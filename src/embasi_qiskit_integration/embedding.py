# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""An EmbASI embedding calculation solved with this package.

End-to-end flow, no longer mocked:

    EmbASI projection embedding (low level)
    -> extract an active-space EmbeddedHamiltonian  [projection_embedding_adapter.py]
    -> solve it with a high-level solver (SQD or FCI)              [solvers.py]
    -> assemble the projection-based-embedding energy       [ProjectionEnergy]
    -> feed the correlated 1-RDM back into the embedding density

This drives a live ``embasi.embedding.ProjectionEmbedding`` through
:class:`ProjectionEmbeddingAdapter`.  The system setup mirrors the EmbASI
developers' methanol ``construct_embedded_fock`` example on the
``qm-code-adapter`` branch -- same methanol monomer, same OH active fragment,
PBE-in-PBE -- except that the projection must be ``level-shift``, since Huzinaga
cannot be exported as a constant offset to an external solver.

:class:`EmbeddingWorkflow` is a pydantic-settings command; ``scripts/
embedding_workflow.py`` is a thin entry point that imports and runs it::

    uv run python scripts/embedding_workflow.py
    uv run python scripts/embedding_workflow.py --solver fci
    uv run python scripts/embedding_workflow.py --sampler runtime --shots 200000
    uv run python scripts/embedding_workflow.py --n_virtual 4 --n_frozen_occ 1
    uv run python scripts/embedding_workflow.py --handoff two-process
    uv run python scripts/embedding_workflow.py --max_cycles 20 --mix_alpha 0.5

The default system is the built-in s26 methanol monomer.  ``--xyz`` runs the same
flow on your own geometry instead, taking the net charge from the file's
``smiles=``/``charge=`` metadata (see
:mod:`embasi_qiskit_integration.molecule.geometry`); ``--active_atoms`` then indexes
into *that* file's atom order::

    uv run python scripts/embedding_workflow.py --xyz mol.xyz --active_atoms '[1,5]'

By default the *full* subsystem-A space is correlated, which is the WF-in-DFT
problem the EmbASI paper solves.  ``--n_frozen_occ`` and ``--n_virtual`` exist
only to fit a solver budget; nothing in the embedding implies an active space.
For SQD on hardware you will want both.

**MPI.** EmbASI's low-level SCF runs collectively on every rank, so
``run_low_level`` is unguarded; only the solve goes through ``ipc.rank0_solve``
(rank 0 solves, result broadcast), and console output / job-dir writes are
guarded to rank 0::

    mpirun -n 2 uv run python scripts/embedding_workflow.py

Remaining work
--------------
The single-pass workflow runs end-to-end against real EmbASI: density-fitted
``eri_mo`` (``--density_fit``), the concentric-localization selector
(``--selector concentric-cl``, the default), the restricted-span eigensolve, and
the low-level energy readout are all in place.  What is left is either blocked on the EmbASI
side or a deliberately-deferred package rewrite; the upstream asks are marked
inline with ``TODO(embasi-api)`` and summarised in the docstring atop
``projection_embedding_adapter.py``:

- ``v_emb`` / ``P_B`` as readable attributes (both reconstructed by subtraction
  today; ``mu`` is cross-checked against ``mu_val``); a retained-AO index map
  under basis truncation; an RI-LVL three-index ERI export (FHI-aims); and
  Huzinaga-as-constant-offset in the same-functional case.

Open-shell / unrestricted embedding is a deferred package rewrite (separate
alpha/beta orbital sets, ``(h1a, h1b)``, SQD spin-symmetry).

References
----------
"The paper" throughout this package -- its Eq. 2 (DFT-in-DFT), Eq. 6 (level
shift), Eq. 8 (WF-in-DFT assembly), Eq. 19 (dissociation energy), §2.2 and
Fig. 3B -- is the EmbASI framework paper:

    G. Bramley, P. Stishenko, O. van Vuren, V. Blum, A. J. Logsdail, "A General
    Pythonic Framework for DFT-in-DFT and WF-in-DFT Embedding", ChemRxiv (2025),
    preprint, doi:10.26434/chemrxiv-2025-c23jf.

The projection / level-shift embedding scheme it implements originates with
F. R. Manby, M. Stella, J. D. Goodpaster, T. F. Miller III, J. Chem. Theory
Comput. 8, 2564 (2012), doi:10.1021/ct300544e.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from embasi_qiskit_integration.circuit_run.base import SamplerKind
from embasi_qiskit_integration.contract import SolverResult
from embasi_qiskit_integration.projection_embedding_adapter import (
    ProjectionEmbeddingAdapter,
    PySCFIntegrals,
)

# Repo root: src/embasi_qiskit_integration/embedding.py -> parents[2] is the
# project root, whose tests/data holds the frozen mock counts (only read on the
# opt-in --sampler mock branch).
DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"


def _rank_size() -> tuple[int, int]:
    """(rank, size) under MPI; (0, 1) when mpi4py is absent (single process)."""
    try:
        from mpi4py import MPI
    except ImportError:
        return 0, 1
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


def _broadcast(obj):
    """Broadcast ``obj`` from rank 0 to all ranks; identity without mpi4py."""
    try:
        from mpi4py import MPI
    except ImportError:
        return obj
    return MPI.COMM_WORLD.bcast(obj, root=0)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class EmbeddingWorkflow(BaseSettings):
    """Run an EmbASI projection-embedding calculation through this package."""

    model_config = SettingsConfigDict(env_prefix="EQI_EMB_", cli_parse_args=True)

    # --- embedding system (mirrors the EmbASI PySCF example) --- #
    xyz: Path | None = None
    charge: int | None = None  # override the charge derived from the .xyz metadata
    s26_index: int = 22  # methanol dimer in the s26 set
    n_atoms: int | None = 6  # first N atoms -> monomer; None for the dimer
    active_atoms: list[int] = [1, 5]  # O and its hydroxyl H -> the OH fragment
    basis: str = "sto-3g"  # matches the EmbASI developers' example
    xc_ll: str = "PBE"
    # HF high level -> the WF-in-DFT path (the SQD integration this package is
    # about).  A KS ``xc_hl`` (e.g. PBE/PBE0) instead routes DFT-in-DFT, where the
    # solver never runs; the dissociation-energy driver overrides this per run.
    xc_hl: str = "HF"
    mu: float = 1.0e6  # level-shift parameter, paper Eq. 6

    # --- active space: solver budget only, not embedding physics --- #
    n_frozen_occ: int = 0
    n_virtual: int | None = None  # solver-budget cap; None -> the selector's own count
    #   (``mulliken``: per-fragment cap; ``spade``: cap after the σ² gap; ``concentric-cl``:
    #   ceiling on the shell-determined k, truncating the outermost shell tail for small
    #   simulations; ``none``: the fixed energy-ordered cut).
    # ``concentric-cl`` by default: the full iterative Concentric Localization of
    # Claudino 2019 (JCTC 15, 6085), the paper-faithful cut.  It keeps a nested,
    # size-consistent active-virtual space (an energy-ordered "none" cut is non-nested
    # across legs) and grows it by ``n_shells`` Fock-coupled shells (the accuracy knob;
    # ``n_shells=0`` is the single-shell case).  ``mulliken`` is the single-shell
    # Mulliken-population cut (ranks canonical F_emb virtuals, cuts on the largest gap;
    # honours ``active_fragment_sizes`` per-fragment); ``spade`` ROTATES the virtual
    # block (Claudino & Mayhall, doi:10.1021/acs.jctc.9b00682) so each rotated virtual
    # carries a definite fragment weight σ², then cuts on the σ² gap -- basis invariant,
    # robust when the canonical virtuals delocalise.  ``none`` disables.
    selector: Literal["none", "mulliken", "spade", "concentric-cl"] = "concentric-cl"
    # ``concentric-cl`` only: number of Fock shell expansions after shell 0.  0 keeps
    # just the fragment-spanned shell (single-shell equivalent); higher values grow the
    # active-virtual space (and the qubit/determinant count) for accuracy.
    n_shells: int = 0
    # How the active atoms partition into PHYSICAL fragments, as consecutive
    # counts in the order they appear in ``active_atoms`` (which the reorder keeps
    # leading, ascending).  ``None`` (default) -> one fragment (current behaviour).
    # e.g. both-OH dimer active_atoms=[1,5,7,11] -> [2, 2] (OH-A | OH-B): the
    # ``mulliken`` selector then cuts each OH shell independently and unions them, so
    # the dimer active-virtual span contains BOTH monomer shells (additivity).  Used
    # ONLY by ``mulliken``; the rotating ``spade``/``concentric-cl`` cuts anchor on the
    # fragment union and ignore the partition.
    active_fragment_sizes: list[int] | None = None

    # --- integral backend --- #
    density_fit: bool = False  # density-fit the active-space ERIs
    df_auxbasis: str | None = None  # fitting auxbasis; None -> PySCF default

    rdm_trace_tol: float = 1.0e-6

    # --- outer self-consistency loop --- #
    max_cycles: int = 15  # cap on feedback cycles (1 -> single pass)
    e_tol: float = 1.0e-6  # |ΔE_total| convergence threshold (Ha)
    rho_tol: float = 1.0e-5  # max |Δγ^A| convergence threshold
    converge_on: Literal["energy_and_density", "energy"] = "energy"
    mix_alpha: float = 0.5  # linear density mixing (1.0 -> undamped)
    reseed_sqd: bool = True  # re-sample the SQD subspace each cycle

    # --- solver / sampling --- #
    solver: Literal["sqd", "fci"] = "sqd"
    handoff: Literal["in-process", "two-process"] = "in-process"
    sampler: SamplerKind = "aer"
    backend: str | None = None  # runtime backend name; else least-busy
    optimization_level: int = 3  # runtime ISA-transpile level
    shots: int = 100_000
    seed: int = 24
    job_dir: Path | None = None

    # ---------------- main ---------------- #
    def cli_cmd(self) -> None:
        self.run()

    def run(self, log=None):
        """Run the full embedding pipeline and return its ``ProjectionEnergy``.

        ``cli_cmd`` (the pydantic-settings entry point) just calls this; callers
        that need the energy back -- e.g. a dissociation-energy driver that
        differences a dimer against its monomers -- can call it directly and use
        the returned :class:`ProjectionEnergy`.  Pass ``log`` to redirect the
        console output (default: print on rank 0).
        """
        rank, size = _rank_size()

        if log is None:

            def log(msg: str = "") -> None:
                """Print only on rank 0 (keeps multi-rank output clean)."""
                if rank == 0:
                    print(msg)

        if size > 1:
            log(f"[running under MPI with {size} ranks; solve is on rank 0]")

        log("== Step 1: EmbASI low-level projection embedding ==")
        source = f"xyz={self.xyz}" if self.xyz is not None else f"s26[{self.s26_index}]"
        log(f"   geometry: {source}")
        emb = self._build_adapter(parallel=size > 1)
        log(
            f"   {self.xc_hl}-in-{self.xc_ll} / {self.basis}, "
            f"active atoms {self.active_atoms}, projection=level-shift"
        )

        # Route on the high-level METHOD, not the solver.  A density functional
        # (pure or hybrid: PBE, PBE0, B3LYP, ...) is DFT-in-DFT (paper Eq. 2): the
        # high-level energy is a Kohn-Sham energy at the embedded density, with no
        # active space, no solver, and no correlated wavefunction.  A wavefunction
        # method (HF as a mean field, or a correlated solver on top) is WF-in-DFT
        # (Eq. 8): downfold subsystem A to an active space and hand a bare
        # electronic Hamiltonian to FCI/SQD.  Routing a hybrid xc through the WF
        # path is a category error (its signature is a large, non-monotonic
        # dependence of Δ_HL on the virtual budget), so it goes down its own path.
        #
        # The two paths drive DIFFERENT collective EmbASI entry points and must
        # branch BEFORE the low-level call, so exactly one fires per rank: the
        # DFT-in-DFT path runs EmbASI's native ``run()`` (which itself runs the
        # supersystem SCF), while the WF path runs ``run_low_level()``.  Calling
        # both would double the supersystem SCF (``run()`` re-invokes
        # ``construct_embedding_potential`` internally).
        if self._is_dft_in_dft():
            return self._dft_in_dft(emb, log=log)

        # WF-in-DFT only.  Collective on every rank: EmbASI's supersystem SCF,
        # SPADE/Pipek-Mezey localisation, and the embedded Fock all run in here.
        emb.run_low_level()
        selector, virtual_localizer = self._build_selector(emb, log=log)
        solver = self._build_solver()
        return self._run_outer_loop(emb, solver, selector, virtual_localizer, rank=rank, log=log)

    def _is_dft_in_dft(self) -> bool:
        """True when the high level is a density functional (-> paper Eq. 2).

        ``HF`` is the sole wavefunction mean field expressible as an ``xc_hl``
        string, so it (and any correlated method layered on it) takes the
        WF-in-DFT path; every other ``xc_hl`` is a KS functional and takes
        DFT-in-DFT.  Kept as an explicit predicate so the routing rule is one
        readable line and the two paths never blur into a shared ``run()`` body.
        """
        return self.xc_hl.strip().upper() != "HF"

    def _dft_in_dft(self, emb, *, log):
        """Paper Eq. 2 reference path: a Kohn-Sham energy at the embedded density.

        Deliberately NOT a variant of the WF outer loop -- it never builds an
        active space, never constructs a downfolded Hamiltonian, never calls a
        solver, and never feeds a correlated density back.  It self-consistently
        relaxes the high-level KS density of subsystem A inside the frozen
        embedding potential and assembles the same :class:`ProjectionEnergy`
        breakdown, so a dissociation-energy driver consumes both paths identically.

        This is the correctness baseline: on s26[22] it reproduces PBE0-in-PBE to
        within the embedding error of the full PBE0 number, and the PBE-in-PBE
        control gives Δ_HL = 0 exactly (the high/low functionals coincide, so the
        embedding is a no-op on the energy -- the paper's Fig. 3B cancellation).
        """
        log("== DFT-in-DFT (paper Eq. 2): embedded Kohn-Sham energy, no active space ==")
        energy = emb.dft_in_dft_energy()
        log("   E = E_low(total) - E_low(A) + E_high(A) + corr")
        log(
            f"     = {energy.e_low_total:.6f} - {energy.e_low_A:.6f} "
            f"+ {energy.e_high_A:.6f} + {energy.correction:.6f}"
        )
        log(f"     = {energy.total:.6f} Ha")
        log(f"   tr[γ̃^A P_B] = {energy.projector_leak:.2e} Ha (should be ~0)")
        log(f"   footing shift applied to E_high(A): {energy.footing_shift:.6f} Ha")
        return energy

    # ---------------- outer self-consistency loop ---------------- #
    def _run_outer_loop(self, emb, solver, selector, virtual_localizer=None, *, rank, log):
        """Steps 2-5, iterated until self-consistent (or ``max_cycles``).

        ``max_cycles == 1`` reproduces the single-pass flow exactly.  With more
        cycles, each iteration builds orbitals, downfolds, solves, assembles the
        PbE energy, and feeds the correlated density back; it stops early once
        the ``converge_on`` criterion is met (``max_cycles`` is the safety cap).

        MPI-safe: the solve goes through ``self._solve`` (rank 0 solves, result
        broadcast), and ``emb.feedback`` runs on every rank (its collective
        ``construct_embedded_fock`` must), so all ranks stay in lock-step.

        A stochastic solver (SQD) makes the fixed point noisy in a way the
        deterministic PbE literature does not address, so convergence is judged
        on running tolerances, not bitwise.  ``--reseed_sqd`` chooses whether the
        sampled subspace is redrawn each cycle (default, honest resampling) or
        carried over (cheaper, but the subspace is frozen at cycle 0's density).

        The undamped map ``γ -> F(γ)`` oscillates and diverges for this problem
        (verified: max|Δγ^A| grows back and E swings ±0.03 Ha), so the fed-back
        density is linearly mixed with the previous one -- ``--mix_alpha`` ∈ (0, 1],
        the fraction of the new density (``1.0`` is undamped).  ``0.5`` converges
        max|Δγ^A| monotonically to ~1e-7 here; smaller is safer but slower.
        """
        prev_total: float | None = None
        prev_dm_a = None
        prev_fed = None
        energy = None

        for cycle in range(self.max_cycles):
            tag = "" if self.max_cycles == 1 else f" [cycle {cycle + 1}/{self.max_cycles}]"

            log(f"== Step 2: build subsystem-A orbitals and downfold =={tag}")
            orbitals = emb.build_orbitals(
                n_frozen_occ=self.n_frozen_occ,
                n_virtual=self.n_virtual,
                selector=selector,
                virtual_localizer=virtual_localizer,
            )
            log(f"   {orbitals}")
            if selector is not None or virtual_localizer is not None:
                n_kept_virt = orbitals.n_active_orbitals - (orbitals.n_occ - orbitals.inactive.size)
                budget = "" if self.n_virtual is None else f" (cap {self.n_virtual})"
                log(
                    f"   selector={self.selector} picked {n_kept_virt} virtuals{budget}; "
                    f"n_frozen_occ={self.n_frozen_occ} frozen"
                )
            elif self.solver == "sqd" and self.n_virtual is None:
                log(
                    "   note: full A virtual space -- pass --n_virtual or "
                    "--selector concentric-cl/mulliken to fit a qubit budget"
                )
            ham = emb.embedded_hamiltonian(orbitals)
            log(
                f"   norb={ham.norb} nelec={ham.nelec} e_core={ham.e_core:.6f} Ha "
                f"(P_B leak {ham.meta['p_b_leak']:.2e})"
            )

            log(f"== Step 3: solve with the high-level {self.solver.upper()} solver =={tag}")
            self._maybe_reseed(solver, cycle)
            result = self._solve(ham, solver, rank=rank, log=log)
            log(
                f"   E_solver  = {result.energy:.6f} Ha "
                f"(solver={result.diagnostics.get('solver', self.solver)})"
            )
            deviation = result.check_particle_number(ham.nelec, atol=self.rdm_trace_tol)
            log(
                f"   trace(rdm1) = {np.trace(result.rdm1):.4f} "
                f"(expected {sum(ham.nelec)}, off by {deviation:+.1e})"
            )

            log(f"== Step 4: assemble the projection-based-embedding energy =={tag}")
            energy = emb.projection_energy(result, orbitals)
            log("   E = E_low(total) - E_low(A) + E_high(A) + corr")
            log(
                f"     = {energy.e_low_total:.6f} - {energy.e_low_A:.6f} "
                f"+ {energy.e_high_A:.6f} + {energy.correction:.6f}"
            )
            log(f"     = {energy.total:.6f} Ha")
            log(f"   tr[γ̃^A P_B] = {energy.projector_leak:.2e} Ha (should be ~0)")
            # E_high(A) is rebased onto E_low(A)'s (ghosted subsystem-A) nuclear
            # frame before the subtraction; without it the two A-terms would be
            # ~4 Ha apart on incompatible frames.  Surface the shift so it is not
            # a silent adjustment.
            log(f"   footing shift applied to E_high(A): {energy.footing_shift:.6f} Ha")

            # Convergence check (only meaningful once we have a previous cycle).
            # `converge_on` selects the criterion: "energy" stops as soon as the
            # PbE total energy stops moving (|ΔE| < e_tol), which is the robust
            # choice under a stochastic solver where the density has sampling
            # noise; "energy_and_density" additionally requires max|Δγ^A| < rho_tol.
            dm_a_now = np.asarray(emb._dm_a)
            if prev_total is not None:
                de = abs(energy.total - prev_total)
                drho = float(np.abs(dm_a_now - prev_dm_a).max())
                log(f"   Δ: |ΔE| = {de:.2e} Ha, max|Δγ^A| = {drho:.2e}")
                energy_ok = de < self.e_tol
                density_ok = drho < self.rho_tol
                converged = energy_ok and (density_ok or self.converge_on == "energy")
                if converged:
                    crit = "|ΔE|" if self.converge_on == "energy" else "|ΔE| and max|Δγ^A|"
                    log(f"   converged ({crit}) after {cycle + 1} cycles.")
                    return energy
            prev_total, prev_dm_a = energy.total, dm_a_now

            if cycle == self.max_cycles - 1:
                if self.max_cycles > 1:
                    log(f"   reached max_cycles={self.max_cycles} without convergence.")
                break

            log(f"== Step 5: feed the correlated 1-RDM back into the embedding =={tag}")
            # Every rank holds the broadcast result, so the density feedback --
            # and the collective construct_embedded_fock call inside it -- stay
            # consistent across the communicator.  Linearly mix the fed-back
            # total density (γ̃^A + γ^B) with the previous cycle's to damp the
            # otherwise-divergent fixed-point iteration; mix_alpha=1.0 is the
            # bare (undamped) feedback emb.feedback would do on its own.
            fed = emb.rdm1_ao(result.rdm1, orbitals)
            if prev_fed is not None and self.mix_alpha != 1.0:
                fed = self.mix_alpha * fed + (1.0 - self.mix_alpha) * prev_fed
            prev_fed = fed
            emb.run_low_level(dma_in=fed, dmb_in=emb._dm_b)
            log(f"   embedded Fock rebuilt at γ̃^A + γ^B (mix_alpha={self.mix_alpha}).")

        return energy

    def _maybe_reseed(self, solver, cycle: int) -> None:
        """Advance the SQD seed each cycle unless the subspace is carried over.

        Only SQDSolver has a ``seed``; FCI is deterministic and ignores this.
        With ``reseed_sqd`` the seed is bumped per cycle so each iteration draws a
        fresh sampled subspace; without it the seed is held so the subspace is
        reused (frozen at the cycle-0 density).
        """
        if cycle == 0 or not hasattr(solver, "seed"):
            return
        if self.reseed_sqd and solver.seed is not None:
            solver.seed = solver.seed + cycle

    # ---------------- construction ---------------- #
    def _build_adapter(self, *, parallel: bool) -> ProjectionEmbeddingAdapter:
        """Set up ProjectionEmbedding exactly as the EmbASI PySCF example does."""
        import pyscf
        from embasi.embedding import ProjectionEmbedding
        from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

        atoms, charge = self._build_atoms()

        # Embedding mask: 1 = high-level, 2 = low-level.
        embed_mask = len(atoms) * [2]
        for idx in self.active_atoms:
            embed_mask[idx] = 1

        # ProjectionEmbedding requires embed_mask sorted so that region 1 atoms
        # come first; reorder atoms to match before building the PySCF Mole so
        # the two stay in sync.
        idx_list = np.argsort(embed_mask)
        sort_embed_mask = np.sort(embed_mask)
        atoms = atoms[idx_list]

        # TODO(embasi-api): the charge is set on the PySCF Mole (which needs it to
        # get nelec right), but ProjectionEmbedding is handed only the ASE Atoms and
        # exposes no charge argument, so its own low-level SCF may still assume a
        # neutral system.  ASK: accept a charge (or read it off the ASE Atoms'
        # initial_charges).  Until that is confirmed against a live EmbASI, treat
        # non-zero-charge embedding runs as unvalidated.
        mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis=self.basis, charge=charge)
        mf_ll = mol.KS(xc=self.xc_ll)
        mf_hl = mol.KS(xc=self.xc_hl)

        projection = ProjectionEmbedding(
            atoms,
            embed_mask=sort_embed_mask,
            calc_base_ll=PySCF(method=mf_ll),
            calc_base_hl=PySCF(method=mf_hl),
            projection="level-shift",
            parallel=parallel,
        )
        # PySCFIntegrals wraps the *same* mf_hl object, so veff_hl undoes exactly
        # what EmbASI folded into F_emb, whether that is KS or HF.
        density_fit: bool | str = self.df_auxbasis or self.density_fit
        integrals = PySCFIntegrals(mf_hl, density_fit=density_fit)
        return ProjectionEmbeddingAdapter(projection, integrals, mu=self.mu)

    def _build_atoms(self) -> tuple[Any, int]:
        """Return ``(ase.Atoms, charge)`` for the requested geometry source."""
        if self.xyz is not None:
            from embasi_qiskit_integration.molecule import geometry

            atoms, charge, _smiles = geometry.read_atoms_from_xyz(
                self.xyz, charge_override=self.charge
            )
            return atoms, charge

        from ase.data.s22 import create_s22_system, s26

        atoms = create_s22_system(s26[self.s26_index])
        if self.n_atoms is not None:
            atoms = atoms[: self.n_atoms]
        return atoms, self.charge or 0

    def _build_selector(self, emb: ProjectionEmbeddingAdapter, *, log=None):
        """Build the active-virtual shaping hook from ``self.selector``.

        Returns a ``(selector, virtual_localizer)`` pair with at most one non-None
        (both ``None`` for the fixed ``--n_virtual`` cut):

        * ``mulliken`` -> a ``Selector`` (per-fragment single-shell Mulliken cut),
        * ``spade`` -> a ``VirtualLocalizer`` (SPADE rotation + σ² gap cut),
        * ``concentric-cl`` -> a ``VirtualLocalizer`` (iterative CL, ``n_shells``),
        * ``none`` -> ``(None, None)``.

        A rotation cannot be expressed as a column-index ``Selector``, so ``spade``
        and ``concentric-cl`` go through the ``build_orbitals(virtual_localizer=...)``
        hook instead, anchored on the fragment UNION (a rotation cannot be unioned
        per-fragment the way index selection can).

        After the atom reorder in ``_build_adapter`` the region-1 (active) atoms
        occupy the first ``len(active_atoms)`` positions of ``mol``, so the
        fragment AO indices come straight from the leading atom slices.
        """
        if self.selector == "none":
            return None, None
        from embasi_qiskit_integration.selectors import (
            concentric_localization_selector,
            fragment_ao_indices,
            per_fragment_mulliken_selector,
            spade_virtual_selector,
        )

        mol = emb.ints.mol
        # run_low_level() (called before this) populates the overlap and F_emb; assert
        # for the type-checker and to fail loudly if the call order is ever broken.
        assert emb._s is not None, "run_low_level() must run before _build_selector()"
        overlap = emb._s
        # Reordered active-atom positions lead, ascending: 0..len(active_atoms)-1.
        # Partition them into PHYSICAL fragments per ``active_fragment_sizes`` (one
        # group by default).
        n_active = len(self.active_atoms)
        sizes = self.active_fragment_sizes or [n_active]
        if sum(sizes) != n_active:
            raise ValueError(
                f"active_fragment_sizes {sizes} sum to {sum(sizes)}, but there are "
                f"{n_active} active atoms"
            )
        groups, start = [], 0
        for sz in sizes:
            positions = list(range(start, start + sz))
            groups.append(fragment_ao_indices(mol, positions))
            start += sz

        if self.selector in ("spade", "concentric-cl"):
            # A rotation cannot be unioned per-fragment the way index selection can,
            # so both localisers anchor on the UNION of the fragment AOs.
            frag_union = np.unique(np.concatenate(groups)) if groups else np.empty(0, int)
            if self.selector == "spade":
                # ``n_virtual`` caps the kept rotated shell.
                return None, spade_virtual_selector(overlap, frag_union, max_virtual=self.n_virtual)
            # concentric-cl: the shell count sets the physically-motivated k; if
            # ``n_virtual`` is given it is a solver-budget CEILING on top of that
            # (for small simulations), truncating the outermost shell tail -- not a
            # replacement for the shell structure.
            assert emb._fock is not None, "run_low_level() must run before _build_selector()"
            if self.n_virtual is not None and log is not None:
                log(
                    f"   [selector] concentric-cl: capping the shell-determined virtuals "
                    f"at n_virtual={self.n_virtual} (solver budget)"
                )
            return None, concentric_localization_selector(
                overlap, frag_union, emb._fock, n_shells=self.n_shells, max_virtual=self.n_virtual
            )

        # mulliken: run the per-fragment single-shell Mulliken cut and union.
        # ``n_virtual`` is the per-FRAGMENT ceiling here (additivity: each fragment
        # gets its own shell, matched to the monomer leg's cut), not a global cap.
        return (
            per_fragment_mulliken_selector(
                overlap, groups, max_virtual_per_fragment=self.n_virtual
            ),
            None,
        )

    def _build_solver(self):
        from embasi_qiskit_integration.solvers import FCISolver, SQDSolver

        if self.solver == "fci":
            return FCISolver()

        from embasi_qiskit_integration.circuit_run import build_sampler

        # ``runtime`` plugs in real quantum hardware: configured IBM Quantum
        # credentials, least-busy backend unless --backend names one.
        sampler = build_sampler(
            self.sampler,
            counts=str(DATA_DIR / "mock_counts.json"),
            backend=self.backend,
            optimization_level=self.optimization_level,
            default_shots=self.shots,
        )
        return SQDSolver(sampler, shots=self.shots, seed=self.seed)

    def _solve(self, ham, solver, *, rank, log) -> SolverResult:
        """Solve in-process, or via the two-process file handoff.

        Both paths are MPI-safe: rank 0 does the solving/IO and the resulting
        :class:`SolverResult` is broadcast to every rank.
        """
        if self.handoff == "in-process":
            from embasi_qiskit_integration.ipc import rank0_solve

            return rank0_solve(solver, ham)

        # Two-process: rank 0 writes a job dir and solves it (standing in for the
        # separate `embasi-qiskit-integration solve <dir> --watch` process), then
        # broadcasts the result so every rank returns the same SolverResult.
        import tempfile

        from embasi_qiskit_integration import ipc

        result = None
        if rank == 0:
            job_dir = self.job_dir or Path(tempfile.mkdtemp(prefix="eqi_jobdir_"))
            ipc.write_job(ham, job_dir)
            log(f"   wrote job to {job_dir} (process B would run 'solve --watch')")
            solved = solver.solve(ipc.read_job(job_dir))
            ipc.write_result(solved, job_dir)
            result = ipc.read_result(job_dir)
        return _broadcast(result)


def main() -> None:
    CliApp.run(EmbeddingWorkflow)


if __name__ == "__main__":
    main()
