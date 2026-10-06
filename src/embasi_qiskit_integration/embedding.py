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
from typing import Any, Literal, Protocol

import numpy as np
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from embasi_qiskit_integration._sweep import FloatSweep, IntSweep
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
    """(rank, size) under MPI; (0, 1) when MPI is unusable (single process).

    Catches more than ``ImportError``: ``mpi4py`` resolves its MPI runtime on import and
    raises ``RuntimeError("cannot load MPI library")`` when the package is installed but no
    ``libmpi`` is present -- an environment without a system MPI, which is single-process
    by definition.
    """
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError, OSError):
        return 0, 1
    comm = MPI.COMM_WORLD
    return comm.Get_rank(), comm.Get_size()


def _broadcast(obj):
    """Broadcast ``obj`` from rank 0 to all ranks; identity when MPI is unusable."""
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError, OSError):
        return obj
    return MPI.COMM_WORLD.bcast(obj, root=0)


# --------------------------------------------------------------------------- #
# CLI
def _check_shell_consistency(spin: int, unrestricted: bool) -> None:
    """Refuse a spin state the restricted downfold cannot represent.

    Runs *before* any SCF, so the failure it prevents costs nothing: the restricted path
    derives ``nelec`` by halving the active electron count, so a doublet would silently
    become a closed shell with a plausible energy and nothing to flag it.

    A negative ``spin`` is rejected outright, since this reaches us from a
    ``BaseSettings`` field a caller can set freely.

    Raises:
        ValueError: if ``spin != 0`` without ``unrestricted=True``.
    """
    if spin < 0:
        raise ValueError(f"spin must be non-negative (2*S = unpaired electrons); got {spin}")
    if spin != 0 and not unrestricted:
        raise ValueError(
            f"spin={spin} needs unrestricted=True: the restricted downfold assigns "
            "n_alpha = n_beta = n_active_electrons // 2, which cannot represent an open "
            "shell and would return a closed-shell answer for an open-shell system. "
            "Set unrestricted=True (spin-resolved path), or spin=0."
        )


def seed_for_cycle(base_seed: int, cycle: int, *, reseed: bool = True) -> int:
    """Absolute SQD seed for a given outer-loop cycle.

    The seed selects the SQD sampled subspace, so it is part of what defines the
    problem each cycle solves -- which makes reproducing the *schedule* a
    correctness requirement for any driver that re-implements the outer loop
    (e.g. an out-of-process one running a single cycle per round).

    The schedule is triangular, not linear.  :meth:`EmbeddingWorkflow._maybe_reseed`
    historically bumped the solver's *own* attribute in place
    (``solver.seed = solver.seed + cycle``), so each cycle adds ``cycle`` to the
    already-bumped value:

        cycle:  0   1   2   3   4   5
        seed:  24  25  27  30  34  39      (base=24, i.e. base + k(k+1)/2)

    It reads as ``base + cycle`` at a glance and is not; expressing it as a pure
    function of ``(base_seed, cycle)`` is what lets a fresh process compute round
    *n*'s seed without having run rounds 0..*n*-1.

    Args:
        base_seed: the configured seed (the cycle-0 value).
        cycle: zero-based cycle index.
        reseed: when False the subspace is deliberately reused, so the seed is
            held at ``base_seed`` for every cycle.

    Returns:
        The absolute seed for ``cycle``; ``base_seed`` at cycle 0 either way.
    """
    if not reseed:
        return base_seed
    return base_seed + (cycle * (cycle + 1)) // 2


def diis_extrapolate(residuals: list[np.ndarray], outputs: list[np.ndarray]) -> np.ndarray | None:
    """Pulay/DIIS extrapolation of the density-feedback fixed point.

    Module-level so an out-of-process outer loop can reuse the *same* extrapolation
    rather than reimplementing it; :meth:`EmbeddingWorkflow._diis_extrapolate`
    delegates here.

    Standard DIIS: find coefficients ``c`` (summing to 1) minimising
    ``|| sum_i c_i * residuals[i] ||``, by solving the bordered linear system

        [B  -1] [c]   [0]
        [-1  0] [λ] = [-1]
    Returns ``None`` (caller falls back to linear mixing) if ``B`` is
    singular -- expected once residuals shrink toward linear dependence
    near convergence, and routine if two cycles happen to produce
    near-identical residuals.
    """
    n = len(residuals)
    b = np.empty((n + 1, n + 1))
    for i, ri in enumerate(residuals):
        for j, rj in enumerate(residuals):
            b[i, j] = float(np.vdot(ri, rj).real)
    b[:n, n] = -1.0
    b[n, :n] = -1.0
    b[n, n] = 0.0
    rhs = np.zeros(n + 1)
    rhs[n] = -1.0
    try:
        coeffs = np.linalg.solve(b, rhs)[:n]
    except np.linalg.LinAlgError:
        return None
    return sum(c * o for c, o in zip(coeffs, outputs))


class EmbeddingSetup(Protocol):
    """The settings :func:`build_adapter` / :func:`build_selector` actually read.

    Structural, not a base class: :class:`EmbeddingWorkflow` satisfies it as-is (it
    is the CLI ``BaseSettings``), and so does any plain dataclass or simple namespace
    carrying these attributes.  That is the point of the seam -- an external driver
    (a workflow engine assembling its own settings model, say) can call the builders
    without importing or instantiating a ``BaseSettings`` whose
    ``cli_parse_args=True`` would try to read ``sys.argv``.

    Only the geometry/adapter/selector fields appear here; the solver and outer-loop
    fields stay private to :class:`EmbeddingWorkflow`, which owns that loop.
    """

    # geometry source (build_atoms)
    @property
    def xyz(self) -> Any: ...
    @property
    def charge(self) -> int | None: ...
    @property
    def s26_index(self) -> int: ...
    @property
    def n_atoms(self) -> int | None: ...

    # adapter (build_adapter)
    @property
    def basis(self) -> str: ...
    @property
    def active_atoms(self) -> list[int]: ...
    @property
    def xc_ll(self) -> str: ...
    @property
    def xc_hl(self) -> str: ...
    @property
    def mu(self) -> float: ...
    @property
    def spin(self) -> int: ...
    @property
    def unrestricted(self) -> bool: ...
    @property
    def density_fit(self) -> bool: ...
    @property
    def df_auxbasis(self) -> str | None: ...
    @property
    def scf_max_cycle(self) -> int | None: ...
    @property
    def init_guess(self) -> str | None: ...
    @property
    def init_guess_breaksym(self) -> Any: ...
    @property
    def localisation(self) -> str: ...
    @property
    def uno_fill(self) -> str: ...
    @property
    def avas_ao_labels(self) -> list[str] | None: ...
    @property
    def avas_threshold(self) -> float: ...
    @property
    def avas_minao(self) -> str: ...
    @property
    def avas_max_size(self) -> tuple[int, ...] | None: ...
    @property
    def n_frozen_occ(self) -> int: ...
    @property
    def uno_n_env(self) -> int | None: ...
    @property
    def scf_stability(self) -> bool: ...

    # selector (build_selector)
    @property
    def selector(self) -> str: ...
    @property
    def active_fragment_sizes(self) -> list[int] | None: ...
    @property
    def n_virtual(self) -> int | None: ...
    @property
    def n_shells(self) -> int: ...
    @property
    def apc_max_size(self) -> tuple[int, int] | None: ...
    @property
    def apc_fixed(self) -> Any: ...


def build_atoms(cfg: EmbeddingSetup) -> tuple[Any, int]:
    """Return ``(ase.Atoms, charge)`` for the requested geometry source."""
    if cfg.xyz is not None:
        from embasi_qiskit_integration.molecule import geometry

        atoms, charge, _smiles = geometry.read_atoms_from_xyz(cfg.xyz, charge_override=cfg.charge)
        return atoms, charge

    from ase.data.s22 import create_s22_system, s26

    atoms = create_s22_system(s26[cfg.s26_index])
    if cfg.n_atoms is not None:
        atoms = atoms[: cfg.n_atoms]
    return atoms, cfg.charge or 0


def build_adapter(cfg: EmbeddingSetup, *, parallel: bool) -> ProjectionEmbeddingAdapter:
    """Set up ProjectionEmbedding exactly as the EmbASI PySCF example does."""
    # Validate the config before importing EmbASI: the import loads mpi4py, which needs
    # an MPI library, and a bad flag should fail the same way with or without one.
    _check_shell_consistency(cfg.spin, cfg.unrestricted)
    if cfg.uno_n_env is not None:
        if cfg.localisation != "UNO-SPADE":
            raise ValueError("--uno_n_env needs --localisation UNO-SPADE")
        if getattr(cfg, "a_nmos", None) is not None:
            raise ValueError("--uno_n_env and --a_nmos both fix the UNO-SPADE cut; give only one")

    import pyscf
    from embasi.embedding import ProjectionEmbedding
    from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

    atoms, charge = build_atoms(cfg)

    # Embedding mask: 1 = high-level, 2 = low-level.
    embed_mask = len(atoms) * [2]
    for idx in cfg.active_atoms:
        embed_mask[idx] = 1

    # ProjectionEmbedding requires embed_mask sorted so that region 1 atoms
    # come first; reorder atoms to match before building the PySCF Mole so
    # the two stay in sync.
    idx_list = np.argsort(embed_mask)
    sort_embed_mask = np.sort(embed_mask)
    atoms = atoms[idx_list]

    mol = pyscf.M(
        atom=ase_atoms_to_pyscf(atoms),
        basis=cfg.basis,
        charge=charge,
        spin=cfg.spin,
    )
    # A non-zero spin needs an unrestricted low level; PySCF's UKS carries the
    # spin axis EmbASI already threads through SPADE (n_spins > 1).
    if cfg.spin != 0 or cfg.unrestricted:
        mf_ll = mol.UKS(xc=cfg.xc_ll)
        mf_hl = mol.UKS(xc=cfg.xc_hl)
    else:
        mf_ll = mol.KS(xc=cfg.xc_ll)
        mf_hl = mol.KS(xc=cfg.xc_hl)
    # EmbASI runs each SCF on these objects (`mf.kernel(dm0=mf.get_init_guess())` when it
    # has no density of its own), so their cycle cap and guess settings are honoured.
    for mf in (mf_ll, mf_hl):
        if cfg.scf_max_cycle is not None:
            mf.max_cycle = cfg.scf_max_cycle
        if cfg.init_guess is not None:
            mf.init_guess = cfg.init_guess
        if cfg.init_guess_breaksym is not None and hasattr(mf, "init_guess_breaksym"):
            mf.init_guess_breaksym = cfg.init_guess_breaksym

    projection = ProjectionEmbedding(
        atoms,
        embed_mask=sort_embed_mask,
        calc_base_ll=PySCF(method=mf_ll),
        calc_base_hl=PySCF(method=mf_hl),
        projection="level-shift",
        localisation=cfg.localisation,
        # EmbASI sets the supersystem Mole's charge from this (AB_LL.input_total_charge),
        # overwriting the one on `mol` above -- leaving it at its default 0 would silently
        # run a charged system as neutral.  The fragments take theirs from SPADE's populations.
        total_charge=charge,
        # Only when set, so a run against an EmbASI without the option is unaffected.
        **({"uno_n_env": cfg.uno_n_env} if cfg.uno_n_env is not None else {}),
        **({"scf_stability": True} if cfg.scf_stability else {}),
        parallel=parallel,
    )
    # PySCFIntegrals wraps the *same* mf_hl/mf_ll objects handed to
    # calc_base_hl/calc_base_ll, so veff_ll undoes exactly what EmbASI folded into
    # F_emb (which is built from the A_LL blocks -- see the h_emb property) and
    # veff_hl remains available as the high-level mean field.
    density_fit: bool | str = cfg.df_auxbasis or cfg.density_fit
    integrals = PySCFIntegrals(mf_hl, mf_ll, density_fit=density_fit)
    return ProjectionEmbeddingAdapter(
        projection, integrals, mu=cfg.mu, unrestricted=cfg.unrestricted
    )


def avas_valence_ao(mol, frag_ao, labels=None) -> np.ndarray:
    """Fragment AO indices the ``avas`` UNO fill projects onto.

    ``labels`` are PySCF AVAS-style AO labels (``"Fe 3d"``, ``"C 2p"``, ...) matched with
    ``mol.search_ao_label`` and restricted to ``frag_ao``; ``None`` takes the fragment's p
    shells, which suits main-group pi/sigma bonds but not a transition metal's d shell.
    """
    frag_ao = np.asarray(frag_ao, dtype=int)
    ao_labels = mol.ao_labels(fmt=False)
    if not labels:
        return np.array([i for i in frag_ao if ao_labels[i][2].endswith("p")], dtype=int)
    hits = np.asarray(mol.search_ao_label(list(labels)), dtype=int)
    valence = np.intersect1d(hits, frag_ao)
    if valence.size == 0:
        shells = sorted({" ".join(ao_labels[i][1:3]) for i in frag_ao})
        raise ValueError(
            f"--avas_ao_labels {list(labels)} match no AO of the fragment; its shells are: "
            f"{', '.join(shells)}"
        )
    return valence


def build_selector(
    cfg: EmbeddingSetup,
    emb: ProjectionEmbeddingAdapter,
    *,
    log=None,
    use_relaxed: bool = False,
):
    """Build the active-virtual shaping hook from ``cfg.selector``.

    ``use_relaxed`` (mirroring
    :meth:`ProjectionEmbeddingAdapter.build_orbitals`'s flag) selects the relaxed
    embedded-HF Fock from :meth:`ProjectionEmbeddingAdapter.relax_active_hf`
    instead of the one-shot low-level ``F_emb``, for every branch below that reads
    a Fock eagerly (``concentric-cl``, ``apc-concentric``).  ``relax_hf`` gates
    both whether ``relax_active_hf`` ran and this flag, so the two always agree.

    Returns a ``(selector, virtual_localizer, orbital_builder)`` triple with at
    most one non-None (all ``None`` for the fixed ``--n_virtual`` cut):

    * ``mulliken`` -> a ``Selector`` (per-fragment single-shell Mulliken cut),
    * ``spade`` -> a ``VirtualLocalizer`` (SPADE rotation + σ² gap cut),
    * ``concentric-cl`` -> a ``VirtualLocalizer`` (iterative CL, ``n_shells``),
    * ``apc-concentric`` -> an ``orbital_builder`` (``emb -> EmbeddedOrbitals``):
      CL locality pre-filter then APC ranking, see
      ``ProjectionEmbeddingAdapter.build_orbitals_apc_concentric``,
    * ``avas`` -> an ``orbital_builder``: occupied and virtual blocks projected onto
      the fragment's ``avas_ao_labels`` AOs, see
      ``ProjectionEmbeddingAdapter.build_orbitals_avas``,
    * ``none`` -> ``(None, None, None)``.

    A rotation cannot be expressed as a column-index ``Selector``, so ``spade``
    and ``concentric-cl`` go through the ``build_orbitals(virtual_localizer=...)``
    hook instead, anchored on the fragment UNION (a rotation cannot be unioned
    per-fragment the way index selection can); ``apc-concentric`` needs a full
    ``EmbeddedOrbitals`` (it calls ``build_orbitals`` itself, then re-selects),
    so it returns a third kind of hook the caller invokes directly on ``emb``.

    After the atom reorder in ``_build_adapter`` the region-1 (active) atoms
    occupy the first ``len(active_atoms)`` positions of ``mol``, so the
    fragment AO indices come straight from the leading atom slices.
    """
    if cfg.selector == "none":
        return None, None, None
    if cfg.selector == "uno":
        # Unrestricted natural orbitals of subsystem A over one orbital set shared by both
        # spins (see `build_orbitals_uno`); budget from `apc_max_size`/`apc_fixed`.
        if cfg.apc_max_size is None:
            raise ValueError("--selector uno requires --apc_max_size (nelec,norb)")
        max_size, fixed = cfg.apc_max_size, cfg.apc_fixed
        # After the atom reorder the active atoms come first; their AOs define the fragment
        # for the fill (all of them for apc-fragment, the valence p shells for avas).
        from embasi_qiskit_integration.selectors import fragment_ao_indices

        frag_ao = fragment_ao_indices(emb.ints.mol, list(range(len(cfg.active_atoms))))
        valence_ao = avas_valence_ao(emb.ints.mol, frag_ao, cfg.avas_ao_labels)

        def _uno_builder(
            emb,
            max_size=max_size,
            fixed=fixed,
            use_relaxed=use_relaxed,
            spin=False,
            fill=cfg.uno_fill,
            frag_ao=frag_ao,
            valence_ao=valence_ao,
        ):
            if not spin:
                raise ValueError(
                    "--selector uno needs the per-spin downfold: pass --unrestricted True "
                    "--spin_downfold True (with --localisation UNO-SPADE)"
                )
            return emb.build_orbitals_uno(
                max_size=max_size,
                fixed=fixed,
                use_relaxed=use_relaxed,
                fill=fill,
                fragment_ao=frag_ao,
                valence_ao=valence_ao,
            )

        return None, None, _uno_builder
    if cfg.selector == "avas":
        if not cfg.avas_ao_labels:
            raise ValueError(
                "--selector avas requires --avas_ao_labels, the target valence AOs "
                '(e.g. \'["C 2p", "O 2p"]\' or \'["Fe 3d"]\')'
            )
        from embasi_qiskit_integration.selectors import avas_ao_projector

        # After the atom reorder the active atoms come first, so the targets are
        # restricted to the fragment's own copies of the labelled AOs.
        ao_projector = avas_ao_projector(
            emb.ints.mol,
            cfg.avas_ao_labels,
            atoms=range(len(cfg.active_atoms)),
            minao=cfg.avas_minao,
        )

        def _avas_builder(
            emb,
            ao_projector=ao_projector,
            threshold=cfg.avas_threshold,
            max_size=cfg.avas_max_size,
            n_frozen_occ=cfg.n_frozen_occ,
            use_relaxed=use_relaxed,
            spin=False,
        ):
            if spin:
                # Per-spin: each channel selected on its own; only norb is shared, so the
                # threshold cut (which cannot promise a common count) is not available.
                if max_size is None:
                    raise ValueError(
                        "--selector avas with --spin_downfold needs --avas_max_size, as "
                        "(nelec,norb) or (nelec_alpha,nelec_beta,norb): the two channels are "
                        "selected independently but must share an active-orbital count, "
                        "which a per-channel threshold cannot guarantee"
                    )
                return emb.build_orbitals_avas_spin(
                    ao_projector=ao_projector,
                    max_size=max_size,
                    n_frozen_occ=n_frozen_occ,
                    use_relaxed=use_relaxed,
                )
            return emb.build_orbitals_avas(
                ao_projector=ao_projector,
                threshold=threshold,
                max_size=max_size,
                n_frozen_occ=n_frozen_occ,
                use_relaxed=use_relaxed,
            )

        return None, None, _avas_builder
    from embasi_qiskit_integration.selectors import (
        concentric_localization_selector,
        fragment_ao_indices,
        per_fragment_mulliken_selector,
        spade_virtual_selector,
    )

    mol = emb.ints.mol
    assert emb._s is not None, "run_low_level() must run before _build_selector()"
    overlap = emb._s
    n_active = len(cfg.active_atoms)
    sizes = cfg.active_fragment_sizes or [n_active]
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

    if cfg.selector in ("spade", "concentric-cl", "apc-concentric"):
        frag_union = np.unique(np.concatenate(groups)) if groups else np.empty(0, int)
        if cfg.selector == "spade":
            return (
                None,
                spade_virtual_selector(overlap, frag_union, max_virtual=cfg.n_virtual),
                None,
            )
        assert emb._fock is not None, "run_low_level() must run before _build_selector()"
        if cfg.selector == "apc-concentric":
            if cfg.apc_max_size is None:
                raise ValueError("--selector apc-concentric requires --apc_max_size (nelec,norb)")
            n_shells, max_size, fixed = cfg.n_shells, cfg.apc_max_size, cfg.apc_fixed

            def _orbital_builder(
                emb,
                frag=frag_union,
                n_shells=n_shells,
                max_size=max_size,
                fixed=fixed,
                use_relaxed=use_relaxed,
                spin=False,
            ):
                # `spin=True` is the `spin_downfold` path: the same CL + APC recipe, run
                # per channel and reconciled through paired slots, returning an
                # (alpha, beta) pair instead of one `EmbeddedOrbitals`.
                build = (
                    emb.build_orbitals_apc_concentric_spin
                    if spin
                    else emb.build_orbitals_apc_concentric
                )
                return build(
                    fragment_ao=frag,
                    n_shells=n_shells,
                    max_size=max_size,
                    fixed=fixed,
                    use_relaxed=use_relaxed,
                )

            return None, None, _orbital_builder
        # concentric-cl: the shell count sets k; ``n_virtual`` is a solver-budget CEILING
        # on top of it, truncating the outermost shell tail -- not a replacement.
        if cfg.n_virtual is not None and log is not None:
            log(
                f"   [selector] concentric-cl: capping the shell-determined virtuals "
                f"at n_virtual={cfg.n_virtual} (solver budget)"
            )
        return (
            None,
            concentric_localization_selector(
                overlap,
                frag_union,
                emb._fock_relaxed_arr if use_relaxed else emb._fock,
                n_shells=cfg.n_shells,
                max_virtual=cfg.n_virtual,
            ),
            None,
        )

    # mulliken: run the per-fragment single-shell Mulliken cut and union.
    # ``n_virtual`` is the per-FRAGMENT ceiling here (additivity: each fragment
    # gets its own shell, matched to the monomer leg's cut), not a global cap.
    return (
        per_fragment_mulliken_selector(overlap, groups, max_virtual_per_fragment=cfg.n_virtual),
        None,
        None,
    )


# --------------------------------------------------------------------------- #
class EmbeddingWorkflow(BaseSettings):
    """Run an EmbASI projection-embedding calculation through this package."""

    model_config = SettingsConfigDict(env_prefix="EQI_EMB_", cli_parse_args=True)

    # --- embedding system (mirrors the EmbASI PySCF example) --- #
    xyz: Path | None = None
    charge: int | None = None  # override the charge derived from the .xyz metadata
    spin: int = 0
    # Opt in to the spin-resolved embedding path (alpha/beta orbital sets, an
    # ``(h1a, h1b)`` pair).  Off by default: the closed-shell path stays bit-identical,
    # and the open-shell energy assembly is not yet validated against a UKS reference.
    unrestricted: bool = False
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
    # WF-in-DFT only: converge subsystem A's own HF problem on A_HL in the frozen
    # v_emb/P_B potential (see `relax_active_hf`).  Without it `build_orbitals` diagonalizes
    # F_emb once against A_LL's density, so Brillouin's theorem does not hold for the
    # orbitals FCI/SQD receives -- an error an exact active-space solve cannot recover.
    # Off by default: costs an extra embedded SCF and moves the published reference.
    relax_hf: bool = False
    # Open shell only: diagonalize each channel in its OWN span(A) and downfold to an
    # (h1a, h1b) + (aa, ab, bb) Hamiltonian.  Needs `unrestricted=True` and an EmbASI that
    # reports a spin axis.  A spin-RESTRICTED downfold is ill-defined on an open shell (SPADE
    # partitions each channel independently, so a spin-summed P_B annihilates neither), but
    # this stays a flag rather than a silent switch: the mulliken index selector is
    # unsupported here, apc-concentric runs per channel through paired alpha/beta slots
    # (DRAFT, see `build_orbitals_apc_concentric_spin`), and only the FCI solver consumes
    # the pair.
    spin_downfold: bool = False

    a_nmos: int | None = None  # Fixes the number of electrons selected by SPADE
    # EmbASI's partition.  "UNO-SPADE" runs SPADE on the doubly occupied unrestricted
    # natural orbitals only and puts every fractional/singly occupied one in A, so the
    # environment is one closed-shell orbital set shared by both spins (with it, a_nmos
    # counts the doubly occupied orbitals given to A).  Needed by ``--selector uno``.
    localisation: Literal["SPADE", "UNO-SPADE"] = "SPADE"
    # ``--selector uno``: how the budget beyond the fractional UNOs is filled.  "avas"
    # projects onto the fragment's valence p AOs (same chemical orbitals at every geometry),
    # "apc-fragment" ranks fragment-local orbitals by APC, "apc" ranks all of A by APC.
    uno_fill: Literal["avas", "apc-fragment", "apc"] = "avas"
    # ``--uno_fill avas`` and ``--selector avas``: the fragment AOs projected onto, as PySCF
    # AVAS labels (e.g. ["Fe 3d", "C 2p", "O 2p"]); restricted to the fragment's atoms.
    # Required by ``--selector avas``.  For the UNO fill, None -> the fragment's p shells
    # (right for C/N/O pi and sigma bonds, not for a transition metal).
    avas_ao_labels: list[str] | None = None
    # ``--selector avas`` only: keep orbitals whose weight on the target AOs exceeds this
    # (PySCF's default), and match the labels in this minimal reference basis.
    avas_threshold: float = 0.2
    avas_minao: str = "minao"
    # ``--selector avas`` only: a fixed (nelec, norb) active space filled with the
    # highest-weight orbitals instead of the threshold cut, so the size cannot change
    # between cycles or along a scan.  Singly occupied orbitals count towards both and
    # are always kept.  None -> the threshold sets the size.  With --spin_downfold the
    # alpha and beta channels are selected independently to a common norb and this is
    # required; (nelec_alpha, nelec_beta, norb) sets the per-spin split explicitly.
    avas_max_size: tuple[int, ...] | None = None
    # UNO-SPADE only: fix the number of doubly occupied natural orbitals in the environment,
    # so B is the same size at every geometry of a scan (A takes the rest).  The doubly
    # occupied count changes with geometry as orbitals become fractional, so a fixed a_nmos
    # does not do this.  Cannot be combined with a_nmos.  None -> SPADE's gap.
    uno_n_env: int | None = None
    # Constrain the solve to <S^2> = target_s2 (e.g. 0 singlet, 2 triplet), keeping the
    # lowest matching root: FCI's fix_spin_ penalty, or SQD's spin_sq (the same penalty in
    # the subspace diagonalisation).  Needs alpha and beta to share orbitals: a restricted
    # downfold, or ``--selector uno``.  None -> unconstrained, which returns the lowest
    # state of the --spin sector whatever its S (e.g. a quintet in a triplet's M_s = 1).
    target_s2: float | None = None
    # With --target_s2 and a genuinely spin-dependent downfold (a KS low level makes the
    # embedding potential depend on A's spin density): average h1a/h1b and solve
    # spin-adapted anyway.  Exact spin states, at the cost of dropping the physical alpha/beta
    # difference of the embedding potential.  Off: the penalised unrestricted solve, whose
    # roots are only approximately pure (reported).
    fci_spin_average: bool = False

    # --- low-level / high-level SCF controls (None -> PySCF's own default) --- #
    scf_max_cycle: int | None = None  # SCF iteration cap (PySCF default 50)
    # Starting guess for each SCF: "minao" (PySCF default), "atom", "huckel", "1e", ...
    init_guess: str | None = None
    # Unrestricted runs at 2S = 0 only: how PySCF breaks alpha/beta symmetry in the guess.
    # 1 (PySCF default) keeps only atom-diagonal blocks of the beta density, "mix" rotates
    # HOMO/LUMO by 45 degrees between the channels, 0 keeps the guess spin-symmetric.
    init_guess_breaksym: int | Literal["mix"] | None = None
    # Supersystem SCF only: after it converges, run PySCF stability analysis in EmbASI and
    # follow any instability (including breaking the spin symmetry of a closed-shell UHF) to
    # a stable solution, so a scan does not jump between a saddle point and the minimum.
    scf_stability: bool = False

    # --- active space: solver budget only, not embedding physics --- #
    n_frozen_occ: int = 0
    n_virtual: int | None = None  # solver-budget cap; None -> the selector's own count
    #   (``mulliken``: per-fragment cap; ``spade``: after the σ² gap; ``concentric-cl``:
    #   a ceiling on the shell-determined k; ``none``: the energy-ordered cut).  With
    #   ``spin_downfold=True`` it caps the COMMON active-orbital count instead, since the
    #   channels' differing occupied counts make one per-channel count meaningless.
    # ``concentric-cl`` (default) keeps a nested, size-consistent active-virtual space and
    # grows it by ``n_shells`` Fock-coupled shells.  ``mulliken`` cuts on the largest gap in
    # the canonical virtuals' fragment population (honours ``active_fragment_sizes``);
    # ``spade`` rotates the virtual block so each virtual carries a definite weight σ² and
    # cuts on the σ² gap -- basis invariant, robust when the canonicals delocalise.
    # ``apc-concentric`` adds APC ranking over BOTH occupied and virtual candidates, so it
    # supersedes ``n_frozen_occ``.  ``avas`` keeps the occupied and virtual orbitals that
    # project onto ``avas_ao_labels`` above ``avas_threshold``, or the highest-weight ones up
    # to ``avas_max_size`` (``n_frozen_occ`` sets its core, ``n_virtual`` is not used).
    # ``none`` disables.  See ``selectors`` for the papers.
    selector: Literal[
        "none", "mulliken", "spade", "concentric-cl", "apc-concentric", "uno", "avas"
    ] = "concentric-cl"
    # ``concentric-cl``/``apc-concentric`` only: Fock shell expansions after shell 0.
    # 0 keeps just the fragment-spanned shell; higher values grow the candidate space
    # (and the qubit count) for accuracy.
    n_shells: int = 0
    # ``apc-concentric`` only: APC's active-space budget, as (nelec, norb).  Required
    # when selector="apc-concentric"; inert otherwise.
    apc_max_size: tuple[int, int] | None = None
    # ``apc-concentric`` only: pin the selection to exactly apc_max_size every cycle
    # (recommended) rather than dropping to an N_CSF budget, which can change the
    # active-space size as F_emb drifts under density feedback.
    apc_fixed: bool = True
    # How the active atoms split into PHYSICAL fragments, as consecutive counts in
    # ``active_atoms`` order.  ``None`` -> one fragment.  e.g. an OH dimer [1,5,7,11] ->
    # [2, 2], so ``mulliken`` cuts each shell independently and unions them (additivity).
    # Used ONLY by ``mulliken``; the rotating cuts anchor on the fragment union.
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
    # Pulay/DIIS acceleration of the density feedback instead of plain linear mixing.  Off
    # by default.  ``mix_alpha`` still governs the bootstrap cycle (before DIIS has two
    # vectors) and any cycle where the subspace matrix is singular.
    diis: bool = False
    diis_size: int = 8  # max (input, output) pairs kept in the DIIS subspace
    reseed_sqd: bool = True  # re-sample the SQD subspace each cycle

    # --- solver / sampling --- #
    solver: Literal["sqd", "fci"] = "sqd"
    handoff: Literal["in-process", "two-process"] = "in-process"
    sampler: SamplerKind = "aer"
    # Which Aer simulator: `--sampler aer` alone does not pin one, and None takes the
    # package default (MPS).  Validated against the installed Aer, so a typo fails at
    # construction.  Seeds do NOT reproduce across methods.
    aer_method: str | None = None
    # MPS only; the cap is what buys the memory saving, at the cost of an approximation.
    mps_max_bond_dimension: int | None = None
    mps_truncation_threshold: float | None = None
    backend: str | None = None  # runtime backend name; else least-busy
    optimization_level: int = 3  # runtime ISA-transpile level
    shots: int = 1_000
    seed: int = 24
    job_dir: Path | None = None
    output_path: Path | None = None
    diagnostics_csv: Path | None = None  # append per-cycle diagnostics here; None disables logging
    # SQD solver parameters
    sqd_method: Literal["exact", "qdrift"] = "qdrift"
    # Evolution time of the SqDRIFT sampling circuits.  It sets how far the samples
    # spread from the reference determinant, hence how much of the determinant space
    # configuration recovery can span -- too short and the subspace misses
    # configurations (measured on a nitrile CAS(6,6): 1.0 -> 35 distinct bitstrings and
    # +3.5 mHa, 3.0 -> 56 and exact).
    # Both are sweep axes, as in the reference workflow: the ensemble is built over
    # time x num_groups (e.g. --evolution_time 1,2,3 --num_groups 10,15,20) and every
    # circuit is pooled into one SQD run.  A scalar means a one-point axis.
    evolution_time: FloatSweep = [1.0]
    num_groups: IntSweep = [15]
    # For method="qdrift": random circuits per (time, num_groups) combination.
    num_randomizations: int = 500

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
        import uuid

        rank, size = _rank_size()

        if log is None:

            def log(msg: str = "") -> None:
                """Print only on rank 0 (keeps multi-rank output clean)."""
                if rank == 0:
                    print(msg)

        if size > 1:
            log(f"[running under MPI with {size} ranks; solve is on rank 0]")

        # Compute per-run identifiers for diagnostics logging (one per execution)
        run_id = uuid.uuid4().hex
        geometry_file = str(self.xyz) if self.xyz is not None else f"s26[{self.s26_index}]"
        geometry_parameter = None
        if self.xyz is not None and self.xyz.stem.isdigit():
            geometry_parameter = int(self.xyz.stem) / 10.0

        self._check_target_s2()  # before any EmbASI work, so a bad value fails fast
        log("== Step 1: EmbASI low-level projection embedding ==")
        source = f"xyz={self.xyz}" if self.xyz is not None else f"s26[{self.s26_index}]"
        log(f"   geometry: {source}")
        emb = self._build_adapter(parallel=size > 1)
        log(
            f"   {self.xc_hl}-in-{self.xc_ll} / {self.basis}, "
            f"active atoms {self.active_atoms}, projection=level-shift"
        )

        # Route on the high-level METHOD, not the solver: a functional is DFT-in-DFT
        # (Eq. 2), a wavefunction method is WF-in-DFT (Eq. 8).  They drive DIFFERENT
        # collective EmbASI entry points and must branch BEFORE the low-level call so
        # exactly one fires per rank, or the supersystem SCF runs twice.
        if self._is_dft_in_dft():
            return self._dft_in_dft(emb, log=log)

        # WF-in-DFT only.  Collective on every rank: EmbASI's supersystem SCF,
        # SPADE/Pipek-Mezey localisation, and the embedded Fock all run in here.
        emb.run_low_level(a_nmos=self.a_nmos)
        if self.relax_hf:
            log(
                "   relaxing the subsystem-A HF reference on A_HL "
                "(self-consistent, frozen v_emb/P_B)..."
            )
            emb.relax_active_hf()
        selector, virtual_localizer, orbital_builder = self._build_selector(
            emb, log=log, use_relaxed=self.relax_hf
        )
        solver = self._build_solver()
        output = self._run_outer_loop(
            emb,
            solver,
            selector,
            virtual_localizer,
            orbital_builder,
            rank=rank,
            log=log,
            run_id=run_id,
            geometry_file=geometry_file,
            geometry_parameter=geometry_parameter,
        )

        if self.output_path is not None:
            with self.output_path.open("a") as out:
                out.write(f"{self.xyz}; {output.total}\n")

        return output

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
    def _run_outer_loop(
        self,
        emb,
        solver,
        selector,
        virtual_localizer=None,
        orbital_builder=None,
        *,
        rank,
        log,
        run_id,
        geometry_file,
        geometry_parameter,
    ):
        """Steps 2-5, iterated until self-consistent (or ``max_cycles``).

        ``max_cycles == 1`` reproduces the single-pass flow exactly.  Otherwise each cycle
        builds orbitals, downfolds, solves, assembles the PbE energy and feeds the
        correlated density back, stopping once ``converge_on`` is met.

        MPI-safe: the solve goes through ``self._solve`` (rank 0 solves, result broadcast)
        and ``emb.feedback`` runs on every rank, since its ``construct_embedded_fock`` is
        collective.

        A stochastic solver (SQD) makes the fixed point noisy, so convergence is judged on
        running tolerances rather than bitwise.  ``--reseed_sqd`` chooses whether the sampled
        subspace is redrawn each cycle or frozen at cycle 0's density.

        The undamped map ``γ -> F(γ)`` oscillates and diverges here, so the fed-back density
        is linearly mixed with the previous one (``--mix_alpha`` ∈ (0, 1], the fraction of
        the new density).  Near a vanishing HOMO-LUMO gap -- bond dissociation -- mixing can
        fail outright, the iterates settling into a limit cycle no practical ``mix_alpha``
        escapes.  ``--diis`` replaces the blend with Pulay extrapolation over past outputs,
        whose coefficients adapt to the residual history; ``mix_alpha`` still governs the
        bootstrap cycle and any singular subspace.

        ``orbital_builder`` (``selector="apc-concentric"``, ``"avas"`` or ``"uno"``) bypasses
        ``emb.build_orbitals`` entirely: the builder calls it internally and returns the
        finished ``EmbeddedOrbitals``, so ``selector``/``virtual_localizer`` are ``None``
        with it.
        """
        prev_total: float | None = None
        prev_dm_a = None
        prev_fed = None
        energy = None
        diis_inputs: list[np.ndarray] = []
        diis_outputs: list[np.ndarray] = []
        diis_residuals: list[np.ndarray] = []

        for cycle in range(self.max_cycles):
            tag = "" if self.max_cycles == 1 else f" [cycle {cycle + 1}/{self.max_cycles}]"

            log(f"== Step 2: build subsystem-A orbitals and downfold =={tag}")
            # Beta's own orbital set, set only on the per-spin path; `None` everywhere
            # else keeps the restricted lift bit-identical.
            orbitals_b = None
            if self.spin_downfold:
                # Two orbital sets, each diagonalized in its own span(A), downfolded to an
                # (h1a, h1b) pair.  A *localiser* (spade, concentric-cl) is applied per
                # channel and reconciled to a common active-orbital count; the APC
                # `orbital_builder` pairs the channels into slots and selects once (see
                # `build_orbitals_apc_concentric_spin`).  An index `selector` (mulliken)
                # is refused: it ranks against one spin-summed Fock and would mix the
                # channels this path exists to separate.
                if selector is not None:
                    raise ValueError(
                        f"spin_downfold=True does not support selector={self.selector!r}: "
                        "index selection (mulliken) ranks columns against one spin-summed "
                        "Fock, which mixes the channels. Use --selector none, spade, "
                        "concentric-cl, apc-concentric, uno or avas, or cap with --n_virtual."
                    )
                if orbital_builder is not None and self.selector == "avas":
                    # Independent per-channel selection (`build_orbitals_avas_spin`): no
                    # pairing, so the principal cosines are the only cross-spin check.
                    spin_orbitals = orbital_builder(emb, spin=True)
                    info = emb._avas_info
                    for name, ch in zip(("alpha", "beta"), info["channels"]):
                        w_occ = ch["occ_weights"][::-1][: ch["n_active_occ"]]
                        w_virt = ch["virt_weights"][: ch["n_active_virt"]]
                        w_occ_s = " ".join(f"{w:.3f}" for w in w_occ) or "-"
                        w_virt_s = " ".join(f"{w:.3f}" for w in w_virt) or "-"
                        log(
                            f"   selector=avas {name}: kept {ch['n_active_occ']} occupied "
                            f"(weights {w_occ_s}) and {ch['n_active_virt']} virtual "
                            f"(weights {w_virt_s}); froze {ch['n_frozen']}"
                        )
                    log(
                        f"   alpha/beta active spaces: principal cosines "
                        f"{' '.join(f'{x:.3f}' for x in info['active_cosines'])} "
                        f"(max_size={tuple(self.avas_max_size)}, core spin {info['core_spin']})"
                    )
                elif orbital_builder is not None and self.selector == "uno":
                    spin_orbitals = orbital_builder(emb, spin=True)
                    info = emb._uno_info
                    act_occ = " ".join(f"{n:.3f}" for n in info["occupations"][info["active"]])
                    log(
                        f"   selector=uno (fill={info['fill']}) kept "
                        f"{spin_orbitals[0].n_active_orbitals} UNOs "
                        f"(occupations {act_occ}), froze {spin_orbitals[0].inactive.size} core; "
                        f"{info['n_fractional']} fractional UNOs in subsystem A; environment "
                        f"spin-common to {info['env_max_angle_deg']:.1e} deg; electrons outside "
                        f"the complement {info['electrons_lost']:.1e}"
                    )
                elif orbital_builder is not None:
                    spin_orbitals = orbital_builder(emb, spin=True)
                    overlap = emb._apc_spin_pair_overlap[spin_orbitals[0].active]
                    log(
                        f"   selector=apc-concentric (per-spin slots) kept "
                        f"{spin_orbitals[0].n_active_orbitals} slots, froze "
                        f"{spin_orbitals[0].inactive.size} doubly-occupied "
                        f"(max_size={self.apc_max_size}, fixed={self.apc_fixed}); "
                        f"min active pair overlap {overlap.min():.3f}"
                    )
                else:
                    spin_orbitals = emb.build_orbitals_spin(
                        n_frozen_occ=self.n_frozen_occ,
                        n_virtual=self.n_virtual,
                        virtual_localizer=virtual_localizer,
                        use_relaxed=self.relax_hf,
                    )
                # Not `str(EmbeddedOrbitals)`: each set describes ONE spin (`n_occ_b=None`),
                # so its restricted `__str__` would double the electron count.
                for name, orb in zip(("alpha", "beta "), spin_orbitals):
                    n_act_occ = orb.n_occ - orb.inactive.size
                    log(
                        f"   {name}: active {orb.n_active_orbitals} orbitals = "
                        f"{n_act_occ} occupied + {orb.n_active_orbitals - n_act_occ} "
                        f"virtual; {n_act_occ} electrons  [subsystem A pool: "
                        f"{orb.n_occ} occupied ({orb.inactive.size} frozen), "
                        f"{orb.coeff.shape[1] - orb.n_occ} virtual]"
                    )
                ham = emb.embedded_hamiltonian_spin(spin_orbitals)
                # Alpha represents the pair wherever one set suffices (logging, the
                # restricted `rdm1_ao` fallback).  `orbitals_b` carries beta's own set: the
                # channels are diagonalized in DIFFERENT spans, so lifting beta's active RDM
                # through alpha's columns gives the right trace and the wrong matrix.
                orbitals, orbitals_b = spin_orbitals
                leaks = ham.meta["p_b_leak_per_spin"]
                log(
                    f"   norb={ham.norb} nelec={ham.nelec} e_core={ham.e_core:.6f} Ha "
                    f"(P_B leak per spin {leaks[0]:.2e} / {leaks[1]:.2e}; "
                    f"spin-resolved ERIs: {ham.has_spin_dependent_eri})"
                )
            elif orbital_builder is not None:
                orbitals = orbital_builder(emb)
            else:
                orbitals = emb.build_orbitals(
                    n_frozen_occ=self.n_frozen_occ,
                    n_virtual=self.n_virtual,
                    selector=selector,
                    virtual_localizer=virtual_localizer,
                    use_relaxed=self.relax_hf,
                )
            if not self.spin_downfold:
                log(f"   {orbitals}")
            if self.spin_downfold:
                pass  # already logged per channel above
            elif orbital_builder is not None and self.selector == "avas":
                info = emb._avas_info
                w_occ = " ".join(
                    f"{w:.3f}" for w in info["occ_weights"][::-1][: info["n_active_occ"]]
                )
                w_virt = " ".join(f"{w:.3f}" for w in info["virt_weights"][: info["n_active_virt"]])
                cut = (
                    f"threshold={self.avas_threshold}"
                    if self.avas_max_size is None
                    else f"max_size={self.avas_max_size}"
                )
                log(
                    f"   selector=avas ({self.avas_ao_labels}, {cut}) "
                    f"kept {info['n_active_occ']} occupied (weights {w_occ or '-'}), "
                    f"{info['n_somo']} singly occupied and {info['n_active_virt']} virtual "
                    f"(weights {w_virt or '-'}); froze {orbitals.inactive.size} occupied"
                )
            elif orbital_builder is not None:
                n_kept_virt = orbitals.n_active_orbitals - (orbitals.n_occ - orbitals.inactive.size)
                log(
                    f"   selector=apc-concentric picked {n_kept_virt} virtuals and froze "
                    f"{orbitals.inactive.size} occupied (max_size={self.apc_max_size}, "
                    f"fixed={self.apc_fixed})"
                )
            elif selector is not None or virtual_localizer is not None:
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
            if not self.spin_downfold:
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
            if "n_distinct_bitstrings" in result.diagnostics:
                # SQD's subspace coverage and spin: a subspace spanned by too few sampled
                # configurations sits above FCI and can drift in <S^2>, invisibly otherwise.
                dg = result.diagnostics
                target = ""
                if dg.get("spin_sq_target") is not None:
                    target = f" (target {dg['spin_sq_target']:g})"
                times = ", ".join(f"{t:g}" for t in self.evolution_time)
                log(
                    f"   SQD: {dg['n_distinct_bitstrings']} distinct sampled bitstrings from "
                    f"{dg.get('n_shots', self.shots)} shots, {dg.get('n_circuits', '?')} "
                    f"circuit(s), evolution_time {times}; <S^2> = "
                    f"{dg.get('spin_square', float('nan')):.4f}{target}; "
                    f"{dg.get('n_iterations', '?')} recovery iteration(s)"
                )
            deviation = result.check_particle_number(ham.nelec, atol=self.rdm_trace_tol)
            if result.is_spin_resolved:
                sz_dev = result.check_spin_sector(ham.nelec, atol=self.rdm_trace_tol)
                log(f"   Sz check: tr(γa) - tr(γb) matches nelec (off by {sz_dev:+.1e})")
            log(
                f"   trace(rdm1) = {np.trace(result.rdm1):.4f} "
                f"(expected {sum(ham.nelec)}, off by {deviation:+.1e})"
            )

            log(f"== Step 4: assemble the projection-based-embedding energy =={tag}")
            energy = emb.projection_energy(result, orbitals, orbitals_b)
            log("   E = E_low(total) - E_low(A) + E_high(A) + corr")
            log(
                f"     = {energy.e_low_total:.6f} - {energy.e_low_A:.6f} "
                f"+ {energy.e_high_A:.6f} + {energy.correction:.6f}"
            )
            log(f"     = {energy.total:.6f} Ha")
            log(f"   tr[γ̃^A P_B] = {energy.projector_leak:.2e} Ha (should be ~0)")
            # E_high(A) is rebased onto E_low(A)'s (ghosted subsystem-A) nuclear frame
            # before the subtraction, or the two A-terms sit on incompatible frames.
            # Surfaced so the shift is not a silent adjustment.
            log(f"   footing shift applied to E_high(A): {energy.footing_shift:.6f} Ha")

            # `converge_on` picks the criterion: "energy" stops once |ΔE| < e_tol, the
            # robust choice under a stochastic solver whose density carries sampling noise;
            # "energy_and_density" also requires max|Δγ^A| < rho_tol.
            dm_a_now = np.asarray(emb._dm_a)
            de = None
            drho = None
            converged = False
            if prev_total is not None:
                de = abs(energy.total - prev_total)
                drho = float(np.abs(dm_a_now - prev_dm_a).max())
                log(f"   Δ: |ΔE| = {de:.2e} Ha, max|Δγ^A| = {drho:.2e}")
                energy_ok = de < self.e_tol
                density_ok = drho < self.rho_tol
                converged = energy_ok and (density_ok or self.converge_on == "energy")

            is_last_cycle = cycle == self.max_cycles - 1
            total_cycles = (cycle + 1) if (converged or is_last_cycle) else None

            self._log_diagnostics_cycle(
                run_id=run_id,
                geometry_file=geometry_file,
                geometry_parameter=geometry_parameter,
                cycle=cycle,
                converged=converged,
                total_cycles=total_cycles,
                orbitals=orbitals,
                ham=ham,
                result=result,
                energy=energy,
                solver=solver,
                delta_e=de,
                max_delta_gamma_a=drho,
                rank=rank,
                log=log,
            )

            if converged:
                crit = "|ΔE|" if self.converge_on == "energy" else "|ΔE| and max|Δγ^A|"
                log(f"   converged ({crit}) after {cycle + 1} cycles.")
                return energy
            prev_total, prev_dm_a = energy.total, dm_a_now

            if is_last_cycle:
                if self.max_cycles > 1:
                    log(f"   reached max_cycles={self.max_cycles} without convergence.")
                break

            log(f"== Step 5: feed the correlated 1-RDM back into the embedding =={tag}")

            # An unrestricted result keeps its spin resolution: `fed` stays the spin-summed
            # total (what the |Δγ| diagnostic reads) while `fed_split` carries the pair
            # actually fed back.  Summing the pair here would discard the polarisation and
            # make the loop a spin-summed fixed point.
            fed_split: tuple[np.ndarray, np.ndarray] | None = None
            if getattr(emb, "unrestricted", False) and result.is_spin_resolved:
                fed_a, fed_b = emb.rdm1_ao_spin(result.rdm1a, result.rdm1b, orbitals, orbitals_b)
                fed = fed_a + fed_b
                fed_split = (fed_a, fed_b)
            else:
                fed = emb.rdm1_ao(result.rdm1, orbitals)
            # Spin-resolved feedback: DIIS and the linear mixing act on the STACKED pair
            # rather than the total.  One coefficient set still covers both channels (they
            # share a fixed point), but the residual now measures each channel's own error,
            # so a badly-converging channel is damped on its own terms instead of having its
            # error masked by cancellation in the sum.
            spin_mixed = fed_split is not None
            vec_now = np.stack(emb._dm_a_spin) if (spin_mixed and emb._dm_a_spin) else dm_a_now
            vec_fed = np.stack(fed_split) if spin_mixed else fed
            if spin_mixed and np.asarray(vec_now).shape != np.asarray(vec_fed).shape:
                # No per-spin input to form a residual against (first cycle after a
                # restricted start): fall back to the spin-summed vector this round.
                vec_now, vec_fed, spin_mixed = dm_a_now, fed, False
                fed_split = None

            mixing_desc = f"mix_alpha={self.mix_alpha}"
            extrapolated = None
            if self.diis:
                # The convergence vector's SHAPE can change between cycles, since
                # `spin_mixed` is demoted whenever the adapter has no per-spin `gamma^A` to
                # form a residual against.  A subspace spanning both shapes is not a
                # subspace: `np.vdot` raises `ValueError` -- NOT the `LinAlgError`
                # `_diis_extrapolate` guards -- so it escapes and aborts the whole loop.
                # Drop the stale history and restart the subspace from this cycle.
                if (
                    diis_residuals
                    and np.asarray(diis_residuals[-1]).shape != np.asarray(vec_fed - vec_now).shape
                ):
                    log(
                        "   DIIS subspace reset: the convergence vector changed shape "
                        f"({'spin-summed -> per-spin' if spin_mixed else 'per-spin -> spin-summed'})."
                    )
                    diis_inputs.clear()
                    diis_outputs.clear()
                    diis_residuals.clear()
                diis_inputs.append(vec_now)
                diis_outputs.append(vec_fed)
                diis_residuals.append(vec_fed - vec_now)
                if len(diis_residuals) > self.diis_size:
                    diis_inputs.pop(0)
                    diis_outputs.pop(0)
                    diis_residuals.pop(0)
                if len(diis_residuals) >= 2:
                    extrapolated = self._diis_extrapolate(diis_residuals, diis_outputs)
                if extrapolated is not None:
                    vec_fed = extrapolated
                    per = " per-spin" if spin_mixed else ""
                    mixing_desc = f"diis(n={len(diis_residuals)}){per}"
                else:
                    mixing_desc = f"mix_alpha={self.mix_alpha} (DIIS bootstrap/fallback)"
            if extrapolated is None and prev_fed is not None and self.mix_alpha != 1.0:
                # Linear mixing: DIIS's bootstrap cycle (< 2 vectors), its singular-subspace
                # fallback, and --diis=False.  `prev_fed` carries the previous cycle's vector
                # shape, so only mix when the two agree.
                if np.asarray(prev_fed).shape == np.asarray(vec_fed).shape:
                    vec_fed = self.mix_alpha * vec_fed + (1.0 - self.mix_alpha) * prev_fed
            prev_fed = vec_fed
            # Unstack: `fed` stays the spin-summed total (the |Δ| diagnostic and the
            # restricted path read it), `fed_split` the mixed channels.
            if spin_mixed:
                arr = np.asarray(vec_fed)
                fed_split = (arr[0], arr[1])
                fed = arr[0] + arr[1]
            else:
                fed = np.asarray(vec_fed)
            # No rescaling needed: the pair itself went through DIIS/mixing above, so
            # `fed_split` and `fed` are consistent by construction (`fed` is their sum).
            fed_in: np.ndarray | tuple[np.ndarray, np.ndarray] = (
                fed_split if fed_split is not None else fed
            )
            # Currently commented out the old outer loop behaviour where the
            # low level potential is re-constructed and subtracted from the
            # supersystem embedding potential.
            # emb.run_low_level(dma_in=fed_in, dmb_in=emb._dm_b)
            emb.run_low_level_a_only(dma_in=fed_in, dmb_in=emb._dm_b)
            spin_note = "" if fed_split is None else " [per-spin pair]"
            log(f"   embedded Fock rebuilt at γ̃^A + γ^B ({mixing_desc}){spin_note}.")

        return energy

    def _log_diagnostics_cycle(
        self,
        *,
        run_id: str,
        geometry_file: str,
        geometry_parameter: float | None,
        cycle: int,
        converged: bool,
        total_cycles: int | None,
        orbitals,
        ham,
        result,
        energy,
        solver,
        delta_e: float | None,
        max_delta_gamma_a: float | None,
        rank: int,
        log,
    ) -> None:
        """Log one diagnostics row for the current cycle, if logging is enabled.

        Logging is strictly optional and non-fatal: if disabled (diagnostics_csv is None),
        this is a no-op; if enabled but fails (bad path, missing data), a warning is logged
        and execution continues. This ensures diagnostic failures never abort the outer loop.
        """
        if self.diagnostics_csv is None or rank != 0:
            return

        try:
            from embasi_qiskit_integration.diagnostics_csv import (
                build_diagnostics_row,
                append_diagnostics_row,
            )

            row = build_diagnostics_row(
                self,
                run_id=run_id,
                geometry_file=geometry_file,
                geometry_parameter=geometry_parameter,
                cycle=cycle,
                converged=converged,
                total_cycles=total_cycles,
                orbitals=orbitals,
                ham=ham,
                result=result,
                energy=energy,
                solver=solver,
            )
            # Add delta_E and max_delta_gamma_A (computed in convergence check, may be None)
            row["delta_E"] = delta_e
            row["max_delta_gamma_A"] = max_delta_gamma_a

            append_diagnostics_row(self.diagnostics_csv, row)
        except Exception as exc:
            log(f"   [diagnostics] skipped cycle {cycle + 1}: {type(exc).__name__}: {exc}")

    @staticmethod
    def _diis_extrapolate(
        residuals: list[np.ndarray], outputs: list[np.ndarray]
    ) -> np.ndarray | None:
        """Pulay/DIIS extrapolation; see :func:`diis_extrapolate`."""
        return diis_extrapolate(residuals, outputs)

    def _maybe_reseed(self, solver, cycle: int) -> None:
        """Advance the SQD seed each cycle unless the subspace is carried over.

        Only SQDSolver has a ``seed``; FCI is deterministic and ignores this.
        With ``reseed_sqd`` the seed advances per cycle so each iteration draws a
        fresh sampled subspace; without it the seed is held so the subspace is
        reused (frozen at the cycle-0 density).

        Assigns :func:`seed_for_cycle`'s *absolute* value rather than bumping
        ``solver.seed`` in place.  Same sequence, but now a pure function of
        ``(self.seed, cycle)``, so an out-of-process driver reproduces the schedule
        from the cycle index alone.  Note it must read ``self.seed`` (the configured
        base), never ``solver.seed`` (already advanced) -- otherwise the triangular
        accumulation happens twice.
        """
        if cycle == 0 or not hasattr(solver, "seed"):
            return
        if self.reseed_sqd and solver.seed is not None:
            solver.seed = seed_for_cycle(self.seed, cycle, reseed=self.reseed_sqd)

    # ---------------- construction ---------------- #
    def _build_adapter(self, *, parallel: bool) -> ProjectionEmbeddingAdapter:
        """Set up ProjectionEmbedding exactly as the EmbASI PySCF example does."""
        return build_adapter(self, parallel=parallel)

    def _build_atoms(self) -> tuple[Any, int]:
        """Return ``(ase.Atoms, charge)`` for the requested geometry source."""
        return build_atoms(self)

    def _build_selector(
        self, emb: ProjectionEmbeddingAdapter, *, log=None, use_relaxed: bool = False
    ):
        """Build the active-virtual shaping hook from ``self.selector``.

        See :func:`build_selector`, which this delegates to.
        """
        return build_selector(self, emb, log=log, use_relaxed=use_relaxed)

    def _check_target_s2(self) -> None:
        """Reject a ``--target_s2`` that no spin state has, or the ``--spin`` sector excludes."""
        if self.target_s2 is None:
            return
        # --target_s2 is S(S+1), not 2S: only 0, 0.75, 2, 3.75, 6, ... exist, and S must be
        # reachable from the M_s the --spin sector fixes (S >= |M_s|, same parity).
        two_s = np.sqrt(1.0 + 4.0 * float(self.target_s2)) - 1.0
        if abs(two_s - round(two_s)) > 1e-6:
            valid = ", ".join(f"{0.25 * n * (n + 2):g}" for n in range(6))
            raise ValueError(
                f"--target_s2 {self.target_s2:g} is not S(S+1) for any spin S (valid: "
                f"{valid}, ...).  It is <S^2>, not 2S: singlet 0, doublet 0.75, "
                "triplet 2, quintet 6."
            )
        if round(two_s) < abs(self.spin) or (round(two_s) - abs(self.spin)) % 2:
            raise ValueError(
                f"--target_s2 {self.target_s2:g} (2S = {round(two_s)}) is not reachable "
                f"from --spin {self.spin} (2M_s): need 2S >= |2M_s| with the same parity"
            )

    def _build_solver(self):
        from embasi_qiskit_integration.solvers import FCISolver, SQDSolver

        if self.target_s2 is not None:
            self._check_target_s2()
            # <S^2> is evaluated assuming alpha and beta share one orbital set, which the
            # per-spin downfold only guarantees with the UNO selector.  Both solvers take
            # it: FCI as a fix_spin_ penalty, SQD as its spin_sq (the same penalty, applied
            # to the subspace diagonalisation).
            if self.spin_downfold and self.selector != "uno":
                raise ValueError(
                    "--target_s2 with --spin_downfold needs --selector uno (shared alpha/beta "
                    f"orbitals); selector={self.selector!r} gives each spin its own orbitals"
                )
        if self.solver == "fci":
            return FCISolver(target_s2=self.target_s2, spin_average=self.fci_spin_average)

        from embasi_qiskit_integration.circuit_run import build_sampler

        # ``runtime`` plugs in real quantum hardware: configured IBM Quantum
        # credentials, least-busy backend unless --backend names one.
        sampler = build_sampler(
            self.sampler,
            counts=str(DATA_DIR / "mock_counts.json"),
            backend=self.backend,
            optimization_level=self.optimization_level,
            default_shots=self.shots,
            aer_method=self.aer_method,
            mps_max_bond_dimension=self.mps_max_bond_dimension,
            mps_truncation_threshold=self.mps_truncation_threshold,
        )
        return SQDSolver(
            sampler,
            shots=self.shots,
            seed=self.seed,
            method=self.sqd_method,
            evolution_time=self.evolution_time,
            num_groups=self.num_groups,
            num_randomizations=self.num_randomizations,
            spin_sq=self.target_s2,
        )

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
