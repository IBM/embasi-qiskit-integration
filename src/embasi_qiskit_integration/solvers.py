# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Active-space solvers over an :class:`EmbeddedHamiltonian`.

Phase 2 provides the classical reference (:class:`FCISolver`) and the
``cheap_ccsd_t2`` helper used to seed the quantum ansatz. :class:`SQDSolver`
is completed in Phase 6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any

import numpy as np
from embasi_qiskit_integration._versions import collect_versions

# Only probes importability (find_spec) -- pulls in neither pyomo nor
# qiskit-fermions, so the classical/mock paths stay dependency-free.
from embasi_qiskit_integration.circuit_generator.relabel import relabel_available
from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult


class ActiveSpaceSolver(ABC):
    """Solve an active-space Hamiltonian for its ground-state energy and RDMs."""

    @abstractmethod
    def solve(self, ham: EmbeddedHamiltonian) -> SolverResult: ...


def _sampler_backend_diagnostics(sampler) -> dict:
    """Simulation-method provenance for whichever sampler ran, if it has any.

    Duck-typed rather than isinstance-checked so a custom sampler exposing the same
    attributes is reported too, and a sampler with none (``MockSampler``,
    ``RuntimeSampler``) contributes nothing rather than ``None`` entries.
    """
    out: dict = {}
    method = getattr(sampler, "method", None)
    if method is not None:
        out["sampler_method"] = method
    for attr, key in (
        ("mps_max_bond_dimension", "mps_max_bond_dimension"),
        ("mps_truncation_threshold", "mps_truncation_threshold"),
    ):
        value = getattr(sampler, attr, None)
        if value is not None:
            out[key] = value
    return out


class FCISolver(ActiveSpaceSolver):
    """Exact full configuration interaction via PySCF — the numerical oracle.

    ``target_s2`` constrains the solve to a spin state: a spin penalty (``fix_spin_``) on
    ``nroots`` roots, keeping the lowest root whose <S^2> matches.  Near dissociation the
    spin multiplets become near-degenerate and the lowest state in a given M_s sector can
    have higher S (a triplet run landing on a quintet); a single penalised root can also
    converge onto an excited state of the right spin.  <S^2> is evaluated for alpha and
    beta sharing one orbital set -- the case this is meant for (a restricted downfold, or
    the UNO selector's shared orbitals).  ``None`` (default) keeps the unconstrained solve.
    With a target, an ``(h1a, h1b)`` pair differing by less than ``spin_free_tol`` (or any
    pair, with ``spin_average``) is averaged and solved spin-adapted (``direct_spin1``), so
    the roots are exact spin states.  A genuinely spin-dependent pair (a KS low level) is
    solved with the penalised ``direct_uhf``, whose roots are only approximately pure: the
    lowest root within ``s2_tol`` of the target is kept, and its <S^2> is reported.  The
    reported energy is always <H> of the kept root, never the penalised eigenvalue.
    """

    def __init__(
        self,
        target_s2: float | None = None,
        nroots: int = 3,
        spin_free_tol: float = 1.0e-3,
        spin_average: bool = False,
        s2_tol: float = 0.05,
    ) -> None:
        self.target_s2 = target_s2
        self.nroots = nroots
        self.spin_free_tol = spin_free_tol
        self.spin_average = spin_average
        self.s2_tol = s2_tol

    def solve(self, ham: EmbeddedHamiltonian) -> SolverResult:
        from pyscf import fci

        norb = ham.norb
        spin_dependent = ham.is_spin_dependent
        averaged = 0.0
        if spin_dependent and self.target_s2 is not None:
            # A spin target asks for a spin eigenstate, which needs a spin-free Hamiltonian.
            # With orbitals shared by both channels (the only case target_s2 is allowed for)
            # an h1a/h1b difference at the level of SCF noise is not physics -- an HF low
            # level with a closed-shell environment is exactly spin-free -- but near
            # dissociation, where the multiplets are near-degenerate, even 1e-4 Ha mixes
            # them completely.  Average it away and solve spin-adapted; a genuinely
            # spin-dependent pair (a KS low level) stays on the penalised direct_uhf path.
            averaged = float(np.abs(np.asarray(ham.h1a) - np.asarray(ham.h1b)).max())
            if averaged < self.spin_free_tol or self.spin_average:
                if averaged >= self.spin_free_tol:
                    import warnings

                    warnings.warn(
                        f"spin_average removes a genuine alpha/beta difference of "
                        f"{averaged:.2e} Ha from h1 (above spin_free_tol={self.spin_free_tol:g}): "
                        "the embedding potential is spin-dependent (a KS low level), so the "
                        "averaged Hamiltonian differs from the embedded one -- 45 mHa on a "
                        "nitrile triplet at |h1a-h1b| = 0.39 Ha.  The default penalised solve "
                        "keeps the physical difference.",
                        stacklevel=2,
                    )
                spin_dependent = False
        if spin_dependent:
            # A genuine spin-dependent downfold: the two channels see different
            # one-body operators, which `direct_spin1` cannot express (it takes a
            # single h1e and distinguishes spin only through `nelec`).
            # `direct_uhf` takes the (h1a, h1b) pair, and (eri_aa, eri_ab, eri_bb)
            # for the two-body part -- the same spatial-orbital ERIs in all three
            # slots here, since h2 is spin-free by construction.
            solver = fci.direct_uhf
            h1e: Any = (ham.h1a, ham.h1b)
            if ham.has_spin_dependent_eri:
                # (aa, ab, bb) over the two different orbital sets -- what direct_uhf
                # actually wants (see its absorb_h1e, which unpacks exactly this order).
                eri: Any = ham.h2_spin
                tag = "pyscf-fci-uhf"
            else:
                # Spin-free h2 in all three slots: correct only when both channels share
                # an orbital set, which a per-spin downfold does not.
                eri = (ham.h2, ham.h2, ham.h2)
                tag = "pyscf-fci-uhf-spinfree-eri"
        else:
            solver = fci.direct_spin1
            h1e, eri, tag = ham.h1, ham.h2, "pyscf-fci"
            if ham.is_spin_dependent:
                # The averaged pair (see above).
                h1e = 0.5 * (np.asarray(ham.h1a) + np.asarray(ham.h1b))
                tag = f"pyscf-fci-spin-averaged(|h1a-h1b|={averaged:.1e})"

        if self.target_s2 is None:
            e, ci = solver.kernel(h1e, eri, norb, ham.nelec)
        else:
            if solver is fci.direct_uhf:
                # PySCF's fix_spin_ refuses direct_uhf, so apply the same penalty here.
                obj = _spin_penalised_uhf(self.target_s2)
            else:
                obj = solver.FCISolver()
                fci.addons.fix_spin_(obj, ss=self.target_s2)
            obj.nroots = self.nroots
            _, cis = obj.kernel(h1e, eri, norb, ham.nelec)
            cis = cis if self.nroots > 1 else [cis]
            # Each eigenvalue is <H> + shift * <(S^2 - t)^2>.  On the spin-adapted path the
            # roots are spin eigenstates and the penalty vanishes, but a genuinely
            # spin-dependent pair breaks [H, S^2]: the roots are contaminated and the penalty
            # would land in the energy (33 mHa on a triplet at |h1a-h1b| ~ 0.4 Ha).  Re-evaluate
            # <H> with the module-level (unpenalised) energy, and choose among roots by it.
            energies = np.array([solver.energy(h1e, eri, c, norb, ham.nelec) for c in cis])
            s2s = np.array([fci.spin_op.spin_square0(c, norb, ham.nelec)[0] for c in cis])
            # A spin-free solve gives exact spin states; a genuinely spin-dependent one
            # (penalised direct_uhf) only approximately pure ones, so allow s2_tol there.
            tol = s2_tol_used = 1e-3 if not spin_dependent else self.s2_tol
            ok = np.flatnonzero(np.abs(s2s - self.target_s2) < tol)
            if not ok.size:
                raise RuntimeError(
                    f"no FCI root with <S^2> = {self.target_s2} (+/- {tol:g}) among "
                    f"{self.nroots}: {s2s}"
                    + (
                        "; the downfold is genuinely spin-dependent (|h1a-h1b| = "
                        f"{averaged:.1e}): consider --fci_spin_average"
                        if spin_dependent
                        else ""
                    )
                )
            best = ok[np.argmin(energies[ok])]
            e, ci = float(energies[best]), cis[best]
            tag += f"-s2={self.target_s2:g}(got {s2s[best]:.4f}, tol {s2_tol_used:g})"
        rdm1a, rdm1b = solver.make_rdm1s(ci, norb, ham.nelec)
        if spin_dependent:
            # `direct_uhf` returns spin-resolved blocks, so combine them here: rdm1 is the
            # sum, and rdm2 is `aa + bb + ab + ab^T` since the alpha-beta block appears
            # once in each ordering.
            _, rdm2s = solver.make_rdm12s(ci, norb, ham.nelec)
            aa, ab, bb = (np.asarray(x) for x in rdm2s)
            rdm2 = aa + bb + ab + ab.transpose(2, 3, 0, 1)
            rdm1 = np.asarray(rdm1a) + np.asarray(rdm1b)
        else:
            rdm1, rdm2 = solver.make_rdm12(ci, norb, ham.nelec)
        return SolverResult(
            energy=e + ham.e_core,
            rdm1=np.asarray(rdm1),
            rdm2=np.asarray(rdm2),
            rdm1a=np.asarray(rdm1a),
            rdm1b=np.asarray(rdm1b),
            diagnostics={"solver": tag},
        )


def _spin_penalised_uhf(target_s2: float, shift: float = 0.2):
    """A ``direct_uhf`` FCI solver with the penalty ``shift * (S^2 - target_s2)^2``.

    PySCF's ``fci.addons.fix_spin_`` does not support ``direct_uhf``.  The penalty is added
    to each Hamiltonian application through ``fci.spin_op.contract_ss``, which treats alpha
    and beta orbital ``p`` as the same spatial orbital -- valid when the two channels share
    one orbital set (the UNO selector), even though their one-body operators differ.
    """
    from pyscf.fci import direct_uhf, spin_op

    class _Penalised(direct_uhf.FCISolver):
        # PySCF diagonalizes small CI spaces (<= pspace_size determinants, 400 by default --
        # a CAS(6,6)) directly from `pspace`, never calling contract_2e, which would silently
        # drop the penalty.  Force the iterative solver so every H.c goes through it.
        davidson_only = True

        def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
            ci1 = super().contract_2e(eri, fcivec, norb, nelec, link_index, **kwargs)
            vec = np.asarray(fcivec).reshape(ci1.shape)
            tmp = spin_op.contract_ss(vec, norb, nelec).reshape(ci1.shape) - target_s2 * vec
            tmp = spin_op.contract_ss(tmp, norb, nelec).reshape(ci1.shape) - target_s2 * tmp
            return ci1 + shift * tmp

    return _Penalised()


def _rhf_from_integrals(ham: EmbeddedHamiltonian):
    """Build a PySCF RHF object over the bare (h1, h2, e_core) integrals.

    Follows the custom-Hamiltonian route: a bare ``Mole`` with the electron
    count set, overridden ``get_hcore``/``get_ovlp``/``energy_nuc`` and the
    two-electron integrals injected via ``_eri`` (8-fold packed).

    Restricted: this seeds the deferred LUCJ ansatz only. An open-shell ``nelec`` sets
    ``mol.spin`` and PySCF will dispatch accordingly, but the resulting amplitudes are
    not validated for that case -- see the LUCJ note in the module docstring.
    """
    from pyscf import ao2mo, gto, scf

    norb = ham.norb
    nelec = sum(ham.nelec)

    mol = gto.M(verbose=0)
    mol.nelectron = nelec
    mol.spin = ham.nelec[0] - ham.nelec[1]
    mol.incore_anyway = True

    mf = scf.RHF(mol)
    h1 = ham.h1
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(8, np.asarray(ham.h2), norb)
    e_core = ham.e_core
    mf.energy_nuc = lambda *args: e_core
    return mf, mol


def cheap_ccsd_t2(ham: EmbeddedHamiltonian) -> np.ndarray:
    """RHF + CCSD on the bare integrals, returning the t2 amplitudes.

    Used to seed the LUCJ ansatz (Phase 4), which is on hold; this stays
    closed-shell, matching :func:`_rhf_from_integrals`'s restricted reference.
    """
    from pyscf import cc

    mf, _mol = _rhf_from_integrals(ham)
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    return np.asarray(mycc.t2)


class SQDSolver(ActiveSpaceSolver):
    """Sample-based Quantum Diagonalization solver.

    Runs the full quantum path for an active-space Hamiltonian:

    1. build the SqDRIFT ansatz as *bare* evolution circuits
       (:mod:`~embasi_qiskit_integration.circuit_generator.sqdrift`),
    2. resolve and prepend the reference determinant
       (:mod:`~embasi_qiskit_integration.circuit_run.prep`),
    3. sample the batch with the injected ``sampler``,
    4. pool the counts and hand them to the SQD driver for the energy and RDMs.

    The ``sampler`` follows the :class:`~embasi_qiskit_integration.circuit_run.
    base.BitstringSampler` protocol. A ``MockSampler`` replays frozen counts and
    declares ``requires_circuit = False``, so no circuit is built at all
    (deterministic CI without the qiskit-fermions dependency); Aer/Runtime
    samplers run the real circuits.

    With ``method="qdrift"`` the ansatz is an *ensemble*: ``num_randomizations``
    circuits are each sampled at ``shots`` and their counts pooled, so the total
    shot budget is ``num_randomizations * shots``. The default of 1 keeps a single
    circuit, matching the exact-evolution path.

    ``optimize`` (default True) lets the generator
    relabel the fermionic modes to shorten each circuit. That makes every circuit
    sample in its *own* permuted mode order, which this class handles end to end:
    the reference determinant is prepared in the permuted order, and each
    circuit's counts are mapped back to the original order *before* the ensemble
    is pooled. ``workers`` shards circuit construction across processes.
    """

    def __init__(
        self,
        sampler,
        *,
        shots: int = 1_000,
        ansatz: str = "sqdrift",
        method: str = "qdrift",
        evolution_time: float = 1.0,
        num_groups: int = 15,
        num_randomizations: int = 500,
        initial_state_bitstring: str | None = None,
        samples_per_batch: int = 300,
        num_batches: int = 5,
        max_iterations: int = 5,
        seed: int | None = None,
        optimize: bool | None = True,
        time_limit: float = 10.0,
        canonical_permutation: bool = False,
        workers: int = 1,
        symmetrize_spin: bool | None = None,
        spin_sq: float | None = None,
    ):
        self.sampler = sampler
        self.shots = shots
        self.ansatz = ansatz
        self.method = method
        self.evolution_time = evolution_time
        self.num_groups = num_groups
        self.num_randomizations = num_randomizations
        self.initial_state_bitstring = initial_state_bitstring
        self.samples_per_batch = samples_per_batch
        self.num_batches = num_batches
        self.max_iterations = max_iterations
        self.seed = seed
        self.optimize = relabel_available() if optimize is None else optimize
        self.time_limit = time_limit
        # Forwarded to the generator: False (default) uses the MILP solver's own
        # permutation, True derives a reproducible one at the cost of a second
        # pass-manager run per randomization. See build_sqdrift_circuits.
        self.canonical_permutation = canonical_permutation
        self.workers = workers
        # None -> run_sqd derives it from the sector (an open shell must not merge the
        # alpha/beta CI pools).  Forwarded verbatim so an explicit choice still wins.
        self.symmetrize_spin = symmetrize_spin
        # Target S(S+1). None (default) imposes no projection: nelec fixes Sz, not S, so
        # set this when a wrong-multiplicity state could lie below the intended one.
        self.spin_sq = spin_sq
        self._permutations: list[list[int] | None] = []

    def solve(self, ham: EmbeddedHamiltonian) -> SolverResult:
        from embasi_qiskit_integration.circuit_run import merge_counts, unpermute_counts_list
        from embasi_qiskit_integration.sqd.driver import run_sqd

        circuits = self.build_circuits(ham)
        counts_list = self.sampler.run(circuits, self.shots, seed=self.seed)

        # Undo each circuit's mode relabeling BEFORE pooling
        counts_list = unpermute_counts_list(counts_list, self._permutations)
        # Pool the ensemble into the single empirical distribution SQD consumes.
        counts = merge_counts(counts_list)

        res = run_sqd(
            ham,
            counts,
            samples_per_batch=self.samples_per_batch,
            num_batches=self.num_batches,
            max_iterations=self.max_iterations,
            symmetrize_spin=self.symmetrize_spin,
            spin_sq=self.spin_sq,
            seed=self.seed,
        )
        res.diagnostics.update(
            shots=self.shots,
            ansatz=self.ansatz,
            method=self.method,
            n_circuits=len(circuits),
            sampler=type(self.sampler).__name__,
            **_sampler_backend_diagnostics(self.sampler),
            versions=collect_versions(),
            seed=self.seed,
            fcidump_sha=ham.meta.get("sha"),
            optimize=self.optimize,
            n_permuted=sum(1 for p in self._permutations if p is not None),
            workers=self.workers,
        )
        return res

    def build_circuits(self, ham: EmbeddedHamiltonian) -> list:
        """Build the sampling circuits: bare ansatz + resolved initial state.

        Public so a caller can inspect, transpile or sample the exact circuits
        :meth:`solve` would use without re-deriving them.

        Also records each circuit's mode permutation in ``self._permutations`` so
        :meth:`solve` can undo the relabeling on the sampled counts.

        A sampler that declares ``requires_circuit = False`` (``MockSampler``)
        gets a single ``None`` placeholder instead, so pure-replay runs never
        import qiskit-fermions.
        """
        if not getattr(self.sampler, "requires_circuit", True):
            # Replayed counts are already in the original mode order.
            self._permutations = [None]
            return [None]

        if self.ansatz != "sqdrift":
            raise ValueError(
                f"unknown ansatz {self.ansatz!r}; only 'sqdrift' is available (LUCJ is on hold)"
            )

        from qiskit import transpile
        from qiskit_aer import AerSimulator

        from embasi_qiskit_integration.circuit_generator.sqdrift import build_sqdrift_circuits
        from embasi_qiskit_integration.circuit_run.prep import (
            compose_full_circuit,
            resolve_initial_state,
        )

        # Bare evolution circuits: the reference determinant is chosen here, at
        # run time, rather than baked in by the generator.
        # Every sweep axis is pinned to a single value here: a solver call wants a
        # definite ensemble size (num_randomizations), not the generator's default
        # time x num_groups sweep, which would build thousands of circuits per solve.
        result = build_sqdrift_circuits(
            ham,
            method=self.method,
            time=self.evolution_time,
            num_groups=self.num_groups,
            num_randomizations=self.num_randomizations,
            seed=self.seed,
            measure=False,
            include_initial_state=False,
            optimize=self.optimize,
            time_limit=self.time_limit,
            canonical_permutation=self.canonical_permutation,
            workers=self.workers,
        )
        self._permutations = list(result.permutations)

        if self.initial_state_bitstring is not None and result.any_permuted:
            raise ValueError(
                "initial_state_bitstring cannot be combined with optimize=True: the "
                "relabeled circuits sample in a permuted mode order, so the explicit "
                "determinant would be applied to the wrong modes. Pass optimize=False "
                "to keep the original mode order, or leave initial_state_bitstring "
                "unset to use the Hartree-Fock reference (which is permuted to match)."
            )

        na, nb = ham.nelec

        full = [
            compose_full_circuit(
                resolve_initial_state(
                    num_qubits=core.num_qubits,
                    initial_state_bitstring=self.initial_state_bitstring,
                    n_alpha=na,
                    n_beta=nb,
                    n_orbitals=ham.norb,
                    core=core,
                ),
                core,
            )
            for core in result.circuits
        ]
        # Decompose the opaque fermionic gates so any statevector backend runs them.
        return list(transpile(full, AerSimulator(), optimization_level=0))
