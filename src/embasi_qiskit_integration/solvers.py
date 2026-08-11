# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Active-space solvers over an :class:`EmbeddedHamiltonian`.

Phase 2 provides the classical reference (:class:`FCISolver`) and the
``cheap_ccsd_t2`` helper used to seed the quantum ansatz. :class:`SQDSolver`
is completed in Phase 6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

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


class FCISolver(ActiveSpaceSolver):
    """Exact full configuration interaction via PySCF — the numerical oracle."""

    def solve(self, ham: EmbeddedHamiltonian) -> SolverResult:
        from pyscf import fci

        norb = ham.norb
        e, ci = fci.direct_spin1.kernel(ham.h1, ham.h2, norb, ham.nelec)
        rdm1, rdm2 = fci.direct_spin1.make_rdm12(ci, norb, ham.nelec)
        return SolverResult(
            energy=e + ham.e_core,
            rdm1=np.asarray(rdm1),
            rdm2=np.asarray(rdm2),
            diagnostics={"solver": "pyscf-fci"},
        )


def _rhf_from_integrals(ham: EmbeddedHamiltonian):
    """Build a PySCF RHF object over the bare (h1, h2, e_core) integrals.

    Follows the custom-Hamiltonian route: a bare ``Mole`` with the electron
    count set, overridden ``get_hcore``/``get_ovlp``/``energy_nuc`` and the
    two-electron integrals injected via ``_eri`` (8-fold packed).
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

    Used to seed the LUCJ ansatz (Phase 4). The active space is treated as a
    closed/open shell as implied by ``nelec``.
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
        shots: int = 100_000,
        ansatz: str = "sqdrift",
        method: str = "exact",
        evolution_time: float = 1.0,
        num_groups: int = 200,
        num_randomizations: int = 1,
        initial_state_bitstring: str | None = None,
        samples_per_batch: int = 300,
        num_batches: int = 5,
        max_iterations: int = 5,
        seed: int | None = None,
        optimize: bool | None = None,
        time_limit: float = 10.0,
        workers: int = 1,
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
        self.workers = workers
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
            seed=self.seed,
        )
        res.diagnostics.update(
            shots=self.shots,
            ansatz=self.ansatz,
            method=self.method,
            n_circuits=len(circuits),
            sampler=type(self.sampler).__name__,
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
