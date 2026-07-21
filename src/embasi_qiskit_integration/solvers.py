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

    Builds the SqDRIFT sampling ansatz for the active-space Hamiltonian,
    samples bitstrings with the injected ``sampler``, and runs the SQD driver to
    recover the ground-state energy and RDMs.

    The ``sampler`` follows the :class:`~embasi_qiskit_integration.sampling.base.
    BitstringSampler` protocol. A ``MockSampler`` replays frozen counts and
    ignores the circuit (deterministic CI); Aer/Runtime samplers run the
    transpiled circuit.
    """

    def __init__(
        self,
        sampler,
        *,
        shots: int = 100_000,
        ansatz: str = "sqdrift",
        evolution_time: float = 1.0,
        samples_per_batch: int = 300,
        num_batches: int = 5,
        max_iterations: int = 5,
        seed: int | None = None,
    ):
        self.sampler = sampler
        self.shots = shots
        self.ansatz = ansatz
        self.evolution_time = evolution_time
        self.samples_per_batch = samples_per_batch
        self.num_batches = num_batches
        self.max_iterations = max_iterations
        self.seed = seed

    def solve(self, ham: EmbeddedHamiltonian) -> SolverResult:
        from embasi_qiskit_integration.sqd.driver import run_sqd

        circuit = self._build_and_prepare_circuit(ham)
        counts = self.sampler.sample(circuit, self.shots, seed=self.seed)

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
            sampler=type(self.sampler).__name__,
            versions=collect_versions(),
            seed=self.seed,
            fcidump_sha=ham.meta.get("sha"),
        )
        return res

    def _build_and_prepare_circuit(self, ham: EmbeddedHamiltonian):
        """Build the ansatz circuit, transpiled for a real sampler.

        Returns ``None`` for a MockSampler (which ignores the circuit), avoiding
        the qiskit-fermions dependency in pure-CI test runs.
        """
        if type(self.sampler).__name__ == "MockSampler":
            return None

        if self.ansatz != "sqdrift":
            raise ValueError(
                f"unknown ansatz {self.ansatz!r}; only 'sqdrift' is available (LUCJ is on hold)"
            )

        from qiskit import transpile
        from qiskit_aer import AerSimulator

        from embasi_qiskit_integration.circuits.sqdrift import build_sqdrift_circuits

        circuits = build_sqdrift_circuits(
            ham, method="exact", time=self.evolution_time, seed=self.seed
        )
        # Decompose the opaque fermionic gates so any statevector backend runs it.
        return transpile(circuits[0], AerSimulator(), optimization_level=0)
