# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Sketch: an EmbASI embedding calculation solved with this package.

This is a runnable *sketch* of the intended end-to-end flow:

    EmbASI projection embedding (low level)
        -> extract an active-space EmbeddedHamiltonian   [hamiltonian/extract.py]
        -> solve it with a high-level solver (FCI or SQD) [solvers.py]
        -> assemble the projection-based-embedding energy [workflow.py]
        -> feed the correlated 1-RDM back into the embedding density

EmbASI + FHI-aims are **mocked** so this runs with no QM driver, no network, and
no hardware. The mock (:class:`MockEmbASI`) exposes the same attributes the real
EmbASI glue reads (see ``hamiltonian/extract.py`` and its ``TODO(embasi-api)``
markers) but backs them with a real PySCF active-space calculation, so the
extracted Hamiltonian is a genuine correlated problem (non-trivial ``h2``) and
the numbers are meaningful.

Where the real workflow differs from this sketch is called out inline with
``# REAL:`` comments. Configured via pydantic-settings::

    uv run python scripts/embedding_workflow.py
    uv run python scripts/embedding_workflow.py --solver sqd --seed 24
    uv run python scripts/embedding_workflow.py --handoff two-process

**MPI.** Real EmbASI runs under MPI, so this sketch is MPI-safe: the solve goes
through ``ipc.rank0_solve`` (rank 0 solves, result is broadcast to all ranks),
and console output / job-dir writes are guarded to rank 0. Run it parallel with::

    mpirun -n 2 uv run python scripts/embedding_workflow.py --solver sqd
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult
from embasi_qiskit_integration.workflow import embedding_energy, feedback_rdm1

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


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
# Mock EmbASI object
# --------------------------------------------------------------------------- #
class MockEmbASI:
    """Stand-in for an EmbASI projection-embedding object.

    Mimics the attribute surface the real glue reads (``mo_coeff``,
    ``hamiltonian_kinetic``, ``fock_embedding_matrix``, ``density_matrix_in``,
    energies) but is backed by a real PySCF RHF + CASCI active space so the
    extracted Hamiltonian is physical. In a real run this object would be an
    ``embasi.embedding.ProjectionEmbedding`` that has completed its low-level SCF.
    """

    def __init__(self, *, atom: str, basis: str, ncas: int, nelecas: int):
        from pyscf import ao2mo, mcscf, scf
        from pyscf import gto as pyscf_gto

        mol = pyscf_gto.M(atom=atom, basis=basis, verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        mc = mcscf.CASCI(mf, ncas, nelecas)
        h1_cas, e_core = mc.get_h1eff()
        h2_cas = ao2mo.restore(1, mc.get_h2eff(), ncas)

        # --- attributes the EmbASI glue reads (see hamiltonian/extract.py) ---
        # REAL: mo_coeff would be the SPADE-localized coefficients for subsystem A
        # (ProjectionEmbedding.spade_localisation); here the CAS integrals are
        # already in the MO basis, so we expose identity + carry the CAS tensors.
        self.mo_coeff = np.eye(ncas)
        self.hamiltonian_kinetic = np.asarray(h1_cas)  # REAL: kinetic in AO
        self.fock_embedding_matrix = np.zeros((ncas, ncas))  # REAL: h_core + v_emb + P_B
        self.e_core = float(e_core)
        self.n_active_alpha = nelecas // 2
        self.n_active_beta = nelecas - nelecas // 2

        # Kept for building the genuine active-space Hamiltonian below and for
        # the PbE energy bookkeeping.
        self._h2_cas = np.asarray(h2_cas)
        self.e_low_total = float(mf.e_tot)  # E_low over the full system (RHF)
        self.e_low_A = float(mf.e_tot)  # REAL: E_low restricted to subsystem A
        self.density_matrix_in = None  # settable; RDM feedback target

    def build_embedded_hamiltonian(self, active_orbitals) -> EmbeddedHamiltonian:
        """Assemble a genuine active-space :class:`EmbeddedHamiltonian`.

        REAL: replace this whole method with

            from embasi_qiskit_integration.hamiltonian.extract import (
                embedded_hamiltonian_from_embasi,
            )
            ham = embedded_hamiltonian_from_embasi(emb, active_orbitals)

        once ``extract`` is wired to a live EmbASI object. ``extract`` currently
        returns a placeholder (zero) two-body tensor because ASI does not export
        the active-space ERIs directly; here we carry the real CAS ``h2`` so the
        solve is meaningful.
        """
        idx = list(active_orbitals)
        h1 = np.asarray(self.hamiltonian_kinetic)[np.ix_(idx, idx)]
        h2 = self._h2_cas[np.ix_(idx, idx, idx, idx)]
        return EmbeddedHamiltonian(
            h1=h1,
            h2=h2,
            e_core=self.e_core,
            nelec=(self.n_active_alpha, self.n_active_beta),
            meta={"source": "mock-embasi", "n_active_orbitals": len(idx)},
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class EmbeddingWorkflow(BaseSettings):
    """Run a mocked EmbASI embedding calculation through this package."""

    model_config = SettingsConfigDict(env_prefix="EQI_EMB_", cli_parse_args=True)

    # Mock embedding "system" (a real run gets these from EmbASI/FHI-aims input).
    atom: str = "N 0 0 0; N 0 0 1.09"
    basis: str = "6-31g"
    ncas: int = 8
    nelecas: int = 10

    solver: Literal["sqd", "fci"] = "sqd"
    handoff: Literal["in-process", "two-process"] = "in-process"
    sampler: Literal["aer", "mock", "runtime"] = "aer"
    backend: str | None = None  # runtime backend name; else least-busy
    optimization_level: int = 3  # runtime ISA-transpile level
    shots: int = 100_000
    seed: int = 24
    job_dir: Path | None = None

    def cli_cmd(self) -> None:
        rank, size = _rank_size()

        def log(msg: str = "") -> None:
            """Print only on rank 0 (keeps multi-rank output clean)."""
            if rank == 0:
                print(msg)

        if size > 1:
            log(f"[running under MPI with {size} ranks; solve is on rank 0]")

        log("== Step 1: EmbASI low-level embedding (mocked) ==")
        # REAL: EmbASI's low-level SCF runs collectively on all ranks; here each
        # rank builds the same mock so the broadcast result is consistent.
        emb = MockEmbASI(atom=self.atom, basis=self.basis, ncas=self.ncas, nelecas=self.nelecas)
        active_orbitals = list(range(self.ncas))
        log(
            f"   system: {self.atom!r} / {self.basis}, "
            f"active space CAS({self.ncas}o, {self.nelecas}e)"
        )
        log(f"   E_low(total) = {emb.e_low_total:.6f} Ha  (RHF, mocked)")

        log("== Step 2: extract the embedded active-space Hamiltonian ==")
        # REAL: embedded_hamiltonian_from_embasi(emb, active_orbitals)
        ham = emb.build_embedded_hamiltonian(active_orbitals)
        log(f"   norb={ham.norb}  nelec={ham.nelec}  e_core={ham.e_core:.6f} Ha")

        log(f"== Step 3: solve with the high-level {self.solver.upper()} solver ==")
        # The solve goes through rank0_solve: rank 0 runs the (possibly hardware)
        # solver, then broadcasts the SolverResult to every rank.
        solver = self._build_solver()
        result = self._solve(emb, active_orbitals, ham, solver, rank=rank, log=log)
        log(
            f"   E_high(A) = {result.energy:.6f} Ha  "
            f"(solver={result.diagnostics.get('solver', self.solver)})"
        )
        log(f"   trace(rdm1) = {np.trace(result.rdm1):.4f}  (expected {sum(ham.nelec)})")

        log("== Step 4: assemble the projection-based-embedding energy ==")
        energy = embedding_energy(result, e_low_total=emb.e_low_total, e_low_A=emb.e_low_A)
        log("   E = E_low(total) - E_low(A) + E_high(A) + corr")
        log(
            f"     = {energy.e_low_total:.6f} - {energy.e_low_A:.6f} "
            f"+ {energy.e_high_A:.6f} + {energy.correction:.6f}"
        )
        log(f"     = {energy.total:.6f} Ha")

        log("== Step 5: feed the correlated 1-RDM back into the embedding ==")
        # Every rank holds the broadcast result, so the density feedback is
        # consistent across the communicator.
        feedback_rdm1(emb, result.rdm1)
        log(
            f"   emb.density_matrix_in updated (shape {emb.density_matrix_in.shape}); "
            "a real driver would start the next embedding cycle here."
        )

    def _build_solver(self):
        from embasi_qiskit_integration.solvers import FCISolver, SQDSolver

        if self.solver == "fci":
            return FCISolver()

        if self.sampler == "mock":
            from embasi_qiskit_integration.sampling.base import MockSampler

            sampler = MockSampler(DATA_DIR / "mock_counts.json")
        elif self.sampler == "runtime":
            # Plug in real quantum hardware: uses configured IBM Quantum
            # credentials, least-busy backend unless --backend names one.
            from embasi_qiskit_integration.sampling.runtime import RuntimeSampler

            sampler = RuntimeSampler(
                backend=self.backend,
                optimization_level=self.optimization_level,
                default_shots=self.shots,
            )
        else:
            from embasi_qiskit_integration.sampling.aer import AerSampler

            sampler = AerSampler()
        return SQDSolver(sampler, shots=self.shots, seed=self.seed)

    def _solve(self, emb, active_orbitals, ham, solver, *, rank, log) -> SolverResult:
        """Solve in-process, or via the two-process file handoff.

        Both paths are MPI-safe: rank 0 does the solving/IO and the resulting
        :class:`SolverResult` is broadcast to every rank.
        """
        if self.handoff == "in-process":
            # REAL: workflow.solve_embedding_in_process(emb, active_orbitals, solver)
            # would call extract internally; we already built `ham`, so solve it
            # directly through the rank-0 guard (rank 0 solves, result broadcast).
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
            log(f"   wrote job to {job_dir} (process B would run the CLI 'solve --watch')")
            solved = solver.solve(ipc.read_job(job_dir))
            ipc.write_result(solved, job_dir)
            result = ipc.read_result(job_dir)

        return _broadcast(result)


def main() -> None:
    CliApp.run(EmbeddingWorkflow)


if __name__ == "__main__":
    main()
