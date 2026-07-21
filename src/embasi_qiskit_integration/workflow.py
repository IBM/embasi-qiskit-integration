# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""EmbASI embedding workflow glue (guarded).

Wires an EmbASI projection-embedding driver to a quantum/classical active-space
solver: extract the embedded Hamiltonian, solve it (in-process on rank 0, or via
the two-process file handoff for hardware), and fold the high-level RDM back
into the projection-based-embedding (PbE) energy expression::

    E = E_low(total) - E_low(A) + E_high(A) + mu/DM corrections

Only the EmbASI *access* is guarded; the energy-assembly logic is pure and is
covered in CI against a :class:`FakeEmbASI` stub. EmbASI itself is imported
nowhere in this module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel

from embasi_qiskit_integration.contract import SolverResult


class EmbeddingEnergy(BaseModel):
    """Decomposed PbE energy and its high-level contribution."""

    total: float
    e_high_A: float
    e_low_total: float
    e_low_A: float
    correction: float


def embedding_energy(
    result: SolverResult,
    *,
    e_low_total: float,
    e_low_A: float,
    correction: float = 0.0,
) -> EmbeddingEnergy:
    """Assemble the PbE total energy from the high-level active-space result.

    ``E = E_low(total) - E_low(A) + E_high(A) + correction``. ``E_high(A)`` is
    ``result.energy`` (the active-space total energy including its ``e_core``).
    """
    e_high_A = float(result.energy)
    total = e_low_total - e_low_A + e_high_A + correction
    return EmbeddingEnergy(
        total=total,
        e_high_A=e_high_A,
        e_low_total=e_low_total,
        e_low_A=e_low_A,
        correction=correction,
    )


def solve_embedding_in_process(emb, active_orbitals, solver) -> SolverResult:
    """Extract the embedded Hamiltonian from ``emb`` and solve it on rank 0."""
    from embasi_qiskit_integration.hamiltonian.extract import (
        embedded_hamiltonian_from_embasi,
    )
    from embasi_qiskit_integration.ipc import rank0_solve

    ham = embedded_hamiltonian_from_embasi(emb, active_orbitals)
    return rank0_solve(solver, ham)


def solve_embedding_two_process(
    emb, active_orbitals, job_dir: str | Path, *, timeout: float = 3600.0, poll: float = 0.5
) -> SolverResult:
    """Two-process handoff: write the job, block until ``result.npz`` appears.

    Process A (this call) writes ``<job_dir>/job.fcidump`` from the extracted
    Hamiltonian; process B (the CLI ``solve`` with ``--watch``) runs the solver
    and writes ``result.npz`` (or ``result.ERROR``). Default for hardware runs.
    """
    import time

    from embasi_qiskit_integration import ipc
    from embasi_qiskit_integration.hamiltonian.extract import (
        embedded_hamiltonian_from_embasi,
    )

    job_dir = Path(job_dir)
    ham = embedded_hamiltonian_from_embasi(emb, active_orbitals)
    ipc.write_job(ham, job_dir)

    result_path = job_dir / ipc.RESULT_NAME
    error_path = job_dir / ipc.ERROR_NAME
    deadline = time.monotonic() + timeout
    while True:
        if result_path.exists():
            return ipc.read_result(job_dir)
        if error_path.exists():
            raise RuntimeError(f"solver process reported an error:\n{error_path.read_text()}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out waiting for {result_path}")
        time.sleep(poll)


def feedback_rdm1(emb, rdm1: np.ndarray) -> None:
    """Feed the high-level 1-RDM back into the embedding density.

    Kept isolated so the (EmbASI-specific) density update is a single
    touchpoint. In-place update of the subsystem-A density on ``emb``.
    """
    # TODO(embasi-api): write rdm1 back to the EmbASI density for subsystem A
    # (e.g. emb.density_matrix_in setter) so the next embedding cycle uses the
    # correlated density. Verify the setter name and basis (localized vs AO).
    emb.density_matrix_in = np.asarray(rdm1)
