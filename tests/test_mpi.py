# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""MPI tests: exercise the rank-0 solve/broadcast path under a real mpirun.

These launch subprocesses with ``mpirun -n 2`` and are skipped unless both
``mpirun`` and ``mpi4py`` are available (they are not in the default CI image).
The single-process behaviour of ``rank0_solve`` is already covered elsewhere;
here we verify the multi-rank broadcast actually works end to end.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _mpi_available() -> bool:
    if shutil.which("mpirun") is None:
        return False
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False
    return True


requires_mpi = pytest.mark.skipif(not _mpi_available(), reason="needs mpirun + mpi4py")


def _run_mpi(code: str, nranks: int = 2) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["mpirun", "-n", str(nranks), sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO_ROOT,
    )


@requires_mpi
def test_rank0_solve_broadcasts_result():
    """Every rank returns the same FCI SolverResult via rank0_solve."""
    code = """
        import numpy as np
        from mpi4py import MPI
        from embasi_qiskit_integration.hamiltonian import fcidump
        from embasi_qiskit_integration.solvers import FCISolver
        from embasi_qiskit_integration.ipc import rank0_solve

        ham = fcidump.read("tests/data/n2_8o10e.fcidump")
        res = rank0_solve(FCISolver(), ham)
        rank = MPI.COMM_WORLD.Get_rank()
        # Each rank must hold a fully-populated result (not None).
        assert res is not None and res.rdm1.shape == (ham.norb, ham.norb)
        print(f"RANK {rank} ENERGY {res.energy:.10f}")
    """
    proc = _run_mpi(code)
    assert proc.returncode == 0, proc.stderr

    energies = [
        float(line.split()[-1]) for line in proc.stdout.splitlines() if line.startswith("RANK ")
    ]
    assert len(energies) == 2  # one line per rank
    # All ranks agree, and the value matches the stored FCI reference.
    assert abs(energies[0] - energies[1]) < 1e-12
    assert abs(energies[0] - (-108.9585095430)) < 1e-6


@requires_mpi
def test_embedding_workflow_mpi_orchestration():
    """The workflow's MPI orchestration is correct under a real mpirun -n 2.

    We deliberately do NOT drive the real ``scripts/embedding_workflow.py`` here:
    with ``parallel=True`` (which the workflow sets under MPI) EmbASI's
    ``roothan_hall_eigensolver_scalapack.hamiltonian_eigensolv_parallel`` does
    ``overlap[0,0].gl_m`` and crashes with ``AttributeError`` -- it expects a
    scalapack-distributed matrix but gets a plain ndarray.  That is an upstream
    EmbASI bug on the parallel SPADE path (reported separately), not a defect in
    this package's MPI code.

    ``tests/_mpi_workflow_mock.py`` replaces only that broken piece with a
    self-consistent mock ProjectionEmbedding (built from a real RHF, so every
    adapter validation passes honestly) and runs the SAME orchestration the
    workflow uses: rank-guarded logging, ``build_orbitals`` / downfold on every
    rank, and ``rank0_solve`` (rank 0 solves, result broadcast).  What this
    verifies: console output appears once and both ranks reach the same point
    with the same result.
    """
    driver = REPO_ROOT / "tests" / "_mpi_workflow_mock.py"
    proc = subprocess.run(
        ["mpirun", "-n", "2", sys.executable, str(driver)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr

    # Rank-0-only banner appears exactly once (rank guarding works).
    assert proc.stdout.count("running under MPI with 2 ranks") == 1
    assert proc.stdout.count("Step done: solve broadcast to all ranks") == 1

    # Both ranks reach the solve and hold the SAME broadcast energy.
    reached = [line for line in proc.stdout.splitlines() if "REACHED_SOLVE" in line]
    assert len(reached) == 2, proc.stdout
    energies = [float(line.split()[-1]) for line in reached]
    assert abs(energies[0] - energies[1]) < 1e-12
