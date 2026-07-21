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
def test_embedding_workflow_runs_under_mpirun():
    """The embedding sketch runs clean under mpirun and prints once (rank 0)."""
    proc = subprocess.run(
        [
            "mpirun",
            "-n",
            "2",
            sys.executable,
            "scripts/embedding_workflow.py",
            "--solver",
            "fci",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    # Output is rank-0-only: the final energy line appears exactly once.
    assert proc.stdout.count("running under MPI with 2 ranks") == 1
    assert proc.stdout.count("Step 5") == 1
    assert "-108.9585" in proc.stdout
