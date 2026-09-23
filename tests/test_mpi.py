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


def _find_launcher() -> str | None:
    """The MPI launcher on PATH.

    Debian/Ubuntu's ``mpich`` ships ``mpirun.mpich``/``mpiexec.mpich`` and provides the
    bare ``mpirun`` only through update-alternatives, which is not always configured, so
    take the first name that exists rather than assuming ``mpirun``.
    """
    for name in ("mpirun", "mpiexec", "mpirun.mpich", "mpiexec.mpich"):
        if shutil.which(name):
            return name
    return None


def _multirank_launch_works() -> bool:
    """Can this environment actually run a 2-rank job that shares one COMM_WORLD?

    Probed by launching one, rather than inferred from version strings.  A launcher and
    an ``mpi4py`` from different MPI installs -- the PyPI wheels bundle their own runtime
    -- initialise a SINGLETON communicator per process instead of failing: the job
    degrades to N independent 1-rank runs, every process reporting rank 0 of size 1.

    That is an environment fault, not a defect in the code under test, and no source
    change fixes it, so the tests skip rather than fail.  The probe is the same question
    they ask, which is why it cannot drift away from them the way a vendor comparison did.
    """
    launcher = _find_launcher()
    if launcher is None:
        return False
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        return False
    probe = "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"
    try:
        proc = subprocess.run(
            [launcher, "-n", "2", sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - probed once
        return False
    # Two ranks in one communicator print "2" twice; a singleton launch prints "1" twice.
    return proc.returncode == 0 and [ln.strip() for ln in proc.stdout.split()] == ["2", "2"]


requires_mpi = pytest.mark.skipif(
    not _multirank_launch_works(),
    reason="needs an MPI that launches 2 ranks in one COMM_WORLD (launcher and mpi4py "
    "from the same install)",
)


def _mpirun_argv(nranks: int) -> list[str]:
    """``mpirun`` plus the flags a containerised OpenMPI needs to launch at all.

    A CI runner often has fewer cores than ranks, and OpenMPI refuses to
    oversubscribe by default; it also probes transports that are unavailable in a
    container, and aborts as root without an explicit opt-in.  Each of those kills
    the launcher *before* the payload runs, so the failure arrives with empty
    stdout AND stderr.  The flags are OpenMPI-specific, so they are only added when
    ``mpirun --version`` says OpenMPI -- MPICH's Hydra rejects unknown flags.
    """
    argv = [_find_launcher() or "mpirun", "-n", str(nranks)]
    try:
        banner = subprocess.run(
            [argv[0], "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - probed once
        return argv
    if "Open MPI" in (banner.stdout + banner.stderr):
        argv += ["--oversubscribe", "--allow-run-as-root"]
    return argv


def _run_mpi(code: str, nranks: int = 2) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*_mpirun_argv(nranks), sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO_ROOT,
    )


def _mpi_failure(proc: subprocess.CompletedProcess) -> str:
    """Both streams, since a launcher-level failure leaves stderr empty."""
    return f"returncode={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"


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
    assert proc.returncode == 0, _mpi_failure(proc)

    ranks = {line.split()[1] for line in proc.stdout.splitlines() if line.startswith("RANK ")}
    assert ranks == {"0", "1"}, (
        f"expected ranks 0 and 1, saw {sorted(ranks)} -- a singleton launch reports every "
        f"process as rank 0.\n{_mpi_failure(proc)}"
    )
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
        [*_mpirun_argv(2), sys.executable, str(driver)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, _mpi_failure(proc)

    # A launcher that cannot form a multi-rank communicator runs N independent 1-rank
    # jobs -- every rank reports itself as rank 0 of size 1.  Name that outright: it is an
    # environment fault, and diagnosing it from a bare count mismatch is guesswork.
    assert "running under MPI with 2 ranks" in proc.stdout, (
        "ranks did not share one COMM_WORLD (mpi4py's bundled MPI runtime not matching "
        f"the launcher will do this).\n{_mpi_failure(proc)}"
    )

    # Rank-0-only lines appear exactly once, i.e. rank guarding works.
    assert proc.stdout.count("running under MPI with 2 ranks") == 1
    assert proc.stdout.count("Step done: solve broadcast to all ranks") == 1

    # Both ranks reach the solve and hold the SAME broadcast energy.
    reached = [line for line in proc.stdout.splitlines() if "REACHED_SOLVE" in line]
    assert len(reached) == 2, proc.stdout
    energies = [float(line.split()[-1]) for line in reached]
    assert abs(energies[0] - energies[1]) < 1e-12
