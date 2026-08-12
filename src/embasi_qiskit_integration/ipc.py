# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Rank-0 guard + file-based job handoff between two processes.

File protocol in a job directory::

    <dir>/job.fcidump + job.npz   input written by process A
    <dir>/result.npz              energy, rdm1, rdm2, diagnostics (process B)
    <dir>/result.ERROR            traceback text on failure

Results are written atomically (temp file + rename) so a reader watching the
directory never observes a partial file.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np

from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult
from embasi_qiskit_integration.hamiltonian import fcidump

JOB_STEM = "job"
RESULT_NAME = "result.npz"
ERROR_NAME = "result.ERROR"


def rank0_solve(solver, ham: EmbeddedHamiltonian) -> SolverResult:
    """Run ``solver.solve(ham)`` on rank 0 and broadcast the result.

    With mpi4py absent this is just ``solver.solve(ham)``. With mpi4py present,
    only rank 0 solves (quantum sampling/hardware calls must be serialized) and
    the :class:`SolverResult` is broadcast to all ranks.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        return solver.solve(ham)

    comm = MPI.COMM_WORLD
    result = solver.solve(ham) if comm.Get_rank() == 0 else None
    return comm.bcast(result, root=0)


def read_job(job_dir: str | Path) -> EmbeddedHamiltonian:
    """Read the input Hamiltonian from ``<dir>/job.fcidump`` (+ ``job.npz``)."""
    job_dir = Path(job_dir)
    return fcidump.read(job_dir / f"{JOB_STEM}.fcidump")


def write_job(ham: EmbeddedHamiltonian, job_dir: str | Path) -> Path:
    """Write ``ham`` as the input job in ``job_dir``; returns the FCIDUMP path."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f"{JOB_STEM}.fcidump"
    fcidump.write(ham, path)
    return path


def write_result(result: SolverResult, job_dir: str | Path) -> Path:
    """Atomically write a :class:`SolverResult` to ``<dir>/result.npz``."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    final = job_dir / RESULT_NAME
    # Use a .npz temp suffix so np.savez does not append another extension.
    tmp = job_dir / "result.tmp.npz"

    payload = {
        "energy": np.asarray(result.energy, dtype=float),
        "rdm1": np.asarray(result.rdm1),
        "diagnostics_json": np.asarray(json.dumps(result.diagnostics, default=_json_default)),
    }
    if result.rdm2 is not None:
        payload["rdm2"] = np.asarray(result.rdm2)

    with tmp.open("wb") as fh:
        # numpy's savez stub mistypes the 2nd positional as bool; kwargs are fine.
        np.savez(fh, **payload)  # type: ignore[arg-type]
    os.replace(tmp, final)
    return final


def read_result(job_dir: str | Path) -> SolverResult:
    """Read a :class:`SolverResult` back from ``<dir>/result.npz``."""
    job_dir = Path(job_dir)
    with np.load(job_dir / RESULT_NAME, allow_pickle=False) as npz:
        energy = float(npz["energy"])
        rdm1 = np.asarray(npz["rdm1"])
        rdm2 = np.asarray(npz["rdm2"]) if "rdm2" in npz.files else None
        diagnostics = json.loads(str(npz["diagnostics_json"]))
    return SolverResult(energy=energy, rdm1=rdm1, rdm2=rdm2, diagnostics=diagnostics)


def write_error(exc: BaseException, job_dir: str | Path) -> Path:
    """Write the traceback of ``exc`` to ``<dir>/result.ERROR``."""
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / ERROR_NAME
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    path.write_text(text)
    return path


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return str(obj)
