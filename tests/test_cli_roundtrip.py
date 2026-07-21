# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Two-process file handoff: CLI writes result.npz / result.ERROR."""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration import ipc
from embasi_qiskit_integration.cli import main
from embasi_qiskit_integration.hamiltonian import fcidump


@pytest.fixture
def n2_ham(data_dir):
    return fcidump.read(data_dir / "n2_8o10e.fcidump")


def test_cli_fci_roundtrip(tmp_path, n2_ham):
    ipc.write_job(n2_ham, tmp_path)
    rc = main(["solve", str(tmp_path), "--solver", "fci"])
    assert rc == 0

    result = ipc.read_result(tmp_path)
    assert abs(result.energy - n2_ham.meta["reference_fci_energy"]) < 1e-8
    assert result.rdm1.shape == (n2_ham.norb, n2_ham.norb)
    assert result.diagnostics["solver"] == "pyscf-fci"


def test_cli_sqd_mock_roundtrip(tmp_path, n2_ham, data_dir):
    ipc.write_job(n2_ham, tmp_path)
    counts = str(data_dir / "mock_counts.json")
    rc = main(
        [
            "solve",
            str(tmp_path),
            "--solver",
            "sqd",
            "--sampler",
            "mock",
            "--counts",
            counts,
            "--seed",
            "24",
        ]
    )
    assert rc == 0

    result = ipc.read_result(tmp_path)
    assert abs(result.energy - n2_ham.meta["reference_fci_energy"]) <= 2e-3
    assert result.rdm2 is not None
    assert result.diagnostics["sampler"] == "MockSampler"


def test_cli_error_on_corrupt_input(tmp_path):
    # No job.fcidump present -> a result.ERROR file with a traceback.
    (tmp_path / "job.fcidump").write_text("this is not a valid fcidump\n")
    rc = main(["solve", str(tmp_path), "--solver", "fci"])
    assert rc == 1
    assert (tmp_path / "result.ERROR").exists()
    assert not (tmp_path / "result.npz").exists()


def test_result_written_atomically(tmp_path, n2_ham):
    result = ipc.read_result  # sanity: function exists
    assert callable(result)
    from embasi_qiskit_integration.contract import SolverResult

    res = SolverResult(energy=-1.0, rdm1=np.eye(2), diagnostics={"x": 1})
    path = ipc.write_result(res, tmp_path)
    assert path.name == "result.npz"
    assert not (tmp_path / "result.tmp.npz").exists()  # temp cleaned up
    back = ipc.read_result(tmp_path)
    assert back.energy == -1.0
    assert back.diagnostics["x"] == 1
