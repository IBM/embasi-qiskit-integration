# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""End-to-end SQDSolver (MockSampler) vs FCISolver on identical integrals."""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.sampling.base import MockSampler
from embasi_qiskit_integration.solvers import FCISolver, SQDSolver


@pytest.fixture
def n2_ham(data_dir):
    return fcidump.read(data_dir / "n2_8o10e.fcidump")


@pytest.fixture
def mock_sampler(data_dir):
    return MockSampler(data_dir / "mock_counts.json")


def test_sqdsolver_matches_fci(n2_ham, mock_sampler):
    fci = FCISolver().solve(n2_ham)
    sqd = SQDSolver(mock_sampler, shots=100_000, seed=24).solve(n2_ham)
    assert abs(sqd.energy - fci.energy) <= 2e-3


def test_sqdsolver_diagnostics(n2_ham, mock_sampler):
    res = SQDSolver(mock_sampler, shots=100_000, seed=24).solve(n2_ham)
    d = res.diagnostics
    assert d["shots"] == 100_000
    assert d["sampler"] == "MockSampler"
    assert d["ansatz"] == "sqdrift"
    assert d["seed"] == 24
    assert "versions" in d and "qiskit" in d["versions"]
    assert d["fcidump_sha"] == n2_ham.meta.get("sha")


def test_sqdsolver_returns_rdms(n2_ham, mock_sampler):
    res = SQDSolver(mock_sampler, seed=24).solve(n2_ham)
    assert res.rdm1.shape == (n2_ham.norb, n2_ham.norb)
    assert abs(np.trace(res.rdm1) - sum(n2_ham.nelec)) < 1e-6
    assert res.rdm2 is not None
