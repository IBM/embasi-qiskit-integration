# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""End-to-end SQDSolver (MockSampler) vs FCISolver on identical integrals."""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.circuit_run.base import MockSampler
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


def test_mock_sampler_needs_no_circuit(n2_ham, mock_sampler):
    """Replay runs build no circuit at all, so they need no fermions extra."""
    solver = SQDSolver(mock_sampler, shots=100_000, seed=24)
    assert solver.build_circuits(n2_ham) == [None]


def test_ensemble_counts_are_pooled(n2_ham, data_dir):
    """N circuits are each sampled and their counts summed, not discarded.

    Uses MockSampler (one counts dict per requested circuit), so pooling N
    replays of a fixed distribution must multiply the shot total by N.
    """
    sampler = MockSampler(data_dir / "mock_counts.json")
    counts_list = sampler.run([None] * 3, shots=1000)
    assert len(counts_list) == 3

    from embasi_qiskit_integration.circuit_run import merge_counts

    pooled = merge_counts(counts_list)
    assert sum(pooled.values()) == 3000
    # Pooling does not invent or drop configurations.
    assert set(pooled) == set(counts_list[0])
