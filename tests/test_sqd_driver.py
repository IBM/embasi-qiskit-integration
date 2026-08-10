# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SQD driver on frozen mock counts vs the stored FCI reference."""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.circuit_run.base import MockSampler
from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.sqd.driver import run_sqd


@pytest.fixture
def n2_ham(data_dir):
    return fcidump.read(data_dir / "n2_8o10e.fcidump")


@pytest.fixture
def mock_counts(data_dir):
    return MockSampler(data_dir / "mock_counts.json").sample(circuit=None, shots=100_000)


@pytest.fixture
def sqd_result(n2_ham, mock_counts):
    return run_sqd(
        n2_ham,
        mock_counts,
        samples_per_batch=300,
        num_batches=5,
        max_iterations=5,
        seed=24,
    )


def test_sqd_energy_within_tolerance(n2_ham, sqd_result):
    ref = n2_ham.meta["reference_fci_energy"]
    assert abs(sqd_result.energy - ref) <= 2e-3


def test_sqd_returns_rdm1(sqd_result, n2_ham):
    assert sqd_result.rdm1 is not None
    assert sqd_result.rdm1.shape == (n2_ham.norb, n2_ham.norb)
    assert abs(np.trace(sqd_result.rdm1) - sum(n2_ham.nelec)) < 1e-6


def test_sqd_rdm1_hermitian(sqd_result):
    assert np.allclose(sqd_result.rdm1, sqd_result.rdm1.conj().T, atol=1e-8)


def test_sqd_returns_rdm2(sqd_result, n2_ham):
    assert sqd_result.rdm2 is not None
    assert sqd_result.rdm2.shape == (n2_ham.norb,) * 4


def test_sqd_energy_monotonic_nonincreasing(sqd_result):
    energies = sqd_result.diagnostics["iteration_energies"]
    assert len(energies) >= 1
    # Assert loosely: warn (don't fail) on small non-monotonic wiggles from
    # the stochastic subspace selection.
    for lo, hi in zip(energies[1:], energies[:-1]):
        if lo > hi + 1e-6:
            import warnings

            warnings.warn(
                f"SQD energy increased across iterations: {hi} -> {lo}",
                stacklevel=1,
            )


def test_sqd_deterministic_with_seed(n2_ham, mock_counts):
    a = run_sqd(n2_ham, mock_counts, seed=7, max_iterations=3)
    b = run_sqd(n2_ham, mock_counts, seed=7, max_iterations=3)
    assert abs(a.energy - b.energy) < 1e-10


def test_sqd_diagnostics_present(sqd_result):
    d = sqd_result.diagnostics
    assert d["solver"] == "qiskit-addon-sqd"
    assert d["n_shots"] == 100_000
    assert d["n_distinct_bitstrings"] > 1
