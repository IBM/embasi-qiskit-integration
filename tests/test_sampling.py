# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Sampling layer: MockSampler determinism and Aer correctness."""

from __future__ import annotations

import pytest

from embasi_qiskit_integration.sampling.base import BitstringSampler, MockSampler


def test_mock_sampler_replays_counts(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    counts = sampler.sample(circuit=None, shots=2000)
    assert counts == {"00": 983, "11": 1017}


def test_mock_sampler_deterministic(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    a = sampler.sample(circuit=None, shots=2000, seed=1)
    b = sampler.sample(circuit=None, shots=2000, seed=999)
    assert a == b  # seed and circuit are ignored — pure replay


def test_mock_sampler_rescales_preserving_total(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    counts = sampler.sample(circuit=None, shots=1000)
    assert sum(counts.values()) == 1000
    # proportions preserved to rounding
    assert abs(counts["11"] / 1000 - 1017 / 2000) < 1e-3


def test_mock_sampler_satisfies_protocol(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    assert isinstance(sampler, BitstringSampler)


@pytest.mark.slow
def test_aer_bell_circuit_only_00_and_11():
    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.sampling.aer import AerSampler

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    counts = AerSampler().sample(qc, shots=4000, seed=42)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 4000
    # both outcomes appear with roughly equal weight
    assert counts.get("00", 0) > 1000
    assert counts.get("11", 0) > 1000


@pytest.mark.slow
def test_aer_sampler_deterministic_with_seed():
    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.sampling.aer import AerSampler

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    a = AerSampler().sample(qc, shots=2000, seed=7)
    b = AerSampler().sample(qc, shots=2000, seed=7)
    assert a == b
