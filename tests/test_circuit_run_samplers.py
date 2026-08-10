# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Samplers: MockSampler determinism, Aer correctness, and the batch seam."""

from __future__ import annotations

import pytest

from embasi_qiskit_integration.circuit_run.base import BitstringSampler, MockSampler


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


# ----- the batch seam: run() and its single-circuit shim ------------------- #


def test_mock_sampler_run_returns_one_dict_per_circuit(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    out = sampler.run([None, None, None], shots=2000)
    assert len(out) == 3
    assert all(counts == {"00": 983, "11": 1017} for counts in out)


def test_mock_sampler_run_empty_batch(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    assert sampler.run([], shots=2000) == []


def test_sample_is_the_first_element_of_run(data_dir):
    """The single-circuit shim must agree with the batch method by construction."""
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    assert sampler.sample(circuit=None, shots=1500) == sampler.run([None], shots=1500)[0]


def test_mock_sampler_declares_it_needs_no_circuit(data_dir):
    """Lets callers skip circuit construction entirely for deterministic replay."""
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    assert sampler.requires_circuit is False


def test_merge_counts_sums_per_bitstring():
    from embasi_qiskit_integration.circuit_run import merge_counts

    merged = merge_counts([{"00": 3, "11": 2}, {"00": 1, "01": 5}, {}])
    assert merged == {"00": 4, "11": 2, "01": 5}
    assert sum(merged.values()) == 11


def test_merge_counts_of_nothing_is_empty():
    from embasi_qiskit_integration.circuit_run import merge_counts

    assert merge_counts([]) == {}


@pytest.mark.slow
def test_aer_run_samples_every_circuit_in_the_batch():
    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.circuit_run.aer import AerSampler

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    zero = QuantumCircuit(2)
    zero.measure_all()

    out = AerSampler().run([bell, zero], shots=1000, seed=5)
    assert len(out) == 2
    assert set(out[0]) <= {"00", "11"}
    assert out[1] == {"00": 1000}  # the all-zero circuit is deterministic
    assert all(sum(counts.values()) == 1000 for counts in out)


@pytest.mark.slow
def test_aer_bell_circuit_only_00_and_11():
    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.circuit_run.aer import AerSampler

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

    from embasi_qiskit_integration.circuit_run.aer import AerSampler

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    a = AerSampler().sample(qc, shots=2000, seed=7)
    b = AerSampler().sample(qc, shots=2000, seed=7)
    assert a == b
