# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Ansatz circuits: HF reference (ffsim) and SqDRIFT (qiskit-fermions)."""

from __future__ import annotations

import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator

from embasi_qiskit_integration.circuits.hf import build_hf_circuit
from embasi_qiskit_integration.circuits.sqdrift import (
    build_sqdrift_circuits,
    sqdrift_available,
)
from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.sampling.aer import AerSampler

requires_fermions = pytest.mark.skipif(
    not sqdrift_available(), reason="qiskit-fermions not installed (fermions extra)"
)


@pytest.fixture
def n2_ham(data_dir):
    return fcidump.read(data_dir / "n2_8o10e.fcidump")


def test_hf_circuit_qubit_count(n2_ham):
    qc = build_hf_circuit(n2_ham)
    assert qc.num_qubits == 2 * n2_ham.norb


def test_hf_circuit_bitstring(n2_ham):
    """The HF-only circuit sampled with Aer yields one deterministic bitstring."""
    qc = build_hf_circuit(n2_ham)
    tqc = transpile(qc, AerSimulator(), optimization_level=0)
    counts = AerSampler().sample(tqc, shots=2000, seed=1)

    assert len(counts) == 1
    bitstring = next(iter(counts))
    # blocked alpha|beta ordering: total occupation == n_alpha + n_beta
    assert bitstring.count("1") == sum(n2_ham.nelec)
    assert len(bitstring) == 2 * n2_ham.norb


@requires_fermions
def test_sqdrift_exact_single_circuit(n2_ham):
    circuits = build_sqdrift_circuits(n2_ham, method="exact")
    assert len(circuits) == 1
    assert circuits[0].num_qubits == 2 * n2_ham.norb


@requires_fermions
def test_sqdrift_qdrift_ensemble(n2_ham):
    circuits = build_sqdrift_circuits(
        n2_ham, method="qdrift", num_terms=10, num_randomizations=3, seed=42
    )
    assert len(circuits) == 3
    for qc in circuits:
        assert qc.num_qubits == 2 * n2_ham.norb


@requires_fermions
def test_sqdrift_exact_spans_ci_space(n2_ham):
    """The exact-evolution ansatz samples many determinants at the right
    Hamming weights (this is what feeds SQD)."""
    circuits = build_sqdrift_circuits(n2_ham, method="exact", time=1.0)
    tqc = transpile(circuits[0], AerSimulator(), optimization_level=0)
    counts = AerSampler().sample(tqc, shots=20_000, seed=1)
    norb = n2_ham.norb
    na, nb = n2_ham.nelec
    # many distinct configurations, each with the correct alpha/beta occupation
    assert len(counts) > 20
    good = sum(1 for b in counts if b[norb:].count("1") == na and b[:norb].count("1") == nb)
    assert good == len(counts)
    assert all(len(b) == 2 * norb for b in counts)
