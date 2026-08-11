# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Hartree-Fock reference circuit (ffsim, Jordan-Wigner)."""

from __future__ import annotations

from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def build_hf_circuit(ham: EmbeddedHamiltonian, *, measure: bool = True):
    """Build the Hartree-Fock reference circuit for ``ham``.

    Returns a :class:`qiskit.QuantumCircuit` on ``2 * ham.norb`` qubits. With
    ``measure=True`` the circuit ends in ``measure_all()``.
    """
    import ffsim
    from qiskit import QuantumCircuit, QuantumRegister

    norb = ham.norb
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, ham.nelec), qubits)
    if measure:
        circuit.measure_all()
    return circuit
