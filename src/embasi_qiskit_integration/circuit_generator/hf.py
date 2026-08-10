# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Hartree-Fock reference circuit (ffsim, Jordan-Wigner).

This is the always-available baseline ansatz: it prepares the closed/open-shell
HF determinant on ``2*norb`` qubits with ffsim's blocked (alpha|beta) spin
ordering. It is used both for the deterministic HF-bitstring test and as a
cheap sampling ansatz that yields the reference determinant. Correlated
ansaetze (SqDRIFT, LUCJ) build on top of this reference state.
"""

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
