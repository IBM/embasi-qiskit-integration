# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Circuit generation: integrals -> fermionic operator -> mapped circuit.

This package owns the whole path from an :class:`EmbeddedHamiltonian` to a
sampling-ready :class:`~qiskit.QuantumCircuit`, including the fermionic operator
construction and the qubit mapping. The Jordan-Wigner mapping is not a separate
stage: it is the pass manager the generator runs
(``generate_preset_jw_pass_manager``), so operator building and mapping live here
rather than being split across packages.

Modules:

- :mod:`.operator` -- integrals -> canonicalized, grouped ``FermionOperator``;
  the single operator builder for the package.
- :mod:`.sqdrift` -- the primary SqDRIFT ansatz (qiskit-fermions).
- :mod:`.hf` -- the Hartree-Fock reference circuit (ffsim), always available.
- :mod:`.utils` -- orderable sort keys backing the canonical ordering.
- :mod:`.lucj` -- on hold.
"""
