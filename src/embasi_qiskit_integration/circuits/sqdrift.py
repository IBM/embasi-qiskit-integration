# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SqDRIFT ansatz circuits via ``qiskit-fermions`` (primary ansatz).

SqDRIFT (arXiv:2508.02578) samples bitstrings from a Hamiltonian
time-evolution circuit. Two synthesis modes are provided:

- ``method="exact"`` (default): the full Jordan-Wigner-synthesised evolution of
  the (diagonal-filtered) Hamiltonian. Deterministic and dense enough to span
  the CI space well — used to generate the frozen SQD counts and as the default
  noiseless sampling ansatz.
- ``method="qdrift"``: the hardware-oriented ensemble of ``QDriftTrotterization``
  randomizations. Shorter circuits, at the cost of per-circuit stochasticity.

Both start from the Hartree-Fock reference via ``InitializeModes`` (occupying
the lowest ``n_alpha`` alpha and ``n_beta`` beta modes); without it the circuit
would evolve the vacuum and sample zero-particle bitstrings.

``qiskit-fermions`` is not on PyPI and requires a Rust toolchain (see the README
prerequisites and the ``fermions`` extra). All imports here are lazy so the rest
of the package works without it; :func:`sqdrift_available` reports whether it is
importable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def sqdrift_available() -> bool:
    """True if ``qiskit-fermions`` is importable in this environment."""
    try:
        import qiskit_fermions  # noqa: F401
    except ImportError:
        return False
    return True


def _require_fermions():
    try:
        import qiskit_fermions  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "SqDRIFT circuits require qiskit-fermions, which is not on PyPI and "
            "needs a Rust toolchain. Install it from source (see README) or via "
            "the 'fermions' extra: pip install -e '.[fermions]'."
        ) from exc


def _hf_occupation(norb: int, nelec: tuple[int, int]) -> list[bool]:
    """HF occupation over the ``2*norb`` modes (alpha block then beta block)."""
    na, nb = nelec
    occ = [False] * (2 * norb)
    for i in range(na):
        occ[i] = True
    for i in range(nb):
        occ[norb + i] = True
    return occ


def build_sqdrift_circuits(
    ham: EmbeddedHamiltonian,
    *,
    method: str = "exact",
    num_terms: int = 200,
    num_randomizations: int = 10,
    time: float = 1.0,
    filter_diagonal_terms: bool = True,
    seed: int | None = None,
    measure: bool = True,
) -> list:
    """Build SqDRIFT sampling circuits for ``ham``.

    Each circuit is a :class:`qiskit.QuantumCircuit` on ``2 * ham.norb`` qubits
    that prepares the HF reference and applies the Hamiltonian time-evolution.

    Args:
        method: ``"exact"`` (single full-evolution circuit) or ``"qdrift"``
            (ensemble of qDRIFT randomizations).
        num_terms: qDRIFT term-groups per circuit (``method="qdrift"`` only).
        num_randomizations: number of circuits to return (``method="qdrift"``;
            ``"exact"`` always returns one circuit).
        time: evolution time fed to the ``Evolution`` gate.
        filter_diagonal_terms: drop occupation-diagonal terms that do not affect
            sampled bitstrings.
        seed: RNG seed for the qDRIFT randomization (reproducible circuits).
        measure: append ``measure_all()`` to each circuit.
    """
    _require_fermions()
    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import Evolution, InitializeModes
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.operators.library import FCIDump
    from qiskit_fermions.operators.terms.grouping import (
        group_terms_by_electronic_structure,
    )
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.passes import QDriftTrotterization
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    from embasi_qiskit_integration.hamiltonian import fcidump

    # qiskit-fermions loads the Hamiltonian from a FCIDUMP file; round-trip our
    # in-memory integrals through a temporary one.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sqdrift.fcidump"
        fcidump.write(ham, path)
        fc = FCIDump.from_file(str(path))
        num_modes = 2 * fc.norb
        hamil = FermionOperator.from_fcidump(fc)

    normal = hamil.normal_ordered().simplify(atol=1e-16)
    rc = group_terms_by_electronic_structure(normal, num_modes, two_body_physicist_order=False)
    if rc is not None:  # pragma: no cover - defensive; success returns None
        raise RuntimeError(f"group_terms_by_electronic_structure failed (rc={rc})")

    occ = _hf_occupation(ham.norb, ham.nelec)

    def _base_circuit() -> "FermionicCircuit":
        circ = FermionicCircuit(num_modes)
        circ.append(InitializeModes(occ), circ.modes)
        circ.append(Evolution(num_modes, normal, time), circ.modes)
        return circ

    if method == "exact":
        pm = generate_preset_jw_pass_manager()
        circuits = [pm.run(_base_circuit())]
    elif method == "qdrift":
        qdrift = QDriftTrotterization(
            num_terms, filter_diagonal_terms=filter_diagonal_terms, rng=seed
        )
        pm = generate_preset_jw_pass_manager()
        pm.optimization = FermionicPassManager([qdrift])
        base = _base_circuit()
        circuits = [pm.run(base) for _ in range(num_randomizations)]
    else:
        raise ValueError(f"unknown method {method!r}; use 'exact' or 'qdrift'")

    if measure:
        out = []
        for qc in circuits:
            qc = qc.copy()
            qc.measure_all()
            out.append(qc)
        return out
    return circuits
