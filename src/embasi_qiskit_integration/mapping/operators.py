# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Fermionic operator construction + qubit mapping.

Note: SQD diagonalizes directly in the *fermionic* CI space, so the qubit
mapping here is needed only for circuit construction and observable evaluation,
not for the SQD solve itself. The primary path uses ``qiskit-fermions``
(``FermionOperator`` + Jordan-Wigner); it is optional (Rust build) and imported
lazily. When it is unavailable, :func:`fermionic_op_from_integrals` raises with
guidance rather than silently using a divergent backend.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def _require_fermions():
    try:
        import qiskit_fermions  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "Fermionic operator mapping requires qiskit-fermions (not on PyPI; "
            "needs Rust). Install from source (see README) or via "
            "the 'fermions' extra: pip install -e '.[fermions]'."
        ) from exc


def fermionic_op_from_integrals(ham: EmbeddedHamiltonian):
    """Build a ``qiskit_fermions.FermionOperator`` from ``ham``'s integrals.

    Routes through a temporary FCIDUMP file (the qiskit-fermions loader entry
    point), so the operator is normal-ordered/simplified by the caller as
    needed.
    """
    _require_fermions()
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.operators.library import FCIDump

    from embasi_qiskit_integration.hamiltonian import fcidump

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "op.fcidump"
        fcidump.write(ham, path)
        fc = FCIDump.from_file(str(path))
        return FermionOperator.from_fcidump(fc)


def to_qubit_op(op, num_qubits: int, mapping: str = "jordan-wigner"):
    """Map a fermionic operator to a qubit operator (SparseObservable).

    Args:
        op: a ``qiskit_fermions.FermionOperator``.
        num_qubits: number of qubits (``2 * norb`` for Jordan-Wigner).
        mapping: currently only ``"jordan-wigner"`` is supported.
    """
    _require_fermions()
    from qiskit_fermions.mappers.library import jordan_wigner

    key = mapping.lower().replace("_", "-")
    if key in ("jordan-wigner", "jw"):
        return jordan_wigner(op, num_qubits)
    raise ValueError(f"unsupported mapping {mapping!r}; only 'jordan-wigner' is available")
