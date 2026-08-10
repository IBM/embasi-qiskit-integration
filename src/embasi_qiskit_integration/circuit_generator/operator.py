# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Fermionic operator construction from active-space integrals.

The single operator builder for the package: integrals -> ``FermionOperator``,
normal-ordered, simplified, optionally diagonal-filtered, grouped by electronic
structure, and relabeled into a canonical order. Circuit modules
(:mod:`..sqdrift`) consume the result; the Jordan-Wigner mapping itself is the
pass manager they run, not a separate stage here.

``qiskit-fermions`` is not on PyPI and needs a Rust toolchain (see the README
prerequisites and the ``fermions`` extra), so every import of it is lazy and this
module is importable without it.

Determinism
-----------
``QDriftTrotterization`` samples a group index ``k`` into an array whose
index->group mapping is ``np.unique(groups)`` -- the *sorted group labels*. Those
labels are assigned by ``group_terms_by_electronic_structure`` in a
process-dependent order, so the same seeded draw maps to a different physical
group across invocations: two rebuilds of one operator already diverge.
:func:`_canonicalize_group_order` fixes this without touching the stock pass, by
relabeling groups into a content-derived order before the operator is handed
over.

Diagonal-term filtering is applied *here*, before grouping and canonicalization,
rather than by pruning the operator inside the qDRIFT pass. If the pass pruned
instead, the canonical ranking would be computed over the *unfiltered* operator
while the pass samples the *filtered* one, breaking the stable
group->sample-index mapping that makes the seeded draw reproducible.

This is separate from the pass's own ``filter_trivial`` option, which rejects
individual *draws* that cannot change the occupation; that one is orthogonal and
safe to use alongside this pruning (see :mod:`.sqdrift`).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from embasi_qiskit_integration.circuit_generator.utils import complex_sort_key, term_sort_key
from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def fermions_available() -> bool:
    """True if ``qiskit-fermions`` is importable in this environment."""
    import importlib.util

    return importlib.util.find_spec("qiskit_fermions") is not None


def _require_fermions() -> None:
    try:
        import qiskit_fermions  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "Fermionic operator construction requires qiskit-fermions, which is not "
            "on PyPI and needs a Rust toolchain. Install it from source (see README) "
            "or via the 'fermions' extra: pip install -e '.[fermions]'."
        ) from exc


def fermionic_op_from_integrals(ham: EmbeddedHamiltonian):
    """Build a raw ``qiskit_fermions.FermionOperator`` from ``ham``'s integrals.

    Routes through a temporary FCIDUMP file (the qiskit-fermions loader entry
    point). The returned operator is *not* normal-ordered, simplified, grouped or
    canonicalized -- use :func:`build_canonical_operator` for the operator the
    SqDRIFT pipeline consumes.

    Returns:
        ``(operator, num_modes)`` where ``num_modes == 2 * ham.norb``.
    """
    _require_fermions()
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.operators.library import FCIDump

    from embasi_qiskit_integration.hamiltonian import fcidump

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "op.fcidump"
        fcidump.write(ham, path)
        fc = FCIDump.from_file(str(path))
        return FermionOperator.from_fcidump(fc), 2 * fc.norb


def _canonicalize_group_order(operator: Any) -> Any:
    """Return a copy of ``operator`` with groups relabeled into a canonical order.

    For a seeded qDRIFT draw to be reproducible, sampled index ``k`` must always
    map to the same physical group -- but ``group_terms_by_electronic_structure``
    assigns labels in a process-dependent order. We recompute a stable rank per
    group from its own term structure (:func:`term_sort_key`) plus its aggregated
    weight -- read from the same ``group_weights()`` accessor the pass itself
    samples, quantized via :func:`complex_sort_key` so float jitter cannot reorder
    groups -- then remap every term's group tag to that rank and rebuild via
    ``from_terms_with_groups``.

    When the operator carries no groups (defensive -- the SqDRIFT pipeline always
    groups first), the flat term order is canonicalized instead via the native
    ``order_terms``, keyed on the same (action pattern, quantized coefficient) so
    the ungrouped path is equally reproducible.
    """
    _require_fermions()
    from qiskit_fermions.operators.terms import order_terms

    if operator.groups is None:
        return order_terms(
            operator,
            key=lambda term: (
                tuple((bool(is_creation), int(mode)) for is_creation, mode in term[0]),
                complex_sort_key(term[1]),
            ),
        )

    # Aggregated per-group weight, read from the same native accessor
    # ``QDriftTrotterization`` samples with (``np.array(hamil.group_weights())``):
    # sum of |coeff| within a group, divided by that group's term count. Taking it
    # from upstream rather than recomputing it keeps the ranking aligned with the
    # pass by construction, and handles a group index no term carries (weight 0.0)
    # -- which a hand-rolled ``np.add.at`` / ``np.unique`` reduction cannot, since
    # those two disagree in length whenever the labels are non-contiguous.
    weights = np.asarray(operator.group_weights())

    # Both ``group_weights()`` and ``split_out_groups()`` are indexed by raw group
    # label and length ``num_groups()``, so they line up; unused labels appear as
    # an empty sub-operator with weight 0.0.
    split = operator.split_out_groups()
    keys = [(term_sort_key(split[i]), complex_sort_key(weights[i])) for i in range(len(split))]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    rank = {old_label: new_label for new_label, old_label in enumerate(order)}

    relabeled_terms = [
        (actions, coeff, rank[group]) for actions, coeff, group in operator.iter_terms_with_groups()
    ]
    return operator.__class__.from_terms_with_groups(relabeled_terms)


def build_canonical_operator(
    ham: EmbeddedHamiltonian,
    *,
    atol: float = 1e-16,
    filter_diagonal_terms: bool = True,
) -> tuple[Any, int]:
    """Build the canonicalized, grouped Hamiltonian the SqDRIFT pipeline samples.

    Performs the full construction: integrals -> ``FermionOperator``, dropping
    empty (identity) terms, ``normal_ordered().simplify(atol)``, optional
    diagonal-term filtering, electronic-structure grouping, and
    :func:`_canonicalize_group_order`.

    Diagonal filtering happens here rather than in the qDRIFT pass so that
    grouping and canonicalization see the same term set the pass will sample --
    required for the seeded draw to be reproducible across rebuilds.

    The returned operator is treated as read-only by callers: each randomization
    wraps it in a fresh ``FermionicCircuit``, so it is never mutated in place.

    Args:
        ham: The active-space Hamiltonian.
        atol: Tolerance for simplifying the normal-ordered operator.
        filter_diagonal_terms: Remove diagonal (number-operator) terms before
            grouping, yielding more expressive circuits.

    Returns:
        ``(operator, num_modes)`` with ``num_modes == 2 * ham.norb``.
    """
    _require_fermions()
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.operators.terms.filtering import (
        filter_diagonal_terms as _filter_diagonal_terms,
    )
    from qiskit_fermions.operators.terms.grouping import group_terms_by_electronic_structure

    hamil, num_modes = fermionic_op_from_integrals(ham)
    # Drop empty (identity) terms: they contribute a global phase only, and an
    # all-identity group would make the excitation-span solve degenerate.
    hamil = FermionOperator.from_terms(
        [(term, coeff) for term, coeff in hamil.iter_terms() if len(term) > 0]
    )
    normal = hamil.normal_ordered().simplify(atol=atol)
    if filter_diagonal_terms:
        _filter_diagonal_terms(normal)
    # Groups ``normal`` in place. Upstream types this as returning None and raises
    # on failure, but it is an unreleased git dependency: if it ever switches to a
    # return code, a silent grouping failure would yield a wrong-but-plausible
    # circuit. Check rather than assume (an ``assert`` would vanish under -O).
    grouping_result = group_terms_by_electronic_structure(  # type: ignore[func-returns-value]
        normal, num_modes, two_body_physicist_order=False
    )
    if grouping_result is not None:  # pragma: no cover - tripwire for upstream drift
        raise RuntimeError(
            f"group_terms_by_electronic_structure returned {grouping_result!r}; expected None. "
            "The qiskit-fermions grouping API may have changed to signal failure by "
            "return code -- verify grouping before trusting generated circuits."
        )

    # Relabel groups into a canonical, run-invariant order so seeded qDRIFT
    # sampling is reproducible across process invocations.
    return _canonicalize_group_order(normal), num_modes
