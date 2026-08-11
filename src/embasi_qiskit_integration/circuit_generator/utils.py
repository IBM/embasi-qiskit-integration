# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

# Tolerance for quantizing coefficients into orderable sort keys (see
# ``complex_sort_key``). Coefficients differing only by float jitter below this
# collapse to the same key, so they cannot reorder groups between rebuilds.
_COMPLEX_SORT_TOL = 1e-12


def complex_sort_key(coeff: complex, tol: float = _COMPLEX_SORT_TOL) -> tuple:
    """Return a tolerance-quantized orderable key for a complex number.

    Python refuses ``complex < complex`` because there is no natural total order.
    We build one by lex-sorting on quantized ``(|z|, Re(z), Im(z))``, rounded to
    ``tol`` so that values differing only by float jitter share a key.
    """
    z = complex(coeff)
    return (
        round(abs(z) / tol) * tol,
        round(z.real / tol) * tol,
        round(z.imag / tol) * tol,
    )


def term_sort_key(term_op: Any) -> tuple:
    """Return an orderable canonical key for a ``FermionOperator`` (group of terms).

    Iterates over ``(actions, coeff)`` yielded by ``iter_terms()``. Each
    ``action`` is a ``(is_creation: bool, mode: int)`` pair, and the mode index is
    kept in the key: groups sharing a creation/annihilation pattern but acting on
    different orbitals must not collide, otherwise stable-sort ties reintroduce
    the process-dependent ordering this key exists to remove.

    Coefficients are wrapped via :func:`complex_sort_key`, and the parts are
    sorted so the key does not depend on ``iter_terms()``'s own iteration order.
    """
    parts = []
    for actions, coeff in term_op.iter_terms():
        action_tuple = tuple((bool(is_creation), int(mode)) for is_creation, mode in actions)
        parts.append((action_tuple, complex_sort_key(coeff)))
    parts.sort()
    return tuple(parts)
