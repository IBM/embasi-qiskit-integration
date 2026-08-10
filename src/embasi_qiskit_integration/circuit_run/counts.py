# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Measurement-counts extraction from ``SamplerV2`` results, and count merging.

One implementation shared by every sampler backend (Aer, IBM Runtime), so the
Aer and hardware paths cannot drift apart in how they read a ``DataBin`` or name
a classical register.

A ``SamplerV2`` PUB result carries its measured bits in a ``BitArray`` on a
classical register. ``measure_all()`` names that register ``meas``;
:func:`find_meas_bitarray` prefers the known names and falls back to the sole
public attribute for circuits using a custom one.

A PUB may also carry *several* binding positions (a parameter-bindings array of
shape ``(N, n_params)`` runs one circuit N times). :func:`counts_per_binding_from_pub_result`
keeps those attributed one dict per binding;
:func:`counts_from_pub_result` collapses them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any

# Classical-register names tried in order. ``measure_all()`` emits ``meas``;
# ``measure_active()`` and hand-built circuits commonly use ``c``/``c0``.
_CLASSICAL_REGISTER_CANDIDATES = ("meas", "c", "c0")


def find_meas_bitarray(data_bin: Any) -> Any:
    """Return the classical-register ``BitArray`` from a PUB result's ``data``.

    Prefers the conventional register names in
    :data:`_CLASSICAL_REGISTER_CANDIDATES`, then falls back to the single public
    attribute on the ``DataBin`` so circuits with a custom register name still
    work.

    Raises:
        ValueError: if the ``DataBin`` exposes no register, or exposes several
            and none of them carries a conventional name (ambiguous -- measure
            into one register, or name it ``meas``).
    """
    for register_name in _CLASSICAL_REGISTER_CANDIDATES:
        if hasattr(data_bin, register_name):
            return getattr(data_bin, register_name)

    # A ``DataBin`` is a mapping over its data fields, so ``keys()`` names exactly
    # the registers present (verified: ``['meas']`` after measure_all, ``['c']``
    # for a hand-built register).
    names = list(data_bin.keys()) if hasattr(data_bin, "keys") else []
    if not names:
        raise ValueError("PUB result data exposes no classical register")
    if len(names) != 1:
        raise ValueError(
            f"expected a single classical register, found {names!r}; "
            "measure into one register or a register named 'meas'"
        )
    return data_bin[names[0]]


def counts_per_binding_from_pub_result(pub_result: Any) -> list[dict[str, int]]:
    """Extract *per-binding* count dicts from a ``SamplerV2`` PUB result.

    A single-binding PUB (``BitArray`` shape ``()``) yields a length-1 list. A
    multi-binding PUB (shape ``(N,)`` from an ``(N, n_params)`` bindings matrix,
    or higher-dimensional from a ``BindingsArray``) yields one dict per binding
    position, flattened in row-major order.
    """
    bit_array = find_meas_bitarray(pub_result.data)
    if bit_array.ndim == 0:
        return [_normalize(bit_array.get_counts())]
    indices = product(*(range(dim) for dim in bit_array.shape))
    return [_normalize(bit_array.get_counts(loc=index)) for index in indices]


def counts_from_pub_result(pub_result: Any) -> dict[str, int]:
    """Return the counts of a PUB result, pooled across all binding positions.

    Convenience for the single-binding case; for a multi-binding PUB prefer
    :func:`counts_per_binding_from_pub_result` to keep per-binding attribution.
    """
    return merge_counts(counts_per_binding_from_pub_result(pub_result))


def merge_counts(counts_list: Iterable[Mapping[str, int]]) -> dict[str, int]:
    """Pool several ``{bitstring: count}`` dicts by summing per bitstring.

    Used to combine an ensemble of sampling circuits (e.g. SqDRIFT
    randomizations) into the single empirical distribution the SQD driver
    consumes. The pooled total is the sum of the inputs' totals, so a caller
    sampling ``N`` circuits at ``shots`` each ends up with ``N * shots``.
    """
    merged: dict[str, int] = {}
    for counts in counts_list:
        for bitstring, count in counts.items():
            merged[str(bitstring)] = merged.get(str(bitstring), 0) + int(count)
    return merged


def _normalize(counts: Mapping[str, int]) -> dict[str, int]:
    """Coerce a counts mapping to plain ``{str: int}``."""
    return {str(bitstring): int(count) for bitstring, count in counts.items()}
