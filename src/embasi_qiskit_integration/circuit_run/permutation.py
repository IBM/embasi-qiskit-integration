# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Undo mode relabeling on sampled bitstrings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def validate_permutation(permutation: Sequence[int], num_qubits: int) -> list[int]:
    """Return ``permutation`` as a list, checking it permutes ``range(num_qubits)``.

    Raises:
        ValueError: if it is not a rearrangement of ``range(num_qubits)`` -- a
            wrong-length or duplicated permutation would silently scramble
            occupations rather than fail.
    """
    perm = [int(index) for index in permutation]
    if sorted(perm) != list(range(num_qubits)):
        raise ValueError(
            f"permutation must be a rearrangement of range({num_qubits}), got {perm!r}"
        )
    return perm


def little_endian_gather(permutation: Sequence[int], num_qubits: int) -> np.ndarray:
    """Column indices that map permuted-order bitstrings back to the original order.

    ``permutation`` is mode-indexed (original mode ``i`` sits at new index
    ``permutation[i]``) while a counts bitstring is MSB-left, so qubit ``0`` is
    its *last* character. Reversing and complementing the indices converts the
    mode-space gather into a single column gather over MSB-left strings.
    """
    perm = validate_permutation(permutation, num_qubits)
    return (num_qubits - 1) - np.asarray(perm[::-1])


def unpermute_bitstrings(bitstrings: Sequence[str], permutation: Sequence[int]) -> list[str]:
    """Map MSB-left ``bitstrings`` from a relabeled mode order back to the original.

    All bitstrings must share one width, which is taken as ``num_qubits``.

    Raises:
        ValueError: if the widths disagree, or ``permutation`` is not a
            permutation of ``range(num_qubits)``.
    """
    if not bitstrings:
        return []

    cleaned = [str(bitstring).replace(" ", "") for bitstring in bitstrings]
    widths = {len(bitstring) for bitstring in cleaned}
    if len(widths) != 1:
        raise ValueError(f"bitstrings must all have the same width, got widths {sorted(widths)}")
    num_qubits = widths.pop()

    gather = little_endian_gather(permutation, num_qubits)
    # A fixed-width unicode array viewed as uint32 is an (n, num_qubits) matrix of
    # code points, so one np.take reorders every bitstring's columns at once; the
    # view back to "<U{num_qubits}" reassembles the strings.
    keys = np.asarray(cleaned, dtype=f"<U{num_qubits}")
    codes = keys.view(np.uint32).reshape(len(cleaned), num_qubits)
    return np.take(codes, gather, axis=1).view(f"<U{num_qubits}").ravel().tolist()


def unpermute_counts(
    counts: Mapping[str, int], permutation: Sequence[int] | None
) -> dict[str, int]:
    """Rewrite a ``{bitstring: count}`` dict from a relabeled mode order to the original.

    ``permutation=None`` (the circuit was not relabeled) returns the counts
    unchanged, so this is safe to call unconditionally on a mixed ensemble.

    Distinct permuted bitstrings can never collide after the inverse map -- it is
    a bijection on fixed-width strings -- but counts are summed on collision
    anyway rather than overwritten, so a caller passing a pre-pooled dict cannot
    silently lose shots.
    """
    if permutation is None:
        return {str(bitstring): int(count) for bitstring, count in counts.items()}
    if not counts:
        return {}

    original = unpermute_bitstrings(list(counts.keys()), permutation)
    restored: dict[str, int] = {}
    for bitstring, count in zip(original, counts.values(), strict=True):
        restored[bitstring] = restored.get(bitstring, 0) + int(count)
    return restored


def unpermute_counts_list(
    counts_list: Sequence[Mapping[str, int]],
    permutations: Sequence[Sequence[int] | None],
) -> list[dict[str, int]]:
    """Apply :func:`unpermute_counts` per circuit, pairing counts with permutations.

    Each ensemble member carries its own permutation (``RelabelModes`` solves the
    MILP per randomization), so the inverse must be applied *before* the counts
    are pooled -- pooling first would mix mutually-inconsistent mode orders.

    Raises:
        ValueError: if the two sequences have different lengths, which would mean
            counts are being paired with another circuit's permutation.
    """
    if len(counts_list) != len(permutations):
        raise ValueError(
            f"got {len(counts_list)} counts dicts but {len(permutations)} permutations; "
            "they must be aligned one-to-one with the sampled circuits"
        )
    return [
        unpermute_counts(counts, permutation)
        for counts, permutation in zip(counts_list, permutations, strict=True)
    ]
