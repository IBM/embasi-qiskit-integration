# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Undoing mode relabeling on sampled bitstrings."""

from __future__ import annotations

import pytest

from embasi_qiskit_integration.circuit_run.permutation import (
    little_endian_gather,
    unpermute_bitstrings,
    unpermute_counts,
    unpermute_counts_list,
    validate_permutation,
)


def test_matches_qiskit_fermions_documented_recipe():
    """Our vectorized inverse equals the recipe in ``RelabelModes``' docstring.

    Upstream documents the un-relabeling as a per-bitstring comprehension:
    ``"".join(bitstring[~i] for i in permutation)[::-1]``. We fold the index
    negation and the reversal into one gather so all bitstrings are mapped at
    once; this pins the two to the same answer, including the exact worked example
    from that docstring (``'000111' -> '001011'`` under ``[0, 2, 4, 1, 3, 5]``).
    """
    permutation = [0, 2, 4, 1, 3, 5]
    bitstrings = ["000111", "001011", "101010", "111000", "010101"]

    documented = ["".join(b[~i] for i in permutation)[::-1] for b in bitstrings]
    assert unpermute_bitstrings(bitstrings, permutation) == documented
    # The worked example from the upstream docstring.
    assert unpermute_counts({"000111": 1}, permutation) == {"001011": 1}


def test_identity_permutation_is_a_no_op():
    counts = {"0011": 5, "1100": 7}
    assert unpermute_counts(counts, [0, 1, 2, 3]) == counts


def test_none_permutation_passes_counts_through():
    """An unrelabeled circuit's counts must survive untouched (mixed ensembles)."""
    counts = {"0011": 5, "1100": 7}
    assert unpermute_counts(counts, None) == counts


def test_round_trip_restores_original_bitstrings():
    """Applying the permutation then its inverse is the identity.

    ``permutation[i]`` is where original mode ``i`` ends up, so *applying* it to a
    bitstring is a scatter; ``unpermute_bitstrings`` is the corresponding gather.
    Composing them must return the input for every bitstring.
    """
    permutation = [3, 0, 5, 1, 4, 2]
    num_qubits = len(permutation)
    originals = ["000111", "101010", "110001", "111111", "000000"]

    def apply_relabeling(bitstring: str) -> str:
        # MSB-left string: character -(mode + 1) is mode `mode`.
        out = [""] * num_qubits
        for mode, new_index in enumerate(permutation):
            out[-(new_index + 1)] = bitstring[-(mode + 1)]
        return "".join(out)

    relabeled = [apply_relabeling(b) for b in originals]
    assert unpermute_bitstrings(relabeled, permutation) == originals


def test_preserves_total_shots_and_hamming_weight():
    """The inverse is a bijection: no shots lost, occupation count unchanged."""
    permutation = [5, 2, 0, 4, 1, 3]
    counts = {"000111": 10, "101010": 20, "110100": 30}

    restored = unpermute_counts(counts, permutation)
    assert sum(restored.values()) == sum(counts.values())
    assert len(restored) == len(counts)
    assert sorted(b.count("1") for b in restored) == sorted(b.count("1") for b in counts)


def test_rejects_malformed_permutation():
    """A wrong-length or duplicated permutation must raise, not scramble silently."""
    with pytest.raises(ValueError, match="rearrangement"):
        validate_permutation([0, 1, 2], 4)
    with pytest.raises(ValueError, match="rearrangement"):
        validate_permutation([0, 1, 1, 2], 4)
    with pytest.raises(ValueError, match="rearrangement"):
        unpermute_counts({"0011": 1}, [0, 1, 2])


def test_rejects_ragged_bitstring_widths():
    with pytest.raises(ValueError, match="same width"):
        unpermute_bitstrings(["0011", "00111"], [0, 1, 2, 3])


def test_gather_indices_are_a_permutation():
    """The little-endian gather is itself a permutation of the columns."""
    permutation = [3, 0, 5, 1, 4, 2]
    gather = little_endian_gather(permutation, 6)
    assert sorted(gather.tolist()) == list(range(6))


def test_counts_list_applies_each_circuits_own_permutation():
    """Ensemble members carry different permutations; each must get its own."""
    counts_list = [{"000111": 1}, {"000111": 1}, {"000111": 1}]
    permutations = [[0, 2, 4, 1, 3, 5], None, [5, 4, 3, 2, 1, 0]]

    restored = unpermute_counts_list(counts_list, permutations)
    assert restored[0] == {"001011": 1}  # documented example
    assert restored[1] == {"000111": 1}  # unpermuted, passed through
    assert restored[2] == {"111000": 1}  # full reversal
    # Using one member's permutation for all of them would be wrong:
    assert restored[0] != restored[2]


def test_counts_list_rejects_misaligned_lengths():
    """Pairing counts with another circuit's permutation must be impossible."""
    with pytest.raises(ValueError, match="aligned one-to-one"):
        unpermute_counts_list([{"01": 1}, {"10": 1}], [[0, 1]])


def test_empty_inputs():
    assert unpermute_counts({}, [0, 1]) == {}
    assert unpermute_bitstrings([], [0, 1]) == []


def test_bitstrings_with_register_spaces_are_accepted():
    """Qiskit counts keys can carry register separators; they must not break the map."""
    assert unpermute_counts({"000 111": 1}, [0, 2, 4, 1, 3, 5]) == {"001011": 1}
