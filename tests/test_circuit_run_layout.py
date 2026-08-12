# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Noise-aware layout selection: chain search, pruning, ranking, metrics.

These are the pure functions of the characterisation pipeline, so they are fully
testable offline. What cannot be tested here is whether pinning a measured-best
layout actually improves a hardware energy -- that needs a real device.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.transpiler import CouplingMap

from embasi_qiskit_integration.circuit_run.layout import (
    find_lines,
    get_modified_coupling_map,
    get_viable_paths,
    has_feasible_chain,
    largest_component_size,
    rank_layouts,
    select_layout,
)
from embasi_qiskit_integration.circuit_run.metrics import (
    compute_manual_readout_errors,
    compute_trex_fidelities,
    compute_xslow_flip_fidelity,
)


def _line(n: int) -> CouplingMap:
    """A 1-D chain ``0-1-...-(n-1)``."""
    return CouplingMap([[i, i + 1] for i in range(n - 1)])


def _shots_with_errors(errors: list[float], n_rand: int = 8, shots: int = 200) -> np.ndarray:
    """Synthetic corrected shots whose per-qubit mean approximates ``errors``."""
    rng = np.random.default_rng(0)
    probabilities = np.asarray(errors)
    return (rng.random((n_rand, shots, len(errors))) < probabilities).astype(int)


# ----- chain search ---------------------------------------------------------- #


def test_find_lines_enumerates_chains_of_exact_length():
    chains = find_lines(3, _line(6))
    assert chains == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    # The full-width chain is unique on a line.
    assert find_lines(6, _line(6)) == [[0, 1, 2, 3, 4, 5]]
    # Nothing longer than the device.
    assert find_lines(7, _line(6)) == []


def test_find_lines_treats_a_chain_and_its_reverse_as_one_layout():
    """Chains are undirected: ``[0,1]`` and ``[1,0]`` are the same placement."""
    chains = find_lines(2, _line(3))
    assert len(chains) == 2  # (0,1) and (1,2), not four directed variants
    assert {tuple(sorted(c)) for c in chains} == {(0, 1), (1, 2)}


def test_find_lines_is_deterministic():
    """Neighbours are visited in sorted order, so the result is reproducible.

    Layout choice feeds the transpile, so an unstable enumeration would make two
    identical runs land on different physical qubits.
    """
    ring = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 0]])
    assert find_lines(3, ring) == find_lines(3, ring)


def test_find_lines_respects_the_candidate_cap():
    capped = find_lines(2, _line(10), max_candidates=3)
    assert len(capped) == 3


def test_find_lines_respects_the_visit_budget():
    """The work budget bounds the no-chain-found branch rather than the results.

    A search for an *impossible* chain on a large graph would otherwise backtrack
    the whole DFS tree (millions of extensions); exhausting the budget returns
    whatever was found so far. The star has no 3-chain at all, so an unbudgeted
    search explores every branch to no end.
    """
    # 20 leaves on one hub: no chain longer than 3 exists, so the search is pure
    # backtracking and the budget is what stops it.
    star = CouplingMap([[0, leaf] for leaf in range(1, 21)])
    assert find_lines(5, star, max_visits=10) == []
    # Unbudgeted, the same search still terminates and still finds nothing.
    assert find_lines(5, star) == []


def test_find_lines_rejects_nonpositive_length():
    assert find_lines(0, _line(4)) == []
    assert find_lines(-1, _line(4)) == []


def test_find_lines_length_one_returns_every_qubit():
    assert find_lines(1, _line(4)) == [[0], [1], [2], [3]]


# ----- component sizing and the cheap feasibility gate ----------------------- #


def test_largest_component_size_and_feasibility():
    # Two disjoint pieces: 0-1 and 3-4-5.
    split = CouplingMap([[0, 1], [3, 4], [4, 5]])
    assert largest_component_size(split) == 3
    assert has_feasible_chain(split, 3) is True
    # No component holds 4 nodes, so the DFS can be skipped entirely.
    assert has_feasible_chain(split, 4) is False
    assert has_feasible_chain(split, 0) is False


def test_feasibility_is_necessary_not_sufficient():
    """A big component need not contain a long *simple path*.

    A star has 4 nodes but no 3-chain, so the gate says "maybe" and ``find_lines``
    is what actually decides.
    """
    star = CouplingMap([[0, 1], [0, 2], [0, 3]])
    assert largest_component_size(star) == 4
    assert has_feasible_chain(star, 4) is True
    assert find_lines(4, star) == []


# ----- pruning --------------------------------------------------------------- #


def test_pruning_removes_a_qubit_and_all_its_edges():
    pruned = get_modified_coupling_map(_line(6), bad_qubits=[2], bad_edges=[])
    remaining = {tuple(sorted(e)) for e in pruned.get_edges()}
    assert not any(2 in edge for edge in remaining)
    # The line is now split into 0-1 and 3-4-5.
    assert largest_component_size(pruned) == 3


def test_pruning_removes_an_edge_in_both_directions():
    pruned = get_modified_coupling_map(_line(4), bad_qubits=[], bad_edges=[(2, 1)])
    remaining = {tuple(sorted(e)) for e in pruned.get_edges()}
    assert (1, 2) not in remaining
    assert (0, 1) in remaining


def test_pruning_does_not_mutate_the_input_map():
    original = _line(4)
    before = sorted(tuple(e) for e in original.get_edges())
    get_modified_coupling_map(original, bad_qubits=[1], bad_edges=[])
    assert sorted(tuple(e) for e in original.get_edges()) == before


def test_get_viable_paths_composes_pruning_and_search():
    assert get_viable_paths(3, _line(6), bad_qubits=[2]) == [[3, 4, 5]]


# ----- metrics --------------------------------------------------------------- #


def test_manual_readout_errors_recover_the_injected_rates():
    """With all-zero preparation, the mean of corrected shots *is* the error rate.

    Tolerance is 3 standard errors of the binomial estimate rather than a flat
    number: at 1600 samples the 0.20 rate alone has a standard error of ~0.01, so a
    fixed atol either passes vacuously for the small rates or flakes on the big one.
    """
    errors = np.array([0.01, 0.05, 0.20, 0.0])
    n_rand, shots = 8, 200
    measured = compute_manual_readout_errors(
        _shots_with_errors(errors.tolist(), n_rand=n_rand, shots=shots)
    )
    assert measured.shape == (4,)

    n_samples = n_rand * shots
    standard_error = np.sqrt(errors * (1 - errors) / n_samples)
    assert np.all(np.abs(measured - errors) <= 3 * standard_error + 1e-12)
    # An error-free qubit reads exactly zero, not merely close to it.
    assert measured[3] == 0.0


def test_xslow_flip_fidelity_is_one_when_the_remeasurement_inverts():
    data = np.zeros((2, 5, 3), dtype=int)
    inverted = np.ones((2, 5, 3), dtype=int)
    assert np.allclose(compute_xslow_flip_fidelity(data, inverted), 1.0)
    # And zero when it fails to invert at all.
    assert np.allclose(compute_xslow_flip_fidelity(data, data), 0.0)


def test_trex_fidelity_is_one_for_error_free_shots():
    shots = np.zeros((2, 10, 3), dtype=int)
    assert np.allclose(compute_trex_fidelities(shots, 3), 1.0)


# ----- ranking --------------------------------------------------------------- #


def test_rank_layouts_prefers_the_cleaner_chain():
    """Score is the joint readout fidelity, so one bad qubit sinks a layout.

    That product is the right figure of merit here: SQD postselects on particle
    number, so a single flipped bit discards the whole shot.
    """
    shots = _shots_with_errors([0.01, 0.01, 0.01, 0.30, 0.30, 0.30])
    candidates = [[3, 4, 5], [0, 1, 2]]
    ranked, scores = rank_layouts(candidates, shots, list(range(6)))

    assert ranked[0] == [0, 1, 2]  # the clean end wins
    assert scores[0] > scores[1]
    assert scores == sorted(scores, reverse=True)


# ----- select_layout: prune + search + adaptive relaxation ------------------- #


def test_select_layout_avoids_the_bad_qubit():
    shots = _shots_with_errors([0.01, 0.01, 0.50, 0.01, 0.01, 0.01])
    pruned, layouts, avoid, threshold = select_layout(
        shots, _line(6), list(range(6)), required_size=3
    )

    assert avoid == [2]
    assert threshold == pytest.approx(0.03)
    assert layouts == [[3, 4, 5]]
    assert not any(2 in edge for edge in pruned.get_edges())


def test_select_layout_relaxes_the_threshold_when_pruning_fragments_the_lattice():
    """Pruning often leaves no chain; the threshold must relax rather than fail.

    Here every qubit is mildly above the 0.03 default, so a strict prune deletes the
    whole device. Relaxing by x1.5 until a chain appears is what keeps the run alive.
    """
    shots = _shots_with_errors([0.05] * 6)
    _pruned, layouts, avoid, threshold = select_layout(
        shots, _line(6), list(range(6)), required_size=4, readout_error_threshold=0.03
    )

    assert layouts, "expected the relaxation to find a chain"
    assert threshold > 0.03
    assert avoid == []  # nothing exceeds the relaxed threshold


def test_select_layout_returns_no_layout_when_the_device_is_too_small():
    """A chain wider than the device yields nothing, even fully unpruned.

    The caller decides whether that is fatal; ``select_readout_layout`` raises.
    """
    shots = _shots_with_errors([0.01] * 4)
    _pruned, layouts, _avoid, threshold = select_layout(
        shots, _line(4), list(range(4)), required_size=8
    )
    assert layouts == []
    assert threshold == pytest.approx(1.0)  # relaxed all the way to the ceiling


def test_select_layout_honours_extra_bad_edges():
    shots = _shots_with_errors([0.01] * 6)
    _pruned, layouts, _avoid, _threshold = select_layout(
        shots, _line(6), list(range(6)), required_size=3, bad_edges=[(2, 3)]
    )
    # Severing 2-3 leaves 0-1-2 and 3-4-5 as the only 3-chains.
    assert [sorted(layout) for layout in layouts] == [[0, 1, 2], [3, 4, 5]]


def test_path_score_prefers_the_better_chain():
    """The richer T1/SPAM/Bell criterion.

    Nothing in this package calls it -- the readout-twirl characterisation collects
    neither T1 times nor per-edge Bell fidelities -- so this pins that it is correct
    and importable for callers that do have that data.
    """
    from embasi_qiskit_integration.circuit_run.layout import path_score

    t1 = np.array([100.0, 120.0, 90.0, 150.0])
    trex = np.array([0.98, 0.99, 0.95, 0.99])
    bell = {(0, 1): 0.95, (1, 2): 0.90, (2, 3): 0.97}
    kwargs = {"cutoff_t1": 50.0, "cutoff_spam": 0.9, "cutoff_bell": 0.85}

    better = path_score([1, 2, 3], t1, trex, bell, **kwargs)
    worse = path_score([0, 1, 2], t1, trex, bell, **kwargs)
    assert better > worse


def test_path_score_accepts_edges_in_either_direction():
    """Bell fidelities are undirected, so a reversed key must still be found."""
    from embasi_qiskit_integration.circuit_run.layout import path_score

    t1 = np.array([100.0, 120.0])
    trex = np.array([0.98, 0.99])
    kwargs = {"cutoff_t1": 50.0, "cutoff_spam": 0.9, "cutoff_bell": 0.85}

    forward = path_score([0, 1], t1, trex, {(0, 1): 0.95}, **kwargs)
    reversed_key = path_score([0, 1], t1, trex, {(1, 0): 0.95}, **kwargs)
    assert forward == pytest.approx(reversed_key)
