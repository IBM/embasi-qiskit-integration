# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Coupling-map pruning and 1-D chain layout selection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from embasi_qiskit_integration.circuit_run.metrics import compute_manual_readout_errors

logger = logging.getLogger(__name__)

# Max distinct chains returned by ``find_lines``; downstream only needs a few
# candidates for ``rank_layouts`` to score, so we stop early once enough exist.
DEFAULT_MAX_CANDIDATES = 256

# Max DFS extensions per ``find_lines`` call. The result cap bounds results, not
# runtime: a large component with no simple path of the requested length yields
# zero chains and backtracks the whole tree (~8-12M extensions on a 120-qubit
# search). This work budget bounds that failing branch.
DEFAULT_MAX_VISITS = 2_000_000

# Default readout-error ceiling for a usable qubit, and the adaptive relaxation
# applied when pruning at that ceiling fragments the lattice.
DEFAULT_READOUT_ERROR_THRESHOLD = 0.03
DEFAULT_MAX_THRESHOLD = 1.0
DEFAULT_RELAX_FACTOR = 1.5


def _adjacency(coupling_map: Any) -> dict[int, list[int]]:
    """Undirected adjacency with deterministically sorted neighbours.

    Sorting is what makes the DFS in :func:`find_lines` reproducible: the same
    coupling map must always yield the same candidate chains in the same order.
    """
    adjacency: dict[int, list[int]] = {}
    for u, v in coupling_map.get_edges():
        adjacency.setdefault(int(u), []).append(int(v))
        adjacency.setdefault(int(v), []).append(int(u))
    for node in adjacency:
        adjacency[node] = sorted(set(adjacency[node]))
    return adjacency


def find_lines(
    length: int,
    coupling_map: Any,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    max_visits: int = DEFAULT_MAX_VISITS,
) -> list[list[int]]:
    """Find simple paths (chains) of exactly ``length`` qubits in the coupling map.

    Deterministic depth-first walk (nodes visited in ascending order) bounded by
    two budgets: ``max_candidates`` (stop once enough chains are collected) and
    ``max_visits`` (stop after this many DFS extensions even if none found). The
    latter keeps a line-free-but-large graph from an exponential backtrack.

    Returns unique undirected chains as ordered physical-qubit lists. The result is
    a deterministic prefix of the full set when either budget was hit, and may be
    empty if ``max_visits`` was exhausted first.
    """
    if length <= 0:
        return []

    adjacency = _adjacency(coupling_map)

    if length == 1:
        return [[node] for node in sorted(adjacency)][:max_candidates]

    seen: set[tuple[int, ...]] = set()
    chains: list[list[int]] = []
    visits = 0  # DFS extensions taken; bounds work on the "no chain found" branch

    def _extend(path: list[int], on_path: set[int]) -> bool:
        """DFS-extend ``path``; return True to stop (either budget hit)."""
        nonlocal visits
        if len(path) == length:
            # Keyed on the *set* of qubits: a chain and its reverse are the same
            # undirected layout, so only one is kept.
            key = tuple(sorted(path))
            if key not in seen:
                seen.add(key)
                chains.append(list(path))
            return len(chains) >= max_candidates
        for neighbour in adjacency[path[-1]]:
            if neighbour in on_path:
                continue
            visits += 1
            if visits > max_visits:
                return True  # work budget exhausted; return chains found so far
            path.append(neighbour)
            on_path.add(neighbour)
            reached_cap = _extend(path, on_path)
            path.pop()
            on_path.discard(neighbour)
            if reached_cap:
                return True
        return False

    for start in sorted(adjacency):
        if _extend([start], {start}):
            break

    return chains


def largest_component_size(coupling_map: Any) -> int:
    """Return the node count of the largest connected component (O(V+E))."""
    adjacency = _adjacency(coupling_map)
    if not adjacency:
        return 0

    seen: set[int] = set()
    largest = 0
    for start in adjacency:
        if start in seen:
            continue
        size = 0
        stack = [start]
        seen.add(start)
        while stack:
            node = stack.pop()
            size += 1
            for neighbour in adjacency.get(node, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        largest = max(largest, size)
    return largest


def has_feasible_chain(coupling_map: Any, required_size: int) -> bool:
    """Cheap necessary check for a ``required_size`` chain.

    True when some component holds at least that many nodes. Necessary but *not*
    sufficient (a large component need not contain a simple path that long), so a
    True still needs :func:`find_lines` to confirm -- but a False lets the caller
    skip the expensive DFS entirely.
    """
    if required_size <= 0:
        return False
    if required_size == 1:
        return coupling_map.size() > 0 or bool(list(coupling_map.get_edges()))
    return largest_component_size(coupling_map) >= required_size


def get_modified_coupling_map(
    coupling_map: Any,
    bad_qubits: Sequence[int],
    bad_edges: Sequence[tuple[int, int]],
) -> Any:
    """Return a copy of ``coupling_map`` with the given qubits and edges removed.

    Args:
        coupling_map: original device topology.
        bad_qubits: physical qubit indices to drop, along with all their edges.
        bad_edges: additional edges to drop, checked in both directions.

    Returns:
        The pruned :class:`~qiskit.transpiler.CouplingMap`.
    """
    from qiskit.transpiler import CouplingMap

    pruned = deepcopy(coupling_map)
    pruned.make_symmetric()

    bad_qubit_set = set(int(q) for q in bad_qubits)
    edges = [
        edge
        for edge in pruned.get_edges()
        if edge[0] not in bad_qubit_set and edge[1] not in bad_qubit_set
    ]
    if bad_edges:
        excluded = set(tuple(e) for e in bad_edges)
        edges = [
            edge for edge in edges if tuple(edge) not in excluded and edge[::-1] not in excluded
        ]

    return CouplingMap(edges)


def get_viable_paths(
    path_length: int,
    coupling_map: Any,
    bad_qubits: Sequence[int],
    bad_edges: Sequence[tuple[int, int]] | None = None,
) -> list[list[int]]:
    """Prune the map by bad qubits/edges, then return all chains of ``path_length``."""
    truncated = get_modified_coupling_map(coupling_map, bad_qubits, bad_edges or [])
    return find_lines(length=path_length, coupling_map=truncated)


def path_score(
    path_qubits: Sequence[int],
    t1_vals: np.ndarray,
    trex: np.ndarray,
    hellinger_fids: dict[tuple[int, int], float],
    cutoff_t1: float,
    cutoff_spam: float,
    cutoff_bell: float,
) -> float:
    """Score a candidate chain from rescaled T1, SPAM and Bell-state fidelities.

    Higher is better. Each metric is linearly rescaled so its cutoff maps to 0 and
    the best observed value to 1, then summed over the chain's qubits plus its
    nearest-neighbour pairs.

    This is a richer criterion than :func:`rank_layouts` uses, and it needs inputs
    (T1 times, per-edge Bell fidelities) that the readout-twirl characterisation in
    :mod:`.characterisation` does not collect -- so nothing in this package calls it
    yet. Provided for callers that do have that data.

    Args:
        path_qubits: the candidate chain, in order.
        t1_vals: per-qubit T1, indexed by physical qubit.
        trex: per-qubit SPAM/readout fidelity, indexed by physical qubit.
        hellinger_fids: per-edge Bell-state fidelity, keyed by qubit pair in either
            direction.
        cutoff_t1: T1 value scoring zero.
        cutoff_spam: SPAM fidelity scoring zero.
        cutoff_bell: Bell fidelity scoring zero.
    """
    t1_rescaled = (t1_vals - cutoff_t1) / np.max(t1_vals - cutoff_t1)
    spam_rescaled = (trex - cutoff_spam) / np.max(trex - cutoff_spam)

    bell_vals = np.array(list(hellinger_fids.values()))
    bell_rescaled_vals = (bell_vals - cutoff_bell) / np.max(bell_vals - cutoff_bell)
    bell_rescaled = {pair: bell_rescaled_vals[i] for i, pair in enumerate(hellinger_fids)}

    score = 0.0
    for index, qubit in enumerate(path_qubits):
        score += t1_rescaled[qubit]
        score += spam_rescaled[qubit]
        if index < len(path_qubits) - 1:
            pair = (path_qubits[index], path_qubits[index + 1])
            if pair not in bell_rescaled:
                pair = pair[::-1]
            score += bell_rescaled[pair]

    return float(score)


def rank_layouts(
    possible_layouts: list[list[int]],
    shots_trex: np.ndarray,
    qubit_layout: Sequence[int],
) -> tuple[list[list[int]], list[float]]:
    """Rank layouts by descending joint readout fidelity.

    A layout scores ``prod(1 - readout_error)`` over its qubits -- approximately
    the probability that *every* qubit in it is read correctly in a single shot,
    which is the quantity that matters when one flipped bit costs the whole shot.

    Args:
        possible_layouts: candidates from :func:`select_layout` or :func:`find_lines`.
        shots_trex: corrected shots, ``(n_rand, shots_per_twirl, n_qubits)``.
        qubit_layout: physical qubit indices matching the last axis of ``shots_trex``.

    Returns:
        ``(layouts, scores)`` with the layouts sorted best-first and the scores
        aligned to them.
    """
    manual_errors = compute_manual_readout_errors(shots_trex)
    error_map = {int(q): float(manual_errors[i]) for i, q in enumerate(qubit_layout)}

    def _score(layout: list[int]) -> float:
        return float(np.prod([1.0 - error_map[int(q)] for q in layout]))

    scored = sorted(possible_layouts, key=_score, reverse=True)
    return scored, [_score(layout) for layout in scored]


def select_layout(
    shots_trex: np.ndarray,
    coupling_map: Any,
    qubit_layout: Sequence[int],
    required_size: int,
    readout_error_threshold: float = DEFAULT_READOUT_ERROR_THRESHOLD,
    bad_edges: Sequence[tuple[int, int]] | None = None,
    max_threshold: float = DEFAULT_MAX_THRESHOLD,
    relax_factor: float = DEFAULT_RELAX_FACTOR,
) -> tuple[Any, list[list[int]], list[int], float]:
    """Select viable 1-D chains from a device by removing high-error qubits.

    Uses the measured per-qubit readout error
    (:func:`~embasi_qiskit_integration.circuit_run.metrics.compute_manual_readout_errors`)
    as the sole criterion for flagging qubits to avoid.

    ``readout_error_threshold`` is a *starting* threshold. If pruning at it leaves
    no chain of ``required_size`` -- common for longer chains, where pruning
    fragments the lattice -- the threshold is relaxed by ``relax_factor`` and the
    search retried, up to ``max_threshold``. Only when even the fully-unpruned map
    has no chain of the required size are no layouts returned; the caller decides
    whether that is fatal.

    Args:
        shots_trex: corrected shots, ``(n_rand, shots_per_twirl, n_qubits)``.
        coupling_map: full device topology (e.g. ``backend.coupling_map``).
        qubit_layout: physical qubit indices matching the last axis of ``shots_trex``.
        required_size: chain length in qubits (the circuit's width).
        readout_error_threshold: initial ceiling; qubits above it are excluded.
        bad_edges: edges to remove regardless of qubit quality.
        max_threshold: ceiling for the relaxation. ``1.0`` eventually admits every
            qubit, since readout errors are probabilities in ``[0, 1]``.
        relax_factor: multiplicative relaxation step.

    Returns:
        ``(pruned_map, possible_layouts, qubits_to_avoid, threshold_used)``.
    """
    manual_errors = compute_manual_readout_errors(shots_trex)
    layout_arr = np.array([int(q) for q in qubit_layout])
    extra_bad_edges = list(bad_edges or [])

    threshold = readout_error_threshold
    pruned_map = coupling_map
    possible_layouts: list[list[int]] = []
    qubits_to_avoid: list[int] = []

    while True:
        qubits_to_avoid = layout_arr[manual_errors > threshold].tolist()
        pruned_map = get_modified_coupling_map(
            coupling_map, bad_qubits=qubits_to_avoid, bad_edges=extra_bad_edges
        )
        # Skip the expensive DFS when no component is large enough to hold the
        # chain (cheap O(V+E) gate); relax the threshold instead.
        if has_feasible_chain(pruned_map, required_size):
            possible_layouts = find_lines(length=required_size, coupling_map=pruned_map)
        else:
            possible_layouts = []

        if possible_layouts:
            if threshold > readout_error_threshold:
                logger.warning(
                    "No viable %d-qubit chain at readout_error_threshold=%g; "
                    "relaxed to %g to find one.",
                    required_size,
                    readout_error_threshold,
                    threshold,
                )
            break
        if threshold >= max_threshold:
            # Even the fully-unpruned map has no chain of the required size.
            logger.error(
                "No viable %d-qubit chain even at the maximum threshold %g; the "
                "device topology cannot host this chain.",
                required_size,
                threshold,
            )
            break
        threshold = min(threshold * relax_factor, max_threshold)

    return pruned_map, possible_layouts, qubits_to_avoid, threshold


def plot_layout(
    backend: Any,
    qubits_to_avoid: Sequence[int],
    transpiled_circuit: Any = None,
    save_path: str | None = None,
) -> Any:
    """Plot the device gate map, highlighting avoided and selected qubits.

    Args:
        backend: IBM backend (supplies ``num_qubits`` and the topology).
        qubits_to_avoid: qubits flagged as bad; drawn red.
        transpiled_circuit: a transpiled circuit whose final layout marks the
            active qubits green. ``None`` skips that.
        save_path: write the figure here instead of returning it for display.

    Returns:
        The matplotlib figure.
    """
    from qiskit.visualization import plot_gate_map

    colors = ["white"] * backend.num_qubits

    if transpiled_circuit is not None:
        physical = list(
            transpiled_circuit.layout.final_virtual_layout().get_virtual_bits().values()
        )
        for qubit in physical:
            colors[int(qubit)] = "green"

    for qubit in qubits_to_avoid:
        colors[int(qubit)] = "red"

    figure = plot_gate_map(
        backend, figsize=(8, 8), font_size=22, qubit_color=colors, font_color="black"
    )
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")
    return figure
