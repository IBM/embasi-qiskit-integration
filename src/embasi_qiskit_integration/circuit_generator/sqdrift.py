# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SqDRIFT ansatz circuits via ``qiskit-fermions`` (primary ansatz)."""

from __future__ import annotations

import logging
import math
import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from embasi_qiskit_integration.circuit_generator.operator import (
    build_canonical_operator,
    fermions_available,
)
from embasi_qiskit_integration.circuit_generator.relabel import (
    NativeHighsSolverAdapter,
    relabel_available,
)
from embasi_qiskit_integration.contract import EmbeddedHamiltonian

logger = logging.getLogger(__name__)


def sqdrift_available() -> bool:
    """True if ``qiskit-fermions`` is importable in this environment."""
    return fermions_available()


@dataclass
class SqdriftBuildResult:
    """The circuits of one SqDRIFT build, with each circuit's mode permutation.

    ``circuits``, ``permutations`` and ``randomization_indices`` are aligned
    position-for-position. ``permutations[i]`` is the mode permutation applied to
    ``circuits[i]`` -- a ``list[int]`` when ``RelabelModes`` solved one, else
    ``None`` (optimization off, or the MILP was infeasible for that draw).

    ``permutations`` holds a plain ``list[int] | None`` per circuit: nothing here
    is written to disk, so an indexable value is what callers actually want.
    """

    circuits: list = field(default_factory=list)
    permutations: list[list[int] | None] = field(default_factory=list)
    num_modes: int = 0
    optimize_requested: bool = False
    # Absolute (global) randomization index of each circuit. For a full build this
    # is ``[0, 1, ..., num_randomizations - 1]`` per sweep combination; for a
    # seed-chunk it is that chunk's contiguous slice (e.g. ``[100, ..., 149]``).
    randomization_indices: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        """Number of circuits in the result."""
        return len(self.circuits)

    def __iter__(self):
        """Iterate the circuits, so a result can stand in for a circuit list."""
        return iter(self.circuits)

    def __getitem__(self, index):
        """Index/slice the circuits directly."""
        return self.circuits[index]

    @property
    def num_permutations_found(self) -> int:
        """How many randomizations ``RelabelModes`` found a permutation for.

        ``0`` when ``optimize`` was off, and also when every solve was infeasible
        (e.g. a degenerate tiny active space whose sampled terms reduce to
        identity) -- those circuits are kept unpermuted.
        """
        return sum(1 for permutation in self.permutations if permutation is not None)

    @property
    def any_permuted(self) -> bool:
        """True if at least one circuit carries a mode permutation.

        When this is True the sampled counts are in a permuted mode order and
        must be passed through
        :func:`~embasi_qiskit_integration.circuit_run.permutation.unpermute_counts_list`
        before they are pooled or handed to SQD.
        """
        return self.num_permutations_found > 0

    def extend(self, other: SqdriftBuildResult) -> None:
        """Append ``other``'s circuits/permutations/indices onto this result."""
        self.circuits.extend(other.circuits)
        self.permutations.extend(other.permutations)
        self.randomization_indices.extend(other.randomization_indices)
        self.num_modes = self.num_modes or other.num_modes
        self.optimize_requested = self.optimize_requested or other.optimize_requested


def _as_float_list(value: float | Sequence[float]) -> list[float]:
    """Coerce a scalar or sequence of evolution times into a list of floats.

    A bare ``1.0`` is accepted as ``[1.0]`` so existing scalar callers keep
    working; the swept form is a sequence.
    """
    if isinstance(value, (int, float)):
        return [float(value)]
    times = [float(t) for t in value]
    if not times:
        raise ValueError("time must contain at least one evolution time")
    return times


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
    num_groups: int | Sequence[int] = 15,
    num_randomizations: int = 500,
    time: float | Sequence[float] = (1.0, 2.0, 3.0),
    filter_diagonal_terms: bool = True,
    filter_trivial: bool | None = None,
    atol: float = 1e-16,
    seed: int | None = 42,
    measure: bool = True,
    include_initial_state: bool = True,
    optimize: bool = True,
    time_limit: float = 10.0,
    canonical_permutation: bool = False,
    workers: int = 1,
) -> SqdriftBuildResult:
    """Build SqDRIFT sampling circuits for ``ham``.

    Each circuit is a :class:`qiskit.QuantumCircuit` on ``2 * ham.norb`` qubits
    carrying the Hamiltonian time-evolution, preceded by the HF reference unless
    ``include_initial_state=False``.

    Defaults are ``time`` ``[1.0, 2.0, 3.0]``, ``num_groups`` ``15``,
    ``num_randomizations`` 500, ``seed`` 42, ``optimize`` True, so a bare
    ``method="qdrift"`` call produces **1500 circuits** (3 times x 1 group-count x
    500 randomizations). Widen ``num_groups`` to a sequence to sweep it as well --
    ``(10, 15, 20)`` would give 4500. Narrow the axes for a smaller budget instead;
    ``SQDSolver`` does exactly that,
    pinning one ``(time, num_groups)`` pair and its own ``num_randomizations``.

    Args:
        method: ``"exact"`` (one full-evolution circuit per ``time``) or
            ``"qdrift"`` (ensemble over time x num_groups x randomizations).
        num_groups: how many term-groups qDRIFT samples per circuit
            (``method="qdrift"`` only) -- the length of the randomized product,
            so it trades circuit depth against accuracy. Scalar or sequence; a
            sequence is a sweep axis. It is passed positionally to
            ``QDriftTrotterization``,
            whose own parameter is called ``num_terms``. Note this is *not*
            ``FermionOperator.num_groups()`` (how many groups the Hamiltonian
            has, typically ~1000) -- it is how many of them get drawn.
        num_randomizations: randomizations per ``(time, num_groups)`` combination
            (``method="qdrift"``; ``"exact"`` yields one circuit per ``time``).
        time: evolution time(s) t for ``exp(-i t H)``, fed to the ``Evolution``
            gate. Scalar or sequence; a sequence is a sweep axis, combined with
            ``num_groups`` as a cartesian product.
        filter_diagonal_terms: drop occupation-diagonal terms that do not affect
            sampled bitstrings. Applied during operator construction, before
            grouping (see :mod:`.operator`).
        filter_trivial: forwarded to ``QDriftTrotterization`` (``method="qdrift"``
            only). Rejects a *sampled* term that cannot change the occupation
            (acting only within the occupied or only within the unoccupied set),
            so it does not waste one of the ``num_groups`` slots. This is a
            distinct mechanism from ``filter_diagonal_terms``: that one prunes the
            operator up front, this one filters draws during sampling.

            The pass can only filter when it can see the occupation, i.e. when the
            reference state is inside the circuit. ``None`` (default) therefore
            tracks ``include_initial_state``; forcing ``True`` on a bare circuit
            has no effect and makes qiskit emit a ``UserWarning``.
        atol: tolerance for simplifying the normal-ordered operator.
        seed: base RNG seed (default 42).
            Randomization ``i`` of each ``(time, num_groups)`` combination uses
            ``seed + i`` for the qDRIFT sampler, the transpiler, and the relabel
            solve, so each draw is independently reproducible -- which is also
            what lets the batch be sharded across processes. The index restarts
            per combination, so combinations sharing a randomization index share a
            draw, which isolates the effect of ``time``/``num_groups`` from
            sampling noise. ``None`` leaves
            everything unseeded (non-reproducible), and disables sharding since
            the draws are then not addressable by index.
        measure: append ``measure_all()`` to each circuit.
        include_initial_state: prepare the HF reference inside the circuit
            (default). ``False`` emits the bare evolution, leaving the reference
            state to the run stage; either way the choice is recorded in
            ``metadata["initial_state_included"]``.
        optimize: append ``RelabelModes`` to the pass chain (default True),
            reordering modes to minimize the excitation span and so shorten the
            circuit. The permutation actually applied is reported per circuit in the
            result and mirrored into ``metadata["permutation"]``.

            **The caller must undo it on the sampled counts** -- a relabeled
            circuit measures in the permuted mode order, so raw counts put
            occupations on the wrong orbitals. Use
            :func:`~embasi_qiskit_integration.circuit_run.permutation.unpermute_counts_list`
            before pooling. Requires the optional ``pyomo``/``highspy`` extra;
            without it this raises rather than silently producing unpermuted
            circuits.
        time_limit: wall-clock limit (seconds) for each ``RelabelModes`` solve.
            A draw whose solve times out or is infeasible is kept unpermuted.
        canonical_permutation: derive each draw's permutation from a fixed candidate
            set instead of using the one the MILP solver returned
            (:func:`canonicalize_permutation`).

            ``False`` (default) applies the solver's own ordering in a single pass
            chain. ``True`` costs a second pass-manager run per randomization and
            makes a seeded build **reproducible**, which the default is not: HiGHS is
            not a pure function of this degenerate model -- solving one model
            repeatedly in a single process was observed to return one optimum twice
            and then a different one, and the alternatives are not interchangeable
            (observed refined scores ``(6, 126)`` vs ``(7, 146)``; depths 346 vs
            274). Since the permutation must be undone on the counts, that
            instability propagates into the pooled distribution.

            Set it when you need byte-identical circuits across runs; leave it off
            for the standard single-solve behaviour.
        workers: processes to shard the randomizations across. ``1`` (default)
            builds in-process. Higher values split each combination's
            randomizations into contiguous seed-chunks, one per worker, and build
            them in a :class:`~concurrent.futures.ProcessPoolExecutor`; the
            process-local operator cache means each worker pays the operator
            build once. Output order is normalised back to the sequential order,
            so the result does not depend on ``workers``. ``0`` means one worker
            per CPU. Ignored for ``method="exact"`` (one circuit per time --
            nothing to shard) and when ``seed is None``.

            Workers are spawned, so a script calling this with ``workers > 1``
            must guard its entry point with ``if __name__ == "__main__":`` -- the
            standard :mod:`multiprocessing` requirement. Without it the workers
            fail with ``BrokenProcessPool``.

    Returns:
        A :class:`SqdriftBuildResult`. It iterates and indexes as the circuit
        list, so existing list-style callers keep working; ``permutations``
        carries each circuit's mode permutation (``None`` when unpermuted). Order
        is the flattened sweep: for each ``time``, for each ``num_groups``, each
        randomization in turn.
    """
    if method not in ("exact", "qdrift"):
        raise ValueError(f"unknown method {method!r}; use 'exact' or 'qdrift'")

    if optimize and not relabel_available():
        raise ImportError(
            "optimize=True requires the optional 'pyomo' and 'highspy' packages "
            "(the relabel solver stack). Install them with: pip install -e '.[relabel]', "
            "or pass optimize=False to build without mode relabeling."
        )

    # The qDRIFT pass filters a draw by comparing it against the occupation, which
    # it can only read from an InitializeModes gate. On a bare circuit the option
    # is inert and qiskit warns, so default it to wherever the reference state is.
    if filter_trivial is None:
        filter_trivial = include_initial_state

    times = _as_float_list(time)
    group_counts = (
        [int(num_groups)] if isinstance(num_groups, int) else [int(n) for n in num_groups]
    )
    if not group_counts:
        raise ValueError("num_groups must contain at least one group count")

    spec = _BuildSpec(
        ham=ham,
        method=method,
        filter_diagonal_terms=filter_diagonal_terms,
        filter_trivial=bool(filter_trivial),
        atol=atol,
        seed=seed,
        measure=measure,
        include_initial_state=include_initial_state,
        optimize=optimize,
        time_limit=time_limit,
        canonical_permutation=canonical_permutation,
    )

    if method == "exact":
        logger.info(
            "SqDRIFT exact: building %d circuit(s), one per evolution time (optimize=%s).",
            len(times),
            optimize,
        )
        return _build_chunk(spec, [(t, group_counts[0]) for t in times], range(1))

    tasks = _plan_tasks(times, group_counts, num_randomizations, workers, seed)

    logger.info(
        "SqDRIFT qdrift sweep: %d time(s) x %d group-count(s) x %d randomization(s) "
        "= %d circuit(s) in %d task(s) (workers requested=%s, effective=%d).",
        len(times),
        len(group_counts),
        num_randomizations,
        len(times) * len(group_counts) * num_randomizations,
        len(tasks),
        workers,
        min(_resolve_workers(workers), len(tasks)) if len(tasks) > 1 else 1,
    )

    if len(tasks) == 1:
        combos, rand_range = tasks[0]
        return _build_chunk(spec, combos, rand_range)

    return _build_parallel(spec, tasks, workers)


@dataclass(frozen=True)
class _BuildSpec:
    """Everything a build needs that does not vary across the sweep.

    Held as one frozen, picklable value so a seed-chunk task can be shipped to a
    worker process as ``(spec, combos, rand_range)`` without re-deriving anything.
    ``EmbeddedHamiltonian`` carries plain arrays, so it pickles fine; the built
    circuits never cross back untransformed -- only the finished result does.
    """

    ham: EmbeddedHamiltonian
    method: str
    filter_diagonal_terms: bool
    filter_trivial: bool
    atol: float
    seed: int | None
    measure: bool
    include_initial_state: bool
    optimize: bool
    time_limit: float
    canonical_permutation: bool


def _plan_tasks(
    times: list[float],
    group_counts: list[int],
    num_randomizations: int,
    workers: int,
    seed: int | None,
) -> list[tuple[list[tuple[float, int]], range]]:
    """Split the sweep into ``(combos, randomization-slice)`` tasks.

    With one worker (or an unseeded build, whose draws are not addressable by
    index) this is a single task covering the whole sweep. Otherwise each
    ``(time, num_groups)`` combination is split into contiguous randomization
    slices of ``ceil(num_randomizations / workers)`` -- one chunk per worker per
    combination. Contiguous slices keep
    every draw's absolute index intact, so the sharded output is the sequential
    output reordered, never a different set of circuits.
    """
    if num_randomizations <= 0:
        raise ValueError(f"num_randomizations must be positive, got {num_randomizations}")

    combos = [(t, n) for t in times for n in group_counts]
    effective_workers = _resolve_workers(workers)
    if effective_workers <= 1 or seed is None:
        return [(combos, range(num_randomizations))]

    chunk = max(1, math.ceil(num_randomizations / effective_workers))
    return [
        ([combo], range(start, min(start + chunk, num_randomizations)))
        for combo in combos
        for start in range(0, num_randomizations, chunk)
    ]


def _resolve_workers(workers: int | None) -> int:
    """Normalise the ``workers`` request; ``0`` (or negative) means "one per CPU".

    ``None`` is treated as the sequential default rather than as "auto", so an
    unset value never silently saturates the machine.
    """
    if workers is None:
        return 1
    requested = int(workers)
    if requested <= 0:
        return os.cpu_count() or 1
    return requested


def _build_parallel(
    spec: _BuildSpec,
    tasks: list[tuple[list[tuple[float, int]], range]],
    workers: int,
) -> SqdriftBuildResult:
    """Build ``tasks`` in a process pool and reassemble them in sequential order.

    Never spawns more processes than there are tasks: a ``ProcessPoolExecutor``
    eagerly starts ``max_workers`` processes, so an oversized pool (e.g.
    ``workers=128`` for 2 randomizations) would fork dozens of idle workers.

    A failed chunk re-raises rather than being logged and dropped: a silently
    short ensemble would flow into SQD as a quietly wrong energy.
    """
    effective_workers = min(_resolve_workers(workers), len(tasks))
    results: dict[int, SqdriftBuildResult] = {}

    with ProcessPoolExecutor(
        max_workers=effective_workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(_build_chunk, spec, combos, rand_range): position
            for position, (combos, rand_range) in enumerate(tasks)
        }
        for future in as_completed(futures):
            position = futures[future]
            try:
                results[position] = future.result()
            except Exception as exc:
                combos, rand_range = tasks[position]
                raise RuntimeError(
                    f"SqDRIFT chunk {combos} randomizations "
                    f"[{rand_range.start}, {rand_range.stop}) failed: {exc}"
                ) from exc

    # Chunks complete out of order; reassemble by their planned position so the
    # circuit order matches the sequential build exactly.
    merged = SqdriftBuildResult(optimize_requested=spec.optimize)
    for position in range(len(tasks)):
        merged.extend(results[position])

    # Re-report in the parent: the per-chunk records were emitted inside spawned
    # workers, whose logging handlers are not the parent's, so this is the only
    # relabel summary a caller is guaranteed to see on the sharded path.
    if spec.optimize and merged.circuits:
        found = merged.num_permutations_found
        total = len(merged.circuits)
        if found == 0:
            logger.warning(
                "SqDRIFT optimize: 0/%d circuits across %d chunk(s) got a mode "
                "permutation -- RelabelModes found no feasible solution, so the "
                "circuits are kept unpermuted and unshortened. Check time_limit=%s, "
                "or pass optimize=False to skip relabeling deliberately.",
                total,
                len(tasks),
                spec.time_limit,
            )
        else:
            logger.info(
                "SqDRIFT optimize: %d/%d circuits across %d chunk(s) got a mode permutation.",
                found,
                total,
                len(tasks),
            )
    return merged


def _build_chunk(
    spec: _BuildSpec,
    combos: list[tuple[float, int]],
    rand_range: range,
) -> SqdriftBuildResult:
    """Build the randomizations ``rand_range`` of every combination in ``combos``.

    The unit of work for both the sequential and the sharded path, and the
    function dispatched to a worker process -- module-level (not a closure) so it
    pickles. It builds the canonicalized operator through the process-local cache
    in :mod:`.operator`, so a worker handling several chunks of one molecule pays
    that cost once.
    """
    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import Evolution, InitializeModes
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.passes import QDriftTrotterization, RelabelModes
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    ham = spec.ham
    normal, num_modes = build_canonical_operator(
        ham, atol=spec.atol, filter_diagonal_terms=spec.filter_diagonal_terms
    )
    occ = _hf_occupation(ham.norb, ham.nelec)

    def _fresh_circuit(evolution_time: float) -> Any:
        """Build a fresh evolution circuit at ``evolution_time``.

        A new circuit (and its own metadata dict) per pass-manager run: sharing
        one circuit across randomizations lets qiskit passes leak metadata between
        results, since a pass that returns its input DAG unchanged still carries
        the previous randomization's entries (aliased by object id). With
        ``RelabelModes`` in the chain that is not cosmetic -- an infeasible solve
        would inherit the previous draw's ``permutation``, and the counts would
        then be un-permuted with a map that was never applied to that circuit.
        """
        circ = FermionicCircuit(num_modes)
        if spec.include_initial_state:
            circ.append(InitializeModes(occ), circ.modes)
        circ.append(Evolution(num_modes, normal, evolution_time), circ.modes)
        return circ

    def _draw_seed(randomization: int) -> int | None:
        """Seed for randomization ``i``: ``seed + i`` (None stays unseeded).

        The index is the randomization number *within* a ``(time, num_groups)``
        combination, and restarts at 0 for each combination, so every combination
        is an independent pass over ``range(num_randomizations)``. Combinations that
        share a
        randomization index also share a qDRIFT draw, which isolates the effect of
        ``time``/``num_groups`` from sampling noise.
        """
        return None if spec.seed is None else spec.seed + randomization

    def _pass_manager(draw_seed: int | None) -> Any:
        """A preset JW pass manager, seeded when ``draw_seed`` is given."""
        if draw_seed is None:
            return generate_preset_jw_pass_manager()
        return generate_preset_jw_pass_manager(seed_transpiler=draw_seed)

    def _sampling_passes(n_groups: int, draw_seed: int | None) -> list:
        """The qDRIFT sampling stage (no relabeling).

        ``filter_trivial`` filters the pass's own *draws*; the occupation-diagonal
        terms of the operator were already pruned during its construction, which
        is a separate mechanism (see :mod:`.operator` -- doing that pruning here
        instead would break the canonical group order the seeded draw depends on).
        """
        if spec.method != "qdrift":
            return []
        return [QDriftTrotterization(n_groups, filter_trivial=spec.filter_trivial, rng=draw_seed)]

    def _solve_permutation(n_groups: int, draw_seed: int | None, evolution_time: float):
        """Find this draw's canonical mode permutation, or ``None`` if unrelabeled.

        Runs a *probe* pass manager whose optimization stage is qDRIFT sampling ->
        excitation collection -> ``RelabelModes``. The collector snapshots the
        excitations from the DAG at exactly the point ``RelabelModes`` reads them
        (they only exist there -- synthesis later lowers the ``Evolution`` gates
        away). The returned circuit is then rebuilt by relabeling with the
        canonicalized permutation, which is what makes an ``optimize=True`` build
        reproducible despite HiGHS being unstable on this degenerate MILP.

        The solve is kept even though :func:`canonicalize_permutation` does not use
        its ordering, because it answers a question nothing else does: *whether*
        this draw can be relabeled at all. ``RelabelModes`` returns no permutation
        when the MILP is infeasible or times out -- a degenerate active space whose
        sampled terms reduce to identity -- and those circuits must stay unpermuted.
        Skipping the solve would relabel them anyway.
        """
        collector = _make_excitation_collector()
        probe = _pass_manager(draw_seed)
        probe.optimization = FermionicPassManager(
            [
                *_sampling_passes(n_groups, draw_seed),
                collector,
                RelabelModes(solver=NativeHighsSolverAdapter(time_limit=spec.time_limit)),
            ]
        )
        solved = _extract_permutation(probe.run(_fresh_circuit(evolution_time)), num_modes)
        if solved is None:
            return None
        return canonicalize_permutation(solved, collector.excitations)

    def _build_one(n_groups: int, draw_seed: int | None, evolution_time: float) -> Any:
        """Synthesise one randomization's circuit.

        Two shapes, selected by ``canonical_permutation``:

        - **Default (False).** A single pass manager whose optimization stage is
          ``[QDriftTrotterization, RelabelModes(solver=...)]``, taking whatever
          permutation the solver returns. One solve, one synthesis.
        - **True.** Solve on a probe pass, canonicalize the result, then synthesise
          by relabeling with that fixed permutation. Two pass-manager runs, and the
          circuit becomes a pure function of the seed -- see
          :func:`canonicalize_permutation` for why the solver's own pick is not.
        """
        pm = _pass_manager(draw_seed)
        passes = _sampling_passes(n_groups, draw_seed)

        if not spec.optimize:
            if passes:
                pm.optimization = FermionicPassManager(passes)
            return pm.run(_fresh_circuit(evolution_time))

        if not spec.canonical_permutation:
            passes.append(RelabelModes(solver=NativeHighsSolverAdapter(time_limit=spec.time_limit)))
            pm.optimization = FermionicPassManager(passes)
            return pm.run(_fresh_circuit(evolution_time))

        permutation = _solve_permutation(n_groups, draw_seed, evolution_time)
        if permutation is not None:
            # No solver runs in this pass, so the synthesised circuit is a pure
            # function of the seed and the permutation just pinned down.
            passes.append(RelabelModes(permutation=permutation))
        if passes:
            pm.optimization = FermionicPassManager(passes)
        return pm.run(_fresh_circuit(evolution_time))

    result = SqdriftBuildResult(num_modes=num_modes, optimize_requested=spec.optimize)
    for evolution_time, n_groups in combos:
        for randomization in rand_range:
            draw_seed = _draw_seed(randomization)
            circuit = _build_one(n_groups, draw_seed, evolution_time)

            result.circuits.append(circuit)
            result.permutations.append(_extract_permutation(circuit, num_modes))
            result.randomization_indices.append(randomization)

    if spec.measure:
        result.circuits = [qc.measure_all(inplace=False) for qc in result.circuits]

    # Record how each circuit is to be interpreted
    for circuit, permutation in zip(result.circuits, result.permutations, strict=True):
        circuit.metadata = {
            **(circuit.metadata or {}),
            "initial_state_included": spec.include_initial_state,
            "permutation": permutation,
        }

    _log_relabel_outcome(spec, result, rand_range)
    return result


def _log_relabel_outcome(spec: _BuildSpec, result: SqdriftBuildResult, rand_range: range) -> None:
    """Report how many draws in this chunk actually got a mode permutation.

    ``RelabelModes`` degrades silently: when the excitation-span MILP is infeasible
    or times out it returns the circuit unchanged, so ``optimize=True`` producing
    *zero* permutations looks exactly like relabeling having worked -- shorter
    circuits simply never materialise. That case is warned about rather than logged
    at info level, since it usually means the ``time_limit`` is too tight or the
    active space is degenerate, and it is the only signal the caller gets.

    With sharding each chunk sees only its own randomization slice, so the counts
    are reported over this chunk's span rather than the whole sweep (a
    ``workers=1`` build is one chunk covering everything).

    Note this runs *inside* the worker on the sharded path, and a spawned process
    does not inherit the parent's logging handlers, so these records are only
    visible there if the host configures logging on import (or the build is
    sequential). :func:`_build_parallel` therefore re-reports the merged totals in
    the parent, which is the line a caller can always rely on.
    """
    if not spec.optimize or not result.circuits:
        return

    found = result.num_permutations_found
    total = len(result.circuits)
    span = f"[{rand_range.start}, {rand_range.stop})" if rand_range else "[]"
    if found == 0:
        logger.warning(
            "SqDRIFT optimize: 0/%d circuits (randomizations %s) got a mode "
            "permutation -- RelabelModes found no feasible solution, so the circuits "
            "are kept unpermuted and unshortened. Check time_limit=%s, or pass "
            "optimize=False to skip relabeling deliberately.",
            total,
            span,
            spec.time_limit,
        )
    else:
        logger.info(
            "SqDRIFT optimize: %d/%d circuits (randomizations %s) got a mode permutation.",
            found,
            total,
            span,
        )


def _make_excitation_collector() -> Any:
    """Build a no-op transpiler pass that snapshots the DAG's excitations.

    Defined as a factory rather than a module-level class so its base is resolved
    lazily -- it is taken from the same alias ``RelabelModes`` itself subclasses,
    which keeps the collector in step if upstream changes that base.
    """
    from qiskit_fermions.transpiler.passes.optimization.relabel_modes import (
        FermionicDAGCircuitPass,
    )

    class _Collector(FermionicDAGCircuitPass):  # type: ignore[misc, valid-type]
        """Records the excitations of the DAG it sees, returning it unchanged.

        Inserted immediately before ``RelabelModes`` so it observes exactly the
        excitation set the MILP is built from -- the only point in the pipeline
        where the ``Evolution`` gates still carry their operator.
        """

        def __init__(self) -> None:
            super().__init__()
            self.excitations: list[tuple[int, ...]] = []

        def run(self, dag: Any) -> Any:
            """Snapshot the excitations and pass the DAG through untouched."""
            self.excitations = gather_excitations(dag)
            return dag

    return _Collector()


def _excitation_spans(permutation: Sequence[int], excitations: Sequence[tuple[int, ...]]) -> tuple:
    """Score ``permutation`` the way the relabel MILP does: ``(max_span, total_span)``.

    ``build_excitation_span_minimization_model`` minimizes
    ``max_span + mix_delta * average_span``; with the excitation count fixed, that
    is order-equivalent to comparing ``(max_span, total_span)`` lexicographically
    scaled the same way, which is all a tie-break needs.
    """
    max_span = 0
    total_span = 0
    for excitation in excitations:
        positions = [permutation[mode] for mode in excitation]
        span = max(positions) - min(positions)
        max_span = max(max_span, span)
        total_span += span
    return (max_span, total_span)


def gather_excitations(dag: Any) -> list[tuple[int, ...]]:
    """Collect the fermionic excitations of a DAG's ``Evolution`` gates.

    Mirrors ``RelabelModes.find_permutation``'s own gathering (each gate's modes
    sliced by its operator's group boundaries), then applies the model's documented
    pre-processing: indices occurring twice within a tuple cancel, and tuples that
    reduce to length 0 or 1 impose no distance constraint and are dropped.

    Takes a **DAG mid-pipeline**, not a finished circuit: the ``Evolution`` gates
    carry the operator only until the synthesis stage lowers them to qubit gates,
    so a fully-transpiled circuit yields nothing to score. This is why it is used
    from inside a pass (see :func:`_make_excitation_collector`) rather than applied to
    the returned circuit.
    """
    from qiskit_fermions.circuit.library import Evolution
    from qiskit_fermions.operators import FermionOperator

    excitations: list[tuple[int, ...]] = []
    for node in dag.op_nodes():
        operation = node.op
        if not isinstance(operation, Evolution):
            continue
        hamil = getattr(operation, "operator", None)
        if not isinstance(hamil, FermionOperator):
            continue
        modes = hamil.get_modes()
        boundaries = list(hamil.get_boundaries())
        for start, stop in zip(boundaries, boundaries[1:], strict=False):
            term = [int(mode) for mode in modes[start:stop]]
            # Cancel indices appearing twice, as the model's pre-processing does.
            reduced = tuple(mode for mode in term if term.count(mode) == 1)
            if len(reduced) in (2, 4):
                excitations.append(reduced)
    return excitations


def _refine_permutation(
    start: Sequence[int], excitations: Sequence[tuple[int, ...]]
) -> tuple[tuple[int, int], list[int]]:
    """Greedily improve ``start`` under the span objective; return ``(score, perm)``.

    Repeatedly applies the single best improving-or-equal transposition, preferring
    the lexicographically smaller ordering whenever the objective ties, until no
    swap helps. Deterministic given ``(start, excitations)``.
    """
    current = [int(index) for index in start]
    best_score = _excitation_spans(current, excitations)
    num_modes = len(current)

    improved = True
    while improved:
        improved = False
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                score = _excitation_spans(candidate, excitations)
                if score < best_score or (score == best_score and candidate < current):
                    current, best_score = candidate, score
                    improved = True
    return best_score, current


def canonicalize_permutation(
    permutation: Sequence[int], excitations: Sequence[tuple[int, ...]]
) -> list[int]:
    """Return a reproducible mode permutation for a draw, ignoring solver luck.

    The excitation-span MILP is highly degenerate *and* HiGHS is not a pure
    function of the model: solving one model repeatedly in a single process was
    observed to return one optimum twice and then a different one, and pinning the
    solver to a single thread narrows but does not remove the variation. Since the
    permutation must be undone on the sampled counts, an unstable one makes a
    seeded run irreproducible -- and the alternatives are not interchangeable, with
    solver picks for one draw scoring ``(12, 224)`` and ``(14, 242)`` and yielding
    circuit depths of 346 vs 274.

    So the solver's pick is discarded and the permutation is derived from scratch:
    a *fixed* set of starting orderings -- the identity (the blocked spin order the
    operator is built in) and the interleaved order ``(u0, d0, u1, d1, ...)``, the
    usual hand-chosen relabeling for shortening these circuits -- is greedily
    refined under the model's own objective, and the best result wins, with the
    lexicographically smallest ordering breaking any tie.

    Keeping the solver's permutation as a third candidate was tried and rejected:
    the refined optimum is frequently reached from several starts with *different*
    orderings at the same score, so including a start that varies run-to-run makes
    the tie-break vary too. Excluding it costs nothing measurable here -- refining
    the fixed starts reached ``(6, 118)`` on the draw examined while the solver's
    own picks refined only to ``(6, 126)`` and ``(7, 146)`` -- and it makes the
    result a function of ``(num_modes, excitations)`` alone, hence identical across
    processes, thread counts and worker pools.

    ``permutation`` is still required: it tells us the pass *did* relabel (and its
    width), and with no excitations to score against it is returned unchanged.
    """
    solved = [int(index) for index in permutation]
    if not excitations:
        return solved

    num_modes = len(solved)
    norb = num_modes // 2

    identity = list(range(num_modes))
    interleaved = [0] * num_modes
    for orbital in range(norb):
        interleaved[orbital] = 2 * orbital
        interleaved[norb + orbital] = 2 * orbital + 1

    scored = [_refine_permutation(candidate, excitations) for candidate in (identity, interleaved)]
    # Lowest objective wins; the refined ordering breaks exact ties, so the result
    # does not depend on the order the candidates were tried in.
    return min(scored, key=lambda item: (item[0], item[1]))[1]


def _extract_permutation(circuit: Any, num_modes: int) -> list[int] | None:
    """Read the applied mode permutation off a transpiled circuit, or ``None``.

    ``RelabelModes`` records what it did in ``metadata["permutation"]`` and only
    when it actually relabeled, so this reads defensively as its documentation
    instructs. Its raw pyomo solver output is dropped here: it is stored under
    ``metadata["permutation.opt_result"]``, is not JSON/QPY-serializable, would
    have to be pickled back from a worker process, and carries nothing we keep.

    A permutation that is not a rearrangement of ``range(num_modes)`` is rejected
    rather than propagated -- applying a malformed map to the counts would scramble
    occupations silently.
    """
    metadata = getattr(circuit, "metadata", None) or {}
    metadata.pop("permutation.opt_result", None)

    permutation = metadata.get("permutation")
    if permutation is None:
        return None

    permutation = [int(index) for index in permutation]
    if sorted(permutation) != list(range(num_modes)):
        raise RuntimeError(
            f"RelabelModes returned {permutation!r}, which is not a permutation of "
            f"range({num_modes}); refusing to build circuits whose bitstrings could "
            "not be mapped back to the original mode order"
        )
    return permutation


__all__ = [
    "SqdriftBuildResult",
    "build_sqdrift_circuits",
    "relabel_available",
    "sqdrift_available",
]
