# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SqDRIFT ansatz circuits via ``qiskit-fermions`` (primary ansatz)."""

from __future__ import annotations

import hashlib
import logging
import math
import multiprocessing
import os
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from embasi_qiskit_integration.circuit_generator.operator import (
    build_canonical_operator,
    fermions_available,
)
from embasi_qiskit_integration.circuit_generator.relabel import (
    RELABEL_SOLVERS,
    NativeHighsSolverAdapter,
    cpsat_available,
    make_caching_relabel_modes,
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
    # Draws rebuilt with a replacement seed because their build crashed (see
    # :func:`_retry_failed`): one ``{"time", "num_groups", "randomization",
    # "seed", "replacement_seed", "attempts"}`` record each.
    reseeded: list[dict] = field(default_factory=list)

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
        self.reseeded.extend(other.reseeded)
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
    method: str = "qdrift",
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
    time_limit: float = 60.0,
    relabel_solver: str = "highs",
    cpsat_workers: int = 8,
    cpsat_deterministic: bool = False,
    per_time_randomization: bool = False,
    relabel_cache_dir: str | Path | None = None,
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
            ``seed + i`` for the qDRIFT sampler and the transpiler (see
            ``per_time_randomization`` for the alternative), so each draw is
            independently reproducible -- which is also what lets the batch be
            sharded across processes. The index restarts per combination, so
            combinations sharing a randomization index share a draw, which
            isolates the effect of ``time``/``num_groups`` from sampling noise.
            ``None`` leaves everything unseeded (non-reproducible), and disables
            sharding since the draws are then not addressable by index.
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
            before pooling. Requires the optional ``relabel`` extra; without it
            this raises rather than silently producing unpermuted circuits.

            The relabel model depends only on the excitations a draw sampled, not
            on ``time`` (which only scales the angles), so its permutation is
            cached by that model's content: a draw shared across times is solved
            once and gets the same permutation at every time.
        time_limit: wall-clock limit (seconds) per relabel solve, for either
            ``relabel_solver`` (``0`` = no limit; default 60). A draw whose solve
            finds no feasible permutation in time is kept unpermuted. It is part of
            the cache key, since it can change the answer.
        relabel_solver: ``"highs"`` (default; the MILP through pyomo + HiGHS) or
            ``"cpsat"`` (the same default objective solved with OR-Tools CP-SAT).
            CP-SAT proves the optimum in under a second at ~16 modes; above that
            neither proves it within a minute, and CP-SAT's parallel search
            (``cpsat_workers``) gives far shallower circuits than HiGHS in the same
            time limit. On a single thread it is no better than HiGHS.
        cpsat_workers: CP-SAT search threads per solve (default 8). They multiply
            with ``workers``; CP-SAT's advantage comes from this parallel search, so
            lower ``workers`` (to about cores / ``cpsat_workers``) rather than this.
        cpsat_deterministic: make ``relabel_solver="cpsat"`` reproducible run to run
            (interleaved search). With several ``cpsat_workers`` CP-SAT's workers
            race, so a model with several equally good orderings returns a different
            one each run; this fixes that at the cost of a weaker search (10-20%
            worse objective at 32 modes, 8x longer to prove the optimum at 16 modes).
            Inert with ``"highs"``. Default False.
        per_time_randomization: give each ``time`` its own qDRIFT draws (default
            False). Off, randomization ``i`` uses ``seed + i`` at every time, so the
            circuits of one randomization share their groups (and permutation) and
            differ only in angle. On, each draw is seeded from a hash of
            ``(seed, i, time)``, so every time is an independent sample (more
            independent circuits, no cross-time cache hits); this changes the seeds
            even for a single-time run.
        relabel_cache_dir: directory of an on-disk permutation cache shared by the
            worker processes, so each distinct model is solved once across the pool
            and across repeated builds. ``None`` (default) uses a temporary
            directory for a sharded build (removed afterwards) and the in-process
            cache only for a sequential one. Safe to delete.
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

            With ``optimize=True``, a worker that dies mid-build (e.g. HiGHS
            segfaulting in presolve on one relabel model) does not fail the build:
            the draws that crash on their own are redrawn with replacement seeds
            until they build, and listed in :attr:`SqdriftBuildResult.reseeded`.
            A sequential build (``workers=1``) runs in-process, where a native crash
            cannot be caught, so it has no such retry.

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
    if relabel_solver not in RELABEL_SOLVERS:
        raise ValueError(f"unknown relabel_solver {relabel_solver!r}; use 'highs' or 'cpsat'")
    if optimize and relabel_solver == "cpsat" and not cpsat_available():
        raise ImportError(
            "relabel_solver='cpsat' needs OR-Tools (ortools). Install it with: "
            "pip install -e '.[relabel]'"
        )
    if int(cpsat_workers) < 1:
        raise ValueError(f"cpsat_workers must be >= 1, got {cpsat_workers}")

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
        time_limit=float(time_limit),
        relabel_solver=relabel_solver,
        cpsat_workers=int(cpsat_workers),
        cpsat_deterministic=bool(cpsat_deterministic),
        per_time_randomization=bool(per_time_randomization),
        relabel_cache_dir=None if relabel_cache_dir is None else str(relabel_cache_dir),
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

    if spec.optimize and spec.relabel_cache_dir is None:
        # Workers solving the same draw (one per time) share their permutations
        # through this directory, so each distinct model is solved once.
        with tempfile.TemporaryDirectory(prefix="sqdrift-relabel-") as cache_dir:
            return _build_parallel(replace(spec, relabel_cache_dir=cache_dir), tasks, workers)
    return _build_parallel(spec, tasks, workers)


def _randomization_seed(seed: int, index: int, time: float, per_time: bool) -> int:
    """Seed of randomization ``index`` at evolution time ``time``.

    ``seed + index`` when ``per_time`` is off, so every time shares the draw (the
    default). When on, a stable hash of ``(seed, index, time)`` keyed on the time
    *value* (not its list position), reduced into ``[0, 2**63)``.
    """
    if not per_time:
        return seed + index
    digest = hashlib.blake2b(f"{seed}:{index}:{float(time).hex()}".encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") % 2**63


def _distinct_draws(spec: _BuildSpec, tasks: list[tuple[list[tuple[float, int]], range]]) -> int:
    """Number of distinct qDRIFT draws in ``tasks``, i.e. of distinct relabel solves.

    A draw is fixed by its seed (:func:`_randomization_seed`) and its group count, so
    without ``per_time_randomization`` the times share each randomization's draw, and
    with it every time is its own draw.
    """
    if spec.seed is None:
        return sum(len(combos) * len(rand_range) for combos, rand_range in tasks)
    return len(
        {
            (n_groups, _randomization_seed(spec.seed, i, t, spec.per_time_randomization))
            for combos, rand_range in tasks
            for t, n_groups in combos
            for i in rand_range
        }
    )


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
    relabel_solver: str = "highs"
    cpsat_workers: int = 8
    cpsat_deterministic: bool = False
    per_time_randomization: bool = False
    relabel_cache_dir: str | None = None


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
    build_fn=None,
) -> SqdriftBuildResult:
    """Build ``tasks`` in a process pool and reassemble them in sequential order.

    Never spawns more processes than there are tasks: a ``ProcessPoolExecutor``
    eagerly starts ``max_workers`` processes, so an oversized pool (e.g.
    ``workers=128`` for 2 randomizations) would fork dozens of idle workers.

    A failed chunk is never dropped: a quietly short ensemble would flow into SQD as
    a quietly wrong energy. With ``optimize=True`` a failure is retried
    (:func:`_retry_failed`): the draws that fail on their own are rebuilt with a
    replacement seed until they build. Without relabeling any failure raises.

    ``build_fn`` stands in for :func:`_build_chunk` (tests inject a worker that
    crashes on chosen draws); it must be a module-level function so it pickles.
    """
    build_fn = build_fn or _build_chunk
    effective_workers = min(_resolve_workers(workers), len(tasks))
    if spec.optimize:
        # With relabeling the solves dominate and each distinct draw is solved once
        # (the permutation cache and its lock): a worker beyond the draw count would
        # only wait on another's solve.
        effective_workers = min(effective_workers, max(1, _distinct_draws(spec, tasks)))
    results: dict[int, SqdriftBuildResult] = {}
    failed: dict[int, BaseException] = {}

    with ProcessPoolExecutor(
        max_workers=effective_workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(build_fn, spec, combos, rand_range): position
            for position, (combos, rand_range) in enumerate(tasks)
        }
        for future in as_completed(futures):
            position = futures[future]
            try:
                results[position] = future.result()
            except Exception as exc:
                # One dead worker breaks the pool, failing every pending chunk with
                # BrokenProcessPool -- not just the chunk that crashed.
                failed[position] = exc

    if failed:
        if not spec.optimize:
            position = min(failed)
            combos, rand_range = tasks[position]
            raise RuntimeError(
                f"SqDRIFT chunk {combos} randomizations "
                f"[{rand_range.start}, {rand_range.stop}) failed: {failed[position]}"
            ) from failed[position]
        results.update(_retry_failed(spec, tasks, failed, effective_workers, build_fn))

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


# Replacement seeds tried per draw before a crashing draw stops the build.
_MAX_RESEED_ATTEMPTS = 10


def _run_isolated(jobs: dict, workers: int) -> dict:
    """Run each ``{key: (fn, args)}`` job in a fresh single-use process, ``workers`` at a time.

    Returns ``{key: (True, result)}`` or ``{key: (False, exception)}``. A job whose
    process dies takes down only its own one-worker pool, so the others still finish.
    """
    context = multiprocessing.get_context("spawn")

    def run_one(fn, args):
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
            return executor.submit(fn, *args).result()

    outcome: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as threads:
        futures = {threads.submit(run_one, fn, args): key for key, (fn, args) in jobs.items()}
        for future in as_completed(futures):
            try:
                outcome[futures[future]] = (True, future.result())
            except Exception as exc:
                outcome[futures[future]] = (False, exc)
    return outcome


def _retry_failed(
    spec: _BuildSpec,
    tasks: list[tuple[list[tuple[float, int]], range]],
    failed: dict[int, BaseException],
    workers: int,
    build_fn,
) -> dict[int, SqdriftBuildResult]:
    """Rebuild failed chunks; give each draw that fails on its own a new seed until it builds.

    First each failed chunk is rebuilt alone in a fresh process: chunks that only
    failed because another worker broke the pool come back unchanged. A chunk that
    fails again is rebuilt one draw per process, which isolates the draws that fail
    on their own. Each of those is redrawn with a replacement seed until it builds,
    keeping its position in the ensemble.

    Draw ``i`` normally uses ``seed + i``; attempt ``k`` uses ``seed + k * N + i``
    (``N`` = randomizations per combination), so a replacement never repeats a seed
    of this build and a rerun picks the same replacement. With
    ``per_time_randomization`` both go through :func:`_randomization_seed` (base
    ``seed`` or ``seed + k * N``, plus the combination's time). After
    ``_MAX_RESEED_ATTEMPTS`` failed replacements the build raises -- a failure on
    every seed is systematic, not one unlucky draw.
    """
    rebuilt = _run_isolated({pos: (build_fn, (spec, *tasks[pos])) for pos in failed}, workers)
    recovered = {pos: result for pos, (ok, result) in rebuilt.items() if ok}

    per_draw = {
        (pos, combo, r): (build_fn, (spec, [combo], range(r, r + 1)))
        for pos, (ok, _) in rebuilt.items()
        if not ok
        for combo in tasks[pos][0]
        for r in tasks[pos][1]
    }
    outcome = _run_isolated(per_draw, workers) if per_draw else {}

    # Draws that failed alone: redraw them, all of one attempt in parallel.
    n_rand = max(rand_range.stop for _, rand_range in tasks)
    pending = {key: str(value) for key, (ok, value) in outcome.items() if not ok}
    attempts: dict = dict.fromkeys(pending, 0)
    while pending:
        attempt = 1 + max(attempts[key] for key in pending)
        if attempt > _MAX_RESEED_ATTEMPTS:
            listing = ", ".join(
                f"t={combo[0]:g} ngroups={combo[1]} #{r}" for (_, combo, r) in sorted(pending)
            )
            raise RuntimeError(
                f"SqDRIFT: {len(pending)} draw(s) failed to build with "
                f"{_MAX_RESEED_ATTEMPTS} replacement seeds each ({listing}). "
                f"Last error: {next(iter(pending.values()))}"
            )
        reseeded_spec = replace(spec, seed=spec.seed + attempt * n_rand)
        retry = _run_isolated(
            {
                key: (build_fn, (reseeded_spec, [key[1]], range(key[2], key[2] + 1)))
                for key in pending
            },
            workers,
        )
        for key, (ok, value) in retry.items():
            attempts[key] = attempt
            if ok:
                _, combo, r = key
                per_time = spec.per_time_randomization
                # Sharding (hence a retry) only happens for a seeded build.
                assert spec.seed is not None and reseeded_spec.seed is not None
                record = {
                    "time": combo[0],
                    "num_groups": combo[1],
                    "randomization": r,
                    "seed": _randomization_seed(spec.seed, r, combo[0], per_time),
                    "replacement_seed": _randomization_seed(
                        reseeded_spec.seed, r, combo[0], per_time
                    ),
                    "attempts": attempt,
                }
                value.reseeded.append(record)
                outcome[key] = (True, value)
                del pending[key]
                logger.warning(
                    "SqDRIFT: draw t=%g ngroups=%d #%d crashed with seed %d; rebuilt with "
                    "replacement seed %d (attempt %d).",
                    combo[0],
                    combo[1],
                    r,
                    record["seed"],
                    record["replacement_seed"],
                    attempt,
                )
            else:
                pending[key] = str(value)

    for pos in sorted({key[0] for key in per_draw}):
        chunk = SqdriftBuildResult(optimize_requested=spec.optimize)
        for combo in tasks[pos][0]:
            for r in tasks[pos][1]:
                chunk.extend(outcome[(pos, combo, r)][1])
        recovered[pos] = chunk
    return recovered


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
    from qiskit_fermions.transpiler.passes import QDriftTrotterization
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

    def _draw_seed(randomization: int, evolution_time: float) -> int | None:
        """Seed for randomization ``i`` at ``evolution_time`` (None stays unseeded).

        ``seed + i`` by default (:func:`_randomization_seed`). The index is the
        randomization number *within* a ``(time, num_groups)`` combination, and
        restarts at 0 for each combination, so combinations that share a
        randomization index also share a qDRIFT draw, which isolates the effect of
        ``time``/``num_groups`` from sampling noise -- unless
        ``per_time_randomization`` seeds each time on its own.
        """
        if spec.seed is None:
            return None
        return _randomization_seed(
            spec.seed, randomization, evolution_time, spec.per_time_randomization
        )

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

    def _relabel_pass() -> Any:
        """The caching ``RelabelModes`` for one draw (see :mod:`.relabel`)."""
        return make_caching_relabel_modes(
            backend=spec.relabel_solver,
            solver=NativeHighsSolverAdapter(time_limit=spec.time_limit),
            time_limit=spec.time_limit,
            cpsat_workers=spec.cpsat_workers,
            cpsat_deterministic=spec.cpsat_deterministic,
            cache_dir=spec.relabel_cache_dir,
        )

    def _build_one(n_groups: int, draw_seed: int | None, evolution_time: float) -> Any:
        """Synthesise one randomization's circuit.

        A single pass manager whose optimization stage is qDRIFT sampling followed,
        with ``optimize``, by the caching ``RelabelModes``: one solve per distinct
        model, one synthesis per circuit.
        """
        pm = _pass_manager(draw_seed)
        passes = _sampling_passes(n_groups, draw_seed)
        if spec.optimize:
            passes.append(_relabel_pass())
        if passes:
            pm.optimization = FermionicPassManager(passes)
        return pm.run(_fresh_circuit(evolution_time))

    result = SqdriftBuildResult(num_modes=num_modes, optimize_requested=spec.optimize)
    for evolution_time, n_groups in combos:
        for randomization in rand_range:
            draw_seed = _draw_seed(randomization, evolution_time)
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
