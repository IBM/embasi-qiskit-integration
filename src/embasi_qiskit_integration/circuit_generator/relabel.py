# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Mode relabeling for SqDRIFT circuits: solver adapter, permutation cache, CP-SAT backend.

``RelabelModes`` solves one excitation-span model per circuit, and that model depends only
on the set of excitations the qDRIFT draw gathered, never on the evolution time (which
scales the angles only). :func:`make_caching_relabel_modes` builds a ``RelabelModes`` that
keys its permutation on a content hash of that model, in memory and optionally on disk, so
a ``time`` sweep sharing its draws solves each model once. ``backend="cpsat"`` solves the
same default objective with OR-Tools CP-SAT instead of HiGHS.

``qiskit-fermions`` stays an optional dependency: nothing here imports it until a relabel
pass is actually built.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Bump when the key payload or the cached value changes meaning.
_CACHE_VERSION = 1

# Process-local permutation cache, keyed by :func:`_excitation_key`. It lives per (worker)
# process; the disk cache under ``cache_dir`` carries hits across worker processes. Values
# are tuples so a caller cannot mutate a cached entry.
_PERMUTATION_CACHE: dict[str, tuple[int, ...]] = {}

RELABEL_SOLVERS = ("highs", "cpsat")


def _importable(module: str) -> bool:
    """``find_spec`` probe that treats a broken install as missing instead of raising."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs only
        return False


def relabel_available() -> bool:
    """True if the optional relabel solver stack (``pyomo`` + HiGHS) is importable.

    ``RelabelModes`` needs ``pyomo`` to build the MILP and a solver to solve it;
    :class:`NativeHighsSolverAdapter` uses pyomo's bundled appsi HiGHS interface,
    which requires ``highspy``. With either missing the pass cannot find a
    permutation (it warns and returns the circuit unchanged), so callers should
    check this before requesting ``optimize=True``.

    Probes with ``find_spec`` rather than importing, so the classical and replay
    paths never pay for pulling pyomo in. A broken or partially-installed package
    counts as unavailable: this gates a default, so it must answer rather than
    propagate.
    """
    return all(_importable(module) for module in ("pyomo", "highspy"))


def cpsat_available() -> bool:
    """True if OR-Tools (the ``relabel_solver="cpsat"`` backend) is importable."""
    return _importable("ortools")


class NativeHighsSolverAdapter:
    """Solver adapter for ``RelabelModes`` backed by native appsi HiGHS.

    ``RelabelModes.find_permutation`` calls ``solver.solve(model)`` and then reads
    the solution back via ``pyomo.environ.value(model.y[i])``. The legacy
    ``pyomo.environ.SolverFactory("appsi_highs")`` wrapper defaults to
    ``load_solutions=True``, which raises ``RuntimeError`` whenever a solve returns
    no feasible solution -- and it does so even for the feasible randomizations in
    a batch, leaving every circuit unpermuted.

    This adapter uses the native ``pyomo.contrib.appsi.solvers.highs.Highs``
    interface with ``load_solution=False`` so an infeasible/unbounded solve does
    not raise inside pyomo. We then:

    - load the solution into the model on success, so the pass can read ``model.y``;
    - re-raise ``RuntimeError`` on an infeasible solve, which ``RelabelModes``
      already catches to fall back to "no permutation" gracefully.

    The net effect: feasible randomizations yield real mode permutations, while
    genuinely-infeasible ones (e.g. a degenerate tiny active space whose sampled
    terms all reduce to identity) still degrade cleanly instead of poisoning the
    whole batch.
    """

    def __init__(self, time_limit: float | None = None, deterministic: bool = False) -> None:
        """Configure each solve.

        Args:
            time_limit: optional wall-clock limit (seconds) per solve; ``0``/``None``
                means no limit. A solve that hits it without a feasible point yields
                no permutation, and the circuit is kept unpermuted.
            deterministic: pin HiGHS to a single thread and disable its concurrent
                MIP mode. Off by default (HiGHS chooses its own threading). The
                excitation-span MILP is highly degenerate, so which optimum comes back
                is not determined by the model alone; pinning the threads removes the
                load-dependent part of that. It does not make HiGHS a pure function of
                its input (repeated solves of one model in one process were seen to
                differ), which is why the permutation is cached per model
                (:func:`make_caching_relabel_modes`) rather than re-solved for every
                circuit that shares it.
        """
        self._time_limit = time_limit
        self._deterministic = deterministic

    def solve(self, model: Any) -> Any:
        """Solve ``model`` with native HiGHS; load vars on success, raise if infeasible."""
        import logging

        from pyomo.contrib.appsi.solvers.highs import Highs

        opt = Highs()
        opt.config.load_solution = False
        opt.config.log_level = logging.DEBUG
        if self._time_limit:
            opt.highs_options["time_limit"] = float(self._time_limit)
        if self._deterministic:
            opt.highs_options["threads"] = 1
            opt.highs_options["parallel"] = "off"
        result = opt.solve(model)
        if result.best_feasible_objective is None:
            raise RuntimeError(f"no feasible solution (termination={result.termination_condition})")
        result.solution_loader.load_vars()
        return result


# ----- the model ``RelabelModes`` builds ------------------------------------------- #


def _gather_excitations(dag: Any) -> tuple[list[tuple[int, ...]], int]:
    """Return the excitations ``RelabelModes`` hands to its model builder, and ``num_modes``.

    Mirrors ``RelabelModes.find_permutation`` of the pinned qiskit-fermions: one tuple of
    mode indices per term of every ``Evolution`` gate, in DAG storage order (the operator's
    own indices; the qDRIFT ``Evolution`` gates act on the whole register).
    """
    from qiskit_fermions.circuit.library import Evolution
    from qiskit_fermions.operators import FermionOperator

    gathered: list[tuple[int, ...]] = []
    for node in dag.op_nodes():
        if not isinstance(node.op, Evolution):
            continue
        hamil = node.op.operator
        if not isinstance(hamil, FermionOperator):
            raise NotImplementedError(
                "The excitation-span model assumes fermionic mode excitations defined by "
                f"`FermionOperator` instances; got an Evolution of {type(hamil)}."
            )
        modes = hamil.get_modes()
        boundaries = list(hamil.get_boundaries())
        for start, stop in zip(boundaries, boundaries[1:], strict=False):
            gathered.append(tuple(int(m) for m in modes[start:stop]))
    return gathered, dag.num_qubits()


def _model_terms(
    excitations: Iterable[tuple[int, ...]],
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Return the ``(P2, P4)`` index sets of ``build_excitation_span_minimization_model``.

    Same preprocessing as the builder: an index occurring twice in a tuple cancels, tuples
    left with 0 or 1 index impose nothing, and the 2- and 4-index tuples are deduplicated in
    their own (ordered) form.
    """
    import numpy as np

    i2: list[tuple[int, ...]] = []
    i4: list[tuple[int, ...]] = []
    for ind in excitations:
        deduplicated: list[int] = []
        ignore = set()
        for i in ind:
            if i in ignore:
                continue
            if i in deduplicated:
                ignore.add(i)
                deduplicated.remove(i)
            else:
                deduplicated.append(i)
        match len(deduplicated):
            case 0 | 1:
                continue
            case 2:
                i2.append(tuple(deduplicated))
            case 4:
                i4.append(tuple(deduplicated))
            case _:
                raise ValueError(
                    f"Unsupported excitation after cancellation: {ind} -> {tuple(deduplicated)}"
                )

    def unique(rows: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
        return [tuple(int(i) for i in row) for row in np.unique(rows, axis=0)] if rows else []

    return unique(i2), unique(i4)


def _excitation_key(
    i2: list[tuple[int, ...]], i4: list[tuple[int, ...]], num_modes: int, spec: dict[str, Any]
) -> str:
    """Content hash of one relabel model plus the solver settings that pick its answer."""
    payload = {
        "version": _CACHE_VERSION,
        "num_modes": int(num_modes),
        "i2": sorted(tuple(int(i) for i in t) for t in i2),
        "i4": sorted(tuple(int(i) for i in t) for t in i4),
        "qiskit_fermions": importlib.metadata.version("qiskit-fermions"),
        **spec,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ----- disk cache and the per-model solve lock ------------------------------------- #


def _is_permutation(permutation: Any, num_modes: int) -> bool:
    """True if ``permutation`` is a list/tuple holding each of ``range(num_modes)`` once."""
    return (
        isinstance(permutation, (list, tuple))
        and all(isinstance(i, int) for i in permutation)
        and sorted(permutation) == list(range(num_modes))
    )


def _read_disk(cache_dir: Path, key: str, num_modes: int) -> tuple[int, ...] | None:
    """Cached permutation from ``<cache_dir>/<key>.json``; anything unusable is a miss."""
    try:
        permutation = json.loads((cache_dir / f"{key}.json").read_text())
    except (OSError, ValueError):
        return None
    return tuple(permutation) if _is_permutation(permutation, num_modes) else None


def _write_disk(cache_dir: Path, key: str, permutation: tuple[int, ...]) -> None:
    """Write ``permutation`` to ``<cache_dir>/<key>.json`` atomically (rename into place)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=cache_dir, suffix=".tmp", delete=False) as tmp:
        json.dump(list(permutation), tmp)
    Path(tmp.name).replace(cache_dir / f"{key}.json")


# A lock older than time_limit + this margin belongs to a worker that died mid-solve.
_LOCK_MARGIN_S = 30.0
_LOCK_POLL_S = 0.5


def _claim(cache_dir: Path, key: str, num_modes: int, time_limit: float) -> tuple[int, ...] | bool:
    """Claim the solve of ``key`` for this worker, or wait for the worker that has it.

    Returns the cached permutation if another worker writes it, ``True`` if this worker
    now holds ``<key>.lock`` (and must call :func:`_release`), or ``False`` to solve
    without a lock when there is no time limit (a stale lock cannot be told from a long
    solve). A holder that releases without a result (its solve found none or failed),
    or whose lock outlives ``time_limit`` plus a margin (it died), is replaced by one
    of the waiters.
    """
    if time_limit <= 0:
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = cache_dir / f"{key}.lock"
    while True:
        # Look for a result before every attempt on the lock: a holder writes its result
        # and then releases, so the lock can be free again with the answer on disk.
        cached = _read_disk(cache_dir, key, num_modes)
        if cached is not None:
            return cached
        try:
            lock.touch(exist_ok=False)  # O_CREAT | O_EXCL: exactly one worker wins
        except FileExistsError:
            pass
        else:
            # The holder may have written and released between the read above and the
            # touch; recheck so this worker does not solve an answer already on disk.
            cached = _read_disk(cache_dir, key, num_modes)
            if cached is not None:
                lock.unlink(missing_ok=True)
                return cached
            return True
        try:
            age = time.time() - lock.stat().st_mtime
        except FileNotFoundError:
            continue  # released: loop to read its result, or to take over if it found none
        if age > time_limit + _LOCK_MARGIN_S:
            lock.unlink(missing_ok=True)  # the holder died; race the others to retake it
            continue
        time.sleep(_LOCK_POLL_S)


def _release(cache_dir: Path, key: str) -> None:
    """Drop this worker's ``<key>.lock`` (after any result is already on disk)."""
    (cache_dir / f"{key}.lock").unlink(missing_ok=True)


# ----- CP-SAT ---------------------------------------------------------------------- #


def _solve_cpsat(
    i2: list[tuple[int, ...]],
    i4: list[tuple[int, ...]],
    num_modes: int,
    *,
    time_limit: float,
    workers: int,
    deterministic: bool = False,
) -> tuple[list[int] | None, int | None]:
    """Solve the default ``RelabelModes`` objective with CP-SAT.

    Returns ``(permutation, objective)``, or ``(None, None)`` when no solution was found or
    the model has no excitation (the MILP is unbounded there and HiGHS returns none).

    Notes
    -----
    Mode :math:`i` sits at position :math:`y_i`, all distinct in :math:`[0, n-1]`. The span of
    a pair is :math:`|y_i - y_j|`, of a quadruple :math:`\\max y - \\min y`. The library's
    default ``multi`` objective is :math:`S + 0.1 \\cdot \\frac{1}{D}\\sum_e (s_e + 1)` with
    :math:`S = \\max_e (s_e + 1)` over the :math:`D` excitations (arXiv:2508.02578,
    Sec. III.B); multiplied by :math:`10 D` it is the integer objective

    .. math:: 10 D \\, S + \\sum_e (s_e + 1),

    with the same minimizers. Mirroring :math:`j \\to n-1-j` leaves it unchanged, so
    :math:`y_0 < y_1` removes that symmetry.
    """
    # highspy must load before cp_model. The OR-Tools wheel bundles its own libhighs, and on
    # macOS dyld coalesces weak C++ symbols process-wide, first image wins: with OR-Tools
    # loaded first, a later HiGHS relabel solve in this process can segfault (seen on
    # ortools 9.12-9.15 x highspy 1.12-1.13, darwin-arm64; not on linux-arm64).
    try:
        import highspy  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:  # pragma: no cover - no HiGHS to protect
        pass

    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - guarded by cpsat_available()
        raise ImportError(
            "relabel_solver='cpsat' needs OR-Tools; install it with: pip install -e '.[relabel]'"
        ) from exc

    if not i2 and not i4:
        return None, None

    model = cp_model.CpModel()
    y = [model.new_int_var(0, num_modes - 1, f"y{i}") for i in range(num_modes)]
    model.add_all_different(y)
    if num_modes >= 2:
        model.add(y[0] < y[1])

    spans: list[Any] = []
    for i, j in i2:
        d = model.new_int_var(0, num_modes - 1, f"d_{i}_{j}")
        model.add_abs_equality(d, y[i] - y[j])
        spans.append(d)
    for quad in i4:
        hi = model.new_int_var(0, num_modes - 1, "")
        lo = model.new_int_var(0, num_modes - 1, "")
        model.add_max_equality(hi, [y[i] for i in quad])
        model.add_min_equality(lo, [y[i] for i in quad])
        spans.append(hi - lo)

    max_span = model.new_int_var(1, num_modes, "max_span")
    for span in spans:
        model.add(max_span >= span + 1)
    denom = len(i2) + len(i4)
    model.minimize(10 * denom * max_span + sum(span + 1 for span in spans))

    solver = cp_model.CpSolver()
    if time_limit > 0:
        solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = 0
    # A parallel portfolio races its workers, so ties between equally good permutations
    # (and, under a time limit, the incumbent found) depend on thread timing. Interleaved
    # search runs the workers in a fixed schedule: reproducible, but a weaker search.
    solver.parameters.interleave_search = bool(deterministic)
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None
    return [int(solver.value(v)) for v in y], int(round(solver.objective_value))


# ----- the caching pass ------------------------------------------------------------ #


@functools.cache
def _caching_relabel_modes_class() -> type:
    """Define :class:`CachingRelabelModes` on first use, so importing this module never
    imports qiskit-fermions."""
    from qiskit_fermions.transpiler.passes import RelabelModes

    class CachingRelabelModes(RelabelModes):  # type: ignore[misc, valid-type]
        """``RelabelModes`` that caches its permutation by model content, on HiGHS or CP-SAT.

        ``backend="highs"`` solves a miss exactly as stock ``RelabelModes(solver=solver)``;
        ``backend="cpsat"`` solves the default objective with :func:`_solve_cpsat`. Either
        way only a found permutation is cached, so a model that failed (time limit,
        infeasible) is retried by the next circuit holding it.
        """

        def __init__(
            self,
            *,
            solver: Any = None,
            backend: str = "highs",
            time_limit: float = 60.0,
            cpsat_workers: int = 8,
            cpsat_deterministic: bool = False,
            cache_dir: str | Path | None = None,
            cache: bool = True,
            **kwargs: Any,
        ) -> None:
            if backend not in RELABEL_SOLVERS:
                raise ValueError(f"unknown relabel backend {backend!r}; use 'highs' or 'cpsat'")
            if backend == "cpsat" and kwargs:
                raise NotImplementedError(
                    f"the cpsat relabel backend solves only the default objective; got {kwargs}"
                )
            super().__init__(solver=solver, **kwargs)
            self.backend = backend
            self.time_limit = float(time_limit)
            self.cpsat_workers = int(cpsat_workers)
            self.cpsat_deterministic = bool(cpsat_deterministic)
            self.cache = bool(cache)
            self.cache_dir = Path(cache_dir) if (cache and cache_dir is not None) else None

        def _spec(self) -> dict[str, Any]:
            """The solver identity that enters the cache key."""
            spec: dict[str, Any] = {"backend": self.backend, "time_limit": self.time_limit}
            if self.backend == "cpsat":
                spec["cpsat_workers"] = self.cpsat_workers
                spec["cpsat_deterministic"] = self.cpsat_deterministic
                spec["ortools"] = importlib.metadata.version("ortools")
            else:
                spec["solver"] = type(self.solver).__name__
                # The adapter's threading can change which tied optimum comes back.
                spec["solver_deterministic"] = bool(getattr(self.solver, "_deterministic", False))
                spec["objective"] = self._model_kwargs.get("objective", "multi")
                spec["mix_delta"] = float(self._model_kwargs.get("mix_delta", 0.1))
            return spec

        def find_permutation(self, dag: Any) -> tuple[list[int] | None, Any]:
            """Return the cached permutation of this DAG's model, solving it on a miss."""
            if self.backend == "highs" and self.solver is None:
                return super().find_permutation(dag)

            excitations, num_modes = _gather_excitations(dag)
            i2, i4 = _model_terms(excitations)
            if not self.cache:
                # Solve every circuit afresh: nothing is read from or written to a cache.
                if self.backend == "highs":
                    return super().find_permutation(dag)
                permutation, _ = _solve_cpsat(
                    i2,
                    i4,
                    num_modes,
                    time_limit=self.time_limit,
                    workers=self.cpsat_workers,
                    deterministic=self.cpsat_deterministic,
                )
                if permutation is None or not _is_permutation(permutation, num_modes):
                    return None, None
                return list(permutation), None
            key = _excitation_key(i2, i4, num_modes, self._spec())

            cached = _PERMUTATION_CACHE.get(key)
            if cached is None and self.cache_dir is not None:
                cached = _read_disk(self.cache_dir, key, num_modes)
                if cached is not None:
                    _PERMUTATION_CACHE[key] = cached
            if cached is not None:
                return list(cached), None

            # Workers sharing cache_dir solve each model once: the first claims it, the
            # others wait for its result (see _claim).
            locked_dir = None
            if self.cache_dir is not None:
                claim = _claim(self.cache_dir, key, num_modes, self.time_limit)
                if isinstance(claim, tuple):
                    _PERMUTATION_CACHE[key] = claim
                    return list(claim), None
                locked_dir = self.cache_dir if claim else None
            try:
                result = None
                if self.backend == "highs":
                    permutation, result = super().find_permutation(dag)
                else:
                    permutation, _ = _solve_cpsat(
                        i2,
                        i4,
                        num_modes,
                        time_limit=self.time_limit,
                        workers=self.cpsat_workers,
                        deterministic=self.cpsat_deterministic,
                    )
                if permutation is None or not _is_permutation(permutation, num_modes):
                    return None, None
                _PERMUTATION_CACHE[key] = tuple(permutation)
                if self.cache_dir is not None:
                    _write_disk(self.cache_dir, key, tuple(permutation))
                return list(permutation), result
            finally:
                if locked_dir is not None:
                    _release(locked_dir, key)

    return CachingRelabelModes


def make_caching_relabel_modes(**kwargs: Any) -> Any:
    """Build a caching ``RelabelModes`` pass (see :func:`_caching_relabel_modes_class`).

    Keyword arguments: ``solver`` (the pyomo solver for ``backend="highs"``), ``backend``
    (``"highs"`` or ``"cpsat"``), ``time_limit`` (seconds per solve, ``0`` = none; part of
    the cache key), ``cpsat_workers`` (CP-SAT threads per solve), ``cpsat_deterministic``
    (reproducible interleaved CP-SAT search), ``cache_dir`` (disk cache shared across
    processes; ``None`` keeps it in memory), ``cache`` (``False`` solves every circuit
    afresh, bypassing both caches). Any other keyword goes to the model builder (HiGHS
    only).
    """
    return _caching_relabel_modes_class()(**kwargs)
