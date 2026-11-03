# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The cached ``RelabelModes``, its CP-SAT backend, and per-time SqDRIFT draws.

The relabel model depends only on the sampled excitation set, which qDRIFT draws
independently of ``time``; the cache keys the permutation on that set so a time sweep
solves each model once. The solve tests run on synthetic, realistically sparse excitation
sets (about ``n`` pairs and ``n/2`` quadruples).
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embasi_qiskit_integration.circuit_generator import relabel
from embasi_qiskit_integration.circuit_generator.relabel import (
    NativeHighsSolverAdapter,
    cpsat_available,
    make_caching_relabel_modes,
    relabel_available,
)
from embasi_qiskit_integration.circuit_generator.sqdrift import (
    _BuildSpec,
    _distinct_draws,
    _plan_tasks,
    _randomization_seed,
    build_sqdrift_circuits,
    sqdrift_available,
)
from embasi_qiskit_integration.hamiltonian import fcidump

requires_fermions = pytest.mark.skipif(
    not sqdrift_available(), reason="qiskit-fermions not installed (fermions extra)"
)
requires_relabel = pytest.mark.skipif(
    not (sqdrift_available() and relabel_available()),
    reason="relabel solver stack not installed (relabel extra)",
)
requires_cpsat = pytest.mark.skipif(
    not (sqdrift_available() and relabel_available() and cpsat_available()),
    reason="OR-Tools not installed (relabel extra)",
)


@pytest.fixture(autouse=True)
def _clear_cache():
    relabel._PERMUTATION_CACHE.clear()
    yield
    relabel._PERMUTATION_CACHE.clear()


@pytest.fixture
def n2_ham(data_dir):
    return fcidump.read(data_dir / "n2_8o10e.fcidump")


def _sparse_excitations(num_modes: int, seed: int = 0) -> list[tuple[int, ...]]:
    """About ``num_modes`` distinct pairs and ``num_modes // 2`` quadruples."""
    rng = np.random.default_rng(seed)
    pairs = [
        tuple(int(i) for i in rng.choice(num_modes, 2, replace=False)) for _ in range(num_modes)
    ]
    quads = [
        tuple(int(i) for i in rng.choice(num_modes, 4, replace=False))
        for _ in range(max(1, num_modes // 2))
    ]
    return pairs + quads


def _scaled_objective(perm, i2, i4) -> int:
    """``10 * D * max(span + 1) + sum(span + 1)``: the library default objective times 10 D."""
    spans = [abs(perm[i] - perm[j]) for i, j in i2]
    spans += [max(perm[k] for k in q) - min(perm[k] for k in q) for q in i4]
    return 10 * len(spans) * max(s + 1 for s in spans) + sum(s + 1 for s in spans)


def _dag(num_modes: int, excitations: list[tuple[int, ...]], time: float = 1.0):
    """A fermionic DAG with one ``Evolution`` per excitation (its time only scales the angle)."""
    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import Evolution
    from qiskit_fermions.operators import FermionOperator
    from qiskit_fermions.transpiler import FermionicCircuitToDAG

    circ = FermionicCircuit(num_modes)
    for ex in excitations:
        half = len(ex) // 2
        actions = tuple((True, m) for m in ex[:half]) + tuple((False, m) for m in ex[half:])
        op = FermionOperator.from_terms([(actions, 1.0)])
        circ.append(Evolution(num_modes, op, time), circ.modes)
    return FermionicCircuitToDAG().run(circ)


def _highs_solver(time_limit: float = 10.0):
    return NativeHighsSolverAdapter(time_limit=time_limit)


def _spec(**overrides) -> _BuildSpec:
    fields = {
        "ham": None,
        "method": "qdrift",
        "filter_diagonal_terms": True,
        "filter_trivial": False,
        "atol": 1e-16,
        "seed": 42,
        "measure": False,
        "include_initial_state": False,
        "optimize": True,
        "time_limit": 60.0,
    }
    return _BuildSpec(**{**fields, **overrides})


# Gathering and preprocessing ----------------------------------------------------


@requires_relabel
def test_gather_and_terms_match_what_upstream_hands_the_model_builder(monkeypatch):
    """The excitations and (P2, P4) sets equal those RelabelModes builds its model from."""
    from qiskit_fermions.transpiler.passes import RelabelModes
    from qiskit_fermions.transpiler.passes.optimization import relabel_modes

    captured = {}
    builder = relabel_modes.build_excitation_span_minimization_model

    def spy(excitations, num_modes, **kwargs):
        captured["excitations"] = list(excitations)
        model = builder(excitations, num_modes, **kwargs)
        captured["p2"] = sorted(model.P2.data())
        captured["p4"] = sorted(model.P4.data())
        raise RuntimeError("stop after building")

    monkeypatch.setattr(relabel_modes, "build_excitation_span_minimization_model", spy)
    dag = _dag(6, [(0, 3), (3, 0), (1, 1), (2, 5, 4, 1), (0, 0, 2, 3)])
    with pytest.raises(RuntimeError, match="stop after building"):
        RelabelModes(solver=object()).find_permutation(dag)

    excitations, num_modes = relabel._gather_excitations(dag)
    assert num_modes == 6
    assert excitations == [tuple(int(i) for i in e) for e in captured["excitations"]]
    i2, i4 = relabel._model_terms(excitations)
    assert sorted(i2) == captured["p2"]
    assert sorted(i4) == captured["p4"]


def test_model_terms_cancel_and_dedupe_like_the_builder():
    """Repeated indices cancel, 0/1-length tuples drop, ordered tuples dedupe, 3 raises."""
    i2, i4 = relabel._model_terms(
        [(0, 1), (0, 1), (1, 0), (2, 2), (3,), (0, 0, 1, 2), (4, 5, 6, 7)]
    )
    assert i2 == [(0, 1), (1, 0), (1, 2)]
    assert i4 == [(4, 5, 6, 7)]
    assert all(type(i) is int for t in i2 + i4 for i in t)
    assert relabel._model_terms([]) == ([], [])
    with pytest.raises(ValueError, match="Unsupported excitation"):
        relabel._model_terms([(0, 1, 2)])


# Cache key ------------------------------------------------------------------------


@requires_fermions
def test_cache_key_is_canonical_and_sensitive_to_the_model_and_solver():
    """List order and numpy ints do not change the key; the model and solver settings do."""
    key_of = relabel._excitation_key
    spec = {"backend": "highs", "time_limit": 60.0, "mix_delta": 0.1}
    i2, i4 = [(0, 1), (2, 3)], [(0, 1, 2, 3)]
    key = key_of(i2, i4, 4, spec)
    assert key_of(i2[::-1], i4, 4, spec) == key
    assert key_of([(np.int64(0), np.int64(1)), (2, 3)], i4, 4, spec) == key
    assert key_of(i2, i4, 5, spec) != key
    assert key_of([(1, 0), (2, 3)], i4, 4, spec) != key
    for change in ({"backend": "cpsat"}, {"time_limit": 5.0}, {"mix_delta": 0.2}):
        assert key_of(i2, i4, 4, {**spec, **change}) != key


# Seeds and the worker cap -------------------------------------------------------------


def test_randomization_seed_default_is_seed_plus_index():
    """Off: exactly ``seed + i``, at any time."""
    seeds = [_randomization_seed(42, i, t, False) for i in range(3) for t in (1.0, 2.0)]
    assert seeds == [42, 42, 43, 43, 44, 44]


def test_randomization_seed_per_time_is_stable_distinct_and_in_range():
    """On: a fixed literal (cross-process stable), distinct per time and index, in [0, 2**63)."""
    assert _randomization_seed(42, 0, 1.0, True) == 5722321965008775927
    seeds = {_randomization_seed(42, i, t, True) for i in range(50) for t in (1.0, 2.0, 3.0)}
    assert len(seeds) == 150
    assert all(0 <= s < 2**63 for s in seeds)
    assert _randomization_seed(42, 0, 1, True) == _randomization_seed(42, 0, 1.0, True)


@pytest.mark.parametrize(
    ("times", "groups", "per_time", "draws"),
    [
        ([1.0, 2.0, 3.0], [20], False, 2),  # the three times share each randomization's draw
        ([1.0, 2.0, 3.0], [20], True, 6),  # every time is its own draw
        ([1.0, 2.0, 3.0], [10, 20], False, 4),  # another group count is another draw
        ([1.0, 1.0], [20], True, 2),  # same time, same seed
    ],
)
@pytest.mark.parametrize("workers", [1, 8])
def test_distinct_draws_follow_the_seed(times, groups, per_time, draws, workers):
    """The draw count comes from the same seed rule the build uses, however it is sharded."""
    tasks = _plan_tasks(times, groups, 2, workers, 42)
    assert _distinct_draws(_spec(per_time_randomization=per_time), tasks) == draws


@pytest.mark.parametrize(
    ("optimize", "per_time", "pool"),
    [(True, False, 2), (True, True, 6), (False, False, 6)],
)
def test_worker_pool_is_capped_at_the_distinct_draws(monkeypatch, optimize, per_time, pool):
    """With relabeling, no more workers than distinct draws (each is solved once);
    without it the per-task cap alone applies. The task split itself does not change."""
    from concurrent.futures import Future

    from embasi_qiskit_integration.circuit_generator import sqdrift
    from embasi_qiskit_integration.circuit_generator.sqdrift import (
        SqdriftBuildResult,
        _build_parallel,
    )

    seen: dict[str, int] = {}

    class FakePool:
        def __init__(self, max_workers, **_):
            seen["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, *_):
            future: Future = Future()
            future.set_result(SqdriftBuildResult())
            return future

    monkeypatch.setattr(sqdrift, "ProcessPoolExecutor", FakePool)
    tasks = _plan_tasks([1.0, 2.0, 3.0], [20], 2, 8, 42)
    assert len(tasks) == 6
    spec = _spec(optimize=optimize, per_time_randomization=per_time)
    _build_parallel(spec, tasks, 8)
    assert seen["max_workers"] == pool


# The caching pass ------------------------------------------------------------------


def _counting(monkeypatch) -> list[int]:
    """Count HiGHS solves (``RelabelModes.find_permutation`` misses) and CP-SAT solves."""
    from qiskit_fermions.transpiler.passes import RelabelModes

    calls: list[int] = []
    stock, cpsat = RelabelModes.find_permutation, relabel._solve_cpsat

    def stock_spy(self, dag):
        calls.append(1)
        return stock(self, dag)

    def cpsat_spy(*args, **kwargs):
        calls.append(1)
        return cpsat(*args, **kwargs)

    monkeypatch.setattr(RelabelModes, "find_permutation", stock_spy)
    monkeypatch.setattr(relabel, "_solve_cpsat", cpsat_spy)
    return calls


@requires_cpsat
def test_cache_hits_across_times_and_isolates_solver_settings(monkeypatch, tmp_path):
    """Same excitations at different times solve once; another time_limit or backend misses;
    a hit returns a fresh list; the disk entry serves a cleared in-memory cache."""
    calls = _counting(monkeypatch)
    excitations = _sparse_excitations(8)

    def find(time: float, **kwargs):
        options = {"solver": _highs_solver(), "cache_dir": tmp_path} | kwargs
        return make_caching_relabel_modes(**options).find_permutation(_dag(8, excitations, time))

    first, result = find(1.0)
    assert result is not None and len(calls) == 1
    again, hit_result = find(2.0)
    assert again == first and hit_result is None and len(calls) == 1
    assert isinstance(again, list)

    again.reverse()
    assert find(3.0)[0] == first and len(calls) == 1

    find(1.0, time_limit=5.0)
    find(1.0, backend="cpsat")
    assert len(calls) == 3

    assert len(list(tmp_path.glob("*.json"))) == 3
    relabel._PERMUTATION_CACHE.clear()
    assert find(1.0)[0] == first and len(calls) == 3


@requires_relabel
def test_corrupt_disk_entry_is_a_miss(monkeypatch, tmp_path):
    """An unreadable or non-permutation entry is re-solved and overwritten."""
    calls = _counting(monkeypatch)
    dag = _dag(8, _sparse_excitations(8))
    relabel_pass = make_caching_relabel_modes(solver=_highs_solver(), cache_dir=tmp_path)
    good, _ = relabel_pass.find_permutation(dag)
    (entry,) = tmp_path.glob("*.json")
    for bad in ("{not json", json.dumps([0, 0, 1, 2, 3, 4, 5, 6]), json.dumps([0, 1])):
        entry.write_text(bad)
        relabel._PERMUTATION_CACHE.clear()
        assert relabel_pass.find_permutation(dag)[0] == good
    assert len(calls) == 4
    assert json.loads(entry.read_text()) == good


# Solve lock -----------------------------------------------------------------------


def test_claim_takes_a_free_lock_and_returns_a_cached_result(tmp_path):
    """A free lock is taken; a result already on disk is returned without locking."""
    assert relabel._claim(tmp_path, "k", 4, time_limit=5.0) is True
    assert (tmp_path / "k.lock").exists()
    relabel._write_disk(tmp_path, "k", (1, 0, 3, 2))
    relabel._release(tmp_path, "k")
    assert relabel._claim(tmp_path, "k", 4, time_limit=5.0) == (1, 0, 3, 2)
    assert not (tmp_path / "k.lock").exists()
    assert relabel._claim(tmp_path, "free", 4, time_limit=0.0) is False  # no limit: no lock


def test_claim_waits_for_the_holder_and_takes_over_a_stale_lock(monkeypatch, tmp_path):
    """A waiter gets the holder's result, takes over the lock if the holder found none,
    and takes over a lock older than time_limit plus the margin (a dead holder)."""
    import os
    import threading
    import time

    monkeypatch.setattr(relabel, "_LOCK_POLL_S", 0.01)

    def holder(key: str, result: tuple | None) -> threading.Thread:
        (tmp_path / f"{key}.lock").touch()

        def finish() -> None:
            time.sleep(0.2)
            if result is not None:
                relabel._write_disk(tmp_path, key, result)
            relabel._release(tmp_path, key)

        thread = threading.Thread(target=finish)
        thread.start()
        return thread

    thread = holder("found", (2, 3, 0, 1))
    assert relabel._claim(tmp_path, "found", 4, time_limit=5.0) == (2, 3, 0, 1)
    thread.join()

    thread = holder("none", None)
    assert relabel._claim(tmp_path, "none", 4, time_limit=5.0) is True
    thread.join()

    stale = tmp_path / "stale.lock"
    stale.touch()
    old = time.time() - (5.0 + relabel._LOCK_MARGIN_S + 1)
    os.utime(stale, (old, old))
    assert relabel._claim(tmp_path, "stale", 4, time_limit=5.0) is True
    assert time.time() - stale.stat().st_mtime < 5  # a fresh lock, now this worker's


@requires_cpsat
def test_failed_solve_releases_its_lock(tmp_path):
    """A solve that finds no permutation leaves no lock behind for others to wait on."""
    relabel_pass = make_caching_relabel_modes(backend="cpsat", time_limit=5.0, cache_dir=tmp_path)
    assert relabel_pass.find_permutation(_dag(4, [(1, 1)])) == (None, None)  # empty model
    assert list(tmp_path.iterdir()) == []


@requires_relabel
def test_highs_miss_matches_stock_relabel_modes():
    """A HiGHS miss returns what stock RelabelModes(solver=...) returns on the same DAG."""
    from qiskit_fermions.transpiler.passes import RelabelModes

    dag = _dag(8, _sparse_excitations(8, seed=2))
    stock, _ = RelabelModes(solver=_highs_solver()).find_permutation(dag)
    cached, _ = make_caching_relabel_modes(solver=_highs_solver()).find_permutation(dag)
    assert stock is not None and cached == stock


@requires_relabel
def test_highs_without_solver_defers_to_upstream():
    """No solver: upstream's warning and no permutation, nothing cached."""
    with pytest.warns(UserWarning, match="solver"):
        assert make_caching_relabel_modes().find_permutation(_dag(4, [(0, 1)])) == (None, None)
    assert relabel._PERMUTATION_CACHE == {}


@requires_fermions
def test_cpsat_refuses_non_default_objective():
    """CP-SAT solves only the default objective, so builder kwargs are refused."""
    with pytest.raises(NotImplementedError, match="default objective"):
        make_caching_relabel_modes(backend="cpsat", mix_delta=0.2)
    with pytest.raises(ValueError, match="unknown relabel backend"):
        make_caching_relabel_modes(backend="gurobi")


# CP-SAT ---------------------------------------------------------------------------


@requires_cpsat
def test_cpsat_matches_brute_force_and_highs_at_n8():
    """CP-SAT's optimum equals the brute-force minimum and HiGHS's (scaled) optimum."""
    from qiskit_fermions.transpiler.passes import RelabelModes

    excitations = _sparse_excitations(8)
    i2, i4 = relabel._model_terms(excitations)
    brute = min(_scaled_objective(p, i2, i4) for p in itertools.permutations(range(8)))
    perm, value = relabel._solve_cpsat(i2, i4, 8, time_limit=10.0, workers=8)
    assert sorted(perm) == list(range(8))
    assert value == _scaled_objective(perm, i2, i4) == brute

    highs, _ = RelabelModes(solver=_highs_solver(30.0)).find_permutation(_dag(8, excitations))
    assert _scaled_objective(highs, i2, i4) == brute


@requires_cpsat
def test_cpsat_empty_model_has_no_permutation():
    """No excitation: no permutation, as HiGHS on the (unbounded) MILP."""
    assert relabel._solve_cpsat([], [], 6, time_limit=1.0, workers=1) == (None, None)


@requires_cpsat
def test_cpsat_deterministic_is_reproducible_on_many_threads():
    """With 8 threads CP-SAT returns one of several tied optima run to run; the
    deterministic mode returns the same one every time, at the same objective."""
    i2, i4 = relabel._model_terms(_sparse_excitations(16, seed=0))
    runs = [
        relabel._solve_cpsat(i2, i4, 16, time_limit=5.0, workers=8, deterministic=True)
        for _ in range(4)
    ]
    assert len({tuple(perm) for perm, _ in runs}) == 1
    assert len({value for _, value in runs}) == 1


@requires_cpsat
def test_cpsat_deterministic_keys_its_own_cache_entry():
    """The flag can change the answer, so it is part of the cache key (and highs ignores it)."""

    def spec(backend: str, deterministic: bool) -> dict:
        return make_caching_relabel_modes(
            backend=backend, solver=_highs_solver(), cpsat_deterministic=deterministic
        )._spec()

    assert spec("cpsat", True) != spec("cpsat", False)
    assert spec("highs", True) == spec("highs", False)


# A fresh interpreter, so no earlier test has loaded highspy already.
_HIGHS_AFTER_CPSAT = """
import sys
sys.path.insert(0, "tests")
from qiskit_fermions.transpiler.passes import RelabelModes

from embasi_qiskit_integration.circuit_generator import relabel
from test_sqdrift_relabel import _dag, _highs_solver, _sparse_excitations

excitations = _sparse_excitations(8, seed=0)
assert relabel._solve_cpsat(*relabel._model_terms(excitations), 8, time_limit=5.0, workers=2)[0]
assert RelabelModes(solver=_highs_solver()).find_permutation(_dag(8, excitations))[0]
"""


@requires_cpsat
def test_highs_runs_after_cpsat_in_one_process():
    """A CP-SAT solve does not break a later HiGHS MIP solve in the same process.

    The OR-Tools wheel bundles its own HiGHS; on macOS, loading it before highspy can
    make the next HiGHS relabel solve segfault (``_solve_cpsat`` imports highspy first).
    """
    result = subprocess.run(
        [sys.executable, "-c", _HIGHS_AFTER_CPSAT],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, f"rc={result.returncode}\n{result.stderr[-2000:]}"


@requires_fermions
def test_cpsat_without_ortools_is_refused(monkeypatch, n2_ham):
    """relabel_solver='cpsat' needs OR-Tools, unless relabeling is off."""
    from embasi_qiskit_integration.circuit_generator import sqdrift

    monkeypatch.setattr(sqdrift, "cpsat_available", lambda: False)
    monkeypatch.setattr(sqdrift, "relabel_available", lambda: True)
    with pytest.raises(ImportError, match="OR-Tools"):
        build_sqdrift_circuits(n2_ham, relabel_solver="cpsat")
    with pytest.raises(ValueError, match="unknown relabel_solver"):
        build_sqdrift_circuits(n2_ham, relabel_solver="gurobi")
    with pytest.raises(ValueError, match="cpsat_workers"):
        build_sqdrift_circuits(n2_ham, cpsat_workers=0)


def test_defaults():
    """The generator's own defaults are the full sweep (3 times x 3 group counts x 500,
    60 s per solve); the solver layers keep their one-point sweep and 10 s limit, which
    runs driven by SQDSolver, the solve CLI and the embedding workflow rely on."""
    import inspect

    from embasi_qiskit_integration.cli import SolveCommand
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow
    from embasi_qiskit_integration.solvers import SQDSolver

    builder = inspect.signature(build_sqdrift_circuits).parameters
    assert builder["time"].default == (1.0, 2.0, 3.0)
    assert tuple(builder["num_groups"].default) == (10, 15, 20)
    assert builder["num_randomizations"].default == 500
    assert builder["seed"].default == 42
    assert builder["time_limit"].default == 60.0

    solver = inspect.signature(SQDSolver.__init__).parameters
    assert solver["evolution_time"].default == 1.0
    assert solver["num_groups"].default == 15
    assert solver["num_randomizations"].default == 500
    assert solver["time_limit"].default == 10.0
    for model in (SolveCommand, EmbeddingWorkflow):
        fields = model.model_fields
        assert fields["time_limit"].default == 10.0
        assert fields["evolution_time"].default == [1.0]
        assert fields["num_groups"].default == [15]
        assert fields["num_randomizations"].default == 500

    for params in (builder, solver):
        assert params["relabel_solver"].default == "highs"
        assert params["cpsat_workers"].default == 8
        assert params["cpsat_deterministic"].default is False
        assert params["per_time_randomization"].default is False


# Through the builder -------------------------------------------------------------------

_SWEEP = {
    "method": "qdrift",
    "time": [1.0, 2.0, 3.0],
    "num_groups": 10,
    "num_randomizations": 2,
    "seed": 42,
    "include_initial_state": False,
    "time_limit": 10.0,
}


@requires_relabel
def test_time_sweep_solves_each_model_once(monkeypatch, n2_ham):
    """Shared draws: one solve per randomization across three times, same permutations."""
    calls = _counting(monkeypatch)
    result = build_sqdrift_circuits(n2_ham, **_SWEEP)
    assert len(calls) == 2
    perms = result.permutations
    assert perms[0:2] == perms[2:4] == perms[4:6]
    assert all(p is not None for p in perms)


@requires_relabel
def test_relabel_cache_dir_serves_a_repeated_build(monkeypatch, n2_ham, tmp_path):
    """The disk cache survives the in-memory one: a second build is all hits."""
    calls = _counting(monkeypatch)
    first = build_sqdrift_circuits(n2_ham, **_SWEEP, relabel_cache_dir=tmp_path)
    assert len(calls) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2
    relabel._PERMUTATION_CACHE.clear()
    second = build_sqdrift_circuits(n2_ham, **_SWEEP, relabel_cache_dir=tmp_path)
    assert len(calls) == 2
    assert second.permutations == first.permutations


@requires_relabel
def test_per_time_randomization_draws_each_time_independently(monkeypatch, n2_ham):
    """On: each time is seeded from its own value (and solved on its own); off: seed + i."""
    from embasi_qiskit_integration.circuit_generator import sqdrift

    recorded: list[tuple] = []
    real = sqdrift._randomization_seed

    def recorder(seed, index, time, per_time):
        value = real(seed, index, time, per_time)
        recorded.append((index, time, per_time, value))
        return value

    monkeypatch.setattr(sqdrift, "_randomization_seed", recorder)
    calls = _counting(monkeypatch)
    build_sqdrift_circuits(n2_ham, **_SWEEP, per_time_randomization=True)
    assert {(i, t) for i, t, on, _ in recorded if on} == {
        (i, t) for i in range(2) for t in (1.0, 2.0, 3.0)
    }
    assert len({value for *_, value in recorded}) == 6
    assert len(calls) == 6

    recorded.clear()
    build_sqdrift_circuits(n2_ham, **_SWEEP, per_time_randomization=False)
    assert {(i, value) for i, _, on, value in recorded} == {(0, 42), (1, 43)}


@requires_cpsat
def test_cpsat_through_the_builder_gives_valid_permutations(n2_ham):
    """relabel_solver='cpsat' gives every circuit a valid permutation, shared across times."""
    result = build_sqdrift_circuits(n2_ham, **_SWEEP, relabel_solver="cpsat", cpsat_workers=2)
    num_modes = 2 * n2_ham.norb
    for perm in result.permutations:
        assert perm is not None and sorted(perm) == list(range(num_modes))
    perms = result.permutations
    assert perms[0:2] == perms[2:4] == perms[4:6]


@requires_relabel
def test_sharded_time_sweep_solves_each_model_once_across_workers(n2_ham, tmp_path):
    """Sharded by time, the workers share one disk cache: two solves for six circuits,
    and the result matches the sequential build's permutations and order."""
    sharded = build_sqdrift_circuits(n2_ham, **_SWEEP, workers=6, relabel_cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert not list(tmp_path.glob("*.lock"))
    perms = sharded.permutations
    assert perms[0:2] == perms[2:4] == perms[4:6]
    assert sharded.randomization_indices == [0, 1] * 3

    relabel._PERMUTATION_CACHE.clear()
    sequential = build_sqdrift_circuits(n2_ham, **_SWEEP, relabel_cache_dir=tmp_path)
    assert sequential.permutations == sharded.permutations
