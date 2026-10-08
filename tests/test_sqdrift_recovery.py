# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Recovery of the sharded SqDRIFT build from draws whose build crashes.

A worker that dies (HiGHS segfaulting in presolve on one relabel model) breaks the
whole process pool. The build narrows the failure down to the draws that fail on
their own and redraws each with a replacement seed until it builds. Driven with a
stand-in worker (``_crashing_build``) that really segfaults on chosen seeds, so no
circuits are synthesised.
"""

from __future__ import annotations

import pytest
from _crashing_build import build_chunk

from embasi_qiskit_integration.circuit_generator import sqdrift
from embasi_qiskit_integration.circuit_generator.sqdrift import (
    _BuildSpec,
    _build_parallel,
    _plan_tasks,
)

N_DRAWS, WORKERS, SEED = 40, 4, 24


def _build(monkeypatch, fail="", mode="segv", optimize=True):
    monkeypatch.setenv("EQI_TEST_FAIL_SEEDS", fail)
    monkeypatch.setenv("EQI_TEST_FAIL_MODE", mode)
    spec = _BuildSpec(
        ham=None,
        method="qdrift",
        filter_diagonal_terms=True,
        filter_trivial=False,
        atol=1e-16,
        seed=SEED,
        measure=False,
        include_initial_state=False,
        optimize=optimize,
        time_limit=10.0,
        canonical_permutation=False,
    )
    tasks = _plan_tasks([1.0], [15], N_DRAWS, WORKERS, SEED)
    return _build_parallel(spec, tasks, WORKERS, build_fn=build_chunk)


def _labels(seeds):
    return [f"t1-g15-s{s}" for s in seeds]


def test_no_failure_is_the_plain_sharded_build(monkeypatch):
    result = _build(monkeypatch)
    assert result.circuits == _labels(SEED + r for r in range(N_DRAWS))
    assert result.reseeded == []


@pytest.mark.parametrize("mode", ["segv", "raise"])
def test_a_crashing_draw_is_redrawn_with_a_reserved_seed(monkeypatch, caplog, mode):
    """Draw 13 (seed 37) crashes; it is rebuilt from seed 24 + 1*40 + 13 = 77 in its
    own place, and every other draw is the plain build."""
    result = _build(monkeypatch, fail="37", mode=mode)
    seeds = [SEED + r for r in range(N_DRAWS)]
    seeds[13] = SEED + N_DRAWS + 13
    assert result.circuits == _labels(seeds)
    assert result.randomization_indices == list(range(N_DRAWS))
    (record,) = result.reseeded
    assert record == {
        "time": 1.0,
        "num_groups": 15,
        "randomization": 13,
        "seed": 37,
        "replacement_seed": 77,
        "attempts": 1,
    }
    assert "rebuilt with replacement seed 77" in caplog.text


def test_retries_until_a_seed_builds(monkeypatch):
    # Seeds 37 and its first two replacements (77, 117) all crash; the third (157) builds.
    result = _build(monkeypatch, fail="37,77,117")
    (record,) = result.reseeded
    assert (record["replacement_seed"], record["attempts"]) == (157, 3)
    assert result.circuits[13] == "t1-g15-s157"


def test_a_failure_on_every_seed_raises(monkeypatch):
    monkeypatch.setattr(sqdrift, "_MAX_RESEED_ATTEMPTS", 2)
    with pytest.raises(RuntimeError, match=r"failed to build with 2 replacement seeds"):
        _build(monkeypatch, fail="all", mode="raise")


def test_without_relabeling_any_failure_raises(monkeypatch):
    with pytest.raises(RuntimeError, match=r"SqDRIFT chunk .* failed"):
        _build(monkeypatch, fail="37", optimize=False)
