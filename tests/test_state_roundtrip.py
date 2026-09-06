# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The cross-process state seam: ``export_state`` / ``restore_state`` / ``seed_for_cycle``.

An out-of-process outer loop (one embedding cycle per OS process, e.g. a workflow
engine driving ``max_cycles=1`` repeatedly) cannot carry a live adapter across a round
boundary -- it holds a live ``ProjectionEmbedding`` and two PySCF ``KS`` objects, none of
them picklable.  What it *can* carry is arrays.  These tests pin the three properties
such a loop depends on:

1. **Round-trip fidelity.** A snapshot restored into an independently-built adapter
   reassembles the *same* ``F_emb``.  If it did not, every round after the first would
   downfold a slightly different Hamiltonian and the loop's trajectory would drift
   for reasons unrelated to the physics.
2. **Refusal on mismatch.** A snapshot paired with a different system is rejected
   rather than assembled.  This is the failure the seam exists to prevent: the arrays
   are all the right *shape* for a same-size basis, so a wrong pairing produces a
   plausible, wrong energy with nothing to flag it.
3. **Seed-schedule purity.** ``seed_for_cycle`` reproduces what the in-place
   ``_maybe_reseed`` bump produces.  The schedule is triangular
   (``base + k(k+1)/2``), not linear, because the historical implementation mutated
   the solver's own attribute -- so it reads as ``base + cycle`` and is not.  A driver
   that assumed the linear form would agree at cycles 0 and 1 and diverge from cycle 2
   on, i.e. pass a two-round smoke test and silently solve a different subspace
   thereafter.

Everything here runs with **no EmbASI**: the round-trip uses ``test_outer_loop``'s
partitioned-RHF mock, and the seed schedule is pure arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from embasi_qiskit_integration.embedding import seed_for_cycle  # noqa:E402

from test_outer_loop import _build_adapter  # noqa:E402


def _ran_adapter(**kwargs):
    """An adapter with ``run_low_level()`` already applied (state populated)."""
    adapter = _build_adapter(**kwargs)
    adapter.run_low_level()
    return adapter


def _trace_preserving_nudge(adapter, scale: float = 0.02):
    """A γ^A perturbation that keeps ``tr(γ^A S)`` integral.

    The adapter validates that the electron count is integral, so a fed-back density
    cannot simply be scaled.  A similarity transform by ``exp(scale * K)`` with ``K``
    antisymmetric in the S-orthogonal frame moves the density without moving its
    trace against ``S`` -- which is what a genuine feedback cycle also does.
    """
    import scipy.linalg as sla

    dm = np.asarray(adapter._dm_a, dtype=float)
    s = np.asarray(adapter._s, dtype=float)
    x = sla.sqrtm(s).real  # S^{1/2}
    x_inv = np.linalg.inv(x)
    orth = x @ dm @ x  # density in the orthogonal frame; trace is the electron count
    k = np.zeros_like(orth)
    k[0, -1], k[-1, 0] = scale, -scale
    u = sla.expm(k)
    return x_inv @ (u @ orth @ u.T) @ x_inv


# --------------------------------------------------------------------------- #
# 1. Round-trip fidelity
# --------------------------------------------------------------------------- #
def test_export_state_carries_every_array_the_fock_needs():
    """The snapshot names exactly the arrays ``_assemble_fock_a_only`` consumes."""
    state = _ran_adapter().export_state()
    for key in ("dm_a", "dm_a_init", "dm_b", "fock", "s", "p_b", "v_emb_embasi"):
        assert key in state, f"snapshot is missing {key!r}"
        assert np.asarray(state[key]).ndim == 2
    assert "fingerprint" in state


def test_export_state_before_run_low_level_is_refused():
    """A snapshot taken before the SCF would be all ``None``; say so rather than emit it."""
    with pytest.raises(RuntimeError, match="before run_low_level"):
        _build_adapter().export_state()


def test_restore_state_reproduces_fock_in_an_independent_adapter():
    """The central claim: a restored snapshot rebuilds the same ``F_emb``.

    Two *independently constructed* adapters stand in for two OS processes.  The
    second runs its own supersystem SCF (as a fresh round must, to populate the A_LL
    one-electron blocks), then restores the first's snapshot over it.
    """
    source = _ran_adapter()
    reference = np.array(source.h_emb, copy=True)
    state = source.export_state()

    fresh = _ran_adapter()  # a separate process's fresh SCF
    fresh.restore_state(state)

    np.testing.assert_allclose(fresh.h_emb, reference, atol=1e-12, rtol=0.0)


def test_restore_state_preserves_dm_a_init_against_the_fresh_scf():
    """``_dm_a_init`` must survive the round, not be reset by the fresh SCF.

    ``projection_energy`` computes the Eq. 8 correction as
    ``tr[(γ̃^A - γ^A_init) v_emb]`` against it, and it is set once, on the first
    ``run_low_level`` ever.  A round that let its own SCF overwrite it would silently
    zero the drift the correction measures.
    """
    source = _ran_adapter()
    fed = _trace_preserving_nudge(source)
    source.run_low_level_a_only(dma_in=fed, dmb_in=source._dm_b)
    state = source.export_state()
    assert not np.allclose(state["dm_a"], state["dm_a_init"])

    fresh = _ran_adapter()
    fresh.restore_state(state)
    np.testing.assert_allclose(fresh._dm_a_init, state["dm_a_init"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(fresh._dm_a, state["dm_a"], atol=1e-12, rtol=0.0)


def test_restored_dm_a_is_the_carried_value_not_the_fresh_scf_density():
    """The DIIS-residual guard.

    An out-of-process loop builds its DIIS input vector and its ``max|Δγ^A|``
    diagnostic from ``_dm_a``.  In-process that array holds what the *previous* cycle
    fed in; across a process boundary the fresh ``run_low_level`` overwrites it with
    this round's SCF density.  A consumer reading the post-SCF value builds its
    residual against the wrong reference -- silently, since both arrays exist in the
    same process and have the same shape.  Pin that ``restore_state`` is what decides
    which one is live.
    """
    source = _ran_adapter()
    fed = _trace_preserving_nudge(source)
    source.run_low_level_a_only(dma_in=fed, dmb_in=source._dm_b)
    state = source.export_state()

    fresh = _ran_adapter()
    post_scf_dm_a = np.array(fresh._dm_a, copy=True)
    fresh.restore_state(state)

    # The restored value is the carried one, and it is genuinely different from what
    # this round's own SCF wrote -- otherwise the test proves nothing.
    assert not np.allclose(post_scf_dm_a, state["dm_a"])
    np.testing.assert_allclose(fresh._dm_a, fed, atol=1e-12, rtol=0.0)


def test_restore_state_survives_an_npz_round_trip(tmp_path):
    """The snapshot must cross a process boundary as a file, which is how it travels."""
    source = _ran_adapter()
    reference = np.array(source.h_emb, copy=True)
    path = tmp_path / "embasi_state.npz"
    np.savez(path, **source.export_state())

    loaded = dict(np.load(path, allow_pickle=True))
    fresh = _ran_adapter()
    fresh.restore_state(loaded)
    np.testing.assert_allclose(fresh.h_emb, reference, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------- #
# 2. Refusal on mismatch
# --------------------------------------------------------------------------- #
def test_restore_state_rejects_a_mismatched_fingerprint():
    """A snapshot from a different partition must not be assembled into this adapter.

    ``n_occ_a`` changes which orbitals are subsystem A, so the arrays keep their
    shape while meaning something else -- exactly the silent-wrong-energy pairing the
    fingerprint exists to catch.  (The mock's fingerprint differs via ``mu`` here,
    which is the fingerprinted field it exposes.)
    """
    state = _ran_adapter(mu=1.0e6).export_state()
    fresh = _ran_adapter(mu=2.0e6)
    with pytest.raises(ValueError, match="does not match this adapter"):
        fresh.restore_state(state)


def test_restore_state_rejects_a_snapshot_with_no_fingerprint():
    """An unverifiable snapshot is refused unless the caller opts out explicitly."""
    state = _ran_adapter().export_state()
    del state["fingerprint"]
    with pytest.raises(ValueError, match="no 'fingerprint'"):
        _ran_adapter().restore_state(state)


def test_restore_state_rejects_a_wrong_nao_even_when_not_strict():
    """A shape mismatch is not a judgement call, so ``strict=False`` does not waive it."""
    state = _ran_adapter().export_state()
    state["fock"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="refusing to assemble F_emb"):
        _ran_adapter().restore_state(state, strict=False)


def test_restore_state_reports_a_missing_array_by_name():
    state = _ran_adapter().export_state()
    del state["p_b"]
    with pytest.raises(KeyError, match="p_b"):
        _ran_adapter().restore_state(state)


# --------------------------------------------------------------------------- #
# 3. Seed-schedule purity
# --------------------------------------------------------------------------- #
def test_seed_for_cycle_is_triangular_not_linear():
    """The documented schedule, spelled out: ``base + k(k+1)/2``."""
    assert [seed_for_cycle(24, k) for k in range(6)] == [24, 25, 27, 30, 34, 39]
    # The linear reading agrees for two cycles and then diverges, which is why a
    # short smoke test cannot distinguish them.
    assert seed_for_cycle(24, 1) == 24 + 1
    assert seed_for_cycle(24, 2) != 24 + 2


def test_seed_for_cycle_holds_the_seed_when_not_reseeding():
    """``reseed_sqd=False`` deliberately reuses the subspace, so the seed never moves."""
    assert [seed_for_cycle(24, k, reseed=False) for k in range(4)] == [24] * 4
    # Cycle 0 never reseeds either way.
    assert seed_for_cycle(24, 0) == seed_for_cycle(24, 0, reseed=False) == 24


@pytest.mark.parametrize("reseed", [True, False])
@pytest.mark.parametrize("base", [0, 24, 1234])
def test_maybe_reseed_matches_seed_for_cycle(base, reseed):
    """The equivalence test that pins the refactor as behaviour-preserving.

    Drives the *real* ``_maybe_reseed`` over several cycles with a stub solver and
    asserts the seeds it produces are exactly ``seed_for_cycle``'s.  This is the test
    that would have caught the original ``base + cycle`` misreading, and it is what
    lets an out-of-process driver trust the pure function instead of the mutation.
    """
    from types import SimpleNamespace

    from test_outer_loop import _workflow

    workflow = _workflow(seed=base, reseed_sqd=reseed)
    solver = SimpleNamespace(seed=base)
    observed = []
    for cycle in range(6):
        workflow._maybe_reseed(solver, cycle)
        observed.append(solver.seed)
    assert observed == [seed_for_cycle(base, c, reseed=reseed) for c in range(6)]
