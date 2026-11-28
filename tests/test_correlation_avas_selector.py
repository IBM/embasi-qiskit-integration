# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""``--selector correlation-avas``: semicanonical embedded-MP2 virtual ranking.

The selector keeps the AVAS target-AO ranking for the occupied block and replaces
the (saturating) virtual weight cut with MP2's marginal correlation.  These tests
pin the ranking algebra on a controlled one-occupied/three-virtual model, verify
the MP2 invariance that makes the semicanonical basis safe, and run the whole
adapter path on a stub where the weight cut provably picks the wrong virtual.

The real SIESTA evidence (H2 sigma* and the butyronitrile C/N pi* pair) lives in
``analysis/butyronitrile-pbe-hf-fci/correlation-selector``; it needs the archived
QCOMP ERI files and is not part of the unit suite.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.backends.pyscf import PySCFIntegrals
from embasi_qiskit_integration.projection_embedding_adapter import (
    ProjectionEmbeddingAdapter,
)
from embasi_qiskit_integration.selectors import (
    mp2_orbital_correlation,
    select_correlated_orbitals,
)


# --------------------------------------------------------------------------- #
# A controlled one-occupied / three-virtual model
# --------------------------------------------------------------------------- #
def _model(*, couple=(0.05, 0.30, 0.02), eps=(-1.0, -0.2, -0.1, 0.5)):
    """S=I, F_emb=diag(eps), and only the (0a|0a)/(00|aa)/(0a|a0) couplings.

    The correlating virtual is index 2 (largest coupling and a low gap); index 1
    carries the largest AVAS-style weight in the adapter test but is almost
    uncoupled.  The MP2 energy of this model is analytically ``sum_a K_a^2 / D_a``
    with ``D_a = 2 (f_oo - f_aa)``.
    """
    nao = 4
    nocc = 1
    eps = np.asarray(eps, dtype=float)
    eri = np.zeros((nao,) * 4)
    for a, k in zip((1, 2, 3), couple):
        # All index permutations of the two distinct integral classes.
        eri[0, a, 0, a] = eri[a, 0, a, 0] = eri[0, a, a, 0] = eri[a, 0, 0, a] = k
        eri[0, 0, a, a] = eri[a, a, 0, 0] = k
    oooo = eri[:nocc, :nocc, :nocc, :nocc]
    ovov = eri[:nocc, nocc:, :nocc, nocc:]
    vvoo = eri[nocc:, nocc:, :nocc, :nocc]
    h1_oo = np.diag(eps[:nocc])
    h1_vv = np.diag(eps[nocc:])
    return h1_oo, h1_vv, ovov, oooo, vvoo


def test_mp2_ranking_prefers_the_coupled_low_gap_virtual():
    h1_oo, h1_vv, ovov, oooo, vvoo = _model()
    correlation = mp2_orbital_correlation(h1_oo, h1_vv, ovov, oooo, vvoo)
    _, selected = select_correlated_orbitals(correlation, n_virtual=1)

    # Positional virtual 1 in the model (original column 2) is the strongly coupled
    # one; positional virtual 0 (original column 1) is the decoy carrying the largest
    # AVAS weight in the adapter test.  The selected index is positional in the
    # virtual block.
    assert selected.tolist() == [1]
    assert correlation.virtual_marginal_correlation[1] < correlation.virtual_marginal_correlation[0]
    assert correlation.virtual_marginal_correlation[1] < correlation.virtual_marginal_correlation[2]


def test_mp2_energy_and_natural_occupations_are_analytic_in_the_model():
    h1_oo, h1_vv, ovov, oooo, vvoo = _model()
    correlation = mp2_orbital_correlation(h1_oo, h1_vv, ovov, oooo, vvoo)

    # One occupied: f_oo = eps0; f_aa = eps_a + 2(00|aa) - (0a|a0) = eps_a + K_a.
    # E_MP2 = sum_a (0a|0a)^2 / (2 (f_oo - f_aa)); NOON_a = (0a|0a)^2 / D_a^2.
    couplings = np.array([0.05, 0.30, 0.02])
    eps = np.array([-1.0, -0.2, -0.1, 0.5])
    denominator = 2.0 * (eps[0] - (eps[1:] + couplings))
    expected = np.sum(couplings**2 / denominator)
    assert correlation.e_corr == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(
        correlation.virtual_natural_occupations,
        couplings**2 / denominator**2,
        rtol=1e-12,
    )
    # The MP2 1-RDM is unrelaxed but electron-number conserving.
    assert (
        correlation.virtual_natural_occupations.sum()
        + correlation.occupied_natural_occupations.sum()
        == pytest.approx(2.0, abs=1e-10)
    )


def test_mp2_is_invariant_to_intra_block_rotations_and_the_selected_subspace_is_covariant():
    h1_oo, h1_vv, ovov, oooo, vvoo = _model()
    reference = mp2_orbital_correlation(h1_oo, h1_vv, ovov, oooo, vvoo)
    _, ref_sel = select_correlated_orbitals(reference, n_virtual=2)

    rng = np.random.default_rng(7)
    qo, _ = np.linalg.qr(rng.standard_normal((1, 1)))  # sign only
    qv, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    rotated = mp2_orbital_correlation(
        qo.T @ h1_oo @ qo,
        qv.T @ h1_vv @ qv,
        np.einsum("iajb,ip,jq,ac,bd->pcqd", ovov, qo, qo, qv, qv),
        np.einsum("ijkl,ip,jq,kr,ls->pqrs", oooo, qo, qo, qo, qo),
        np.einsum("abij,ap,bq,ir,js->pqrs", vvoo, qv, qv, qo, qo),
    )
    assert rotated.e_corr == pytest.approx(reference.e_corr, rel=1e-12)
    _, rot_sel = select_correlated_orbitals(rotated, n_virtual=2)

    # The selected span is the same physical subspace: u_v(corr) has the old index
    # as its row, so the original-basis projector is Qv @ u_v[:, selected].
    p_ref = reference.virtual_rotation[:, ref_sel]
    p_ref = p_ref @ p_ref.T
    p_rot = qv @ rotated.virtual_rotation[:, rot_sel]
    p_rot = p_rot @ p_rot.T
    np.testing.assert_allclose(p_rot, p_ref, atol=1e-12)


def test_mp2_skips_an_exact_degeneracy_instead_of_dividing_by_zero():
    h1_oo = np.array([[-0.1]])
    h1_vv = np.array([[-0.1]])
    zeros = np.zeros((1, 1, 1, 1))
    correlation = mp2_orbital_correlation(h1_oo, h1_vv, zeros, zeros, zeros)
    assert correlation.n_skipped == 1
    assert correlation.e_corr == 0.0
    assert np.all(np.isfinite(correlation.virtual_natural_occupations))


def test_siesta_eri_mo_block_handles_unequal_left_and_right_widths():
    """The sparse QCOMP transform needs equal left widths; the mixed (ia|jb) block does not."""
    from embasi_qiskit_integration.backends._siesta_qcomp import SiestaEriProvider

    rng = np.random.default_rng(11)
    n_ao = 4
    eri = rng.standard_normal((n_ao,) * 4)

    def direct(ca, cb, cc, cd):
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, ca, cb, cc, cd)

    provider = SiestaEriProvider.__new__(SiestaEriProvider)
    provider.n_ao = n_ao
    provider._load = lambda: None
    provider._transform_four = direct

    blocks = [rng.standard_normal((n_ao, n)) for n in (1, 3, 1, 3)]
    np.testing.assert_allclose(provider.eri_mo_block(*blocks), direct(*blocks), atol=1e-12)
    # Equal widths take the direct path too.
    blocks = [rng.standard_normal((n_ao, 2)) for _ in range(4)]
    np.testing.assert_allclose(provider.eri_mo_block(*blocks), direct(*blocks), atol=1e-12)


def test_pyscf_eri_mo_block_matches_the_full_transform_for_mixed_blocks():
    """The mixed ``(ia|jb)`` orientation is easy to get wrong; pin it against eri_mo."""
    import pyscf

    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = mol.RHF().run(conv_tol=1e-10)
    ints = PySCFIntegrals(mf)
    c = mf.mo_coeff
    occ, virt = c[:, :1], c[:, 1:]
    unique = np.hstack([occ, virt])
    full = ints.eri_mo(unique)
    # (ia|jb): one occupied and one virtual on each side.
    np.testing.assert_allclose(
        ints.eri_mo_block(occ, virt, occ, virt),
        full[np.ix_([0], [1], [0], [1])],
        atol=1e-12,
    )
    # (ab|ij): the block the virtual Fock uses.
    np.testing.assert_allclose(
        ints.eri_mo_block(virt, virt, occ, occ),
        full[np.ix_([1], [1], [0], [0])],
        atol=1e-12,
    )


def test_select_correlated_orbitals_rejects_an_over_budget_request():
    h1_oo, h1_vv, ovov, oooo, vvoo = _model()
    correlation = mp2_orbital_correlation(h1_oo, h1_vv, ovov, oooo, vvoo)
    with pytest.raises(ValueError, match="n_virtual=4"):
        select_correlated_orbitals(correlation, n_virtual=4)


# --------------------------------------------------------------------------- #
# The adapter path on a stub: the weight cut picks the decoy, MP2 does not
# --------------------------------------------------------------------------- #
class _FakeIntegrals:
    """S=I, a bare one-body F_emb, and the same controlled ERI tensor as ``_model``."""

    capabilities = None

    def __init__(self, eps):
        self.nao = 4
        self.eps = np.asarray(eps, dtype=float)
        self._s = np.eye(self.nao)
        self._h = np.diag(self.eps)
        self._eri = np.zeros((self.nao,) * 4)
        for a, k in zip((1, 2, 3), (0.05, 0.30, 0.02)):
            self._eri[0, a, 0, a] = self._eri[a, 0, a, 0] = k
            self._eri[0, a, a, 0] = self._eri[a, 0, 0, a] = k
            self._eri[0, 0, a, a] = self._eri[a, a, 0, 0] = k

    def overlap(self):
        return self._s

    def hcore(self):
        return self._h

    def energy_nuc(self):
        return 0.0

    def veff_ll(self, dm):
        return np.zeros((self.nao, self.nao))

    def veff_hf(self, dm):
        return np.zeros((self.nao, self.nao))

    def eri_mo(self, coeff):
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", self._eri, coeff, coeff, coeff, coeff)

    def eri_mo_block(self, a, b, c, d):
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", self._eri, a, b, c, d)


def _stub_adapter(eps, n_occ=1):
    ints = _FakeIntegrals(eps)
    identity = np.eye(ints.nao)
    dm_a = 2.0 * np.outer(identity[:, 0], identity[:, 0])

    class _Stub(ProjectionEmbeddingAdapter):
        _s_arr = property(lambda self: ints.overlap())
        _fock_arr = property(lambda self: np.diag(ints.eps))
        _dm_a = property(lambda self: dm_a)
        _p_b = property(lambda self: np.zeros((ints.nao, ints.nao)))
        mo_a_ll = property(lambda self: identity[:, :n_occ])
        mo_b_ll = property(lambda self: np.zeros((ints.nao, 0)))

        def _validate_span(self, coeff):
            return None

    adapter = object.__new__(_Stub)
    adapter.p = SimpleNamespace(A_spin=0)
    adapter.ints = ints
    adapter.unrestricted = False
    adapter.mu = 1.0e6
    adapter._dm_a_spin = None
    return adapter


# Target metric 0 (vacuum), 1 (decoy), 0.9 (correlating), 0 (vacuum): the AVAS
# weight cut prefers the decoy virtual 1, exactly the failure mode seen on SIESTA.
_PROJECTOR = np.diag([0.0, 1.0, 0.9, 0.0])


def test_weight_cut_picks_the_decoy_but_correlation_avas_picks_the_coupled_virtual():
    adapter = _stub_adapter((-1.0, -0.2, -0.1, 0.5))
    baseline = adapter.build_orbitals_avas(ao_projector=_PROJECTOR, max_size=(2, 2))
    baseline_info = dict(adapter._avas_info)
    selected = adapter.build_orbitals_correlation_avas(ao_projector=_PROJECTOR, max_size=(2, 2))
    selected_info = dict(adapter._avas_info)

    identity = np.eye(4)
    # Both keep one occupied and one virtual.
    assert baseline.n_active_orbitals == selected.n_active_orbitals == 2
    # The weight cut keeps input virtual 1 (weight 1.0)...
    assert abs(baseline.c_active[:, 1] @ identity[:, 1]) == pytest.approx(1.0)
    np.testing.assert_allclose(baseline_info["virt_selection_weights"], [1.0])
    # ...while MP2 keeps input virtual 2 (the large coupling / small gap).
    assert abs(selected.c_active[:, 1] @ identity[:, 2]) == pytest.approx(1.0)
    assert selected_info["virt_selection_weights"][0] < 0.0
    assert selected_info["correlation_metric"] == "mp2-marginal"
    # The default 0.01 Ha guard passes: the coupled virtual's |marginal| is ~0.0375.
    assert selected_info["correlation_fallback"] is False


def test_guard_falls_back_to_the_exact_avas_space_when_the_signal_is_weak():
    adapter = _stub_adapter((-1.0, -0.2, -0.1, 0.5))
    baseline = adapter.build_orbitals_avas(ao_projector=_PROJECTOR, max_size=(2, 2))
    # The coupled virtual's marginal is ~0.0375 Ha; a 0.05 Ha guard rejects it.
    guarded = adapter.build_orbitals_correlation_avas(
        ao_projector=_PROJECTOR, max_size=(2, 2), min_marginal=0.05
    )
    info = adapter._avas_info

    assert info["correlation_fallback"] is True
    assert info["correlation_min_marginal"] == pytest.approx(0.05)
    np.testing.assert_allclose(info["correlation_attempted_marginals"], [-0.0375], rtol=1e-3)
    # Fallback is exactly the standard AVAS selection, not a MP2-rotated block.
    assert guarded.active.tolist() == baseline.active.tolist()
    assert guarded.inactive.tolist() == baseline.inactive.tolist()
    np.testing.assert_allclose(guarded.coeff, baseline.coeff, atol=1e-12)
    np.testing.assert_allclose(guarded.energy, baseline.energy, atol=1e-12)


def test_guard_threshold_is_configurable():
    adapter = _stub_adapter((-1.0, -0.2, -0.1, 0.5))
    # 0.02 Ha < 0.0375: the MP2 selection is kept.
    selected = adapter.build_orbitals_correlation_avas(
        ao_projector=_PROJECTOR, max_size=(2, 2), min_marginal=0.02
    )
    assert adapter._avas_info["correlation_fallback"] is False
    assert abs(selected.c_active[:, 1] @ np.eye(4)[:, 2]) == pytest.approx(1.0)
    # 0.05 Ha > 0.0375: the guard falls back.
    adapter.build_orbitals_correlation_avas(
        ao_projector=_PROJECTOR, max_size=(2, 2), min_marginal=0.05
    )
    assert adapter._avas_info["correlation_fallback"] is True


def test_guard_rejects_a_negative_threshold():
    adapter = _stub_adapter((-1.0, -0.2, -0.1, 0.5))
    with pytest.raises(ValueError, match="min_marginal"):
        adapter.build_orbitals_correlation_avas(
            ao_projector=_PROJECTOR, max_size=(2, 2), min_marginal=-0.01
        )


def test_workflow_exposes_the_guard_threshold():
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    field = EmbeddingWorkflow.model_fields["correlation_min_marginal"]
    assert field.default == pytest.approx(0.01)


def test_correlation_avas_preserves_S_orthonormality_and_the_occupied_span():
    adapter = _stub_adapter((-1.0, -0.2, -0.1, 0.5))
    canonical = adapter.build_orbitals()
    selected = adapter.build_orbitals_correlation_avas(ao_projector=_PROJECTOR, max_size=(2, 2))
    c = selected.coeff
    np.testing.assert_allclose(c.T @ c, np.eye(c.shape[1]), atol=1e-12)
    # The occupied determinant is unchanged; only the virtual block rotates.
    np.testing.assert_allclose(
        c[:, : selected.n_occ] @ c[:, : selected.n_occ].T,
        canonical.coeff[:, : canonical.n_occ] @ canonical.coeff[:, : canonical.n_occ].T,
        atol=1e-12,
    )
    assert selected.inactive.max(initial=-1) < selected.n_occ
    assert selected.active[selected.active < selected.n_occ].size == 1


def test_correlation_avas_requires_a_fixed_budget():
    adapter = _stub_adapter((-1.0, -0.2, -0.1, 0.5))
    with pytest.raises(ValueError, match="fixed active-space budget"):
        adapter.build_orbitals_correlation_avas(ao_projector=_PROJECTOR, max_size=None)


def test_correlation_avas_rejects_an_open_shell_subsystem():
    # Two alpha occupieds, one of them unpaired: build_orbitals reports n_occ_b=1.
    adapter = _stub_adapter((-1.0, -0.2, -0.1, 0.5), n_occ=2)
    adapter._n_occ_b_from_embasi = lambda _n_occ: 1
    with pytest.raises(NotImplementedError, match="closed-shell only"):
        adapter.build_orbitals_correlation_avas(ao_projector=_PROJECTOR, max_size=(2, 2))
