# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Per-spin ``apc-concentric``: CL per channel, alpha/beta slot pairing, one APC selection.

Stub-based (no EmbASI): each channel gets its own A/B split, and beta's partition is a
small rotation of alpha's, as on a real open shell where SPADE partitions the spins
separately but nearly alike.  DRAFT tests for ``build_orbitals_apc_concentric_spin``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from embasi_qiskit_integration.projection_embedding_adapter import (
    EmbeddedOrbitals,
    ProjectionEmbeddingAdapter,
)

NAO = 10
N_ENV = 2
FRAG = np.arange(4)


class _FakeInts:
    """A linear, symmetric stand-in for the exchange build: ``K[dm] = W dm W``."""

    def __init__(self, w):
        self.w = w

    def get_k(self, dm):
        return self.w @ dm @ self.w


def _stub(n_occ=(4, 4), polarise=0.05, seed=3):
    """Adapter with per-spin state; ``polarise=0`` and equal counts give alpha == beta."""
    rng = np.random.default_rng(seed)
    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = True
    ad.mu = 1.0e6
    ad._s = np.eye(NAO)
    ad._floor = np.inf

    q_a = np.linalg.qr(rng.standard_normal((NAO, NAO)))[0]
    gen = rng.standard_normal((NAO, NAO))
    q_b = q_a @ expm(polarise * (gen - gen.T))  # beta's partition: a small rotation
    f_base = rng.standard_normal((NAO, NAO))
    f_base = f_base + f_base.T + np.diag(np.arange(NAO, dtype=float))
    f_pol = rng.standard_normal((NAO, NAO))
    f_pol = polarise * (f_pol + f_pol.T)

    mo_a, mo_b, pb, fock = {}, {}, [], []
    for ispin, (q, n) in enumerate(((q_a, n_occ[0]), (q_b, n_occ[1]))):
        c_env = q[:, NAO - N_ENV :]
        proj_a = np.eye(NAO) - c_env @ c_env.T
        pb.append(ad.mu * (c_env @ c_env.T))
        f = proj_a @ (f_base + (f_pol if ispin else -f_pol)) @ proj_a + pb[-1]
        fock.append(f)
        # mo_coeffs_A_LL only supplies each channel's occupied COUNT (and S-orthonormality).
        mo_a[(ispin, 0)] = np.linalg.eigh(f)[1][:, :n]
        mo_b[(ispin, 0)] = c_env

    ad._p_b_spin = (pb[0], pb[1])
    ad._fock_spin = (fock[0], fock[1])
    w = rng.standard_normal((NAO, NAO))
    ad.ints = _FakeInts(0.1 * (w @ w.T))

    class _P:
        pass

    ad.p = _P()
    ad.p.mo_coeffs_A_LL = mo_a
    ad.p.mo_coeffs_B_LL = mo_b
    return ad


def _build(ad, max_size=(4, 4), fixed=True, n_shells=1):
    return ad.build_orbitals_apc_concentric_spin(
        fragment_ao=FRAG, n_shells=n_shells, max_size=max_size, fixed=fixed
    )


def test_closed_shell_limit_matches_the_restricted_apc_selection():
    """alpha == beta: every pairing is the identity, so the restricted recipe must win.

    The restricted ``build_orbitals_apc_concentric`` is run for real; only its
    ``build_orbitals`` call is redirected onto the same per-spin eigenbasis, so the two
    rankings see identical candidates and any difference is the ranking itself.
    """
    ad = _stub(n_occ=(4, 4), polarise=0.0)
    alpha, beta = _build(ad)

    class _Restricted(ProjectionEmbeddingAdapter):
        _fock_arr = property(lambda self: self._fock_spin[0])

        def build_orbitals(self, *, virtual_localizer, **_kw):
            eps, c = self._eigh_subsystem_a_spin(0)
            n_occ = self.p.mo_coeffs_A_LL[(0, 0)].shape[1]
            c_loc, _ = virtual_localizer(c, n_occ)
            k = int(virtual_localizer.max_virtual)
            return EmbeddedOrbitals(
                coeff=c_loc,
                energy=eps,
                n_occ=n_occ,
                inactive=np.empty(0, int),
                active=np.arange(n_occ + k),
            )

    ref_ad = _stub(n_occ=(4, 4), polarise=0.0)
    ref_ad.__class__ = _Restricted
    ref = ref_ad.build_orbitals_apc_concentric(
        fragment_ao=FRAG, n_shells=1, max_size=(4, 4), fixed=True
    )

    np.testing.assert_array_equal(alpha.active, ref.active)
    np.testing.assert_array_equal(alpha.inactive, ref.inactive)
    np.testing.assert_allclose(alpha.c_active, ref.c_active, atol=1e-10)
    np.testing.assert_allclose(beta.c_active, alpha.c_active, atol=1e-10)
    np.testing.assert_allclose(ad._apc_spin_pair_overlap, 1.0, atol=1e-10)


@pytest.mark.parametrize("n_occ", [(4, 4), (5, 4), (5, 3)])
def test_both_channels_share_one_selection_and_keep_the_spin(n_occ):
    ad = _stub(n_occ=n_occ)
    nelec = 4 + (n_occ[0] - n_occ[1])  # keep nelec - A_spin even
    alpha, beta = _build(ad, max_size=(nelec, 5))

    np.testing.assert_array_equal(alpha.active, beta.active)
    np.testing.assert_array_equal(alpha.inactive, beta.inactive)
    na, nb = alpha.n_active_electrons_spin[0], beta.n_active_electrons_spin[0]
    assert na - nb == n_occ[0] - n_occ[1], "the SOMOs (A_spin) must survive the selection"
    assert na + nb == nelec
    assert alpha.n_active_orbitals == beta.n_active_orbitals == 5


def test_pairing_preserves_each_channels_density_span_and_partition():
    """The slot rotations must be invisible to the physics: same occupied span per
    channel, S-orthonormal columns, and each channel annihilated by its OWN P_B."""
    ad = _stub(n_occ=(5, 3))
    pair = _build(ad, max_size=(4, 5))
    for ispin, orb in enumerate(pair):
        c = orb.coeff
        np.testing.assert_allclose(c.T @ c, np.eye(c.shape[1]), atol=1e-8)
        c_ref = ad._eigh_subsystem_a_spin(ispin)[1][:, : orb.n_occ]
        c_occ = c[:, : orb.n_occ]
        np.testing.assert_allclose(c_occ @ c_occ.T, c_ref @ c_ref.T, atol=1e-8)
        leak = np.abs(orb.c_active.T @ ad._p_b_spin[ispin] @ orb.c_active).max()
        assert leak < 1e-6


def test_parity_of_a_fixed_budget_must_match_the_unpaired_count():
    ad = _stub(n_occ=(5, 4))  # A_spin = 1 needs an odd nelec
    with pytest.raises(ValueError, match="nelec - A_spin must be even"):
        _build(ad, max_size=(4, 4))


def test_beta_majority_is_refused():
    ad = _stub(n_occ=(3, 4))
    with pytest.raises(ValueError, match="more beta"):
        _build(ad)


def test_weakly_paired_slots_warn():
    ad = _stub(n_occ=(5, 4), polarise=0.6)
    with pytest.warns(UserWarning, match="pairing is weak"):
        _build(ad, max_size=(5, 5))
