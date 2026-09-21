# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Stage 2: a genuine spin-dependent one-body pair through the solvers.

The downfold can produce two different one-body operators, one per spin channel.
``pyscf.fci.direct_uhf`` is the only solver here that can consume them; SQD's
``diagonalize_fermionic_hamiltonian`` and the FCIDUMP format each take a single
one-body tensor, so those paths fall back to the spin-averaged ``h1`` and must say
so rather than silently presenting a restricted answer as an unrestricted one.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.selectors import somo_occupation_pattern
from embasi_qiskit_integration.solvers import FCISolver

NORB = 2
NELEC = (2, 1)  # a doublet: the sector that makes the two channels differ


def _ham(*, spin_dependent: bool) -> EmbeddedHamiltonian:
    h1a = np.array([[-1.2, 0.1], [0.1, -0.4]])
    h1b = np.array([[-1.0, 0.05], [0.05, -0.3]])
    h2 = np.zeros((NORB,) * 4)
    h2[0, 0, 0, 0], h2[1, 1, 1, 1] = 0.7, 0.6
    h2[0, 0, 1, 1] = h2[1, 1, 0, 0] = 0.3
    pair = {"h1a": h1a, "h1b": h1b} if spin_dependent else {}
    return EmbeddedHamiltonian(h1=0.5 * (h1a + h1b), h2=h2, e_core=0.0, nelec=NELEC, **pair)


def test_fci_dispatches_to_direct_uhf_on_a_spin_pair():
    """The pair must route to ``direct_uhf``, not the spin-averaged ``direct_spin1``.

    ``_ham`` carries no ``h2_spin``, so the tag records that the ERIs are still
    spin-free -- the one-body pair alone is enough to need ``direct_uhf``.
    """
    res = FCISolver().solve(_ham(spin_dependent=True))
    assert res.diagnostics["solver"] == "pyscf-fci-uhf-spinfree-eri"
    restricted = FCISolver().solve(_ham(spin_dependent=False))
    assert restricted.diagnostics["solver"] == "pyscf-fci"


def test_spin_dependent_solve_differs_from_the_averaged_one():
    """Averaging the channels is a real approximation, not a relabelling.

    If these agreed, the whole ``(h1a, h1b)`` contract change would buy nothing --
    so the gap is the test.
    """
    e_uhf = FCISolver().solve(_ham(spin_dependent=True)).energy
    e_avg = FCISolver().solve(_ham(spin_dependent=False)).energy
    assert abs(e_avg - e_uhf) > 1e-3
    # The averaged operator is variationally worse for this pair.
    assert e_uhf < e_avg


def test_spin_dependent_solve_preserves_the_sector_and_rdm_identities():
    """``direct_uhf`` has no ``make_rdm12``; the hand-assembled spin sum must be right.

    ``rdm1a + rdm1b == rdm1`` is enforced by ``SolverResult``'s own validator (so
    construction would already have failed), and the ``rdm2`` combination
    ``aa + bb + ab + ab^T`` is checked here against ``sum_pq rdm2[p,p,q,q] = N(N-1)``.
    """
    res = FCISolver().solve(_ham(spin_dependent=True))
    assert res.is_spin_resolved
    assert res.check_spin_sector(NELEC) == pytest.approx(0.0, abs=1e-9)
    n = sum(NELEC)
    assert float(np.einsum("ppqq->", res.rdm2)) == pytest.approx(n * (n - 1), abs=1e-8)
    assert float(np.trace(res.rdm1)) == pytest.approx(n, abs=1e-9)


def test_fcidump_warns_that_it_cannot_carry_the_pair(tmp_path):
    """The format has one integral block; the sector survives, the pair does not."""
    from embasi_qiskit_integration.hamiltonian import fcidump

    ham = _ham(spin_dependent=True)
    with pytest.warns(UserWarning, match="FCIDUMP format cannot represent"):
        fcidump.write(ham, tmp_path / "h.fcidump")

    back = fcidump.read(tmp_path / "h.fcidump")
    assert back.nelec == NELEC  # sector preserved
    assert not back.is_spin_dependent  # but the pair is gone, as warned
    np.testing.assert_allclose(back.h1, ham.h1, atol=1e-10)


def test_fcidump_does_not_warn_without_a_pair(tmp_path, recwarn):
    from embasi_qiskit_integration.hamiltonian import fcidump

    fcidump.write(_ham(spin_dependent=False), tmp_path / "h.fcidump")
    assert not [w for w in recwarn if "FCIDUMP format cannot represent" in str(w.message)]


# --------------------------------------------------------------------------- #
# per-spin downfold plumbing (no EmbASI needed)
# --------------------------------------------------------------------------- #
def _spin_stub(nao=6, n_occ_a=(3, 2)):
    """Adapter with per-spin state whose two channels are genuinely orthogonal.

    Mimics what EmbASI hands back on an open shell: each spin channel gets its own
    A/B split, so ``span(A_ispin)`` is S-orthogonal to ``span(B_ispin)`` but NOT to
    the other channel's environment -- the property that forces a per-spin downfold.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    rng = np.random.default_rng(11)
    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = True
    ad.mu = 1.0e6
    ad._s = np.eye(nao)
    ad._floor = np.inf

    mo, pb, fock = {}, [], []
    for ispin, n_a in enumerate(n_occ_a):
        q = np.linalg.qr(rng.standard_normal((nao, nao)))[0]
        c_a, c_b = q[:, :n_a], q[:, n_a:]
        mo[("A", ispin)], mo[("B", ispin)] = c_a, c_b
        # Level-shift projector onto THIS channel's environment.
        pb.append(ad.mu * (c_b @ c_b.T))
        f = rng.standard_normal((nao, nao))
        f = f + f.T
        # Make span(A) invariant-ish and push B up, as the real F_emb does.
        fock.append(c_a @ np.diag(np.arange(n_a) - 5.0) @ c_a.T + pb[-1])

    ad._p_b_spin = (pb[0], pb[1])
    ad._fock_spin = (fock[0], fock[1])
    ad._dm_a = np.eye(nao) * 0.5
    ad._mo = mo

    class _P:
        pass

    p = _P()
    p.mo_coeffs_A_LL = {(0, 0): mo[("A", 0)], (1, 0): mo[("A", 1)]}
    p.mo_coeffs_B_LL = {(0, 0): mo[("B", 0)], (1, 0): mo[("B", 1)]}
    ad.p = p
    return ad


def test_per_spin_orbitals_are_annihilated_by_their_own_projector():
    """The point of the per-spin path: each channel's own P_B must vanish on it.

    A spin-summed projector leaks by ~2e-02 on a real doublet (cross-spin mixing);
    a per-channel one leaks at round-off. This pins that the per-spin eigensolver
    keeps the two consistent.
    """
    ad = _spin_stub()
    alpha, beta = ad.build_orbitals_spin()
    for ispin, orb in ((0, alpha), (1, beta)):
        c = orb.c_active
        leak = float(np.abs(c.T @ ad._p_b_spin[ispin] @ c).max())
        assert leak < 1e-6, f"spin {ispin} leaks {leak:.2e} into its own environment"


def test_per_spin_orbitals_report_their_own_occupied_counts():
    ad = _spin_stub(n_occ_a=(3, 2))
    alpha, beta = ad.build_orbitals_spin()
    assert alpha.n_occ == 3
    assert beta.n_occ == 2
    # Each set describes ONE spin, so neither carries a beta count of its own.
    assert alpha.n_occ_b is None and beta.n_occ_b is None


def test_build_orbitals_spin_refuses_without_per_spin_state():
    """A restricted run has no pair to build from and must say so, not guess."""
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = False
    ad._fock_spin = None
    with pytest.raises(ValueError, match="no per-spin embedded Fock"):
        ad.build_orbitals_spin()


def test_rebuilding_the_fock_clears_the_pair_when_no_spin_state_remains():
    """With no per-spin ``v_emb``/``P_B`` to rebuild from, the pair must clear.

    A stale ``_fock_spin`` from a previous cycle would silently downfold new densities
    against old per-spin operators. ``restore_state`` lands here with spin-summed inputs
    only, so it must end up with ``None`` rather than yesterday's pair.
    """
    ad = _spin_stub()
    assert ad._fock_spin is not None
    ad._v_emb_embasi = np.zeros((6, 6))
    ad._p_b = np.zeros((6, 6))
    ad._dm_b = np.eye(6) * 0.5
    ad._p_b_spin = None  # as restore_state leaves it
    ad._v_emb_spin = None
    ad.p.A_LL = SimpleNamespace(
        hamiltonian_kinetic=np.zeros((6, 6)), hamiltonian_estat_plus_xc=np.zeros((6, 6))
    )
    ad.p.mu_val = None
    ad._validate_densities = lambda: None
    ad._assemble_fock_a_only()
    assert ad._fock_spin is None


def test_rebuilding_the_fock_keeps_the_pair_for_a_multicycle_loop():
    """The per-spin path must survive past cycle 1 of the outer loop.

    Cycle >= 2 goes through ``run_low_level_a_only`` -> ``_assemble_fock_a_only``. The
    per-spin ``v_emb``/``P_B`` are frozen for that step and ``A_LL``'s per-spin blocks
    have just been refreshed, so the pair must be *rebuilt*, not dropped -- dropping it
    made ``build_orbitals_spin`` raise "no per-spin embedded Fock available" on cycle 2.
    """
    ad = _spin_stub()
    nao = 6
    ad._v_emb_embasi = np.zeros((nao, nao))
    ad._p_b = np.zeros((nao, nao))
    ad._dm_b = np.eye(nao) * 0.5
    ad._v_emb_spin = (np.zeros((nao, nao)), np.zeros((nao, nao)))
    # A_LL now reports a genuine spin axis, as it does on a live unrestricted run.
    kin = np.stack([np.eye(nao), np.eye(nao)])
    estat = np.stack([np.eye(nao) * 2.0, np.eye(nao) * 3.0])
    ad.p.A_LL = SimpleNamespace(hamiltonian_kinetic=kin, hamiltonian_estat_plus_xc=estat)
    ad.p.mu_val = None
    ad._validate_densities = lambda: None

    ad._assemble_fock_a_only()
    assert ad._fock_spin is not None, "the pair must survive the A-only rebuild"
    # The two channels must stay distinct (estat differs by construction above).
    assert not np.allclose(ad._fock_spin[0], ad._fock_spin[1])


def test_occupation_pattern_is_rotation_invariant():
    """APC must ask the density which columns are singly occupied, not column position.

    The positional rule ("first ``n_occ`` columns are doubly occupied, the last
    ``n_occ - n_occ_b`` of them singly") holds only for canonical orbitals in energy
    order. ``build_orbitals_apc_concentric`` runs concentric localization first, which
    rotates within blocks and destroys that ordering. Measured on the real OH radical
    the projected occupation is ``[2, 2, 2, 1, 2, 0, 0, 0]`` -- the SOMO sits at index
    3, so the positional reading would hand APC the wrong orbital to protect.

    Here the same shape is built synthetically: a SOMO deliberately placed *below* a
    doubly-occupied column.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    nao = 5
    q = np.linalg.qr(np.random.default_rng(5).standard_normal((nao, nao)))[0]
    # A doublet: alpha occupies columns 0, 1, 2, 4; beta occupies 0, 1, 2.  Column 3 is
    # empty and column 4 is the SOMO -- so the singly-occupied column sits ABOVE an
    # empty one, which no positional rule can express.
    a_cols, b_cols = [0, 1, 2, 4], [0, 1, 2]
    dm_a = q[:, a_cols] @ q[:, a_cols].T
    dm_b = q[:, b_cols] @ q[:, b_cols].T

    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = True
    ad._s = np.eye(nao)
    ad._dm_a_spin = (dm_a, dm_b)

    # n_alpha = 4, n_beta = 3, consistent with the densities above.
    pattern = ad._occupation_pattern(q, nao, 4, 3)
    assert pattern.tolist() == [2, 2, 2, 0, 1]
    # The positional rule would have said [2, 2, 2, 1, 0] for (n_occ, n_occ_b) = (4, 3):
    # it protects column 3, which is actually EMPTY, and misses the real SOMO at 4.
    assert somo_occupation_pattern(nao, 4, 3).tolist() == [2, 2, 2, 1, 0]


def test_occupation_pattern_falls_back_without_a_spin_density():
    """No per-spin density -> the positional reading, unchanged from before."""
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = False
    ad._dm_a_spin = None
    pattern = ad._occupation_pattern(np.eye(4), 4, 3, 2)
    assert pattern.tolist() == [2, 2, 1, 0]


def test_occupation_pattern_needs_a_beta_count_for_the_fallback():
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = False
    ad._dm_a_spin = None
    with pytest.raises(ValueError, match="needs the beta count"):
        ad._occupation_pattern(np.eye(4), 4, 3, None)


# --------------------------------------------------------------------------- #
# spin-resolved two-body integrals
# --------------------------------------------------------------------------- #
def _eri_triple(norb=2):
    aa = np.zeros((norb,) * 4)
    aa[0, 0, 0, 0], aa[1, 1, 1, 1] = 0.7, 0.6
    aa[0, 0, 1, 1] = aa[1, 1, 0, 0] = 0.3
    bb = aa * 0.9
    ab = aa * 0.8
    return aa, ab, bb


def test_eri_triple_is_accepted_and_used_by_fci():
    """A genuine (aa, ab, bb) triple must reach ``direct_uhf``, not the spin-free h2."""
    aa, ab, bb = _eri_triple()
    h1a = np.array([[-1.2, 0.1], [0.1, -0.4]])
    h1b = np.array([[-1.0, 0.05], [0.05, -0.3]])
    ham = EmbeddedHamiltonian(
        h1=0.5 * (h1a + h1b),
        h2=aa,
        e_core=0.0,
        nelec=NELEC,
        h1a=h1a,
        h1b=h1b,
        h2_spin=(aa, ab, bb),
    )
    assert ham.has_spin_dependent_eri
    res = FCISolver().solve(ham)
    assert res.diagnostics["solver"] == "pyscf-fci-uhf"

    # Without the triple the same h1 pair goes through the spin-free ERIs and is
    # tagged differently -- and gives a different energy, so the triple matters.
    spin_free = EmbeddedHamiltonian(
        h1=0.5 * (h1a + h1b), h2=aa, e_core=0.0, nelec=NELEC, h1a=h1a, h1b=h1b
    )
    res_free = FCISolver().solve(spin_free)
    assert res_free.diagnostics["solver"] == "pyscf-fci-uhf-spinfree-eri"
    assert abs(res.energy - res_free.energy) > 1e-6


def test_eri_triple_requires_the_one_body_pair():
    """A spin-dependent h2 with a spin-averaged h1 is not a coherent Hamiltonian."""
    aa, ab, bb = _eri_triple()
    with pytest.raises(ValueError, match="h2_spin needs h1a/h1b"):
        EmbeddedHamiltonian(h1=np.eye(2), h2=aa, e_core=0.0, nelec=NELEC, h2_spin=(aa, ab, bb))


def test_eri_triple_shape_is_checked():
    aa, ab, bb = _eri_triple()
    h1a = np.eye(2)
    with pytest.raises(ValueError, match="h2_ab must have shape"):
        EmbeddedHamiltonian(
            h1=h1a,
            h2=aa,
            e_core=0.0,
            nelec=NELEC,
            h1a=h1a,
            h1b=h1a,
            h2_spin=(aa, np.zeros((3,) * 4), bb),
        )


def test_mixed_spin_eri_has_only_fourfold_symmetry():
    """``(aa|bb)`` is built from two different orbital sets.

    ``(pq|rs) == (rs|pq)`` fails there by construction, which is why the contract
    checks the mixed block separately from the same-spin ones -- applying the 8-fold
    test to it would warn on every correct open-shell Hamiltonian.
    """
    import pyscf

    from embasi_qiskit_integration.projection_embedding_adapter import PySCFIntegrals

    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.9; H 0 0 1.8", basis="sto-3g", spin=1)
    ints = PySCFIntegrals(mol.UKS(xc="HF"))
    rng = np.random.default_rng(1)
    nao = mol.nao
    mo_a = np.linalg.qr(rng.standard_normal((nao, nao)))[0][:, :3]
    mo_b = np.linalg.qr(rng.standard_normal((nao, nao)))[0][:, :3]

    ab = ints.eri_mo_mixed(mo_a, mo_b)
    assert np.allclose(ab, ab.transpose(1, 0, 2, 3))  # (pq|rs) == (qp|rs)
    assert np.allclose(ab, ab.transpose(0, 1, 3, 2))  # (pq|rs) == (pq|sr)
    assert not np.allclose(ab, ab.transpose(2, 3, 0, 1))  # but NOT (rs|pq)


# --------------------------------------------------------------------------- #
# per-spin density feedback
# --------------------------------------------------------------------------- #
def test_density_pair_is_wrapped_at_n_spin_two():
    """EmbASI reads ``[0,0]`` and ``[1,0]`` when ``n_spins == 2``; a pair must use that."""
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    dm_a, dm_b = np.eye(3), np.eye(3) * 0.5
    wrapped = ProjectionEmbeddingAdapter._as_spin_kpoint_pair(dm_a, dm_b)
    assert getattr(wrapped, "n_spins", None) == 2
    np.testing.assert_allclose(np.asarray(wrapped[0, 0]), dm_a)
    np.testing.assert_allclose(np.asarray(wrapped[1, 0]), dm_b)


def test_density_pair_shapes_must_match():
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    with pytest.raises(ValueError, match="share a shape"):
        ProjectionEmbeddingAdapter._as_spin_kpoint_pair(np.eye(3), np.eye(2))


def test_wrap_density_dispatches_on_tuple_vs_matrix():
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    assert ad._wrap_density(None) is None
    single = ad._wrap_density(np.eye(3))
    assert getattr(single, "n_spins", None) == 1
    pair = ad._wrap_density((np.eye(3), np.eye(3) * 0.5))
    assert getattr(pair, "n_spins", None) == 2
    with pytest.raises(ValueError, match="must be \\(alpha, beta\\)"):
        ad._wrap_density((np.eye(3),))


# --------------------------------------------------------------------------- #
# spin-resolved energy decomposition
# --------------------------------------------------------------------------- #
def test_energy_split_sums_to_the_reported_totals():
    """The per-spin breakdown must be an exact decomposition, not a second opinion.

    ``correction`` and ``projector_leak`` are both linear in the density, so splitting
    the density and contracting against the *same* operators is exact. An earlier
    attempt contracted each channel against its own ``v_emb_spin``, which does NOT sum
    to the spin-summed ``v_emb`` (``h_emb`` subtracts ``h_core``/``veff_ll`` once, so two
    channels subtract them twice -- ~33 Ha apart on a real doublet), and the "split" then
    disagreed with the total it claimed to decompose.
    """
    from embasi_qiskit_integration.contract import SolverResult
    from embasi_qiskit_integration.projection_embedding_adapter import (
        EmbeddedOrbitals,
        ProjectionEmbeddingAdapter,
    )

    nao, nact = 4, 2
    rng = np.random.default_rng(17)
    q = np.linalg.qr(rng.standard_normal((nao, nao)))[0]
    v_emb = rng.standard_normal((nao, nao))
    v_emb = v_emb + v_emb.T
    p_b = rng.standard_normal((nao, nao))
    p_b = p_b + p_b.T

    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = True
    ad.mu = 1.0e6
    ad._s = np.eye(nao)
    ad._dm_a_init = np.eye(nao) * 0.5
    ad._dm_a = np.eye(nao) * 0.5
    ad._v_emb_spin = None
    ad._p_b_spin = None
    ad._fock_spin = None
    # `v_emb` and `p_b` are read-only properties, so drive them through the state they
    # derive from: v_emb = (fock - veff_ll) - hcore - p_b, with hcore/veff_ll zeroed.
    ad._p_b = p_b
    ad._fock = v_emb + p_b
    ad._a_fragment_footing = lambda: (np.zeros((nao, nao)), 0.0)
    ad._low_level_energies = lambda: (-10.0, -4.0)

    class _Ints:
        def hcore(self):
            return np.zeros((nao, nao))

        def veff_ll(self, dm):
            return np.zeros((nao, nao))

        def energy_nuc(self):
            return 0.0

    ad.ints = _Ints()
    assert np.allclose(ad.v_emb, v_emb)  # the stub really does yield the intended v_emb

    orbitals = EmbeddedOrbitals(
        coeff=q,
        energy=np.arange(float(nao)),
        n_occ=2,
        inactive=np.array([], dtype=int),
        active=np.arange(nact),
    )
    ra = np.diag([1.0, 0.0])
    rb = np.diag([0.6, 0.1])
    result = SolverResult(energy=-3.0, rdm1=ra + rb, rdm1a=ra, rdm1b=rb)

    energy = ad.projection_energy(result, orbitals)
    assert energy.is_spin_resolved
    assert sum(energy.correction_spin) == pytest.approx(energy.correction, abs=1e-12)
    assert sum(energy.projector_leak_spin) == pytest.approx(energy.projector_leak, abs=1e-12)


def test_energy_split_is_absent_without_a_spin_resolved_result():
    from embasi_qiskit_integration.projection_embedding_adapter import ProjectionEnergy

    e = ProjectionEnergy(
        e_low_total=-1.0, e_low_A=-0.5, e_high_A=-0.4, correction=0.0, projector_leak=0.0
    )
    assert not e.is_spin_resolved
    assert e.correction_spin is None and e.projector_leak_spin is None


def test_v_emb_spin_is_not_an_additive_decomposition():
    """Documented trap: the per-channel potentials do not sum to ``v_emb``.

    Pinned so nobody "fixes" the energy split to use them.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    nao = 4
    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = True
    ad._s = np.eye(nao)
    ad._dm_a = np.eye(nao)
    hcore = np.eye(nao) * 3.0
    veff = np.eye(nao) * 2.0
    fock_a, fock_b = np.eye(nao) * 7.0, np.eye(nao) * 9.0
    pb_a, pb_b = np.eye(nao) * 0.5, np.eye(nao) * 0.25
    ad._fock_spin = (fock_a, fock_b)
    ad._p_b_spin = (pb_a, pb_b)

    class _Ints:
        def hcore(self):
            return hcore

        def veff_ll(self, dm):
            return veff

    ad.ints = _Ints()

    v_a, v_b = ad.v_emb_spin(0), ad.v_emb_spin(1)
    # Each channel subtracts hcore and veff once, so the sum double-counts them.
    summed_fock = fock_a + fock_b
    v_total_if_additive = summed_fock - veff - hcore - (pb_a + pb_b)
    assert not np.allclose(v_a + v_b, v_total_if_additive)


# --------------------------------------------------------------------------- #
# frozen core on the per-spin path
# --------------------------------------------------------------------------- #
def _frozen_core_stub(nao=6, n_occ_a=(3, 3)):
    """A ``_spin_stub`` with an integral backend whose ``veff_hf`` is density-linear.

    ``veff_hf`` has to actually depend on the density it is handed, or the frozen-core
    fold cannot be wrong in a way a test can see.  A linear map is enough: it makes the
    folded potential and ``e_core`` exact functions of ``dm_in``, so the assertions
    below compare closed forms rather than tolerances.

    Equal occupied counts per channel, because ``embedded_hamiltonian_spin`` requires a
    common active-space dimension.  The two channels still localise *different* orbitals
    (each gets its own QR draw), which is the property under test: their frozen cores
    overlap by only ~0.43 here, so a wrong core is plainly visible.
    """
    ad = _spin_stub(nao=nao, n_occ_a=n_occ_a)
    scale = np.diag(1.0 + np.arange(float(nao)))  # not proportional to the identity

    class _Ints:
        def veff_ll(self, dm):
            return np.zeros((nao, nao))

        def veff_hf(self, dm):
            return scale @ np.asarray(dm) @ scale

        def eri_mo(self, c):
            n = c.shape[1]
            return np.zeros((n,) * 4)

        def energy_nuc(self):
            return 0.0

    ad.ints = _Ints()
    ad._dm_a = np.eye(nao) * 0.5
    return ad


def test_frozen_core_on_the_spin_path_sums_both_channels():
    """The frozen core is ``c_in_a c_in_a^T + c_in_b c_in_b^T``, not ``2 c_in c_in^T``.

    Each spin channel freezes its *own* orbitals (SPADE partitions the spins
    separately), so doubling either channel's core is the restricted expression applied
    where it does not hold.  It has the right trace -- one electron per channel either
    way -- so no electron-count check catches it; only the matrix differs, and only
    where the two cores do.

    Regression: both the folded ``veff_hf`` and ``e_core`` used ``2 * c_in_a c_in_a^T``.
    On real doublets that is ~0.4 kcal/mol for a deep 1s core (the two cores overlap to
    0.999998) and ~40 kcal/mol once a frozen orbital is valence-like (~0.96), so the
    deep-core case would have hidden it indefinitely.
    """
    ad = _frozen_core_stub()
    alpha, beta = ad.build_orbitals_spin(n_frozen_occ=1)
    ham = ad.embedded_hamiltonian_spin((alpha, beta))

    c_in_a, c_in_b = alpha.c_inactive, beta.c_inactive
    # The two channels really do freeze different orbitals, or there is nothing to test.
    assert not np.allclose(c_in_a, c_in_b)

    dm_true = c_in_a @ c_in_a.T + c_in_b @ c_in_b.T
    dm_wrong = 2.0 * (c_in_a @ c_in_a.T)
    assert np.trace(dm_true) == pytest.approx(np.trace(dm_wrong), abs=1e-12)  # same count
    assert not np.allclose(dm_true, dm_wrong)  # ...different matrix

    veff_true = ad.ints.veff_hf(dm_true)
    fock_a = ad._fock_spin[0]

    def _h1(c, fock, veff):
        h = c.T @ (fock - ad.ints.veff_ll(ad._dm_a) + veff) @ c
        return 0.5 * (h + h.T)

    # Both channels see the SAME potential: veff is a functional of the *total* core
    # density, so alpha is screened by the beta core too.
    assert np.allclose(ham.h1a, _h1(alpha.c_active, fock_a, veff_true))
    assert np.allclose(ham.h1b, _h1(beta.c_active, ad._fock_spin[1], veff_true))
    # ...and NOT what the doubled single-channel core would have produced, nor what
    # giving each channel its own veff_hf(c_in_ispin) would (the other plausible wrong
    # fix -- also not a mean field).
    veff_wrong = ad.ints.veff_hf(dm_wrong)
    assert not np.allclose(ham.h1a, _h1(alpha.c_active, fock_a, veff_wrong))
    per_channel_a = ad.ints.veff_hf(c_in_a @ c_in_a.T)
    assert not np.allclose(per_channel_a, veff_true)
    assert not np.allclose(ham.h1a, _h1(alpha.c_active, fock_a, per_channel_a))

    e_core_true = np.einsum(
        "ij,ji->", dm_true, (fock_a - ad.ints.veff_ll(ad._dm_a)) + 0.5 * veff_true
    )
    assert ham.e_core == pytest.approx(float(e_core_true), abs=1e-10)
    e_core_wrong = np.einsum(
        "ij,ji->", dm_wrong, (fock_a - ad.ints.veff_ll(ad._dm_a)) + 0.5 * veff_wrong
    )
    assert abs(ham.e_core - float(e_core_wrong)) > 1e-8


def test_frozen_core_is_inert_at_zero_frozen_occupied():
    """At ``n_frozen_occ=0`` both the old and new forms vanish.

    This is why every pinned open-shell number is unaffected by the fix: the default
    path never built a frozen-core density at all.
    """
    ad = _frozen_core_stub()
    alpha, beta = ad.build_orbitals_spin(n_frozen_occ=0)
    assert alpha.c_inactive.shape[1] == 0 and beta.c_inactive.shape[1] == 0
    ham = ad.embedded_hamiltonian_spin((alpha, beta))

    fock_a = ad._fock_spin[0]
    h_emb_a = fock_a - ad.ints.veff_ll(ad._dm_a)
    h1a_bare = alpha.c_active.T @ h_emb_a @ alpha.c_active
    assert np.allclose(ham.h1a, 0.5 * (h1a_bare + h1a_bare.T))
    assert ham.e_core == pytest.approx(ad.ints.energy_nuc(), abs=1e-12)


def test_spin_downfold_refuses_active_orbitals_that_leak_into_b():
    """The per-spin projector guard must actually fire on a leaking active space.

    ``embedded_hamiltonian_spin`` refuses a downfold whose active orbitals are not
    annihilated by their own channel's ``P_B``: a level-shifted environment orbital
    carries ``mu`` (1e6 here), so leaking one in swamps the Hamiltonian rather than
    perturbing it.  Healthy input never trips the guard, so without this the branch was
    unexercised -- disabling it entirely left the whole suite green.

    Feeding the *other* channel's environment is the realistic failure: that is exactly
    the cross-spin mixing the per-spin path exists to avoid (a spin-summed ``P_B`` leaks
    ~2e-02 on a real doublet), so this is the guard's actual job, not an invented one.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import EmbeddedOrbitals

    ad = _frozen_core_stub()
    alpha, beta = ad.build_orbitals_spin(n_frozen_occ=0)

    # Splice one beta-environment column into alpha's active space, keeping the active
    # dimension equal so the downfold reaches the leak check rather than the size check.
    c_b_env = ad._mo[("B", 1)][:, :1]
    bad_coeff = alpha.coeff.copy()
    bad_coeff[:, alpha.active[0]] = c_b_env[:, 0]
    leaky = EmbeddedOrbitals(
        coeff=bad_coeff,
        energy=alpha.energy,
        n_occ=alpha.n_occ,
        inactive=alpha.inactive,
        active=alpha.active,
    )
    # The splice really does leak through alpha's own projector, or the guard is not
    # what is being tested.
    c_act = leaky.c_active
    assert float(np.abs(c_act.T @ ad._p_b_spin[0] @ c_act).max()) > 1e-6

    with pytest.raises(ValueError, match="leak into subsystem B per spin"):
        ad.embedded_hamiltonian_spin((leaky, beta))
