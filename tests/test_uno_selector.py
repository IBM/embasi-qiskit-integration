# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""``--selector uno`` (``build_orbitals_uno``) and the FCI spin target (``target_s2``).

The adapter stub mimics EmbASI with ``localisation="UNO-SPADE"``: one environment shared by
both spins (optionally twisted for beta, to mimic plain per-spin SPADE), and subsystem A's
occupied orbitals built directly -- identical per spin (closed shell), HOMO/LUMO mixed by
+/- theta (a broken-symmetry pair), or with extra alpha orbitals (a triplet).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy.linalg import expm

from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.projection_embedding_adapter import ProjectionEmbeddingAdapter
from embasi_qiskit_integration.selectors import (
    apc_active_space,
    apc_orbital_entropies,
    apc_pair_coefficients,
)
from embasi_qiskit_integration.solvers import FCISolver

NAO, N_ENV, MU = 14, 3, 1.0e6


class _FakeInts:
    def __init__(self, w):
        self.w = w

    def get_k(self, dm):
        return self.w @ dm @ self.w


def _stub(*, n_occ=5, thetas=(), n_somo=0, env_twist=0.0, seed=5):
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.standard_normal((NAO, NAO)))[0]
    a_space, env = q[:, : NAO - N_ENV], q[:, NAO - N_ENV :]
    gen = rng.standard_normal((NAO, NAO))
    env_b = np.linalg.qr(expm(env_twist * (gen - gen.T)) @ env)[0]

    def occupied(sign):
        c = a_space.copy()
        for k, theta in enumerate(thetas):
            homo, lumo = n_occ - 1 - k, n_occ + k
            ch, cl = c[:, homo].copy(), c[:, lumo].copy()
            c[:, homo] = np.cos(theta) * ch + sign * np.sin(theta) * cl
            c[:, lumo] = -sign * np.sin(theta) * ch + np.cos(theta) * cl
        return c

    c_alpha, c_beta = occupied(+1)[:, : n_occ + n_somo], occupied(-1)[:, :n_occ]
    eps = np.sort(rng.uniform(-2.0, 2.0, NAO - N_ENV))
    fock_a = a_space @ np.diag(eps) @ a_space.T
    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = True
    ad._s = np.eye(NAO)
    ad._dm_a = c_alpha @ c_alpha.T + c_beta @ c_beta.T
    ad._dm_a_relaxed = None
    ad._fock_relaxed_spin = None
    ad._p_b_spin = (MU * env @ env.T, MU * env_b @ env_b.T)
    ad._fock_spin = (fock_a + ad._p_b_spin[0], fock_a + ad._p_b_spin[1])
    w = rng.standard_normal((NAO, NAO))
    ad.ints = _FakeInts(0.05 * (w @ w.T))

    class _P:
        pass

    ad.p = _P()
    ad.p.mo_coeffs_A_LL = {(0, 0): c_alpha, (1, 0): c_beta}
    ad.p.mo_coeffs_B_LL = {(0, 0): env, (1, 0): env_b}
    return ad


def test_one_orbital_set_serves_both_spins_and_both_projectors():
    ad = _stub(thetas=(0.4,))
    alpha, beta = ad.build_orbitals_uno(max_size=(4, 4), fill="apc")
    np.testing.assert_array_equal(alpha.coeff, beta.coeff)
    np.testing.assert_array_equal(alpha.active, beta.active)
    c = alpha.coeff
    np.testing.assert_allclose(c.T @ c, np.eye(c.shape[1]), atol=1e-10)
    for pb in ad._p_b_spin:
        assert np.abs(alpha.c_active.T @ pb @ alpha.c_active).max() < 1e-6
    assert abs(ad._uno_info["electrons_lost"]) < 1e-10


def test_refuses_a_per_spin_partition_and_points_to_uno_spade():
    ad = _stub(thetas=(0.4,), env_twist=2e-3)
    with pytest.raises(ValueError, match="UNO-SPADE"):
        ad.build_orbitals_uno(max_size=(4, 4), fill="apc")


def test_broken_symmetry_pair_is_selected():
    theta = 0.4
    ad = _stub(thetas=(theta,))
    alpha, beta = ad.build_orbitals_uno(max_size=(4, 4), fill="apc")
    occ = ad._uno_info["occupations"]
    for n in (1 + np.cos(2 * theta), 1 - np.cos(2 * theta)):
        assert np.isclose(occ[alpha.active], n, atol=1e-10).any()
    assert alpha.n_active_electrons_spin[0] == beta.n_active_electrons_spin[0] == 2


def test_triplet_keeps_its_singly_occupied_unos():
    ad = _stub(n_occ=4, n_somo=2)
    alpha, beta = ad.build_orbitals_uno(max_size=(4, 4), fill="apc")
    occ = ad._uno_info["occupations"]
    somos = np.flatnonzero(np.isclose(occ, 1.0, atol=1e-8))
    assert len(somos) == 2 and set(somos) <= set(alpha.active.tolist())
    na, nb = alpha.n_occ - alpha.inactive.size, beta.n_occ - beta.inactive.size
    assert (na, nb) == (3, 1)


def test_closed_shell_falls_back_to_plain_apc():
    ad = _stub(n_occ=5)
    alpha, _ = ad.build_orbitals_uno(max_size=(4, 4), fill="apc")
    assert ad._uno_info["n_fractional"] == 0
    a_space = np.hstack([ad.p.mo_coeffs_A_LL[(0, 0)], alpha.coeff[:, 5:]])
    f = 0.5 * (ad._fock_spin[0] + ad._fock_spin[1])
    eps, u = np.linalg.eigh(a_space.T @ f @ a_space)
    c = a_space @ u
    kd = np.einsum("pi,pq,qi->i", c, ad.ints.get_k(2.0 * c[:, :5] @ c[:, :5].T), c)
    so, sv = apc_orbital_entropies(apc_pair_coefficients(eps[:5], eps[5:], kd[5:]))
    ref = apc_active_space(
        np.r_[np.full(5, 2), np.zeros(len(eps) - 5, int)], np.r_[so, sv], (4, 4), fixed=True
    )
    assert np.linalg.svd(c[:, ref].T @ alpha.c_active, compute_uv=False).min() > 1 - 1e-8


def _hund_model(norb=4, u=1.0, v=0.5, k=0.3):
    """Degenerate orbitals with ferromagnetic exchange: high spin lies lowest."""
    h1 = np.zeros((norb, norb))
    eri = np.zeros((norb,) * 4)
    for i in range(norb):
        eri[i, i, i, i] = u
        for j in range(norb):
            if i != j:
                eri[i, i, j, j] = v
                eri[i, j, j, i] = eri[i, j, i, j] = k
    return EmbeddedHamiltonian(h1=h1, h2=eri, e_core=0.0, nelec=(3, 1))


def test_fci_spin_target_avoids_the_high_spin_ground_state():
    from pyscf import fci

    ham = _hund_model()
    free = FCISolver().solve(ham)
    con = FCISolver(target_s2=2.0, nroots=6).solve(ham)
    s2_free = fci.spin_op.spin_square0(
        fci.direct_spin1.kernel(ham.h1, ham.h2, 4, (3, 1))[1], 4, (3, 1)
    )[0]
    assert s2_free == pytest.approx(6.0, abs=1e-6)  # unconstrained M_s = 1 ground: a quintet
    assert con.energy > free.energy
    assert "s2=2" in con.diagnostics["solver"]


def test_target_s2_needs_shared_orbitals(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    wf = EmbeddingWorkflow(
        solver="fci",
        target_s2=0.0,
        spin_downfold=True,
        unrestricted=True,
        selector="apc-concentric",
    )
    with pytest.raises(ValueError, match="--selector uno"):
        wf._build_solver()
    assert isinstance(
        EmbeddingWorkflow(
            solver="fci", target_s2=0.0, spin_downfold=True, unrestricted=True, selector="uno"
        )._build_solver(),
        FCISolver,
    )


def test_sqd_takes_the_spin_target_and_evolution_time(monkeypatch):
    """--target_s2 reaches SQD as spin_sq and --evolution_time as the circuit time; the
    shared-orbital requirement for a spin target holds for SQD as for FCI."""
    monkeypatch.setattr(sys, "argv", ["prog"])
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow
    from embasi_qiskit_integration.solvers import SQDSolver

    common = {
        "solver": "sqd",
        "sampler": "aer",
        "aer_method": "statevector",
        "spin": 2,
        "unrestricted": True,
        "spin_downfold": True,
    }
    solver = EmbeddingWorkflow(
        **common, selector="uno", target_s2=2.0, evolution_time=3.0
    )._build_solver()
    assert isinstance(solver, SQDSolver)
    assert solver.spin_sq == 2.0
    assert solver.evolution_time == [3.0]  # a scalar is a one-point sweep

    default = EmbeddingWorkflow(**common, selector="uno")._build_solver()
    assert default.spin_sq is None and default.evolution_time == [1.0]  # unchanged defaults
    assert default.num_groups == [15]

    # The sweep axes take lists from the command line, as the reference workflow's do.
    monkeypatch.setattr(
        sys, "argv", ["prog", "--evolution_time", "1,2,3", "--num_groups", "[10, 20]"]
    )
    swept = EmbeddingWorkflow(**common, selector="uno")._build_solver()
    assert swept.evolution_time == [1.0, 2.0, 3.0]
    assert swept.num_groups == [10, 20]

    with pytest.raises(ValueError, match="--selector uno"):
        EmbeddingWorkflow(**common, selector="apc-concentric", target_s2=2.0)._build_solver()


def test_fci_spin_target_works_for_a_spin_dependent_hamiltonian():
    """The UNO path's downfold carries (h1a, h1b): direct_uhf, which PySCF's fix_spin_
    refuses, so the penalty is applied by our own solver subclass."""
    base = _hund_model()
    h1b = base.h1 + 1e-9 * np.eye(4)
    ham = EmbeddedHamiltonian(
        h1=0.5 * (base.h1 + h1b), h1a=base.h1, h1b=h1b, h2=base.h2, e_core=0.0, nelec=(3, 1)
    )
    assert ham.is_spin_dependent
    con = FCISolver(target_s2=2.0, nroots=6).solve(ham)
    ref = FCISolver(target_s2=2.0, nroots=6).solve(base)
    assert con.energy == pytest.approx(ref.energy, abs=1e-7)


FRAG_AO = np.arange(6)  # the stub's "fragment": its first six (orthonormal) AOs


@pytest.mark.parametrize("fill", ["avas", "apc-fragment"])
def test_local_fills_keep_fractional_unos_and_share_orbitals(fill):
    """The fill only rotates the NON-fractional blocks: the broken-symmetry pair is kept
    exactly, and both spins still get one orbital set orthogonal to B."""
    theta = 0.4
    ad = _stub(thetas=(theta,))
    alpha, beta = ad.build_orbitals_uno(
        max_size=(4, 4), fill=fill, fragment_ao=FRAG_AO, valence_ao=FRAG_AO
    )
    np.testing.assert_array_equal(alpha.coeff, beta.coeff)
    occ = ad._uno_info["occupations"]
    for n in (1 + np.cos(2 * theta), 1 - np.cos(2 * theta)):
        assert np.isclose(occ[alpha.active], n, atol=1e-10).any()
    c = alpha.coeff
    np.testing.assert_allclose(c.T @ c, np.eye(c.shape[1]), atol=1e-10)
    for pb in ad._p_b_spin:
        assert np.abs(alpha.c_active.T @ pb @ alpha.c_active).max() < 1e-6


def test_avas_fills_with_the_most_fragment_like_orbitals():
    """Closed shell (no fractional UNO): every active orbital comes from the fill, and each
    is the most fragment-like one available in its block."""
    ad = _stub(n_occ=5)
    alpha, _ = ad.build_orbitals_uno(
        max_size=(4, 4), fill="avas", fragment_ao=FRAG_AO, valence_ao=FRAG_AO
    )
    c = alpha.coeff
    weight = (c[FRAG_AO] ** 2).sum(axis=0)  # S = I in the stub
    occ_blk, vir_blk = np.arange(5), np.arange(5, c.shape[1])
    act = set(alpha.active.tolist())
    for blk in (occ_blk, vir_blk):
        chosen = [i for i in blk if i in act]
        rest = [i for i in blk if i not in act]
        assert min(weight[chosen]) >= max(weight[rest]) - 1e-10


def test_fills_needing_ao_indices_say_so():
    ad = _stub(n_occ=5)
    with pytest.raises(ValueError, match="AO indices"):
        ad.build_orbitals_uno(max_size=(4, 4), fill="avas")


def test_noise_level_spin_dependence_is_averaged_for_a_spin_target():
    """Near-degenerate multiplets (the Hund model) plus a 1e-4 Ha h1a/h1b difference -- the
    SCF-noise level measured on a stretched nitrile -- must still give an exact spin state:
    the pair is averaged and solved spin-adapted."""

    base = _hund_model()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((4, 4))
    noise = 1e-4 * (noise + noise.T)
    ham = EmbeddedHamiltonian(
        h1=base.h1,
        h1a=base.h1 + noise / 2,
        h1b=base.h1 - noise / 2,
        h2=base.h2,
        e_core=0.0,
        nelec=(3, 1),
    )
    res = FCISolver(target_s2=2.0, nroots=6).solve(ham)
    assert "spin-averaged" in res.diagnostics["solver"]
    ref = FCISolver(target_s2=2.0, nroots=6).solve(base)
    assert res.energy == pytest.approx(ref.energy, abs=1e-8)
    # A genuine difference above spin_free_tol keeps the penalised unrestricted solver.
    big = EmbeddedHamiltonian(
        h1=base.h1,
        h1a=base.h1 + 0.1 * np.eye(4),
        h1b=base.h1 - 0.1 * np.eye(4),
        h2=base.h2,
        e_core=0.0,
        nelec=(3, 1),
    )
    assert "uhf" in FCISolver(target_s2=2.0, nroots=6).solve(big).diagnostics["solver"]


def test_uno_n_env_is_validated_before_embasi_runs(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow, build_adapter

    with pytest.raises(ValueError, match="UNO-SPADE"):
        build_adapter(EmbeddingWorkflow(uno_n_env=12), parallel=False)
    with pytest.raises(ValueError, match="give only one"):
        build_adapter(
            EmbeddingWorkflow(uno_n_env=12, a_nmos=7, localisation="UNO-SPADE"), parallel=False
        )


def test_avas_labels_select_the_requested_fragment_shells():
    from pyscf import gto

    from embasi_qiskit_integration.embedding import avas_valence_ao
    from embasi_qiskit_integration.selectors import fragment_ao_indices

    mol = gto.M(atom="Fe 0 0 0; O 0 0 1.6; H 0 0.8 2.1", basis="sto-3g", spin=3, verbose=0)
    frag = fragment_ao_indices(mol, [0, 1])  # Fe and O; H is environment
    labels = mol.ao_labels(fmt=False)
    d = avas_valence_ao(mol, frag, ["Fe 3d"])
    assert len(d) == 5 and all(labels[i][1] == "Fe" and labels[i][2] == "3d" for i in d)
    both = avas_valence_ao(mol, frag, ["Fe 3d", "O 2p", "H 1s"])  # H 1s is outside the fragment
    assert len(both) == 8
    default = avas_valence_ao(mol, frag)  # the fragment p shells
    assert all(labels[i][2].endswith("p") for i in default)
    with pytest.raises(ValueError, match="match no AO of the fragment"):
        avas_valence_ao(mol, frag, ["N 2p"])


def test_fci_spin_average_gives_exact_spin_for_a_spin_dependent_downfold():
    base = _hund_model()
    ham = EmbeddedHamiltonian(
        h1=base.h1,
        h1a=base.h1 + 0.05 * np.eye(4),
        h1b=base.h1 - 0.05 * np.eye(4),
        h2=base.h2,
        e_core=0.0,
        nelec=(3, 1),
    )
    res = FCISolver(target_s2=2.0, nroots=6, spin_average=True).solve(ham)
    assert "spin-averaged" in res.diagnostics["solver"]
    # A diagonal +/- shift averages to exactly the spin-free model.
    ref = FCISolver(target_s2=2.0, nroots=6).solve(base)
    assert res.energy == pytest.approx(ref.energy, abs=1e-8)


@pytest.mark.parametrize(
    ("spin", "target", "ok"),
    [
        (0, 0.0, True),
        (0, 2.0, True),
        (2, 2.0, True),
        (1, 0.75, True),
        (4, 6.0, True),
        (0, 1.0, False),
        (2, 0.0, False),
        (1, 2.0, False),
    ],
)
def test_target_s2_must_be_a_spin_state_reachable_from_the_sector(monkeypatch, spin, target, ok):
    monkeypatch.setattr(sys, "argv", ["prog"])
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    wf = EmbeddingWorkflow(
        solver="fci",
        target_s2=target,
        spin=spin,
        unrestricted=True,
        spin_downfold=True,
        selector="uno",
    )
    if ok:
        assert isinstance(wf._build_solver(), FCISolver)
    else:
        with pytest.raises(ValueError, match="S\\(S\\+1\\)|not reachable"):
            wf._build_solver()


def test_spin_penalty_applies_on_the_spin_dependent_path():
    """A small CI space is diagonalised directly by PySCF's pspace, bypassing contract_2e; the
    penalised UHF solver must still see the penalty (regression: it once matched plain direct_uhf)."""
    from pyscf import fci
    from embasi_qiskit_integration.solvers import _spin_penalised_uhf

    base = _hund_model()
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((4, 4))
    noise = 1e-4 * (noise + noise.T)
    solver = _spin_penalised_uhf(0.0)
    solver.nroots = 4
    _, civecs = solver.kernel((base.h1 + noise / 2, base.h1 - noise / 2), (base.h2,) * 3, 4, (2, 2))
    for c in civecs:
        assert fci.spin_op.spin_square0(c, 4, (2, 2))[0] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize(("nelec", "target", "scale"), [((2, 2), 0.0, 0.02), ((3, 1), 2.0, 0.2)])
def test_penalised_spin_target_reports_the_energy_of_its_state(nelec, target, scale):
    """On the penalised direct_uhf path each eigenvalue is <H> + shift * <(S^2 - t)^2>.  An
    orbital-dependent h1a/h1b split (a KS low level) breaks [H, S^2], so the kept root is
    slightly contaminated and that penalty is nonzero -- it must not reach the energy.  (A
    uniform +/- c*I shift only adds c*(Na - Nb) and still commutes with S^2, which is why the
    tests above could not see it.)  Checked against the energy rebuilt from the result's own
    RDMs, which describe exactly the returned state."""
    base = _hund_model()
    rng = np.random.default_rng(1)
    d = rng.standard_normal((4, 4))
    d = scale * (d + d.T)
    ham = EmbeddedHamiltonian(
        h1=base.h1,
        h1a=base.h1 + d / 2,
        h1b=base.h1 - d / 2,
        h2=base.h2,
        e_core=0.3,
        nelec=nelec,
    )
    res = FCISolver(target_s2=target, nroots=6).solve(ham)
    assert "uhf" in res.diagnostics["solver"]
    e_rdm = (
        ham.e_core
        + np.einsum("pq,pq->", ham.h1a, res.rdm1a)
        + np.einsum("pq,pq->", ham.h1b, res.rdm1b)
        + 0.5 * np.einsum("pqrs,pqrs->", ham.h2, res.rdm2)
    )
    assert res.energy == pytest.approx(e_rdm, abs=1e-8)
