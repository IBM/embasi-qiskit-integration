# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""``--selector avas`` (``build_orbitals_avas`` / ``avas_ao_projector``).

The adapter stub treats a whole molecule as subsystem A (an empty environment), with the
molecule's own mean-field Fock as ``F_emb``, so the selection can be checked against
PySCF's ``mcscf.avas`` on the same orbitals: same active-space size and electron
count, and the same core and active spans.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyscf
import pytest
from pyscf.mcscf import avas

from embasi_qiskit_integration.embedding import build_selector
from embasi_qiskit_integration.projection_embedding_adapter import ProjectionEmbeddingAdapter
from embasi_qiskit_integration.selectors import avas_ao_projector

FORMALDEHYDE = "C 0 0 0; O 0 0 1.21; H 0 0.94 -0.59; H 0 -0.94 -0.59"
METHYLENE = "C 0 0 0.1; H 0 0.99 -0.5; H 0 -0.99 -0.5"


def _stub(mf, a_spin=0):
    """An adapter whose subsystem A is all of ``mf.mol``, with ``mf``'s Fock as F_emb."""
    mol = mf.mol
    overlap = mol.intor_symmetric("int1e_ovlp")
    fock = mf.get_fock()
    n_alpha = mol.nelec[0]

    class _Stub(ProjectionEmbeddingAdapter):
        _s_arr = property(lambda self: overlap)
        _fock_arr = property(lambda self: fock)
        mo_a_ll = property(lambda self: np.zeros((mol.nao, n_alpha)))
        mo_b_ll = property(lambda self: np.zeros((mol.nao, 0)))  # no environment

        def _validate_span(self, coeff):  # needs the real embedding
            return None

    adapter = object.__new__(_Stub)
    adapter.p = SimpleNamespace(A_spin=a_spin)
    adapter.ints = SimpleNamespace(mol=mol)
    return adapter, overlap


def _span(c):
    """AO-basis projector onto the span of S-orthonormal columns ``c``."""
    return c @ c.T


# The stub re-diagonalizes ``mf.get_fock()`` while PySCF's AVAS rotates ``mf.mo_coeff``;
# the two orbital sets agree only to SCF convergence (measured ~2e-8 at conv_tol=1e-10).
SPAN_TOL = 1e-6


@pytest.fixture(scope="module")
def formaldehyde_rhf():
    mol = pyscf.M(atom=FORMALDEHYDE, basis="6-31g", verbose=0)
    return mol.RHF().run(conv_tol=1e-10)


@pytest.fixture(scope="module")
def methylene_rohf():
    mol = pyscf.M(atom=METHYLENE, basis="6-31g", spin=2, verbose=0)
    return mol.ROHF().run(conv_tol=1e-10)


@pytest.mark.parametrize(
    "labels,threshold", [(["C 2p", "O 2p"], 0.2), (["O 2p"], 0.2), (["C 2p", "O 2p"], 0.5)]
)
def test_matches_pyscf_avas_closed_shell(formaldehyde_rhf, labels, threshold):
    mf = formaldehyde_rhf
    ad, _ = _stub(mf)
    orb = ad.build_orbitals_avas(
        ao_projector=avas_ao_projector(mf.mol, labels), threshold=threshold
    )
    ncas, nelecas, mo = avas.kernel(mf, labels, threshold=threshold, minao="minao")

    assert orb.n_active_orbitals == ncas
    assert orb.n_active_electrons == nelecas
    ncore = (mf.mol.nelectron - nelecas) // 2
    np.testing.assert_allclose(
        _span(orb.c_active), _span(mo[:, ncore : ncore + ncas]), atol=SPAN_TOL
    )
    np.testing.assert_allclose(_span(orb.c_inactive), _span(mo[:, :ncore]), atol=SPAN_TOL)


def test_matches_pyscf_avas_restricted_open_shell(methylene_rohf):
    mf = methylene_rohf
    ad, _ = _stub(mf, a_spin=2)
    orb = ad.build_orbitals_avas(ao_projector=avas_ao_projector(mf.mol, ["C 2p", "C 2s"]))
    ncas, nelecas, mo = avas.kernel(mf, ["C 2p", "C 2s"], minao="minao", openshell_option=3)

    assert orb.is_open_shell
    assert orb.n_active_orbitals == ncas
    assert orb.n_active_electrons == nelecas
    ncore = (mf.mol.nelectron - nelecas) // 2
    np.testing.assert_allclose(
        _span(orb.c_active), _span(mo[:, ncore : ncore + ncas]), atol=SPAN_TOL
    )
    # Both singly occupied orbitals are active, whatever their weight.
    assert set(range(orb.n_occ_b, orb.n_occ)) <= set(orb.active.tolist())
    assert ad._avas_info["n_somo"] == 2


def test_rotations_keep_the_reference(formaldehyde_rhf):
    mf = formaldehyde_rhf
    ad, s = _stub(mf)
    canonical = ad.build_orbitals()
    orb = ad.build_orbitals_avas(ao_projector=avas_ao_projector(mf.mol, ["C 2p", "O 2p"]))
    c = orb.coeff
    np.testing.assert_allclose(c.T @ s @ c, np.eye(c.shape[1]), atol=1e-10)
    # Same occupied space (the determinant) before and after the block rotations.
    occ = slice(0, orb.n_occ)
    np.testing.assert_allclose(_span(c[:, occ]), _span(canonical.coeff[:, occ]), atol=1e-10)
    # Semi-canonical groups carry real Fock eigenvalues.
    assert np.all(np.isfinite(orb.energy))
    np.testing.assert_allclose(np.diag(c.T @ mf.get_fock() @ c), orb.energy, atol=1e-10)
    # Inactive columns are occupied, and the active occupied ones hold the active electrons.
    assert orb.inactive.max(initial=-1) < orb.n_occ
    assert orb.active[orb.active < orb.n_occ].size == orb.n_active_electrons // 2


def test_frozen_core_matches_pyscf_ncore(formaldehyde_rhf):
    mf = formaldehyde_rhf
    ad, _ = _stub(mf)
    labels = ["C 2p", "O 2p"]
    canonical = ad.build_orbitals()
    frozen = ad.build_orbitals_avas(ao_projector=avas_ao_projector(mf.mol, labels), n_frozen_occ=2)
    ncas, nelecas, mo = avas.kernel(mf, labels, minao="minao", ncore=2)

    # The frozen core is the canonical 1s pair, untouched by the projection...
    assert frozen.inactive[:2].tolist() == [0, 1]
    np.testing.assert_allclose(frozen.coeff[:, :2], canonical.coeff[:, :2], atol=1e-12)
    # ...and is excluded from it, exactly as PySCF's ncore.
    assert (frozen.n_active_orbitals, frozen.n_active_electrons) == (ncas, nelecas)
    ncore = (mf.mol.nelectron - nelecas) // 2
    np.testing.assert_allclose(
        _span(frozen.c_active), _span(mo[:, ncore : ncore + ncas]), atol=SPAN_TOL
    )
    with pytest.raises(ValueError, match="n_frozen_occ"):
        ad.build_orbitals_avas(ao_projector=avas_ao_projector(mf.mol, labels), n_frozen_occ=99)


def test_max_size_matching_the_threshold_cut_selects_the_same_space(formaldehyde_rhf):
    mf = formaldehyde_rhf
    ad, _ = _stub(mf)
    proj = avas_ao_projector(mf.mol, ["C 2p", "O 2p"])
    by_threshold = ad.build_orbitals_avas(ao_projector=proj)
    size = (by_threshold.n_active_electrons, by_threshold.n_active_orbitals)
    by_budget = ad.build_orbitals_avas(ao_projector=proj, max_size=size, threshold=0.99)

    assert by_budget.active.tolist() == by_threshold.active.tolist()
    np.testing.assert_allclose(_span(by_budget.c_active), _span(by_threshold.c_active), atol=1e-12)
    assert ad._avas_info["max_size"] == size


def test_max_size_keeps_the_highest_weight_orbitals(formaldehyde_rhf):
    mf = formaldehyde_rhf
    ad, _ = _stub(mf)
    proj = avas_ao_projector(mf.mol, ["C 2p", "O 2p"])
    full = ad.build_orbitals_avas(ao_projector=proj, threshold=0.0)
    w_occ, w_virt = ad._avas_info["occ_weights"], ad._avas_info["virt_weights"]

    small = ad.build_orbitals_avas(ao_projector=proj, max_size=(4, 3))
    assert (small.n_active_electrons, small.n_active_orbitals) == (4, 3)
    assert (ad._avas_info["n_active_occ"], ad._avas_info["n_active_virt"]) == (2, 1)
    # The kept orbitals span exactly the top-weight eigenvectors of each block.
    occ_top = w_occ[-2:]
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(small.c_active[:, :2].T @ proj @ small.c_active[:, :2])),
        np.sort(occ_top),
        atol=1e-10,
    )
    virt = small.c_active[:, 2:]
    np.testing.assert_allclose(np.linalg.eigvalsh(virt.T @ proj @ virt), w_virt[:1], atol=1e-10)
    assert full.n_active_orbitals > small.n_active_orbitals


def test_max_size_always_keeps_the_singly_occupied_orbitals(methylene_rohf):
    mf = methylene_rohf
    ad, _ = _stub(mf, a_spin=2)
    proj = avas_ao_projector(mf.mol, ["C 2p", "C 2s"])
    orb = ad.build_orbitals_avas(ao_projector=proj, max_size=(4, 3))
    assert orb.n_active_electrons_spin == (3, 1)
    assert set(range(orb.n_occ_b, orb.n_occ)) <= set(orb.active.tolist())

    # Two unpaired electrons: nelec must leave an even count for the paired orbitals.
    with pytest.raises(ValueError, match="unpaired"):
        ad.build_orbitals_avas(ao_projector=proj, max_size=(3, 3))


@pytest.mark.parametrize("size,match", [((30, 16), "doubly occupied"), ((4, 40), "virtual")])
def test_max_size_that_cannot_be_filled_is_refused(formaldehyde_rhf, size, match):
    mf = formaldehyde_rhf
    ad, _ = _stub(mf)
    with pytest.raises(ValueError, match=match):
        ad.build_orbitals_avas(
            ao_projector=avas_ao_projector(mf.mol, ["O 2p"]), max_size=size, n_frozen_occ=2
        )


def test_threshold_too_high_is_refused(formaldehyde_rhf):
    mf = formaldehyde_rhf
    ad, _ = _stub(mf)
    with pytest.raises(ValueError, match="no orbital"):
        ad.build_orbitals_avas(ao_projector=avas_ao_projector(mf.mol, ["O 2p"]), threshold=1.1)


def test_projector_is_restricted_to_the_fragment(formaldehyde_rhf):
    mol = formaldehyde_rhf.mol
    both = avas_ao_projector(mol, ["C 2p", "O 2p"])
    o_only = avas_ao_projector(mol, ["C 2p", "O 2p"], atoms=[1])
    np.testing.assert_allclose(o_only, avas_ao_projector(mol, ["O 2p"]), atol=1e-12)
    s = mol.intor_symmetric("int1e_ovlp")
    # Rank = number of target AOs: 6 (C and O 2p) vs 3 (O 2p).
    assert np.linalg.matrix_rank(np.linalg.solve(s, both), tol=1e-8) == 6
    assert np.linalg.matrix_rank(np.linalg.solve(s, o_only), tol=1e-8) == 3

    with pytest.raises(ValueError, match="active fragment; available shells: .*O 2p"):
        avas_ao_projector(mol, ["H 1s"], atoms=[1])


# ----- workflow wiring (build_selector) -------------------------------------- #


def _cfg(**kw):
    base = {
        "selector": "avas",
        "avas_ao_labels": ["O 2p"],
        "avas_threshold": 0.2,
        "avas_minao": "minao",
        "avas_max_size": None,
        "n_frozen_occ": 0,
        "active_atoms": [1, 2],
    }
    return SimpleNamespace(**{**base, **kw})


def test_build_selector_returns_an_avas_builder(formaldehyde_rhf):
    mf = formaldehyde_rhf
    ad, _ = _stub(mf)
    # The workflow puts the active atoms first, so the fragment is atoms 0 and 1 (C, O).
    selector, localizer, builder = build_selector(_cfg(), ad)
    assert selector is None and localizer is None and builder is not None

    orb = builder(ad)
    expected = ad.build_orbitals_avas(
        ao_projector=avas_ao_projector(mf.mol, ["O 2p"], atoms=[0, 1])
    )
    assert orb.active.tolist() == expected.active.tolist()

    with pytest.raises(ValueError, match="needs --avas_max_size"):
        builder(ad, spin=True)

    _, _, budget_builder = build_selector(_cfg(avas_max_size=(4, 3)), ad)
    budgeted = budget_builder(ad)
    assert (budgeted.n_active_electrons, budgeted.n_active_orbitals) == (4, 3)


def test_build_selector_requires_labels(formaldehyde_rhf):
    ad, _ = _stub(formaldehyde_rhf)
    with pytest.raises(ValueError, match="--avas_ao_labels"):
        build_selector(_cfg(avas_ao_labels=None), ad)


# ----- per-spin (build_orbitals_avas_spin): channels selected independently ---- #


def _spin_stub(mol, focks, c_occ):
    """A per-spin adapter over all of ``mol``: channel ``s`` has Fock ``focks[s]`` and
    occupied orbitals ``c_occ[s]``, with an empty environment in both channels."""
    ad = object.__new__(ProjectionEmbeddingAdapter)
    ad.unrestricted = True
    ad._s = mol.intor_symmetric("int1e_ovlp")
    ad._fock_spin = tuple(focks)
    ad._fock_relaxed_spin = None
    env = np.zeros((mol.nao, 0))
    ad.p = SimpleNamespace(
        mo_coeffs_A_LL={(0, 0): c_occ[0], (1, 0): c_occ[1]},
        mo_coeffs_B_LL={(0, 0): env, (1, 0): env},
    )
    ad.ints = SimpleNamespace(mol=mol)
    return ad


@pytest.fixture(scope="module")
def methylene_uhf():
    mol = pyscf.M(atom=METHYLENE, basis="6-31g", spin=2, verbose=0)
    return mol.UHF().run(conv_tol=1e-10)


def _uhf_spin_stub(mf):
    focks = mf.get_fock()
    occ = [mf.mo_coeff[s][:, mf.mo_occ[s] > 0] for s in (0, 1)]
    return _spin_stub(mf.mol, focks, occ)


def test_spin_closed_shell_limit_matches_restricted(formaldehyde_rhf):
    mf = formaldehyde_rhf
    proj = avas_ao_projector(mf.mol, ["C 2p", "O 2p"])
    restricted = _stub(mf)[0].build_orbitals_avas(ao_projector=proj, max_size=(8, 6))

    fock = mf.get_fock()
    c_occ = mf.mo_coeff[:, mf.mo_occ > 0]
    ad = _spin_stub(mf.mol, (fock, fock), (c_occ, c_occ))
    alpha, beta = ad.build_orbitals_avas_spin(ao_projector=proj, max_size=(8, 6))

    for orb in (alpha, beta):
        assert orb.n_active_orbitals == 6
        assert orb.n_occ - orb.inactive.size == 4
        np.testing.assert_allclose(_span(orb.c_active), _span(restricted.c_active), atol=1e-8)
        np.testing.assert_allclose(_span(orb.c_inactive), _span(restricted.c_inactive), atol=1e-8)
    np.testing.assert_allclose(ad._avas_info["active_cosines"], 1.0, atol=1e-8)
    assert ad._avas_info["core_spin"] == 0


def test_spin_channels_are_selected_independently(methylene_uhf):
    mf = methylene_uhf
    ad = _uhf_spin_stub(mf)
    proj = avas_ao_projector(mf.mol, ["C 2p", "C 2s", "H 1s"])
    alpha, beta = ad.build_orbitals_avas_spin(ao_projector=proj, max_size=(4, 5))

    # (nelec, norb) puts subsystem A's whole n_alpha - n_beta = 2 in the active space.
    assert (alpha.n_occ, beta.n_occ) == (5, 3)
    assert (alpha.n_occ - alpha.inactive.size, beta.n_occ - beta.inactive.size) == (3, 1)
    assert alpha.n_active_orbitals == beta.n_active_orbitals == 5
    assert ad._avas_info["core_spin"] == 0

    # Each channel keeps its OWN highest-weight occupied and virtual orbitals: the
    # selection depends only on that channel's orbitals and projector.
    s = ad._s
    for ispin, orb in ((0, alpha), (1, beta)):
        _, c = ad._eigh_subsystem_a_spin(ispin)
        n_occ, n_act_occ = orb.n_occ, orb.n_occ - orb.inactive.size
        for block, n_keep in ((c[:, :n_occ], n_act_occ), (c[:, n_occ:], 5 - n_act_occ)):
            w, u = np.linalg.eigh(block.T @ proj @ block)
            top = block @ u[:, np.argsort(w)[::-1][:n_keep]]
            kept = orb.c_active @ orb.c_active.T @ s @ top
            np.testing.assert_allclose(kept, top, atol=1e-8)  # top lies in the active span
        c_act = orb.c_active
        np.testing.assert_allclose(c_act.T @ s @ c_act, np.eye(5), atol=1e-10)

    # The two spins' active spaces are close for this UHF, but not identical.
    cos = ad._avas_info["active_cosines"]
    assert cos.min() < 1 - 1e-6 and cos.min() > 0.9


def test_spin_explicit_split_and_spin_polarised_core(methylene_uhf):
    ad = _uhf_spin_stub(methylene_uhf)
    proj = avas_ao_projector(methylene_uhf.mol, ["C 2p", "C 2s", "H 1s"])
    with pytest.warns(UserWarning) as record:
        alpha, beta = ad.build_orbitals_avas_spin(ao_projector=proj, max_size=(2, 2, 4))
    messages = " | ".join(str(w.message) for w in record)
    assert "frozen cores carry spin" in messages
    assert (alpha.n_occ - alpha.inactive.size, beta.n_occ - beta.inactive.size) == (2, 2)
    assert ad._avas_info["core_spin"] == 2
    # Freezing part of alpha's open shell leaves orbitals that are active for beta only:
    # exactly the asymmetric correlation the cosine diagnostic exists to flag.
    assert "active spaces diverge" in messages
    assert ad._avas_info["active_cosines"].min() < 0.1


def test_spin_divergent_active_spaces_warn(methylene_uhf, monkeypatch):
    import embasi_qiskit_integration.projection_embedding_adapter as pea

    ad = _uhf_spin_stub(methylene_uhf)
    proj = avas_ao_projector(methylene_uhf.mol, ["C 2p", "C 2s", "H 1s"])
    monkeypatch.setattr(pea, "_FROZEN_OVERLAP_TOL", 1.1)  # every cosine now counts as low
    with pytest.warns(UserWarning, match="active spaces diverge"):
        ad.build_orbitals_avas_spin(ao_projector=proj, max_size=(4, 5))


@pytest.mark.parametrize(
    "size,match",
    [
        ((3, 5), "n_alpha - n_beta = 2"),  # odd nelec cannot carry a triplet
        ((4, 5, 6, 7), "must be"),
        ((6, 2), "fit in norb"),
        ((4, 60), "virtual"),
        ((12, 13), "occupied"),
    ],
)
def test_spin_budget_errors(methylene_uhf, size, match):
    ad = _uhf_spin_stub(methylene_uhf)
    proj = avas_ao_projector(methylene_uhf.mol, ["C 2p"])
    with pytest.raises(ValueError, match=match):
        ad.build_orbitals_avas_spin(ao_projector=proj, max_size=size)


def test_restricted_budget_refuses_a_per_spin_split(formaldehyde_rhf):
    ad, _ = _stub(formaldehyde_rhf)
    with pytest.raises(ValueError, match="per-spin downfold"):
        ad.build_orbitals_avas(
            ao_projector=avas_ao_projector(formaldehyde_rhf.mol, ["O 2p"]), max_size=(2, 2, 3)
        )


def test_build_selector_spin_builder(methylene_uhf):
    ad = _uhf_spin_stub(methylene_uhf)
    cfg = _cfg(avas_ao_labels=["C 2p", "C 2s", "H 1s"], avas_max_size=(4, 5), active_atoms=[0])
    _, _, builder = build_selector(cfg, ad)
    alpha, beta = builder(ad, spin=True)
    assert alpha.n_active_orbitals == beta.n_active_orbitals == 5


# ----- live EmbASI (deselected by default; see test_open_shell_embedding_live) ---- #


@pytest.fixture(scope="module")
def live_open_shell():
    """The live OH-radical-in-water adapter after its low-level embedding."""
    from test_open_shell_embedding_live import _build_open_shell_adapter

    adapter = _build_open_shell_adapter()
    adapter.run_low_level()
    return adapter


@pytest.mark.embasi
@pytest.mark.slow
def test_live_full_space_spin_avas_reproduces_the_per_spin_downfold(live_open_shell):
    """A budget covering all of subsystem A spans the plain per-spin space, so the
    per-channel AVAS rotations must leave the FCI energy exactly where it was."""
    from embasi_qiskit_integration.solvers import FCISolver
    from test_open_shell_embedding_live import REF_E_SOLVER, REF_NELEC, REF_NORB

    ad = live_open_shell
    proj = avas_ao_projector(ad.ints.mol, ["O 2p", "H 1s"], atoms=[0, 1])
    pair = ad.build_orbitals_avas_spin(ao_projector=proj, max_size=(sum(REF_NELEC), REF_NORB))
    ham = ad.embedded_hamiltonian_spin(pair)
    assert ham.nelec == REF_NELEC and ham.norb == REF_NORB
    assert FCISolver().solve(ham).energy == pytest.approx(REF_E_SOLVER, abs=1e-8)


@pytest.mark.embasi
@pytest.mark.slow
def test_live_reduced_spin_avas_is_variational_and_leak_free(live_open_shell):
    from embasi_qiskit_integration.solvers import FCISolver
    from test_open_shell_embedding_live import REF_E_SOLVER

    ad = live_open_shell
    proj = avas_ao_projector(ad.ints.mol, ["O 2p", "H 1s"], atoms=[0, 1])
    pair = ad.build_orbitals_avas_spin(ao_projector=proj, max_size=(5, 4))
    ham = ad.embedded_hamiltonian_spin(pair)
    assert ham.nelec == (3, 2) and ham.norb == 4
    assert max(ham.meta["p_b_leak_per_spin"]) < 1e-8
    assert ad._avas_info["core_spin"] == 0
    assert ad._avas_info["active_cosines"].min() > 0.9
    # A CASCI on a subset of span(A), with an unrestricted frozen core, is a trial
    # wavefunction of the same Hamiltonian: it cannot go below the full-space FCI.
    energy = FCISolver().solve(ham).energy
    assert REF_E_SOLVER - 1e-9 <= energy < REF_E_SOLVER + 0.1
