# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""AO-defined active space: ``--selector ao-avas`` / :func:`fragment_ao_projector`.

The butyronitrile cross-code benchmark exposed that an *index* active space is not
well defined: ``apc-concentric`` and ``--selector none`` can report the same column
indices ``[2..11]`` while their active spans differ, because APC has rotated the
virtual block.  ``ao-avas`` fixes that by selecting the occupied and virtual columns
with the largest weight on a fixed set of **fragment AOs**, in fixed counts.  The
selection is then a function of the AO target and the occupied/virtual *spans*, not
of the arbitrary basis those spans happen to be expressed in, and it needs no MINAO
reference (so SIESTA's NAO backend can use it too).

These tests are deliberately small: synthetic block decompositions pin the selector
properties exactly, H2 at two bond lengths exercises the real PySCF adapter path, and
a fake integral engine checks the active-space FCI energy under an intra-span
rotation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pyscf
import pytest

from embasi_qiskit_integration.backends.pyscf import PySCFIntegrals
from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.embedding import build_selector
from embasi_qiskit_integration.projection_embedding_adapter import (
    EmbeddedOrbitals,
    ProjectionEmbeddingAdapter,
)
from embasi_qiskit_integration.selectors import (
    avas_ao_projector,
    fragment_ao_projector,
)
from embasi_qiskit_integration.solvers import FCISolver


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _span_cosines(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """Principal cosines between two coefficient spans (Euclidean index space).

    QR removes the arbitrary intra-span basis; identical subspaces give all ones.
    """
    q1 = np.linalg.qr(np.asarray(c1, dtype=float))[0]
    q2 = np.linalg.qr(np.asarray(c2, dtype=float))[0]
    return np.linalg.svd(q1.T @ q2, compute_uv=False)


def _random_orthogonal(n: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return q


def _select(coeff, fock, projector, *, n_occ=3, n_keep=1):
    """Run the AVAS rotation/selection on one block decomposition (fixed counts)."""
    return ProjectionEmbeddingAdapter._avas_select(
        coeff,
        np.arange(coeff.shape[1], dtype=float),
        fock,
        projector,
        n_occ=n_occ,
        n_docc=n_occ,
        n_frozen_occ=0,
        choose=lambda _w_occ, _w_virt: (n_keep, n_keep),
    )


class _FakeIntegrals:
    """Minimal AOIntegrals stand-in: S = I, a Coulomb-like ERI, J - K/2 exchange."""

    def __init__(self, hcore: np.ndarray, eri: np.ndarray):
        self._h = hcore
        self._eri = eri
        self._s = np.eye(hcore.shape[0])

    def overlap(self):
        return self._s

    def hcore(self):
        return self._h

    def energy_nuc(self):
        return 0.0

    def veff_hf(self, dm):
        dm = np.asarray(dm)
        j = np.einsum("pqrs,rs->pq", self._eri, dm)
        k = np.einsum("prqs,rs->pq", self._eri, dm)
        return j - 0.5 * k

    def veff_ll(self, dm):
        return self.veff_hf(dm)

    def eri_mo(self, c):
        c = np.asarray(c)
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", self._eri, c, c, c, c)


def _downfold_adapter(coeff, n_occ, inactive, active, hcore, eri):
    """A stub adapter with an empty environment and :class:`_FakeIntegrals`."""
    ints = _FakeIntegrals(hcore, eri)
    dm_a = 2.0 * coeff[:, :n_occ] @ coeff[:, :n_occ].T

    class _Stub(ProjectionEmbeddingAdapter):
        _s_arr = property(lambda self: ints.overlap())
        _fock_arr = property(lambda self: hcore + ints.veff_ll(dm_a))
        _dm_a_arr = property(lambda self: dm_a)
        mo_a_ll = property(lambda self: coeff[:, :n_occ])
        mo_b_ll = property(lambda self: np.zeros((coeff.shape[0], 0)))

        def _validate_span(self, c):
            return None

    ad = object.__new__(_Stub)
    ad.p = SimpleNamespace(A_spin=0)
    ad.ints = ints
    ad.unrestricted = False
    ad.mu = 1.0e6
    ad._dm_a_spin = None
    ad._dm_a = dm_a
    ad._p_b = np.zeros_like(hcore)
    return ad


def _orbitals(coeff, n_occ, inactive, active):
    return EmbeddedOrbitals(
        coeff=coeff,
        energy=np.diag(coeff.T @ coeff).astype(float),
        n_occ=n_occ,
        inactive=np.asarray(inactive, dtype=int),
        active=np.asarray(active, dtype=int),
        n_occ_b=None,
    )


# --------------------------------------------------------------------------- #
# the same-basis projector
# --------------------------------------------------------------------------- #
def test_projector_matches_minao_when_the_working_basis_is_the_reference():
    # With the working basis itself set to PySCF's MINAO, the two constructions are
    # the same target AO space by definition -- an exact cross-check of the algebra.
    mol = pyscf.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="minao",
        verbose=0,
    )
    ints = PySCFIntegrals(mol.RHF())
    same = fragment_ao_projector(ints, ["O 2p"], atoms=[0])
    minao = avas_ao_projector(ints, ["O 2p"])
    np.testing.assert_allclose(same, minao, atol=1e-12)


def test_projector_without_labels_targets_every_fragment_ao():
    mol = pyscf.M(
        atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis="sto-3g",
        verbose=0,
    )
    ints = PySCFIntegrals(mol.RHF())
    all_oxygen = ["O 1s", "O 2s", "O 2p"]
    np.testing.assert_allclose(
        fragment_ao_projector(ints, None, atoms=[0]),
        fragment_ao_projector(ints, all_oxygen, atoms=[0]),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        fragment_ao_projector(ints, None, atoms=[1, 2]),
        fragment_ao_projector(ints, ["H 1s"], atoms=[1, 2]),
        atol=1e-14,
    )


def test_projector_is_a_metric_bounded_on_the_target_space():
    # c^T M c is the squared S-norm of the projection onto span(target AOs): 1 for
    # an S-normalised orbital inside that span, 0 for one S-orthogonal to it.
    rng = np.random.default_rng(4)
    nao = 7
    a = rng.standard_normal((nao, nao))
    s = a @ a.T + nao * np.eye(nao)

    class _Basis:
        capabilities = None

        def __init__(self):
            self.nao = nao

        def overlap(self):
            return s

        def ao_atom_map(self):
            return np.array([0, 0, 1, 1, 1, 1, 1])

        def atom_ao_indices(self, atoms):
            return np.flatnonzero(np.isin(self.ao_atom_map(), list(atoms)))

    basis = _Basis()
    m = fragment_ao_projector(basis, None, atoms=[0])
    target = np.zeros((nao, 2))
    target[0, 0] = target[1, 1] = 1.0
    # M is the identity metric on the target span and zero on its S-orthogonal
    # complement.  Build an S-orthonormal basis of that span: B S22^{-1/2}.
    s22 = target.T @ s @ target
    w22, u22 = np.linalg.eigh(s22)
    inside = target @ (u22 * (1.0 / np.sqrt(w22))) @ u22.T
    np.testing.assert_allclose(inside.T @ s @ inside, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(inside.T @ m @ inside, np.eye(2), atol=1e-10)
    # Any vector S-orthogonal to the target AOs has zero weight.
    _vt = np.linalg.svd(target.T @ s)[2]  # (nao, nao): row space of B^T S
    null = _vt[target.shape[1] :].T  # remaining rows span its right null space
    null = null / np.sqrt(np.einsum("ij,ij->j", null, s @ null))
    np.testing.assert_allclose(null.T @ m @ null, np.zeros((nao - 2, nao - 2)), atol=1e-10)


def test_projector_rejects_labels_a_backend_cannot_report():
    # SIESTA's ASI descriptors carry only the atom index: no AO labels.  Requesting
    # labels must fail with the portable fallback named, not silently target nothing.
    class _NoLabelBasis:
        nao = 3
        capabilities = None

        def overlap(self):
            return np.eye(3)

        def ao_atom_map(self):
            return np.array([0, 0, 1])

        def atom_ao_indices(self, atoms):
            return np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="no AO shell labels.*omit ao_labels"):
        fragment_ao_projector(_NoLabelBasis(), ["C 2p"], atoms=[0])
    # ...while the index-only fallback still builds the projector.
    m = fragment_ao_projector(_NoLabelBasis(), None, atoms=[0])
    np.testing.assert_allclose(m, np.diag([1.0, 1.0, 0.0]))


def test_projector_works_on_a_siesta_shaped_callback_basis():
    # SIESTA's ASI backend is a CallbackIntegrals with basis_atoms but ao_labels=None.
    # The atom-index fallback must still build a projector; labels must fail clearly.
    from embasi_qiskit_integration.backends.callback import CallbackIntegrals, Evaluation

    s = np.eye(4)
    atoms = np.array([0, 0, 1, 1], dtype=int)

    def evaluate(_dm):
        return Evaluation((np.zeros_like(s),), s, 0.0)

    basis = CallbackIntegrals(evaluate, evaluate, None, nao=4, basis_atoms=atoms, ao_labels=None)
    np.testing.assert_allclose(
        fragment_ao_projector(basis, None, atoms=[0]), np.diag([1.0, 1.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError, match="no AO shell labels"):
        fragment_ao_projector(basis, ["C 2p"], atoms=[0])


# --------------------------------------------------------------------------- #
# fixed AO target => fixed active span (deterministic two-"geometry" stub)
# --------------------------------------------------------------------------- #
def _clean_target_setup(nao=6):
    """Target AOs {0, 4}, one inside each block; extra orbitals differ per geometry."""
    projector = np.zeros((nao, nao))
    projector[0, 0] = projector[4, 4] = 1.0
    return projector


def test_selected_span_is_identical_when_the_target_space_is_block_clean():
    # "Geometry A": occupied span {e0,e2,e3}, virtual {e1,e4,e5}.
    # "Geometry B": occupied {e0,e2,e5},      virtual {e1,e4,e3}.
    # The target AOs e0/e4 sit one in each block at both geometries, so the selected
    # span must be exactly span{e0,e4} -- even though the surrounding orbitals (and
    # their canonical energies/basis) change completely.
    projector = _clean_target_setup()
    eye = np.eye(6)

    coeff_a = eye.copy()
    coeff_b = eye[:, [0, 2, 5, 1, 4, 3]]  # columns 0..2 occupied, 3..5 virtual
    fock_a = _random_orthogonal(6, 10)
    fock_a = fock_a @ np.diag(np.arange(6.0)) @ fock_a.T
    fock_b = _random_orthogonal(6, 11)
    fock_b = fock_b @ np.diag(np.arange(6.0)[::-1]) @ fock_b.T

    sel_a = _select(coeff_a, fock_a, projector)
    sel_b = _select(coeff_b, fock_b, projector)
    act_a = sel_a["coeff"][:, sel_a["active"]]
    act_b = sel_b["coeff"][:, sel_b["active"]]

    np.testing.assert_allclose(_span_cosines(act_a, act_b), np.ones(2), atol=1e-12)
    np.testing.assert_allclose(
        act_a @ np.linalg.inv(act_a.T @ act_a) @ act_a.T,
        act_b @ np.linalg.inv(act_b.T @ act_b) @ act_b.T,
        atol=1e-12,
    )
    # One occupied and one virtual column, as the fixed (1,1) budget requests.
    assert np.count_nonzero(sel_a["active"] < 3) == 1
    assert np.count_nonzero(sel_a["active"] >= 3) == 1


def test_selected_span_is_invariant_to_arbitrary_block_rotations():
    # The APC-vs-none ambiguity in one assertion: re-expressing the occupied and
    # virtual blocks in a random basis (what APC/CL rotations do) cannot change an
    # AO-weighted selection, because its weights depend on the block spans only.
    projector = _clean_target_setup()
    eye = np.eye(6)
    fock = _random_orthogonal(6, 12) @ np.diag(np.arange(6.0)) @ _random_orthogonal(6, 12).T
    fock = 0.5 * (fock + fock.T)
    base = _select(eye, fock, projector)
    act_base = base["coeff"][:, base["active"]]

    q_occ = _random_orthogonal(3, 13)
    q_vir = _random_orthogonal(3, 14)
    rotated = eye.copy()
    rotated[:, :3] = eye[:, :3] @ q_occ
    rotated[:, 3:] = eye[:, 3:] @ q_vir
    # The Fock rotates with the block basis, as it does in a canonical eigensolve.
    fock_rot = rotated.T @ fock @ rotated
    sel = _select(rotated, fock_rot, projector)
    act = sel["coeff"][:, sel["active"]]
    np.testing.assert_allclose(_span_cosines(act_base, act), np.ones(2), atol=1e-12)


def test_active_fci_energy_is_invariant_to_rotations_inside_the_span():
    # Rotating columns *inside* the active span is a basis change of the same CAS;
    # the downfolded h1/h2 and the FCI energy must not move.
    nao, n_occ = 6, 3
    rng = np.random.default_rng(20)
    b = rng.standard_normal((nao, nao))
    hcore = b @ b.T / nao  # SPD, Hermitian
    # A random chemist-notation (pq|rs) tensor with the 8-fold index symmetry.
    x = rng.standard_normal((nao, nao, nao, nao))
    eri = 0.25 * (x + x.transpose(1, 0, 2, 3) + x.transpose(0, 1, 3, 2) + x.transpose(1, 0, 3, 2))
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    coeff = np.eye(nao)
    inactive = np.array([1, 2])
    active = np.array([0, 4])

    def energy(c):
        ad = _downfold_adapter(c, n_occ, inactive, active, hcore, eri)
        ham = ad.embedded_hamiltonian(_orbitals(c, n_occ, inactive, active))
        assert isinstance(ham, EmbeddedHamiltonian)
        assert ham.norb == 2 and ham.nelec == (1, 1)
        return FCISolver().solve(ham).energy

    e0 = energy(coeff)
    q = _random_orthogonal(2, 21)
    rotated = coeff.copy()
    rotated[:, active] = coeff[:, active] @ q
    e1 = energy(rotated)
    assert e1 == pytest.approx(e0, abs=1e-10)


# --------------------------------------------------------------------------- #
# real PySCF backend: H2 at two bond lengths
# --------------------------------------------------------------------------- #
def _h2_stub(mf):
    mol = mf.mol
    overlap = mol.intor_symmetric("int1e_ovlp")
    fock = mf.get_fock()
    c_occ = mf.mo_coeff[:, mf.mo_occ > 0]
    dm = 2.0 * c_occ @ c_occ.T

    class _Stub(ProjectionEmbeddingAdapter):
        _s_arr = property(lambda self: overlap)
        _fock_arr = property(lambda self: fock)
        _dm_a_arr = property(lambda self: dm)
        mo_a_ll = property(lambda self: c_occ)
        mo_b_ll = property(lambda self: np.zeros((mol.nao, 0)))

        def _validate_span(self, c):
            return None

    ad = object.__new__(_Stub)
    ad.p = SimpleNamespace(A_spin=0)
    ad.ints = PySCFIntegrals(mf)
    ad.unrestricted = False
    return ad


def test_h2_ao_span_is_identical_at_two_bond_lengths():
    # STO-3G H2: the two H 1s AOs are the whole basis, so the fixed AO target with a
    # full (2,2) budget is the whole active space -- exactly the same AO span at both
    # geometries, whatever the S-orthonormal basis chosen inside it.
    spans, target_sets = [], []
    for r in (0.7, 2.0):
        mol = pyscf.M(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g", verbose=0)
        mf = mol.RHF().run(conv_tol=1e-12)
        ad = _h2_stub(mf)
        projector = fragment_ao_projector(ad.ints, ["H 1s"])
        orb = ad.build_orbitals_avas(ao_projector=projector, max_size=(2, 2), n_frozen_occ=0)
        assert (orb.n_active_electrons, orb.n_active_orbitals) == (2, 2)
        assert ad._avas_info["n_active_occ"] == 1 and ad._avas_info["n_active_virt"] == 1
        spans.append(orb.c_active)
        target_sets.append(ad.ints.search_ao_label(["H 1s"]).tolist())
    assert target_sets[0] == target_sets[1]  # the AO definition itself is fixed
    np.testing.assert_allclose(_span_cosines(*spans), np.ones(2), atol=1e-10)


# --------------------------------------------------------------------------- #
# workflow wiring
# --------------------------------------------------------------------------- #
def _cfg(**kw):
    base = {
        "selector": "ao-avas",
        "avas_ao_labels": ["H 1s"],
        "avas_threshold": 0.2,
        "avas_minao": "minao",
        "avas_max_size": (2, 2),
        "n_frozen_occ": 0,
        "active_atoms": [0, 1],
    }
    return SimpleNamespace(**{**base, **kw})


def test_build_selector_returns_an_ao_avas_builder():
    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = mol.RHF().run(conv_tol=1e-12)
    ad = _h2_stub(mf)
    selector, localizer, builder = build_selector(_cfg(), ad)
    assert selector is None and localizer is None and builder is not None

    orb = builder(ad)
    assert (orb.n_active_electrons, orb.n_active_orbitals) == (2, 2)
    assert ad._avas_info["selector"] == "ao-avas"
    assert ad._avas_info["projector"] == "working-basis fragment AOs"


def test_ao_avas_without_labels_is_the_siesta_portable_path():
    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = mol.RHF().run(conv_tol=1e-12)
    ad = _h2_stub(mf)
    _selector, _localizer, builder = build_selector(_cfg(avas_ao_labels=None), ad)
    orb = builder(ad)
    assert orb.n_active_orbitals == 2
    assert ad._avas_info["projector"] == "working-basis fragment AOs"


def test_build_selector_passes_the_opt_in_energy_window():
    # H2/STO-3G has one virtual, far above the HOMO, so a 0.3 Ha window finds no
    # eligible candidate: the fixed budget is still filled (shortfall=1) and the
    # untouched default reports no window at all.
    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    mf = mol.RHF().run(conv_tol=1e-12)

    ad_default = _h2_stub(mf)
    _sel, _loc, builder = build_selector(_cfg(), ad_default)
    builder(ad_default)
    assert "energy_window" not in ad_default._avas_info

    ad_window = _h2_stub(mf)
    _sel, _loc, builder = build_selector(_cfg(avas_energy_window=0.3), ad_window)
    orb = builder(ad_window)
    assert orb.n_active_orbitals == 2
    assert ad_window._avas_info["energy_window"] == 0.3
    assert ad_window._avas_info["window_shortfall"] == 1
    assert ad_window._avas_info["window_n_candidates"] == 0


def test_build_selector_refuses_the_window_for_other_selectors():
    cfg = _cfg(selector="mulliken", avas_energy_window=0.3)
    with pytest.raises(ValueError, match="applies only to the AVAS selectors"):
        build_selector(cfg, object())


# --------------------------------------------------------------------------- #
# benchmark runner CLI
# --------------------------------------------------------------------------- #
def _load_runner():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/benchmarks/run_structural_scan.py"
    spec = importlib.util.spec_from_file_location("benchmark_runner_ao_avas", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_routes_cas_size_to_ao_avas_and_passes_labels(tmp_path, capsys):
    runner = _load_runner()
    rc = runner.main(
        [
            "--backend",
            "pyscf",
            "--out-dir",
            str(tmp_path),
            "--selector",
            "ao-avas",
            "--cas-size",
            "6",
            "6",
            "--avas-ao-labels",
            "N 2p",
            "C 2p",
            "--geometries",
            "08",
            "--dry-run",
        ]
    )
    assert rc == 0
    command = capsys.readouterr().out
    assert "--selector ao-avas" in command
    assert "--avas_max_size [6,6]" in command
    assert '"N 2p", "C 2p"' in command


def test_runner_refuses_ao_avas_with_the_reference_protocol(tmp_path):
    runner = _load_runner()
    with pytest.raises(SystemExit, match="builds its own canonical concentric active space"):
        runner.main(
            [
                "--backend",
                "pyscf",
                "--out-dir",
                str(tmp_path),
                "--selector",
                "ao-avas",
                "--cas-size",
                "4",
                "4",
                "--wf-in-dft-energy",
                "reference",
                "--geometries",
                "08",
                "--dry-run",
            ]
        )
