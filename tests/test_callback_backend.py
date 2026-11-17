# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The generic callback backend (``backends/callback.py``) against direct PySCF.

``CallbackIntegrals`` never calls PySCF: every quantity comes back through EmbASI's own
``AtomsEmbed.run_noscf`` run/extract path.  Driving it with EmbASI's PySCF adapter and
comparing with :class:`~embasi_qiskit_integration.backends.pyscf.PySCFIntegrals` (which
calls ``get_hcore``/``get_veff``/``get_jk`` directly) is therefore an independent check of
the derivation ``hcore = H[0]``, ``veff = H[dm] - H[0]``, ``(J, K)`` from a UHF evaluation
at ``(d, 0)``, ``E_nuc = E[0]``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import require_module

from embasi_qiskit_integration.backends.base import MissingCapabilityError
from embasi_qiskit_integration.backends.callback import (
    CallbackIntegrals,
    Evaluation,
    LayerEvaluator,
)

TOL = 1e-10


def _cfg(**over):
    base = dict(
        basis="sto-3g",
        spin=0,
        unrestricted=False,
        xc_ll="PBE",
        xc_hl="HF",
        scf_max_cycle=None,
        init_guess=None,
        init_guess_breaksym=None,
        scf_newton=False,
        density_fit=False,
        df_auxbasis=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _water():
    from ase import Atoms

    return Atoms("OH2", positions=[(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)])


def _oh():
    from ase import Atoms

    return Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)])


def _pair(atoms, charge, cfg):
    require_module("embasi.atoms_embedding_asi", "needs a usable EmbASI")
    from embasi_qiskit_integration.backends.registry import make_backend

    direct = make_backend("pyscf", atoms, charge, cfg).integrals
    cb = make_backend("pyscf-callback", atoms, charge, cfg).integrals
    return direct, cb


@pytest.fixture(scope="module")
def closed_shell():
    return _pair(_water(), 0, _cfg())


@pytest.fixture(scope="module")
def open_shell():
    return _pair(_oh(), 0, _cfg(spin=1, unrestricted=True))


def _density(direct):
    """A non-trivial 2-occupancy density: the converged low-level one, perturbed."""
    mf = direct.mf_ll
    dm = np.asarray(mf.kernel() and mf.make_rdm1())
    total = dm.sum(axis=0) if dm.ndim == 3 else dm
    rng = np.random.default_rng(7)
    x = rng.normal(scale=1e-2, size=total.shape)
    return total + (x + x.T), (dm[0], dm[1]) if dm.ndim == 3 else (0.5 * total, 0.5 * total)


@pytest.mark.parametrize(
    "method, takes_dm",
    [
        ("overlap", False),
        ("hcore", False),
        ("veff_ll", True),
        ("veff_hl", True),
        ("veff_hf", True),
        ("get_k", True),
    ],
)
def test_matrices_match_pyscf(closed_shell, method, takes_dm):
    direct, cb = closed_shell
    args = (_density(direct)[0],) if takes_dm else ()
    np.testing.assert_allclose(getattr(cb, method)(*args), getattr(direct, method)(*args), atol=TOL)


def test_energies_and_uhf_potentials(closed_shell):
    direct, cb = closed_shell
    assert cb.energy_nuc() == pytest.approx(direct.energy_nuc(), abs=TOL)
    dm, _ = _density(direct)
    assert cb.energy_elec_ll(dm) == pytest.approx(direct.energy_elec_ll(dm), abs=1e-9)
    da, db = 0.55 * dm, 0.45 * dm
    va, vb, e2 = cb.veff_uhf(da, db)
    ra, rb, re2 = direct.veff_uhf(da, db)
    np.testing.assert_allclose(va, ra, atol=TOL)
    np.testing.assert_allclose(vb, rb, atol=TOL)
    assert e2 == pytest.approx(re2, abs=1e-9)


def test_open_shell_spin_pair_potentials(open_shell):
    direct, cb = open_shell
    assert cb.ll_spin_resolved is direct.ll_spin_resolved is True
    total, pair = _density(direct)
    np.testing.assert_allclose(cb.veff_ll(pair), direct.veff_ll(pair), atol=TOL)
    for got, want in zip(cb.veff_ll_spin(pair), direct.veff_ll_spin(pair), strict=True):
        np.testing.assert_allclose(got, want, atol=TOL)
    # A summed density is split evenly on an unrestricted evaluator, as PySCF does.
    np.testing.assert_allclose(cb.veff_ll(total), direct.veff_ll(total), atol=TOL)


@pytest.mark.parametrize("atoms, charge, spin", [(_water, 1, 1), (_oh, -1, 0)], ids=["H2O+", "OH-"])
def test_charged_systems_evaluate_at_their_own_charge(atoms, charge, spin):
    """Regression: the standalone layers never got the total charge, so EmbASI re-ran
    them neutral -- and PySCF refused the Mole outright for an odd charge."""
    direct, cb = _pair(atoms(), charge, _cfg(spin=spin, unrestricted=bool(spin)))
    np.testing.assert_allclose(cb.hcore(), direct.hcore(), atol=TOL)
    total, pair = _density(direct)
    arg = pair if spin else total
    np.testing.assert_allclose(cb.veff_ll(arg), direct.veff_ll(arg), atol=TOL)
    np.testing.assert_allclose(cb.veff_hf(total), direct.veff_hf(total), atol=TOL)


def test_basis_metadata_and_eris(closed_shell):
    direct, cb = closed_shell
    np.testing.assert_array_equal(cb.ao_atom_map(), direct.ao_atom_map())
    assert cb.ao_labels() == direct.ao_labels()
    for labels in (["O 2p"], ["O 2px"], ["O"], ["O 2p", "H 1s"]):
        np.testing.assert_array_equal(cb.search_ao_label(labels), direct.search_ao_label(labels))
    np.testing.assert_allclose(
        cb.valence_ao_projector(["O 2p"]), direct.valence_ao_projector(["O 2p"]), atol=TOL
    )
    c = np.linalg.qr(np.random.default_rng(1).normal(size=(cb.nao, 3)))[0]
    np.testing.assert_allclose(cb.eri_mo(c), direct.eri_mo(c), atol=TOL)
    fp_d, fp_c = direct.describe(), cb.describe()
    for key in ("basis", "natm", "charge", "spin", "atom_symbols", "xc_hl", "density_fit"):
        assert fp_c[key] == fp_d[key], key
    np.testing.assert_array_equal(fp_c["atom_coords"], fp_d["atom_coords"])


def test_capabilities_match_pyscf(closed_shell):
    direct, cb = closed_shell
    assert cb.capabilities == direct.capabilities


# --------------------------------------------------------------------------- #
# Reduced-capability behaviour, on a fake evaluator (no EmbASI needed)
# --------------------------------------------------------------------------- #
def _fake(nao=2, n_spin=1):
    """H[dm] = hcore + 2*dm (a linear 'mean field'), E[dm] = 1 + tr(dm)."""
    hcore = np.diag(np.arange(1.0, nao + 1))

    def ev(dm):
        if isinstance(dm, tuple):
            blocks = tuple(hcore + 2 * np.asarray(d) for d in dm[:n_spin])
            tr = sum(float(np.trace(d)) for d in dm)
        else:
            d = np.asarray(dm)
            blocks = (hcore + 2 * d,) * n_spin
            tr = float(np.trace(d))
        return Evaluation(blocks, np.eye(nao), 1.0 + tr)

    return ev, hcore


def test_dft_only_backend_declares_and_refuses_exchange_and_eris():
    ev, hcore = _fake()
    ints = CallbackIntegrals(ev, ev, None, nao=2, basis_atoms=np.array([0, 1]))
    assert ints.capabilities == frozenset({"hcore", "energy_elec"})
    np.testing.assert_array_equal(ints.hcore(), hcore)
    assert ints.energy_nuc() == 1.0
    np.testing.assert_allclose(ints.veff_ll(np.eye(2)), 2 * np.eye(2))
    with pytest.raises(MissingCapabilityError, match="exact exchange"):
        ints.get_k(np.eye(2))
    with pytest.raises(MissingCapabilityError, match="ERIs"):
        ints.eri_mo(np.eye(2))
    with pytest.raises(MissingCapabilityError, match="AO labels"):
        ints.ao_labels()


def test_hf_evaluator_must_be_unrestricted():
    ev, _ = _fake()
    ints = CallbackIntegrals(ev, ev, ev, nao=2)  # a restricted "HF" evaluator
    with pytest.raises(ValueError, match="unrestricted"):
        ints.get_k(np.eye(2))


def test_basis_atoms_shape_is_checked():
    ev, _ = _fake()
    with pytest.raises(ValueError, match=r"expected \(3,\)"):
        CallbackIntegrals(ev, ev, None, nao=3, basis_atoms=np.array([0, 1]))


def test_zero_density_reference_is_cached():
    calls = []
    ev, _ = _fake()

    def counting(dm):
        calls.append(dm)
        return ev(dm)

    ints = CallbackIntegrals(counting, counting, None, nao=2)
    ints.hcore()
    ints.overlap()
    ints.energy_nuc()
    ints.veff_ll(np.eye(2))
    ints.veff_ll(np.eye(2))
    assert len(calls) == 3  # one H[0], two H[dm]


def test_layer_evaluator_drops_embasis_cached_eigenvalue_sum():
    """EmbASI caches `ev_sum` per layer; a reused layer must not carry it across densities."""
    seen = []

    class _Layer:
        hamiltonian_total = {(0, 0): np.eye(1)}
        overlap = {(0, 0): np.eye(1)}
        total_energy = 0.0

        def run_noscf(self, dm_in):
            seen.append(hasattr(self, "ev_sum"))
            self.ev_sum = 1.23  # what EmbASI's run() leaves behind

    ev = object.__new__(LayerEvaluator)
    ev.n_spin, ev.label, ev._layer = 1, "t", _Layer()
    ev._wrap = lambda dm: dm  # type: ignore[method-assign]
    ev(np.zeros((1, 1)))
    ev(np.zeros((1, 1)))
    assert seen == [False, False]


# --------------------------------------------------------------------------- #
# End to end: the whole embedding on the callback route lands on the same number
# --------------------------------------------------------------------------- #
def _oh_water_xyz(tmp_path):
    from ase import Atoms
    from ase.io import write

    atoms = Atoms(
        "OHOHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.97],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.96],
            [4.9, 0.0, -0.3],
        ],
    )
    xyz = tmp_path / "oh_water.xyz"
    write(str(xyz), atoms)
    return str(xyz)


def _total(qm_code, **settings):
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    wf = EmbeddingWorkflow(_cli_parse_args=False, qm_code=qm_code, **settings)
    return float(wf.run(log=lambda *_a, **_k: None).total)


@pytest.mark.embasi
@pytest.mark.parametrize(
    "case",
    ["wf_in_dft_closed_shell", "wf_in_dft_open_shell_per_spin", "dft_in_dft"],
)
def test_workflow_total_matches_on_the_callback_route(case, tmp_path):
    common = dict(basis="sto-3g", max_cycles=1, solver="fci")
    if case == "wf_in_dft_closed_shell":
        settings = dict(common, xc_ll="PBE", xc_hl="HF", selector="none", n_virtual=2)
    elif case == "wf_in_dft_open_shell_per_spin":
        settings = dict(
            common,
            xyz=_oh_water_xyz(tmp_path),
            n_atoms=None,
            active_atoms=[0, 1],
            xc_ll="PBE",
            xc_hl="HF",
            spin=1,
            unrestricted=True,
            spin_downfold=True,
            selector="none",
            diis=True,
        )
    else:
        settings = dict(common, xc_ll="PBE", xc_hl="PBE0")
    direct = _total("pyscf", **settings)
    callback = _total("pyscf-callback", **settings)
    assert callback == pytest.approx(direct, abs=1e-8)
    if case == "wf_in_dft_open_shell_per_spin":
        from test_open_shell_embedding_live import REF_TOTAL

        assert callback == pytest.approx(REF_TOTAL, abs=1e-6)


@pytest.mark.embasi
def test_unrestricted_closed_shell_matches_the_restricted_total():
    """Regression: without --spin_downfold an unrestricted closed shell summed the spin
    channels of every operator, putting the water WF-in-DFT total 40 Ha low
    (-153.69 vs -113.77 Ha)."""
    settings = dict(
        basis="sto-3g", xc_hl="HF", selector="none", n_virtual=2, solver="fci", max_cycles=1
    )
    restricted = _total("pyscf", **settings)
    unrestricted = _total("pyscf", unrestricted=True, **settings)
    per_spin = _total("pyscf", unrestricted=True, spin_downfold=True, **settings)
    assert unrestricted == pytest.approx(per_spin, abs=1e-8)
    assert unrestricted == pytest.approx(restricted, abs=1e-5)  # UKS vs RKS SCF residual
