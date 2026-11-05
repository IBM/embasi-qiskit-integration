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

from embasi_qiskit_integration.backends.base import MissingCapabilityError
from embasi_qiskit_integration.backends.callback import CallbackIntegrals, Evaluation

TOL = 1e-10

# EmbASI imports mpi4py; without a working MPI library the layer cannot be built.
embasi_layers = pytest.importorskip("embasi.atoms_embedding_asi", exc_type=ImportError)


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


def test_one_electron_and_energies(closed_shell):
    direct, cb = closed_shell
    np.testing.assert_allclose(cb.overlap(), direct.overlap(), atol=TOL)
    np.testing.assert_allclose(cb.hcore(), direct.hcore(), atol=TOL)
    assert cb.energy_nuc() == pytest.approx(direct.energy_nuc(), abs=TOL)
    dm, _ = _density(direct)
    assert cb.energy_elec_ll(dm) == pytest.approx(direct.energy_elec_ll(dm), abs=1e-9)


def test_mean_field_potentials(closed_shell):
    direct, cb = closed_shell
    dm, _ = _density(direct)
    np.testing.assert_allclose(cb.veff_ll(dm), direct.veff_ll(dm), atol=TOL)
    np.testing.assert_allclose(cb.veff_hl(dm), direct.veff_hl(dm), atol=TOL)


def test_exact_exchange(closed_shell):
    direct, cb = closed_shell
    dm, _ = _density(direct)
    np.testing.assert_allclose(cb.veff_hf(dm), direct.veff_hf(dm), atol=TOL)
    np.testing.assert_allclose(cb.get_k(dm), direct.get_k(dm), atol=TOL)
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


def test_basis_metadata_and_eris(closed_shell):
    direct, cb = closed_shell
    np.testing.assert_array_equal(cb.ao_atom_map(), direct.ao_atom_map())
    assert cb.ao_labels() == direct.ao_labels()
    np.testing.assert_array_equal(cb.search_ao_label(["O 2p"]), direct.search_ao_label(["O 2p"]))
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
