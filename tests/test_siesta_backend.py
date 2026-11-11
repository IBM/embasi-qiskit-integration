# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The SIESTA backend (``backends/siesta.py``): file readers, capabilities, refusals.

SIESTA is GPL; it is only ever run as an external program here, and these tests never
import it.  The parser runs on a committed ``.ORB_INDX`` (SIESTA *output*, water/DZP);
the real-run checks skip unless the ``siesta`` executable and ``$SIESTA_PP_PATH``
(``<Element>.psml`` files) are available.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.backends import siesta as sb

DATA = Path(__file__).parent / "data" / "siesta"

needs_siesta = pytest.mark.skipif(
    not (shutil.which("siesta") and os.environ.get(sb.PSEUDO_ENV)),
    reason="needs the siesta executable on PATH and $SIESTA_PP_PATH",
)


def _cfg(**over):
    base = dict(
        basis="sto-3g",
        spin=0,
        unrestricted=False,
        xc_ll="PBE",
        xc_hl="PBE",
        scf_max_cycle=None,
        siesta_basis="DZP",
        siesta_mesh_cutoff=200.0,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _water():
    from ase import Atoms

    return Atoms("OH2", positions=[(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)], cell=[10] * 3)


# --------------------------------------------------------------------------- #
# .ORB_INDX parsing (no SIESTA needed)
# --------------------------------------------------------------------------- #
def test_orb_indx_parses_the_dzp_water_basis():
    orbs = sb.read_orb_indx(DATA / "h2o_dzp.ORB_INDX")
    assert len(orbs) == 23  # O: 2s(2) 2p(6) 3d*(5);  H: 1s(2) 2p*(3), twice
    np.testing.assert_array_equal(sb.basis_atoms_from_orbitals(orbs), [0] * 13 + [1] * 5 + [2] * 5)
    o = orbs[:13]
    assert [x.shell for x in o] == ["2s"] * 2 + ["2p"] * 6 + ["3d*"] * 5
    assert [x.zeta for x in o[:8]] == [1, 2, 1, 1, 1, 2, 2, 2]
    assert all(x.polarisation for x in o[8:]) and not any(x.polarisation for x in o[:8])


def test_labels_strip_the_ase_species_suffix_and_flag_polarisation():
    orbs = sb.read_orb_indx(DATA / "h2o_dzp.ORB_INDX")
    labels = sb.ao_labels_from_orbitals(orbs)
    assert labels[2] == (0, "O", "2p", "py")
    assert sb._element("O.1") == "O" and sb._element("Fe") == "Fe"
    # H's polarisation shell is "2p*": a valence-label search for "H 2p" must not hit it.
    assert {lab[2] for lab in labels if lab[1] == "H"} == {"1s", "2p*"}


def test_orb_indx_rejects_a_non_siesta_file(tmp_path):
    bad = tmp_path / "x.ORB_INDX"
    bad.write_text("not an orbital table\n")
    with pytest.raises(ValueError, match="no unit-cell orbitals"):
        sb.read_orb_indx(bad)


def test_search_ao_label_on_siesta_labels():
    """The generic backend's label search, fed SIESTA's labels."""
    from embasi_qiskit_integration.backends.callback import CallbackIntegrals, Evaluation

    orbs = sb.read_orb_indx(DATA / "h2o_dzp.ORB_INDX")
    n = len(orbs)

    def ev(_dm):
        return Evaluation((np.zeros((n, n)),), np.eye(n), 0.0)

    ints = CallbackIntegrals(
        ev,
        ev,
        None,
        nao=n,
        basis_atoms=sb.basis_atoms_from_orbitals(orbs),
        ao_labels=sb.ao_labels_from_orbitals(orbs),
    )
    np.testing.assert_array_equal(ints.search_ao_label(["O 2p"]), np.arange(2, 8))
    assert ints.search_ao_label(["H 2p"]).size == 0  # polarisation, not valence
    assert ints.search_ao_label(["H 2p*"]).size == 6


# --------------------------------------------------------------------------- #
# Refusals (no SIESTA needed: they fire before any run)
# --------------------------------------------------------------------------- #
def test_wf_in_dft_is_refused_for_lack_of_exchange_and_eris():
    from embasi_qiskit_integration.backends.registry import make_backend

    with pytest.raises(NotImplementedError, match="DFT-in-DFT only"):
        make_backend("siesta", _water(), 0, _cfg(xc_hl="HF"))


def test_missing_pseudopotential_path_is_reported(monkeypatch):
    monkeypatch.delenv(sb.PSEUDO_ENV, raising=False)
    with pytest.raises(RuntimeError, match="SIESTA_PP_PATH"):
        sb.siesta_calculator(_cfg(), xc="PBE", spin_polarised=False, directory="x")


def test_hybrid_functional_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv(sb.PSEUDO_ENV, str(tmp_path))
    monkeypatch.setenv(sb.COMMAND_ENV, "siesta < PREFIX.fdf > PREFIX.out")
    with pytest.raises(ValueError, match="no exact exchange"):
        sb.siesta_calculator(_cfg(), xc="PBE0", spin_polarised=False, directory="x")


def test_siesta_settings_are_refused_for_pyscf():
    from embasi_qiskit_integration.backends.registry import check_backend_settings

    check_backend_settings("pyscf", _cfg())  # defaults: fine
    with pytest.raises(ValueError, match="--siesta_basis is a siesta option"):
        check_backend_settings("pyscf", _cfg(siesta_basis="SZ"))


def test_refuses_until_embasi_has_a_siesta_adapter(monkeypatch, tmp_path):
    """Our side is wired; EmbASI's adapter registry is what is missing."""
    from embasi_qiskit_integration.backends.registry import make_backend

    monkeypatch.setenv(sb.PSEUDO_ENV, str(tmp_path))
    monkeypatch.setenv(sb.COMMAND_ENV, "siesta < PREFIX.fdf > PREFIX.out")
    monkeypatch.setattr(sb, "_embasi_supports", lambda calc: False)
    with pytest.raises(NotImplementedError, match="no QMCodeAdapter for ASE's Siesta"):
        make_backend("siesta", _water(), 0, _cfg())


def test_adapter_refusal_comes_before_the_install_checks(monkeypatch):
    """Without SIESTA installed the user must hear the real blocker, not 'install SIESTA'."""
    from embasi_qiskit_integration.backends.registry import make_backend

    monkeypatch.delenv(sb.PSEUDO_ENV, raising=False)
    monkeypatch.delenv(sb.COMMAND_ENV, raising=False)
    monkeypatch.setattr(sb, "_embasi_supports", lambda calc: False)
    with pytest.raises(NotImplementedError, match="no QMCodeAdapter for ASE's Siesta"):
        make_backend("siesta", _water(), 0, _cfg())


def test_a_cell_less_molecule_gets_a_vacuum_box():
    from ase import Atoms

    atoms = Atoms("OH2", positions=[(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)])
    before = atoms.get_positions().copy()
    sb.ensure_vacuum_cell(atoms)
    np.testing.assert_allclose(np.diag(atoms.cell), [10.0, 11.514, 10.587])
    np.testing.assert_array_equal(atoms.get_positions(), before)  # positions untouched
    boxed = Atoms("H", cell=[7.0, 7.0, 7.0])
    sb.ensure_vacuum_cell(boxed)
    np.testing.assert_allclose(np.diag(boxed.cell), [7.0, 7.0, 7.0])  # explicit cell kept


def test_basis_probe_uses_ases_scf_must_converge_spelling(monkeypatch, tmp_path):
    """ASE writes `SCFMustConverge True` itself; a dotted key would conflict with it."""
    seen = {}

    class _Calc:
        def __init__(self):
            self.parameters = {"fdf_arguments": {}}

    def fake_calculator(cfg, **kw):
        seen["calc"] = _Calc()
        return seen["calc"]

    monkeypatch.setattr(sb, "siesta_calculator", fake_calculator)
    with pytest.raises(Exception):  # noqa: B017 - the fake cannot run; only the keys matter
        sb.probe_basis(_water(), _cfg(), directory=tmp_path)
    fdf = seen["calc"].parameters["fdf_arguments"]
    assert fdf["SCFMustConverge"] is False and "SCF.MustConverge" not in fdf


# --------------------------------------------------------------------------- #
# Against the real code (skipped without SIESTA)
# --------------------------------------------------------------------------- #
@needs_siesta
def test_basis_probe_matches_the_committed_full_run(tmp_path):
    probe = sb.probe_basis(_water(), _cfg(), directory=tmp_path)
    ref = sb.read_orb_indx(DATA / "h2o_dzp.ORB_INDX")

    def key(orbs):
        return [(o.atom, o.n, o.l, o.m, o.zeta, o.polarisation) for o in orbs]

    assert key(probe) == key(ref)


@needs_siesta
def test_matrix_reader_on_a_real_run(tmp_path):
    pytest.importorskip("sisl")
    from ase.calculators.siesta import Siesta

    atoms = _water()
    calc = sb.siesta_calculator(_cfg(), xc="PBE", spin_polarised=False, directory=str(tmp_path))
    assert isinstance(calc, Siesta)
    atoms.calc = calc
    atoms.get_potential_energy()
    m = sb.read_matrices(tmp_path / "siesta")
    s, h, d = m["overlap"], m["hamiltonian"][0], m["density"][0]
    assert s.shape == h.shape == d.shape == (23, 23)
    np.testing.assert_allclose(s, s.T, atol=1e-8)
    np.testing.assert_allclose(h, h.T, atol=1e-7)
    assert float(np.trace(d @ s)) == pytest.approx(8.0, abs=1e-6)  # valence electrons
    # The Kohn-Sham H itself, not sisl's Fermi-shifted H - Ef*S: its generalised
    # eigenvalues are SIESTA's own band energies.
    _assert_matches_eig(m, tmp_path / "siesta.EIG")


@needs_siesta
def test_open_shell_run_is_genuinely_spin_polarised(tmp_path):
    """Regression: without Spin.Fix an OH doublet converged unpolarised (3.5/3.5,
    H_alpha == H_beta), which would silently turn an open shell into a closed one."""
    pytest.importorskip("sisl")
    from ase import Atoms

    atoms = Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)], cell=[10] * 3)
    atoms.calc = sb.siesta_calculator(
        _cfg(), xc="PBE", spin_polarised=True, directory=str(tmp_path), total_spin=1
    )
    atoms.get_potential_energy()
    m = sb.read_matrices(tmp_path / "siesta")
    s = m["overlap"]
    n_a, n_b = (float(np.trace(d @ s)) for d in m["density"])
    assert (n_a, n_b) == pytest.approx((4.0, 3.0), abs=1e-6)
    assert float(np.abs(m["hamiltonian"][0] - m["hamiltonian"][1]).max()) > 1e-3
    # Spin.Fix gives each channel its own Fermi level; sisl removes only one of them, so a
    # missing per-spin correction shows up as a ~1 eV error in the beta levels.
    _assert_matches_eig(m, tmp_path / "siesta.EIG")


def _assert_matches_eig(matrices, eig_file):
    import scipy.linalg as sla
    from ase.units import Hartree

    toks = eig_file.read_text().split()
    n_band, n_spin = int(toks[1]), int(toks[2])
    ref = np.array(toks[5 : 5 + n_band * n_spin], dtype=float).reshape(n_spin, n_band)
    for spin in range(n_spin):
        e = sla.eigh(matrices["hamiltonian"][spin], matrices["overlap"], eigvals_only=True)
        np.testing.assert_allclose(e * Hartree, ref[spin], atol=1e-5)
