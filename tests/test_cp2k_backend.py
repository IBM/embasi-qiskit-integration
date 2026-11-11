# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The CP2K backend (``backends/cp2k.py``): input generation, run, and its FCIDUMP.

CP2K is GPL and only run as an external program; nothing here imports it.  Fixtures are
real CP2K 7.1 output for water, CAS(4e,4o), DZVP-MOLOPT-GTH / GTH-PBE, isolated (MT):

* ``h2o_cas44.fcidump`` -- written by :func:`run_active_space` (negative EPS_INTEGRAL);
* ``h2o_cas44_cp2k71_signed_screen.fcidump`` -- the same system with CP2K's default
  EPS_INTEGRAL, where 7.1's signed screen ``erint > EPS_INTEGRAL`` dropped every negative
  two-electron integral.

The real-run tests skip unless a CP2K executable and ``$CP2K_DATA_DIR`` are available.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from embasi_qiskit_integration.backends import cp2k
from embasi_qiskit_integration.hamiltonian import fcidump

DATA = Path(__file__).parent / "data" / "cp2k"
GOOD = DATA / "h2o_cas44.fcidump"
SIGNED = DATA / "h2o_cas44_cp2k71_signed_screen.fcidump"
BASIS = {"O": "DZVP-MOLOPT-GTH", "H": "DZVP-MOLOPT-GTH"}
POTENTIAL = {"O": "GTH-PBE-q6", "H": "GTH-PBE-q1"}


def _have_cp2k() -> bool:
    try:
        cp2k.find_executable()
    except RuntimeError:
        return False
    data = os.environ.get(cp2k.DATA_ENV)
    return bool(data) and (Path(data) / "BASIS_MOLOPT").is_file()


needs_cp2k = pytest.mark.skipif(not _have_cp2k(), reason="needs CP2K and $CP2K_DATA_DIR")


def _det_energy(ham, n_docc):
    g, f = ham.h2, ham.h1
    occ = range(n_docc)
    return (
        ham.e_core
        + sum(2 * f[i, i] for i in occ)
        + sum(2 * g[i, i, j, j] - g[i, j, j, i] for i in occ for j in occ)
    )


def _fci(ham):
    from embasi_qiskit_integration.solvers import FCISolver

    return FCISolver().solve(ham).energy


# --------------------------------------------------------------------------- #
# The real FCIDUMPs
# --------------------------------------------------------------------------- #
def test_fcidump_reference_determinant_equals_cp2ks_total():
    """CP2K's KS total equals the active-space closed-shell determinant: e_core, h1, the
    Coulomb/exchange integrals and the index layout all read correctly."""
    ham = fcidump.read(GOOD)
    assert ham.nelec == (2, 2) and ham.norb == 4 and not ham.is_spin_dependent
    total = cp2k.read_total_energy(DATA / "h2o_cas44_energy.out")
    assert _det_energy(ham, 2) == pytest.approx(total, abs=1e-12)


def test_fci_on_the_corrected_fcidump():
    """With every integral kept, FCI is the phase-invariant value (same with 1 or 2 OpenMP
    threads and any ADDED_MOS, checked when the fixture was made)."""
    ham = fcidump.read(GOOD)
    assert not cp2k.dropped_negative_integrals(ham.h2)
    assert (ham.h2 < -1e-8).sum() == 16
    assert _fci(ham) == pytest.approx(-17.488435975, abs=1e-8)


def test_cp2k71_signed_screen_is_detected_and_its_energy_is_wrong():
    """The default 7.1 file shares every Coulomb/exchange integral -- so its determinant
    still matches -- but has lost all negative couplings, and FCI is off by 0.17 Ha."""
    bad, good = fcidump.read(SIGNED), fcidump.read(GOOD)
    assert cp2k.dropped_negative_integrals(bad.h2)
    assert (bad.h2 < -1e-8).sum() == 0
    assert _det_energy(bad, 2) == pytest.approx(_det_energy(good, 2), abs=1e-7)
    assert abs(_fci(bad) - _fci(good)) > 0.1


def test_import_fcidump_warns_about_the_signed_screen(tmp_path, capsys):
    from embasi_qiskit_integration.cli import main

    assert main(["import_fcidump", str(SIGNED), str(tmp_path / "bad")]) == 0
    assert "signed EPS_INTEGRAL" in capsys.readouterr().err
    assert main(["import_fcidump", str(GOOD), str(tmp_path / "good")]) == 0
    assert "signed EPS_INTEGRAL" not in capsys.readouterr().err


def test_dropped_negative_integrals_ignores_tiny_spaces_and_jk_only_tensors():
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 0, 0] = h2[1, 1, 1, 1] = 0.7
    h2[0, 0, 1, 1] = h2[1, 1, 0, 0] = 0.6  # Coulomb only: never negative, never flagged
    assert not cp2k.dropped_negative_integrals(h2)


# --------------------------------------------------------------------------- #
# Input generation
# --------------------------------------------------------------------------- #
def _input(section=("FORCE_EVAL", "DFT"), multiplicity=1, n_act=4, n_val=8):
    return cp2k.active_space_input(
        ["O", "H", "H"],
        [[5.0, 5.0, 5.0], [5.0, 5.757, 5.587], [5.0, 4.243, 5.587]],
        spec=cp2k.ActiveSpaceSpec(n_act, 4, n_val, multiplicity=multiplicity),
        basis=BASIS,
        potential=POTENTIAL,
        data_dir="/data",
        section=section,
    )


def test_input_places_active_space_where_the_installed_cp2k_wants_it():
    modern = _input()
    legacy = _input(("FORCE_EVAL", "DFT", "PRINT"))
    assert "&PRINT" not in modern and "    &ACTIVE_SPACE" in modern
    assert "    &PRINT\n      &ACTIVE_SPACE" in legacy
    for text in (modern, legacy):
        assert "EPS_INTEGRAL -1.0E+10" in text  # keep negative integrals
        assert "ADDED_MOS 2" in text  # the two active virtuals
        assert "INACTIVE_ELECTRONS 4\n" in text
        assert "POTENTIAL GTH-PBE-q6" in text and "/data/BASIS_MOLOPT" in text
    with pytest.raises(ValueError, match="unsupported ACTIVE_SPACE location"):
        _input(("FORCE_EVAL", "PROPERTIES"))


def test_unrestricted_input_uses_per_spin_lists():
    """CP2K reads ACTIVE_ORBITALS / INACTIVE_ELECTRONS as per-spin lists under UKS."""
    text = _input(multiplicity=2, n_act=3, n_val=7)
    assert "UKS" in text and "MULTIPLICITY 2" in text
    assert "INACTIVE_ELECTRONS 2 2" in text and "ACTIVE_ORBITALS 4 4" in text


def test_spin_split_and_its_refusals():
    assert cp2k.ActiveSpaceSpec(3, 3, 7, multiplicity=2).spin_split() == ((2, 1), (2, 2))
    with pytest.raises(ValueError, match="odd or negative inactive"):
        cp2k.ActiveSpaceSpec(3, 3, 8).spin_split()
    # a triplet needs two unpaired electrons: two active electrons is fine, one is not
    assert cp2k.ActiveSpaceSpec(2, 3, 8, multiplicity=3).spin_split() == ((2, 0), (3, 3))
    with pytest.raises(ValueError, match="cannot hold multiplicity"):
        cp2k.ActiveSpaceSpec(1, 3, 7, multiplicity=3).spin_split()


def test_valence_from_potential():
    assert cp2k.valence_from_potential("GTH-PBE-q6") == 6
    with pytest.raises(ValueError):
        cp2k.valence_from_potential("ALL")


# --------------------------------------------------------------------------- #
# Section discovery, output parsing, refusal
# --------------------------------------------------------------------------- #
def _xml(tmp_path, under_print):
    inner = "<SECTION><NAME>ACTIVE_SPACE</NAME></SECTION>"
    if under_print:
        inner = f"<SECTION><NAME>PRINT</NAME>{inner}</SECTION>"
    path = tmp_path / "cp2k_input.xml"
    path.write_text(
        "<CP2K_INPUT><SECTION><NAME>FORCE_EVAL</NAME><SECTION><NAME>DFT</NAME>"
        f"{inner}</SECTION></SECTION></CP2K_INPUT>"
    )
    return path


def test_active_space_section_from_xml(tmp_path):
    assert cp2k.active_space_section_in_xml(_xml(tmp_path, True)) == ("FORCE_EVAL", "DFT", "PRINT")
    assert cp2k.active_space_section_in_xml(_xml(tmp_path, False)) == ("FORCE_EVAL", "DFT")


def _modern_xml(tmp_path):
    """The ACTIVE_SPACE part of a CP2K >= 2026.1 ``--xml`` (no INACTIVE_ELECTRONS, no
    ISOLATED_SYSTEM, a single-valued ACTIVE_ORBITALS, &ERI PERIODICITY)."""

    def kw(name, n_var):
        return (
            f"<KEYWORD><NAME type='default'>{name}</NAME>"
            f"<DATA_TYPE kind='integer'><N_VAR>{n_var}</N_VAR></DATA_TYPE></KEYWORD>"
        )

    path = tmp_path / "modern.xml"
    path.write_text(
        "<CP2K_INPUT><SECTION><NAME>FORCE_EVAL</NAME><SECTION><NAME>DFT</NAME>"
        "<SECTION><NAME>ACTIVE_SPACE</NAME>"
        + kw("ACTIVE_ELECTRONS", 1)
        + kw("ACTIVE_ORBITALS", 1)
        + kw("INACTIVE_ORBITALS", -1)
        + "<SECTION><NAME>ERI</NAME>"
        + kw("METHOD", 1)
        + kw("EPS_INTEGRAL", 1)
        + kw("PERIODICITY", 3)
        + "</SECTION></SECTION></SECTION></SECTION></CP2K_INPUT>"
    )
    return path


@pytest.mark.parametrize("multiplicity, n_act, n_val", [(1, 4, 8), (2, 3, 7)])
def test_input_writes_only_the_keywords_the_installed_cp2k_has(
    tmp_path, multiplicity, n_act, n_val
):
    """Regression: INACTIVE_ELECTRONS (gone in 2024.1), a per-spin ACTIVE_ORBITALS list
    (single-valued from 2024.1) and ISOLATED_SYSTEM (gone in 2026.1) were always written,
    and CP2K's parser aborts on any of them."""
    layout = cp2k.active_space_layout_in_xml(_modern_xml(tmp_path))
    assert layout.section == ("FORCE_EVAL", "DFT") and not layout.orbitals_per_spin
    text = cp2k.active_space_input(
        ["O", "H", "H"],
        [[5.0, 5.0, 5.0], [5.0, 5.757, 5.587], [5.0, 4.243, 5.587]],
        spec=cp2k.ActiveSpaceSpec(n_act, 4, n_val, multiplicity=multiplicity),
        basis=BASIS,
        potential=POTENTIAL,
        data_dir="/data",
        layout=layout,
    )
    assert "INACTIVE_ELECTRONS" not in text and "ISOLATED_SYSTEM" not in text
    assert "ACTIVE_ORBITALS 4\n" in text and "PERIODICITY 0 0 0" in text
    assert ("UKS" in text) == (multiplicity != 1)


def test_cp2k71_xml_keeps_the_legacy_keywords():
    """CP2K 7.1's own --xml (ACTIVE_SPACE section only): lists and ISOLATED_SYSTEM stay."""
    layout = cp2k.active_space_layout_in_xml(DATA / "cp2k71_active_space.xml")
    assert layout.section == ("FORCE_EVAL", "DFT", "PRINT")
    assert cp2k.LEGACY_KEYWORDS <= layout.keywords and layout.orbitals_per_spin
    assert layout.version == (7, 1)


def test_unrestricted_spec_is_refused_on_cp2k7_before_running(tmp_path, monkeypatch):
    """CP2K 7.1's UKS active space aborts -- or, about one run in seven, segfaults --
    after the SCF, so it is refused up front instead of run."""
    import subprocess

    from ase import Atoms

    data = tmp_path / "data"
    data.mkdir()
    (data / "BASIS_MOLOPT").write_text("")
    (data / "GTH_POTENTIALS").write_text("")
    layout = cp2k.active_space_layout_in_xml(DATA / "cp2k71_active_space.xml")
    monkeypatch.setattr(cp2k, "active_space_layout", lambda exe=None: layout)

    def never(*a, **k):
        raise AssertionError("CP2K must not be run")

    monkeypatch.setattr(subprocess, "run", never)
    with pytest.raises(RuntimeError, match="Spin option not yet available"):
        cp2k.run_active_space(
            Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)]),
            cp2k.ActiveSpaceSpec(3, 3, 7, multiplicity=2),
            workdir=tmp_path / "w",
            basis=BASIS,
            potential=POTENTIAL,
            data_dir=data,
            executable="cp2k",
        )


@pytest.mark.parametrize(
    "unit",
    ["(a.u.):", "[a.u.]:", "[hartree]"],
    ids=["7.1", "8.1-2024.1", "2025.1-on"],
)
def test_total_energy_in_every_cp2k_output_format(tmp_path, unit):
    """Each spelling CP2K has printed; regression: 8.1-2024.1's ``[a.u.]:`` was missed."""
    out = tmp_path / "e.out"
    out.write_text(
        f" ENERGY| Total FORCE_EVAL ( QS ) energy {unit}          -17.220800284869728\n"
        " ENERGY| Total FORCE_EVAL ( QS ) free energy [hartree]     -99.0\n"  # not this one
    )
    assert cp2k.read_total_energy(out) == pytest.approx(-17.220800284869728, abs=1e-15)


def test_abort_message_and_total_energy(tmp_path):
    out = tmp_path / "x.out"
    out.write_text(
        " *  /   \\       *\n * [ABORT]       *\n"
        " *  \\___/   Spin option not yet available   *\n *    |                *\n"
    )
    assert cp2k.abort_message(out) == "Spin option not yet available"
    assert cp2k.abort_message(DATA / "h2o_cas44_energy.out") is None
    assert cp2k.read_total_energy(DATA / "h2o_cas44_energy.out") == pytest.approx(
        -17.220800284869728, abs=1e-15
    )


def test_cp2k_backend_refuses_projection_embedding_and_points_to_its_route():
    from embasi_qiskit_integration.backends.registry import make_backend

    with pytest.raises(NotImplementedError, match="no ASI.*import_fcidump"):
        make_backend("cp2k", atoms=None, charge=0, cfg=object())


# --------------------------------------------------------------------------- #
# Against the real code
# --------------------------------------------------------------------------- #
@needs_cp2k
def test_real_cp2k_reports_where_active_space_lives():
    assert cp2k.active_space_section()[:2] == ("FORCE_EVAL", "DFT")


@needs_cp2k
def test_run_active_space_end_to_end(tmp_path):
    from ase import Atoms

    atoms = Atoms("OH2", positions=[(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)])
    run = cp2k.run_active_space(
        atoms, cp2k.ActiveSpaceSpec(4, 4, 8), workdir=tmp_path, basis=BASIS, potential=POTENTIAL
    )
    ham = fcidump.read(run.fcidump)
    assert _det_energy(ham, 2) == pytest.approx(run.total_energy, abs=1e-10)
    assert _fci(ham) == pytest.approx(_fci(fcidump.read(GOOD)), abs=1e-6)


@needs_cp2k
def test_unrestricted_active_space_abort_is_reported(tmp_path):
    from ase import Atoms

    if shutil.which("cp2k.ssmp") is None and not os.environ.get(cp2k.EXE_ENV):
        pytest.skip("needs cp2k.ssmp")
    atoms = Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)])
    try:
        run = cp2k.run_active_space(
            atoms,
            cp2k.ActiveSpaceSpec(3, 3, 7, multiplicity=2),
            workdir=tmp_path,
            basis=BASIS,
            potential=POTENTIAL,
        )
    except RuntimeError as exc:  # CP2K 7.1: its active-space module is restricted-only
        assert "Spin option not yet available" in str(exc)
    else:  # a newer CP2K: a genuine UHF FCIDUMP
        assert fcidump.read(run.fcidump).is_spin_dependent
