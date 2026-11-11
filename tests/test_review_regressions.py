# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Regressions for run-time bugs found in review of the QM-backend branch.

Each test reproduces the failing input that was confirmed before the fix.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.hamiltonian import fcidump


# --------------------------------------------------------------------------- #
# Footing: h_core must include ECP terms (pre-existing; 50.5 Ha on HI/def2-SVP)
# --------------------------------------------------------------------------- #
def test_layer_footing_hcore_includes_the_ecp():
    from pyscf import gto, scf

    from embasi_qiskit_integration.backends.pyscf import pyscf_layer_footing

    mol = gto.M(atom="I 0 0 0; H 0 0 1.6", basis="def2-svp", ecp={"I": "def2-svp"}, verbose=0)
    assert mol._ecp  # the test is only meaningful with a real ECP
    layer = SimpleNamespace(atoms=SimpleNamespace(calc=SimpleNamespace(mol=mol)))
    hcore, enuc = pyscf_layer_footing(layer)
    np.testing.assert_allclose(hcore, scf.RHF(mol).get_hcore(), atol=1e-12)
    assert enuc == pytest.approx(mol.energy_nuc())


# --------------------------------------------------------------------------- #
# Callback backend: label search parity and the per-layer ev_sum cache
# --------------------------------------------------------------------------- #
def test_pyscf_callback_label_search_matches_pyscf():
    pytest.importorskip("embasi.atoms_embedding_asi", exc_type=ImportError)
    from ase import Atoms

    from embasi_qiskit_integration.backends.registry import make_backend

    cfg = SimpleNamespace(
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
    atoms = Atoms("OH2", positions=[(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)])
    direct = make_backend("pyscf", atoms, 0, cfg).integrals
    cb = make_backend("pyscf-callback", atoms, 0, cfg).integrals
    for labels in (["O 2px"], ["O"], ["O 2p", "H 1s"]):
        np.testing.assert_array_equal(cb.search_ao_label(labels), direct.search_ao_label(labels))


def test_layer_evaluator_drops_embasis_cached_eigenvalue_sum():
    """EmbASI caches `ev_sum` per layer; a reused layer must not carry it across densities."""
    from embasi_qiskit_integration.backends.callback import LayerEvaluator

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
# FCIDUMP: logical header values, IUHF, orbital-energy lines
# --------------------------------------------------------------------------- #
def _write(path: Path, header: str, body: str) -> Path:
    path.write_text(header + "\n" + body)
    return path


def test_logical_header_values_are_read(tmp_path):
    p = _write(
        tmp_path / "x.fcidump",
        "&FCI NORB=2,NELEC=2,MS2=0,UHF=.FALSE.,TREL=.FALSE.,\n ORBSYM=1,1,\n ISYM=1,\n &END",
        " 0.5 1 1 1 1\n 0.5 2 2 2 2\n-1.0 1 1 0 0\n-0.5 2 2 0 0\n 0.0 0 0 0 0\n",
    )
    ham = fcidump.read(p)
    assert ham.norb == 2 and not ham.is_spin_dependent


def test_iuhf_flag_and_logical_uhf_select_the_unrestricted_reader(tmp_path):
    body = " 0.5 1 1 1 1\n 0.5 3 3 3 3\n 0.4 1 1 3 3\n-1.0 1 1 0 0\n-0.9 3 3 0 0\n 0.0 0 0 0 0\n"
    for flag in ("IUHF=1", "UHF=.TRUE."):
        p = _write(tmp_path / f"{flag[:4]}.fcidump", f"&FCI NORB=2,NELEC=2,MS2=0,{flag},\n /", body)
        ham = fcidump.read(p)
        assert ham.is_spin_dependent, flag
        assert ham.h1b[0, 0] == pytest.approx(-0.9)


def test_orbital_energy_lines_do_not_overwrite_matrix_elements(tmp_path):
    """`eps i 0 0 0` lines used to land in h1[i-1, -1] (the last column)."""
    body = (
        " 0.5 1 1 1 1\n 0.5 3 3 3 3\n 0.4 1 1 3 3\n"
        "-1.0 1 1 0 0\n-0.3 2 2 0 0\n-0.9 3 3 0 0\n-0.2 4 4 0 0\n"
        "-7.77 2 0 0 0\n-6.66 4 0 0 0\n 0.0 0 0 0 0\n"
    )
    p = _write(tmp_path / "eps.fcidump", "&FCI NORB=2,NELEC=2,MS2=0,UHF=1,\n /", body)
    ham = fcidump.read(p)
    np.testing.assert_allclose(np.diag(ham.h1a), [-1.0, -0.3])
    np.testing.assert_allclose(np.diag(ham.h1b), [-0.9, -0.2])


# --------------------------------------------------------------------------- #
# CP2K: stale FCIDUMP, charge, lean guard
# --------------------------------------------------------------------------- #
def test_cp2k_input_writes_the_charge():
    from embasi_qiskit_integration.backends import cp2k

    text = cp2k.active_space_input(
        ["O", "H"],
        [[5, 5, 5], [5, 5, 5.97]],
        spec=cp2k.ActiveSpaceSpec(2, 2, 6),
        basis={"O": "b", "H": "b"},
        potential={"O": "p", "H": "p"},
        data_dir="/d",
        charge=1,
    )
    assert "    CHARGE 1\n" in text
    neutral = cp2k.active_space_input(
        ["O", "H"],
        [[5, 5, 5], [5, 5, 5.97]],
        spec=cp2k.ActiveSpaceSpec(2, 2, 6),
        basis={"O": "b", "H": "b"},
        potential={"O": "p", "H": "p"},
        data_dir="/d",
    )
    assert "CHARGE" not in neutral


def test_cp2k_run_never_returns_a_stale_fcidump(tmp_path, monkeypatch):
    """A crash that writes no FCIDUMP must not hand back the previous run's file."""
    import subprocess

    from ase import Atoms

    from embasi_qiskit_integration.backends import cp2k

    data = tmp_path / "data"
    data.mkdir()
    (data / "BASIS_MOLOPT").write_text("")
    (data / "GTH_POTENTIALS").write_text("")
    work = tmp_path / "work"
    work.mkdir()
    (work / "active.fcidump").write_text("stale")
    monkeypatch.setattr(cp2k, "active_space_section", lambda exe=None: ("FORCE_EVAL", "DFT"))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=139),  # segfault
    )
    atoms = Atoms("OH", positions=[(0, 0, 0), (0, 0, 0.97)])
    with pytest.raises(RuntimeError, match="exited with status 139"):
        cp2k.run_active_space(
            atoms,
            cp2k.ActiveSpaceSpec(2, 2, 6),
            workdir=work,
            basis={"O": "b", "H": "b"},
            potential={"O": "p", "H": "p"},
            data_dir=data,
            executable="cp2k",
        )
    assert not (work / "active.fcidump").exists()


def test_dropped_negative_integrals_unchanged_by_the_lean_grid():
    from embasi_qiskit_integration.backends import cp2k

    data = Path(__file__).parent / "data" / "cp2k"
    assert cp2k.dropped_negative_integrals(
        fcidump.read(data / "h2o_cas44_cp2k71_signed_screen.fcidump").h2
    )
    assert not cp2k.dropped_negative_integrals(fcidump.read(data / "h2o_cas44.fcidump").h2)


# --------------------------------------------------------------------------- #
# Backend validation runs before EmbASI/MPI is imported
# --------------------------------------------------------------------------- #
def test_refused_qm_code_reports_its_reason_before_embasi_is_imported(monkeypatch):
    import builtins
    import sys

    from embasi_qiskit_integration.embedding import EmbeddingWorkflow, build_adapter

    real_import = builtins.__import__

    def no_embasi(name, *args, **kwargs):
        if name == "embasi" or name.startswith("embasi."):
            raise ImportError("simulated: no MPI library")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(builtins, "__import__", no_embasi)
    with pytest.raises(NotImplementedError, match="plane-wave code"):
        build_adapter(EmbeddingWorkflow(_cli_parse_args=False, qm_code="espresso"), parallel=False)


# --------------------------------------------------------------------------- #
# SIESTA: --scf_max_cycle honoured, a Gaussian --basis refused
# --------------------------------------------------------------------------- #
def test_siesta_honours_scf_max_cycle_and_refuses_a_gaussian_basis(monkeypatch, tmp_path):
    from ase import Atoms

    from embasi_qiskit_integration.backends import siesta as sb
    from embasi_qiskit_integration.backends.registry import make_backend

    monkeypatch.setenv(sb.PSEUDO_ENV, str(tmp_path))
    monkeypatch.setenv(sb.COMMAND_ENV, "siesta < PREFIX.fdf > PREFIX.out")
    cfg = SimpleNamespace(
        basis="sto-3g",
        spin=0,
        unrestricted=False,
        xc_ll="PBE",
        xc_hl="PBE",
        scf_max_cycle=300,
        siesta_basis="DZP",
        siesta_mesh_cutoff=200.0,
    )
    calc = sb.siesta_calculator(cfg, xc="PBE", spin_polarised=False, directory="x")
    assert calc.parameters["fdf_arguments"]["MaxSCFIterations"] == 300
    cfg.basis = "cc-pvdz"
    atoms = Atoms(
        "OH2", positions=[(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)], cell=[10] * 3
    )
    with pytest.raises(ValueError, match="--siesta_basis"):
        make_backend("siesta", atoms, 0, cfg)
