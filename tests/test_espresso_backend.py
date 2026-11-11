# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The Quantum ESPRESSO backend (``backends/espresso.py``): QE/WEST output readers.

QE, WEST and westpy are GPL; none is imported.  ``atomic_proj.xml`` fixtures are real
``projwfc.x`` output (QE 7.6; water, and an OH doublet with LSDA).  WEST is not
installable here, so the ``wfreq.json`` reader is checked on a file written in WEST's
documented layout from a *known* Hamiltonian, then solved: the round trip must give
that Hamiltonian's own FCI energy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embasi_qiskit_integration.backends import espresso as qe
from embasi_qiskit_integration.contract import EmbeddedHamiltonian

DATA = Path(__file__).parent / "data" / "espresso"


# --------------------------------------------------------------------------- #
# atomic_proj.xml (real projwfc.x output)
# --------------------------------------------------------------------------- #
def test_atomic_proj_closed_shell_water():
    a = qe.read_atomic_proj(DATA / "h2o_atomic_proj.xml")
    assert a.eigenvalues.shape == (1, 8)
    assert a.projections.shape == (1, 6, 8)  # O 2s,2p(3) + 2x H 1s
    assert a.n_electrons == 8.0
    # QE's scf.out printed the HOMO at -7.3188 eV; band 4 of 8 electrons / 2.
    assert a.eigenvalues[0, 3] * 27.211386245988 == pytest.approx(-7.3188, abs=1e-3)
    s = a.overlaps[0]
    np.testing.assert_allclose(s, s.conj().T, atol=1e-12)


def test_atomic_proj_lsda_has_two_spin_channels():
    a = qe.read_atomic_proj(DATA / "oh_lsda_atomic_proj.xml")
    assert a.eigenvalues.shape == (2, 8)
    assert a.projections.shape == (2, 5, 8)
    assert a.overlaps.shape == (2, 5, 5)
    assert a.n_electrons == 7.0
    assert not np.allclose(a.eigenvalues[0], a.eigenvalues[1])  # genuinely polarised


def test_projectability_matches_projwfc_own_output():
    """``projectability`` is QE's own ``|psi|^2`` (printed to 3 digits in projwfc.x's
    output for these runs).  Regression: combining the Loewdin-orthogonalised projections
    with the non-orthogonal OVPS overlap gave 0.81/0.76/0.97/0.99 for water's occupied
    bands.  The occupied bands are ~98% spanned; the virtuals much less."""
    water = qe.read_atomic_proj(DATA / "h2o_atomic_proj.xml").projectability()
    np.testing.assert_allclose(
        water[0], [0.988, 0.976, 0.984, 0.987, 0.329, 0.257, 0.208, 0.003], atol=6e-4
    )
    oh = qe.read_atomic_proj(DATA / "oh_lsda_atomic_proj.xml").projectability()
    np.testing.assert_allclose(
        oh,
        [
            [0.992, 0.977, 0.997, 0.994, 0.265, 0.290, 0.000, 0.001],
            [0.990, 0.977, 0.991, 0.984, 0.238, 0.272, 0.002, 0.004],
        ],
        atol=6e-4,
    )


def test_atomic_proj_rejects_another_file(tmp_path):
    bad = tmp_path / "x.xml"
    bad.write_text("<ROOT/>")
    with pytest.raises(ValueError, match="no HEADER"):
        qe.read_atomic_proj(bad)


# --------------------------------------------------------------------------- #
# wfreq.json (WEST/QDET), in WEST's layout, from a known Hamiltonian
# --------------------------------------------------------------------------- #
def _pairs(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def _known(nspin):
    """A small spin-(un)restricted Hamiltonian from real integrals (H2O sto-3g / OH)."""
    from pyscf import ao2mo, gto, scf

    if nspin == 1:
        mol = gto.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run()
        c = (mf.mo_coeff,)
    else:
        mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
        mf = scf.UHF(mol).run(conv_tol=1e-12)
        c = tuple(mf.mo_coeff)
    n = c[0].shape[1]
    hcore = mf.get_hcore()
    h1 = [ci.T @ hcore @ ci for ci in c]
    eri = {}
    for s1 in range(nspin):
        for s2 in range(nspin):
            a, b = c[s1], c[s2]
            eri[(s1, s2)] = ao2mo.general(mol, (a, a, b, b), compact=False).reshape((n,) * 4)
    occ = (
        np.asarray(mf.mo_occ, dtype=float).reshape(nspin, n)
        if nspin == 2
        else (np.asarray(mf.mo_occ, dtype=float)[None, :])
    )
    return n, h1, eri, occ, mol, mf


def _write_wfreq(path, n, h1, eri, occ, *, two_body="eri_w"):
    """WEST's documented wfreq.json layout (energies in eV)."""
    ev = qe._HARTREE_EV
    nspin = len(h1)
    pairs = _pairs(n)
    key = qe._spin_key
    qdet = {
        "h1e": {key(s): [h1[s][i, j] * ev for i, j in pairs] for s in range(nspin)},
        two_body: {
            key(s1): {
                key(s2): {
                    f"pair{p + 1:09d}": [eri[(s1, s2)][i, j, k, m] * ev for k, m in pairs]
                    for p, (i, j) in enumerate(pairs)
                }
                for s2 in range(nspin)
            }
            for s1 in range(nspin)
        },
    }
    data = {
        "system": {"electron": {"nspin": nspin}},
        "input": {"wfreq_control": {"qp_bands": [list(range(1, n + 1))]}},
        "output": {"Q": {key(s): {"occupation": occ[s].tolist()} for s in range(nspin)}},
        "qdet": qdet,
    }
    Path(path).write_text(json.dumps(data))


def _fci(ham):
    from embasi_qiskit_integration.solvers import FCISolver

    return FCISolver().solve(ham).energy


def test_wfreq_restricted_roundtrip(tmp_path):
    n, h1, eri, occ, mol, mf = _known(1)
    path = tmp_path / "wfreq.json"
    _write_wfreq(path, n, h1, eri, occ)
    q = qe.read_wfreq_json(path)
    np.testing.assert_allclose(q.h1[0], h1[0], atol=1e-12)
    np.testing.assert_allclose(q.eri[(0, 0)], eri[(0, 0)], atol=1e-12)
    ham = qe.qdet_to_embedded_hamiltonian(q)
    assert ham.nelec == (2, 2) and ham.e_core == 0.0
    ref = EmbeddedHamiltonian(h1=h1[0], h2=eri[(0, 0)], e_core=0.0, nelec=(2, 2))
    assert _fci(ham) == pytest.approx(_fci(ref), abs=1e-10)
    # Adding E_nuc back recovers the molecule's FCI total -- the integrals are physical.
    from pyscf import fci

    e_ref = fci.FCI(mf).kernel()[0]
    assert _fci(ham) + mol.energy_nuc() == pytest.approx(e_ref, abs=1e-8)


def test_wfreq_unrestricted_keeps_the_spin_blocks_and_solves(tmp_path):
    n, h1, eri, occ, _mol, _mf = _known(2)
    path = tmp_path / "wfreq.json"
    _write_wfreq(path, n, h1, eri, occ)
    ham = qe.qdet_to_embedded_hamiltonian(qe.read_wfreq_json(path))
    assert ham.nelec == (5, 4)
    assert ham.is_spin_dependent
    np.testing.assert_allclose(ham.h1a, h1[0], atol=1e-12)
    np.testing.assert_allclose(ham.h2_spin[1], eri[(0, 1)], atol=1e-12)
    ref = EmbeddedHamiltonian(
        h1=0.5 * (h1[0] + h1[1]),
        h2=eri[(0, 0)],
        e_core=0.0,
        nelec=(5, 4),
        h1a=h1[0],
        h1b=h1[1],
        h2_spin=(eri[(0, 0)], eri[(0, 1)], eri[(1, 1)]),
    )
    assert _fci(ham) == pytest.approx(_fci(ref), abs=1e-10)


def test_wfreq_job_then_solve_cli(tmp_path):
    from embasi_qiskit_integration import ipc
    from embasi_qiskit_integration.cli import main

    n, h1, eri, occ, _mol, _mf = _known(2)
    path = tmp_path / "wfreq.json"
    _write_wfreq(path, n, h1, eri, occ)
    job = tmp_path / "job"
    qe.write_job_from_wfreq(path, job)
    assert main(["solve", str(job), "--solver", "fci"]) == 0
    direct = _fci(qe.qdet_to_embedded_hamiltonian(qe.read_wfreq_json(path)))
    assert ipc.read_result(job).energy == pytest.approx(direct, abs=1e-10)


def test_wfreq_selects_the_two_body_variant_and_reports_missing_ones(tmp_path):
    n, h1, eri, occ, *_ = _known(1)
    path = tmp_path / "wfreq.json"
    _write_wfreq(path, n, h1, eri, occ, two_body="eri_vc")
    assert qe.read_wfreq_json(path, two_body="eri_vc").two_body == "eri_vc"
    with pytest.raises(ValueError, match="no qdet.eri_w; available: eri_vc"):
        qe.read_wfreq_json(path)


def test_wfreq_rejects_a_non_qdet_file(tmp_path):
    path = tmp_path / "wfreq.json"
    path.write_text(json.dumps({"system": {}}))
    with pytest.raises(ValueError, match="not a QDET wfreq.json"):
        qe.read_wfreq_json(path)


def test_odd_restricted_electron_count_is_refused_not_dropped(tmp_path):
    n, h1, eri, occ, *_ = _known(1)
    occ = occ.copy()
    occ[0, 0] = 1.0  # 3 electrons in a spin-restricted file
    path = tmp_path / "wfreq.json"
    _write_wfreq(path, n, h1, eri, occ)
    with pytest.raises(ValueError, match="odd 3 electrons"):
        qe.read_wfreq_json(path).nelec


# --------------------------------------------------------------------------- #
# The registered backend refuses, with the reason and the alternative
# --------------------------------------------------------------------------- #
def test_espresso_backend_refuses_with_the_qdet_alternative():
    from embasi_qiskit_integration.backends.registry import make_backend

    with pytest.raises(NotImplementedError, match="no AO basis.*import_fcidump"):
        make_backend("espresso", atoms=None, charge=0, cfg=object())


def test_west_fcidump_is_read_as_rydberg(tmp_path):
    """WEST's FCIDUMP carries its integrals in Rydberg (westpy halves them on read)."""
    from embasi_qiskit_integration import ipc
    from embasi_qiskit_integration.hamiltonian import fcidump

    rng = np.random.default_rng(3)
    h1 = rng.normal(size=(2, 2))
    h1 = h1 + h1.T
    h2 = np.zeros((2, 2, 2, 2))
    h2[0, 0, 0, 0], h2[1, 1, 1, 1], h2[0, 0, 1, 1] = 0.7, 0.6, 0.4
    h2[1, 1, 0, 0] = 0.4
    for p, q, r, s in [(0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0)]:
        h2[p, q, r, s] = 0.1
    ham = EmbeddedHamiltonian(h1=h1, h2=h2, e_core=0.0, nelec=(1, 1))
    path = tmp_path / "FCIDUMP"
    fcidump.write(ham.model_copy(update={"h1": 2 * h1, "h2": 2 * h2}), path)
    fcidump._sidecar_path(path).unlink(missing_ok=True)
    got = qe.read_west_fcidump(path)
    np.testing.assert_allclose(got.h1, h1, atol=1e-12)
    np.testing.assert_allclose(got.h2, h2, atol=1e-12)
    job = qe.write_job_from_west_fcidump(path, tmp_path / "job")
    np.testing.assert_allclose(ipc.read_job(job.parent).h1, h1, atol=1e-12)
