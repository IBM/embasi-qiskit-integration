# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The CPMD backend (``backends/cpmd.py``): readers for CPMD's Wannier/KS output.

Fixtures are real CPMD 4.3 output (water, PBE, isolated, Tuckerman Poisson solver): a
``PROPERTIES`` run with ``&PROP LOCALIZE`` + ``WANNIER DOS`` wrote the Wannier files, and a
separate ``KOHN-SHAM ENERGIES`` run from the same restart printed the eigenvalues.  The
two runs are independent, so the eigenvalue check below is a genuine cross-check of
``WANNIER_HAM``'s units and layout, not a round trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embasi_qiskit_integration.backends import cpmd

DATA = Path(__file__).parent / "data" / "cpmd"
HARTREE_EV = 27.211386245988


def test_wannier_hamiltonian_reproduces_the_independent_ks_eigenvalues():
    """Localisation is a unitary rotation of the occupied space, so WANNIER_HAM's
    eigenvalues must be the occupied KS eigenvalues of the separate KS-energies run.
    This pins WANNIER_HAM's unit (eV) and that the reader keeps the right block."""
    w = cpmd.read_wannier(DATA)
    eig, occ = cpmd.read_ks_eigenvalues(DATA / "ks_eigenvalues.out")
    occupied = eig[occ > 0]
    assert w.nstate == occupied.size == 4
    np.testing.assert_allclose(w.hamiltonian, w.hamiltonian.T, atol=1e-14)
    np.testing.assert_allclose(w.eigenvalues(), occupied, atol=1e-8)  # Hartree


def test_ks_eigenvalues_parse_values_and_occupations():
    eig, occ = cpmd.read_ks_eigenvalues(DATA / "ks_eigenvalues.out")
    np.testing.assert_allclose(occ, [2, 2, 2, 2, 0, 0])
    assert eig[3] * HARTREE_EV == pytest.approx(-7.0797781, abs=1e-6)  # the HOMO line
    assert np.all(np.diff(eig) > 0)


def test_appended_blocks_resolve_to_the_localised_one():
    """CPMD appends every localisation; the first block in these files is the
    pre-optimisation guess (total spread 6.1088, as CPMD prints), the last the result."""
    blocks = cpmd.read_wc_spread(DATA / "WC_SPREAD")
    assert [b.spreads.size for b in blocks] == [4, 4]
    assert blocks[0].spreads.sum() == pytest.approx(6.1088, abs=1e-3)
    assert blocks[-1].spreads.sum() < blocks[0].spreads.sum()  # localisation reduced it
    w = cpmd.read_wannier(DATA)
    np.testing.assert_allclose(w.spreads, blocks[-1].spreads)


def test_localised_water_is_two_lone_pairs_and_two_bonds():
    """The textbook picture, from the reader alone: two centres symmetric out of the
    molecular (yz) plane (lone pairs) and two in it, each displaced towards one H."""
    w = cpmd.read_wannier(DATA)
    assert w.symbols == ["O", "H", "H"]
    o, h1, h2 = w.positions
    off_plane = np.abs(w.centers[:, 0] - o[0]) > 0.1
    assert off_plane.sum() == 2
    lone = w.centers[off_plane]
    assert lone[0, 0] - o[0] == pytest.approx(-(lone[1, 0] - o[0]), abs=1e-4)
    bonds = w.centers[~off_plane]
    for h in (h1, h2):
        # each H has one in-plane centre on its side of the O
        assert any(np.sign(b[1] - o[1]) == np.sign(h[1] - o[1]) for b in bonds)
    # Fragment assignment: the lone pairs belong to O, and each O-H bond centre sits
    # past the bond midpoint here, nearest its own H (0.43 vs 0.53 Angstrom from O).
    nearest = w.nearest_atoms()
    assert nearest[off_plane].tolist() == [0, 0]
    assert sorted(nearest[~off_plane].tolist()) == [1, 2]


def test_wrong_state_count_is_rejected():
    with pytest.raises(ValueError, match="not a multiple of 5\\^2"):
        cpmd.read_wannier_ham(DATA / "WANNIER_HAM", 5)
    with pytest.raises(ValueError, match="not a multiple of nstate=3"):
        cpmd.read_wannier_centers(DATA / "WANNIER_CENTER", 3)


def test_mismatched_files_are_detected(tmp_path):
    """A WC_SPREAD from another run (different spreads) must not pair silently."""
    for name in ("WANNIER_HAM", "WANNIER_CENTER", "IONS+CENTERS.xyz"):
        (tmp_path / name).write_text((DATA / name).read_text())
    (tmp_path / "WC_SPREAD").write_text("0 4\n 1.0 1.0 1.0 1.0\n")
    with pytest.raises(ValueError, match="disagree on the last spreads"):
        cpmd.read_wannier(tmp_path)


@pytest.mark.parametrize(
    "text, match",
    [
        ("no eigenvalues here\n", "EIGENVALUES"),
        (" EIGENVALUES(EV) AND OCCUPATION:\n K POINT:   1  0.0 0.0 0.0\n", "k-point"),
    ],
    ids=["no-block", "k-points"],
)
def test_ks_eigenvalue_reader_rejects(tmp_path, text, match):
    out = tmp_path / "x.out"
    out.write_text(text)
    with pytest.raises(ValueError, match=match):
        cpmd.read_ks_eigenvalues(out)


_LSD_BLOCK = """
 EIGENVALUES(EV) AND OCCUPATION:
 ALPHA STATES:
       1    -25.5594566       1.00000000          2    -13.0690492       1.00000000
       3     -9.2097962       1.00000000          4     -7.4797781       1.00000000
 BETA STATES:
       1    -24.5594566       1.00000000          2    -12.0690492       1.00000000
       3     -8.2097962       1.00000000          4     -6.0797781       0.00000000
 ALPHA CHEMICAL POTENTIAL =                -7.4797781 EV
"""


def test_spin_polarised_eigenvalues_give_one_row_per_spin(tmp_path):
    """Regression: the ALPHA STATES: line ended the loop and two empty arrays came back."""
    out = tmp_path / "lsd.out"
    out.write_text(_LSD_BLOCK)
    eig, occ = cpmd.read_ks_eigenvalues(out)
    assert eig.shape == occ.shape == (2, 4)
    np.testing.assert_allclose(occ, [[1, 1, 1, 1], [1, 1, 1, 0]])
    assert eig[1, 0] * HARTREE_EV == pytest.approx(-24.5594566, abs=1e-6)


def test_not_converged_states_are_read_with_a_warning(tmp_path):
    """CPMD glues 'NC' to the value of a state that did not converge (-0.9067432NC)."""
    out = tmp_path / "nc.out"
    out.write_text(
        " EIGENVALUES(EV) AND OCCUPATION:\n"
        "       1    -25.5594566       2.00000000          2    -13.0690492       2.00000000\n"
        "       3     -0.9067432NC     0.00000000\n"
        " CHEMICAL POTENTIAL = -7 EV\n"
    )
    with pytest.warns(RuntimeWarning, match=r"states \[3\] as not converged"):
        eig, occ = cpmd.read_ks_eigenvalues(out)
    assert eig[2] * HARTREE_EV == pytest.approx(-0.9067432, abs=1e-6)
    np.testing.assert_allclose(occ, [2, 2, 0])


def test_cpmd_backend_refuses_with_the_reasons():
    from embasi_qiskit_integration.backends.registry import make_backend

    with pytest.raises(NotImplementedError, match="no AO basis.*no native active-space"):
        make_backend("cpmd", atoms=None, charge=0, cfg=object())
