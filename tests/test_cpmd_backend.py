# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The CPMD backend (``backends/cpmd.py``): readers for CPMD's Wannier/KS output.

Fixtures are real CPMD 4.3 output (water, PBE, isolated, Tuckerman Poisson solver): a
``PROPERTIES`` run with ``&PROP LOCALIZE`` + ``WANNIER DOS`` wrote the Wannier files, and a
separate ``KOHN-SHAM ENERGIES`` run from the same restart printed the eigenvalues.  The
two runs are independent, so the eigenvalue check below is a genuine cross-check of
``WANNIER_HAM``'s units and layout, not a round trip.
``lsd/`` is the same water with ``LSD`` (WANNIER files, plus excerpts of the run's
occupation block and of an LSD ``KOHN-SHAM ENERGIES`` run).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embasi_qiskit_integration.backends import cpmd

DATA = Path(__file__).parent / "data" / "cpmd"
HARTREE_EV = 27.211386245988
CLOSED = np.full(4, 2.0)  # the closed-shell fixture's occupations (4 doubly occupied)


def test_wannier_hamiltonian_reproduces_the_independent_ks_eigenvalues():
    """Localisation is a unitary rotation of the occupied space, so WANNIER_HAM's
    eigenvalues must be the occupied KS eigenvalues of the separate KS-energies run.
    This pins WANNIER_HAM's unit (eV) and that the reader keeps the right block."""
    w = cpmd.read_wannier(DATA, occupations=CLOSED)
    eig, occ = cpmd.read_ks_eigenvalues(DATA / "ks_eigenvalues.out")
    occupied = eig[occ > 0]
    assert w.nstate == occupied.size == 4
    np.testing.assert_allclose(w.hamiltonian, w.hamiltonian.T, atol=1e-14)
    np.testing.assert_allclose(w.eigenvalues(), occupied, atol=1e-8)  # Hartree


def test_lsd_wannier_hamiltonian_is_rescaled_by_its_occupations():
    """An LSD run writes WANNIER_HAM scaled by f=1, half the closed-shell eV; with the
    run's occupations it reproduces the independent LSD KS eigenvalues (alpha + beta),
    and without them the zero alpha/beta block makes the reader refuse, not halve."""
    lsd = DATA / "lsd"
    occ = cpmd.read_occupations(lsd / "prop_occupations.out")
    np.testing.assert_array_equal(occ, np.ones(8))
    w = cpmd.read_wannier(lsd, output=lsd / "prop_occupations.out")
    eig, f = cpmd.read_ks_eigenvalues(lsd / "ks_eigenvalues.out")
    occupied = np.sort(np.concatenate([eig[0][f[0] > 0], eig[1][f[1] > 0]]))
    np.testing.assert_allclose(w.eigenvalues(), occupied, atol=1e-8)
    with pytest.raises(ValueError, match="LSD"):
        cpmd.read_wannier(lsd, occupations=np.full(8, 2.0))
    with pytest.raises(ValueError, match="exactly one"):
        cpmd.read_wannier(lsd)


def test_open_shell_lsd_layout_is_refused_without_occupations(tmp_path):
    """A doublet's alpha/beta blocks split unevenly (here 4 + 3, cut from the real LSD
    matrix), so the refusal must not assume nsup == nsdown; with f=1 it reads exactly."""
    raw = np.array(cpmd._floats((DATA / "lsd" / "WANNIER_HAM").read_text())[-64:]).reshape(8, 8)
    (tmp_path / "WANNIER_HAM").write_text("\n".join(" ".join(map(str, r)) for r in raw[:7, :7]))
    with pytest.raises(ValueError, match="states 1-4 and 5-7 do not couple"):
        cpmd.read_wannier_ham(tmp_path / "WANNIER_HAM", 7, np.full(7, 2.0))
    h = cpmd.read_wannier_ham(tmp_path / "WANNIER_HAM", 7, np.ones(7))
    np.testing.assert_allclose(h, raw[:7, :7] / (0.5 * HARTREE_EV), atol=1e-12)


def test_read_occupations_closed_shell_and_its_refusals(tmp_path):
    out = tmp_path / "cpmd.out"
    out.write_text(" NUMBER OF STATES:      15\n OCCUPATION\n" + "  2.0" * 13 + "\n  2.0  1.0\n\n")
    np.testing.assert_array_equal(cpmd.read_occupations(out), [2.0] * 14 + [1.0])
    out.write_text(" NUMBER OF STATES:      4\n OCCUPATION\n  2.0  2.0\n\n")
    with pytest.raises(ValueError, match="read 2 occupations for 4"):
        cpmd.read_occupations(out)
    # 1.96 prints as 2.0; the 5-decimal electron count gives it away
    out.write_text(
        " NUMBER OF STATES:  2\n NUMBER OF ELECTRONS:  3.96000\n OCCUPATION\n  2.0  2.0\n\n"
    )
    with pytest.raises(ValueError, match="they are rounded"):
        cpmd.read_occupations(out)
    out.write_text(" NUMBER OF STATES:      2\n OCCUPATION\n  2.0  0.7\n\n")
    with pytest.raises(ValueError, match="one decimal only"):
        cpmd.read_occupations(out)


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
    w = cpmd.read_wannier(DATA, occupations=CLOSED)
    np.testing.assert_allclose(w.spreads, blocks[-1].spreads)


def test_localised_water_is_two_lone_pairs_and_two_bonds():
    """The textbook picture, from the reader alone: two centres symmetric out of the
    molecular (yz) plane (lone pairs) and two in it, each displaced towards one H."""
    w = cpmd.read_wannier(DATA, occupations=CLOSED)
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
        cpmd.read_wannier_ham(DATA / "WANNIER_HAM", 5, np.full(5, 2.0))
    with pytest.raises(ValueError, match="not a multiple of nstate=3"):
        cpmd.read_wannier_centers(DATA / "WANNIER_CENTER", 3)


def test_mismatched_files_are_detected(tmp_path):
    """A WC_SPREAD from another run (different spreads) must not pair silently."""
    for name in ("WANNIER_HAM", "WANNIER_CENTER", "IONS+CENTERS.xyz"):
        (tmp_path / name).write_text((DATA / name).read_text())
    (tmp_path / "WC_SPREAD").write_text("0 4\n 1.0 1.0 1.0 1.0\n")
    with pytest.raises(ValueError, match="disagree on the last spreads"):
        cpmd.read_wannier(tmp_path, occupations=CLOSED)


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
