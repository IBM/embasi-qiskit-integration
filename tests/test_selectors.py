# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`embasi_qiskit_integration.selectors`.

Pure numpy -- no EmbASI, no PySCF -- so these run in the default suite.  The
selector's contract is: given canonical subsystem-A orbitals ``(coeff, energy,
n_occ)`` and the AO overlap ``S``, return the active column indices as
``[0, n_occ)`` (all occupied A) followed by the virtuals whose *fragment
population* survives the gap/cap cut.

We build synthetic orbitals directly, so the fragment population of each virtual
is known by construction and the cut is checked against a value we set, not a
value the physics happens to produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.selectors import (
    concentric_selector,
    fragment_ao_indices,
)


def _orthonormal(n: int, seed: int = 0) -> np.ndarray:
    """A random orthonormal (nao, nao) matrix via QR -- columns are MO coeffs."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def test_orthonormal_basis_ranks_virtuals_by_fragment_population():
    # S = I so fragment population of virtual v is sum_{mu in frag} C[mu, v]^2:
    # exactly the norm of the virtual restricted to the fragment AOs.  Build the
    # coeff matrix so that population is a clean, strictly-decreasing ladder.
    nao = 6
    n_occ = 2
    frag = np.array([0, 1])  # two fragment AOs

    coeff = np.zeros((nao, nao))
    # Occupied block: arbitrary but orthonormal within their own 2D subspace.
    coeff[2, 0] = 1.0
    coeff[3, 1] = 1.0
    # Virtuals (columns 2..5): fragment weight = C[0,v]^2 + C[1,v]^2 on AO 0/1,
    # remainder on non-fragment AOs so each column stays unit norm.
    #  v2: w=1.00   v3: w=0.64   v4: w=0.04   v5: w=0.00
    ladder = [1.00, 0.64, 0.04, 0.00]
    filler_ao = [4, 5, 4, 5]
    for j, (w, fao) in enumerate(zip(ladder, filler_ao)):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        coeff[filler_ao[j], v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    sel = concentric_selector(s, frag, gap_tol=0.1)
    active = sel(coeff, energy, n_occ)

    # Occupied block is always kept, in order.
    assert list(active[:n_occ]) == [0, 1]
    # The first gap >= 0.1 from the top is w2->w3 (0.36) -- wait, that's the
    # top pair; the ladder is 1.00, 0.64, 0.04, 0.00, so gaps are 0.36, 0.60,
    # 0.04.  First gap >= gap_tol is after index 0 (0.36), keeping just v2.
    assert set(active[n_occ:]) == {2}


def test_gap_cut_keeps_the_tight_shell():
    # A clear two-shell structure: two strongly-coupled virtuals (~0.9) then a
    # long weak tail (~0.05).  The big gap sits between the shells; the selector
    # must keep exactly the strong shell.
    nao = 8
    n_occ = 2
    frag = np.array([0])
    coeff = np.zeros((nao, nao))
    coeff[6, 0] = 1.0
    coeff[7, 1] = 1.0
    weights = [0.92, 0.88, 0.06, 0.05, 0.04, 0.03]  # 2 strong, 4 weak
    for j, w in enumerate(weights):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        # spread the remainder onto a distinct non-fragment AO per column
        coeff[2 + (j % 4), v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    sel = concentric_selector(s, frag, gap_tol=0.1)
    active = sel(coeff, energy, n_occ)
    kept_virt = set(active[n_occ:])
    # Only the two strong virtuals clear the gap.
    assert kept_virt == {2, 3}


def test_max_virtual_caps_after_the_gap():
    nao = 7
    n_occ = 1
    frag = np.array([0])
    coeff = np.zeros((nao, nao))
    coeff[6, 0] = 1.0
    weights = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # smooth ramp, no clear gap
    for j, w in enumerate(weights):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        coeff[1 + (j % 5), v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    # gap_tol high enough that nothing cuts on the gap; cap does the work.
    sel = concentric_selector(s, frag, gap_tol=0.5, max_virtual=2)
    active = sel(coeff, energy, n_occ)
    assert active[0] == 0                       # occupied kept
    assert len(active) - n_occ == 2             # capped at 2 virtuals
    # the two kept are the two strongest-coupled (columns 1 and 2).
    assert set(active[n_occ:]) == {1, 2}


def test_no_significant_gap_keeps_everything():
    # A smooth ramp with no gap >= gap_tol: the selector must refuse to truncate
    # on noise and keep every virtual.
    nao = 6
    n_occ = 1
    frag = np.array([0])
    coeff = np.zeros((nao, nao))
    coeff[5, 0] = 1.0
    weights = [0.60, 0.58, 0.56, 0.54, 0.52]
    for j, w in enumerate(weights):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        coeff[1 + (j % 4), v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    sel = concentric_selector(s, frag, gap_tol=0.1)  # gaps are all 0.02
    active = sel(coeff, energy, n_occ)
    assert len(active) == nao                        # all occ + all virt
    assert set(active[n_occ:]) == set(range(n_occ, nao))


def test_min_virtual_floor():
    # Even when the first gap would keep just one virtual, min_virtual keeps more.
    nao = 6
    n_occ = 1
    frag = np.array([0])
    coeff = np.zeros((nao, nao))
    coeff[5, 0] = 1.0
    weights = [0.9, 0.2, 0.15, 0.1, 0.05]  # big gap after the first virtual
    for j, w in enumerate(weights):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        coeff[1 + (j % 4), v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    sel = concentric_selector(s, frag, gap_tol=0.1, min_virtual=3)
    active = sel(coeff, energy, n_occ)
    assert len(active) - n_occ == 3
    # the three strongest by fragment weight (columns 1, 2, 3).
    assert set(active[n_occ:]) == {1, 2, 3}


def test_no_virtuals_returns_occupied_only():
    nao = 3
    n_occ = 3  # fully occupied, no virtuals
    frag = np.array([0])
    coeff = _orthonormal(nao)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)
    sel = concentric_selector(s, frag)
    active = sel(coeff, energy, n_occ)
    assert list(active) == [0, 1, 2]


def test_nonorthogonal_overlap_uses_S_weighted_population():
    # With a non-identity S the population is w_v = sum_{mu in frag}(S C)_{mu v}
    # C_{mu v}, not sum C^2.  Verify the selector uses the S-weighted form by
    # comparing against a hand-computed ranking.
    nao = 4
    n_occ = 1
    frag = np.array([0, 1])
    rng = np.random.default_rng(3)
    a = rng.standard_normal((nao, nao))
    s = a @ a.T + nao * np.eye(nao)  # SPD overlap
    # S-orthonormalize a random set of columns: C^T S C = I.
    x = rng.standard_normal((nao, nao))
    # Lowdin-style: C = L^{-T} Q  with S = L L^T is fussy; instead use eig.
    w, u = np.linalg.eigh(s)
    s_inv_half = u @ np.diag(1.0 / np.sqrt(w)) @ u.T
    q, _ = np.linalg.qr(x)
    coeff = s_inv_half @ q  # C^T S C = Q^T Q = I
    energy = np.arange(nao, dtype=float)

    # Hand-compute the fragment population ranking (same form the selector uses).
    c_virt = coeff[:, n_occ:]
    sc = s @ c_virt
    pop = np.einsum("mv,mv->v", sc[frag, :], c_virt[frag, :])
    expected_order = np.argsort(-pop)

    sel = concentric_selector(s, frag, gap_tol=1e9, max_virtual=1)  # keep top-1
    active = sel(coeff, energy, n_occ)
    kept = active[n_occ:]
    assert kept.tolist() == [n_occ + int(expected_order[0])]


def test_fragment_ao_indices_from_mock_mol():
    # fragment_ao_indices only needs mol.aoslice_by_atom(); fake it.
    class _Mol:
        def aoslice_by_atom(self):
            # columns 2 and 3 are (ao_start, ao_end) per atom.
            return np.array(
                [
                    [0, 0, 0, 2],   # atom 0 -> AOs 0,1
                    [0, 0, 2, 5],   # atom 1 -> AOs 2,3,4
                    [0, 0, 5, 6],   # atom 2 -> AO 5
                ]
            )

    idx = fragment_ao_indices(_Mol(), [0, 2])
    assert idx.tolist() == [0, 1, 5]

    empty = fragment_ao_indices(_Mol(), [])
    assert empty.size == 0
