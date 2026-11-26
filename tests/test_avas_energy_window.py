# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Opt-in correlation-window virtual ranking for AVAS (``--avas_energy_window``).

The butyronitrile SIESTA-vs-PySCF diagnosis found that AO-AVAS virtual weights
saturate on SIESTA's DZP basis: several high-energy N-localised pi* orbitals
carry weight 1.000, so the standard "keep the highest-weight virtuals" rule
discards a C-localised pi* pair that is only ~0.04 Ha above the occupied pi.
The window policy keeps the highest-weight virtuals only among candidates whose
F_emb energy is at most the highest occupied energy plus ``Delta``.

These tests are synthetic: they pin the policy arithmetic and the selection
wiring without needing SIESTA.  The two archived spectra are the tabulated
0.25/0.8 A butyronitrile candidates from
``analysis/butyronitrile-pbe-hf-fci/active-space-character`` (2.5 A, label-aware
AVAS).
"""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.projection_embedding_adapter import (
    ProjectionEmbeddingAdapter,
    _windowed_virtual_positions,
)


# Tabulated 2.5 A virtual candidates (F_emb diagonal / target-AO weight).
SIESTA_25_WEIGHTS = np.array([1.000000, 1.000000, 0.868533, 0.887436, 0.891946])
SIESTA_25_ENERGIES = np.array([0.582774, 0.584171, -0.156792, -0.145000, -0.144592])
SIESTA_25_HOMO = -0.182484

PYSCF_25_WEIGHTS = np.array([0.967420, 0.952159, 0.949684, 0.304169, 0.196701])
PYSCF_25_ENERGIES = np.array([-0.032740, -0.021246, -0.152683, 0.016453, 0.052561])
PYSCF_25_HOMO = -0.183106


# --------------------------------------------------------------------------- #
# the policy on the archived spectra
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta", [0.2, 0.3, 0.5])
def test_siesta_25_window_replaces_the_saturated_high_energy_pair(delta):
    # Positions 0,1 are the N-localised pi* pair at +0.58 Ha (weight 1.000); the
    # C-localised pi* pair is positions 3,4 (columns 10,11).  Every window tested
    # excludes 0,1 and keeps the C-localised pair.
    positions, shortfall = _windowed_virtual_positions(
        SIESTA_25_WEIGHTS,
        SIESTA_25_ENERGIES,
        n_keep=2,
        ceiling=SIESTA_25_HOMO + delta,
    )
    assert sorted(positions.tolist()) == [3, 4]
    assert shortfall == 0


@pytest.mark.parametrize("delta", [0.2, 0.3, 0.5])
def test_pyscf_25_window_leaves_the_protocol_selection_unchanged(delta):
    # PySCF's two highest-weight virtuals already lie near the HOMO, so restricting
    # the ranking to the window must still pick positions 0,1.
    positions, shortfall = _windowed_virtual_positions(
        PYSCF_25_WEIGHTS,
        PYSCF_25_ENERGIES,
        n_keep=2,
        ceiling=PYSCF_25_HOMO + delta,
    )
    assert sorted(positions.tolist()) == [0, 1]
    assert shortfall == 0


# --------------------------------------------------------------------------- #
# fallback behaviour
# --------------------------------------------------------------------------- #
def test_window_shorter_than_budget_fills_from_the_highest_weight_excluded():
    weights = np.array([0.90, 0.10, 0.05, 0.05])
    energies = np.array([-0.15, 0.60, 0.70, 0.80])
    positions, shortfall = _windowed_virtual_positions(weights, energies, n_keep=2, ceiling=-0.10)
    assert positions.tolist() == [0, 1]
    assert shortfall == 1


def test_empty_window_fills_the_budget_and_reports_the_full_shortfall():
    weights = np.array([0.10, 0.90, 0.20])
    energies = np.array([1.0, 2.0, 3.0])
    positions, shortfall = _windowed_virtual_positions(weights, energies, n_keep=2, ceiling=-0.10)
    assert positions.tolist() == [1, 2]
    assert shortfall == 2


def test_window_requires_matching_shapes():
    with pytest.raises(ValueError, match="same virtual candidates"):
        _windowed_virtual_positions(np.ones(2), np.ones(3), n_keep=1, ceiling=0.0)


# --------------------------------------------------------------------------- #
# _avas_select wiring
# --------------------------------------------------------------------------- #
def _synthetic_avas_setup():
    """Canonical orbitals with two low-energy and two high-energy virtuals.

    The target projector gives the high-energy virtuals the larger expectation
    (1.0 vs 0.9), so the standard AVAS eigen-ranking picks them while the window
    can reach the low-energy pair.
    """
    energies = np.array([-0.60, -0.20, -0.15, 0.00, 0.60, 2.00])
    coeff = np.eye(6)
    fock = np.diag(energies)
    projector = np.zeros((6, 6))
    projector[2, 2] = projector[3, 3] = 0.90
    projector[4, 4] = projector[5, 5] = 1.00
    return coeff, energies, fock, projector


def _select(coeff, energies, fock, projector, *, energy_window):
    return ProjectionEmbeddingAdapter._avas_select(
        coeff,
        energies,
        fock,
        projector,
        n_occ=2,
        n_docc=2,
        n_frozen_occ=0,
        choose=lambda _w_occ, _w_virt: (2, 2),
        energy_window=energy_window,
    )


def test_avas_select_default_is_unchanged_without_the_window():
    coeff, energies, fock, projector = _synthetic_avas_setup()
    info = _select(coeff, energies, fock, projector, energy_window=None)
    # The AVAS rotation reorders the virtual block in place, so the active column
    # indices are the first two virtual slots; their energies show they are the
    # high-weight, high-energy pair.
    assert info["active"].tolist() == [0, 1, 2, 3]
    assert info["energy"][2:4].tolist() == [0.60, 2.00]
    assert info["virt_selection_weights"].tolist() == [1.0, 1.0]
    assert "energy_window" not in info


def test_avas_select_window_keeps_the_low_energy_virtuals():
    coeff, energies, fock, projector = _synthetic_avas_setup()
    info = _select(coeff, energies, fock, projector, energy_window=0.2)
    assert info["active"].tolist() == [0, 1, 2, 3]
    # Same indices, different span: the window kept the canonical low-energy pair.
    assert info["energy"][2:4].tolist() == [-0.15, 0.00]
    assert info["virt_selection_weights"].tolist() == [0.9, 0.9]
    assert info["window_ceiling"] == pytest.approx(0.0)
    assert info["window_n_candidates"] == 2
    assert info["window_shortfall"] == 0
    # Candidate diagnostics are weight-sorted and align column with energy.
    assert info["virt_candidate_column"].tolist() == [4, 5, 2, 3]
    assert info["virt_candidate_energy"].tolist() == [0.60, 2.00, -0.15, 0.00]
    assert info["virt_selection_column"].tolist() == [2, 3]


def test_avas_select_rejects_a_non_positive_window():
    coeff, energies, fock, projector = _synthetic_avas_setup()
    with pytest.raises(ValueError, match="must be positive"):
        _select(coeff, energies, fock, projector, energy_window=0.0)
