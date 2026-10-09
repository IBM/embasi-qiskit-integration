# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pieces behind ``--wf_in_dft_energy reference`` and the SCF guard.

No EmbASI: the concentric-localization options run on a small PySCF molecule, and the
convergence guard on stand-in layers.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.projection_embedding_adapter import _require_scf_converged
from embasi_qiskit_integration.selectors import (
    concentric_localization_selector,
    fragment_ao_indices,
)


@pytest.fixture(scope="module")
def water():
    from pyscf import gto, scf

    mol = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf, mol.nelectron // 2


def _localise(water, **kw):
    mol, mf, n_occ = water
    s, f = mf.get_ovlp(), mf.get_fock()
    frag = fragment_ao_indices(mol, [0])  # the oxygen
    coeff, sigma2 = concentric_localization_selector(s, frag, f, **kw)(mf.mo_coeff, n_occ)
    return coeff, sigma2, s, f, n_occ


def test_canonical_orders_the_kept_virtuals_by_energy(water):
    coeff, sigma2, s, f, n_occ = _localise(water, canonical=True)
    k = int(np.sum(sigma2 > 0.5))
    kept = coeff[:, n_occ : n_occ + k]
    np.testing.assert_allclose(coeff.T @ s @ coeff, np.eye(coeff.shape[1]), atol=1e-8)
    f_kept = kept.T @ f @ kept
    np.testing.assert_allclose(f_kept, np.diag(np.diag(f_kept)), atol=1e-10)  # canonical
    assert np.all(np.diff(np.diag(f_kept)) >= -1e-12)  # ascending


def test_canonical_cap_keeps_the_lowest_energy_virtuals(water):
    full, sigma2_full, _s, f, n_occ = _localise(water, canonical=True)
    capped, sigma2_cap, _, _, _ = _localise(water, canonical=True, max_virtual=2)
    assert int(np.sum(sigma2_cap > 0.5)) == 2
    k = int(np.sum(sigma2_full > 0.5))
    lowest = np.sort(np.diag(full[:, n_occ : n_occ + k].T @ f @ full[:, n_occ : n_occ + k]))[:2]
    cap = capped[:, n_occ : n_occ + 2]
    np.testing.assert_allclose(np.diag(cap.T @ f @ cap), lowest, atol=1e-10)


def test_shell0_size_keeps_exactly_that_many_columns(water):
    _coeff, sigma2, *_ = _localise(water)
    rank = int(np.sum(sigma2 > 0.5))
    _coeff, sigma2_forced, s, _f, _ = _localise(water, shell0_size=rank + 1)
    assert int(np.sum(sigma2_forced > 0.5)) == rank + 1


def _layer(**method):
    return SimpleNamespace(
        atoms=SimpleNamespace(calc=SimpleNamespace(method=SimpleNamespace(**method)))
    )


def test_scf_guard_raises_on_a_failed_pyscf_scf():
    with pytest.raises(RuntimeError, match="did not converge in 50 cycles.*--scf_newton"):
        _require_scf_converged(_layer(converged=False, max_cycle=50, e_tot=-1.0), "test SCF")


def test_scf_guard_passes_converged_noscf_and_other_backends():
    _require_scf_converged(_layer(converged=True, max_cycle=50, e_tot=-1.0), "test SCF")
    _require_scf_converged(_layer(converged=False, max_cycle=0, e_tot=-1.0), "no-SCF evaluation")
    _require_scf_converged(SimpleNamespace(atoms=SimpleNamespace(calc=SimpleNamespace())), "aims")
