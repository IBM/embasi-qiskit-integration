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
from embasi_qiskit_integration.backends.base import (
    CAP_ARBITRARY_DM_VEFF,
    CAP_ENERGY_ELEC,
    CAP_ERI,
    CAP_EXACT_EXCHANGE,
    CAP_HCORE,
)
from embasi_qiskit_integration.reference_protocol import (
    reference_wf_in_dft_hamiltonian,
)
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


class _ReferenceFakeIntegrals:
    """A 4-AO closed-shell stand-in for the reference-protocol wrapper.

    ``F = diag(-1, -0.5, 0.5, 1)`` in an orthonormal basis with a zero low-level
    potential, so the initial density is already the fixed point.  The active space
    is two occupied orbitals plus the fragment-spanned virtual AO 2.  A nonzero
    ``high_level_reference_energy`` stands in for SIESTA's pseudopotential /
    double-counting offset.
    """

    capabilities = frozenset(
        {CAP_HCORE, CAP_EXACT_EXCHANGE, CAP_ARBITRARY_DM_VEFF, CAP_ENERGY_ELEC, CAP_ERI}
    )

    def __init__(self, *, reference_energy: float) -> None:
        self._reference_energy = float(reference_energy)

    def overlap(self) -> np.ndarray:
        return np.eye(4)

    def hcore(self) -> np.ndarray:
        return np.diag([-1.0, -0.5, 0.5, 1.0])

    def energy_nuc(self) -> float:
        return 0.5

    def veff_ll(self, dm: np.ndarray) -> np.ndarray:
        return np.zeros((4, 4))

    def energy_elec_ll(self, dm: np.ndarray) -> float:
        return 0.0

    def veff_hf(self, dm: np.ndarray) -> np.ndarray:
        return np.zeros((4, 4))

    def eri_mo(self, coeff: np.ndarray) -> np.ndarray:
        n = coeff.shape[1]
        return np.zeros((n, n, n, n))

    def high_level_reference_energy(self, layer) -> float:
        return self._reference_energy


def _reference_fake_adapter(*, reference_energy: float):
    ints = _ReferenceFakeIntegrals(reference_energy=reference_energy)
    # One occupied B orbital under the level-shift projector, so the wrapper's
    # projector-leak guard sees a genuine B occupied (not a shifted virtual).
    dm_b = np.diag([0.0, 0.0, 0.0, 2.0])
    p_b = 1.0e6 * np.diag([0.0, 0.0, 0.0, 1.0])
    adapter = SimpleNamespace(
        ints=ints,
        _dm_a=np.diag([2.0, 2.0, 0.0, 0.0]),
        _dm_b=dm_b,
        p_b=p_b,
        p=SimpleNamespace(A_LL=object()),
    )
    return adapter


def test_reference_protocol_subtracts_the_backend_high_level_reference():
    """SIESTA's same-density offset must reach E[γ_A], not only E_high(A).

    The single-shot path subtracts ``high_level_reference_energy`` from
    ``E_high(A)``.  Without the same subtraction in the variational path the
    reference total sits tens of hartree away on a pseudopotential code (measured
    -61.6 vs -34.4 Ha at butyronitrile 0.8 A in SZ), while PySCF is unaffected.
    """
    adapter = _reference_fake_adapter(reference_energy=10.0)
    ham, info = reference_wf_in_dft_hamiltonian(
        adapter, np.array([0, 1, 2]), n_frozen_occ=0, n_virtual=1, n_shells=0
    )

    # Raw E[γ_A] = tr[h γ_A] + E_nuc = -3.0 + 0.5; the backend offset is 10 Ha.
    assert info.e_variational == pytest.approx(-12.5, abs=1e-12)
    assert info.scf_history[-1].energy == pytest.approx(-12.5, abs=1e-12)
    # The applied offset is reported, so a scan can recover the raw energy: the
    # same-density offset is geometry dependent on a pseudopotential code and must
    # not enter relative energies.
    assert info.energy_offset == pytest.approx(-10.0, abs=1e-12)
    assert info.e_variational - info.energy_offset == pytest.approx(-2.5, abs=1e-12)
    # Active-space HF reference: two occupied orbitals of diag(-1, -0.5, 0.5).
    assert info.e_reference == pytest.approx(-3.0, abs=1e-12)
    assert ham.e_core == pytest.approx(-9.5, abs=1e-12)
    assert ham.e_core + info.e_reference == pytest.approx(info.e_variational, abs=1e-12)

    # A zero reference (the PySCF/plain-callback path) leaves the raw energy alone.
    plain = _reference_fake_adapter(reference_energy=0.0)
    _ham, plain_info = reference_wf_in_dft_hamiltonian(
        plain, np.array([0, 1, 2]), n_frozen_occ=0, n_virtual=1, n_shells=0
    )
    assert plain_info.e_variational == pytest.approx(-2.5, abs=1e-12)
    assert plain_info.energy_offset == 0.0
    assert _ham.e_core == pytest.approx(0.5, abs=1e-12)


def test_reference_protocol_rejects_an_explicit_energy_offset():
    adapter = _reference_fake_adapter(reference_energy=10.0)
    with pytest.raises(ValueError, match="energy_offset"):
        reference_wf_in_dft_hamiltonian(
            adapter,
            np.array([0, 1, 2]),
            n_frozen_occ=0,
            n_virtual=1,
            scf_options={"energy_offset": 1.0},
        )
