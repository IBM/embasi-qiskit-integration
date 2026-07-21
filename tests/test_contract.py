# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Validation behaviour of the core data contracts."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult


def _symmetric_h2(norb: int, rng: np.random.Generator) -> np.ndarray:
    b = rng.standard_normal((norb, norb, norb, norb))
    b = b + b.transpose(1, 0, 2, 3)
    b = b + b.transpose(0, 1, 3, 2)
    b = b + b.transpose(2, 3, 0, 1)
    return b


def _hermitian_h1(norb: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((norb, norb))
    return a + a.T


def test_valid_hamiltonian_construction(rng):
    norb = 4
    h1 = _hermitian_h1(norb, rng)
    h2 = _symmetric_h2(norb, rng)
    ham = EmbeddedHamiltonian(h1=h1, h2=h2, e_core=1.5, nelec=(3, 2))
    assert ham.norb == norb
    assert ham.nelec == (3, 2)


def test_rejects_non_square_h1(rng):
    h1 = rng.standard_normal((3, 4))
    h2 = _symmetric_h2(3, rng)
    with pytest.raises(ValueError, match="square"):
        EmbeddedHamiltonian(h1=h1, h2=h2, e_core=0.0, nelec=(1, 1))


def test_rejects_mismatched_h2(rng):
    norb = 3
    h1 = _hermitian_h1(norb, rng)
    h2 = _symmetric_h2(norb + 1, rng)
    with pytest.raises(ValueError, match="h2 must have shape"):
        EmbeddedHamiltonian(h1=h1, h2=h2, e_core=0.0, nelec=(1, 1))


def test_rejects_non_hermitian_h1(rng):
    norb = 3
    h1 = rng.standard_normal((norb, norb))  # generically non-symmetric
    h2 = _symmetric_h2(norb, rng)
    with pytest.raises(ValueError, match="Hermitian"):
        EmbeddedHamiltonian(h1=h1, h2=h2, e_core=0.0, nelec=(1, 1))


def test_rejects_out_of_range_nelec(rng):
    norb = 3
    h1 = _hermitian_h1(norb, rng)
    h2 = _symmetric_h2(norb, rng)
    with pytest.raises(ValueError, match="out of range"):
        EmbeddedHamiltonian(h1=h1, h2=h2, e_core=0.0, nelec=(4, 1))


def test_warns_on_asymmetric_h2(rng):
    norb = 3
    h1 = _hermitian_h1(norb, rng)
    h2 = rng.standard_normal((norb, norb, norb, norb))  # no permutational symmetry
    with pytest.warns(UserWarning, match="8-fold"):
        EmbeddedHamiltonian(h1=h1, h2=h2, e_core=0.0, nelec=(1, 1))


def test_solver_result_coercion():
    res = SolverResult(energy=np.float64(-1.5), rdm1=[[1.0, 0.0], [0.0, 1.0]])
    assert isinstance(res.energy, float)
    assert isinstance(res.rdm1, np.ndarray)
    assert res.rdm2 is None


def test_no_spurious_symmetry_warning(rng):
    """A properly symmetric h2 must not trigger the warning."""
    norb = 4
    h1 = _hermitian_h1(norb, rng)
    h2 = _symmetric_h2(norb, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EmbeddedHamiltonian(h1=h1, h2=h2, e_core=0.0, nelec=(2, 2))
