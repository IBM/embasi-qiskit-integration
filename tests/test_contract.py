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


# ----- particle-number check on a solver result ------------------------------- #


def _diagonal_result(occupations: list[float]) -> SolverResult:
    """A ``SolverResult`` whose rdm1 has the given orbital occupations."""
    return SolverResult(energy=-1.0, rdm1=np.diag(occupations))


def test_check_particle_number_accepts_a_consistent_rdm():
    """A correct RDM passes and reports its (tiny) deviation."""
    result = _diagonal_result([2.0, 2.0, 1.0])
    assert result.check_particle_number((3, 2)) == pytest.approx(0.0)
    # The total may be given directly instead of as a spin pair.
    assert result.check_particle_number(5) == pytest.approx(0.0)


def test_check_particle_number_rejects_the_wrong_sector():
    """A whole-electron discrepancy must raise, not pass quietly.

    The trace of a spin-summed 1-RDM *is* the particle number, so this catches a
    solver that returned an RDM for a different sector than the Hamiltonian
    describes. It matters because the RDM is fed back into the next embedding
    cycle's density, so an unchecked error propagates while the printed energies
    still look plausible.
    """
    result = _diagonal_result([2.0, 2.0, 1.0])  # 5 electrons
    with pytest.raises(ValueError, match="wrong particle-number sector"):
        result.check_particle_number((3, 3))  # expected 6

    # The message quantifies the miss, so a log is enough to diagnose it.
    with pytest.raises(ValueError, match=r"off by \+?-1"):
        result.check_particle_number(6)


def test_check_particle_number_tolerance_is_configurable():
    """``atol`` admits noise but not a whole electron at the default."""
    result = _diagonal_result([2.0, 2.0, 0.999_999_9])
    # Within the default tolerance.
    assert abs(result.check_particle_number(5)) < 1e-6
    # A deliberately loose tolerance accepts a real discrepancy...
    off = _diagonal_result([2.0, 2.0, 0.5])
    assert off.check_particle_number(5, atol=1.0) == pytest.approx(-0.5)
    # ...but the default does not.
    with pytest.raises(ValueError):
        off.check_particle_number(5)


def test_check_particle_number_rejects_a_non_square_rdm():
    result = SolverResult(energy=-1.0, rdm1=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="square"):
        result.check_particle_number(2)
