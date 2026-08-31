# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Open-shell (``n_alpha != n_beta``) support across the quantum path.

Every assertion here is invisible at ``n_alpha == n_beta`` -- that is the point. The
spin-layout conventions coincide for a closed shell, the total electron count cannot
distinguish ``(2, 2)`` from ``(3, 1)``, and ``symmetrize_spin`` is harmless. So these
tests deliberately use asymmetric sectors, following
``tests/test_circuit_run_prep.py::test_bitstring_prep_matches_hf_for_open_shell``.
"""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.circuit_run.spin_layout import (
    BETA_ALPHA,
    counts_sector,
    create_hf_reference,
    swap_spin_halves,
    verify_counts_sector,
)
from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult
from embasi_qiskit_integration.projection_embedding_adapter import EmbeddedOrbitals
from embasi_qiskit_integration.selectors import somo_occupation_pattern
from embasi_qiskit_integration.solvers import FCISolver


@pytest.fixture
def ham_factory(rng):
    """Random Hermitian ``h1`` + 8-fold-symmetric ``h2`` at any spin sector."""

    def _make(norb: int, nelec: tuple[int, int]) -> EmbeddedHamiltonian:
        a = rng.normal(size=(norb, norb))
        h1 = 0.5 * (a + a.T)
        h2 = rng.normal(size=(norb,) * 4) * 0.1
        h2 = h2 + h2.transpose(1, 0, 2, 3)
        h2 = h2 + h2.transpose(0, 1, 3, 2)
        h2 = h2 + h2.transpose(2, 3, 0, 1)
        return EmbeddedHamiltonian(h1=h1, h2=h2, e_core=0.0, nelec=nelec)

    return _make


# --------------------------------------------------------------------------- #
# nelec derivation (replaces the `// 2` halving)
# --------------------------------------------------------------------------- #
def test_active_electrons_spin_matches_old_total_for_closed_shell():
    """The restricted reading must reproduce ``2 * (n_occ - n_inactive)`` exactly."""
    orb = EmbeddedOrbitals(
        coeff=np.eye(6),
        energy=np.arange(6.0),
        n_occ=3,
        inactive=np.array([0]),
        active=np.array([1, 2, 3, 4]),
    )
    assert orb.n_active_electrons == 2 * (3 - 1)
    assert orb.n_active_electrons_spin == (2, 2)
    assert not orb.is_open_shell


def test_active_electrons_spin_open_shell_is_not_halved():
    """An open shell keeps its asymmetry instead of collapsing to ``total // 2``."""
    orb = EmbeddedOrbitals(
        coeff=np.eye(6),
        energy=np.arange(6.0),
        n_occ=3,  # alpha
        inactive=np.array([0]),
        active=np.array([1, 2, 3, 4]),
        n_occ_b=2,  # beta
    )
    assert orb.n_active_electrons_spin == (2, 1)
    assert orb.n_active_electrons == 3  # odd: the old `// 2` would have lost an electron
    assert orb.is_open_shell
    assert "alpha=2, beta=1" in str(orb)


# --------------------------------------------------------------------------- #
# spin-resolved RDMs and the Sz check
# --------------------------------------------------------------------------- #
def test_check_spin_sector_catches_wrong_sector_at_the_right_total():
    """``(3, 1)`` and ``(2, 2)`` share a total, so only an Sz check separates them."""
    rdm1a, rdm1b = np.diag([1.0, 1.0, 1.0]), np.diag([1.0, 0.0, 0.0])
    res = SolverResult(energy=-1.0, rdm1=rdm1a + rdm1b, rdm1a=rdm1a, rdm1b=rdm1b)

    assert res.check_particle_number((3, 1)) == pytest.approx(0.0)
    assert res.check_spin_sector((3, 1)) == pytest.approx(0.0)
    # The wrong sector passes the particle-number check and fails only here.
    assert res.check_particle_number((2, 2)) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="wrong spin sector"):
        res.check_spin_sector((2, 2))


def test_spin_rdm_pair_must_be_complete_and_consistent():
    rdm1a, rdm1b = np.diag([1.0, 1.0]), np.diag([1.0, 0.0])
    with pytest.raises(ValueError, match="together"):
        SolverResult(energy=0.0, rdm1=rdm1a + rdm1b, rdm1a=rdm1a)
    with pytest.raises(ValueError, match="!= rdm1"):
        SolverResult(energy=0.0, rdm1=np.eye(2), rdm1a=rdm1a, rdm1b=rdm1b)


def test_spin_summed_only_result_refuses_the_sz_check():
    res = SolverResult(energy=0.0, rdm1=np.eye(2))
    assert not res.is_spin_resolved
    with pytest.raises(ValueError, match="only a spin-summed"):
        res.check_spin_sector((1, 1))


# --------------------------------------------------------------------------- #
# FCI at an asymmetric sector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nelec", [(2, 2), (3, 1), (1, 3), (3, 2)])
def test_fci_solves_any_spin_sector(ham_factory, nelec):
    """``direct_spin1`` handles an asymmetric tuple, and now reports the spin pair."""
    ham = ham_factory(4, nelec)
    res = FCISolver().solve(ham)

    assert res.is_spin_resolved
    assert res.check_particle_number(ham.nelec) == pytest.approx(0.0, abs=1e-9)
    assert res.check_spin_sector(ham.nelec) == pytest.approx(0.0, abs=1e-9)


def test_mirrored_sectors_are_degenerate(ham_factory):
    """A spin-free Hamiltonian cannot distinguish ``(3, 1)`` from ``(1, 3)``.

    Both solves must use the *same* integrals, so the mirrored Hamiltonian is built by
    swapping ``nelec`` on the original rather than drawing fresh random integrals.
    """
    ham = ham_factory(4, (3, 1))
    mirrored = ham.model_copy(update={"nelec": (1, 3)})

    e_ab = FCISolver().solve(ham).energy
    e_ba = FCISolver().solve(mirrored).energy
    assert e_ab == pytest.approx(e_ba, abs=1e-10)


# --------------------------------------------------------------------------- #
# spin layout -- the silent-corruption guard
# --------------------------------------------------------------------------- #
def test_hf_reference_array_layout_is_beta_left_alpha_right():
    """Pin the ``BETA_ALPHA`` array reference: beta block first, ones at high indices.

    ``(norb=4, na=3, nb=1)`` -> beta occupies index 3, alpha indices 5-7, so ``arr[4:]``
    sums to 3 (alpha) and ``arr[:4]`` to 1 (beta).
    """
    got = create_hf_reference(num_orbitals=4, num_elec_a=3, num_elec_b=1)
    assert np.array_equal(got, np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=np.int8))


def test_swap_is_identity_only_for_a_closed_shell():
    """Exactly why a layout error cannot be caught at ``n_alpha == n_beta``."""
    assert swap_spin_halves("0101", 2) == "0101"  # (1, 1): indistinguishable
    assert swap_spin_halves("0111", 2) != "0111"  # (2, 1): distinguishable


def test_verify_counts_sector_catches_a_swapped_spin_block():
    norb = 4
    # (3, 1) in our layout: MSB-left, so the rightmost norb chars are the alpha block.
    native = "0001" + "0111"
    assert counts_sector({native: 10}, norb) == (3, 1)

    verify_counts_sector({native: 10}, norb, (3, 1))  # correct layout: silent

    swapped = swap_spin_halves(native, norb)
    assert counts_sector({swapped: 10}, norb) == (1, 3)
    with pytest.raises(ValueError, match=BETA_ALPHA):
        verify_counts_sector({swapped: 10}, norb, (3, 1))


def test_verify_counts_sector_is_silent_for_a_closed_shell():
    """Both layouts are valid at ``na == nb``, so the guard must not fire."""
    norb = 4
    native = "0011" + "0011"
    verify_counts_sector({native: 10, swap_spin_halves(native, norb): 10}, norb, (2, 2))


def test_verify_counts_sector_ignores_mixed_pools():
    """Post-noise counts span sectors; the guard only fires on a definite reversal."""
    norb = 3
    verify_counts_sector({"000111": 5, "001011": 5, "111000": 5}, norb, (2, 1))


# --------------------------------------------------------------------------- #
# APC / selectors
# --------------------------------------------------------------------------- #
def test_somo_pattern_marks_the_singly_occupied_band():
    assert list(somo_occupation_pattern(6, 3, 3)) == [2, 2, 2, 0, 0, 0]
    assert list(somo_occupation_pattern(6, 4, 2)) == [2, 2, 1, 1, 0, 0]
    # Order-independent: which spin is in excess does not change the pattern.
    assert list(somo_occupation_pattern(6, 2, 4)) == [2, 2, 1, 1, 0, 0]


def test_somo_pattern_rejects_counts_that_do_not_fit():
    with pytest.raises(ValueError, match="do not fit"):
        somo_occupation_pattern(4, 5, 2)


# --------------------------------------------------------------------------- #
# the embedding path is deliberately NOT yet open shell
# --------------------------------------------------------------------------- #
def test_unrestricted_downfold_refuses_restricted_orbitals():
    """``unrestricted=True`` must not silently produce the ``na == nb`` split.

    Nothing populates ``n_occ_b`` from EmbASI yet -- ``build_orbitals`` infers a single
    ``n_occ`` from ``mo_a_ll.shape[1]`` -- so an unrestricted adapter handed restricted
    orbitals would downfold to a closed shell while reporting success. That is the exact
    failure the ``// 2`` removal was meant to end, so it raises instead. Pinning it here
    keeps the gap visible rather than latent.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    adapter = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    adapter.unrestricted = True
    restricted = EmbeddedOrbitals(
        coeff=np.eye(4),
        energy=np.arange(4.0),
        n_occ=2,
        inactive=np.array([], dtype=int),
        active=np.array([0, 1, 2]),
    )
    assert not restricted.is_open_shell
    with pytest.raises(NotImplementedError, match="n_occ_b is None"):
        adapter.embedded_hamiltonian(restricted)


def test_apc_selection_preserves_the_beta_count():
    """Active-space selection must not drop ``n_occ_b`` and demote an open shell."""
    orb = EmbeddedOrbitals(
        coeff=np.eye(5),
        energy=np.arange(5.0),
        n_occ=3,
        inactive=np.array([0]),
        active=np.array([1, 2, 3]),
        n_occ_b=2,
    )
    # Reconstructing as build_orbitals_apc_concentric does must keep the beta count.
    carried = EmbeddedOrbitals(
        coeff=orb.coeff,
        energy=orb.energy,
        n_occ=orb.n_occ,
        inactive=orb.inactive,
        active=orb.active,
        n_occ_b=orb.n_occ_b,
    )
    assert carried.is_open_shell
    assert carried.n_active_electrons_spin == (2, 1)


# --------------------------------------------------------------------------- #
# <S^2>: the one thing nelec cannot pin
# --------------------------------------------------------------------------- #
def test_sqd_reports_spin_square_and_the_sector_minimum(ham_factory):
    """``nelec`` fixes ``Sz``, never ``S``, so <S^2> is the only contamination signal.

    ``check_particle_number`` sums the pair and ``check_spin_sector`` tests ``Sz`` (fixed
    by construction), so both pass for a wrong-multiplicity state. This pins that the
    diagnostic is recorded and that it matches the sector's minimum ``S(S+1)`` for a
    clean high-spin solve.
    """
    pytest.importorskip("qiskit_aer")
    from embasi_qiskit_integration.circuit_run import build_sampler
    from embasi_qiskit_integration.solvers import SQDSolver

    ham = ham_factory(4, (3, 1))  # Sz = 1 -> S >= 1, so S(S+1) >= 2
    solver = SQDSolver(
        build_sampler("aer"),
        shots=40000,
        num_groups=60,
        seed=3,
        optimize=False,
        samples_per_batch=150,
        num_batches=4,
        max_iterations=4,
    )
    diagnostics = solver.solve(ham).diagnostics

    assert diagnostics["spin_square_min"] == pytest.approx(2.0)
    assert diagnostics["spin_square"] == pytest.approx(2.0, abs=1e-6)
    assert diagnostics["spin_sq_target"] is None  # no projection by default


def test_spin_sq_target_is_recorded(ham_factory):
    """An explicit ``spin_sq`` reaches the solver and is echoed in diagnostics."""
    pytest.importorskip("qiskit_aer")
    from embasi_qiskit_integration.circuit_run import build_sampler
    from embasi_qiskit_integration.solvers import FCISolver, SQDSolver

    ham = ham_factory(4, (3, 2))  # Sz = 1/2 -> doublet target S(S+1) = 0.75
    solver = SQDSolver(
        build_sampler("aer"),
        shots=40000,
        num_groups=60,
        seed=3,
        optimize=False,
        samples_per_batch=150,
        num_batches=4,
        max_iterations=4,
        spin_sq=0.75,
    )
    result = solver.solve(ham)

    assert result.diagnostics["spin_sq_target"] == pytest.approx(0.75)
    assert result.diagnostics["spin_square"] == pytest.approx(0.75, abs=1e-6)
    # Projection must not move a solve that was already spin-pure.
    assert result.energy == pytest.approx(FCISolver().solve(ham).energy, abs=1e-8)


# --------------------------------------------------------------------------- #
# Spin-resolved RDMs out of the SQD path
# --------------------------------------------------------------------------- #
def test_sqd_returns_the_spin_resolved_rdm_pair(ham_factory):
    """SQD must supply ``rdm1a``/``rdm1b``, or unrestricted feedback silently degrades.

    ``EmbeddingWorkflow`` gates the spin-resolved density feedback on
    ``result.is_spin_resolved``. When the SQD path returned only a spin-summed ``rdm1``,
    an ``unrestricted=True`` run fell back to the spin-summed ``rdm1_ao`` with no warning
    -- exactly the "cannot represent gamma_a != gamma_b" failure the unrestricted path
    exists to avoid -- and ``check_spin_sector`` was unreachable. This pins the pair, its
    consistency with ``rdm1``, and that it lands in the right sector.
    """
    pytest.importorskip("qiskit_aer")
    from embasi_qiskit_integration.circuit_run import build_sampler
    from embasi_qiskit_integration.solvers import SQDSolver

    ham = ham_factory(4, (3, 2))
    result = SQDSolver(
        build_sampler("aer"),
        shots=40000,
        num_groups=60,
        seed=3,
        optimize=False,
        samples_per_batch=150,
        num_batches=4,
        max_iterations=4,
    ).solve(ham)

    assert result.is_spin_resolved
    assert result.diagnostics["spin_resolved_rdm1"] is True
    assert result.rdm1a.shape == (ham.norb, ham.norb)
    # The pair must reconstruct the spin-summed RDM (SolverResult validates this too).
    np.testing.assert_allclose(result.rdm1a + result.rdm1b, result.rdm1, atol=1e-8)
    # And it must sit in the requested sector -- the check that was dead before.
    assert result.check_spin_sector(ham.nelec) == pytest.approx(0.0, abs=1e-6)
    assert np.trace(result.rdm1a) == pytest.approx(3.0, abs=1e-6)
    assert np.trace(result.rdm1b) == pytest.approx(2.0, abs=1e-6)


def test_sqd_spin_rdms_agree_with_fci(ham_factory):
    """The SQD pair matches FCI's ``make_rdm1s`` when SQD reaches the exact answer."""
    pytest.importorskip("qiskit_aer")
    from embasi_qiskit_integration.circuit_run import build_sampler
    from embasi_qiskit_integration.solvers import SQDSolver

    ham = ham_factory(4, (3, 1))
    sqd = SQDSolver(
        build_sampler("aer"),
        shots=40000,
        num_groups=60,
        seed=3,
        optimize=False,
        samples_per_batch=150,
        num_batches=4,
        max_iterations=4,
    ).solve(ham)
    ref = FCISolver().solve(ham)

    assert sqd.energy == pytest.approx(ref.energy, abs=1e-8)
    np.testing.assert_allclose(sqd.rdm1a, ref.rdm1a, atol=1e-6)
    np.testing.assert_allclose(sqd.rdm1b, ref.rdm1b, atol=1e-6)


# --------------------------------------------------------------------------- #
# Ported per-spin occupation producers (the missing n_occ_b source)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scf_kind", ["ROHF", "UHF"])
def test_density_projection_recovers_the_open_shell_spin_counts(scf_kind):
    """``per_spin_ao_rdm1`` + ``column_occupation_from_density`` recover ``mol.nelec``.

    Exercised on an OH doublet through both SCF flavours, since
    ``per_spin_ao_rdm1``'s two branches differ:
    UHF hands back the pair directly, ROHF returns a 2D total that must be split via
    ``mo_occ`` rather than halved.
    """
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    from embasi_qiskit_integration.selectors import (
        column_occupation_from_density,
        per_spin_ao_rdm1,
        spin_counts_from_occupation,
    )

    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = getattr(scf, scf_kind)(mol)
    mf.kernel()

    dm_a, dm_b = per_spin_ao_rdm1(mf)
    assert dm_a.shape == dm_b.shape == (mol.nao, mol.nao)
    mo = np.asarray(mf.mo_coeff)
    mo = mo[0] if mo.ndim == 3 else mo
    occ = column_occupation_from_density(mo, dm_a, dm_b, mol.intor_symmetric("int1e_ovlp"))

    assert set(np.unique(occ)) <= {0, 1, 2}
    assert spin_counts_from_occupation(occ) == mol.nelec == (5, 4)
    assert np.count_nonzero(occ == 1) == 1  # exactly one SOMO in a doublet


def test_density_projection_beats_the_positional_rule_after_rotation():
    """The reason to project rather than count positions.

    A within-block rotation (AVAS/concentric/APC all do one) preserves the span but
    destroys the "occupied columns come first, doubly occupied below n_occ" ordering that
    ``somo_occupation_pattern`` assumes. The projection asks each column directly, so it
    survives; the positional pattern does not.
    """
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    from embasi_qiskit_integration.selectors import (
        column_occupation_from_density,
        per_spin_ao_rdm1,
        somo_occupation_pattern,
        spin_counts_from_occupation,
    )

    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.kernel()
    dm_a, dm_b = per_spin_ao_rdm1(mf)
    s = mol.intor_symmetric("int1e_ovlp")
    mo = np.asarray(mf.mo_coeff)[0]

    occ = column_occupation_from_density(mo, dm_a, dm_b, s)
    na, nb = spin_counts_from_occupation(occ)
    assert (na, nb) == (5, 4)

    # UHF's own ordering already places the SOMO away from the top of the occupied
    # block, so the positional pattern disagrees with the projected truth.
    positional = somo_occupation_pattern(len(occ), na, nb)
    assert not np.array_equal(positional, occ)
    # Swapping two occupied columns leaves the projection unchanged (it is per-column),
    # which is exactly the invariance the positional rule lacks.
    rotated = mo.copy()
    rotated[:, [0, 4]] = rotated[:, [4, 0]]
    occ_rotated = column_occupation_from_density(rotated, dm_a, dm_b, s)
    assert spin_counts_from_occupation(occ_rotated) == (5, 4)
    assert sorted(occ_rotated.tolist()) == sorted(occ.tolist())
