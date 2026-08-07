# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Physics/algebra validation for :mod:`projection_embedding_adapter`.

These are *not* plumbing tests -- the CLI round-trip in ``test_cli_roundtrip``
already covers that the workflow wiring runs.  Here we assert the embedding
algebra: orbital orthonormality and span, the downfold round-trip, that ``h1``
is genuinely not the embedded Fock, and inactive-core rotation invariance.

Everything runs against a *real* ``embasi.embedding.ProjectionEmbedding`` (no
mock EmbASI), built once per session on the smallest sane system -- the
methanol monomer with the OH fragment active, sto-3g -- so the supersystem SCF
is paid only once.  The tests are marked ``embasi`` and ``slow`` and skipped
unless ``EMBASI_AVAILABLE=1``.

The low-level energies E_low(A) / E_low(AB) are read out of EmbASI's own
internals (``subsys_A_lowlvl_totalen`` / ``subsys_AB_lowlvl_scftotalen``, eV)
by :meth:`ProjectionEmbeddingAdapter._low_level_energies`, so
``projection_energy`` runs end-to-end with no monkeypatch -- verified by
``test_projection_energy_reads_embasi_low_level``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.embasi, pytest.mark.slow]


# --------------------------------------------------------------------------- #
# One real embedding, built once.
# --------------------------------------------------------------------------- #
def _build_adapter(xc_hl: str = "PBE", xc_ll: str = "PBE"):
    """A live ProjectionEmbeddingAdapter on the methanol/OH sto-3g system.

    Mirrors ``EmbeddingWorkflow._build_adapter`` but self-contained so the test
    does not depend on the CLI settings object.
    """
    import pyscf
    from ase.data.s22 import create_s22_system, s26
    from embasi.embedding import ProjectionEmbedding
    from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
        PySCFIntegrals,
    )

    atoms = create_s22_system(s26[22])[:6]  # methanol monomer
    active_atoms = [1, 5]  # O and hydroxyl H
    embed_mask = len(atoms) * [2]
    for i in active_atoms:
        embed_mask[i] = 1
    idx = np.argsort(embed_mask)
    embed_mask = np.sort(embed_mask)
    atoms = atoms[idx]

    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="sto-3g")
    mf_ll = mol.KS(xc=xc_ll)
    mf_hl = mol.KS(xc=xc_hl)
    projection = ProjectionEmbedding(
        atoms,
        embed_mask=embed_mask,
        calc_base_ll=PySCF(method=mf_ll),
        calc_base_hl=PySCF(method=mf_hl),
        projection="level-shift",
        parallel=False,
    )
    return ProjectionEmbeddingAdapter(projection, PySCFIntegrals(mf_hl), mu=1.0e6)


@pytest.fixture(scope="module")
def adapter():
    """Adapter with the low-level embedding already run (PBE-in-PBE)."""
    a = _build_adapter()
    a.run_low_level()
    return a


@pytest.fixture(scope="module")
def orbitals_full(adapter):
    """Full subsystem-A space: the WF-in-DFT problem, no active-space cuts."""
    return adapter.build_orbitals(n_frozen_occ=0, n_virtual=None)


# --------------------------------------------------------------------------- #
# a. Orbital sanity
# --------------------------------------------------------------------------- #
def test_orbitals_are_S_orthonormal(adapter, orbitals_full):
    """C^T S C = 1 on the surviving (level-shift-kept) orbital block.

    Tolerance 1e-8: these come straight from a generalized eigensolve
    ``sla.eigh(F, S)``, so orthonormality holds to solver precision, not just
    the 1e-6 the adapter's own guards use.
    """
    c, s = orbitals_full.coeff, adapter._s
    gram = c.T @ s @ c
    assert np.allclose(gram, np.eye(c.shape[1]), atol=1e-8)


def test_occupied_block_spans_localized_A(adapter, orbitals_full):
    """The occupied F_emb block spans the same space as mo_coeffs_A_LL.

    All singular values of ``C_A_LL^T S C_occ`` equal 1 (two orthonormal bases
    of the same subspace are related by a unitary).  Tolerance 1e-6 matches the
    adapter's own span guard; the localized orbitals carry SPADE round-off.
    """
    c_occ = orbitals_full.coeff[:, : orbitals_full.n_occ]
    overlap = adapter.mo_a_ll.T @ adapter._s @ c_occ
    sv = np.linalg.svd(overlap, compute_uv=False)
    assert np.allclose(sv, 1.0, atol=1e-6)


def test_subsystem_populations_are_integral(adapter):
    """tr(γ^A S) and tr(γ^B S) are integer electron counts.

    Tolerance 1e-6: SPADE localisation conserves the trace up to its own
    linear-algebra round-off, which the workflow output shows at ~1e-14 here.
    """
    n_a = np.einsum("ij,ji->", adapter._dm_a, adapter._s)
    n_b = np.einsum("ij,ji->", adapter._dm_b, adapter._s)
    assert n_a == pytest.approx(round(n_a), abs=1e-6)
    assert n_b == pytest.approx(round(n_b), abs=1e-6)
    assert round(n_a) > 0 and round(n_b) > 0


# --------------------------------------------------------------------------- #
# b. Downfold round-trip -- the important one.
# --------------------------------------------------------------------------- #
def _single_determinant_energy(ham, c_occ_active) -> float:
    """Restricted single-determinant energy of a downfolded Hamiltonian.

    Given the active-space integrals ``(h1, h2, e_core)`` and the *active*
    columns that are doubly occupied, form the closed-shell RHF density in the
    active MO basis and evaluate

        E = e_core + sum_pq D_pq h1_pq + 1/2 sum_pqrs D_pq D_rs [(pq|rs) - 1/2 (pr|qs)]

    with D the 2-occupancy density.  This is the reference-determinant energy of
    the downfolded Hamiltonian -- exactly the quantity a mean-field solve of
    ``ham`` would return, with no correlation.
    """
    _ = ham.norb
    d = 2.0 * (c_occ_active @ c_occ_active.T)  # (norb, norb), MO basis
    j = np.einsum("pqrs,rs->pq", ham.h2, d)
    k = np.einsum("prqs,rs->pq", ham.h2, d)
    e1 = np.einsum("pq,pq->", d, ham.h1)
    e2 = 0.5 * np.einsum("pq,pq->", d, j - 0.5 * k)
    return float(ham.e_core + e1 + e2)


def test_downfold_roundtrip_reproduces_embedded_mean_field(adapter, orbitals_full):
    """The downfold preserves the embedded mean-field energy exactly.

    What cancels, and why the tolerance is tight:

    The AO-basis embedded mean-field energy of subsystem A is
        E_AO = tr[γ^A h_emb] + 1/2 tr[γ^A G_HF[γ^A]]
    where ``h_emb`` already carries ``v_emb + P_B`` and ``G_HF`` is the bare
    (J - K/2) built from the *full* A density.  The CASCI downfold rewrites the
    identical operator over an active/inactive split: ``e_core`` absorbs the
    inactive-core mean field, ``h1`` folds the inactive Coulomb/exchange into
    the active block, and the active-active two-electron part stays in ``h2``.
    With the FULL A space active (n_virtual=None) the reference determinant of
    ``ham`` is the *same* Slater determinant as γ^A, just expressed in the
    diagonalized F_emb basis.  So the two energies are algebraically identical
    rearrangements of one another -- the difference is pure floating-point noise
    from two dense contraction paths.  Tolerance 1e-9 Ha (well below the
    ~1e-6 kJ/mol the paper reports for the physical PBE-in-PBE cancellation,
    which is a different, weaker statement about full SCF energies).
    """
    ham = adapter.embedded_hamiltonian(orbitals_full)

    # The A-occupied orbitals, expressed in the active MO basis.  With the full
    # space active and no frozen occupied, active columns 0..n_occ-1 ARE the
    # occupied set, so in the active basis they are the first identity columns.
    n_occ = orbitals_full.n_occ
    n_active = orbitals_full.n_active_orbitals
    occ_in_active = np.zeros((n_active, n_occ))
    occ_in_active[:n_occ, :n_occ] = np.eye(n_occ)
    e_downfold = _single_determinant_energy(ham, occ_in_active)

    # Same energy computed directly in the AO basis from the embedded operators.
    dm_a = 2.0 * (orbitals_full.coeff[:, :n_occ] @ orbitals_full.coeff[:, :n_occ].T)
    h_emb = adapter.h_emb
    veff_hf = adapter.ints.veff_hf(dm_a)
    e_ao = float(
        adapter.ints.energy_nuc()
        + np.einsum("ij,ji->", dm_a, h_emb)
        + 0.5 * np.einsum("ij,ji->", dm_a, veff_hf)
    )
    assert e_downfold == pytest.approx(e_ao, abs=1e-9)


# --------------------------------------------------------------------------- #
# c. h1 is not the Fock.
# --------------------------------------------------------------------------- #
def test_h1_differs_from_projected_fock_by_mean_field(adapter, orbitals_full):
    """``ham.h1`` is the embedded one-body op, NOT C^T F_emb C.

    Regression guard against reintroducing the double-counting bug: if h1 were
    just the projected Fock, the solver would rebuild J/K from the ERIs on top
    of a Fock that already contains them.  The gap between the two is precisely
    the high-level mean field G_HL[γ^A] (undone in ``h_emb``) plus the
    inactive-core HF potential (folded into ``h1``).  Assert the gap is real
    (not a rounding artefact): its norm must dwarf the 1e-6 tolerances used
    elsewhere.
    """
    c_act = orbitals_full.c_active
    fock_projected = c_act.T @ adapter._fock @ c_act
    ham = adapter.embedded_hamiltonian(orbitals_full)
    diff = np.abs(ham.h1 - fock_projected).max()
    assert diff > 1e-3, f"h1 suspiciously close to projected Fock (max |Δ| = {diff:.2e})"


# --------------------------------------------------------------------------- #
# d. e_core invariance under inactive-block rotation.
# --------------------------------------------------------------------------- #
def test_e_core_invariant_under_inactive_rotation(adapter, rng):
    """Rotating the inactive block by a random orthogonal U leaves e_core fixed.

    ``e_core`` depends on the inactive orbitals only through the density
    ``dm_in = 2 C_in C_in^T``, which is invariant under C_in -> C_in U for
    orthogonal U.  We freeze one occupied orbital so there IS an inactive block
    to rotate, mutate a copy of the orbital object, and re-downfold.  Tolerance
    1e-9 Ha: a similarity transform of a density is exact up to floating point.
    """
    from dataclasses import replace

    orbitals = adapter.build_orbitals(n_frozen_occ=1, n_virtual=0)
    assert orbitals.inactive.size >= 1, "need an inactive block to rotate"
    e_core_ref = adapter.embedded_hamiltonian(orbitals).e_core

    # Random orthogonal rotation within the inactive columns.
    n_in = orbitals.inactive.size
    a = rng.standard_normal((n_in, n_in))
    q, _ = np.linalg.qr(a)
    coeff = orbitals.coeff.copy()
    coeff[:, orbitals.inactive] = coeff[:, orbitals.inactive] @ q
    rotated = replace(orbitals, coeff=coeff)

    e_core_rot = adapter.embedded_hamiltonian(rotated).e_core
    assert e_core_rot == pytest.approx(e_core_ref, abs=1e-9)


# --------------------------------------------------------------------------- #
# End-to-end energy -- low-level energies read from EmbASI's own internals.
# --------------------------------------------------------------------------- #
def test_projection_energy_reads_embasi_low_level(adapter, orbitals_full):
    """``projection_energy`` runs with NO monkeypatch, reading real EmbASI data.

    ``construct_embedded_fock()`` already stored both low-level total energies on
    the projection object (in eV): ``subsys_AB_lowlvl_scftotalen`` = E_L[γ^A+γ^B]
    and ``subsys_A_lowlvl_totalen`` = E_L[γ^A].  The adapter's
    ``_low_level_energies()`` reads and converts them (eV -> Ha via EmbASI's own
    factor), so the whole Eq. 8 assembly runs on genuine EmbASI numbers.

    This test asserts (a) the reported ``e_low_total`` / ``e_low_A`` are exactly
    the eV -> Ha conversions of those attributes, and (b) the assembled total is
    finite with a numerically-zero projector leak.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import _EV2HA
    from embasi_qiskit_integration.solvers import FCISolver

    p = adapter.p
    # The real attributes must be present on the construct_embedded_fock() path.
    assert hasattr(p, "subsys_AB_lowlvl_scftotalen")
    assert hasattr(p, "subsys_A_lowlvl_totalen")
    expected_ab = float(np.real(p.subsys_AB_lowlvl_scftotalen)) * _EV2HA
    expected_a = float(np.real(p.subsys_A_lowlvl_totalen)) * _EV2HA

    ham = adapter.embedded_hamiltonian(orbitals_full)
    result = FCISolver().solve(ham)
    energy = adapter.projection_energy(result, orbitals_full)

    # The reported low-level energies are exactly the converted EmbASI internals.
    assert energy.e_low_total == pytest.approx(expected_ab, rel=0, abs=1e-12)
    assert energy.e_low_A == pytest.approx(expected_a, rel=0, abs=1e-12)

    # The assembly must be finite and the projector leak numerically zero.
    assert np.isfinite(energy.total)
    assert abs(energy.projector_leak) < 1e-6
    # e_high_A recovers a bound energy; total is a sum of finite Ha-scale terms.
    assert np.isfinite(energy.e_high_A)
    # Sanity: total is the documented combination.
    assert energy.total == pytest.approx(
        energy.e_low_total - energy.e_low_A + energy.e_high_A + energy.correction
    )


# --------------------------------------------------------------------------- #
# Density-fitted eri_mo agrees with the dense transform.
# --------------------------------------------------------------------------- #
def test_density_fit_eri_matches_dense(adapter, orbitals_full):
    """DF ``eri_mo`` reproduces the dense N^5 transform to DF tolerance.

    Density fitting is an *approximation* -- ``(pq|rs) ~= sum_P M^P_pq M^P_rs``
    with an auxiliary basis -- so the two transforms are NOT expected to be
    bitwise equal.  The comparison tolerance (1e-2 max abs on the two-electron
    integrals) is the density-fitting error itself, not a tuned knob: on this
    sto-3g system the DF/dense discrepancy sits at ~1e-3 Ha, well inside it.  A
    tighter tolerance would be asserting DF is exact, which it is not.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import PySCFIntegrals

    c_act = orbitals_full.c_active
    dense = adapter.ints.eri_mo(c_act)

    # Same mf_hl, but DF-backed integrals (PySCF default auxbasis).
    df_ints = PySCFIntegrals(adapter.ints.mf, density_fit=True)
    fitted = df_ints.eri_mo(c_act)

    assert fitted.shape == dense.shape
    max_err = float(np.abs(fitted - dense).max())
    assert max_err < 1e-2, f"DF-vs-dense ERI discrepancy {max_err:.2e} exceeds DF tolerance"


# --------------------------------------------------------------------------- #
# Restricted-span eigensolve reproduces the full-basis solve.
# --------------------------------------------------------------------------- #
def test_restricted_span_reproduces_full_basis_orbitals(adapter):
    """``restrict_to_a=True`` returns the same A-space physics as the full solve.

    The restricted path diagonalizes ``F_emb`` in span(A) (span(B) deflated),
    the full path solves the generalized problem over the whole AO basis and
    cuts everything with ``eps < floor``.  They are two routes to the *same*
    subspace, so they must agree.  We compare the recovered orbital energies
    (order-independent, so sorted) at solver precision.

    Tolerance 1e-6: both routes carry the same finite level-shift ``mu=1e6``,
    which perturbs the retained A eigenvalues at O(1/mu) ~ 1e-6; the two routes
    inherit that identically, so their *difference* is far smaller, but 1e-6 is
    the honest floor set by the shared shift rather than a value tuned to pass.
    """
    orb_restricted = adapter.build_orbitals(n_frozen_occ=0, n_virtual=None, restrict_to_a=True)
    orb_full = adapter.build_orbitals(n_frozen_occ=0, n_virtual=None, restrict_to_a=False)

    # Same number of A orbitals recovered.
    assert orb_restricted.coeff.shape[1] == orb_full.coeff.shape[1]

    e_r = np.sort(orb_restricted.energy)
    e_f = np.sort(orb_full.energy)
    assert np.allclose(e_r, e_f, atol=1e-6), (
        f"restricted vs full eigenvalues differ by {np.abs(e_r - e_f).max():.2e}"
    )

    # Both orbital sets are S-orthonormal (the restricted route by construction).
    for orb in (orb_restricted, orb_full):
        gram = orb.coeff.T @ adapter._s @ orb.coeff
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-8)
