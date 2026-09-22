# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Live open-shell (UKS-in-UKS) embedding: the per-spin downfold, end to end.

These need real EmbASI and are marked ``embasi``/``slow``, so they are deselected by
default (``addopts = "-m 'not embasi'"``) and skipped unless ``EMBASI_AVAILABLE=1``.
Run them with BOTH::

    EMBASI_AVAILABLE=1 pytest -m embasi tests/test_open_shell_embedding_live.py

Everything else in the open-shell suite tests *plumbing* against stubs -- that the
energy split is additive, that the pair reaches ``direct_uhf``, that shapes are checked.
None of that says the open-shell **energy** is right. This file does: it pins the
numbers a real doublet produces, and cross-checks the downfold against an
independently rebuilt Hamiltonian so a bookkeeping error in the adapter cannot hide.

System: an OH radical (doublet, subsystem A) 4 A from a water molecule (closed-shell
environment), sto-3g, PBE low level, HF high level. Small enough to run in seconds,
and the same system the EmbASI open-shell example uses.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.embasi, pytest.mark.slow]

# Reference values, measured 2026-09-20 against EmbASI 2973169 / scalapack4py
# v1.0_release. They are regression anchors, not independently-derived physics: the
# *independent* check is test_downfold_matches_an_independently_rebuilt_hamiltonian.
REF_NELEC = (5, 4)
REF_NORB = 8
REF_E_CORE = 25.2020184581
# Re-measured 2026-09-22, after the `veff_ll` SPIN-PAIR INPUT fix.  `h_emb` (and every
# `h_emb_s`) subtracts the low-level mean field, and that mean field was being evaluated at
# a *depolarised* density: an unrestricted `get_veff` handed the spin-summed `_dm_a_arr`
# silently substitutes `d/2` for both channels, and at `xc_ll=PBE` the xc functional is
# nonlinear in the spin densities, so this is a real error (not the identity it is for an HF
# low level).  `E_solver` moves 0.0283 Ha; `e_core` does not move at all, because it is
# built from the same `h_emb_s` on both sides of the change at n_frozen_occ=0.
REF_E_SOLVER = -53.5507789338
# Re-measured 2026-09-21, after the PER-CHANNEL CONTRACTION fix, and this time checked
# against physics rather than only re-pinned.  Every previous `REF_TOTAL` in this file was
# wrong by ~39 Ha: the assembly contracted the spin-summed density against the spin-summed
# `v_emb`/`P_B`, which on a per-spin downfold picks up cross-channel terms (SPADE makes
# `span(A_alpha)` S-orthogonal to `span(B_alpha)`, NOT to `span(B_beta)`) and removes a
# `v_emb` term the solver never added.
#
# Why -149.61 is right and -110.51 was not.  This system is an OH radical 4 A from a
# water molecule, so at that separation the supersystem is very nearly non-interacting:
#
#     E(OH, UHF/sto-3g)        = -74.362669
#     E(H2O, RHF/sto-3g)       = -74.960573   ->  sum = -149.323242
#     E(supersystem, UHF)      = -149.322412
#     E(supersystem, UPBE)     = -149.793506
#
# The run is HF-in-PBE, so the total must land between the UHF and UPBE supersystem
# numbers, near PBE (only fragment A is HF).  NEW = -149.6094 sits 0.18 Ha from UPBE and
# 0.29 Ha from UHF -- in the bracket.  OLD = -110.5119 was 39.3 Ha outside it, i.e. not a
# physically possible total for this system at all.  The projector leak tells the same
# story: -3.6e-16 now, against +7.99e-02 before, for a quantity that is a numerical zero
# by construction.
#
# Superseded values, for the record (all wrong, each for a different reason):
#   -110.5897439424  before the per-channel AO lift fix
#   -110.5118889208  after it, still spin-summed in the energy assembly
#   -149.6093681127  after the per-channel contraction, still depolarising veff_ll's input
# `total` moves only 8.3e-05 Ha even though `E_solver` moves 0.0283 Ha: the assembly
# subtracts the same per-channel `v_emb_s` back off, so the shift very largely cancels and
# what remains is the genuine physical change.  Still inside the non-interacting bracket
# above (0.18 Ha from UPBE, 0.29 Ha from UHF), and the leak tightened from -3.6e-16 to
# -1.1e-15 -- both numerical zeros.
REF_TOTAL = -149.6092847931
REF_CORRECTION = -0.0020832881
# Each channel referenced to its own round-0 density AND contracted against its own
# `v_emb_spin`.  Superseded values: (0.6160922018, -0.5928344203) with a halved reference,
# then (0.0117289574, 0.0115288241) with the polarised reference but spin-summed operators.
REF_CORRECTION_SPIN = (0.0001734614, -0.0022567495)


def _build_open_shell_adapter():
    """A live unrestricted ProjectionEmbeddingAdapter on OH-radical-in-water."""
    import pyscf
    from ase import Atoms
    from embasi.embedding import ProjectionEmbedding
    from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
        PySCFIntegrals,
    )

    atoms = Atoms(
        "OHOHH",
        positions=[
            [0.0, 0.0, 0.0],  # O (OH radical)  -- region 1
            [0.0, 0.0, 0.97],  # H (OH radical)  -- region 1
            [4.0, 0.0, 0.0],  # O (water)
            [4.0, 0.0, 0.96],  # H (water)
            [4.9, 0.0, -0.3],  # H (water)
        ],
    )
    # spin=1: Nalpha - Nbeta for the whole system, the one unpaired electron on OH.
    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="sto-3g", spin=1)
    mf_ll = mol.UKS(xc="PBE")
    mf_hl = mol.UKS(xc="HF")
    projection = ProjectionEmbedding(
        atoms,
        embed_mask=[1, 1, 2, 2, 2],
        calc_base_ll=PySCF(method=mf_ll),
        calc_base_hl=PySCF(method=mf_hl),
        projection="level-shift",
        mu_val=1.0e6,
        parallel=False,
    )
    return ProjectionEmbeddingAdapter(
        projection, PySCFIntegrals(mf_hl, mf_ll), mu=1.0e6, unrestricted=True
    )


@pytest.fixture(scope="module")
def open_shell():
    """``(adapter, alpha, beta, ham, result)`` from one real open-shell embedding."""
    from embasi_qiskit_integration.solvers import FCISolver

    adapter = _build_open_shell_adapter()
    adapter.run_low_level()
    alpha, beta = adapter.build_orbitals_spin()
    ham = adapter.embedded_hamiltonian_spin((alpha, beta))
    result = FCISolver().solve(ham)
    return adapter, alpha, beta, ham, result


def test_embasi_reports_the_expected_spin_partition(open_shell):
    """``A_spin`` must put the whole supersystem spin on the radical, none on water.

    This is the assumption every downstream spin decision rests on -- if EmbASI ever
    partitions differently, the sector below is wrong for a reason that has nothing to
    do with our code.
    """
    adapter = open_shell[0]
    assert int(adapter.p.A_spin) == 1
    assert int(adapter.p.B_spin) == 0


def test_per_spin_orbital_counts_come_from_the_partition(open_shell):
    _, alpha, beta, _, _ = open_shell
    assert alpha.n_occ == 5  # n_alpha
    assert beta.n_occ == 4  # n_beta = n_alpha - A_spin
    # Each set describes ONE spin, so neither carries a beta count of its own.
    assert alpha.n_occ_b is None and beta.n_occ_b is None


def test_each_channel_is_annihilated_by_its_own_projector(open_shell):
    """The property that makes the per-spin downfold well-posed at all.

    A spin-summed ``P_B`` leaks ~4e-02 here (SPADE partitions the channels
    independently, so alpha-A is not orthogonal to beta-B). Per channel it is
    round-off, which is what lets ``embedded_hamiltonian_spin``'s leak check pass.
    """
    _, _, _, ham, _ = open_shell
    leaks = ham.meta["p_b_leak_per_spin"]
    assert max(leaks) < 1e-8, f"per-spin projector leak too large: {leaks}"


def test_downfold_is_spin_resolved_in_both_one_and_two_body(open_shell):
    _, _, _, ham, _ = open_shell
    assert ham.nelec == REF_NELEC
    assert ham.norb == REF_NORB
    assert ham.is_spin_dependent
    assert ham.has_spin_dependent_eri
    # The channels must genuinely differ -- equal blocks would mean the spin resolution
    # was lost somewhere and the "unrestricted" result is restricted in disguise.
    assert not np.allclose(ham.h1a, ham.h1b)
    h2_aa, _, h2_bb = ham.h2_spin
    assert not np.allclose(h2_aa, h2_bb)


def test_solver_takes_the_unrestricted_path(open_shell):
    _, _, _, _, result = open_shell
    assert result.diagnostics["solver"] == "pyscf-fci-uhf"
    assert result.is_spin_resolved
    assert result.check_spin_sector(REF_NELEC) == pytest.approx(0.0, abs=1e-9)


def test_downfold_matches_an_independently_rebuilt_hamiltonian(open_shell):
    """The real correctness check: rebuild the integrals outside the adapter.

    ``embedded_hamiltonian_spin`` does the per-channel ``h_emb`` strip, the frozen-core
    ``veff`` fold and three ERI transforms. Here the same active space is assembled from
    the adapter's *raw* per-spin Fock and integral backend, then solved with
    ``direct_uhf`` directly -- so a bookkeeping slip in the downfold shows up as a
    disagreement rather than as a plausible number.

    Tolerance 1e-9 Ha: both sides are the same contractions of the same matrices, so
    only floating-point associativity separates them.
    """
    from pyscf import fci

    adapter, alpha, beta, ham, result = open_shell
    c_a, c_b = alpha.c_active, beta.c_active
    # The PAIR, not the spin-summed `_dm_a_arr`: at `xc_ll=PBE` the xc functional is
    # nonlinear in the spin densities, so a summed input has PySCF substitute `d/2` for
    # both channels and silently depolarise the low-level mean field (worth ~0.065 Ha on
    # triplet CH2).  Passing `_dm_a_arr` here would restate the old bug and make this
    # rebuild agree with a downfold that was wrong.
    veff_ll = adapter.ints.veff_ll(adapter._dm_a_for_veff)

    # Frozen-core mean field, exactly as the downfold folds it in: the sum of the two
    # channels' own inactive densities (one electron per channel), shared by both
    # channels.  This fixture runs at n_frozen_occ=0, so both blocks are empty and
    # `veff_in` is zero -- spelled out in the correct form anyway, so that adding frozen
    # orbitals here tests the downfold instead of re-deriving its assumption.
    c_in_a, c_in_b = alpha.c_inactive, beta.c_inactive
    veff_in = adapter.ints.veff_hf(c_in_a @ c_in_a.T + c_in_b @ c_in_b.T)

    def _h1(c, fock):
        h = c.T @ (fock - veff_ll + veff_in) @ c
        return 0.5 * (h + h.T)

    h1a = _h1(c_a, adapter._fock_spin[0])
    h1b = _h1(c_b, adapter._fock_spin[1])
    eri = (
        adapter.ints.eri_mo(c_a),
        adapter.ints.eri_mo_mixed(c_a, c_b),
        adapter.ints.eri_mo(c_b),
    )
    e_elec, _ = fci.direct_uhf.kernel((h1a, h1b), eri, ham.norb, ham.nelec)

    assert float(e_elec) == pytest.approx(result.energy - ham.e_core, abs=1e-9)


def test_energy_split_is_an_exact_decomposition(open_shell):
    """The per-spin breakdown must sum to the totals it decomposes.

    Now exact *by construction*: the spin-summed ``correction`` / ``projector_leak`` are
    derived as the channel sums, rather than each being an independent contraction that
    happens to agree.  Earlier versions of this test bought the identity by contracting
    both channels against the spin-summed ``v_emb``/``P_B``, which is what hid a ~39 Ha
    error in ``total`` (see the note on ``REF_TOTAL``): the sum check is invariant under
    mis-attributing between the channels, so it cannot see that class of bug at all.
    """
    adapter, alpha, beta, _, result = open_shell
    energy = adapter.projection_energy(result, alpha, beta)

    assert energy.is_spin_resolved
    assert sum(energy.correction_spin) == pytest.approx(energy.correction, abs=1e-10)
    assert sum(energy.projector_leak_spin) == pytest.approx(energy.projector_leak, abs=1e-10)

    # Each channel against ITS OWN operators, and referenced to its OWN round-0 density.
    ref_a, ref_b = adapter._dm_a_spin_init
    dm_a_hl, dm_b_hl = adapter.rdm1_ao_spin(result.rdm1a, result.rdm1b, alpha, beta)
    v_a, v_b = adapter.v_emb_spin(0), adapter.v_emb_spin(1)
    p_a, p_b = adapter._p_b_spin
    assert energy.correction_spin[0] == pytest.approx(
        float(np.einsum("ij,ji->", dm_a_hl - ref_a, v_a)), abs=1e-10
    )
    assert energy.correction_spin[1] == pytest.approx(
        float(np.einsum("ij,ji->", dm_b_hl - ref_b, v_b)), abs=1e-10
    )
    assert energy.projector_leak_spin[0] == pytest.approx(
        float(np.einsum("ij,ji->", dm_a_hl, p_a)), abs=1e-12
    )
    assert energy.projector_leak_spin[1] == pytest.approx(
        float(np.einsum("ij,ji->", dm_b_hl, p_b)), abs=1e-12
    )

    # The projector leak is a numerical zero ONLY per channel.  The spin-summed
    # contraction is 0.08 Ha here -- small enough to read as noise on this system, which is
    # exactly why it went unnoticed, and +2.4e4 on a stretched C-N bond.  Assert both, so
    # the distinction is pinned rather than the magnitude of one system.
    assert abs(energy.projector_leak) < 1e-10
    leak_summed = float(np.einsum("ij,ji->", dm_a_hl + dm_b_hl, adapter.p_b))
    assert abs(leak_summed) > 1e-3
    assert abs(leak_summed - energy.projector_leak) > 1e-3

    # ...and the halved reference would still give a different split, so the polarised
    # reference is pinned too.
    half = 0.5 * adapter._dm_a_arr_init
    assert abs(float(np.einsum("ij,ji->", dm_a_hl - half, v_a)) - energy.correction_spin[0]) > 1e-3


def test_open_shell_energies_are_pinned(open_shell):
    """Regression anchors for the assembled open-shell energy.

    Deliberately tight (1e-6 Ha): these are deterministic closed-form contractions plus
    an exact FCI, with no sampling anywhere, so drift means something changed -- either
    here or in EmbASI. Widen only with a reason, and record it.
    """
    adapter, alpha, beta, ham, result = open_shell
    energy = adapter.projection_energy(result, alpha, beta)

    assert float(ham.e_core) == pytest.approx(REF_E_CORE, abs=1e-6)
    assert float(result.energy) == pytest.approx(REF_E_SOLVER, abs=1e-6)
    assert float(energy.total) == pytest.approx(REF_TOTAL, abs=1e-6)
    assert energy.correction == pytest.approx(REF_CORRECTION, abs=1e-6)
    assert energy.correction_spin[0] == pytest.approx(REF_CORRECTION_SPIN[0], abs=1e-6)
    assert energy.correction_spin[1] == pytest.approx(REF_CORRECTION_SPIN[1], abs=1e-6)


def test_spin_free_eri_is_a_real_approximation(open_shell):
    """Dropping ``h2_spin`` must change the answer, or the triple buys nothing.

    Quantifies what the spin-free two-body tensor costs: it was the state of this code
    before the ERI triple landed, so this is the size of that former error.
    """
    from embasi_qiskit_integration.contract import EmbeddedHamiltonian
    from embasi_qiskit_integration.solvers import FCISolver

    _, _, _, ham, result = open_shell
    spin_free = EmbeddedHamiltonian(
        h1=ham.h1,
        h2=ham.h2,
        e_core=ham.e_core,
        nelec=ham.nelec,
        h1a=ham.h1a,
        h1b=ham.h1b,
    )
    res_free = FCISolver().solve(spin_free)
    assert res_free.diagnostics["solver"] == "pyscf-fci-uhf-spinfree-eri"
    # ~2.96 Ha on this system -- far too large to leave implicit.
    assert abs(res_free.energy - result.energy) > 1.0


@pytest.fixture(scope="module")
def open_shell_frozen():
    """The same embedding, but with one frozen occupied per channel.

    Every other test here runs at ``n_frozen_occ=0``, where the whole frozen-core term
    vanishes identically -- so the core fold and its ``e_core`` contribution had no
    live numerical coverage at all. This fixture is that coverage.
    """
    adapter = _build_open_shell_adapter()
    adapter.run_low_level()
    alpha, beta = adapter.build_orbitals_spin(n_frozen_occ=1)
    ham = adapter.embedded_hamiltonian_spin((alpha, beta))
    return adapter, alpha, beta, ham


def test_frozen_core_e_core_charges_each_channel_to_its_own_fock(open_shell_frozen):
    """``e_core``'s one-body part is per channel, rebuilt independently here.

    The two channels see *different* embedded Focks, so beta's frozen core must be
    charged to ``h_emb_b``, not to alpha's. Contracting the summed core density against
    ``h_emb_a`` alone -- which this did -- overcharges by exactly
    ``tr[d_b (h_emb_a - h_emb_b)]``: **0.0162 Ha (10.2 kcal/mol)** here, with the
    electron count, the spin sector and the projector leak all still exact, which is why
    nothing else caught it.

    Rebuilt from the adapter's raw per-spin Fock and integral backend, so a bookkeeping
    slip shows up as a disagreement rather than as a plausible number.
    """
    adapter, alpha, beta, ham = open_shell_frozen

    c_in_a, c_in_b = alpha.c_inactive, beta.c_inactive
    # The frozen blocks are actually populated, or there is nothing under test.
    assert c_in_a.shape[1] == 1 and c_in_b.shape[1] == 1

    dm_core_a, dm_core_b = c_in_a @ c_in_a.T, c_in_b @ c_in_b.T
    dm_core = dm_core_a + dm_core_b
    # The PAIR, not the spin-summed `_dm_a_arr`: at `xc_ll=PBE` the xc functional is
    # nonlinear in the spin densities, so a summed input has PySCF substitute `d/2` for
    # both channels and silently depolarise the low-level mean field (worth ~0.065 Ha on
    # triplet CH2).  Passing `_dm_a_arr` here would restate the old bug and make this
    # rebuild agree with a downfold that was wrong.
    veff_ll = adapter.ints.veff_ll(adapter._dm_a_for_veff)
    h_emb_a = adapter._fock_spin[0] - veff_ll
    h_emb_b = adapter._fock_spin[1] - veff_ll

    # Two-body core term from the raw AO ERIs, independent of the adapter's own fold (see
    # the note in `test_frozen_core_downfold_matches_an_independent_rebuild`): the
    # unrestricted core energy is `0.5 tr[d J[d]] - 0.5 sum_sigma tr[d_sigma K[d_sigma]]`,
    # which is NOT `0.5 tr[d veff_hf(d)]`.
    from pyscf import ao2mo

    eri_ao = ao2mo.restore(1, adapter.ints.mol.intor("int2e"), adapter.ints.mol.nao)
    j_core = np.einsum("pqrs,rs->pq", eri_ao, dm_core)
    k_a = np.einsum("prqs,rs->pq", eri_ao, dm_core_a)
    k_b = np.einsum("prqs,rs->pq", eri_ao, dm_core_b)
    e_core_two_body = 0.5 * np.einsum("ij,ji->", dm_core, j_core) - 0.5 * (
        np.einsum("ij,ji->", dm_core_a, k_a) + np.einsum("ij,ji->", dm_core_b, k_b)
    )

    e_core_expected = (
        adapter.ints.energy_nuc()
        + np.einsum("ij,ji->", dm_core_a, h_emb_a)
        + np.einsum("ij,ji->", dm_core_b, h_emb_b)
        + e_core_two_body
    )
    assert ham.e_core == pytest.approx(float(e_core_expected), abs=1e-9)

    # The regression: both cores charged to alpha's operator. Assert the gap is real and
    # that it is precisely the cross term, so this fails loudly if the fix is reverted.
    e_core_alpha_only = (
        adapter.ints.energy_nuc() + np.einsum("ij,ji->", dm_core, h_emb_a) + e_core_two_body
    )
    cross = float(np.einsum("ij,ji->", dm_core_b, h_emb_a - h_emb_b))
    assert float(e_core_alpha_only) - float(e_core_expected) == pytest.approx(cross, abs=1e-9)
    # Physically significant on this system: ~10 kcal/mol, not round-off.
    assert abs(cross) > 1e-3


def test_frozen_core_downfold_matches_an_independent_rebuild(open_shell_frozen):
    """End-to-end at ``n_frozen_occ=1``: the full solve, rebuilt outside the adapter.

    The companion to :func:`test_downfold_matches_an_independently_rebuilt_hamiltonian`
    with the frozen-core path actually live, so an error in the core fold cannot hide in
    ``e_core`` while the electronic part stays right (or vice versa). Checks the *total*,
    which is what a caller reads.
    """
    from pyscf import ao2mo, fci

    from embasi_qiskit_integration.solvers import FCISolver

    adapter, alpha, beta, ham = open_shell_frozen
    result = FCISolver().solve(ham)

    c_a, c_b = alpha.c_active, beta.c_active
    c_in_a, c_in_b = alpha.c_inactive, beta.c_inactive
    dm_core_a, dm_core_b = c_in_a @ c_in_a.T, c_in_b @ c_in_b.T
    dm_core = dm_core_a + dm_core_b

    # Build the frozen-core mean field from the raw AO ERIs, NOT from
    # `adapter.ints.veff_uhf` (nor `veff_hf`).  Calling the adapter's own helper here
    # would restate the formula under test: an earlier version of this test used
    # `veff_hf`, and so could not see that the per-spin core was being folded with the
    # *restricted* `J - K/2` instead of the unrestricted `J[d] - K[d_sigma]`.  Only an
    # independent contraction distinguishes the two.
    eri_ao = ao2mo.restore(1, adapter.ints.mol.intor("int2e"), adapter.ints.mol.nao)

    def _j(d):
        return np.einsum("pqrs,rs->pq", eri_ao, d)

    def _k(d):
        return np.einsum("prqs,rs->pq", eri_ao, d)

    veff_in_a = _j(dm_core) - _k(dm_core_a)
    veff_in_b = _j(dm_core) - _k(dm_core_b)
    e_core_two_body = 0.5 * np.einsum("ij,ji->", dm_core, _j(dm_core)) - 0.5 * (
        np.einsum("ij,ji->", dm_core_a, _k(dm_core_a))
        + np.einsum("ij,ji->", dm_core_b, _k(dm_core_b))
    )
    # The PAIR, not the spin-summed `_dm_a_arr`: at `xc_ll=PBE` the xc functional is
    # nonlinear in the spin densities, so a summed input has PySCF substitute `d/2` for
    # both channels and silently depolarise the low-level mean field (worth ~0.065 Ha on
    # triplet CH2).  Passing `_dm_a_arr` here would restate the old bug and make this
    # rebuild agree with a downfold that was wrong.
    veff_ll = adapter.ints.veff_ll(adapter._dm_a_for_veff)
    h_emb_a = adapter._fock_spin[0] - veff_ll
    h_emb_b = adapter._fock_spin[1] - veff_ll

    def _h1(c, h_emb, veff_in):
        h = c.T @ (h_emb + veff_in) @ c
        return 0.5 * (h + h.T)

    eri = (
        adapter.ints.eri_mo(c_a),
        adapter.ints.eri_mo_mixed(c_a, c_b),
        adapter.ints.eri_mo(c_b),
    )
    e_elec, _ = fci.direct_uhf.kernel(
        (_h1(c_a, h_emb_a, veff_in_a), _h1(c_b, h_emb_b, veff_in_b)),
        eri,
        ham.norb,
        ham.nelec,
    )
    e_core_indep = (
        adapter.ints.energy_nuc()
        + np.einsum("ij,ji->", dm_core_a, h_emb_a)
        + np.einsum("ij,ji->", dm_core_b, h_emb_b)
        + e_core_two_body
    )
    assert float(e_elec + e_core_indep) == pytest.approx(result.energy, abs=1e-9)

    # The spin-averaged fold really is a different answer, so the agreement above is
    # evidence about the unrestricted core and not a tolerance that would absorb either.
    # On this doublet the frozen orbital is a deep 1s (the two channels' cores overlap to
    # ~1e-7), so the gap is small here -- it reaches ~1.9 kcal/mol at n_frozen_occ=2 and
    # ~0.8 Ha once a frozen orbital is genuinely valence-like.
    veff_avg = adapter.ints.veff_hf(dm_core)
    e_core_avg = (
        adapter.ints.energy_nuc()
        + np.einsum("ij,ji->", dm_core_a, h_emb_a)
        + np.einsum("ij,ji->", dm_core_b, h_emb_b)
        + 0.5 * np.einsum("ij,ji->", dm_core, veff_avg)
    )
    e_elec_avg, _ = fci.direct_uhf.kernel(
        (_h1(c_a, h_emb_a, veff_avg), _h1(c_b, h_emb_b, veff_avg)),
        eri,
        ham.norb,
        ham.nelec,
    )
    assert abs(float(e_elec_avg + e_core_avg) - result.energy) > 1e-9
    assert ham.meta["veff_core_spin_free"] is False

    # Freezing an occupied really did shrink the active space, so this is a different
    # downfold from the n_frozen_occ=0 fixture rather than an accidental repeat.
    assert ham.norb < REF_NORB
    assert ham.nelec == (REF_NELEC[0] - 1, REF_NELEC[1] - 1)


def test_workflow_drives_the_open_shell_path(tmp_path):
    """``spin_downfold=True`` must reach the per-spin path from a config alone.

    The adapter-level tests above bypass ``EmbeddingWorkflow``; this pins that the CLI
    surface wires it and lands on the *same number*.  The multi-cycle feedback is
    exercised separately by :func:`test_workflow_open_shell_survives_a_second_cycle`.
    """
    from ase import Atoms
    from ase.io import write

    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    atoms = Atoms(
        "OHOHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.97],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.96],
            [4.9, 0.0, -0.3],
        ],
    )
    xyz = tmp_path / "oh_water.xyz"
    write(str(xyz), atoms)

    workflow = EmbeddingWorkflow(
        _cli_parse_args=False,
        xyz=str(xyz),
        n_atoms=None,
        active_atoms=[0, 1],
        basis="sto-3g",
        xc_ll="PBE",
        xc_hl="HF",
        spin=1,
        unrestricted=True,
        spin_downfold=True,
        selector="none",
        solver="fci",
        max_cycles=1,
        diis=True,
    )
    energy = workflow.run(log=lambda *_a, **_k: None)
    assert energy.is_spin_resolved
    # At one cycle the workflow must reproduce the adapter-level pinned total exactly.
    # The previous loose bracket (-120 < total < -100) passed while the workflow lifted
    # BOTH spin channels through alpha's active space -- a 0.0779 Ha error that sat well
    # inside a 20 Ha window.  Pin it, so the CLI surface is held to the same number the
    # adapter tests are.
    assert float(energy.total) == pytest.approx(REF_TOTAL, abs=1e-6)
    assert energy.correction_spin[1] == pytest.approx(REF_CORRECTION_SPIN[1], abs=1e-6)


def test_workflow_open_shell_survives_a_second_cycle(tmp_path):
    """Cycle >= 2 goes through ``run_low_level_a_only``, which must rebuild the per-spin
    Fock rather than drop it -- a regression that made cycle 2 raise.

    Kept separate from the single-cycle pin above because the two assert different
    things: that one pins a number, this one pins that the loop *advances*.  The fed-back
    density is the per-channel pair, so a channel lifted through the wrong active space
    would move this total too.
    """
    from ase import Atoms
    from ase.io import write

    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    atoms = Atoms(
        "OHOHH",
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.97],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.96],
            [4.9, 0.0, -0.3],
        ],
    )
    xyz = tmp_path / "oh_water.xyz"
    write(str(xyz), atoms)

    workflow = EmbeddingWorkflow(
        _cli_parse_args=False,
        xyz=str(xyz),
        n_atoms=None,
        active_atoms=[0, 1],
        basis="sto-3g",
        xc_ll="PBE",
        xc_hl="HF",
        spin=1,
        unrestricted=True,
        spin_downfold=True,
        selector="none",
        solver="fci",
        max_cycles=2,
        diis=True,
    )
    energy = workflow.run(log=lambda *_a, **_k: None)
    assert energy.is_spin_resolved
    # It moved off the single-shot value (the feedback did something) but stayed close
    # (it is a correction, not a different calculation).
    assert float(energy.total) != pytest.approx(REF_TOTAL, abs=1e-9)
    assert float(energy.total) == pytest.approx(REF_TOTAL, abs=0.5)
    # The split stays exact through the loop.
    assert sum(energy.correction_spin) == pytest.approx(energy.correction, abs=1e-10)


def test_workflow_refuses_an_index_selector_on_the_spin_path(tmp_path):
    """Index selection and APC rank columns against ONE spin-summed Fock.

    A *localiser* (spade, concentric-cl) is fine -- it is applied per channel and the
    counts reconciled (see ``test_localiser_is_applied_per_channel_and_reconciled``).
    ``mulliken`` and ``apc-concentric`` have no per-channel form, so they would mix the
    channels the per-spin path exists to separate, and must refuse rather than degrade.
    """
    from ase import Atoms
    from ase.io import write

    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    atoms = Atoms(
        "OHOHH", positions=[[0, 0, 0], [0, 0, 0.97], [4, 0, 0], [4, 0, 0.96], [4.9, 0, -0.3]]
    )
    xyz = tmp_path / "g.xyz"
    write(str(xyz), atoms)

    workflow = EmbeddingWorkflow(
        _cli_parse_args=False,
        xyz=str(xyz),
        n_atoms=None,
        active_atoms=[0, 1],
        basis="sto-3g",
        xc_ll="PBE",
        xc_hl="HF",
        spin=1,
        unrestricted=True,
        spin_downfold=True,
        selector="mulliken",
        solver="fci",
        max_cycles=1,
    )
    with pytest.raises(ValueError, match="does not support selector"):
        workflow.run(log=lambda *_a, **_k: None)


def test_localiser_is_applied_per_channel_and_reconciled(open_shell):
    """A virtual localiser works on the per-spin path, at a common orbital count.

    Each channel's own σ² gap can fall at a different shell, so the two counts are
    reconciled to their ``min`` -- the downfold needs one ``norb``. Without that
    reconciliation ``embedded_hamiltonian_spin`` would get two different dimensions.
    """
    from embasi_qiskit_integration.selectors import (
        concentric_localization_selector,
        fragment_ao_indices,
    )

    adapter = open_shell[0]
    frag = fragment_ao_indices(adapter.ints.mol, [0, 1])
    loc = concentric_localization_selector(adapter._s_arr, frag, adapter._fock_spin[0], n_shells=1)

    # The localiser must actually RUN, once per channel, and actually change the result.
    # Every assertion below about a common `norb`, the sector and the leak holds whether or
    # not it ran -- the reconciliation equalises `norb` on its own and the sector comes from
    # the partition -- so without these two checks this test passed vacuously while
    # `build_orbitals_spin` accepted `virtual_localizer` and silently ignored it (the
    # parameter appeared only in the signature and the docstring).  That left
    # `--spin_downfold --selector spade/concentric-cl` producing an UNCUT active space: a
    # 6-qubit overrun on this fixture, with no warning.
    calls: list[int] = []

    def counting(c_virt, n_occ):
        calls.append(1)
        return loc(c_virt, n_occ)

    for attr in ("gap_tol", "max_virtual", "min_virtual"):
        if hasattr(loc, attr):
            setattr(counting, attr, getattr(loc, attr))

    baseline_alpha, _ = adapter.build_orbitals_spin()
    alpha, beta = adapter.build_orbitals_spin(virtual_localizer=counting)
    assert len(calls) == 2, f"localiser ran {len(calls)} times; expected once per channel"
    assert not np.array_equal(alpha.coeff, baseline_alpha.coeff), (
        "the virtual block must be rotated, not merely re-sliced"
    )

    # ...and a localiser with a real virtual budget must shrink `norb`.  This CL config
    # happens to keep all three virtuals, so the rotation above is what distinguishes
    # "ran" from "ignored" for it; `max_virtual` is what proves the CUT is honoured.
    capped = concentric_localization_selector(
        adapter._s_arr, frag, adapter._fock_spin[0], n_shells=1, max_virtual=1
    )
    cut_alpha, cut_beta = adapter.build_orbitals_spin(virtual_localizer=capped)
    assert cut_alpha.n_active_orbitals < baseline_alpha.n_active_orbitals, (
        "max_virtual must cut the active space, not leave it at the uncut size"
    )
    assert cut_alpha.n_active_orbitals == cut_beta.n_active_orbitals

    assert alpha.n_active_orbitals == beta.n_active_orbitals

    ham = adapter.embedded_hamiltonian_spin((alpha, beta))
    assert ham.nelec == REF_NELEC
    assert max(ham.meta["p_b_leak_per_spin"]) < 1e-8


def test_export_restore_round_trips_the_per_spin_state(open_shell):
    """The cross-process seam must carry the spin pair, or degrade loudly.

    Before this, ``export_state`` held only spin-summed arrays: an out-of-process
    open-shell loop would restore, assemble a plausible *restricted* Fock and report a
    wrong energy with nothing to flag it.
    """
    adapter = open_shell[0]
    state = adapter.export_state()
    for key in ("p_b_spin", "v_emb_spin", "dm_a_spin", "dm_b_spin", "fock_spin"):
        assert key in state, f"{key} missing from the snapshot"
        assert np.asarray(state[key]).shape[0] == 2

    fresh = _build_open_shell_adapter()
    fresh.run_low_level()
    fresh.restore_state(state)
    assert fresh._fock_spin is not None
    np.testing.assert_allclose(fresh._fock_spin[0], adapter._fock_spin[0], atol=1e-10)
    np.testing.assert_allclose(fresh._fock_spin[1], adapter._fock_spin[1], atol=1e-10)


def test_relax_hf_targets_the_right_spin_sector():
    """``relax_active_hf`` must relax against the per-spin density, not a summed one.

    It used to wrap with the single-channel ``_as_spin_kpoint_array``, feeding a
    spin-summed density into an unrestricted embedded SCF. It now hands EmbASI the pair
    and sets ``input_fragment_spin``, and produces a relaxed per-spin Fock that
    ``build_orbitals_spin(use_relaxed=True)`` can diagonalize.
    """
    from embasi_qiskit_integration.solvers import FCISolver

    adapter = _build_open_shell_adapter()
    adapter.run_low_level()
    adapter.relax_active_hf()

    assert adapter._fock_relaxed_spin is not None
    alpha, beta = adapter.build_orbitals_spin(use_relaxed=True)
    ham = adapter.embedded_hamiltonian_spin((alpha, beta))
    assert ham.nelec == REF_NELEC
    result = FCISolver().solve(ham)
    assert result.diagnostics["solver"] == "pyscf-fci-uhf"
    assert result.check_spin_sector(REF_NELEC) == pytest.approx(0.0, abs=1e-9)


def test_build_orbitals_spin_refuses_relaxed_without_relaxation(open_shell):
    adapter = open_shell[0]
    with pytest.raises(ValueError, match="needs a relaxed per-spin Fock"):
        adapter.build_orbitals_spin(use_relaxed=True)


def test_dft_in_dft_works_open_shell():
    """The Eq. 2 reference path is already spin-correct; pin that it stays so.

    It re-slices EmbASI's own scalar energies rather than contracting matrices here, so
    an unrestricted ``run()`` yields an unrestricted answer with no per-spin work needed
    on this side. ``.total`` must equal ``DFT_AinB_total_energy`` exactly.
    """
    import pyscf
    from ase import Atoms
    from embasi.embedding import ProjectionEmbedding
    from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

    from embasi_qiskit_integration.projection_embedding_adapter import (
        _EV2HA,
        ProjectionEmbeddingAdapter,
        PySCFIntegrals,
    )

    atoms = Atoms(
        "OHOHH",
        positions=[[0, 0, 0], [0, 0, 0.97], [4, 0, 0], [4, 0, 0.96], [4.9, 0, -0.3]],
    )
    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="sto-3g", spin=1)
    mf_ll, mf_hl = mol.UKS(xc="PBE"), mol.UKS(xc="PBE0")
    projection = ProjectionEmbedding(
        atoms,
        embed_mask=[1, 1, 2, 2, 2],
        calc_base_ll=PySCF(method=mf_ll),
        calc_base_hl=PySCF(method=mf_hl),
        projection="level-shift",
        mu_val=1.0e6,
        parallel=False,
    )
    adapter = ProjectionEmbeddingAdapter(
        projection, PySCFIntegrals(mf_hl, mf_ll), mu=1.0e6, unrestricted=True
    )
    energy = adapter.dft_in_dft_energy()

    assert int(projection.A_spin) == 1  # the run really was open shell
    native = float(np.real(projection.DFT_AinB_total_energy)) * _EV2HA
    assert energy.total == pytest.approx(native, abs=1e-9)
    # UKS-in-UKS reference for this system, -4076.697213 eV.
    assert energy.total == pytest.approx(-149.8158689, abs=1e-6)


def test_apc_selection_forwards_the_beta_count_for_real(open_shell):
    """Behavioural counterpart to ``test_apc_selection_forwards_the_beta_count``.

    That test parses ``build_orbitals_apc_concentric``'s *source* for an ``n_occ_b=``
    keyword, because without live EmbASI the APC path cannot be reached (it needs real
    integrals and a concentric-localization stage). Source assertions are brittle to
    neighbouring edits -- one briefly broke when a helper was inserted above the method
    -- so now that EmbASI runs, pin the actual behaviour too.

    Uses the restricted-orbital APC path (`build_orbitals`, which carries ``n_occ_b``
    from ``A_spin``), not ``build_orbitals_spin``: APC is unsupported on the per-spin
    path, and this is exactly where a dropped ``n_occ_b`` would silently demote an
    open-shell partition to the restricted reading.
    """
    from embasi_qiskit_integration.selectors import fragment_ao_indices

    adapter = open_shell[0]
    frag = fragment_ao_indices(adapter.ints.mol, [0, 1])
    orbitals = adapter.build_orbitals_apc_concentric(
        fragment_ao=frag, n_shells=1, max_size=(4, 4), restrict_to_a=False
    )
    # The beta count must survive the APC re-selection.
    assert orbitals.n_occ_b is not None, "APC dropped n_occ_b; the partition is demoted"
    assert orbitals.is_open_shell
    n_alpha, n_beta = orbitals.n_active_electrons_spin
    assert n_alpha > n_beta, f"expected a doublet-like split, got {(n_alpha, n_beta)}"


# --------------------------------------------------------------------------- #
# A geometry where the two spin channels' spans genuinely differ.
#
# Everything above runs on the OH-radical fixture, whose channels overlap enough that a
# cross-channel error reads as noise: the spin-summed projector leak there is 0.08 Ha
# against a per-channel ~1e-16, which is why contracting across channels went unnoticed
# through several rounds of review.  This block uses the stretched end of the C-N
# dissociation scan (2.20 A, the `data/22.inp` geometry), where the same error is +2.4e4
# and a wrong assembly cannot hide.
# --------------------------------------------------------------------------- #

# Stretched butyronitrile, C-N at 2.20 A: `data/22.inp`, inlined so the test does not
# depend on an untracked data directory.
_STRETCHED_CN = [
    ("N", (3.20473, -0.47458, -0.00000)),
    ("C", (1.14199, 0.29032, -0.00000)),
    ("C", (-0.28626, 0.80954, -0.00000)),
    ("H", (-0.43038, 1.45243, -0.90096)),
    ("H", (-0.43037, 1.45244, 0.90095)),
    ("C", (-1.33813, -0.36286, 0.00001)),
    ("H", (-1.17391, -1.00037, -0.89892)),
    ("H", (-1.17391, -1.00036, 0.89895)),
    ("C", (-2.80434, 0.18582, 0.00001)),
    ("H", (-3.53027, -0.65561, 0.00001)),
    ("H", (-2.99050, 0.80923, 0.90131)),
    ("H", (-2.99050, 0.80921, -0.90131)),
]


def _build_stretched_adapter(*, spin: int, unrestricted: bool):
    """A live adapter on the stretched C-N geometry, HF-in-HF, active atoms [0, 1]."""
    import pyscf
    from ase import Atoms
    from embasi.embedding import ProjectionEmbedding
    from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
        PySCFIntegrals,
    )

    atoms = Atoms("".join(s for s, _ in _STRETCHED_CN), positions=[p for _, p in _STRETCHED_CN])
    kwargs = {"atom": ase_atoms_to_pyscf(atoms), "basis": "sto-3g"}
    if spin:
        kwargs["spin"] = spin
    mol = pyscf.M(**kwargs)
    make = mol.UKS if unrestricted else mol.RKS
    mf_ll, mf_hl = make(xc="HF"), make(xc="HF")
    projection = ProjectionEmbedding(
        atoms,
        embed_mask=[1, 1] + [2] * (len(_STRETCHED_CN) - 2),
        calc_base_ll=PySCF(method=mf_ll),
        calc_base_hl=PySCF(method=mf_hl),
        projection="level-shift",
        mu_val=1.0e6,
        parallel=False,
    )
    adapter = ProjectionEmbeddingAdapter(
        projection, PySCFIntegrals(mf_hl, mf_ll), mu=1.0e6, unrestricted=unrestricted
    )
    adapter.run_low_level()
    return adapter


def test_stretched_spans_are_not_cross_orthogonal():
    """The premise of the per-channel contraction, measured where it is unmissable.

    Each channel's density is annihilated by *its own* projector but not by the other's,
    because SPADE partitions the spins separately. On the OH fixture the cross terms are
    0.08 / 9e-04; here they are ~2.4e4 / ~5e2. Pinning both magnitudes is the point: a
    tolerance that passes on OH says nothing about this regime.
    """
    adapter = _build_stretched_adapter(spin=2, unrestricted=True)
    d_a, d_b = adapter._dm_a_spin
    p_a, p_b = adapter._p_b_spin

    def tr(x, y):
        return float(np.einsum("ij,ji->", x, y))

    assert abs(tr(d_a, p_a)) < 1e-8
    assert abs(tr(d_b, p_b)) < 1e-8
    # ...and the cross terms are enormous, which is what makes a spin-summed contraction
    # of these operators wrong rather than merely imprecise.
    assert abs(tr(d_a, p_b)) > 1e3
    assert tr(d_a + d_b, adapter._p_b) == pytest.approx(
        tr(d_a, p_a) + tr(d_b, p_b) + tr(d_a, p_b) + tr(d_b, p_a), abs=1e-6
    )


def test_stretched_open_shell_total_matches_a_closed_shell_control():
    """The assembled per-spin total must be physical, cross-checked two ways.

    This is the test the OH fixture cannot provide. It pins the fix that matters most:
    contracting the spin-summed density against the spin-summed operators gave
    ``total = -24288 Ha`` on this geometry -- for a system whose low-level supersystem
    energy is ``-206.99 Ha`` -- with ``projector_leak = +2.41e4`` for a field that is a
    numerical zero by construction.

    Two independent references, because a single one could be wrong the same way:

    1. **A closed-shell control on the identical geometry**, through the restricted path
       (untouched by this change). The triplet must land near it, not 24 kHa away.
    2. **The projector leak**, which must be a numerical zero per channel.
    """
    from embasi_qiskit_integration.solvers import FCISolver

    adapter = _build_stretched_adapter(spin=2, unrestricted=True)
    alpha, beta = adapter.build_orbitals_spin(n_frozen_occ=4)
    ham = adapter.embedded_hamiltonian_spin((alpha, beta))
    result = FCISolver().solve(ham)
    energy = adapter.projection_energy(result, alpha, orbitals_b=beta)

    # The leak is a numerical zero -- the regression this guards was +2.41e+04.
    assert abs(energy.projector_leak) < 1e-8, energy.projector_leak
    assert energy.is_spin_resolved
    assert sum(energy.projector_leak_spin) == pytest.approx(energy.projector_leak, abs=1e-12)
    assert sum(energy.correction_spin) == pytest.approx(energy.correction, abs=1e-10)

    # Reference 1: the restricted path on the same nuclei.
    control = _build_stretched_adapter(spin=0, unrestricted=False)
    orbitals = control.build_orbitals(n_frozen_occ=4)
    control_result = FCISolver().solve(control.embedded_hamiltonian(orbitals))
    control_energy = control.projection_energy(control_result, orbitals)

    # `e_high_A` is the term the bug corrupted (-24090 against -9.11), and it is the one
    # that should agree closely: both are HF on the same fragment, differing only in spin
    # state.  Measured 0.022 Ha apart; 0.5 Ha is loose enough not to be brittle and tight
    # enough that the 24 kHa regression cannot pass.
    assert energy.e_high_A == pytest.approx(control_energy.e_high_A, abs=0.5)
    assert float(energy.total) == pytest.approx(float(control_energy.total), abs=1.0)
    # An absolute sanity bracket, in case BOTH paths ever regress together.
    assert -215.0 < float(energy.total) < -200.0


def test_stretched_state_roundtrip_reproduces_the_per_spin_energy():
    """``projection_energy_from_state`` must reproduce the live per-spin assembly exactly.

    The snapshot path is a second implementation of the same algebra, so it is where a
    per-channel fix silently fails to propagate. It must also *refuse* an open-shell
    snapshot handed only a spin-summed density, rather than assembling the wrong number.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import (
        projection_energy_from_state,
    )
    from embasi_qiskit_integration.solvers import FCISolver

    adapter = _build_stretched_adapter(spin=2, unrestricted=True)
    alpha, beta = adapter.build_orbitals_spin(n_frozen_occ=4)
    ham = adapter.embedded_hamiltonian_spin((alpha, beta))
    result = FCISolver().solve(ham)
    live = adapter.projection_energy(result, alpha, orbitals_b=beta)

    dm_a, dm_b = adapter.rdm1_ao_spin(result.rdm1a, result.rdm1b, alpha, beta)
    state = {k: v for k, v in adapter.export_state().items() if k != "fingerprint"}
    from_state = projection_energy_from_state(
        state,
        solver_energy=result.energy,
        rdm1_ao=dm_a + dm_b,
        rdm1_ao_spin=(dm_a, dm_b),
    )
    for term in ("e_high_A", "correction", "projector_leak", "footing_shift", "total"):
        assert getattr(from_state, term) == pytest.approx(getattr(live, term), abs=1e-10), term
    assert from_state.correction_spin == pytest.approx(live.correction_spin, abs=1e-10)

    # Withholding the pair must raise, not silently assemble the spin-summed (wrong) form.
    with pytest.raises(ValueError, match="open-shell"):
        projection_energy_from_state(state, solver_energy=result.energy, rdm1_ao=dm_a + dm_b)


def test_apc_ranks_against_subsystem_a_real_electron_count(open_shell):
    """The APC exchange ranking must use ``n_alpha + n_beta``, not ``2 * n_alpha``.

    ``build_orbitals_apc_concentric`` built its ranking density as the restricted
    ``2 * c_occ c_occ^T``, which on an open shell is ``2 * n_alpha`` and so invents
    ``n_alpha - n_beta`` extra electrons.  ``k_diag_virt`` feeds ``apc_pair_coefficients``
    directly, so the entropies inherit the error: measured on this doublet it fed 10
    electrons where EmbASI reports ``A_pop = 9.0``, and ``k_diag`` came out up to 0.88 Ha
    high, most of it on the SOMO.

    The ranking *order* survives it on a system this small, so the selected space is
    unchanged -- which is why only a check on the density itself can see it, and why it
    would bite where two candidates are near-tied.
    """
    from embasi_qiskit_integration.selectors import fragment_ao_indices

    adapter = open_shell[0]
    s = np.asarray(adapter._s_arr)
    seen: list[float] = []
    original = adapter.ints.get_k

    def spy(dm):
        seen.append(float(np.einsum("ij,ji->", np.asarray(dm), s)))
        return original(dm)

    adapter.ints.get_k = spy
    try:
        frag = fragment_ao_indices(adapter.ints.mol, [0, 1])
        adapter.build_orbitals_apc_concentric(
            fragment_ao=frag, n_shells=1, max_size=(4, 4), fixed=True
        )
    finally:
        adapter.ints.get_k = original

    assert seen, "the APC ranking never called get_k"
    a_pop = float(adapter.p.A_pop)
    assert a_pop == pytest.approx(9.0, abs=1e-6)  # the doublet: 5 alpha + 4 beta
    for n in seen:
        assert n == pytest.approx(a_pop, abs=1e-6), (
            f"ranking density carries {n} electrons; subsystem A has {a_pop}"
        )
        # And specifically not the restricted 2 * n_alpha reading.
        assert n != pytest.approx(10.0, abs=1e-6)
