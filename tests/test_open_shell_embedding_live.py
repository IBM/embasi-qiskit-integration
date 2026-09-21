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
REF_E_SOLVER = -53.5224926352
REF_TOTAL = -110.5897439424
REF_CORRECTION = -0.0142385801
REF_CORRECTION_SPIN = (0.6160922018, -0.6303307820)


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
    veff_ll = adapter.ints.veff_ll(adapter._dm_a_arr)

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

    Both split terms are linear in the density, so this is exact. It caught a real bug:
    contracting each channel against its own ``v_emb_spin`` (which does NOT sum to
    ``v_emb`` -- ``h_emb`` subtracts ``h_core``/``veff_ll`` once, so two channels
    subtract them twice) gave parts that disagreed with the whole.
    """
    adapter, alpha, _, _, result = open_shell
    energy = adapter.projection_energy(result, alpha)

    assert energy.is_spin_resolved
    assert sum(energy.correction_spin) == pytest.approx(energy.correction, abs=1e-10)
    assert sum(energy.projector_leak_spin) == pytest.approx(energy.projector_leak, abs=1e-10)
    # The split is informative, not cosmetic: two large nearly-cancelling channel
    # contributions that the spin-summed value hides.  Measured ~43x here (+0.616 and
    # -0.630 against a summed -0.014); assert an order of magnitude so the point is
    # pinned without over-fitting the ratio.
    assert abs(energy.correction_spin[0]) > 10 * abs(energy.correction)
    # ...and they really do have opposite signs, which is what makes the sum small.
    assert energy.correction_spin[0] * energy.correction_spin[1] < 0


def test_open_shell_energies_are_pinned(open_shell):
    """Regression anchors for the assembled open-shell energy.

    Deliberately tight (1e-6 Ha): these are deterministic closed-form contractions plus
    an exact FCI, with no sampling anywhere, so drift means something changed -- either
    here or in EmbASI. Widen only with a reason, and record it.
    """
    adapter, alpha, _, ham, result = open_shell
    energy = adapter.projection_energy(result, alpha)

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


def test_workflow_drives_the_open_shell_path(tmp_path):
    """``spin_downfold=True`` must reach the per-spin path from a config alone.

    The adapter-level tests above bypass ``EmbeddingWorkflow``; this pins that the CLI
    surface actually wires it, including the multi-cycle feedback (cycle >= 2 goes
    through ``run_low_level_a_only``, which must rebuild the per-spin Fock rather than
    drop it -- that regression made cycle 2 raise).
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
    # Cycle 1 reproduces the pinned single-shot total; cycle 2 moves off it, so only
    # assert the loop ran and landed somewhere physical.
    assert energy.is_spin_resolved
    assert -120.0 < energy.total < -100.0


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
    alpha, beta = adapter.build_orbitals_spin(virtual_localizer=loc)
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
