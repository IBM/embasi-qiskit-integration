# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Regression for the DFT-in-DFT dissociation energy (paper Eq. 2 / Eq. 19 direct).

A density-functional high level (PBE0, PBE, ...) is a *density* functional, not a
wavefunction method: the workflow evaluates ``E_high[γ̃^A]`` as a Kohn-Sham energy at
the embedded density over the FULL occupied space of subsystem A -- no active space,
no virtual budget, no solver.  ``ProjectionEmbeddingAdapter.dft_in_dft_energy`` reads
this total straight from EmbASI's native ``run()`` (``DFT_AinB_total_energy``), so the
embedded high-level SCF is EmbASI's own ``A_HL.run_emb_scf``, not an adapter-side one.
The dissociation energy is then formed directly on the complete embedding totals
(Eq. 19): ``ΔE_dissoc = E_emb(dimer) - 2*E_emb(monoA)`` for the methanol homodimer,
both hydroxyl groups high-level in the dimer, bare (isolated) monomer reference.

Two legs, both at the NATIVE s26[22] geometry, 6-31G, closed-shell:

* **PBE-in-PBE control** (``xc_hl == xc_ll == PBE``): the high and low functionals
  coincide, so the embedding is analytically a no-op on the energy.  ``Δ_HL`` must
  vanish on every leg -- the paper's Fig. 3B ``ΔΔE_dissoc``, negligible (~1e-6
  kJ/mol) rather than a literal machine zero, asserted to a TIGHT tolerance -- and
  ``ΔE_dissoc`` equals the pure-PBE low-level bracket, ``-40.18 kJ/mol``.  This is
  the paper's Fig. 3B cancellation reproduced end to end -- the strongest single
  check here: it exercises the entire embedded-SCF + footing + Eq. 2 assembly and
  demands the analytic cancellation from the functional-difference term, so any
  bookkeeping error in the assembly shows up.

* **PBE0-in-PBE** (``xc_hl == PBE0``): the genuine high-vs-low functional difference.
  DFT-in-DFT reproduces the full-PBE0 number ``ΔE_PBE0 = -39.06 kJ/mol`` up to the
  projection-embedding error, measured ``ΔE_dissoc = -39.25`` here (residual +0.19).
  Asserted with a small window around the measured value.

Marked ``embasi``: skipped unless ``EMBASI_AVAILABLE=1`` (needs a live EmbASI).  Each
leg is one supersystem SCF + EmbASI's embedded high-level SCF of subsystem A (a handful
of small 6-31G runs, no correlated solve), so it is not marked ``slow``.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.embasi]

_HA2KJMOL = 2625.4996394799
S26_METHANOL = 22
MONOMER_N_ATOMS = 6
DIMER_ACTIVE = [1, 5, 7, 11]  # both OH groups (monomer fragment [1,5] mirrored)
MONO_ACTIVE = [1, 5]  # single OH fragment on the isolated monomer

# PBE-in-PBE is an analytic no-op embedding, so the functional-difference term Δ_HL
# must vanish (Fig. 3B ΔΔE_dissoc: ~1e-6 kJ/mol, not a literal machine zero -- up to
# SCF/round-off noise).  1e-3 kJ/mol is ~4e-7 Ha -- far inside embedded-SCF
# convergence noise, yet still catches any real assembly regression.
TOL_CONTROL_DHL_KJMOL = 1.0e-3
# The low-level bracket must reproduce the pure-PBE supermolecular binding -40.18.
# It telescopes exactly from the e_low_total totals, so a 0.05 kJ/mol window (~2e-5
# Ha, SCF-convergence scale) is appropriate; -40.18 is quoted to 2 dp in the refs.
PBE_BRACKET_KJMOL = -40.18
TOL_BRACKET_KJMOL = 0.1
# PBE0-in-PBE lands at the MEASURED -39.25 (residual +0.19 from the -39.06 full-PBE0
# target, = the projection-embedding error on this system).  This value comes from
# EmbASI's native embedded high-level SCF (A_HL.run_emb_scf, read out of
# DFT_AinB_total_energy); it is closer to the full-PBE0 reference than the earlier
# adapter-side SCF was.  Asserted as a small window around the measured value -- this
# is a value regression, not a physics-zero.
PBE0_IN_PBE_MEASURED_KJMOL = -39.25
TOL_PBE0_KJMOL = 0.3


def _leg(*, n_atoms, active_atoms, xc_hl):
    """One embedding leg at the native geometry: (total, e_low_total, Δ_HL) in Ha.

    Routes on ``xc_hl`` inside ``run()``: a density functional takes the DFT-in-DFT
    path (embedded KS-SCF + Eq. 2 assembly, no active space / solver).  The full
    ``run()`` is used -- not a hand-assembled slice -- so the test exercises the same
    routing an end-user hits.
    """
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    wf = EmbeddingWorkflow(
        _cli_parse_args=False,
        s26_index=S26_METHANOL,
        n_atoms=n_atoms,
        active_atoms=active_atoms,
        basis="6-31G",
        xc_ll="PBE",
        xc_hl=xc_hl,
        max_cycles=1,
    )
    # A density-functional high level must take the DFT-in-DFT path; guard it so a
    # routing regression fails here rather than silently in the number.
    assert wf._is_dft_in_dft(), f"xc_hl={xc_hl} should route to DFT-in-DFT"
    energy = wf.run(log=lambda *_a, **_k: None)
    return energy.total, energy.e_low_total, energy.total - energy.e_low_total


def _dissociation(*, xc_hl):
    """(ΔE_dissoc, low-level bracket, Δ_HL residual) in kJ/mol via Eq. 19 direct."""
    e_dim, elow_dim, dhl_dim = _leg(n_atoms=None, active_atoms=DIMER_ACTIVE, xc_hl=xc_hl)
    e_mon, elow_mon, dhl_mon = _leg(n_atoms=MONOMER_N_ATOMS, active_atoms=MONO_ACTIVE, xc_hl=xc_hl)
    de = (e_dim - 2.0 * e_mon) * _HA2KJMOL
    bracket = (elow_dim - 2.0 * elow_mon) * _HA2KJMOL
    dhl_residual = (dhl_dim - 2.0 * dhl_mon) * _HA2KJMOL
    return de, bracket, dhl_residual


def test_pbe_in_pbe_control_is_pure_pbe():
    """PBE-in-PBE: Δ_HL vanishes (analytic 0) and ΔE_dissoc = the -40.18 PBE bracket.

    The high and low functionals coincide, so the embedding must not change the
    energy: the functional-difference term vanishes analytically and the dissociation
    energy collapses onto the pure-PBE low-level bracket.  This is the tightest check
    -- it demands the Fig. 3B cancellation from the full DFT-in-DFT assembly (embedded
    SCF + footing rebase + Eq. 2 correction), so any bookkeeping drift shows up here.
    """
    de, bracket, dhl_residual = _dissociation(xc_hl="PBE")
    assert abs(dhl_residual) < TOL_CONTROL_DHL_KJMOL, (
        f"PBE-in-PBE Δ_HL residual {dhl_residual:.2e} kJ/mol does not vanish -- the "
        f"embedding is a no-op when high==low, so the functional-difference term "
        f"MUST cancel (Fig. 3B ΔΔE_dissoc ~1e-6 kJ/mol); a value this large is a bug "
        f"in the Eq. 2 assembly, not physics."
    )
    assert abs(bracket - PBE_BRACKET_KJMOL) < TOL_BRACKET_KJMOL, (
        f"low-level bracket {bracket:.3f} kJ/mol != pure-PBE binding "
        f"{PBE_BRACKET_KJMOL} (size-consistency check on e_low_total)."
    )
    assert abs(de - PBE_BRACKET_KJMOL) < TOL_BRACKET_KJMOL, (
        f"PBE-in-PBE ΔE_dissoc {de:.3f} kJ/mol != the -40.18 PBE bracket; with Δ_HL "
        f"vanishing the dissociation energy must equal the bracket."
    )


def test_pbe0_in_pbe_reproduces_full_pbe0():
    """PBE0-in-PBE via DFT-in-DFT lands at -39.25 (full-PBE0 -39.06 + embedding err).

    The low-level bracket still telescopes to -40.18 (size-consistency holds
    independently of the high level), and the genuine PBE0-vs-PBE functional
    difference shifts ΔE_dissoc to the measured -39.25 -- within the projection-
    embedding error of the full-PBE0 reference -39.06.
    """
    de, bracket, dhl_residual = _dissociation(xc_hl="PBE0")
    assert abs(bracket - PBE_BRACKET_KJMOL) < TOL_BRACKET_KJMOL, (
        f"low-level bracket {bracket:.3f} kJ/mol != {PBE_BRACKET_KJMOL} -- the "
        f"bracket is high-level-independent, so PBE0 must give the same PBE bracket."
    )
    assert abs(de - PBE0_IN_PBE_MEASURED_KJMOL) < TOL_PBE0_KJMOL, (
        f"PBE0-in-PBE ΔE_dissoc {de:.3f} kJ/mol is outside "
        f"{PBE0_IN_PBE_MEASURED_KJMOL} +/- {TOL_PBE0_KJMOL} -- expected the DFT-in-DFT "
        f"value that reproduces full-PBE0 (-39.06) up to the ~0.3 kJ/mol embedding "
        f"error.  A large miss suggests PBE0 was routed through the WF path."
    )
    # The functional difference is real and nonzero (contrast the PBE control): its
    # sign/magnitude (~+1.4 kJ/mol) is the PBE0-PBE effect surfacing through Δ_HL.
    assert dhl_residual > 0.5, (
        f"PBE0-in-PBE Δ_HL residual {dhl_residual:.3f} kJ/mol is near zero -- PBE0 "
        f"should differ from PBE; a ~0 value means the functional was inert (PBE0 "
        f"collapsed to PBE, e.g. the WF-path nv=0 coincidence)."
    )
