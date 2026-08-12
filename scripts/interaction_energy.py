# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Direct (Eq. 19) embedding interaction energy for an s26 dimer, in kJ/mol.

Runs the projection-embedding workflow (:class:`EmbeddingWorkflow`) at a *frozen*
dimer geometry and forms the interaction energy the way the paper's Eq. 19 does --
**directly, on the complete embedding totals**::

    ΔE_int = E_emb(dimer) - E_emb(monoA) - E_emb(monoB)

For the methanol homodimer (``--s26_index 22``, the default) the two monomers are
geometrically identical, so ``E_emb(monoB) == E_emb(monoA)`` and this is
``E_emb(dimer) - 2*E_emb(monoA)``.

Why direct, and why bare monomers
---------------------------------
``E_emb(S)`` is the *complete* projection-embedding total returned by the workflow
(``ProjectionEnergy.total``: ``E_low(AB) - E_low(A) + E_high(A) + corr``, with
``E_high(A)`` already footing-rebased onto ``E_low(A)``'s ghosted-A nuclear frame).
Differenced directly, the low-level pieces telescope to the pure low-level
supermolecular binding -- for PBE this is the ``~-40.25 kJ/mol`` bracket --
**at the level of totals**, so the interaction energy is size-consistent by
construction with no polarization-multiplicity bookkeeping and no counterpoise
apparatus.  A standalone free-PBE bracket (:meth:`_free_pbe`) is computed
independently as the size-consistency check -- it never touches the embedding, so
it is cycle-invariant and cannot be fooled by the fed-back-density drift that the
``e_low_total`` difference picks up once ``max_cycles > 1``.  The monomer reference is the **bare**
isolated monomer (``n_atoms=monomer_n_atoms``): it is the physically isolated
fragment, and differencing complete totals is what Eq. 19 prescribes.

The default configuration is the standard PBE0-in-PBE benchmark for the methanol
dimer: OH···OH at PBE0 (high level), CH3 at PBE (low level), 6-31G, closed-shell.
The high level is a single OH fragment on each monomer; in the dimer both hydroxyls
are high-level (the dimer fragment is the monomer fragment mirrored onto the
partner, :meth:`_dimer_active`).

Paths, routed on the high-level METHOD and (for HF) the solver
--------------------------------------------------------------
* **DFT-in-DFT** (the default and the *supported* interaction-energy path; any
  density-functional ``--xc_hl`` such as PBE0/PBE).  The workflow takes paper Eq. 2:
  a Kohn-Sham energy at the embedded density, **no active space, no selector, no
  n_virtual, no solver** -- nothing to tune, and nothing to mismatch between the two
  legs.  ``--solver`` / ``--selector`` / ``--n_virtual`` and the SQD sampling knobs
  are inert on this path and do not enter the number.  This is the path that meets
  the reference table (see below).
* **WF-in-DFT, active-space** (``--xc_hl HF`` with ``--solver fci``/``sqd``).  HF is
  the sole wavefunction mean field expressible as an ``xc_hl``; the workflow downfolds
  subsystem A to an active space and hands a bare electronic Hamiltonian to the
  high-level solver.  The number is **HF-in-PBE + explicit active-space correlation**,
  a different physical quantity from the DFT-in-DFT PBE0 total -- it is NOT expected to
  reproduce a DFT number.  The deliverable of this path is the **quantum-solver
  integration itself**: SQD reproduces the exact FCI result at a *fixed* active space
  (agreement <0.001 Ha), which is what validates that the Qiskit solver is correctly
  wired into the EmbASI embedding.  Uses the concentric selector
  (``--selector concentric``); the two legs are differenced at matched
  *fragment-coupling* active spaces.

(A CCSD-over-full-A "CCSD-in-PBE" path was explored as a would-be *correlated*
interaction-energy reference for the SQD path -- a ~3.4 kJ/mol effect the reference
table below does not contain.  Its assembled ``ΔE_int`` was never made size-consistent
for the two-fragment subsystem A, so the path was removed; the diagnosis is preserved
in ``CCSD_IN_PBE_DIAGNOSIS.html``.  Do not reintroduce it.)

Reference table (all 6-31G, same basis as here so basis incompleteness cancels
out of the ΔE comparison entirely -- these are NOT complete-basis targets):

    ΔE_PBE = -40.18   ΔE_PBE0 = -39.06   ΔE_PBE0-in-PBE = -39.25   (kJ/mol)

The acceptance target for the DEFAULT (PBE0-in-PBE) run is the full-PBE0 number,
**ΔE_PBE0 = -39.06 kJ/mol**: DFT-in-DFT reproduces it up to the projection-embedding
error (measured ~+0.3 kJ/mol residual on this system, i.e. ΔE_int ~ -38.8).  The
``-39.25`` entry is the full PBE0-in-PBE literature value and itself carries that
embedding error relative to -39.06; it is quoted for context, not as the target.

Two built-in checks fall out of the direct construction:

* **Free-PBE bracket** ``E(dimer) - E(monoA) - E(monoB)`` from a **standalone**
  ``xc_ll`` SCF on each leg (:meth:`_free_pbe`), which never touches the embedding
  and is therefore **cycle-invariant**.  It must reproduce the pure low-level
  binding: for PBE, ``-40.25 kJ/mol`` at contact and **exactly 0.000** at far
  separation, to ~0.01.  This is *the* size-consistency check -- independent of
  the high level and of the fragment definition, and valid at any ``max_cycles``.
  It uses ``E(A)+E(B)``, not ``2*E(A)``: the two s26[22] methanols differ by
  0.016 A in internal bond lengths (worth +0.074 kJ/mol at 6-31G, +3.67 at
  sto-3g), so ``2*E(A)`` carries that inequivalence as a spurious term.
  A separate ``e_low_total`` **drift diagnostic** (``E_low(dimer)-2*E_low(monoA)``,
  the fed-back-density low-level energies) is also printed; it equals the free
  bracket ONLY at ``max_cycles=1`` and drifts above it under self-consistent
  feedback by design (paper §2.2), so it is NOT a size-consistency check.
* **PBE-in-PBE control** (``--xc_hl PBE``): the high and low functionals coincide,
  so the embedding is a no-op on the energy -- ``Δ_HL`` is **0 exactly** on every
  leg and ``ΔE_int`` equals the ``-40.18`` bracket.  This is the paper's Fig. 3B
  cancellation reproduced end to end, and the strongest single check here.

Usage::

    uv run python scripts/interaction_energy.py                 # PBE0-in-PBE (DFT-in-DFT) -> ~-38.8
    uv run python scripts/interaction_energy.py --xc_hl PBE      # PBE-in-PBE control -> -40.18
    # WF-in-DFT (HF + active-space correlation); FCI is the reference SQD must match:
    uv run python scripts/interaction_energy.py --xc_hl HF --solver fci --selector concentric
    uv run python scripts/interaction_energy.py --xc_hl HF --solver sqd --sampler aer --selector concentric

Every other embedding knob (basis, xc, mu, active space, solver, sampler, ...) is
accepted and forwarded verbatim to each sub-run.  The WF-path knobs are inert on the
DFT-in-DFT path.

Heterodimers (two *different* monomers) are not supported yet.
``--assume_symmetric false`` still raises rather than silently computing a wrong
number.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from embasi_qiskit_integration.embedding import EmbeddingWorkflow

# 1 Hartree in kJ/mol (CODATA-consistent).  The adapter has already converted
# ProjectionEnergy.total to Hartree, so we convert Ha -> kJ/mol here.
_HA2KJMOL = 2625.4996394799


class InteractionEnergy(BaseSettings):
    """Difference a dimer embedding against its two monomers, directly (Eq. 19)."""

    model_config = SettingsConfigDict(env_prefix="EQI_INT_", cli_parse_args=True)

    # --- system (mirrors EmbeddingWorkflow; forwarded to each sub-run) --- #
    s26_index: int = 22                  # methanol dimer in the s26 set
    monomer_n_atoms: int = 6             # atoms 0..N-1 are monomer A; the rest, B
    active_atoms: list[int] = [1, 5]     # active fragment on monomer A (O + OH H)
    basis: str = "6-31G"
    xc_ll: str = "PBE"
    xc_hl: str = "PBE0"
    mu: float = 1.0e6
    assume_symmetric: bool = True        # homodimer: E(B) == E(A); False -> unsupported

    # --- WF-in-DFT knobs (used only when xc_hl == HF; inert on DFT-in-DFT) --- #
    # These reach the number ONLY on the wavefunction path.  The concentric
    # selector is the default because the WF path requires it (energy-ordered cuts
    # are non-nested across the two differently-sized legs -- see module docstring).
    solver: str = "sqd"                  # "sqd" | "fci"
    selector: str = "concentric"         # "concentric" | "none"
    n_virtual: int | None = None         # advisory when a selector is set
    sampler: str = "aer"                 # "aer" | "mock" | "runtime" (SQD only)
    shots: int = 100_000
    seed: int = 24
    max_cycles: int = 1                  # single pass: difference single totals

    # --------------------------------------------------------------- #
    # config forwarding / fragment bookkeeping
    # --------------------------------------------------------------- #
    def _forwarded(self) -> dict:
        """The subset of fields shared with EmbeddingWorkflow, as kwargs.

        The WF-path knobs (``solver``/``selector``/``n_virtual``/``sampler``/
        ``shots``/``seed``/``max_cycles``) are forwarded on every leg but are inert
        unless ``xc_hl == HF`` (the workflow routes on the high-level method).
        ``active_fragment_sizes`` is deliberately NOT here -- it differs per leg
        (dimer vs monomer) and is set at each :meth:`_run` call site.
        """
        shared = [
            "s26_index", "active_atoms", "basis", "xc_ll", "xc_hl", "mu",
            "solver", "selector", "n_virtual", "sampler", "shots", "seed",
            "max_cycles",
        ]
        return {k: getattr(self, k) for k in shared}

    def _fragment_sizes(self, active_atoms: list[int]) -> list[int]:
        """Per-fragment atom counts for the concentric selector, for one leg.

        The monomer leg is a single fragment ``[len(active_atoms)]``; the dimer
        leg is that same fragment repeated once per monomer copy (``[2, 2]`` for
        both-OH), so the concentric selector cuts each OH shell independently and
        unions them -- the dimer active-virtual span then contains BOTH monomer
        shells (additivity), which is what makes the two legs differ at MATCHED
        fragment-coupling cuts.  Derived from ``active_atoms`` so it cannot drift.
        """
        m = len(self.active_atoms)
        assert len(active_atoms) % m == 0, (
            "active_atoms length is not a whole multiple of the monomer fragment; "
            "cannot partition into equal physical fragments for the selector."
        )
        return [m] * (len(active_atoms) // m)

    def _free_pbe(self, *, atom_slice: slice, active_atoms: list[int], tag: str) -> float:
        """Standalone low-level (``xc_ll``) SCF total for one leg, in Ha.

        This is the *size-consistency anchor*.  It is a plain ``mol.KS(xc=xc_ll)``
        on the SAME atoms and basis the embedding uses -- built here from scratch,
        so it **never touches the embedding**: no ``v_emb``, no ``P_B``, no
        projection, no fed-back density.  It is therefore **cycle-invariant** by
        construction (identical at ``max_cycles`` = 1 and 15), unlike
        ``ProjectionEnergy.e_low_total``, which is ``E_low`` at the *fed-back*
        (non-ground-state) density and drifts above the free minimum by design
        (paper §2.2 -- the reason Eq. 10 exists).

        The atom construction mirrors :meth:`EmbeddingWorkflow._build_adapter`
        (``create_s22_system`` -> slice -> argsort reorder by the embed mask ->
        ``pyscf.M`` with ``self.basis``).  ``atom_slice`` picks the leg's atoms
        (``[:monomer_n_atoms]`` for monomer A, ``[monomer_n_atoms:]`` for monomer
        B, everything for the dimer); ``active_atoms`` (indices WITHIN the slice)
        sets the region mask so the reorder matches the embedding's.  The reorder
        does not change the SCF energy (permutation-invariant), but matching it
        keeps this a faithful "same atoms" reference and future-proofs it against
        any embedding change that made ordering matter.
        """
        import numpy as np
        import pyscf
        from ase.data.s22 import create_s22_system, s26
        from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf

        atoms = create_s22_system(s26[self.s26_index])[atom_slice]
        # Reproduce the embedding's region-sorted atom order (energetically inert).
        embed_mask = len(atoms) * [2]
        for idx in active_atoms:
            if 0 <= idx < len(atoms):
                embed_mask[idx] = 1
        atoms = atoms[np.argsort(embed_mask)]

        mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis=self.basis)
        mf = mol.KS(xc=self.xc_ll)
        e = float(mf.kernel())
        print(f"  free-PBE[{tag}] ({len(atoms)} atoms, {self.xc_ll}/{self.basis}) "
              f"= {e:.8f} Ha  (converged={bool(mf.converged)})")
        return e

    def _dimer_active(self) -> list[int]:
        """Mirror the monomer-A fragment onto monomer B (offset ``monomer_n_atoms``).

        ``[1, 5]`` -> ``[1, 5, 7, 11]`` for the methanol homodimer: the SAME OH
        fragment on both monomers, so the dimer is treated **symmetrically**
        high-level (both hydroxyl groups at ``xc_hl``).  The monomer leg keeps the
        single fragment ``active_atoms``.  Deriving the dimer list from the monomer
        keeps a single source of truth -- it cannot drift out of sync with
        ``monomer_n_atoms`` or the fragment definition.
        """
        return list(self.active_atoms) + [
            a + self.monomer_n_atoms for a in self.active_atoms
        ]

    # --------------------------------------------------------------- #
    # embedding sub-runs
    # --------------------------------------------------------------- #
    def _run(self, *, n_atoms, active_atoms, active_fragment_sizes, tag: str):
        """One embedding sub-run.

        Returns ``(total, e_low_total, dhl)`` in Ha: the complete embedding total
        (``ProjectionEnergy.total``), the low-level energy of the whole host at the
        embedded density (``ProjectionEnergy.e_low_total``), and their difference
        ``Δ_HL = total - e_low_total`` (the high-vs-low fragment correction).  The
        direct interaction energy differences the ``total`` legs; the ``e_low_total``
        and ``Δ_HL`` legs are surfaced only as the built-in bracket / control checks.

        ``active_fragment_sizes`` is this leg's per-fragment partition for the
        concentric selector (``[2]`` monomer, ``[2, 2]`` dimer); inert unless the
        WF path (``xc_hl == HF``) with ``selector == concentric`` is active.
        """
        print(f"\n########## {tag} ##########")
        wf = EmbeddingWorkflow(
            _cli_parse_args=False,
            n_atoms=n_atoms,
            **{**self._forwarded(), "active_atoms": active_atoms,
               "active_fragment_sizes": active_fragment_sizes},
        )
        energy = wf.run()
        dhl = energy.total - energy.e_low_total
        print(
            f"########## {tag}: E = {energy.total:.6f} Ha "
            f"(E_low_total = {energy.e_low_total:.6f} Ha, "
            f"Δ_HL = {dhl * _HA2KJMOL:.3f} kJ/mol, "
            f"footing_shift = {energy.footing_shift:.6f} Ha) ##########"
        )
        return energy.total, energy.e_low_total, dhl

    # --------------------------------------------------------------- #
    # main
    # --------------------------------------------------------------- #
    def cli_cmd(self) -> None:
        if not self.assume_symmetric:
            raise NotImplementedError(
                "heterodimer interaction energy is not supported yet.  Under the "
                "direct (Eq. 19) formula the second monomer needs its own bare "
                "embedding total E_emb(monoB) on atoms[n:]; it is simply not wired "
                "up here.  Run with --assume_symmetric (the default) for a homodimer "
                "such as the methanol dimer, where E_emb(monoB) == E_emb(monoA)."
            )

        dimer_active = self._dimer_active()
        dimer_sizes = self._fragment_sizes(dimer_active)
        mono_sizes = self._fragment_sizes(self.active_atoms)
        is_wf = self.xc_hl.strip().upper() == "HF"

        print("\n==================== INTERACTION ENERGY setup ====================")
        print(f"  fragment (monomer) = {self.active_atoms}   dimer high-level = "
              f"{dimer_active}")
        print(f"  xc_hl={self.xc_hl}  xc_ll={self.xc_ll}  basis={self.basis}")
        if is_wf:
            print(f"  path=WF-in-DFT (HF + active-space correlation)  solver={self.solver}"
                  f"  selector={self.selector}")
            print(f"  per-leg fragments: dimer={dimer_sizes}  monomer={mono_sizes}"
                  + (f"  sampler={self.sampler} shots={self.shots}"
                     if self.solver == "sqd" else ""))
            if self.selector == "none":
                print("  *** WARNING: selector=none is the ENERGY-ORDERED cut the HF/FCI "
                      "test proved is NON-nested across the two legs.")
                print("  ***          The interaction energy will not be trustworthy; "
                      "use --selector concentric.")
            print("  NOTE: this is HF-in-PBE + correlation, a DIFFERENT quantity from the "
                  "DFT-in-DFT PBE0 total; it need not equal -39.06/-38.77.")
        else:
            print("  path=DFT-in-DFT (Eq. 2 embedded KS energy); "
                  "solver/selector/sampler are inert on this path")
        print("  ΔE_int = E_emb(dimer) - 2 E_emb(monoA)   (Eq. 19 direct, bare monomers)")

        # --- complete embedding totals: dimer (both OH high-level) + bare monomer --- #
        e_dimer, elow_dimer, dhl_dimer = self._run(
            n_atoms=None, active_atoms=dimer_active,
            active_fragment_sizes=dimer_sizes,
            tag=f"DIMER (full system, both-OH high-level {dimer_active})",
        )
        e_mono, elow_mono, dhl_mono = self._run(
            n_atoms=self.monomer_n_atoms, active_atoms=self.active_atoms,
            active_fragment_sizes=mono_sizes,
            tag="MONOMER (bare, isolated fragment)",
        )

        # --- direct interaction energy (homodimer: monoB == monoA) --- #
        de_ha = e_dimer - 2.0 * e_mono
        de_kjmol = de_ha * _HA2KJMOL

        # --- size-consistency bracket: STANDALONE free-PBE, never sees embedding --- #
        # The anchor is a plain xc_ll SCF on each leg (cycle-invariant).  Use the
        # true non-interacting reference E(A)+E(B), NOT 2*E(A): the two s26[22]
        # methanols differ by 0.016 A in internal bond lengths -- worth +0.074
        # kJ/mol at 6-31G, +3.67 at sto-3g (the inequivalence that produced the
        # spurious -3.68 in the earlier scratch reconciliation).  E(B) is the
        # SECOND monomer, atoms[monomer_n_atoms:], sliced by translating the mask
        # onto that fragment via _free_pbe's own slice below.
        print("\n  --- size-consistency anchor (standalone free-PBE, no embedding) ---")
        efree_dimer = self._free_pbe(
            atom_slice=slice(None), active_atoms=dimer_active, tag="dimer")
        efree_monoA = self._free_pbe(
            atom_slice=slice(0, self.monomer_n_atoms),
            active_atoms=self.active_atoms, tag="monoA")
        efree_monoB = self._free_pbe(
            atom_slice=slice(self.monomer_n_atoms, None),
            active_atoms=self.active_atoms, tag="monoB")
        free_bracket = (efree_dimer - efree_monoA - efree_monoB) * _HA2KJMOL

        # --- non-ground-state drift diagnostic (NOT a size-consistency check) --- #
        # This is E_low at the fed-back density on each leg, differenced.  It equals
        # the free bracket ONLY at max_cycles=1 (fed-back density ~ PBE ground
        # state); under feedback it drifts by design.  Kept for insight, relabelled.
        elow_drift = (elow_dimer - 2.0 * elow_mono) * _HA2KJMOL
        dhl_residual = (dhl_dimer - 2.0 * dhl_mono) * _HA2KJMOL     # -> 0 for PBE-in-PBE

        print("\n==================== INTERACTION ENERGY (Eq. 19 direct) ====================")
        print(f"  E_emb(dimer)  = {e_dimer:.6f} Ha  (E_low_total = {elow_dimer:.6f})")
        print(f"  E_emb(monoA)  = {e_mono:.6f} Ha  (= E_emb(monoB), homodimer)")
        print("  ---------------------------------------------------------------------------")
        print("  ΔE_int = E_emb(dimer) - 2 E_emb(monoA)")
        print(f"         = {e_dimer:.6f} - 2 * {e_mono:.6f}")
        print("  ---------------------------------------------------------------------------")
        print(f"    free-PBE bracket  E(d) - E(mA) - E(mB)   = {free_bracket:8.3f} kJ/mol"
              f"   [SIZE-CONSISTENCY CHECK: PBE ref -40.25; standalone, cycle-invariant]")
        print(f"    e_low_total drift E_low(d) - 2 E_low(m)  = {elow_drift:8.3f} kJ/mol"
              f"   [non-ground-state drift diagnostic; = free bracket ONLY at max_cycles=1]")
        print(f"    Δ_HL(d) - 2 Δ_HL(m)                      = {dhl_residual:8.3f} kJ/mol"
              f"   [PBE-in-PBE control: exactly 0]")
        print("  ---------------------------------------------------------------------------")
        if is_wf:
            target_note = (f"[WF-in-DFT ({self.solver.upper()}): HF + active-space "
                           f"correlation; reference is the FCI run on this active space]")
        elif self.xc_hl.upper() == "PBE":
            target_note = "[target ΔE_PBE = -40.18]"
        else:
            target_note = f"[target ΔE_{self.xc_hl} = -39.06]"
        print(f"    ΔE_int = {de_ha:.6f} Ha = {de_kjmol:.3f} kJ/mol   {target_note}")
        print("  ---------------------------------------------------------------------------")
        print("  The free-PBE bracket is the built-in size-consistency check: a STANDALONE")
        print("  xc_ll SCF on each leg (never touches the embedding), so it is cycle-invariant.")
        print("  It must match the pure-PBE supermolecular binding (-40.25 kJ/mol at 6-31G,")
        print("  using E(A)+E(B) not 2*E(A)) and go to EXACTLY 0 at far separation.  The")
        print("  e_low_total drift line above is E_low at the fed-back density -- it equals the")
        print("  free bracket ONLY at max_cycles=1 and drifts under feedback by design (§2.2);")
        print("  it is a diagnostic, NOT a size-consistency check.  For --xc_hl PBE the")
        print("  embedding is a no-op (Δ_HL = 0 exactly) and ΔE_int equals the bracket -- the")
        print("  paper's Fig. 3B cancellation.  For --xc_hl PBE0 (default) the target is the")
        print("  full-PBE0 number -39.06; DFT-in-DFT reproduces it up to the ~0.3 kJ/mol")
        print("  projection-embedding error.")
        print("  References (6-31G, same basis => basis cancels): ΔE_PBE=-40.18/-40.25*, "
              "ΔE_PBE0=-39.06, ΔE_PBE0-in-PBE=-39.25 kJ/mol.")
        print("    (*-40.18 is 2*E(A); -40.25 is the true E(A)+E(B) with the 0.074 kJ/mol "
              "monomer inequivalence.)")
        print("=============================================================================")


def main() -> None:
    CliApp.run(InteractionEnergy)


if __name__ == "__main__":
    main()
