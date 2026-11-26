# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Temporary compatibility shim for EmbASI's missing SIESTA (ASI flavour 3) support.

EmbASI's ``qm-code-adapter`` branch registers adapters for FHI-aims and PySCF
only, and its generic ASI extraction assumes the last three Hamiltonian exports
are kinetic, electrostatic+XC and total.  SIESTA's ASI implementation exports
one *total* Hamiltonian per SCF step, so the generic path either fails
(``KeyError: 1``) or silently mislabels successive totals.

This module is deliberately self-contained and removable:

* :func:`install_siesta_adapter` registers a :class:`SiestaAdapter` under
  ``"Siesta"`` only when EmbASI does not already provide one;
* :class:`SiestaAdapter` maps EmbASI's input directives onto SIESTA fdf keys
  and overrides :meth:`extract_matrices` for the one-H-per-step interface;
* :func:`prepare_projection_embedding` fills the one piece of layer state
  EmbASI's ``ProjectionEmbedding`` omits (``AB_LL.input_fragment_nelectrons``),
  which is also why the shim is needed for any ``needs_nonscf_ene_corr`` code;
* :func:`prime_layer` does the same for a standalone ``AtomsEmbed`` used by
  :class:`~embasi_qiskit_integration.backends.callback.LayerEvaluator`.

Delete this module and its call sites once EmbASI ships a Siesta adapter
and the upstream fixes referenced in the docstrings below.

TODO(embasi-siesta-shim) removal checklist -- grep this tag:
  * delete ``backends/_embasi_compat.py`` (this module);
  * ``backends/siesta.py``: remove the ``install_siesta_adapter()`` call and
    import, and the ``calc.write_inputfiles = calc.write_input`` alias;
  * ``backends/callback.py``: remove the ``prime_layer(self._layer)`` call;
  * ``embedding.py``: remove the ``prepare_projection_embedding(projection)``
    call;
  * ``tests/test_siesta_asi.py``: remove/adapt the tests marked with the same
    tag (shim registration, directive mapping, extraction/priming);
  * README: drop the "compatibility shim" sentence from the SIESTA section.
  The ASI calculator construction and ``probe_basis_asi`` are NOT shim work:
  keep them once the shim is gone.

Upstream issues this works around (EmbASI ``qm-code-adapter`` @ 0245a38):

* ``atoms_embedding_asi.py:155`` -- ``calc_initializer`` only calls
  ``write_inputfiles``; ASE's ``Siesta`` calculator only has ``write_input``.
* ``embedding.py`` ``construct_embedding_potential`` -- ``AB_LL.run_scf()``
  runs before ``AB_LL.input_fragment_nelectrons`` is assigned.
* ``atoms_embedding_asi.py`` ``AtomsEmbed.free_atom_nelectrons`` sums
  ``atoms.numbers``, which is right for all-electron codes but overcounts the
  core for a pseudopotential code (H2O: 10 instead of SIESTA's 8 valence
  electrons).  SPADE then fills ten electrons and selects an 8-electron A
  fragment.  The shim makes the property valence-aware and uses the same count
  in ``prime_layer``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from embasi.ks_array import SpinKpointArray
from embasi.qmcode_adapters import QMCodeASIAdapter

#: ASI flavour ID for SIESTA (ASI upstream ``include/asi.h``).
ASI_SIESTA = 3

#: Opt-in: run the non-SCF one-step layers (A_LL/B_LL, the imported-density
#: evaluations) in a *neutral* cell even when the layer carries a fragment charge.
#:
#: EmbASI derives a layer's SIESTA ``NetCharge`` from
#: ``fragment_total_charge = input_total_charge + free_atom_nelectrons -
#: input_fragment_nelectrons``.  For A_LL that is ``0 + 28 - 14 = +14``: a 12-atom
#: supersystem cell with 28 valence ions holding only SPADE's 14 A-fragment
#: electrons.  SIESTA neutralises the missing 14 e with a uniform background and
#: refuses the Makov-Payne correction for the orthorhombic cells this backend
#: builds (``madelung: WARNING: Charged system, but not SC, FCC or BCC cell``),
#: which leaves a cell-size-dependent electrostatic gauge in the A_LL
#: Hamiltonian (measured -0.63 Ha between the 10 A and 20 A cells, overlap
#: identical); EmbASI's exported ``v_emb = H_AB - H_A_LL`` carries it.
#:
#: With this flag set, a layer whose ``density_matrix_in`` is live -- exactly the
#: ``run_noscf`` one-step path that imports a fragment density -- omits
#: ``NetCharge`` (SIESTA's neutral default), matching what ``eval_ll`` already
#: does.  The full supersystem ``run_scf`` (``density_matrix_in is None``) keeps
#: its charge, and ``input_fragment_nelectrons`` is untouched, so EmbASI's
#: non-SCF eigenvalue-sum correction still fills the fragment electron count.
#: Off by default.  Validated at 0.8 and 2.5 A, plus a charged/neutral one-step
#: probe at the shipped 10 A cell: the A_LL H and S at the imported density are
#: bit-identical between the charged and neutral cells (max|dH| = 0), the
#: audit's -0.63 Ha 10->20 A shift being a cell-size effect, not a charge
#: effect.  The only thing the flag moves is SIESTA's post-step ``E_KS`` scalar
#: (a neutral cell nominally occupies 28 electrons); that scalar cancels
#: between ``E_low_A`` and the same-density reference, and the WF assembly
#: cancels ``H_A_LL`` in ``F_emb = H_A_LL + (H_AB - H_A_LL) + P_B``, so
#: ``v_emb``/``h1``/``E_emb`` are unchanged.  The flag is a control for that
#: check, not a validated default.
NEUTRAL_LAYER_ENV = "EQI_SIESTA_NEUTRAL_A_LAYER"


def _neutral_layer_enabled() -> bool:
    """Whether :data:`NEUTRAL_LAYER_ENV` opts the one-step layers into a neutral cell."""
    raw = os.environ.get(NEUTRAL_LAYER_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


#: fdf keys the adapter always keeps at these values.
#:
#: ``SCF.Mix density-matrix`` + ``DM.MixSCF1 False`` are load-bearing, not
#: stylistic.  SIESTA >= 4.1 defaults to *Hamiltonian* mixing, which evaluates
#: the one-step run's energy decomposition (``total.log``) at the post-step
#: output density ``DM_out``.  The shim's same-density reference reads ``Exc``
#: from that decomposition and pairs it with the imported input density
#: ``D_in``, so under H mixing it silently evaluates the XC term at the wrong
#: density (measured on H2O: 97 eV = 3.58 Ha, geometry dependent).  DM mixing
#: evaluates the in-loop decomposition at ``D_in``; disabling first-step mixing
#: keeps the final post-SCF energy at ``DM_out``, which is the density
#: EmbASI's ``needs_nonscf_ene_corr`` correction is written for.
#:
#: TODO(embasi-siesta-shim): delete these two keys with this module once EmbASI
#: either registers its own Siesta adapter (and gets the protocol right) or
#: consumes the input-density energy terms directly instead of parsing Exc.
_NOSCF_FDF = {
    "MaxSCFIterations": 1,
    "SCFMustConverge": False,
    "DM.UseSaveDM": False,
    "SCF.Mix": "density-matrix",
    "DM.MixSCF1": False,
}


def _fdf(calc: Any) -> dict[str, Any]:
    return dict(calc.parameters.get("fdf_arguments", {}) or {})


def _set_fdf(calc: Any, values: dict[str, Any]) -> Any:
    calc.parameters["fdf_arguments"] = {**_fdf(calc), **values}
    return calc


class SiestaAdapter(QMCodeASIAdapter):
    """EmbASI adapter for SIESTA's ASI flavour 3 (temporary shim).

    TODO(embasi-siesta-shim): delete with this module once EmbASI registers a
    ``"Siesta"`` adapter (or adopts this one upstream).
    """

    @property
    def asi_flavour(self) -> int:
        return ASI_SIESTA

    @property
    def needs_nonscf_ene_corr(self) -> bool:
        # ASI_init rejects nscf <= 0, so a "non-SCF" evaluation is one SCF step
        # at the imported density; the eigenvalue-sum energy must be replaced
        # by Tr(DM_in H[DM_in]).
        return True

    # ------------------------------------------------------------------ #
    # Input directives
    # ------------------------------------------------------------------ #
    def set_ghost_atoms(self, atomsembed: Any, calc: Any, ghost_list: Any) -> Any:
        if any(bool(ghost) for ghost in ghost_list):
            raise NotImplementedError(
                "SIESTA ASI has no ghost/empty-species support; BSSE/ghost-based "
                "basis truncation cannot run with --qm_code siesta"
            )
        return calc

    def set_noscf_calc_params(self, calc: Any) -> Any:
        return _set_fdf(calc, _NOSCF_FDF)

    def set_max_scf_cycles(self, calc: Any, max_cycles: int) -> Any:
        return _set_fdf(
            calc,
            {"MaxSCFIterations": max(1, int(max_cycles)), "SCFMustConverge": True},
        )

    def set_full_scf_calc(self, calc: Any) -> Any:
        return _set_fdf(calc, {"SCFMustConverge": True, "DM.UseSaveDM": False})

    def set_postscf_keyword(self, calc: Any, postscf_method: Any) -> Any:
        return calc

    def set_scalapack_blocksize(self, calc: Any, blacs_blocksize: Any) -> Any:
        return calc

    def set_embasi_calculation_type(self, calc: Any, calc_type: str) -> Any:
        return calc

    def override_basis_order(self, calc: Any) -> Any:
        return calc

    def set_qm_localise(self, calc: Any) -> Any:
        return calc

    def set_qm_total_charge(self, atomsembed: Any, calc: Any, charge: float) -> Any:
        # EmbASI calls this with each *layer's* fragment charge, which is not
        # the supersystem charge (a neutral system still has charged fragments).
        charge = float(charge)
        # Opt-in (see NEUTRAL_LAYER_ENV): a one-step layer that imports a
        # fragment density runs in a neutral supersystem cell, so the charged
        # fragment's uniform-background gauge cannot enter H_A_LL/v_emb.  The
        # discriminator is EmbASI's own run_noscf path, which sets
        # ``density_matrix_in`` before this initializer; the full supersystem
        # ``run_scf`` leaves it None.  The non-SCF energy correction keeps the
        # layer's ``input_fragment_nelectrons`` (the adapter never touches it),
        # so its eigenvalue sum still fills the fragment count.
        if _neutral_layer_enabled() and getattr(atomsembed, "density_matrix_in", None) is not None:
            fdf = _fdf(calc)
            fdf.pop("NetCharge", None)
            calc.parameters["fdf_arguments"] = fdf
            return calc
        values = {"NetCharge": charge} if abs(charge) > 1e-12 else {}
        return _set_fdf(calc, values)

    # ------------------------------------------------------------------ #
    # Output directives
    # ------------------------------------------------------------------ #
    def get_energy(self, atomsembed: Any) -> float:
        return atomsembed.atoms.calc.get_potential_energy()

    def get_forces(self, atomsembed: Any) -> Any:
        return atomsembed.atoms.calc.get_forces()

    def get_decomposed_energy_from_file(self, atomsembed: Any) -> Any:
        # The base implementation parses FHI-aims' asi.log; SIESTA has no
        # equivalent decomposition, and Layers only need the energy above.
        return atomsembed

    def extract_matrices(self, atomsembed: Any) -> dict[str, Any]:
        asi = atomsembed.atoms.calc.asi
        n_spins = int(asi.n_spin)
        n_kpoints = int(asi.n_kpts)

        if not asi.ham_storage:
            raise RuntimeError(
                "SIESTA ASI exported no Hamiltonian; ASI_run() did not reach the "
                "Hamiltonian callback (check that the SCF actually ran)"
            )
        # EmbASI's ham_saving_callback keeps a rolling window of at most three
        # exports.  SIESTA exports only H_total, once per SCF step, so the last
        # entry is the Hamiltonian at the final density of this run.
        last_index = max(0, int(getattr(asi, "ham_count", len(asi.ham_storage))) - 1)
        last = asi.ham_storage[last_index]
        ham_tot = SpinKpointArray(last, n_spins, n_kpoints)
        zero = SpinKpointArray(
            {key: np.zeros_like(np.asarray(value)) for key, value in last.items()},
            n_spins,
            n_kpoints,
        )

        # TODO(embasi-siesta-shim): delete this override once EmbASI ships a
        # Siesta adapter with the correct one-H-per-step semantics.
        if not asi.dm_storage:
            raise RuntimeError("SIESTA ASI exported no density matrix")

        # EmbASI clears density_matrix_in as soon as run_noscf/run_scf returns,
        # but the energy-reference alignment needs the *input* density used for
        # this layer.  extract_matrices runs while it is still live.
        if getattr(atomsembed, "density_matrix_in", None) is not None:
            raw = atomsembed.density_matrix_in
            # SpinKpointArray has n_spins/n_kpoints and no __array__; take the
            # spin-summed (nao, nao) block.  Restricted SIESTA has n_spin=1.
            if getattr(raw, "n_spins", None) is not None:
                blocks = [np.asarray(raw[s, 0]) for s in range(int(raw.n_spins))]
            else:
                arr = np.asarray(raw)
                while arr.ndim > 2:
                    arr = arr[0] if arr.shape[0] == 1 else arr.sum(axis=0)
                blocks = [arr]
            density = sum(blocks)
            atomsembed._siesta_input_density = np.array(density.real, copy=True)

        return {
            "n_spins": n_spins,
            "n_kpoints": n_kpoints,
            "n_basis": int(asi.n_basis),
            "basis_atoms": asi.basis_atoms,
            # K + V must equal H_total for EmbASI's construct_embedded_fock();
            # SIESTA gives no split, so put the whole operator in the kinetic
            # slot and leave the second empty.
            "ham_kin": ham_tot,
            "ham_estat_xc": zero,
            "ham_tot": ham_tot,
            "ovlp": SpinKpointArray(asi.overlap_storage, n_spins, n_kpoints),
            "dm": SpinKpointArray(asi.dm_storage[0], n_spins, n_kpoints),
            "dm_localised": None,
        }


def install_siesta_adapter() -> bool:
    """Register :class:`SiestaAdapter` if EmbASI has no ``"Siesta"`` adapter.

    TODO(embasi-siesta-shim): delete with this module; the call site in
    ``backends/siesta.py`` must go too.

    Returns:
        True when this shim installed the adapter, False when EmbASI already
        provides one (the shim is then bypassed).
    """
    from embasi import qmcode_adapters

    qmcode_adapters.ASI_flavour.setdefault("SIESTA", ASI_SIESTA)
    if "Siesta" in qmcode_adapters.implemented_calculators:
        return False
    qmcode_adapters.implemented_calculators["Siesta"] = SiestaAdapter
    _install_siesta_valence_electron_count()
    return True


def _neutral_electrons(atoms: Any) -> float:
    return float(np.sum(np.asarray(atoms.numbers, dtype=float)))


def _pseudo_dir(layer: Any) -> str | None:
    """The pseudopotential directory used by a SIESTA layer's calculator."""
    from .siesta import PSEUDO_ENV

    # ``atoms.calc`` is ASI's ASE wrapper after a run; the ASE Siesta
    # calculator that owns ``pseudo_path`` is the layer's runtime/initial calc.
    owners = (
        getattr(layer, "runtime_calc", None),
        getattr(layer, "initial_calc", None),
        getattr(getattr(layer, "atoms", None), "calc", None),
    )
    for owner in owners:
        parameters = getattr(owner, "parameters", None)
        try:
            path = parameters.get("pseudo_path") if parameters is not None else None
        except AttributeError:
            path = None
        if path:
            return str(path)
    return os.environ.get(PSEUDO_ENV)


def _siesta_valence_electrons(layer: Any) -> float:
    """Valence electron count of a SIESTA layer's active atoms.

    EmbASI's :attr:`AtomsEmbed.free_atom_nelectrons` sums ``atoms.numbers``,
    which is correct for all-electron codes but counts the core electrons for a
    pseudopotential code: H2O would contribute 10 instead of SIESTA's 8 valence
    electrons.  SPADE then fills ten electrons and partitions them as 8 on O,
    selecting an 8-electron active space instead of neutral oxygen's 6.  Use the
    pseudopotentials' ``z-pseudo``/``Z_valence`` instead.
    """
    from .siesta import _valence_charge

    pseudo_dir = _pseudo_dir(layer)
    if not pseudo_dir:
        raise RuntimeError(
            "a SIESTA layer has no pseudopotential directory; set "
            "$SIESTA_PP_PATH (or the calculator's pseudo_path)"
        )

    atoms = layer.atoms
    info = getattr(layer, "basis_info", None)
    if info is None or (
        getattr(layer, "truncate", False) and len(atoms) == getattr(info, "trunc_natoms", None)
    ):
        active = np.arange(len(atoms))
    else:
        raw = np.asarray(info.active_atoms)
        active = np.flatnonzero(raw) if raw.dtype == bool else raw.astype(int).ravel()

    symbols = [atoms[int(index)].symbol for index in active]
    return float(sum(_valence_charge(symbol, pseudo_dir) for symbol in symbols))


def _free_atom_electrons(layer: Any) -> float:
    """Neutral electron count for a layer, valence-aware for SIESTA."""
    if isinstance(getattr(layer, "qm_adapter", None), SiestaAdapter):
        return _siesta_valence_electrons(layer)
    return _neutral_electrons(layer.atoms)


_VALENCE_COUNT_PATCHED = False


def _install_siesta_valence_electron_count() -> None:
    """Make ``AtomsEmbed.free_atom_nelectrons`` valence-aware for SIESTA.

    TODO(embasi-siesta-shim): delete once EmbASI's adapter can report a
    pseudopotential-aware neutral electron count (the same value SPADE, the
    fragment charges and the non-SCF energy correction must share).
    """
    global _VALENCE_COUNT_PATCHED
    if _VALENCE_COUNT_PATCHED:
        return

    from embasi.atoms_embedding_asi import AtomsEmbed

    original = AtomsEmbed.free_atom_nelectrons.fget

    def free_atom_nelectrons(self: Any) -> float:
        if isinstance(getattr(self, "qm_adapter", None), SiestaAdapter):
            return _siesta_valence_electrons(self)
        return original(self)

    AtomsEmbed.free_atom_nelectrons = property(free_atom_nelectrons)
    _VALENCE_COUNT_PATCHED = True


def prime_layer(layer: Any) -> bool:
    """Give an ``AtomsEmbed`` the state EmbASI's non-SCF correction assumes.

    TODO(embasi-siesta-shim): delete with this module (and the call in
    ``backends/callback.py``) once EmbASI sets this state itself or exposes the
    non-SCF energy correction without it.

    ``AtomsEmbed.run`` calls ``ev_solve_and_sum_evs`` whenever the adapter's
    ``needs_nonscf_ene_corr`` is true, which reads
    ``layer.input_fragment_nelectrons`` and (through
    ``fragment_total_charge``) ``layer.basis_info``.  ``ProjectionEmbedding``
    sets both for its layers, but a standalone ``AtomsEmbed`` (as used by
    :class:`LayerEvaluator`) has neither.
    """
    adapter = getattr(layer, "qm_adapter", None)
    if not getattr(adapter, "needs_nonscf_ene_corr", False):
        return False
    if not hasattr(layer, "_basis_info"):
        from embasi.basis_info import Basis_info

        info = object.__new__(Basis_info)  # avoid Basis_info's console banner
        info.full_natoms = info.trunc_natoms = len(layer.atoms)
        info.active_atoms = np.ones(len(layer.atoms), dtype=bool)
        layer.basis_info = info
    if not hasattr(layer, "_input_fragment_nelectrons"):
        # A LayerEvaluator covers the whole system, so the "fragment" is the
        # whole system: neutral electron count keeps fragment_total_charge
        # equal to the layer's input_total_charge (EmbASI's convention).  For
        # SIESTA this must be the pseudopotential valence count, not Z.
        layer.input_fragment_nelectrons = _free_atom_electrons(layer)
    return True


def prepare_projection_embedding(projection: Any) -> bool:
    """Prime ``ProjectionEmbedding.AB_LL`` before its first ``run_scf``.

    TODO(embasi-siesta-shim): delete with this module (and the call in
    ``embedding.py``) once EmbASI assigns ``AB_LL.input_fragment_nelectrons``
    before ``construct_embedding_potential`` calls ``AB_LL.run_scf()``.

    ``construct_embedding_potential`` calls ``AB_LL.run_scf()`` before assigning
    ``AB_LL.input_fragment_nelectrons``; for a code that needs the non-SCF
    energy correction that raises ``AttributeError``.  ``AB_LL`` is the whole
    supersystem, so its neutral count is the right value.
    """
    layer = getattr(projection, "AB_LL", None)
    if layer is None:
        return False
    return prime_layer(layer)
