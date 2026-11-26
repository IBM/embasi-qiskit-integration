# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""CSV logging of per-cycle diagnostics from the embedding outer loop."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

DIAGNOSTICS_CSV_COLUMNS = [
    "run_id",
    "geometry_file",
    "geometry_parameter",
    "algorithm",
    "backend_name",
    "cycle",
    "converged",
    "total_cycles",
    "cas_norb",
    "cas_nelec",
    "active_space_indices",
    "selector",
    "selected_virtuals",
    "frozen_occupied",
    "norb",
    "nelec_alpha",
    "nelec_beta",
    "E_solver",
    "E_low_total",
    "E_low_A",
    "E_high_A",
    "e_core",
    "correction",
    "footing_shift",
    "projector_leak",
    "p_b_leak",
    "delta_E",
    "embedding_energy",
    "number_unique_bitstrings",
    "final_sqd_subspace_dimension",
    "rdm1_trace",
    "max_delta_gamma_A",
    "natural_occupations",
    "rdm1",
]


def append_diagnostics_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one diagnostics row to a CSV file, writing the header if needed.

    Args:
        path: Path to the CSV file. Parent directories are created if needed.
        row: Dictionary mapping column names to values. Missing keys are rendered as empty cells.

    The file is opened in append mode with newline="" (per csv module convention). If the file
    does not exist or is empty, the header row is written first. All rows are flushed to disk
    immediately after writing to ensure partial results survive a crash.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=DIAGNOSTICS_CSV_COLUMNS, restval="", extrasaction="ignore"
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def _round_floats(obj: Any, decimals: int = 3) -> Any:
    """Recursively round floats in a nested structure to the specified decimal places."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [_round_floats(item, decimals) for item in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _round_floats(obj.tolist(), decimals)
    return obj


def json_cell(value: Any) -> str:
    """Serialize a value to JSON for CSV embedding, or return empty string if None.

    For NumPy arrays, converts to list first. For all other values, uses json.dumps.
    All floats are rounded to 3 decimal places. Returns empty string for None.
    """
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    value = _round_floats(value)
    return json.dumps(value)


def _downfolded_onebody_payload(emb, orbitals, coefficients: np.ndarray) -> dict[str, Any]:
    """The operator ingredients needed to score arbitrary virtual candidates offline.

    The standard dump stores only the *selected* active columns, so a candidate pair
    that was not selected cannot be scored.  This extended block adds the full
    embedded one-body ingredients -- ``h_emb``, the high-level HF mean field at the
    inactive density (``veff_hf(dm_in)``), the low-level mean field, the canonical
    virtual block and the AVAS target metric -- so an analysis script can form
    ``C^T (h_emb + veff_hf(dm_in)) C`` for any occupied/virtual choice and compare
    its downfolded gap with the F_emb spectrum.  Also carries ``v_emb`` and the
    initial subsystem-A density, which fix the paper-Eq.-8 correction for a candidate
    without rerunning the embedding.

    Every array is optional: a backend/selector that cannot supply a piece drops it
    rather than failing the diagnostic.
    """

    def _array(value) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value, dtype=float)

    def _put(key: str, value) -> None:
        array = _array(value)
        if array is not None:
            out[key] = array

    out: dict[str, Any] = {}
    ints = getattr(emb, "ints", None)
    h_emb = _array(getattr(emb, "h_emb", None))
    f_emb = _array(getattr(emb, "_fock_arr", None))
    if h_emb is not None:
        out["h_emb"] = h_emb
    if f_emb is not None:
        out["f_emb"] = f_emb
        if h_emb is not None:
            out["veff_ll_a"] = f_emb - h_emb
    _put("v_emb", getattr(emb, "v_emb", None))
    _put("p_b", getattr(emb, "p_b", None))
    _put("dm_a_init", getattr(emb, "_dm_a_init", None))
    if ints is not None:
        hcore = getattr(ints, "hcore", None)
        if callable(hcore):
            try:
                _put("hcore", hcore())
            except Exception:  # noqa: BLE001 - extended diagnostic is optional
                pass

    c_in = np.asarray(coefficients[:, orbitals.inactive], dtype=float)
    dm_in = 2.0 * (c_in @ c_in.T)
    out["dm_in"] = dm_in
    n_occ = int(orbitals.n_occ)
    active = np.asarray(orbitals.active, dtype=int)
    active_occ = active[active < n_occ]
    if active_occ.size:
        c_act_occ = np.asarray(coefficients[:, active_occ], dtype=float)
        dm_act_occ = 2.0 * (c_act_occ @ c_act_occ.T)
        out["dm_act_occ"] = dm_act_occ
    else:
        dm_act_occ = None

    if ints is not None:
        veff_hf = getattr(ints, "veff_hf", None)
        if callable(veff_hf):
            try:
                out["veff_hf_in"] = np.asarray(veff_hf(dm_in), dtype=float)
            except Exception:  # noqa: BLE001 - extended diagnostic is optional
                pass
            if dm_act_occ is not None:
                try:
                    out["veff_hf_act_occ"] = np.asarray(veff_hf(dm_act_occ), dtype=float)
                except Exception:  # noqa: BLE001
                    pass
        veff_ll = getattr(ints, "veff_ll", None)
        if callable(veff_ll):
            try:
                out["veff_ll_in"] = np.asarray(veff_ll(dm_in), dtype=float)
            except Exception:  # noqa: BLE001
                pass
            if dm_act_occ is not None:
                try:
                    out["veff_ll_act_occ"] = np.asarray(veff_ll(dm_act_occ), dtype=float)
                except Exception:  # noqa: BLE001
                    pass

    info = getattr(emb, "_avas_info", None)
    if isinstance(info, dict):
        for key in ("canonical_coeff", "canonical_energy", "ao_projector"):
            _put(key, info.get(key))
    return out


def dump_active_space(
    path: Path,
    *,
    emb,
    orbitals,
    result,
    cfg,
    ham=None,
    cycle: int = 0,
    geometry_file: str = "",
    geometry_parameter: float | None = None,
    downfolded_onebody: bool = False,
) -> Path:
    """Write one cycle's active-space orbitals and AO labels to an ``.npz`` dump.

    Opt-in diagnostic for active-space *character* comparisons across backends
    (``EQI_EMB_DUMP_ACTIVE_MOS`` / ``--dump_active_mos``).  It writes the embedded
    orbital coefficients and overlap in the working AO basis, the active/inactive
    column indices, the solver 1-RDM, and -- when the backend can report them --
    the per-AO ``(atom, element, shell, m)`` labels and the AO-to-atom map.  When the
    downfolded Hamiltonian is supplied it also stores the orbital energies and (for
    small active spaces) ``h1``/``h2``, so the one- and two-body drivers of the
    correlation can be compared offline.  Nothing here re-derives physics; a
    companion analysis script computes populations and bonding/antibonding character.

    ``downfolded_onebody`` (off by default) additionally stores the full operator
    ingredients -- ``h_emb``, ``veff_hf(dm_in)``, the low-level and high-level mean
    fields, the canonical (pre-AVAS) virtual block/energies, the AVAS target metric,
    ``v_emb`` and the initial A density -- so arbitrary virtual candidate pairs can be
    scored offline (see :func:`_downfolded_onebody_payload`).  The default dump is
    unchanged when this flag is off.

    Args:
        path: Destination ``.npz`` path.  Parent directories are created.
        emb: The :class:`ProjectionEmbeddingAdapter` the orbitals came from (used
            for the AO basis and overlap).
        orbitals: The cycle's :class:`EmbeddedOrbitals` (restricted path only).
        result: The :class:`SolverResult`; its ``rdm1`` is in the active-MO basis.
        cfg: The workflow settings (selector and AVAS target labels go into ``meta``).
        ham: The :class:`EmbeddedHamiltonian` solved this cycle, or ``None``.  Its
            ``h2`` is stored only when ``norb <= 16`` (a 64k-element tensor), so the
            dump never becomes the large object it was meant to avoid.
        cycle: Outer-loop cycle index.
        geometry_file: Source geometry, recorded in ``meta``.
        geometry_parameter: Bond-distance label, recorded in ``meta``.

    Returns:
        The path written.
    """
    ints = emb.ints
    coefficients = np.asarray(orbitals.coeff, dtype=float)
    overlap = np.asarray(ints.overlap(), dtype=float)
    active = np.asarray(orbitals.active, dtype=int)
    inactive = np.asarray(orbitals.inactive, dtype=int)
    atom_map = np.asarray(ints.ao_atom_map(), dtype=int)
    payload: dict[str, Any] = {
        "coeff": coefficients,
        "coeff_active": coefficients[:, active],
        "overlap": overlap,
        "active": active,
        "inactive": inactive,
        "n_occ": np.asarray(int(orbitals.n_occ)),
        "n_occ_b": np.asarray(-1 if orbitals.n_occ_b is None else int(orbitals.n_occ_b)),
        "ao_atom_map": atom_map,
        "rdm1_active": (
            np.empty((0, 0))
            if result is None or result.rdm1 is None
            else np.asarray(result.rdm1, dtype=float)
        ),
        "cycle": np.asarray(int(cycle)),
    }
    # Orbital energies and the downfolded Hamiltonian: the one-electron spectrum sets
    # the excitation denominators, while h2 carries the couplings.  Without these an
    # orbital-character comparison cannot say why the same character correlates in one
    # backend and not the other.
    energy = getattr(orbitals, "energy", None)
    if energy is not None:
        energy = np.asarray(energy, dtype=float)
        payload["orbital_energy"] = energy
        payload["active_energy"] = energy[active]
    if ham is not None:
        payload["ham_nelec"] = np.asarray([int(x) for x in ham.nelec], dtype=int)
        payload["ham_e_core"] = np.asarray(float(ham.e_core))
        payload["h1"] = np.asarray(ham.h1, dtype=float)
        if int(ham.norb) <= 16:
            payload["h2"] = np.asarray(ham.h2, dtype=float)

    # Labels are optional and backend-specific.  A length mismatch means the backend
    # cannot be trusted to order them like the AO matrices, so drop them rather than
    # silently mislabel the analysis.
    try:
        labels = ints.ao_labels()
    except Exception:  # noqa: BLE001 - labels are diagnostic, never required
        labels = None
    if labels is not None and len(labels) == overlap.shape[0]:
        payload["label_atom"] = np.asarray([int(lab[0]) for lab in labels], dtype=int)
        payload["label_element"] = np.asarray([str(lab[1]) for lab in labels])
        payload["label_shell"] = np.asarray([str(lab[2]) for lab in labels])
        payload["label_m"] = np.asarray([str(lab[3]) for lab in labels])

    try:
        describe = ints.describe()
    except Exception:  # noqa: BLE001 - optional metadata
        describe = {}
    symbols = describe.get("atom_symbols")
    coords = describe.get("atom_coords")
    if symbols is not None:
        payload["atom_symbols"] = np.asarray([str(s) for s in symbols])
    if coords is not None:
        payload["atom_coords"] = np.asarray(coords, dtype=float)

    info = getattr(emb, "_avas_info", None)
    if isinstance(info, dict) and "channels" not in info:
        for key in (
            "occ_weights",
            "virt_weights",
            "virt_selection_weights",
            "virt_candidate_energy",
        ):
            value = info.get(key)
            if value is not None:
                payload[key] = np.asarray(value, dtype=float)
        for key in ("virt_candidate_column", "virt_selection_column"):
            value = info.get(key)
            if value is not None:
                payload[key] = np.asarray(value, dtype=int)
    if downfolded_onebody:
        payload.update(_downfolded_onebody_payload(emb, orbitals, coefficients))
        payload["downfolded_onebody"] = np.asarray(True)
    meta = {
        "selector": str(getattr(cfg, "selector", "")),
        "avas_ao_labels": getattr(cfg, "avas_ao_labels", None),
        "avas_max_size": getattr(cfg, "avas_max_size", None),
        # Opt-in correlation-window virtual ranking (Ha); None means standard AVAS.
        "avas_energy_window": getattr(cfg, "avas_energy_window", None),
        "energy_window_ceiling": (
            None if not isinstance(info, dict) else info.get("window_ceiling")
        ),
        "energy_window_n_candidates": (
            None if not isinstance(info, dict) else info.get("window_n_candidates")
        ),
        "energy_window_shortfall": (
            None if not isinstance(info, dict) else info.get("window_shortfall")
        ),
        "n_frozen_occ": getattr(cfg, "n_frozen_occ", None),
        "projector": None if not isinstance(info, dict) else info.get("projector"),
        "geometry_file": geometry_file,
        "geometry_parameter": geometry_parameter,
        "qm_code": getattr(cfg, "qm_code", None),
        "basis": describe.get("basis"),
        # The workflow reorders the active atoms to the front, so the fragment AO
        # indices are 0..len(active_atoms)-1 in this dump's atom map.
        "active_atoms": getattr(cfg, "active_atoms", None),
    }
    payload["meta"] = np.asarray(json.dumps(meta))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return path


def natural_occupations(rdm1: np.ndarray) -> list[float] | None:
    """Compute natural-orbital occupations from the 1-RDM.

    Eigenvalues of the spin-summed spatial-orbital 1-RDM (SolverResult.rdm1) are the
    natural-orbital occupations by definition. However, numpy.linalg.eigvalsh only reads
    the lower (or upper) triangle of its input, so calling it on a not-actually-Hermitian
    array would silently produce a wrong but well-formed result. This function therefore
    verifies both shape and Hermiticity before computing eigenvalues.

    Args:
        rdm1: Spin-summed one-particle reduced density matrix, shape (norb, norb).

    Returns:
        List of natural-orbital occupations sorted in descending order, or None if the
        input is not a square Hermitian matrix.
    """
    if rdm1 is None:
        return None
    if not isinstance(rdm1, np.ndarray):
        return None
    if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
        return None
    if not np.allclose(rdm1, rdm1.conj().T, atol=1e-6):
        return None

    try:
        eigenvalues = np.linalg.eigvalsh(rdm1)
        return sorted(eigenvalues.tolist(), reverse=True)
    except Exception:
        return None


def build_diagnostics_row(
    cfg,
    *,
    run_id: str,
    geometry_file: str,
    geometry_parameter: float | None,
    cycle: int,
    converged: bool,
    total_cycles: int | None,
    orbitals,
    ham,
    result,
    energy,
    solver,
) -> dict[str, Any]:
    """Assemble one diagnostics row from current cycle data.

    All values are read directly from the passed objects; no re-derivation of physics occurs.
    Array-like values are serialized via json_cell(). Missing optional diagnostics keys
    are left as empty cells via csv.DictWriter's restval="".

    Args:
        cfg: EmbeddingWorkflow instance (for solver, sampler, selector, etc.).
        run_id: UUID-based run identifier string.
        geometry_file: Geometry source (e.g., "data/12.inp" or "s26[22]").
        geometry_parameter: Bond distance in Angstrom if available (e.g., 1.2), else None.
        cycle: Current outer-loop cycle index (0-based).
        converged: Whether the outer loop has converged at this cycle.
        total_cycles: Total number of cycles, or None if not yet final.
        orbitals: EmbeddedOrbitals instance (active, inactive, n_occ, etc.).
        ham: EmbeddedHamiltonian instance (h1, h2, nelec, e_core, norb, meta).
        result: SolverResult instance (energy, rdm1, diagnostics).
        energy: ProjectionEnergy instance (e_low_total, e_low_A, e_high_A, etc.).
        solver: ActiveSpaceSolver instance (FCISolver or SQDSolver).

    Returns:
        Dictionary ready to pass to csv.DictWriter.writerow().
    """

    # Algorithm and backend name
    if cfg.solver == "fci":
        algorithm = "FCI"
        backend_name = ""
    elif cfg.sampler == "aer":
        algorithm = "qiskit_aer"
        backend_name = ""
    elif cfg.sampler == "mock":
        algorithm = "mock"
        backend_name = ""
    else:  # "runtime"
        from embasi_qiskit_integration.circuit_run.backend import _is_fake_backend

        rt_sampler = solver.sampler
        backend_obj = getattr(rt_sampler, "_backend_obj", None)
        algorithm = (
            "fake_backend"
            if (backend_obj is not None and _is_fake_backend(backend_obj))
            else "ibm_backend"
        )
        backend_name = rt_sampler.resolved_backend_name or ""

    # Active-space geometry
    cas_norb = orbitals.n_active_orbitals
    cas_nelec = orbitals.n_active_electrons
    active_space_indices = orbitals.active
    frozen_occupied = orbitals.inactive
    selector = cfg.selector
    norb = ham.norb
    nelec_alpha, nelec_beta = ham.nelec

    # Selected virtuals: active indices >= n_occ (occupied + virtual boundary)
    # Note: this assumes active is a NumPy array of column indices; in EmbeddedOrbitals
    # they are indeed stored that way and represent MO indices into the orbital space.
    selected_virtuals_mask = active_space_indices >= orbitals.n_occ
    selected_virtuals = active_space_indices[selected_virtuals_mask]

    # Energy fields (all already computed every cycle for the console log)
    e_solver = result.energy
    e_low_total = energy.e_low_total
    e_low_a = energy.e_low_A
    e_high_a = energy.e_high_A
    e_core = ham.e_core
    correction = energy.correction
    footing_shift = energy.footing_shift
    projector_leak = energy.projector_leak
    p_b_leak = ham.meta.get("p_b_leak")
    embedding_energy = energy.total

    # Density and convergence
    rdm1_matrix = result.rdm1
    rdm1_trace = float(np.trace(result.rdm1)) if result.rdm1 is not None else None
    natorb = natural_occupations(result.rdm1)

    # SQD-specific diagnostics (empty for FCI, read from result.diagnostics for SQD)
    number_unique_bitstrings = result.diagnostics.get("n_distinct_bitstrings")
    final_sqd_subspace_dimension = result.diagnostics.get("final_sqd_subspace_dimension")

    # Assemble the row; unused columns will be filled with restval="" by DictWriter
    return {
        "run_id": run_id,
        "geometry_file": geometry_file,
        "geometry_parameter": geometry_parameter,
        "algorithm": algorithm,
        "backend_name": backend_name,
        "cycle": cycle,
        "converged": converged,
        "total_cycles": total_cycles,
        "cas_norb": cas_norb,
        "cas_nelec": cas_nelec,
        "active_space_indices": json_cell(active_space_indices),
        "selector": selector,
        "selected_virtuals": json_cell(selected_virtuals),
        "frozen_occupied": json_cell(frozen_occupied),
        "norb": norb,
        "nelec_alpha": nelec_alpha,
        "nelec_beta": nelec_beta,
        "E_solver": e_solver,
        "E_low_total": e_low_total,
        "E_low_A": e_low_a,
        "E_high_A": e_high_a,
        "e_core": e_core,
        "correction": correction,
        "footing_shift": footing_shift,
        "projector_leak": projector_leak,
        "p_b_leak": p_b_leak,
        "delta_E": None,  # Set by caller after convergence check
        "embedding_energy": embedding_energy,
        "number_unique_bitstrings": number_unique_bitstrings,
        "final_sqd_subspace_dimension": final_sqd_subspace_dimension,
        "rdm1_trace": rdm1_trace,
        "rdm1": json_cell(rdm1_matrix),
        "max_delta_gamma_A": None,  # Set by caller
        "natural_occupations": json_cell(natorb),
        # FCI-vs-SQD comparison columns always empty (no dual-solver mechanism)
    }
