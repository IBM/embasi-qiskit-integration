# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Component-wise diagnosis of a PySCF-vs-SIESTA embedding scan.

The embedding total is assembled from four terms (paper Eq. 8)::

    E_embed = E_low_total - E_low_A + E_high_A + correction
            = E_low_total + (E_high_A - E_low_A) + correction

This script reads one ``summary.csv`` per code (written by
:mod:`run_structural_scan`) and reports every term per distance, then compares
the two codes component by component.

**Energy zeros are not comparable.**  PySCF here is all-electron/Gaussian and
SIESTA is pseudopotential/NAO, so ``E_low_total``, ``E_low_A`` and ``E_high_A``
carry code-dependent core/pseudopotential offsets of tens to hundreds of
hartree.  Only differences between geometries (relative energies), mean-centred
cross-code differences, and well depths are physically comparable.  The script
prints the raw offset for every component so the offset is never mistaken for
physics.

Two modes:

* component diagnosis (``--pyscf-dir`` and ``--siesta-dir``): full per-distance
  breakdown, cross-code centred RMSE / Pearson r, well depths, and a
  covariance decomposition of the relative-energy gap;
* active-space scan (``--pyscf-cas``/``--siesta-cas`` repeated): the same
  comparison for several CAS sizes, so ``CAS(10,10)`` can be checked against
  ``CAS(4,4)`` for narrowing or widening the cross-code gap.

Example::

    .venv/bin/python scripts/benchmarks/diagnose_embedding_components.py \
        --pyscf-dir /tmp/embasi-scan-pyscf-ao-cas4 \
        --siesta-dir /tmp/embasi-scan-siesta-ao-cas4 \
        --out-dir analysis/butyronitrile-pbe-hf-fci/component-diagnosis/final-ao-cas4

    .venv/bin/python scripts/benchmarks/diagnose_embedding_components.py \
        --pyscf-cas 4=/tmp/cas-check-pyscf/cas4 \
        --pyscf-cas 6=/tmp/cas-check-pyscf/cas6 \
        --pyscf-cas 10=/tmp/cas-check-pyscf/cas10 \
        --siesta-cas 4=/tmp/cas-check-siesta/cas4 \
        --siesta-cas 6=/tmp/cas-check-siesta/cas6 \
        --siesta-cas 10=/tmp/cas-check-siesta/cas10 \
        --out-dir analysis/butyronitrile-pbe-hf-fci/component-diagnosis/cas-4-6-10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HARTREE_TO_KCAL = 627.5094740631

# Reference palette, validated for up to four series on the light surface.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

#: Components that make up ``E_embed`` and are compared one by one.
COMPONENTS = (
    "E_low_total",
    "E_low_A",
    "E_high_A",
    "E_high_A_minus_E_low_A",
    "correction",
    "embedding_energy",
)

#: The three terms whose relative energies sum exactly to the relative total.
DECOMPOSITION_TERMS = ("E_low_total", "E_high_A_minus_E_low_A", "correction")

DISPLAY = {
    "E_low_total": "E_low(total)",
    "E_low_A": "E_low(A)",
    "E_high_A": "E_high(A)",
    "E_high_A_minus_E_low_A": "E_high(A) - E_low(A)",
    "correction": "correction (tr[(g~-g) v_emb])",
    "embedding_energy": "E_embed",
}

CODE_DISPLAY = {"pyscf": "PySCF", "siesta": "SIESTA"}


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_summary(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"missing {summary_path}; run run_structural_scan.py first")
    rows: list[dict[str, Any]] = []
    with summary_path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("status") not in {"ok", "cached"}:
                continue
            distance = _num(raw.get("distance"))
            if distance is None:
                continue
            row: dict[str, Any] = dict(raw)
            row["distance"] = round(float(distance), 6)
            for key in (
                "E_solver",
                "E_low_total",
                "E_low_A",
                "E_high_A",
                "e_core",
                "correction",
                "embedding_energy",
                "embedding_correction",
                "high_level_term",
                "projector_leak",
                "cycles",
            ):
                row[key] = _num(raw.get(key))
            missing = [
                key
                for key in ("E_low_total", "E_low_A", "E_high_A", "correction", "embedding_energy")
                if row[key] is None
            ]
            if missing:
                raise SystemExit(f"{summary_path}: row {row.get('stem')} lacks {missing}")
            row["E_high_A_minus_E_low_A"] = row["E_high_A"] - row["E_low_A"]
            stored = row.get("high_level_term")
            if stored is not None and abs(stored - row["E_high_A_minus_E_low_A"]) > 1e-6:
                raise SystemExit(
                    f"{summary_path}: high_level_term ({stored}) does not equal "
                    f"E_high_A - E_low_A ({row['E_high_A_minus_E_low_A']})"
                )
            row["converged"] = _as_bool(raw.get("converged"))
            rows.append(row)
    if not rows:
        raise SystemExit(f"{summary_path} has no successful, complete rows")
    rows.sort(key=lambda item: item["distance"])
    metadata: dict[str, Any] = {}
    meta_path = run_dir / "summary.json"
    if meta_path.is_file():
        payload = json.loads(meta_path.read_text())
        metadata = payload.get("metadata", payload)
    return rows, metadata


def _fractional_electrons(occupations: list[float]) -> float:
    """Electrons in fractionally occupied active orbitals (5-95% occupancy).

    A closed-shell single determinant gives 0; a fully broken pair gives 2.
    This is a compact static-correlation indicator for the high-level solve.
    """
    return float(sum(value for value in occupations if 0.05 < value < 1.95))


def _read_occupations(
    run_dir: Path, rows: list[dict[str, Any]]
) -> dict[float, tuple[list[float], float]]:
    """Last-cycle natural occupations per distance, when diagnostics exist."""
    out: dict[float, tuple[list[float], float]] = {}
    for row in rows:
        path = run_dir / str(row.get("stem", "")) / "diagnostics.csv"
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            diagnostics = list(csv.DictReader(handle))
        if not diagnostics:
            continue
        raw = diagnostics[-1].get("natural_occupations")
        if not raw:
            continue
        try:
            occupations = [float(value) for value in json.loads(raw)]
        except (TypeError, ValueError):
            continue
        out[row["distance"]] = (occupations, _fractional_electrons(occupations))
    return out


def _label(metadata: dict[str, Any], fallback: str) -> str:
    backend = str(metadata.get("backend", fallback)).lower()
    display = CODE_DISPLAY.get(backend, backend)
    xc_ll = metadata.get("xc_ll", "PBE")
    xc_hl = metadata.get("xc_hl", "HF")
    basis = metadata.get("basis", "?")
    return f"{display} ({xc_ll}/{xc_hl}, {basis})"


def _reference_row(rows: list[dict[str, Any]], spec: str) -> dict[str, Any]:
    if spec == "min":
        return min(rows, key=lambda row: row["distance"])
    if spec == "max":
        return max(rows, key=lambda row: row["distance"])
    target = float(spec)
    return min(rows, key=lambda row: abs(row["distance"] - target))


def _merge(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    left_by = {row["distance"]: row for row in left}
    right_by = {row["distance"]: row for row in right}
    common = sorted(set(left_by) & set(right_by))
    if not common:
        raise SystemExit("the two summaries have no common geometry distances")
    return [(left_by[d], right_by[d]) for d in common]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _rmse(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(array)))) if array.size else float("nan")


def _max_abs(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.max(np.abs(array))) if array.size else float("nan")


def _pearson(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.size < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _well_depth(rows: list[dict[str, Any]], key: str, reference: str) -> float:
    """``E(reference) - min(E)`` over the scan, in hartree."""
    ref = _reference_row(rows, reference)
    return float(ref[key] - min(row[key] for row in rows))


# --------------------------------------------------------------------------- #
# Component diagnosis
# --------------------------------------------------------------------------- #
def _component_records(
    code: str,
    rows: list[dict[str, Any]],
    reference: str,
) -> list[dict[str, Any]]:
    ref = _reference_row(rows, reference)
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            "code": code,
            "distance": row["distance"],
            "E_low_total": row["E_low_total"],
            "E_low_A": row["E_low_A"],
            "E_high_A": row["E_high_A"],
            "E_high_A_minus_E_low_A": row["E_high_A_minus_E_low_A"],
            "correction": row["correction"],
            "embedding_energy": row["embedding_energy"],
        }
        for key in COMPONENTS:
            record[f"rel_{key}"] = row[key] - ref[key]
        records.append(record)
    return records


def _component_metrics(
    pyscf: list[dict[str, Any]],
    siesta: list[dict[str, Any]],
    reference: str,
) -> dict[str, Any]:
    pairs = _merge(pyscf, siesta)
    metrics: dict[str, Any] = {"components": {}}
    for key in COMPONENTS:
        p_raw = [p[key] for p, _ in pairs]
        s_raw = [s[key] for _, s in pairs]
        delta = np.asarray(s_raw) - np.asarray(p_raw)
        centered = delta - np.mean(delta)

        p_ref = _reference_row(pyscf, reference)[key]
        s_ref = _reference_row(siesta, reference)[key]
        p_rel = [p[key] - p_ref for p, _ in pairs]
        s_rel = [s[key] - s_ref for _, s in pairs]
        delta_rel = np.asarray(s_rel) - np.asarray(p_rel)
        centered_rel = delta_rel - np.mean(delta_rel)

        depth_p = _well_depth(pyscf, key, reference)
        depth_s = _well_depth(siesta, key, reference)
        metrics["components"][key] = {
            "display": DISPLAY[key],
            "mean_delta_ha": float(np.mean(delta)),
            "mean_delta_kcal": float(np.mean(delta) * HARTREE_TO_KCAL),
            "rmse_delta_ha": _rmse(delta),
            "rmse_delta_kcal": _rmse(delta) * HARTREE_TO_KCAL,
            "rmse_delta_centered_ha": _rmse(centered),
            "rmse_delta_centered_kcal": _rmse(centered) * HARTREE_TO_KCAL,
            "max_abs_delta_centered_ha": _max_abs(centered),
            "max_abs_delta_centered_kcal": _max_abs(centered) * HARTREE_TO_KCAL,
            "pearson_r_raw": _pearson(p_raw, s_raw),
            "pearson_r_relative": _pearson(p_rel, s_rel),
            "mean_delta_rel_ha": float(np.mean(delta_rel)),
            "rmse_delta_rel_ha": _rmse(delta_rel),
            "rmse_delta_rel_centered_ha": _rmse(centered_rel),
            "rmse_delta_rel_centered_kcal": _rmse(centered_rel) * HARTREE_TO_KCAL,
            "max_abs_delta_rel_ha": _max_abs(delta_rel),
            "max_abs_delta_rel_centered_ha": _max_abs(centered_rel),
            "well_depth_pyscf_ha": depth_p,
            "well_depth_siesta_ha": depth_s,
            "well_depth_pyscf_kcal": depth_p * HARTREE_TO_KCAL,
            "well_depth_siesta_kcal": depth_s * HARTREE_TO_KCAL,
            "well_depth_gap_kcal": (depth_s - depth_p) * HARTREE_TO_KCAL,
        }
    return metrics


def _decomposition(
    pyscf: list[dict[str, Any]],
    siesta: list[dict[str, Any]],
    reference: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split the cross-code relative-energy gap into its three additive terms.

    ``rel(E_embed) = rel(E_low_total) + rel(E_high(A)-E_low(A)) + rel(correction)``
    for every distance, so the same identity holds for the cross-code gap.
    """
    pairs = _merge(pyscf, siesta)
    p_ref = _reference_row(pyscf, reference)
    s_ref = _reference_row(siesta, reference)
    rows: list[dict[str, Any]] = []
    totals: list[float] = []
    contribs: dict[str, list[float]] = {key: [] for key in DECOMPOSITION_TERMS}
    for p, s in pairs:
        total = (s["embedding_energy"] - s_ref["embedding_energy"]) - (
            p["embedding_energy"] - p_ref["embedding_energy"]
        )
        row = {"distance": p["distance"], "delta_rel_embedding_ha": total}
        for key in DECOMPOSITION_TERMS:
            contribution = (s[key] - s_ref[key]) - (p[key] - p_ref[key])
            row[f"delta_rel_{key}_ha"] = contribution
            contribs[key].append(contribution)
        row["delta_rel_reconstructed_ha"] = sum(
            row[f"delta_rel_{key}_ha"] for key in DECOMPOSITION_TERMS
        )
        rows.append(row)
        totals.append(total)

    total_array = np.asarray(totals)
    variance = float(np.var(total_array))
    summary: dict[str, Any] = {
        "max_abs_identity_error_ha": _max_abs(
            np.asarray([row["delta_rel_reconstructed_ha"] for row in rows]) - total_array
        ),
        "rmse_delta_rel_centered_kcal": _rmse(total_array - np.mean(total_array)) * HARTREE_TO_KCAL,
        "terms": {},
    }
    for key, values in contribs.items():
        array = np.asarray(values)
        share = (
            float(np.cov(array, total_array, ddof=0)[0, 1] / variance)
            if variance > 0
            else float("nan")
        )
        summary["terms"][key] = {
            "rmse_centered_ha": _rmse(array - np.mean(array)),
            "rmse_centered_kcal": _rmse(array - np.mean(array)) * HARTREE_TO_KCAL,
            "max_abs_centered_kcal": _max_abs(array - np.mean(array)) * HARTREE_TO_KCAL,
            "mean_ha": float(np.mean(array)),
            "covariance_share": share,
        }
    ranked = sorted(
        summary["terms"],
        key=lambda key: abs(summary["terms"][key]["rmse_centered_kcal"]),
        reverse=True,
    )
    summary["dominant_component"] = ranked[0]
    summary["ranking"] = ranked
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"refusing to write empty CSV {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _write_component_markdown(
    path: Path,
    *,
    label_pyscf: str,
    label_siesta: str,
    reference: str,
    reference_distance: float,
    records_p: list[dict[str, Any]],
    records_s: list[dict[str, Any]],
    metrics: dict[str, Any],
    decomposition_rows: list[dict[str, Any]],
    decomposition: dict[str, Any],
) -> None:
    components = metrics["components"]
    lines = [
        "# Component-wise diagnosis: SIESTA vs PySCF",
        "",
        f"- PySCF: {label_pyscf}",
        f"- SIESTA: {label_siesta}",
        f"- Reference separation: {reference_distance:.2f} A (`--reference {reference}`)",
        f"- Points: {len(records_p)}",
        "",
        "**Raw offsets are not physics.** E_low(total), E_low(A) and E_high(A) sit on "
        "different core/pseudopotential energy zeros in the two codes, so their raw "
        "cross-code differences are tens to hundreds of hartree. Only relative "
        "energies, mean-centred cross-code differences, and well depths are "
        "comparable; every raw offset below is reported only so it can be excluded.",
        "",
        "## Per-distance components (hartree)",
        "",
        "| R (A) | code | E_low(total) | E_low(A) | E_high(A) | E_high(A)-E_low(A) | correction | E_embed |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for code, records in (("pyscf", records_p), ("siesta", records_s)):
        for record in records:
            lines.append(
                f"| {record['distance']:.2f} | {CODE_DISPLAY[code]} | "
                f"{record['E_low_total']:.8f} | {record['E_low_A']:.8f} | "
                f"{record['E_high_A']:.8f} | {record['E_high_A_minus_E_low_A']:.8f} | "
                f"{record['correction']:.8f} | {record['embedding_energy']:.8f} |"
            )

    lines += [
        "",
        "## Cross-code comparison per component",
        "",
        "`offset` is the raw mean difference (SIESTA - PySCF); it is the "
        "core/pseudopotential zero and is **not** an error. `centred RMSE` removes "
        "that constant and measures the shape mismatch. Pearson `r` is between the "
        "two codes' series across distances; it is invariant to the constant shift, "
        "so the raw and relative series give the same value.",
        "",
        "| component | offset (Ha) | offset (kcal/mol) | raw RMSE (kcal/mol) | "
        "centred RMSE (kcal/mol) | max |centred| (kcal/mol) | r |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in COMPONENTS:
        item = components[key]
        lines.append(
            f"| {DISPLAY[key]} | {_fmt(item['mean_delta_ha'])} | "
            f"{_fmt(item['mean_delta_kcal'], 1)} | {_fmt(item['rmse_delta_kcal'], 1)} | "
            f"{_fmt(item['rmse_delta_centered_kcal'], 1)} | "
            f"{_fmt(item['max_abs_delta_centered_kcal'], 1)} | "
            f"{_fmt(item['pearson_r_raw'])} |"
        )

    lines += [
        "",
        "## Relative-energy gap decomposition",
        "",
        "For every distance the cross-code relative-energy gap splits additively:",
        "",
        "`delta_rel(E_embed) = delta_rel(E_low_total) + delta_rel(E_high(A)-E_low(A)) + delta_rel(correction)`",
        "",
        "The covariance share is the fraction of the gap variance carried by each "
        "term; because the terms are correlated the individual shares need not lie "
        "in [0, 1], but they sum to 1 when the terms are additive.",
        "",
        "| term | centred RMSE (kcal/mol) | max |centred| (kcal/mol) | covariance share |",
        "|:---|---:|---:|---:|",
    ]
    for key in decomposition["ranking"]:
        item = decomposition["terms"][key]
        lines.append(
            f"| {DISPLAY[key]} | {_fmt(item['rmse_centered_kcal'], 1)} | "
            f"{_fmt(item['max_abs_centered_kcal'], 1)} | {_fmt(item['covariance_share'], 3)} |"
        )

    dominant = components[decomposition["dominant_component"]]
    total = components["embedding_energy"]
    lines += [
        "",
        f"**Dominant component: {DISPLAY[decomposition['dominant_component']]}.** "
        f"Gap identity residual: {decomposition['max_abs_identity_error_ha']:.2e} Ha.",
        "",
        f"Interpretation: the raw cross-code offsets "
        f"({components['E_low_total']['mean_delta_ha']:.3f} Ha for E_low(total), "
        f"{components['E_low_A']['mean_delta_ha']:.3f} Ha for E_low(A), "
        f"{components['E_high_A']['mean_delta_ha']:.3f} Ha for E_high(A), "
        f"{components['E_high_A_minus_E_low_A']['mean_delta_ha']:.3f} Ha for "
        f"E_high(A)-E_low(A), {total['mean_delta_ha']:.3f} Ha for E_embed) are "
        f"core/pseudopotential zeros and are excluded from the comparison. The "
        f"comparable centred shape mismatch of E_embed is "
        f"{total['rmse_delta_centered_kcal']:.1f} kcal/mol, and it is dominated by "
        f"{DISPLAY[decomposition['dominant_component']]} "
        f"(centred RMSE {dominant['rmse_delta_centered_kcal']:.1f} kcal/mol; "
        f"covariance share {decomposition['terms'][decomposition['dominant_component']]['covariance_share']:.2f}). "
        f"The low-level PBE shape agrees well "
        f"(E_low(total) centred RMSE "
        f"{components['E_low_total']['rmse_delta_centered_kcal']:.1f} kcal/mol, "
        f"r = {components['E_low_total']['pearson_r_raw']:.3f}).",
        "",
        "### Per-distance gap terms (kcal/mol)",
        "",
        "| R (A) | delta E_embed | E_low(total) | E_high(A)-E_low(A) | correction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in decomposition_rows:
        lines.append(
            f"| {row['distance']:.2f} | {row['delta_rel_embedding_ha'] * HARTREE_TO_KCAL:.1f} | "
            f"{row['delta_rel_E_low_total_ha'] * HARTREE_TO_KCAL:.1f} | "
            f"{row['delta_rel_E_high_A_minus_E_low_A_ha'] * HARTREE_TO_KCAL:.1f} | "
            f"{row['delta_rel_correction_ha'] * HARTREE_TO_KCAL:.1f} |"
        )

    lines += [
        "",
        "## Well depths",
        "",
        f"Well depth is `E(reference) - min(E)` over the scan, with reference "
        f"R = {reference_distance:.2f} A.",
        "",
        "| quantity | PySCF (kcal/mol) | SIESTA (kcal/mol) | gap (kcal/mol) |",
        "|:---|---:|---:|---:|",
    ]
    for key in ("embedding_energy", "E_low_total", "E_high_A_minus_E_low_A", "correction"):
        item = components[key]
        lines.append(
            f"| {DISPLAY[key]} | {_fmt(item['well_depth_pyscf_kcal'], 1)} | "
            f"{_fmt(item['well_depth_siesta_kcal'], 1)} | {_fmt(item['well_depth_gap_kcal'], 1)} |"
        )
    lines += [
        "",
        "Component well-depth gaps are **not additive**: each quantity reaches its "
        "minimum at a different separation, so use the per-distance decomposition "
        "above for the additive attribution.",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Active-space scan
# --------------------------------------------------------------------------- #
def _parse_cas_runs(values: list[str] | None) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"expected CAS=DIR, got {value!r}")
        cas, directory = value.split("=", 1)
        runs[int(cas)] = Path(directory).expanduser().resolve()
    return runs


def _cas_comparison(
    pyscf_runs: dict[int, Path],
    siesta_runs: dict[int, Path],
    reference: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if set(pyscf_runs) != set(siesta_runs):
        raise SystemExit(
            f"CAS sets differ: PySCF {sorted(pyscf_runs)} vs SIESTA {sorted(siesta_runs)}"
        )
    common_cas = sorted(pyscf_runs)
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"cas": {}}
    labels: dict[str, Any] = {}
    for cas in common_cas:
        p_rows, p_meta = _read_summary(pyscf_runs[cas])
        s_rows, s_meta = _read_summary(siesta_runs[cas])
        if cas == common_cas[0]:
            labels = {
                "pyscf": _label(p_meta, "pyscf"),
                "siesta": _label(s_meta, "siesta"),
                "pyscf_metadata": p_meta,
                "siesta_metadata": s_meta,
            }
        pairs = _merge(p_rows, s_rows)
        p_ref = _reference_row(p_rows, reference)
        s_ref = _reference_row(s_rows, reference)
        p_occ = _read_occupations(pyscf_runs[cas], p_rows)
        s_occ = _read_occupations(siesta_runs[cas], s_rows)
        rel_p: list[float] = []
        rel_s: list[float] = []
        delta_rel: list[float] = []
        hlt_delta: list[float] = []
        corr_delta: list[float] = []
        for p, s in pairs:
            rp = p["embedding_energy"] - p_ref["embedding_energy"]
            rs = s["embedding_energy"] - s_ref["embedding_energy"]
            dh = (s["E_high_A_minus_E_low_A"] - s_ref["E_high_A_minus_E_low_A"]) - (
                p["E_high_A_minus_E_low_A"] - p_ref["E_high_A_minus_E_low_A"]
            )
            dc = (s["correction"] - s_ref["correction"]) - (p["correction"] - p_ref["correction"])
            p_frac = p_occ.get(p["distance"], (None, None))[1]
            s_frac = s_occ.get(s["distance"], (None, None))[1]
            rows.append(
                {
                    "cas": cas,
                    "distance": p["distance"],
                    "backend": "pyscf",
                    "cas_norb": p.get("cas_norb"),
                    "cas_nelec": p.get("cas_nelec"),
                    "E_low_total": p["E_low_total"],
                    "E_low_A": p["E_low_A"],
                    "E_high_A": p["E_high_A"],
                    "E_high_A_minus_E_low_A": p["E_high_A_minus_E_low_A"],
                    "correction": p["correction"],
                    "embedding_energy": p["embedding_energy"],
                    "rel_embedding_energy": rp,
                    "fractional_electrons": p_frac,
                }
            )
            rows.append(
                {
                    "cas": cas,
                    "distance": s["distance"],
                    "backend": "siesta",
                    "cas_norb": s.get("cas_norb"),
                    "cas_nelec": s.get("cas_nelec"),
                    "E_low_total": s["E_low_total"],
                    "E_low_A": s["E_low_A"],
                    "E_high_A": s["E_high_A"],
                    "E_high_A_minus_E_low_A": s["E_high_A_minus_E_low_A"],
                    "correction": s["correction"],
                    "embedding_energy": s["embedding_energy"],
                    "rel_embedding_energy": rs,
                    "fractional_electrons": s_frac,
                }
            )
            rel_p.append(rp)
            rel_s.append(rs)
            delta_rel.append(rs - rp)
            hlt_delta.append(dh)
            corr_delta.append(dc)
        delta_array = np.asarray(delta_rel)
        hlt_array = np.asarray(hlt_delta)
        corr_array = np.asarray(corr_delta)
        depth_p = _well_depth(p_rows, "embedding_energy", reference)
        depth_s = _well_depth(s_rows, "embedding_energy", reference)
        hlt_depth_p = _well_depth(p_rows, "E_high_A_minus_E_low_A", reference)
        hlt_depth_s = _well_depth(s_rows, "E_high_A_minus_E_low_A", reference)
        p_frac_values = [value for _, value in p_occ.values()]
        s_frac_values = [value for _, value in s_occ.values()]
        metrics["cas"][cas] = {
            "n_points": len(pairs),
            "reference_distance": p_ref["distance"],
            "cas_norb_pyscf": p_rows[0].get("cas_norb"),
            "cas_nelec_pyscf": p_rows[0].get("cas_nelec"),
            "cas_norb_siesta": s_rows[0].get("cas_norb"),
            "cas_nelec_siesta": s_rows[0].get("cas_nelec"),
            "max_fractional_electrons_pyscf": (max(p_frac_values) if p_frac_values else None),
            "max_fractional_electrons_siesta": (max(s_frac_values) if s_frac_values else None),
            "rmse_delta_rel_kcal": _rmse(delta_array) * HARTREE_TO_KCAL,
            "rmse_delta_rel_centered_kcal": _rmse(delta_array - np.mean(delta_array))
            * HARTREE_TO_KCAL,
            "max_abs_delta_rel_kcal": _max_abs(delta_array) * HARTREE_TO_KCAL,
            "pearson_r_rel": _pearson(rel_p, rel_s),
            "rmse_delta_hlt_centered_kcal": _rmse(hlt_array - np.mean(hlt_array)) * HARTREE_TO_KCAL,
            "max_abs_delta_hlt_centered_kcal": _max_abs(hlt_array - np.mean(hlt_array))
            * HARTREE_TO_KCAL,
            "mean_delta_hlt_ha": float(np.mean(hlt_array)),
            "rmse_delta_correction_centered_kcal": _rmse(corr_array - np.mean(corr_array))
            * HARTREE_TO_KCAL,
            "well_depth_pyscf_ha": depth_p,
            "well_depth_siesta_ha": depth_s,
            "well_depth_pyscf_kcal": depth_p * HARTREE_TO_KCAL,
            "well_depth_siesta_kcal": depth_s * HARTREE_TO_KCAL,
            "well_depth_gap_kcal": (depth_s - depth_p) * HARTREE_TO_KCAL,
            "hlt_well_depth_pyscf_kcal": hlt_depth_p * HARTREE_TO_KCAL,
            "hlt_well_depth_siesta_kcal": hlt_depth_s * HARTREE_TO_KCAL,
            "hlt_well_depth_gap_kcal": (hlt_depth_s - hlt_depth_p) * HARTREE_TO_KCAL,
        }
    smallest = common_cas[0]
    largest = common_cas[-1]
    metrics["comparison"] = {
        "smallest_cas": smallest,
        "largest_cas": largest,
        "rmse_delta_rel_centered_change_kcal": (
            metrics["cas"][largest]["rmse_delta_rel_centered_kcal"]
            - metrics["cas"][smallest]["rmse_delta_rel_centered_kcal"]
        ),
        "well_depth_gap_change_kcal": (
            metrics["cas"][largest]["well_depth_gap_kcal"]
            - metrics["cas"][smallest]["well_depth_gap_kcal"]
        ),
    }
    return rows, metrics, labels


def _write_cas_markdown(
    path: Path,
    *,
    labels: dict[str, Any],
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    cas_values = sorted(metrics["cas"])
    smallest = metrics["comparison"]["smallest_cas"]
    largest = metrics["comparison"]["largest_cas"]
    pyscf_labels = labels.get("pyscf_metadata", {}).get("avas_ao_labels")
    lines = [
        "# Active-space (AO-AVAS) scan: CAS comparison",
        "",
        f"- PySCF: {labels['pyscf']}",
        f"- SIESTA: {labels['siesta']}",
        f"- CAS values: {', '.join(f'({c},{c})' for c in cas_values)}",
    ]
    if pyscf_labels:
        lines.append(f"- PySCF AO-AVAS labels: {', '.join(pyscf_labels)}")
    lines += [
        "",
        "Raw energies are not comparable across codes (all-electron/Gaussian vs "
        "pseudopotential/NAO); the tables use relative energies and mean-centred "
        "cross-code differences only. Pearson `r` uses one point per separation "
        "(three here); treat it as indicative.",
        "",
        "## Cross-code metrics per active space",
        "",
        "| CAS | points | RMSE delta_rel (kcal/mol) | centred RMSE (kcal/mol) | "
        "max |delta_rel| (kcal/mol) | r_rel | centred RMSE E_H-E_L (kcal/mol) | "
        "centred RMSE correction (kcal/mol) | max fractional e- (P/S) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cas in cas_values:
        item = metrics["cas"][cas]
        lines.append(
            f"| ({cas},{cas}) | {item['n_points']} | "
            f"{_fmt(item['rmse_delta_rel_kcal'], 1)} | "
            f"{_fmt(item['rmse_delta_rel_centered_kcal'], 1)} | "
            f"{_fmt(item['max_abs_delta_rel_kcal'], 1)} | "
            f"{_fmt(item['pearson_r_rel'], 3)} | "
            f"{_fmt(item['rmse_delta_hlt_centered_kcal'], 1)} | "
            f"{_fmt(item['rmse_delta_correction_centered_kcal'], 1)} | "
            f"{_fmt(item['max_fractional_electrons_pyscf'], 2)} / "
            f"{_fmt(item['max_fractional_electrons_siesta'], 2)} |"
        )

    lines += [
        "",
        "`max fractional e-` is the largest number of electrons in fractionally "
        "occupied active orbitals (5-95% occupancy) over the scan, a "
        "static-correlation indicator read from the high-level RDM. A large PySCF "
        "value with a near-zero SIESTA value means the two active spaces do not "
        "develop the same multireference character, even at the same CAS size.",
    ]

    lines += [
        "",
        "## Well depths",
        "",
        "Well depth is `E(reference) - min(E)` over the scan. All CAS runs share "
        "the same distance grid, so the CAS-to-CAS trend is internally consistent; "
        "the absolute depths are grid-limited (a minimum between sampled points is "
        "missed) and are not comparable to the denser 18-point scans.",
        "",
        "| CAS | depth PySCF (kcal/mol) | depth SIESTA (kcal/mol) | gap (kcal/mol) | "
        "E_H-E_L depth PySCF | E_H-E_L depth SIESTA | E_H-E_L depth gap |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for cas in cas_values:
        item = metrics["cas"][cas]
        lines.append(
            f"| ({cas},{cas}) | {_fmt(item['well_depth_pyscf_kcal'], 1)} | "
            f"{_fmt(item['well_depth_siesta_kcal'], 1)} | {_fmt(item['well_depth_gap_kcal'], 1)} | "
            f"{_fmt(item['hlt_well_depth_pyscf_kcal'], 1)} | "
            f"{_fmt(item['hlt_well_depth_siesta_kcal'], 1)} | "
            f"{_fmt(item['hlt_well_depth_gap_kcal'], 1)} |"
        )

    lines += [
        "",
        "## Relative embedding energy and cross-code gap",
        "",
        "Each code's total is referenced to its own value at the reference "
        "separation; `delta_rel` is SIESTA minus PySCF.",
    ]
    distances = sorted({row["distance"] for row in rows})
    for cas in cas_values:
        pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for distance in distances:
            p = next(
                row
                for row in rows
                if row["cas"] == cas and row["distance"] == distance and row["backend"] == "pyscf"
            )
            s = next(
                row
                for row in rows
                if row["cas"] == cas and row["distance"] == distance and row["backend"] == "siesta"
            )
            pairs.append((distance, p, s))
        correction_mean = float(np.mean([s["correction"] - p["correction"] for _, p, s in pairs]))
        hlt_mean = float(
            np.mean(
                [s["E_high_A_minus_E_low_A"] - p["E_high_A_minus_E_low_A"] for _, p, s in pairs]
            )
        )
        lines += [
            "",
            f"### CAS({cas},{cas})",
            "",
            "`delta corr` and `delta E_H-E_L` are mean-centred across the scan, "
            "removing the core/pseudopotential offset.",
            "",
            "| R (A) | rel E_embed PySCF (Ha) | rel E_embed SIESTA (Ha) | "
            "delta_rel (kcal/mol) | E_H-E_L PySCF (Ha) | E_H-E_L SIESTA (Ha) | "
            "delta E_H-E_L centred (kcal/mol) | delta corr centred (kcal/mol) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for distance, p, s in pairs:
            delta_rel = s["rel_embedding_energy"] - p["rel_embedding_energy"]
            hlt_centered = (
                s["E_high_A_minus_E_low_A"] - p["E_high_A_minus_E_low_A"] - hlt_mean
            ) * HARTREE_TO_KCAL
            corr_centered = (s["correction"] - p["correction"] - correction_mean) * HARTREE_TO_KCAL
            lines.append(
                f"| {distance:.2f} | {p['rel_embedding_energy']:.6f} | "
                f"{s['rel_embedding_energy']:.6f} | {delta_rel * HARTREE_TO_KCAL:.1f} | "
                f"{p['E_high_A_minus_E_low_A']:.6f} | {s['E_high_A_minus_E_low_A']:.6f} | "
                f"{hlt_centered:.1f} | {corr_centered:.1f} |"
            )

    change = metrics["comparison"]["rmse_delta_rel_centered_change_kcal"]
    depth_change = metrics["comparison"]["well_depth_gap_change_kcal"]
    direction = "narrows" if change < 0 else "widens"
    depth_direction = "narrows" if depth_change < 0 else "widens"
    lines += [
        "",
        "## Verdict",
        "",
        f"Going from CAS({smallest},{smallest}) to CAS({largest},{largest}) "
        f"{direction} the centred cross-code relative-energy gap by "
        f"{abs(change):.1f} kcal/mol (centred RMSE "
        f"{metrics['cas'][smallest]['rmse_delta_rel_centered_kcal']:.1f} -> "
        f"{metrics['cas'][largest]['rmse_delta_rel_centered_kcal']:.1f} kcal/mol) and "
        f"{depth_direction} the well-depth gap by {abs(depth_change):.1f} kcal/mol "
        f"({metrics['cas'][smallest]['well_depth_gap_kcal']:.1f} -> "
        f"{metrics['cas'][largest]['well_depth_gap_kcal']:.1f} kcal/mol).",
        "",
        f"PySCF's E_embed well depth goes "
        f"{metrics['cas'][smallest]['well_depth_pyscf_kcal']:.1f} -> "
        f"{metrics['cas'][largest]['well_depth_pyscf_kcal']:.1f} kcal/mol while "
        f"SIESTA's goes {metrics['cas'][smallest]['well_depth_siesta_kcal']:.1f} -> "
        f"{metrics['cas'][largest]['well_depth_siesta_kcal']:.1f} kcal/mol: enlarging "
        f"the active space moves the two codes apart, it does not converge them "
        f"together. The static-correlation indicator tells the same story: PySCF's "
        f"max fractional electrons go "
        f"{_fmt(metrics['cas'][smallest]['max_fractional_electrons_pyscf'], 2)} -> "
        f"{_fmt(metrics['cas'][largest]['max_fractional_electrons_pyscf'], 2)}, "
        f"SIESTA's stay "
        f"{_fmt(metrics['cas'][smallest]['max_fractional_electrons_siesta'], 2)} -> "
        f"{_fmt(metrics['cas'][largest]['max_fractional_electrons_siesta'], 2)}.",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _style_axis(ax, *, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5, length=3)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9.5)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9.5)
    ax.title.set_color(TEXT_PRIMARY)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["system-ui", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    return plt


def _save(fig, out_dir: Path, stem: str, formats: tuple[str, ...]) -> None:
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=200, bbox_inches="tight")


def _plot_components(
    records_p: list[dict[str, Any]],
    records_s: list[dict[str, Any]],
    *,
    label_pyscf: str,
    label_siesta: str,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), sharex=True)
    panels = (
        ("E_low_total", "E_low(total) relative"),
        ("E_low_A", "E_low(A) relative"),
        ("E_high_A", "E_high(A) relative"),
        ("E_high_A_minus_E_low_A", "E_high(A) - E_low(A)"),
        ("correction", "Correction tr[(g~-g) v_emb]"),
        ("embedding_energy", "E_embed relative"),
    )
    for ax, (key, title) in zip(axes.flat, panels, strict=True):
        p_x = [row["distance"] for row in records_p]
        p_y = [row[f"rel_{key}"] * HARTREE_TO_KCAL for row in records_p]
        s_x = [row["distance"] for row in records_s]
        s_y = [row[f"rel_{key}"] * HARTREE_TO_KCAL for row in records_s]
        ax.plot(p_x, p_y, color=SERIES[0], linewidth=2.0, label=label_pyscf)
        ax.plot(s_x, s_y, color=SERIES[1], linewidth=2.0, label=label_siesta)
        ax.axhline(0.0, color=AXIS, linewidth=0.8, linestyle="--")
        _style_axis(ax, xlabel="R(C-N) (A)", ylabel="Relative energy (kcal/mol)")
        ax.set_title(title, loc="left", fontsize=9.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        labelcolor=TEXT_SECONDARY,
        bbox_to_anchor=(0.995, 0.995),
    )
    fig.suptitle(
        "Embedding components, each code referenced to the same separation",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.01,
        0.005,
        "Raw component values carry different core/pseudopotential zeros; only these "
        "relative shapes (and the centred differences) are comparable.",
        color=TEXT_MUTED,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    _save(fig, out_dir, "components_relative", formats)
    plt.close(fig)


def _plot_centred(
    decomposition_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = [row["distance"] for row in decomposition_rows]
    ax.plot(
        x,
        [row["delta_rel_embedding_ha"] * HARTREE_TO_KCAL for row in decomposition_rows],
        color=TEXT_PRIMARY,
        linewidth=2.4,
        marker="o",
        markersize=5,
        label="delta_rel E_embed (total)",
    )
    for index, key in enumerate(DECOMPOSITION_TERMS):
        ax.plot(
            x,
            [row[f"delta_rel_{key}_ha"] * HARTREE_TO_KCAL for row in decomposition_rows],
            color=SERIES[index],
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=f"delta_rel {DISPLAY[key]}",
        )
    ax.axhline(0.0, color=AXIS, linewidth=0.8, linestyle="--")
    _style_axis(ax, xlabel="R(C-N) (A)", ylabel="SIESTA - PySCF relative energy (kcal/mol)")
    ax.set_title("Cross-code relative-energy gap and its additive terms", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "gap_decomposition", formats)
    plt.close(fig)


def _plot_well_depths(
    values: dict[str, dict[str, float]],
    *,
    out_dir: Path,
    formats: tuple[str, ...],
    title: str,
    stem: str,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    names = list(values)
    p = [values[name]["pyscf"] for name in names]
    s = [values[name]["siesta"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, p, width, color=SERIES[0], label="PySCF")
    ax.bar(x + width / 2, s, width, color=SERIES[1], label="SIESTA")
    for xi, yi in zip(x - width / 2, p, strict=True):
        ax.annotate(
            f"{yi:.0f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
    for xi, yi in zip(x + width / 2, s, strict=True):
        ax.annotate(
            f"{yi:.0f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    _style_axis(ax, xlabel="Quantity", ylabel="Well depth (kcal/mol)")
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, stem, formats)
    plt.close(fig)


def _plot_cas(
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    plt = _plt()
    cas_values = sorted(metrics["cas"])
    distances = sorted({row["distance"] for row in rows})

    # Relative embedding energy, one panel per code.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=False)
    for ax, backend, display in (
        (axes[0], "pyscf", "PySCF"),
        (axes[1], "siesta", "SIESTA"),
    ):
        for index, cas in enumerate(cas_values):
            subset = sorted(
                (row for row in rows if row["backend"] == backend and row["cas"] == cas),
                key=lambda row: row["distance"],
            )
            ax.plot(
                [row["distance"] for row in subset],
                [row["rel_embedding_energy"] * HARTREE_TO_KCAL for row in subset],
                color=SERIES[index],
                linewidth=2.0,
                marker="o",
                markersize=5,
                label=f"CAS({cas},{cas})",
            )
        ax.axhline(0.0, color=AXIS, linewidth=0.8, linestyle="--")
        _style_axis(ax, xlabel="R(C-N) (A)", ylabel="Relative E_embed (kcal/mol)")
        ax.set_title(f"{display}: active-space convergence", loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "cas_relative_energy", formats)
    plt.close(fig)

    # Cross-code relative-energy gap per CAS.
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for index, cas in enumerate(cas_values):
        x: list[float] = []
        y: list[float] = []
        for distance in distances:
            p = next(
                row
                for row in rows
                if row["cas"] == cas and row["distance"] == distance and row["backend"] == "pyscf"
            )
            s = next(
                row
                for row in rows
                if row["cas"] == cas and row["distance"] == distance and row["backend"] == "siesta"
            )
            x.append(distance)
            y.append((s["rel_embedding_energy"] - p["rel_embedding_energy"]) * HARTREE_TO_KCAL)
        item = metrics["cas"][cas]
        ax.plot(
            x,
            y,
            color=SERIES[index],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=f"CAS({cas},{cas}) centred RMSE {item['rmse_delta_rel_centered_kcal']:.0f}",
        )
    ax.axhline(0.0, color=AXIS, linewidth=0.8, linestyle="--")
    _style_axis(ax, xlabel="R(C-N) (A)", ylabel="SIESTA - PySCF rel. E_embed (kcal/mol)")
    ax.set_title("Cross-code gap vs active-space size", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "cas_cross_code_gap", formats)
    plt.close(fig)

    # E_H - E_L (raw, one panel per code) and the centred cross-code delta.
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    for ax, backend, display in (
        (axes[0], "pyscf", "PySCF"),
        (axes[1], "siesta", "SIESTA"),
    ):
        for index, cas in enumerate(cas_values):
            subset = sorted(
                (row for row in rows if row["backend"] == backend and row["cas"] == cas),
                key=lambda row: row["distance"],
            )
            ax.plot(
                [row["distance"] for row in subset],
                [row["E_high_A_minus_E_low_A"] for row in subset],
                color=SERIES[index],
                linewidth=2.0,
                marker="o",
                markersize=5,
                label=f"CAS({cas},{cas})",
            )
        _style_axis(ax, xlabel="R(C-N) (A)", ylabel="E_H(A) - E_L(A) (Ha)")
        ax.set_title(f"{display}: high-level term", loc="left", fontsize=9.5)
        ax.legend(frameon=False, fontsize=7.5, labelcolor=TEXT_SECONDARY, loc="best")
    ax = axes[2]
    for index, cas in enumerate(cas_values):
        x = []
        y = []
        for distance in distances:
            p = next(
                row
                for row in rows
                if row["cas"] == cas and row["distance"] == distance and row["backend"] == "pyscf"
            )
            s = next(
                row
                for row in rows
                if row["cas"] == cas and row["distance"] == distance and row["backend"] == "siesta"
            )
            x.append(distance)
            y.append(s["E_high_A_minus_E_low_A"] - p["E_high_A_minus_E_low_A"])
        centred = np.asarray(y) - np.mean(y)
        ax.plot(
            x,
            centred * HARTREE_TO_KCAL,
            color=SERIES[index],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=f"CAS({cas},{cas})",
        )
    ax.axhline(0.0, color=AXIS, linewidth=0.8, linestyle="--")
    _style_axis(ax, xlabel="R(C-N) (A)", ylabel="Centred delta (kcal/mol)")
    ax.set_title("Cross-code E_H-E_L, mean-centred", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "cas_high_level_term", formats)
    plt.close(fig)

    # Well depths per CAS.
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(cas_values))
    width = 0.36
    p = [metrics["cas"][cas]["well_depth_pyscf_kcal"] for cas in cas_values]
    s = [metrics["cas"][cas]["well_depth_siesta_kcal"] for cas in cas_values]
    ax.bar(x - width / 2, p, width, color=SERIES[0], label="PySCF")
    ax.bar(x + width / 2, s, width, color=SERIES[1], label="SIESTA")
    for xi, yi in zip(x - width / 2, p, strict=True):
        ax.annotate(
            f"{yi:.0f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
    for xi, yi in zip(x + width / 2, s, strict=True):
        ax.annotate(
            f"{yi:.0f}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7.5,
            color=TEXT_SECONDARY,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"CAS({cas},{cas})" for cas in cas_values])
    _style_axis(ax, xlabel="Active space", ylabel="E_embed well depth (kcal/mol)")
    ax.set_title("Well depth vs active-space size", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    _save(fig, out_dir, "cas_well_depths", formats)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Component-wise comparison of PySCF and SIESTA embedding scans, "
            "with an optional CAS-size sweep."
        )
    )
    parser.add_argument("--pyscf-dir", type=Path, help="PySCF run with summary.csv")
    parser.add_argument("--siesta-dir", type=Path, help="SIESTA run with summary.csv")
    parser.add_argument(
        "--pyscf-cas",
        action="append",
        metavar="CAS=DIR",
        help="PySCF CAS run (repeatable), e.g. --pyscf-cas 4=/tmp/cas4",
    )
    parser.add_argument(
        "--siesta-cas",
        action="append",
        metavar="CAS=DIR",
        help="SIESTA CAS run (repeatable), e.g. --siesta-cas 4=/tmp/cas4",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis/butyronitrile-pbe-hf-fci/component-diagnosis"),
        help="output directory (git-ignored)",
    )
    parser.add_argument(
        "--reference",
        default="max",
        help="reference separation for relative energies and well depths: "
        "'min', 'max', or a distance in A (default: max)",
    )
    parser.add_argument(
        "--formats",
        default="png",
        help="comma-separated plot formats (default: png)",
    )
    args = parser.parse_args(argv)
    if (args.pyscf_dir is None) != (args.siesta_dir is None):
        parser.error("--pyscf-dir and --siesta-dir must be given together")
    if bool(args.pyscf_cas) != bool(args.siesta_cas):
        parser.error("--pyscf-cas and --siesta-cas must be given together")
    if args.pyscf_dir is None and not args.pyscf_cas:
        parser.error("give --pyscf-dir/--siesta-dir or --pyscf-cas/--siesta-cas")
    return args


def _run_components(args: argparse.Namespace, out_dir: Path, formats: tuple[str, ...]) -> None:
    pyscf, p_meta = _read_summary(args.pyscf_dir)
    siesta, s_meta = _read_summary(args.siesta_dir)
    label_pyscf = _label(p_meta, "pyscf")
    label_siesta = _label(s_meta, "siesta")

    records_p = _component_records("pyscf", pyscf, args.reference)
    records_s = _component_records("siesta", siesta, args.reference)
    metrics = _component_metrics(pyscf, siesta, args.reference)
    decomposition_rows, decomposition = _decomposition(pyscf, siesta, args.reference)
    metrics["decomposition"] = decomposition

    reference_distance = _reference_row(pyscf, args.reference)["distance"]
    s_reference = _reference_row(siesta, args.reference)["distance"]
    if abs(reference_distance - s_reference) > 1e-6:
        raise SystemExit(
            f"reference geometry differs between runs: {reference_distance} vs {s_reference}"
        )
    metrics.update(
        {
            "mode": "components",
            "labels": {"pyscf": label_pyscf, "siesta": label_siesta},
            "reference": args.reference,
            "reference_distance": reference_distance,
            "n_points": len(records_p),
            "caveat": (
                "Raw offsets between the codes are core/pseudopotential energy zeros, "
                "not errors. Only relative energies, mean-centred differences, and "
                "well depths are comparable."
            ),
        }
    )

    _write_csv(out_dir / "components.csv", records_p + records_s)

    difference_rows: list[dict[str, Any]] = []
    for p, s in _merge(pyscf, siesta):
        row = {"distance": p["distance"]}
        for key in COMPONENTS:
            row[f"{key}_pyscf"] = p[key]
            row[f"{key}_siesta"] = s[key]
            row[f"delta_{key}_ha"] = s[key] - p[key]
            row[f"delta_{key}_kcal"] = (s[key] - p[key]) * HARTREE_TO_KCAL
        difference_rows.append(row)
    _write_csv(out_dir / "component_differences.csv", difference_rows)
    _write_csv(out_dir / "gap_decomposition.csv", decomposition_rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_component_markdown(
        out_dir / "component_diagnosis.md",
        label_pyscf=label_pyscf,
        label_siesta=label_siesta,
        reference=args.reference,
        reference_distance=reference_distance,
        records_p=records_p,
        records_s=records_s,
        metrics=metrics,
        decomposition_rows=decomposition_rows,
        decomposition=decomposition,
    )

    _plot_components(
        records_p,
        records_s,
        label_pyscf=label_pyscf,
        label_siesta=label_siesta,
        out_dir=out_dir,
        formats=formats,
    )
    _plot_centred(decomposition_rows, out_dir=out_dir, formats=formats)
    _plot_well_depths(
        {
            "E_embed": {
                "pyscf": metrics["components"]["embedding_energy"]["well_depth_pyscf_kcal"],
                "siesta": metrics["components"]["embedding_energy"]["well_depth_siesta_kcal"],
            },
            "E_low(total)": {
                "pyscf": metrics["components"]["E_low_total"]["well_depth_pyscf_kcal"],
                "siesta": metrics["components"]["E_low_total"]["well_depth_siesta_kcal"],
            },
            "E_H-E_L": {
                "pyscf": metrics["components"]["E_high_A_minus_E_low_A"]["well_depth_pyscf_kcal"],
                "siesta": metrics["components"]["E_high_A_minus_E_low_A"]["well_depth_siesta_kcal"],
            },
        },
        out_dir=out_dir,
        formats=formats,
        title="Embedding well depths by component",
        stem="well_depths",
    )


def _run_cas(args: argparse.Namespace, out_dir: Path, formats: tuple[str, ...]) -> None:
    pyscf_runs = _parse_cas_runs(args.pyscf_cas)
    siesta_runs = _parse_cas_runs(args.siesta_cas)
    rows, metrics, labels = _cas_comparison(pyscf_runs, siesta_runs, args.reference)
    metrics.update(
        {
            "mode": "cas",
            "labels": {key: labels[key] for key in ("pyscf", "siesta")},
            "reference": args.reference,
            "caveat": (
                "Raw offsets between the codes are core/pseudopotential energy zeros, "
                "not errors. Only relative energies, mean-centred differences, and "
                "well depths are comparable."
            ),
        }
    )
    _write_csv(out_dir / "cas_components.csv", rows)
    (out_dir / "cas_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_cas_markdown(
        out_dir / "cas_comparison.md",
        labels=labels,
        metrics=metrics,
        rows=rows,
    )
    _plot_cas(rows, metrics, out_dir=out_dir, formats=formats)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(part.strip() for part in args.formats.split(",") if part.strip())
    if args.pyscf_dir is not None:
        _run_components(args, out_dir, formats)
    if args.pyscf_cas:
        _run_cas(args, out_dir, formats)
    print(f"wrote diagnosis to {out_dir}")


if __name__ == "__main__":
    main()
