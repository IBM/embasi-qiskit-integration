# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Compare two :mod:`run_structural_scan` summaries and plot the scan.

The two calculations use each backend's native basis: PySCF is
all-electron/Gaussian and SIESTA is pseudopotential/NAO.  Their absolute totals
therefore sit on different energy zeros and must not be subtracted.  The
comparison uses:

* the relative-energy curve (each backend referenced to the same separation),
  which tests the shape of the bond-breaking profile;
* the embedding correction ``E_embed - E_low_total``, which directly exercises
  the downfolded Hamiltonian and high-level solver.  Its raw value still carries
  a code-dependent core/basis offset, so the plot compares mean-centred shapes
  and the CSV keeps both.

Both raw and mean-centred differences are reported.  The centred number is the
useful one for "do the two implementations agree on the shape?"; the raw number
also contains the constant basis/pseudopotential offset.  The exact
same-pseudopotential validation is a separate test, not this scan.

Example::

    uv run python scripts/benchmarks/compare_structural_scan.py \
        --pyscf-dir /tmp/scan-pyscf \
        --siesta-dir /tmp/scan-siesta \
        --out-dir /tmp/scan-compare
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
# Reference palette, validated for two series (blue, orange) on the light surface.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")


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
    with summary_path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            if raw.get("status") not in {"ok", "cached"}:
                continue
            distance = _num(raw.get("distance"))
            energy = _num(raw.get("embedding_energy"))
            low_total = _num(raw.get("E_low_total"))
            if distance is None or energy is None or low_total is None:
                continue
            row = dict(raw)
            row["distance"] = distance
            for key in (
                "embedding_energy",
                "E_low_total",
                "E_low_A",
                "E_high_A",
                "E_solver",
                "e_core",
                "correction",
                "embedding_correction",
                "high_level_term",
                "projector_leak",
                "cycles",
            ):
                row[key] = _num(raw.get(key))
            row["converged"] = _as_bool(raw.get("converged"))
            rows.append(row)
    if not rows:
        raise SystemExit(f"{summary_path} has no successful, complete rows")
    rows.sort(key=lambda row: row["distance"])
    metadata: dict[str, Any] = {}
    meta_path = run_dir / "summary.json"
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text()).get("metadata", {})
    single_shot = metadata.get("max_cycles") == 1
    for row in rows:
        row["single_shot"] = single_shot
    return rows, metadata


def _reference_row(rows: list[dict[str, Any]], spec: str) -> dict[str, Any]:
    if spec == "min":
        return min(rows, key=lambda row: row["distance"])
    if spec == "max":
        return max(rows, key=lambda row: row["distance"])
    target = float(spec)
    return min(rows, key=lambda row: abs(row["distance"] - target))


def _merge(
    pyscf: list[dict[str, Any]], siesta: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_distance = {
        "pyscf": {round(row["distance"], 6): row for row in pyscf},
        "siesta": {round(row["distance"], 6): row for row in siesta},
    }
    common = sorted(set(by_distance["pyscf"]) & set(by_distance["siesta"]))
    if not common:
        raise SystemExit("the two summaries have no common geometry distances")
    return [(by_distance["pyscf"][d], by_distance["siesta"][d]) for d in common]


def _rmse(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def _max_abs(values: list[float]) -> float:
    return float(np.max(np.abs(values))) if values else float("nan")


def _label(metadata: dict[str, Any], fallback: str) -> str:
    backend = metadata.get("backend", fallback)
    display = {"pyscf": "PySCF", "siesta": "SIESTA"}.get(str(backend).lower(), str(backend))
    xc_ll = metadata.get("xc_ll", "PBE")
    xc_hl = metadata.get("xc_hl", "HF")
    basis = metadata.get("basis", "?")
    return f"{display} ({xc_ll}/{xc_hl}, {basis})"


def _build_comparison(
    pyscf: list[dict[str, Any]],
    siesta: list[dict[str, Any]],
    reference: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ref_pyscf = _reference_row(pyscf, reference)
    ref_siesta = _reference_row(siesta, reference)
    if abs(ref_pyscf["distance"] - ref_siesta["distance"]) > 1e-6:
        raise SystemExit(
            "reference geometry differs between runs: "
            f"{ref_pyscf['distance']} vs {ref_siesta['distance']}"
        )
    e_ref_pyscf = ref_pyscf["embedding_energy"]
    e_ref_siesta = ref_siesta["embedding_energy"]

    rows: list[dict[str, Any]] = []
    for p, s in _merge(pyscf, siesta):
        rel_p = p["embedding_energy"] - e_ref_pyscf
        rel_s = s["embedding_energy"] - e_ref_siesta
        corr_p = p["embedding_energy"] - p["E_low_total"]
        corr_s = s["embedding_energy"] - s["E_low_total"]
        row = {
            "distance": p["distance"],
            "stem_pyscf": p["stem"],
            "stem_siesta": s["stem"],
            "converged_pyscf": p["converged"],
            "converged_siesta": s["converged"],
            "single_shot_pyscf": p["single_shot"],
            "single_shot_siesta": s["single_shot"],
            "plot_converged_pyscf": bool(p["converged"] or p["single_shot"]),
            "plot_converged_siesta": bool(s["converged"] or s["single_shot"]),
            "cycles_pyscf": p["cycles"],
            "cycles_siesta": s["cycles"],
            "embedding_energy_pyscf": p["embedding_energy"],
            "embedding_energy_siesta": s["embedding_energy"],
            "E_low_total_pyscf": p["E_low_total"],
            "E_low_total_siesta": s["E_low_total"],
            "E_low_A_pyscf": p["E_low_A"],
            "E_low_A_siesta": s["E_low_A"],
            "E_high_A_pyscf": p["E_high_A"],
            "E_high_A_siesta": s["E_high_A"],
            "E_solver_pyscf": p["E_solver"],
            "E_solver_siesta": s["E_solver"],
            "e_core_pyscf": p["e_core"],
            "e_core_siesta": s["e_core"],
            "correction_pyscf": p["correction"],
            "correction_siesta": s["correction"],
            "rel_energy_pyscf_ha": rel_p,
            "rel_energy_siesta_ha": rel_s,
            "delta_rel_ha": rel_s - rel_p,
            "embedding_correction_pyscf_ha": corr_p,
            "embedding_correction_siesta_ha": corr_s,
            "delta_embedding_correction_ha": corr_s - corr_p,
            "delta_E_low_total_ha": s["E_low_total"] - p["E_low_total"],
        }
        for ha_key in (
            "rel_energy_pyscf_ha",
            "rel_energy_siesta_ha",
            "delta_rel_ha",
            "embedding_correction_pyscf_ha",
            "embedding_correction_siesta_ha",
            "delta_embedding_correction_ha",
        ):
            row[ha_key.replace("_ha", "_kcal")] = row[ha_key] * HARTREE_TO_KCAL
        rows.append(row)

    delta_rel = [row["delta_rel_ha"] for row in rows]
    delta_corr = [row["delta_embedding_correction_ha"] for row in rows]
    mean_rel = float(np.mean(delta_rel))
    mean_corr = float(np.mean(delta_corr))
    centered_rel = [value - mean_rel for value in delta_rel]
    centered_corr = [value - mean_corr for value in delta_corr]

    rel_series_p = [row["rel_energy_pyscf_ha"] for row in rows]
    rel_series_s = [row["rel_energy_siesta_ha"] for row in rows]
    corr_series_p = [row["embedding_correction_pyscf_ha"] for row in rows]
    corr_series_s = [row["embedding_correction_siesta_ha"] for row in rows]
    total_series_p = [row["embedding_energy_pyscf"] for row in rows]
    total_series_s = [row["embedding_energy_siesta"] for row in rows]
    low_series_p = [row["E_low_total_pyscf"] for row in rows]
    low_series_s = [row["E_low_total_siesta"] for row in rows]

    def correlation(a: list[float], b: list[float]) -> float:
        if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    pyscf_min = min(pyscf, key=lambda row: row["embedding_energy"])
    siesta_min = min(siesta, key=lambda row: row["embedding_energy"])
    metrics = {
        "n_points": len(rows),
        "reference_distance": ref_pyscf["distance"],
        "reference": reference,
        "pearson_r_total": correlation(total_series_p, total_series_s),
        "pearson_r_low_level_total": correlation(low_series_p, low_series_s),
        "pearson_r_rel_energy": correlation(rel_series_p, rel_series_s),
        "pearson_r_embedding_correction": correlation(corr_series_p, corr_series_s),
        "n_unconverged_pyscf": sum(
            not row["converged_pyscf"] and not row["single_shot_pyscf"] for row in rows
        ),
        "n_unconverged_siesta": sum(
            not row["converged_siesta"] and not row["single_shot_siesta"] for row in rows
        ),
        "single_shot_pyscf": bool(rows[0]["single_shot_pyscf"]),
        "single_shot_siesta": bool(rows[0]["single_shot_siesta"]),
        "mean_delta_rel_ha": mean_rel,
        "mean_delta_rel_kcal": mean_rel * HARTREE_TO_KCAL,
        "rmse_delta_rel_ha": _rmse(delta_rel),
        "rmse_delta_rel_kcal": _rmse(delta_rel) * HARTREE_TO_KCAL,
        "rmse_delta_rel_centered_ha": _rmse(centered_rel),
        "rmse_delta_rel_centered_kcal": _rmse(centered_rel) * HARTREE_TO_KCAL,
        "max_abs_delta_rel_ha": _max_abs(delta_rel),
        "max_abs_delta_rel_kcal": _max_abs(delta_rel) * HARTREE_TO_KCAL,
        "max_abs_delta_rel_centered_ha": _max_abs(centered_rel),
        "max_abs_delta_rel_centered_kcal": _max_abs(centered_rel) * HARTREE_TO_KCAL,
        "mean_delta_embedding_correction_ha": mean_corr,
        "mean_delta_embedding_correction_kcal": mean_corr * HARTREE_TO_KCAL,
        "rmse_delta_embedding_correction_ha": _rmse(delta_corr),
        "rmse_delta_embedding_correction_kcal": _rmse(delta_corr) * HARTREE_TO_KCAL,
        "rmse_delta_embedding_correction_centered_ha": _rmse(centered_corr),
        "rmse_delta_embedding_correction_centered_kcal": _rmse(centered_corr)
        * HARTREE_TO_KCAL,
        "max_abs_delta_embedding_correction_ha": _max_abs(delta_corr),
        "max_abs_delta_embedding_correction_kcal": _max_abs(delta_corr) * HARTREE_TO_KCAL,
        "min_distance_pyscf": pyscf_min["distance"],
        "min_distance_siesta": siesta_min["distance"],
        "min_energy_rel_pyscf_ha": pyscf_min["embedding_energy"] - e_ref_pyscf,
        "min_energy_rel_siesta_ha": siesta_min["embedding_energy"] - e_ref_siesta,
        "well_depth_pyscf_ha": e_ref_pyscf - pyscf_min["embedding_energy"],
        "well_depth_siesta_ha": e_ref_siesta - siesta_min["embedding_energy"],
        "well_depth_pyscf_kcal": (e_ref_pyscf - pyscf_min["embedding_energy"])
        * HARTREE_TO_KCAL,
        "well_depth_siesta_kcal": (e_ref_siesta - siesta_min["embedding_energy"])
        * HARTREE_TO_KCAL,
    }
    return rows, metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    label_pyscf: str,
    label_siesta: str,
) -> None:
    lines = [
        "# Butyronitrile embedding scan comparison",
        "",
        f"- PySCF: {label_pyscf}",
        f"- SIESTA: {label_siesta}",
        f"- Reference separation: {metrics['reference_distance']:.2f} A",
        f"- Points compared: {metrics['n_points']}",
        "",
        "The absolute totals are not comparable (all-electron/Gaussian vs "
        "pseudopotential/NAO); differences below use relative energies and "
        "`E_embed - E_low_total`.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| Pearson r, absolute totals | {metrics['pearson_r_total']:.4f} |",
        f"| Pearson r, low-level PBE totals | {metrics['pearson_r_low_level_total']:.4f} |",
        f"| Pearson r, relative energy | {metrics['pearson_r_rel_energy']:.4f} |",
        f"| Pearson r, embedding correction | {metrics['pearson_r_embedding_correction']:.4f} |",
        f"| mean delta E_rel (kcal/mol) | {metrics['mean_delta_rel_kcal']:.4f} |",
        f"| RMSE delta E_rel (kcal/mol) | {metrics['rmse_delta_rel_kcal']:.4f} |",
        "| RMSE delta E_rel, mean-centred (kcal/mol) | "
        f"{metrics['rmse_delta_rel_centered_kcal']:.4f} |",
        "| RMSE delta embedding correction (kcal/mol) | "
        f"{metrics['rmse_delta_embedding_correction_kcal']:.4f} |",
        "| RMSE delta embedding correction, mean-centred (kcal/mol) | "
        f"{metrics['rmse_delta_embedding_correction_centered_kcal']:.4f} |",
        f"| minimum R, PySCF (A) | {metrics['min_distance_pyscf']:.2f} |",
        f"| minimum R, SIESTA (A) | {metrics['min_distance_siesta']:.2f} |",
        f"| well depth, PySCF (kcal/mol) | {metrics['well_depth_pyscf_kcal']:.4f} |",
        f"| well depth, SIESTA (kcal/mol) | {metrics['well_depth_siesta_kcal']:.4f} |",
        "",
        "`S` marks a single-shot (`max_cycles=1`) protocol.  `N` marks an "
        "iterative outer loop that did not converge and uses its final iterate; "
        "those points are drawn with open markers.  A density-dependent selector "
        "can change the active orbitals between geometries; check each point's "
        "`diagnostics.csv` before reading a jump as physics.",
        "",
        "| R (A) | delta E_rel (kcal/mol) | delta corr, centred (kcal/mol) | conv P/S |",
        "|---:|---:|---:|:---:|",
    ]
    mean_corr_kcal = metrics["mean_delta_embedding_correction_kcal"]
    for row in rows:
        lines.append(
            f"| {row['distance']:.2f} | {row['delta_rel_kcal']:.4f} | "
            f"{row['delta_embedding_correction_kcal'] - mean_corr_kcal:.4f} | "
            f"{'S' if row['single_shot_pyscf'] else ('Y' if row['converged_pyscf'] else 'N')}/"
            f"{'S' if row['single_shot_siesta'] else ('Y' if row['converged_siesta'] else 'N')} |"
        )
    path.write_text("\n".join(lines) + "\n")


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


def _series_plot(
    ax,
    rows: list[dict[str, Any]],
    value_key: str,
    *,
    label: str,
    color: str,
    unconverged_key: str,
) -> None:
    x = [row["distance"] for row in rows]
    y = [row[value_key] for row in rows]
    ax.plot(x, y, color=color, linewidth=2.0, label=label, zorder=2)
    converged = [row[unconverged_key] for row in rows]
    ax.scatter(
        [xi for xi, ok in zip(x, converged) if ok],
        [yi for yi, ok in zip(y, converged) if ok],
        s=34,
        color=color,
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=3,
    )
    ax.scatter(
        [xi for xi, ok in zip(x, converged) if not ok],
        [yi for yi, ok in zip(y, converged) if not ok],
        s=34,
        facecolor=SURFACE,
        edgecolor=color,
        linewidth=1.5,
        zorder=3,
    )



def _annotate_ends(ax, entries: list[tuple[str, float, float]]) -> None:
    """Direct-label each series at its last point, separating close labels.

    Text stays in ink; the coloured line and marker carry identity.  When the two
    endpoints nearly coincide (common at a dissociation limit), split the labels
    vertically so they remain readable.
    """
    if not entries:
        return
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin if ymax > ymin else 1.0
    ordered = sorted(entries, key=lambda item: item[2])
    offsets: dict[str, tuple[int, int]] = {}
    if len(ordered) > 1 and abs(ordered[-1][2] - ordered[0][2]) < 0.06 * span:
        offsets[ordered[0][0]] = (6, -9)
        offsets[ordered[-1][0]] = (6, 9)
        if len(ordered) > 2:
            for name, _x, _y in ordered[1:-1]:
                offsets[name] = (6, 0)
    else:
        for name, _x, _y in entries:
            offsets[name] = (6, 0)
    for label, x, y in entries:
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offsets[label],
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT_SECONDARY,
        )


def _plot(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    label_pyscf: str,
    label_siesta: str,
    metrics: dict[str, Any],
    formats: tuple[str, ...],
) -> None:
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
    x = [row["distance"] for row in rows]

    # 0. Absolute totals, one panel per backend (their energy zeros differ by
    # ~174 Ha here, so a shared axis would hide every feature).
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True)
    for ax, key, label, color, conv_key in (
        (
            axes[0],
            "embedding_energy_pyscf",
            label_pyscf,
            SERIES[0],
            "plot_converged_pyscf",
        ),
        (
            axes[1],
            "embedding_energy_siesta",
            label_siesta,
            SERIES[1],
            "plot_converged_siesta",
        ),
    ):
        y = [row[key] for row in rows]
        ax.plot(x, y, color=color, linewidth=2.0)
        ax.scatter(
            [xi for xi, row in zip(x, rows) if row[conv_key]],
            [yi for yi, row in zip(y, rows) if row[conv_key]],
            s=30,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=3,
        )
        ax.scatter(
            [xi for xi, row in zip(x, rows) if not row[conv_key]],
            [yi for yi, row in zip(y, rows) if not row[conv_key]],
            s=30,
            facecolor=SURFACE,
            edgecolor=color,
            linewidth=1.4,
            zorder=3,
        )
        _style_axis(ax, xlabel="R(C-N) (A)", ylabel="Embedding total (Ha)")
        ax.set_title(label, loc="left", fontsize=9.5)
    fig.suptitle(
        "Absolute embedding totals (separate y-scales; compare the shape, not the offset)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.125,
        0.005,
        "All-electron/Gaussian and pseudopotential/NAO totals have different zeros; "
        "the relative-energy plot is the like-for-like trend.",
        color=TEXT_MUTED,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for fmt in formats:
        fig.savefig(out_dir / f"totals.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    delta_rel = [row["delta_rel_kcal"] for row in rows]
    delta_corr = [row["delta_embedding_correction_kcal"] for row in rows]
    mean_rel = float(np.mean(delta_rel))
    mean_corr = float(np.mean(delta_corr))

    # 1. Relative energy: two series, the actual curve shape.
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    _series_plot(
        ax,
        rows,
        "rel_energy_pyscf_kcal",
        label=label_pyscf,
        color=SERIES[0],
        unconverged_key="plot_converged_pyscf",
    )
    _series_plot(
        ax,
        rows,
        "rel_energy_siesta_kcal",
        label=label_siesta,
        color=SERIES[1],
        unconverged_key="plot_converged_siesta",
    )
    _annotate_ends(
        ax,
        [
            (label_pyscf, rows[-1]["distance"], rows[-1]["rel_energy_pyscf_kcal"]),
            (label_siesta, rows[-1]["distance"], rows[-1]["rel_energy_siesta_kcal"]),
        ],
    )
    _style_axis(ax, xlabel="R(C-N) (A)", ylabel="Relative energy (kcal/mol)")
    ax.set_title("Butyronitrile bond stretch: relative embedding energy", loc="left", fontsize=11)
    fig.text(
        0.125,
        0.005,
        "Each backend referenced to its own R = "
        f"{metrics['reference_distance']:.1f} A total. Open marker: outer loop not converged.",
        color=TEXT_MUTED,
        fontsize=7.5,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for fmt in formats:
        fig.savefig(out_dir / f"relative_energy.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Embedding correction.  Its raw magnitude has a code-dependent core/basis
    # offset (here ~2.3 Ha), so shift each series by its mean to expose the shape.
    corr_p = [row["embedding_correction_pyscf_kcal"] for row in rows]
    corr_s = [row["embedding_correction_siesta_kcal"] for row in rows]
    mean_corr_p = float(np.mean(corr_p))
    mean_corr_s = float(np.mean(corr_s))
    for row, cp, cs in zip(rows, corr_p, corr_s):
        row["embedding_correction_pyscf_plot_kcal"] = cp - mean_corr_p
        row["embedding_correction_siesta_plot_kcal"] = cs - mean_corr_s
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    _series_plot(
        ax,
        rows,
        "embedding_correction_pyscf_plot_kcal",
        label=label_pyscf,
        color=SERIES[0],
        unconverged_key="plot_converged_pyscf",
    )
    _series_plot(
        ax,
        rows,
        "embedding_correction_siesta_plot_kcal",
        label=label_siesta,
        color=SERIES[1],
        unconverged_key="plot_converged_siesta",
    )
    _annotate_ends(
        ax,
        [
            (label_pyscf, rows[-1]["distance"], rows[-1]["embedding_correction_pyscf_plot_kcal"]),
            (label_siesta, rows[-1]["distance"], rows[-1]["embedding_correction_siesta_plot_kcal"]),
        ],
    )
    _style_axis(
        ax,
        xlabel="R(C-N) (A)",
        ylabel="E_embed - E_low_total, centred (kcal/mol)",
    )
    ax.set_title("High-level embedding correction (each series mean-centred)", loc="left", fontsize=11)
    fig.text(
        0.125,
        0.005,
        f"Raw means differ by {(mean_corr_s - mean_corr_p):.1f} kcal/mol "
        f"({mean_corr_p:.1f} vs {mean_corr_s:.1f}); compare the shape, not the offset.",
        color=TEXT_MUTED,
        fontsize=7.5,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    for fmt in formats:
        fig.savefig(out_dir / f"embedding_correction.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Differences: raw and mean-centred, 2x2 small multiples.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    panels = (
        (axes[0, 0], delta_rel, "delta E_rel, raw", "SIESTA - PySCF (kcal/mol)", False),
        (
            axes[0, 1],
            [value - mean_rel for value in delta_rel],
            "delta E_rel, mean-centred",
            "centred delta E_rel (kcal/mol)",
            True,
        ),
        (axes[1, 0], delta_corr, "delta embedding correction, raw", "SIESTA - PySCF (kcal/mol)", False),
        (
            axes[1, 1],
            [value - mean_corr for value in delta_corr],
            "delta embedding correction, mean-centred",
            "centred delta corr (kcal/mol)",
            True,
        ),
    )
    for ax, values, title, ylabel, centred in panels:
        ax.plot(x, values, color=SERIES[1] if centred else SERIES[0], linewidth=2.0)
        ax.axhline(0.0, color=AXIS, linewidth=0.8, linestyle="--")
        ax.scatter(
            x,
            values,
            s=30,
            color=SERIES[1] if centred else SERIES[0],
            edgecolor=SURFACE,
            linewidth=1.0,
            zorder=3,
        )
        _style_axis(ax, xlabel="", ylabel=ylabel)
        ax.set_title(title, loc="left", fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("R(C-N) (A)", color=TEXT_SECONDARY, fontsize=9.5)
    fig.suptitle(
        "Backend difference (basis/pseudopotential offset removed in the right column)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.125,
        0.005,
        f"Mean raw offsets: delta E_rel = {mean_rel:.3f} kcal/mol, "
        f"delta corr = {mean_corr:.3f} kcal/mol.",
        color=TEXT_MUTED,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    for fmt in formats:
        fig.savefig(out_dir / f"differences.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Outer-loop iteration count, useful to spot a non-converged point.
    if any(row["cycles_pyscf"] is not None for row in rows) or any(
        row["cycles_siesta"] is not None for row in rows
    ):
        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        for key, label, color, unconv_key in (
            ("cycles_pyscf", label_pyscf, SERIES[0], "converged_pyscf"),
            ("cycles_siesta", label_siesta, SERIES[1], "converged_siesta"),
        ):
            xs = [row["distance"] for row in rows if row[key] is not None]
            ys = [row[key] for row in rows if row[key] is not None]
            ax.plot(xs, ys, color=color, linewidth=2.0, label=label)
            ax.scatter(
                xs,
                ys,
                s=30,
                color=color,
                edgecolor=SURFACE,
                linewidth=1.0,
                zorder=3,
            )
        _style_axis(ax, xlabel="R(C-N) (A)", ylabel="Outer-loop cycles")
        ax.set_title("Configuration-recovery iterations", loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(out_dir / f"cycles.{fmt}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyscf-dir", type=Path, required=True)
    parser.add_argument("--siesta-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reference",
        default="max",
        help="Reference separation: 'max', 'min', or a distance in Angstrom.",
    )
    parser.add_argument("--pyscf-label", default=None)
    parser.add_argument("--siesta-label", default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pyscf, pyscf_meta = _read_summary(args.pyscf_dir)
    siesta, siesta_meta = _read_summary(args.siesta_dir)
    rows, metrics = _build_comparison(pyscf, siesta, args.reference)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label_pyscf = args.pyscf_label or _label(pyscf_meta, "pyscf")
    label_siesta = args.siesta_label or _label(siesta_meta, "siesta")
    _write_csv(out_dir / "comparison.csv", rows)
    (out_dir / "comparison.json").write_text(
        json.dumps(
            {
                "labels": {"pyscf": label_pyscf, "siesta": label_siesta},
                "metrics": metrics,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    _write_markdown(
        out_dir / "comparison_summary.md",
        rows=rows,
        metrics=metrics,
        label_pyscf=label_pyscf,
        label_siesta=label_siesta,
    )
    if not args.no_plots:
        _plot(
            rows,
            out_dir=out_dir,
            label_pyscf=label_pyscf,
            label_siesta=label_siesta,
            metrics=metrics,
            formats=tuple(args.formats),
        )
    print(
        f"wrote {out_dir / 'comparison.csv'} and {out_dir / 'comparison_summary.md'}\n"
        f"trend r(total) = {metrics['pearson_r_total']:.4f}, "
        f"r(low-level) = {metrics['pearson_r_low_level_total']:.4f}, "
        f"r(E_rel) = {metrics['pearson_r_rel_energy']:.4f}, "
        f"r(correction) = {metrics['pearson_r_embedding_correction']:.4f}; "
        f"RMSE delta E_rel: {metrics['rmse_delta_rel_kcal']:.3f} kcal/mol raw, "
        f"{metrics['rmse_delta_rel_centered_kcal']:.3f} mean-centred; "
        f"delta corr: {metrics['rmse_delta_embedding_correction_kcal']:.3f} raw, "
        f"{metrics['rmse_delta_embedding_correction_centered_kcal']:.3f} mean-centred."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
