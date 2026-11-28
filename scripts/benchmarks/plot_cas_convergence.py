# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Plot active-space convergence for structural-scan summaries.

Each ``CAS=DIR`` argument names a directory produced by
:mod:`run_structural_scan` with ``--cas-size CAS CAS``.  The script reports, for
selected C-N distances, the shift of the embedding total and of the embedding
correction relative to the largest active space in the supplied set.

Example::

    uv run python scripts/benchmarks/plot_cas_convergence.py \
        --pyscf 4=/tmp/scan-pyscf-cas4 --pyscf 6=/tmp/scan-pyscf-cas6 \
        --pyscf 8=/tmp/scan-pyscf-cas8 --pyscf 10=/tmp/scan-pyscf-cas10 \
        --siesta 4=/tmp/scan-siesta-cas4 --siesta 6=/tmp/scan-siesta-cas6 \
        --siesta 8=/tmp/scan-siesta-cas8 --siesta 10=/tmp/scan-siesta-cas10 \
        --out-dir analysis/cas-convergence
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

HARTREE_TO_KCAL = 627.5094740631
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def _read_run(directory: Path) -> tuple[dict[float, float], dict[float, float]]:
    path = directory / "summary.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}; run run_structural_scan.py first")
    energies: dict[float, float] = {}
    corrections: dict[float, float] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") not in {"ok", "cached"}:
                continue
            try:
                distance = round(float(row["distance"]), 2)
                energy = float(row["embedding_energy"])
                low_total = float(row["E_low_total"])
            except (TypeError, ValueError):
                continue
            energies[distance] = energy
            corrections[distance] = energy - low_total
    if not energies:
        raise SystemExit(f"{path} has no successful rows")
    return energies, corrections


def _parse_runs(values: list[str]) -> dict[int, Path]:
    runs: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected CAS=DIR, got {value!r}")
        cas, directory = value.split("=", 1)
        runs[int(cas)] = Path(directory).expanduser().resolve()
    return runs


def _convergence_rows(
    backend: str,
    runs: dict[int, Path],
    distances: list[float],
) -> list[dict[str, Any]]:
    data = {cas: _read_run(path) for cas, path in sorted(runs.items())}
    max_cas = max(data)
    rows: list[dict[str, Any]] = []
    for distance in distances:
        reference = data[max_cas][0].get(distance)
        reference_corr = data[max_cas][1].get(distance)
        if reference is None or reference_corr is None:
            raise SystemExit(f"CAS({max_cas},{max_cas}) has no R={distance} row")
        for cas, (energies, corrections) in sorted(data.items()):
            if distance not in energies:
                continue
            delta = energies[distance] - reference
            delta_corr = corrections[distance] - reference_corr
            rows.append(
                {
                    "backend": backend,
                    "cas": cas,
                    "distance": distance,
                    "embedding_energy": energies[distance],
                    "correction": corrections[distance],
                    "delta_total_ha": delta,
                    "delta_total_kcal": delta * HARTREE_TO_KCAL,
                    "delta_correction_ha": delta_corr,
                    "delta_correction_kcal": delta_corr * HARTREE_TO_KCAL,
                }
            )
    return rows


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


def _plot(
    rows: list[dict[str, Any]],
    distances: list[float],
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["system-ui", "DejaVu Sans"],
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
    backends = [name for name in ("pyscf", "siesta") if any(r["backend"] == name for r in rows)]
    fig, axes = plt.subplots(len(backends), 2, figsize=(9.2, 3.8 * len(backends)), squeeze=False)
    for irow, backend in enumerate(backends):
        for icol, (field, title) in enumerate(
            (
                ("delta_total_kcal", "Embedding total"),
                ("delta_correction_kcal", "Embedding correction"),
            )
        ):
            ax = axes[irow, icol]
            for j, distance in enumerate(distances):
                subset = [
                    row
                    for row in rows
                    if row["backend"] == backend and row["distance"] == distance
                ]
                subset.sort(key=lambda row: row["cas"])
                if not subset:
                    continue
                x = [row["cas"] for row in subset]
                y = [row[field] for row in subset]
                color = SERIES[j % len(SERIES)]
                ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=5.5, label=f"R = {distance:.1f} A")
                # Put the direct label at the first CAS point, where the series
                # are separated (all series meet at the largest CAS).
                ax.annotate(
                    f"R={distance:.1f}",
                    xy=(x[0], y[0]),
                    xytext=(8, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color=TEXT_SECONDARY,
                )
            ax.axhline(0.0, color=AXIS, linewidth=0.8, linestyle="--")
            _style_axis(ax, xlabel="Active-space size (CAS n,n)", ylabel="Shift vs largest CAS (kcal/mol)")
            display = {"pyscf": "PySCF", "siesta": "SIESTA"}.get(backend, backend)
            ax.set_title(f"{display}: {title}", loc="left", fontsize=9.5)
            ax.set_xticks(sorted({row["cas"] for row in rows if row["backend"] == backend}))
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="best")
    fig.suptitle(
        "Active-space convergence of the embedding energy",
        x=0.01,
        ha="left",
        fontsize=11,
        color=TEXT_PRIMARY,
    )
    fig.text(
        0.01,
        0.005,
        "Each point is shifted to the largest supplied CAS at the same geometry; "
        "a flat line means the active space is converged.",
        color=TEXT_MUTED,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    for fmt in formats:
        fig.savefig(out_dir / f"cas_convergence.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyscf", action="append", default=[], metavar="CAS=DIR")
    parser.add_argument("--siesta", action="append", default=[], metavar="CAS=DIR")
    parser.add_argument("--distances", nargs="+", type=float, default=[1.3, 2.5])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    distances = sorted(round(d, 2) for d in args.distances)
    rows: list[dict[str, Any]] = []
    for backend, values in (("pyscf", args.pyscf), ("siesta", args.siesta)):
        runs = _parse_runs(values)
        if runs:
            rows.extend(_convergence_rows(backend, runs, distances))
    if not rows:
        raise SystemExit("pass at least one --pyscf CAS=DIR or --siesta CAS=DIR")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out_dir / "cas_convergence.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, distances, out_dir, tuple(args.formats))
    print(f"wrote {out_dir / 'cas_convergence.csv'} and plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
