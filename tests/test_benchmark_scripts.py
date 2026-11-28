# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit checks for the structural-scan benchmark helpers.

These do not run either quantum-chemistry backend; they pin the summary/merge and
relative-energy arithmetic so a plotting change cannot silently reinterpret the
scan.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("benchmark_runner", "scripts/benchmarks/run_structural_scan.py")
compare = _load("benchmark_compare", "scripts/benchmarks/compare_structural_scan.py")

SUMMARY_FIELDS = [
    "stem",
    "distance",
    "backend",
    "status",
    "returncode",
    "cycles",
    "converged",
    "cas_norb",
    "cas_nelec",
    "norb",
    "nelec_alpha",
    "nelec_beta",
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
    "rdm1_trace",
    "wall_time_s",
    "command",
]


def _write_run(path: Path, *, backend: str, energies: dict[float, float], max_cycles: int = 1):
    path.mkdir(parents=True)
    with (path / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for distance, energy in energies.items():
            writer.writerow(
                {
                    "stem": f"{int(distance * 10):02d}",
                    "distance": distance,
                    "backend": backend,
                    "status": "ok",
                    "returncode": 0,
                    "cycles": 0,
                    "converged": False,
                    "E_low_total": -100.0 + distance,
                    "E_low_A": -10.0,
                    "E_high_A": -9.5,
                    "E_solver": 1.0,
                    "e_core": 2.0,
                    "correction": 0.0,
                    "embedding_energy": energy,
                    "projector_leak": 0.0,
                    "rdm1_trace": 4.0,
                    "wall_time_s": 1.0,
                    "command": "workflow",
                }
            )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "backend": backend,
                    "xc_ll": "PBE",
                    "xc_hl": "HF",
                    "basis": "test",
                    "max_cycles": max_cycles,
                },
                "rows": [],
            }
        )
    )


def test_comparison_uses_the_common_reference_and_relative_energies(tmp_path):
    pyscf = tmp_path / "pyscf"
    siesta = tmp_path / "siesta"
    _write_run(pyscf, backend="pyscf", energies={1.0: -100.0, 2.0: -101.0})
    _write_run(siesta, backend="siesta", energies={1.0: -30.0, 2.0: -30.5})
    p_rows, _ = compare._read_summary(pyscf)
    s_rows, _ = compare._read_summary(siesta)
    rows, metrics = compare._build_comparison(p_rows, s_rows, "max")

    assert [row["distance"] for row in rows] == [1.0, 2.0]
    assert rows[0]["rel_energy_pyscf_ha"] == pytest.approx(1.0)
    assert rows[0]["rel_energy_siesta_ha"] == pytest.approx(0.5)
    assert rows[0]["delta_rel_ha"] == pytest.approx(-0.5)
    assert metrics["reference_distance"] == 2.0
    assert metrics["single_shot_pyscf"] is True
    assert metrics["rmse_delta_rel_ha"] == pytest.approx(0.5 / 2**0.5)


def test_single_shot_rows_are_not_counted_as_nonconverged(tmp_path):
    run = tmp_path / "single"
    _write_run(run, backend="pyscf", energies={1.0: -100.0, 2.0: -101.0}, max_cycles=1)
    rows, _meta = compare._read_summary(run)
    assert all(row["single_shot"] for row in rows)
    assert all(row["converged"] is False for row in rows)


def test_partial_summary_merge_preserves_other_geometries(tmp_path):
    out = tmp_path / "scan"
    out.mkdir()
    with (out / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=runner.SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "stem": "08",
                "distance": 0.8,
                "backend": "pyscf",
                "status": "ok",
            }
        )
    rows = runner._load_summary_rows(out)
    rows["09"] = {
        "stem": "09",
        "distance": 0.9,
        "backend": "pyscf",
        "status": "ok",
    }
    runner._write_summary(out, rows, {"backend": "pyscf"})
    with (out / "summary.csv").open(newline="") as fh:
        written = list(csv.DictReader(fh))
    assert [row["stem"] for row in written] == ["08", "09"]
