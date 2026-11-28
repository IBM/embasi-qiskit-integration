# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Run the embedding workflow over a directory of related geometries.

The script is a reproducible driver for the structural scans used to compare
PySCF and SIESTA at the *protocol* level: same active atoms, active-space
selection, low/high-level method, and solver, with each backend's own basis.

Each geometry gets its own working directory, log, diagnostics CSV, and
one-line energy file, so a long scan is resumable and the EmbASI/SIESTA scratch
files never land in the repository root.  The final per-geometry row is also
collected into ``summary.csv`` for :mod:`compare_structural_scan`.

Example (PySCF, PBE low level, HF high level, exact FCI reference)::

    uv run python scripts/benchmarks/run_structural_scan.py \
        --backend pyscf \
        --data-dir /path/to/geometries \
        --out-dir /tmp/scan-pyscf \
        --solver fci --max-cycles 30 --basis sto-3g

Example (SIESTA, same protocol; export ``ASI_LIB_PATH`` and
``SIESTA_PP_PATH`` first)::

    uv run python scripts/benchmarks/run_structural_scan.py \
        --backend siesta \
        --data-dir /path/to/geometries \
        --out-dir /tmp/scan-siesta \
        --solver fci --max-cycles 30 --siesta-basis SZ

Example (PySCF, AO-defined active space, fixed CAS(4,4); see
``benchmarks/README.md`` for the SIESTA and CAS(6,6) variants)::

    uv run python scripts/benchmarks/run_structural_scan.py \
        --backend pyscf \
        --out-dir /tmp/scan-pyscf-ao-cas4 \
        --selector ao-avas --cas-size 4 4 --n-frozen-occ 0 \
        --avas-ao-labels "N 2p" "C 2p" \
        --solver fci --max-cycles 1 --basis sto-3g

This is not an exact cross-code comparison: PySCF is all-electron/Gaussian and
SIESTA is pseudopotential/NAO.  Keep absolute totals out of any comparison and
use relative energies or embedding corrections; the exact same-pseudopotential
checks live in the test suite instead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / "scripts" / "embedding_workflow.py"
SUMMARY_COLUMNS = [
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


def _compact_list(values: list[int]) -> str:
    return "[" + ",".join(str(int(v)) for v in values) + "]"


def _parse_distance(stem: str) -> float | None:
    """Distance label from a stem such as ``08`` (0.8 A) or ``2.4`` (2.4 A)."""
    try:
        value = float(stem)
    except ValueError:
        return None
    if stem.isdigit():
        return value / 10.0
    return value


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _last_diagnostics_row(path: Path) -> dict[str, str] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-1] if rows else None


def _geometries(data_dir: Path, pattern: str, requested: list[str] | None) -> list[Path]:
    if requested:
        paths = []
        for name in requested:
            path = data_dir / (name if Path(name).suffix else f"{name}.inp")
            if not path.is_file():
                raise SystemExit(f"geometry not found: {path}")
            paths.append(path)
        return paths
    paths = sorted(p for p in data_dir.glob(pattern) if p.is_file())
    if not paths:
        raise SystemExit(f"no geometries matching {pattern!r} in {data_dir}")
    return paths


def _build_command(
    args: argparse.Namespace, xyz: Path, energy_path: Path, diag_path: Path
) -> list[str]:
    cmd = [
        sys.executable,
        str(WORKFLOW),
        "--xyz",
        str(xyz),
        "--n_atoms",
        str(args.n_atoms),
        "--active_atoms",
        _compact_list(args.active_atoms),
        "--selector",
        args.selector,
        "--n_shells",
        str(args.n_shells),
        "--apc_fixed",
        str(bool(args.apc_fixed)),
        "--max_cycles",
        str(args.max_cycles),
        "--xc_ll",
        args.xc_ll,
        "--xc_hl",
        args.xc_hl,
        "--wf_in_dft_energy",
        args.wf_in_dft_energy,
        "--a_nmos",
        str(args.a_nmos),
        "--apc_max_size",
        _compact_list(args.apc_max_size),
        "--mix_alpha",
        str(args.mix_alpha),
        "--diis",
        str(bool(args.diis)),
        "--spin",
        str(args.spin),
        "--solver",
        args.solver,
        "--output_path",
        str(energy_path),
        "--diagnostics_csv",
        str(diag_path),
    ]
    if args.charge is not None:
        cmd += ["--charge", str(args.charge)]
    if args.scf_max_cycle is not None:
        cmd += ["--scf_max_cycle", str(args.scf_max_cycle)]
    if args.scf_chkfile_out is not None and args.backend == "pyscf":
        cmd += ["--scf_chkfile_out", str(args.scf_chkfile_out)]
    if args.n_frozen_occ:
        cmd += ["--n_frozen_occ", str(args.n_frozen_occ)]
    if args.n_virtual is not None:
        cmd += ["--n_virtual", str(args.n_virtual)]
    if args.selector in ("avas", "ao-avas", "correlation-avas"):
        if args.avas_ao_labels:
            # pydantic-settings parses list-valued options as JSON.
            cmd += ["--avas_ao_labels", json.dumps(list(args.avas_ao_labels))]
        if args.avas_max_size is not None:
            cmd += ["--avas_max_size", _compact_list(args.avas_max_size)]
        if args.avas_threshold is not None:
            cmd += ["--avas_threshold", str(args.avas_threshold)]
    if args.scf_newton and args.backend == "pyscf":
        cmd += ["--scf_newton", "True"]
    if args.backend == "siesta":
        cmd += [
            "--qm_code",
            "siesta",
            "--siesta_basis",
            args.siesta_basis,
            "--siesta_mesh_cutoff",
            str(args.siesta_mesh_cutoff),
            "--siesta_energy_shift",
            str(args.siesta_energy_shift),
        ]
    else:
        cmd += ["--qm_code", "pyscf", "--basis", args.basis]
    if args.dump_active_mos:
        cmd += ["--dump_active_mos", str(energy_path.parent / "active_mos.npz")]
    return cmd


def _row_from_final(
    *,
    args: argparse.Namespace,
    stem: str,
    distance: float | None,
    status: str,
    returncode: int | None,
    final: dict[str, str] | None,
    wall_time_s: float,
    command: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stem": stem,
        "distance": distance if distance is not None else "",
        "backend": args.backend,
        "status": status,
        "returncode": "" if returncode is None else returncode,
        "cycles": "",
        "converged": "",
        "cas_norb": "",
        "cas_nelec": "",
        "norb": "",
        "nelec_alpha": "",
        "nelec_beta": "",
        "E_solver": "",
        "E_low_total": "",
        "E_low_A": "",
        "E_high_A": "",
        "e_core": "",
        "correction": "",
        "embedding_energy": "",
        "embedding_correction": "",
        "high_level_term": "",
        "projector_leak": "",
        "rdm1_trace": "",
        "wall_time_s": round(wall_time_s, 3),
        "command": " ".join(command),
    }
    if not final:
        return row

    for key in (
        "cycle",
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
        "projector_leak",
        "rdm1_trace",
    ):
        if key in final:
            row["cycles" if key == "cycle" else key] = final[key]
    row["converged"] = final.get("converged", "")
    if final.get("geometry_parameter"):
        row["distance"] = _float_or_none(final["geometry_parameter"]) or row["distance"]

    e_emb = _float_or_none(final.get("embedding_energy"))
    e_low_total = _float_or_none(final.get("E_low_total"))
    e_low_a = _float_or_none(final.get("E_low_A"))
    e_high_a = _float_or_none(final.get("E_high_A"))
    if e_emb is not None and e_low_total is not None:
        row["embedding_correction"] = e_emb - e_low_total
    if e_high_a is not None and e_low_a is not None:
        row["high_level_term"] = e_high_a - e_low_a
    return row


def _load_summary_rows(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Rows already in ``summary.csv``, keyed by geometry stem.

    Partial reruns (``--geometries 19``) must not drop the other points: a scan
    is commonly resumed one stretched geometry at a time with different SCF
    settings.
    """
    path = out_dir / "summary.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as fh:
        return {row["stem"]: dict(row) for row in csv.DictReader(fh) if row.get("stem")}


def _write_summary(
    out_dir: Path,
    rows: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    ordered = sorted(
        rows.values(),
        key=lambda row: (_float_or_none(str(row.get("distance"))) or 0.0, row["stem"]),
    )
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})
    with (out_dir / "summary.json").open("w") as fh:
        json.dump({"metadata": metadata, "rows": ordered}, fh, indent=2, sort_keys=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "data" / "butyronitrile",
        help="Directory holding the geometries (default: the bundled butyronitrile scan).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("pyscf", "siesta"), required=True)
    parser.add_argument(
        "--geometries",
        nargs="+",
        default=None,
        help="Stems or filenames (default: every file matching --pattern).",
    )
    parser.add_argument("--pattern", default="*.inp")
    parser.add_argument("--solver", choices=("fci", "sqd"), default="fci")
    parser.add_argument("--max-cycles", type=int, default=30)
    parser.add_argument("--xc-ll", default="PBE")
    parser.add_argument("--xc-hl", default="HF")
    parser.add_argument(
        "--wf-in-dft-energy",
        choices=("first-order", "reference"),
        default="first-order",
        help="Reference protocol (Rossmannek et al.) is single-shot; use --max-cycles 1.",
    )
    parser.add_argument("--basis", default="sto-3g", help="PySCF high-level basis.")
    parser.add_argument("--siesta-basis", default="DZP")
    parser.add_argument("--siesta-mesh-cutoff", type=float, default=200.0)
    parser.add_argument(
        "--siesta-energy-shift",
        type=float,
        default=0.1,
        help="SIESTA PAO.EnergyShift in eV (default 0.1; smaller is more diffuse).",
    )
    parser.add_argument("--n-atoms", type=int, default=12)
    parser.add_argument("--active-atoms", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--selector", default="apc-concentric")
    parser.add_argument("--n-shells", type=int, default=1)
    parser.add_argument("--apc-fixed", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--a-nmos", type=int, default=7)
    parser.add_argument(
        "--n-frozen-occ",
        type=int,
        default=0,
        help="Freeze this many occupied orbitals (a fixed-space control; also the "
        "core floor excluded from the ao-avas AO ranking).",
    )
    parser.add_argument(
        "--n-virtual",
        type=int,
        default=None,
        help="Cap active virtuals (a fixed-space control, e.g. selector=mulliken).",
    )
    parser.add_argument(
        "--avas-ao-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Target fragment AOs for --selector ao-avas (e.g. 'N 2p' 'C 2p').  "
        "Omit on SIESTA, whose ASI basis descriptors report no shell labels: every AO "
        "of the active fragment is then used.",
    )
    parser.add_argument(
        "--avas-max-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("NELEC", "NORB"),
        help="Fixed (nelec, norb) budget for --selector ao-avas/avas.  --cas-size is "
        "used when this is omitted and the selector is avas/ao-avas.",
    )
    parser.add_argument(
        "--avas-threshold",
        type=float,
        default=None,
        help="AVAS weight threshold for avas/ao-avas (default: the workflow's 0.2).",
    )
    parser.add_argument(
        "--dump-active-mos",
        action="store_true",
        help="Write each geometry's --dump_active_mos .npz (active-space character "
        "diagnostic) next to its diagnostics.csv.",
    )
    parser.add_argument(
        "--cas-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("NELEC", "NORB"),
        help="High-level CAS size, e.g. '--cas-size 6 6' (default: 4 4).  Routed to "
        "--avas-max-size for avas/ao-avas and to --apc-max-size otherwise.",
    )
    parser.add_argument("--apc-max-size", type=int, nargs=2, default=[4, 4])
    parser.add_argument("--mix-alpha", type=float, default=0.08)
    parser.add_argument("--diis", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--charge", type=int, default=None)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--scf-max-cycle", type=int, default=None)
    parser.add_argument(
        "--scf-chkfile-out",
        type=Path,
        default=None,
        help="PySCF only: SCF checkpoint path (the last EmbASI layer written wins).",
    )
    parser.add_argument(
        "--scf-newton",
        action="store_true",
        help="PySCF only: second-order SCF for stretched geometries with small gaps.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment entry for the workflow (repeatable; e.g. ASI_LIB_PATH=...).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Geometries to run concurrently (each SIESTA process uses ~1-2 GB).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _run_one(
    *,
    args: argparse.Namespace,
    xyz: Path,
    geom_dir: Path,
    energy_path: Path,
    diag_path: Path,
    log_path: Path,
    command: list[str],
    env: dict[str, str],
) -> tuple[str, dict[str, Any], bool]:
    """Run one geometry, returning ``(stem, summary row, failed)``."""
    stem = xyz.stem
    print(f"[{stem}] running {args.backend} ...", flush=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        proc = subprocess.run(
            command,
            cwd=geom_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - started
    final = _last_diagnostics_row(diag_path)
    failed = proc.returncode != 0 or final is None
    if failed:
        print(f"[{stem}] FAILED (exit {proc.returncode}); see {log_path}", file=sys.stderr)
    else:
        emb = _float_or_none(final.get("embedding_energy")) if final else None
        print(
            f"[{stem}] ok in {elapsed:.1f}s" + (f", E = {emb:.8f} Ha" if emb is not None else ""),
            flush=True,
        )
    row = _row_from_final(
        args=args,
        stem=stem,
        distance=_parse_distance(stem),
        status="failed" if failed else "ok",
        returncode=proc.returncode,
        final=final,
        wall_time_s=elapsed,
        command=command,
    )
    return stem, row, failed


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.wf_in_dft_energy == "reference" and args.selector == "ao-avas":
        raise SystemExit(
            "--wf-in-dft-energy reference builds its own canonical concentric active "
            "space from --n_frozen_occ/--n_virtual and ignores --selector; use "
            "--selector none for the reference protocol, or drop the reference mode "
            "to run ao-avas."
        )
    if args.cas_size is not None:
        size = [int(args.cas_size[0]), int(args.cas_size[1])]
        if args.selector in ("avas", "ao-avas", "correlation-avas"):
            if args.avas_max_size is None:
                args.avas_max_size = size
        else:
            args.apc_max_size = size
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    geometries = _geometries(data_dir, args.pattern, args.geometries)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONUNBUFFERED"] = "1"
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value

    if args.backend == "siesta":
        missing = [key for key in ("ASI_LIB_PATH", "SIESTA_PP_PATH") if not env.get(key)]
        if missing:
            print(
                "warning: SIESTA usually needs " + ", ".join(missing) + " in the environment",
                file=sys.stderr,
            )

    if args.selector in ("avas", "ao-avas", "correlation-avas") and args.avas_max_size is not None:
        cas_size = list(args.avas_max_size)
    else:
        cas_size = list(args.apc_max_size)
    metadata = {
        "backend": args.backend,
        "data_dir": str(data_dir),
        "workflow": str(WORKFLOW),
        "solver": args.solver,
        "xc_ll": args.xc_ll,
        "xc_hl": args.xc_hl,
        "basis": args.basis if args.backend == "pyscf" else args.siesta_basis,
        "max_cycles": args.max_cycles,
        "active_atoms": args.active_atoms,
        "selector": args.selector,
        "avas_ao_labels": args.avas_ao_labels,
        "avas_max_size": (None if args.avas_max_size is None else list(args.avas_max_size)),
        "n_frozen_occ": args.n_frozen_occ,
        "cas_size": cas_size,
    }
    rows = _load_summary_rows(out_dir)
    pending: list[tuple[Path, Path, Path, Path, Path, list[str]]] = []
    for xyz in geometries:
        stem = xyz.stem
        geom_dir = out_dir / stem
        geom_dir.mkdir(parents=True, exist_ok=True)
        energy_path = geom_dir / "energy.csv"
        diag_path = geom_dir / "diagnostics.csv"
        log_path = geom_dir / "run.log"
        command = _build_command(args, xyz.resolve(), energy_path, diag_path)

        if args.dry_run:
            print(f"[{stem}] cd {geom_dir} && " + " ".join(command))
            continue

        if energy_path.exists() and diag_path.exists() and not args.overwrite:
            final = _last_diagnostics_row(diag_path)
            rows[stem] = _row_from_final(
                args=args,
                stem=stem,
                distance=_parse_distance(stem),
                status="cached",
                returncode=0,
                final=final,
                wall_time_s=0.0,
                command=command,
            )
            print(f"[{stem}] cached")
            continue

        if energy_path.exists():
            energy_path.unlink()
        if diag_path.exists():
            diag_path.unlink()
        pending.append((xyz, geom_dir, energy_path, diag_path, log_path, command))

    if args.dry_run:
        return 0

    failures = 0
    if args.jobs <= 1:
        for xyz, geom_dir, energy_path, diag_path, log_path, command in pending:
            stem, row, failed = _run_one(
                args=args,
                xyz=xyz,
                geom_dir=geom_dir,
                energy_path=energy_path,
                diag_path=diag_path,
                log_path=log_path,
                command=command,
                env=env,
            )
            rows[stem] = row
            failures += int(failed)
            _write_summary(out_dir, rows, metadata)
            if failed and not args.continue_on_error:
                break
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(
                    _run_one,
                    args=args,
                    xyz=xyz,
                    geom_dir=geom_dir,
                    energy_path=energy_path,
                    diag_path=diag_path,
                    log_path=log_path,
                    command=command,
                    env=env,
                )
                for xyz, geom_dir, energy_path, diag_path, log_path, command in pending
            ]
            for future in as_completed(futures):
                stem, row, failed = future.result()
                rows[stem] = row
                failures += int(failed)
                _write_summary(out_dir, rows, metadata)

    _write_summary(out_dir, rows, metadata)
    print(f"wrote {out_dir / 'summary.csv'} ({len(rows)} geometries, {failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
