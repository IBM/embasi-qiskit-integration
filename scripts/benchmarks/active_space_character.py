# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Label the active-space orbitals of one or more ``--dump_active_mos`` files.

The butyronitrile SIESTA-vs-PySCF comparison selects a CAS(4,4) by AO weight on the
active fragment.  Column indices alone do not say *which* physical orbitals were
chosen, so this script reads the opt-in dumps written by
``EmbeddingWorkflow.dump_active_mos`` (embedded coefficients, overlap, AO labels,
solver 1-RDM) and reports:

* Mulliken populations of every active orbital on the fragment's ``n l`` shells
  (N/C 2s, 2p, 3d*, ...), and the projector weight on the combined target set;
* the C-N bond population and the p-orbital density tensor, which separates bonding
  from antibonding and sigma from pi character;
* the same quantities for the *natural* orbitals of the active-space FCI 1-RDM,
  whose occupations show which active pair actually carries the static correlation.

SIESTA dumps include labels because the backend resolves ``.ORB_INDX`` lazily; for a
dump written before that change, pass ``--orb-indx`` with the layer's file.

Usage::

    .venv/bin/python scripts/benchmarks/active_space_character.py \
        /tmp/aschar/siesta-08/active_mos.npz /tmp/aschar/pyscf-08/active_mos.npz \
        --out-csv /tmp/aschar/character.csv

Not a benchmark runner: it never launches an embedding, and it writes only the paths
given on the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _meta(record: dict) -> dict:
    raw = record.get("meta")
    if raw is None:
        return {}
    try:
        return json.loads(raw.item() if getattr(raw, "shape", ()) == () else str(raw))
    except (ValueError, AttributeError):
        return {}


def _labels_from_orb_indx(path: Path) -> tuple[list[tuple[int, str, str, str]], np.ndarray]:
    from embasi_qiskit_integration.backends.siesta import (
        ao_labels_from_orbitals,
        basis_atoms_from_orbitals,
        read_orb_indx,
    )

    orbitals = read_orb_indx(path)
    return ao_labels_from_orbitals(orbitals), basis_atoms_from_orbitals(orbitals)


def load_dump(path: Path, orb_indx: Path | None = None) -> dict:
    """Read one dump and return a normalised record (arrays + label table)."""
    with np.load(path, allow_pickle=False) as data:
        record = {key: np.asarray(data[key]) for key in data.files}
    meta = _meta(record)
    coeff = record["coeff"]
    overlap = record["overlap"]
    nao = coeff.shape[0]
    if overlap.shape != (nao, nao):
        raise SystemExit(f"{path}: overlap has shape {overlap.shape}, expected ({nao}, {nao})")

    labels = None
    if {"label_atom", "label_element", "label_shell", "label_m"} <= record.keys():
        if len(record["label_element"]) == nao:
            labels = [
                (
                    int(record["label_atom"][i]),
                    str(record["label_element"][i]),
                    str(record["label_shell"][i]),
                    str(record["label_m"][i]),
                )
                for i in range(nao)
            ]
    atom_map = record.get("ao_atom_map")
    if labels is None and orb_indx is not None:
        labels, mapped = _labels_from_orb_indx(orb_indx)
        if len(labels) != nao:
            raise SystemExit(f"{path}: {orb_indx} has {len(labels)} AOs but the dump has {nao}")
        if atom_map is not None and not np.array_equal(mapped, atom_map):
            raise SystemExit(f"{path}: {orb_indx} atom map disagrees with the dump")

    return {
        "path": str(path),
        "meta": meta,
        "coeff": coeff,
        "overlap": overlap,
        "active": np.asarray(record["active"], dtype=int),
        "inactive": np.asarray(record["inactive"], dtype=int),
        "n_occ": int(record["n_occ"]),
        "n_occ_b": int(record.get("n_occ_b", -1)),
        "rdm1": record.get("rdm1_active"),
        "ao_atom_map": atom_map,
        "labels": labels,
        "atom_symbols": None
        if "atom_symbols" not in record
        else [str(x) for x in record["atom_symbols"]],
        "atom_coords": record.get("atom_coords"),
        "occ_weights": record.get("occ_weights"),
        "virt_weights": record.get("virt_weights"),
        # Opt-in AVAS correlation-window diagnostics (only present when the run used
        # --avas_energy_window): weights/columns of the kept virtuals and the full
        # weight-sorted candidate spectrum.
        "virt_selection_weights": record.get("virt_selection_weights"),
        "virt_selection_column": record.get("virt_selection_column"),
        "virt_candidate_column": record.get("virt_candidate_column"),
        "virt_candidate_energy": record.get("virt_candidate_energy"),
        # Extended diagnostic (optional): orbital energies and the downfolded
        # active-space Hamiltonian, when the dump was written with them.
        "orbital_energy": record.get("orbital_energy"),
        "active_energy": record.get("active_energy"),
        "h1": record.get("h1"),
        "h2": record.get("h2"),
        "ham_nelec": record.get("ham_nelec"),
        "ham_e_core": record.get("ham_e_core"),
    }


# --------------------------------------------------------------------------- #
# character analysis
# --------------------------------------------------------------------------- #
def _p_component(label_m: str) -> str | None:
    text = label_m.strip()
    if text and text[-1] in "xyz":
        return text[-1]
    return None


def _shell_core(shell: str) -> str:
    """``"3d*"`` -> ``"3d"``: the polarization star is not part of the n l label."""
    return shell.rstrip("*")


def _groups(record: dict, fragment_atoms: list[int]) -> dict[str, np.ndarray]:
    """AO indices per ``"<element> <nl>"`` group on the fragment atoms."""
    labels = record["labels"]
    atom_map = record["ao_atom_map"]
    if labels is None or atom_map is None:
        return {}
    groups: dict[str, list[int]] = {}
    for mu, (atom, element, shell, _m) in enumerate(labels):
        if atom not in fragment_atoms:
            continue
        groups.setdefault(f"{element} {_shell_core(shell)}", []).append(mu)
    return {key: np.asarray(idx, dtype=int) for key, idx in sorted(groups.items())}


def _projector_weights(
    c: np.ndarray, s: np.ndarray, groups: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """``c^T M_group c`` per MO column: squared S-norm of its projection on the group."""
    out = {}
    for key, idx in groups.items():
        b = np.zeros((s.shape[0], idx.size))
        b[idx, np.arange(idx.size)] = 1.0
        s22 = b.T @ s @ b
        rhs = b.T @ s @ c
        try:
            weights = np.einsum("ij,ij->j", rhs, np.linalg.solve(s22, rhs))
        except np.linalg.LinAlgError:
            weights = np.einsum("ij,ij->j", rhs, np.linalg.pinv(s22) @ rhs)
        out[key] = np.clip(weights, 0.0, None)
    return out


def _bond_axis(record: dict, bond_atoms: tuple[int, int]) -> np.ndarray | None:
    coords = record.get("atom_coords")
    if coords is None or coords.shape[0] <= max(bond_atoms):
        return None
    axis = coords[bond_atoms[1]] - coords[bond_atoms[0]]
    norm = float(np.linalg.norm(axis))
    return None if norm < 1.0e-12 else axis / norm


def _p_tensor(c: np.ndarray, s: np.ndarray, labels, atom: int) -> np.ndarray | None:
    """3x3 Mulliken p-density tensor of each MO column on ``atom`` (x, y, z)."""
    comps = "xyz"
    ao_by_comp: dict[str, list[int]] = {x: [] for x in comps}
    for mu, (a, _el, shell, m) in enumerate(labels):
        if a != atom or not _shell_core(shell).endswith("p"):
            continue
        comp = _p_component(m)
        if comp is not None:
            ao_by_comp[comp].append(mu)
    if not any(ao_by_comp.values()):
        return None
    tensor = np.zeros((c.shape[1], 3, 3))
    for i, ci in enumerate(comps):
        for j, cj in enumerate(comps):
            if not ao_by_comp[ci] or not ao_by_comp[cj]:
                continue
            mu = np.asarray(ao_by_comp[ci])
            nu = np.asarray(ao_by_comp[cj])
            # Density matrix restricted to the atom's p subspace,
            # c_mu S_mu,nu c_nu; its trace is that subspace's occupation.
            tensor[:, i, j] = np.einsum("mi,mn,ni->i", c[mu], s[np.ix_(mu, nu)], c[nu])
    return 0.5 * (tensor + np.transpose(tensor, (0, 2, 1)))


def _mulliken(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-AO Mulliken population per MO column, shape ``(nao, nmo)``."""
    return c * (s @ c)


def _character(
    *,
    fragment_pop: float,
    bond_all: float,
    bond_p: float,
    p_parallel: float,
    p_perp: float,
    axis_angle: float | None,
) -> str:
    """A compact chemistry label from the numbers above (documented heuristics)."""
    if bond_all < -0.15 or bond_p < -0.10:
        bond = "antibonding"
    elif bond_all > 0.25 or bond_p > 0.15:
        bond = "bonding"
    else:
        bond = "nonbonding"
    if p_parallel > 2.0 * p_perp and p_parallel > 0.15:
        symmetry = "sigma"
    elif p_perp > 2.0 * p_parallel and p_perp > 0.15:
        symmetry = "pi"
    else:
        symmetry = ""
    if fragment_pop < 0.25:
        base = "environment"
    elif bond == "nonbonding" and fragment_pop > 0.75:
        base = "fragment-local"
    else:
        base = f"C-N {bond}"
    if symmetry and base.startswith("C-N"):
        base += f" {symmetry}"
    elif symmetry and base == "fragment-local":
        base = f"fragment-local {symmetry}"
    return base


def _orbital_rows(
    record: dict,
    coeff: np.ndarray,
    occupations: np.ndarray | None,
    kind: str,
    *,
    fragment_atoms: list[int],
    bond_atoms: tuple[int, int],
) -> list[dict]:
    s = record["overlap"]
    labels = record["labels"]
    groups = _groups(record, fragment_atoms)
    projectors = _projector_weights(coeff, s, groups) if groups else {}
    pops = _mulliken(coeff, s)
    axis = _bond_axis(record, bond_atoms)
    tensors = {}
    if labels is not None:
        for atom in dict.fromkeys(bond_atoms):
            tensors[atom] = _p_tensor(coeff, s, labels, atom)
    atom_map = record["ao_atom_map"]
    frag_mask = None if atom_map is None else np.isin(atom_map, fragment_atoms)
    bond_idx = [
        np.flatnonzero(atom_map == atom) if atom_map is not None else np.empty(0, dtype=int)
        for atom in bond_atoms
    ]
    # p-only bond population through the labelled p AOs of the two bond atoms.
    p_first = p_second = None
    if labels is not None and atom_map is not None:
        p_all = [
            mu
            for mu, (atom, _el, shell, _m) in enumerate(labels)
            if atom in bond_atoms and _shell_core(shell).endswith("p")
        ]
        p_first = [mu for mu in p_all if atom_map[mu] == bond_atoms[0]]
        p_second = [mu for mu in p_all if atom_map[mu] == bond_atoms[1]]

    rows = []
    has_bond = bond_idx[0].size > 0 and bond_idx[1].size > 0
    for k in range(coeff.shape[1]):
        c = coeff[:, k]
        row = {
            "source": record["path"],
            "kind": kind,
            "index": k,
            "occupation": None if occupations is None else float(occupations[k]),
            "fragment_pop": None if frag_mask is None else float(pops[frag_mask, k].sum()),
            "bond_pop_all": float(
                2.0 * c[bond_idx[0]] @ s[np.ix_(bond_idx[0], bond_idx[1])] @ c[bond_idx[1]]
            )
            if has_bond
            else None,
        }
        row["bond_pop_p"] = (
            float(2.0 * c[p_first] @ s[np.ix_(p_first, p_second)] @ c[p_second])
            if p_first and p_second
            else None
        )
        for key in groups:
            row[f"w[{key}]"] = float(projectors[key][k])
            row[f"pop[{key}]"] = float(pops[groups[key], k].sum())

        p_parallel = p_perp = 0.0
        angle = None
        if axis is not None and tensors:
            total = np.zeros((3, 3))
            for tensor in tensors.values():
                if tensor is not None:
                    total += tensor[k]
            total = 0.5 * (total + total.T)
            p_parallel = float(axis @ total @ axis)
            p_perp = float(np.trace(total) - p_parallel)
            eigenvalues, eigenvectors = np.linalg.eigh(total)
            principal = eigenvectors[:, int(np.argmax(eigenvalues))]
            cosang = abs(float(axis @ principal))
            angle = float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))
        row["p_parallel"] = p_parallel
        row["p_perp"] = p_perp
        row["p_axis_angle_deg"] = angle
        row["character"] = _character(
            fragment_pop=row["fragment_pop"] or 0.0,
            bond_all=row["bond_pop_all"] or 0.0,
            bond_p=row["bond_pop_p"] or 0.0,
            p_parallel=p_parallel,
            p_perp=p_perp,
            axis_angle=angle,
        )
        rows.append(row)
    return rows


def analyse(
    record: dict,
    *,
    fragment_atoms: list[int],
    bond_atoms: tuple[int, int],
    target_labels: list[str] | None = None,
    n_frozen_occ: int = 0,
) -> dict:
    coeff = record["coeff"]
    active = record["active"]
    c_active = coeff[:, active]
    rows = _orbital_rows(
        record, c_active, None, "active", fragment_atoms=fragment_atoms, bond_atoms=bond_atoms
    )
    for row, column in zip(rows, active):
        row["column"] = int(column)

    natural = []
    rdm = record.get("rdm1")
    if rdm is not None and rdm.shape == (active.size, active.size):
        rdm = 0.5 * (rdm + rdm.T)
        occupations, vectors = np.linalg.eigh(rdm)
        order = np.argsort(occupations)[::-1]
        c_nat = c_active @ vectors[:, order]
        natural = _orbital_rows(
            record,
            c_nat,
            occupations[order],
            "natural",
            fragment_atoms=fragment_atoms,
            bond_atoms=bond_atoms,
        )

    relabelled = []
    relabel_info = None
    if target_labels:
        relabelled, relabel_info = reselect_with_labels(
            record,
            target_labels,
            fragment_atoms=fragment_atoms,
            bond_atoms=bond_atoms,
            n_frozen_occ=n_frozen_occ,
        )
    return {
        "path": record["path"],
        "meta": record["meta"],
        "n_occ": record["n_occ"],
        "active": active.tolist(),
        "inactive": record["inactive"].tolist(),
        "occ_weights": None if record["occ_weights"] is None else record["occ_weights"].tolist(),
        "virt_weights": None if record["virt_weights"] is None else record["virt_weights"].tolist(),
        "orbitals": rows,
        "natural_orbitals": natural,
        "relabelled_orbitals": relabelled,
        "relabel_info": relabel_info,
    }


def _target_ao_indices(record: dict, target_labels: list[str], fragment_atoms: list[int]):
    """AO indices matching ``"<element> <shell>"`` labels on the fragment atoms."""
    labels = record["labels"]
    if labels is None:
        raise SystemExit(
            f"{record['path']}: --target-labels needs AO labels in the dump "
            "(or --orb-indx for a SIESTA dump written before labels were recorded)"
        )
    wanted = {" ".join(label.split()) for label in target_labels}
    target = [
        mu
        for mu, (atom, element, shell, _m) in enumerate(labels)
        if atom in fragment_atoms and f"{element} {shell}" in wanted
    ]
    if not target:
        shells = sorted(
            {f"{element} {shell}" for atom, element, shell, _m in labels if atom in fragment_atoms}
        )
        raise SystemExit(
            f"{record['path']}: {target_labels} match no fragment AO; available: {shells}"
        )
    return np.asarray(target, dtype=int)


def _ao_projector(s: np.ndarray, target: np.ndarray) -> np.ndarray:
    b = np.zeros((s.shape[0], target.size))
    b[target, np.arange(target.size)] = 1.0
    s22 = b.T @ s @ b
    try:
        m = s @ b @ np.linalg.solve(s22, b.T @ s)
    except np.linalg.LinAlgError:
        m = s @ b @ np.linalg.pinv(s22) @ b.T @ s
    return 0.5 * (m + m.T)


def reselect_with_labels(
    record: dict,
    target_labels: list[str],
    *,
    fragment_atoms: list[int],
    bond_atoms: tuple[int, int],
    n_frozen_occ: int = 0,
) -> tuple[list[dict], dict]:
    """Predict the label-aware AVAS selection from an existing dump.

    The dump's ``coeff`` is already an AVAS rotation of the same occupied and virtual
    subspaces (:math:`F_emb` eigenvectors rotated within each block).  AVAS selection
    is invariant to that prior rotation, so diagonalising the *new* target-AO
    projector in the dumped occupied and virtual blocks reproduces exactly the span
    the fixed-label workflow would select -- no SIESTA rerun needed to inspect it.
    """
    coeff = record["coeff"]
    s = record["overlap"]
    n_occ = record["n_occ"]
    active = record["active"]
    n_act_occ = int((active < n_occ).sum())
    n_act_virt = int((active >= n_occ).sum())
    target = _target_ao_indices(record, target_labels, fragment_atoms)
    m = _ao_projector(s, target)

    occ_idx = np.arange(n_frozen_occ, n_occ)
    virt_idx = np.arange(n_occ, coeff.shape[1])
    w_occ, u_occ = np.linalg.eigh(coeff[:, occ_idx].T @ m @ coeff[:, occ_idx])
    w_virt, u_virt = np.linalg.eigh(coeff[:, virt_idx].T @ m @ coeff[:, virt_idx])
    keep_occ = np.argsort(w_occ)[::-1][:n_act_occ]
    keep_virt = np.argsort(w_virt)[::-1][:n_act_virt]
    selected = np.concatenate([occ_idx[keep_occ], virt_idx[keep_virt]])
    kept = coeff[:, selected].copy()
    kept[:, :n_act_occ] = coeff[:, occ_idx] @ u_occ[:, keep_occ]
    kept[:, n_act_occ:] = coeff[:, virt_idx] @ u_virt[:, keep_virt]

    rows = _orbital_rows(
        record, kept, None, "relabelled", fragment_atoms=fragment_atoms, bond_atoms=bond_atoms
    )
    for row, column in zip(rows, selected):
        row["column"] = int(column)
    info = {
        "labels": list(target_labels),
        "target_aos": target.tolist(),
        "occ_weights": np.sort(w_occ)[::-1][:n_act_occ].tolist(),
        "virt_weights": np.sort(w_virt)[::-1][:n_act_virt].tolist(),
        "selected_columns": selected.tolist(),
    }
    return rows, info


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #
def _fmt(value, width: int = 7, precision: int = 3) -> str:
    if value is None:
        return " " * (width - 1) + "-"
    if isinstance(value, str):
        return value[:width].ljust(width)
    return f"{value:{width}.{precision}f}"


def print_table(result: dict, groups: list[str]) -> None:
    meta = result["meta"]
    print(f"\n{result['path']}")
    print(
        f"  qm_code={meta.get('qm_code')} selector={meta.get('selector')} "
        f"labels={meta.get('avas_ao_labels')} basis={meta.get('basis')} "
        f"active={result['active']} n_occ={result['n_occ']}"
    )
    header = (
        f"  {'kind':8s} {'k':>2s} {'occ':>6s} {'frag':>6s} {'bond':>7s} "
        + " ".join(f"{g:>9s}" for g in groups)
        + f" {'p_par':>6s} {'p_perp':>6s} {'ang':>5s}  character"
    )
    print(header)
    for row in result["orbitals"]:
        _print_row(row, groups)
    if result["natural_orbitals"]:
        print("  -- natural orbitals of the active-space FCI 1-RDM --")
        for row in result["natural_orbitals"]:
            _print_row(row, groups)
    if result["relabelled_orbitals"]:
        info = result["relabel_info"]
        print(
            f"  -- label-aware re-selection: {info['labels']} "
            f"(occ weights {[round(w, 3) for w in info['occ_weights']]}, "
            f"virt weights {[round(w, 3) for w in info['virt_weights']]}, "
            f"columns {info['selected_columns']}) --"
        )
        for row in result["relabelled_orbitals"]:
            _print_row(row, groups)


def _print_row(row: dict, groups: list[str]) -> None:
    weights = " ".join(f"{row.get(f'w[{g}]', float('nan')):9.3f}" for g in groups)
    print(
        f"  {row['kind']:8s} {row['index']:2d} {_fmt(row['occupation'], 6)} "
        f"{_fmt(row['fragment_pop'], 6)} {_fmt(row['bond_pop_all'], 7)} {weights} "
        f"{_fmt(row['p_parallel'], 6)} {_fmt(row['p_perp'], 6)} "
        f"{_fmt(row['p_axis_angle_deg'], 5, 0)}  {row['character']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dumps", nargs="+", type=Path, help="--dump_active_mos .npz files")
    parser.add_argument(
        "--orb-indx",
        type=Path,
        default=None,
        help="SIESTA .ORB_INDX for a dump that carries no labels (one dump only).",
    )
    parser.add_argument("--bond-atoms", type=int, nargs=2, default=(0, 1), metavar=("I", "J"))
    parser.add_argument(
        "--fragment-atoms",
        type=int,
        nargs="+",
        default=None,
        help="Fragment atom indices (default: the dump's reordered active atoms).",
    )
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--target-labels",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Re-run the AVAS selection against these AO labels on the dumped occupied/"
        "virtual blocks (e.g. 'N 2p' 'C 2p'), and report the orbitals it would pick.",
    )
    parser.add_argument(
        "--n-frozen-occ",
        type=int,
        default=None,
        help="Core floor for --target-labels (default: the dump's meta value).",
    )
    args = parser.parse_args(argv)
    if args.orb_indx is not None and len(args.dumps) != 1:
        parser.error("--orb-indx applies to one dump; run the script once per dump")

    records = [load_dump(path, args.orb_indx) for path in args.dumps]
    results = []
    group_keys: set[str] = set()
    for record in records:
        meta = record["meta"]
        if args.fragment_atoms is not None:
            fragment_atoms = list(args.fragment_atoms)
        else:
            active_atoms = meta.get("active_atoms") or []
            # The workflow reorders active atoms to the front of the basis.
            fragment_atoms = (
                list(range(len(active_atoms))) if active_atoms else list(args.bond_atoms)
            )
        result = analyse(
            record,
            fragment_atoms=fragment_atoms,
            bond_atoms=tuple(args.bond_atoms),
            target_labels=args.target_labels,
            n_frozen_occ=(
                int(meta.get("n_frozen_occ") or 0)
                if args.n_frozen_occ is None
                else args.n_frozen_occ
            ),
        )
        results.append(result)
        for row in result["orbitals"] + result["natural_orbitals"] + result["relabelled_orbitals"]:
            group_keys.update(key[2:-1] for key in row if key.startswith("w["))

    groups = sorted(group_keys)
    for result in results:
        print_table(result, groups)

    if args.out_csv is not None:
        rows = [
            row
            for result in results
            for row in (
                result["orbitals"] + result["natural_orbitals"] + result["relabelled_orbitals"]
            )
        ]
        columns = (
            [
                "source",
                "kind",
                "column",
                "index",
                "occupation",
                "fragment_pop",
                "bond_pop_all",
                "bond_pop_p",
                "p_parallel",
                "p_perp",
                "p_axis_angle_deg",
                "character",
            ]
            + [f"w[{g}]" for g in groups]
            + [f"pop[{g}]" for g in groups]
        )
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out_csv}")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
