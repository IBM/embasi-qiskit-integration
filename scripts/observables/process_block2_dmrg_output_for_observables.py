#!/usr/bin/env python3
"""
Prepare inputs for compute_observables.py from a raw block2 DMRG run.

Reads the per-root 2-RDMs that block2 leaves in the scratch dir, contracts each
to a 1-RDM, and writes a small bundle that compute_observables.py consumes:

  <prefix>_rdm1.npz      per-root 1-RDMs (keys dm1_root<i>)
  <prefix>_states.npz    per-root energies (keys e_states, delta_eV)
  <prefix>_manifest.txt  what was found, plus the command line to run next

Why separate: the spatial_twopdm text files scale as n_orbs^4 (~2 GB for
CAS(50e,50o)) and only the 1-RDM is needed downstream, so this runs once and the
bundle is ~0.2 MB.

Install
-------
numpy only (pyscf is needed by compute_observables.py, not here):

    python3 -m venv venv && source venv/bin/activate
    pip install numpy

Run
---
    python process_block2_dmrg_output_for_observables.py \
        --run-root-folder path/to/run_folder --out-prefix prepared

run_folder holds dmrg_scratch/ and dmrg_rundir/ as siblings; the RDMs, FCIDUMP,
dmrg.e and the geometry/MO/basis inputs are all discovered from it. The scratch
dir or the run dir may be given instead. Add --check to report what was found
without writing. The manifest prints the compute_observables.py command with
every path filled in.

Inputs
------
--run-root-folder  run folder, scratch dir or run dir (alias: --dmrg-dir).
--rdm-dir          exact dir of spatial_<kind>.<i>.<i>.txt, if not found.
--rdm-kind         twopdm (default, contracted here) or onepdm (read directly).
--fcidump          FCIDUMP supplying NORB/NELEC. Default: in the run dir.
--n-elec/--n-orbs  active space. Default: FCIDUMP header; explicit values win,
                   and a disagreement is reported, not applied silently.
--energies         override the energy file: comma list, text file, or npz. Use
                   --energies="-4350.1,..." ('=' keeps argparse from reading the
                   leading '-' as a flag).
--geom/--mo/--basis  passed through to compute_observables.py. Default:
                   discovered in the run folder. --basis is guessed from the
                   folder name -- verify it against your run.
--out-prefix       output prefix (default: prepared).
--check            locate and report inputs, then stop.

Energies
--------
Taken from block2's binary dmrg.e (converged per-root values). dmrg.out is a
cross-check only: it logs one block per sweep, including warmup, and the last
block is not the converged one (~9 mHa off on the reference run). If no energy
file is found none is invented -- the tables then show zero excitation energies.

Notes
-----
* The 1-RDM trace must equal n_elec; the manifest reports it per root. Off by
  more than 0.1 usually means --n-elec is wrong or the RDM is spin-resolved.
* Missing roots are skipped, not faked; the manifest lists which were found.
* No quantum chemistry here -- geometry/basis/MO are only passed through, and
  are required by compute_observables.py to turn a 1-RDM into a dipole or charge.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

DEF_DMRG_DIR = "dmrg_scratch"
DEF_RDM_KIND = "twopdm"
DEF_OUT_PREFIX = "prepared"

_report = []


def log(msg=""):
    print(msg, flush=True)
    _report.append(msg)


def section(title):
    log()
    log(f"=== {title} " + "=" * max(0, 60 - len(title)))


RUN_MARKERS = ("FCIDUMP", "dmrg.conf", "dmrg.out")
FCIDUMP_NAMES = ("FCIDUMP", "fcidump_block2.dat", "fcidump_pyscf.dat",
                 "FCIDUMP.dat", "fcidump.dat")


def _is_run_dir(d):
    return any(os.path.exists(os.path.join(d, m)) for m in RUN_MARKERS)


def find_run_dir(root):
    """Directory holding FCIDUMP / dmrg.conf / dmrg.out, at root or below it."""
    if _is_run_dir(root):
        return root
    for dirpath, _dirs, files in os.walk(root):
        if any(m in files for m in RUN_MARKERS):
            return dirpath
    return None


def find_fcidump(*dirs):
    """First recognised FCIDUMP filename in any of `dirs`."""
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for name in FCIDUMP_NAMES:
            cand = os.path.join(d, name)
            if os.path.exists(cand):
                return cand
    return None


def find_run_dir_upward(start, levels=3):
    """Nearest run dir at or beside an ancestor of `start`.

    block2 as driven by pyscf-dmrgscf writes the RDMs to <scratch>/node0/ but
    keeps FCIDUMP/dmrg.out in a separate run directory that is a SIBLING of the
    scratch dir, so ancestors alone are not enough.
    """
    cur = os.path.abspath(start)
    for _ in range(levels):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        if _is_run_dir(parent):
            return parent
        for sib in sorted(glob.glob(os.path.join(parent, "*"))):
            if os.path.isdir(sib) and "_saved" not in os.path.basename(sib):
                if _is_run_dir(sib):
                    return sib
        cur = parent
    return None


BASIS_PATTERNS = (
    "def2-tzvpp", "def2-tzvp", "def2-svp", "def2-qzvp",
    "cc-pvtz", "cc-pvdz", "cc-pvqz",
    "6-311g", "6-31g", "sto-3g",
)


def find_geometry(*dirs):
    """First .xyz found in `dirs` (single match preferred)."""
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        hits = sorted(glob.glob(os.path.join(d, "*.xyz")))
        if len(hits) == 1:
            return hits[0], None
        if len(hits) > 1:
            return hits[0], [os.path.basename(h) for h in hits]
    return None, None


def find_mo_file(*dirs):
    """MO-coefficient npz: prefers *natorb*.npz, then any npz holding mo_coeff."""
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        cands = sorted(glob.glob(os.path.join(d, "*natorb*.npz")))
        cands += [f for f in sorted(glob.glob(os.path.join(d, "*.npz")))
                  if f not in cands]
        for f in cands:
            try:
                with np.load(f, allow_pickle=False) as z:
                    if "mo_coeff" in z:
                        return f
            except Exception:
                continue
    return None


def guess_basis(*names):
    """Basis name inferred from a directory or file name, or None."""
    for name in names:
        if not name:
            continue
        low = os.path.basename(os.path.abspath(name)).lower()
        for b in BASIS_PATTERNS:
            if b in low:
                return b
    return None


def find_rdm_dir(root, kind):
    """Directory holding spatial_<kind>.<i>.<i>.txt; prefers node0/."""
    pat = f"spatial_{kind}.*.txt"
    node0 = os.path.join(root, "node0")
    for cand in (node0, root):
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, pat)):
            return cand
    for dirpath, _dirs, _files in os.walk(root):
        if glob.glob(os.path.join(dirpath, pat)):
            return dirpath
    return None


def find_rdm_dir_upward(start, kind, levels=3):
    """RDM dir at or beside an ancestor of `start`.

    The mirror of find_run_dir_upward: when the user points at the run dir, the
    spatial_*pdm files sit in the scratch dir, which is its SIBLING.
    """
    cur = os.path.abspath(start)
    for _ in range(levels):
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        for sib in sorted(glob.glob(os.path.join(parent, "*"))):
            if os.path.isdir(sib) and "_saved" not in os.path.basename(sib):
                hit = find_rdm_dir(sib, kind)
                if hit:
                    return hit
        cur = parent
    return None


def rdm_roots_present(rdm_dir, kind):
    """Sorted root indices i for which spatial_<kind>.i.i.txt exists."""
    roots = []
    for path in glob.glob(os.path.join(rdm_dir, f"spatial_{kind}.*.*.txt")):
        m = re.search(rf"spatial_{kind}\.(\d+)\.(\d+)\.txt$", os.path.basename(path))
        if m and m.group(1) == m.group(2):
            roots.append(int(m.group(1)))
    return sorted(roots)


def read_fcidump_header(path):
    """Return (norb, nelec) from a FCIDUMP header, or (None, None)."""
    norb = nelec = None
    try:
        with open(path) as fh:
            for line in fh:
                flat = line.upper().replace(" ", "")
                for key in ("NORB", "NELEC"):
                    for part in flat.replace("&FCI", "").split(","):
                        if part.startswith(key + "="):
                            try:
                                val = int(part.split("=")[1])
                            except ValueError:
                                continue
                            if key == "NORB":
                                norb = val
                            else:
                                nelec = val
                if "&END" in flat or line.strip() == "/":
                    break
    except OSError:
        pass
    return norb, nelec


def contract_twopdm(path, n_orbs, n_elec):
    """block2 spatial 2-RDM text -> spin-summed 1-RDM.

    Columns are 'i k l j value' with twopdm[i,j,k,l] = 2*value, so
    dm1[k,i] = sum_j twopdm[i,k,j,j] / (n_elec - 1).
    """
    if n_elec < 2:
        raise ValueError(
            "the 2-RDM contraction divides by (n_elec - 1); use --rdm-kind onepdm "
            "for a 1-electron active space"
        )
    twopdm = np.zeros((n_orbs,) * 4)
    n_lines = 0
    with open(path) as fh:
        header = fh.readline().split()
        norb_hdr = int(header[0]) if header else n_orbs
        if norb_hdr != n_orbs:
            log(f"    WARNING: file norb={norb_hdr} but --n-orbs={n_orbs}")
        for line in fh:
            sp = line.split()
            if len(sp) < 5:
                continue
            i, k, l, j = (int(sp[n]) for n in range(4))
            twopdm[i, j, k, l] = 2.0 * float(sp[4])
            n_lines += 1
    if not n_lines:
        raise ValueError(f"{path}: no 2-RDM elements parsed")
    return np.einsum("ikjj->ki", twopdm) / (n_elec - 1)


def read_onepdm(path, n_orbs):
    """block2 spatial 1-RDM text -> 1-RDM, symmetrized if only one triangle."""
    dm1 = np.zeros((n_orbs, n_orbs))
    seen = set()
    with open(path) as fh:
        header = fh.readline().split()
        norb_hdr = int(header[0]) if header else n_orbs
        if norb_hdr != n_orbs:
            log(f"    WARNING: file norb={norb_hdr} but --n-orbs={n_orbs}")
        for line in fh:
            sp = line.split()
            if len(sp) < 3:
                continue
            i, j = int(sp[0]), int(sp[1])
            dm1[i, j] = float(sp[2])
            seen.add((i, j))
    if not seen:
        raise ValueError(f"{path}: no 1-RDM elements parsed")
    if not any((j, i) in seen for i, j in seen if i != j):
        dm1 = dm1 + dm1.T - np.diag(np.diag(dm1))
    return dm1


def read_dmrg_e(path, n_roots=None):
    """Per-root converged energies from block2's binary dmrg.e (raw float64)."""
    e = np.fromfile(path, dtype=np.float64)
    if e.size == 0:
        return None
    if n_roots and e.size < n_roots:
        return None
    return e[:n_roots] if n_roots else e


def find_energy_source(rdm_dir, run_dir):
    """Preferred energy file: dmrg.e / E_dmrg.npy beside the RDMs, else npz."""
    for d in (rdm_dir, run_dir):
        if not d or not os.path.isdir(d):
            continue
        cand = os.path.join(d, "dmrg.e")
        if os.path.exists(cand):
            return cand, "dmrg.e"
    for d in (rdm_dir, run_dir, os.path.dirname(rdm_dir or "") or None,
              os.path.dirname(os.path.dirname(rdm_dir or "")) or None):
        if not d or not os.path.isdir(d):
            continue
        for pat in ("dmrg_states*.npz", "*states*.npz", "e_states*.npz",
                    "energies.txt", "energies.dat"):
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                return hits[0], "npz/text"
    return None, None


def scrape_dmrg_out_blocks(path):
    """All '.. E[ i] = <energy>' blocks in a block2 dmrg.out.

    Returns a list of dicts {root_index: energy}, in file order. A new block
    starts whenever the index resets to 0. These are per-sweep values: the last
    block is NOT necessarily the converged one, so this is only ever used to
    cross-check a proper energy file, never as the primary source.
    """
    try:
        with open(path, errors="replace") as fh:
            txt = fh.read()
    except OSError:
        return []
    blocks, cur = [], {}
    for m in re.finditer(r"E\[\s*(\d+)\s*\]\s*=\s*(-?\d+\.\d+)", txt):
        idx, val = int(m.group(1)), float(m.group(2))
        if idx == 0 and cur:
            blocks.append(cur)
            cur = {}
        cur[idx] = val
    if cur:
        blocks.append(cur)
    return blocks


def check_against_dmrg_out(path, e_states, tol=1e-6):
    """Cross-check e_states against the sweep energies in dmrg.out.

    Reports the closest matching sweep block. Never overrides e_states -- a
    mismatch is surfaced for the user to judge, because block2's sweep blocks
    include warmup and non-converged passes.
    """
    blocks = scrape_dmrg_out_blocks(path)
    if not blocks:
        log(f"  consistency: no 'E[i] =' blocks found in {os.path.basename(path)}")
        return
    n = len(e_states)
    full = [b for b in blocks if len(b) == n]
    log(f"  consistency: {os.path.basename(path)} has {len(blocks)} sweep blocks, "
        f"{len(full)} with {n} roots")
    if not full:
        sizes = sorted({len(b) for b in blocks})
        log(f"    no block has {n} roots (block sizes seen: {sizes}) -- cannot "
            "cross-check; cannot confirm the root count either")
        return
    best, best_d = None, None
    for k, b in enumerate(full):
        arr = np.array([b[i] for i in sorted(b)])
        d = float(np.abs(arr - e_states).max())
        if best_d is None or d < best_d:
            best, best_d = k, d
    gs = None
    m = re.search(r"DMRG Energy\s*=\s*(-?\d+\.\d+)",
                  open(path, errors="replace").read())
    if m:
        gs = float(m.group(1))
    if best_d <= tol:
        where = "the last" if best == len(full) - 1 else f"block {best + 1}/{len(full)}"
        log(f"    OK: matches {where} sweep block to {best_d:.2e} Ha")
    else:
        log(f"    MISMATCH: closest sweep block differs by {best_d:.2e} Ha "
            f"(block {best + 1}/{len(full)})")
        log("    The energy file and dmrg.out disagree. Check they come from the "
            "same run before trusting the excitation energies.")
    if gs is not None:
        dg = abs(gs - e_states[0])
        tag = "OK" if dg <= tol else "MISMATCH"
        log(f"    {tag}: 'DMRG Energy' line {gs:.9f} vs root 0 {e_states[0]:.9f} "
            f"(diff {dg:.2e} Ha)")


def load_energies(spec):
    """Energies (Ha) from a comma list, a text file, or an npz."""
    if os.path.exists(spec):
        if spec.endswith(".npz"):
            with np.load(spec, allow_pickle=False) as d:
                for key in ("e_states", "e_tot", "energies"):
                    if key in d:
                        return np.atleast_1d(np.asarray(d[key], dtype=float))
                raise KeyError(
                    f"{spec} has no 'e_states' key (found: {list(d.keys())})"
                )
        with open(spec) as fh:
            return np.array(
                [float(ln.split()[0]) for ln in fh if ln.strip()], dtype=float
            )
    return np.array([float(v) for v in spec.split(",") if v.strip()], dtype=float)


def build_parser():
    p = argparse.ArgumentParser(
        description="Turn a raw block2 dmrg_scratch into inputs for "
                    "compute_observables.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for install instructions and examples.",
    )
    p.add_argument("--run-root-folder", "--dmrg-dir", dest="dmrg_dir", default=None,
                   metavar="DIR",
                   help="the DMRG run folder holding dmrg_scratch/ and "
                        "dmrg_rundir/ as siblings. The RDMs, FCIDUMP, dmrg.e "
                        "and the geometry/MO/basis inputs for compute_observables.py are all "
                        "discovered from it. The scratch dir or the run dir "
                        "itself are also accepted. (--dmrg-dir is an alias; "
                        f"default: {DEF_DMRG_DIR})")
    p.add_argument("--rdm-dir", default=None,
                   help="exact dir holding spatial_<kind>.<i>.<i>.txt; may be "
                        "combined with --run-root-folder, and its parents are "
                        "searched for a run dir when that is absent")
    p.add_argument("--rdm-kind", choices=("twopdm", "onepdm"), default=DEF_RDM_KIND,
                   help=f"block2 RDM file type (default: {DEF_RDM_KIND})")
    p.add_argument("--fcidump", default=None,
                   help="FCIDUMP file supplying norb/nelec (default: FCIDUMP in "
                        "the run dir, when there is one)")
    p.add_argument("--n-elec", type=int, default=None,
                   help="active electrons (default: from FCIDUMP)")
    p.add_argument("--n-orbs", type=int, default=None,
                   help="active orbitals (default: from FCIDUMP)")
    p.add_argument("--roots", default=None,
                   help="comma list of root indices to keep (default: all found)")
    p.add_argument("--energies", default=None,
                   help="energies in Ha, overriding the discovered energy file: comma "
                        "list, text file, or npz")
    p.add_argument("--geom", default=None,
                   help="geometry .xyz to record for compute_observables.py (default: the single "
                        ".xyz found in the run folder)")
    p.add_argument("--mo", default=None,
                   help="MO-coefficient npz to record for compute_observables.py (default: the "
                        "*natorb*.npz found in the run folder)")
    p.add_argument("--basis", default=None,
                   help="basis set name to record for compute_observables.py (default: inferred "
                        "from the run folder name)")
    p.add_argument("--out-prefix", default=DEF_OUT_PREFIX,
                   help=f"output file prefix (default: {DEF_OUT_PREFIX})")
    p.add_argument("--check", action="store_true",
                   help="report what was found, then stop before writing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    section("Locate DMRG output")
    if not (args.dmrg_dir or args.rdm_dir):
        args.dmrg_dir = DEF_DMRG_DIR
        log(f"No source given -- trying default --run-root-folder {DEF_DMRG_DIR}")
        if not os.path.isdir(args.dmrg_dir):
            raise FileNotFoundError(
                f"default DMRG directory '{DEF_DMRG_DIR}' not found in "
                f"{os.getcwd()}. Pass --run-root-folder <your DMRG run folder>, or "
                "--rdm-dir <dir with spatial_*pdm files>."
            )

    run_dir = None
    rdm_dir = None
    if args.dmrg_dir:
        if not os.path.isdir(args.dmrg_dir):
            raise FileNotFoundError(
                f"--run-root-folder not a directory: {args.dmrg_dir}")
        # Search below first, then beside an ancestor: when the given dir is the
        # scratch dir, FCIDUMP/dmrg.out live in a sibling run dir, not under it.
        run_dir = find_run_dir(args.dmrg_dir) or find_run_dir_upward(args.dmrg_dir)
        if run_dir:
            note = "" if _is_run_dir(args.dmrg_dir) else "   (found beside the run folder)"
            log(f"Run dir      : {run_dir}{note}")
        else:
            run_dir = args.dmrg_dir
            log(f"Run dir      : {run_dir}   (no FCIDUMP/dmrg.out found in it)")
        if not args.rdm_dir:
            rdm_dir = (find_rdm_dir(args.dmrg_dir, args.rdm_kind)
                       or find_rdm_dir_upward(args.dmrg_dir, args.rdm_kind))
        if rdm_dir is None and not args.rdm_dir:
            raise FileNotFoundError(
                f"no spatial_{args.rdm_kind}.<i>.<i>.txt found under "
                f"{args.dmrg_dir}. Check --rdm-kind (twopdm vs onepdm), or point "
                "--rdm-dir at the directory holding them (block2 usually writes "
                "them to node0/)."
            )
    if args.rdm_dir:
        if not os.path.isdir(args.rdm_dir):
            raise FileNotFoundError(f"--rdm-dir not a directory: {args.rdm_dir}")
        rdm_dir = args.rdm_dir
        if run_dir is None:
            run_dir = find_run_dir(rdm_dir) or find_run_dir_upward(rdm_dir)
            if run_dir:
                log(f"Run dir      : {run_dir}   (found from --rdm-dir)")
    log(f"RDM dir      : {rdm_dir}")
    log(f"RDM kind     : {args.rdm_kind}")

    # The "run folder" is the common parent of the scratch and run dirs: it is
    # where the pipeline leaves the geometry, the MO npz and the states npz.
    cands = []
    if args.dmrg_dir:
        cands.append(os.path.abspath(args.dmrg_dir))
    for d in (run_dir, rdm_dir):
        if d:
            cands.append(os.path.abspath(d))
    top_dir = None
    if cands:
        top_dir = os.path.commonpath(cands) if len(cands) > 1 else cands[0]
        # walk up out of node0/ or dmrg_scratch/ to the folder holding them
        for _ in range(3):
            if find_geometry(top_dir)[0] or find_mo_file(top_dir):
                break
            parent = os.path.dirname(top_dir)
            if parent == top_dir:
                break
            top_dir = parent
    if top_dir:
        log(f"Run folder   : {top_dir}")

    fcidump = args.fcidump
    if fcidump:
        if not os.path.exists(fcidump):
            raise FileNotFoundError(f"--fcidump not found: {fcidump}")
        if os.path.isdir(fcidump):
            found = find_fcidump(fcidump)
            if found is None:
                raise FileNotFoundError(
                    f"--fcidump is a directory with no FCIDUMP inside: "
                    f"{args.fcidump} (looked for {', '.join(FCIDUMP_NAMES)})"
                )
            fcidump = found
    elif run_dir:
        fcidump = find_fcidump(run_dir, rdm_dir, os.path.dirname(rdm_dir))

    if fcidump:
        src = "given" if args.fcidump else "run dir"
        norb, nelec = read_fcidump_header(fcidump)
        if norb is None and nelec is None:
            log(f"FCIDUMP      : {fcidump}  [{src}]")
            log("  WARNING: no NORB/NELEC in the header -- pass --n-orbs/--n-elec")
        else:
            log(f"FCIDUMP      : {fcidump}  [{src}]  NORB={norb} NELEC={nelec}")
        if norb:
            if args.n_orbs is None:
                args.n_orbs = norb
                log(f"n_orbs       : {norb}   (from FCIDUMP)")
            elif args.n_orbs != norb:
                log(f"  NOTE: --n-orbs {args.n_orbs} overrides FCIDUMP NORB={norb}")
        if nelec:
            if args.n_elec is None:
                args.n_elec = nelec
                log(f"n_elec       : {nelec}   (from FCIDUMP)")
            elif args.n_elec != nelec:
                log(f"  NOTE: --n-elec {args.n_elec} overrides FCIDUMP NELEC={nelec}")

    why = (
        f"the header of {fcidump} has no usable NORB/NELEC"
        if fcidump
        else "no FCIDUMP was found"
    )
    if args.n_orbs is None:
        raise ValueError(
            f"could not determine --n-orbs: {why}. Pass --n-orbs explicitly"
            + ("." if fcidump else ", or --fcidump <path> to read it from the header.")
        )
    if args.n_elec is None and args.rdm_kind == "twopdm":
        raise ValueError(
            f"could not determine --n-elec: {why}. It is required to contract "
            "the 2-RDM -- pass --n-elec explicitly"
            + ("." if fcidump else ", or --fcidump <path>.")
        )
    log(f"Active space : CAS({args.n_elec}e,{args.n_orbs}o)")

    roots_found = rdm_roots_present(rdm_dir, args.rdm_kind)
    if not roots_found:
        raise FileNotFoundError(
            f"no spatial_{args.rdm_kind}.<i>.<i>.txt files in {rdm_dir}"
        )
    if args.roots:
        want = [int(v) for v in args.roots.split(",") if v.strip()]
        missing = [r for r in want if r not in roots_found]
        if missing:
            raise ValueError(f"--roots asks for {missing}, not present in {rdm_dir}")
        roots = want
    else:
        roots = roots_found
    log(f"Roots found  : {roots_found}")
    if roots != roots_found:
        log(f"Roots kept   : {roots}")
    gaps = [i for i in range(max(roots_found) + 1) if i not in roots_found]
    if gaps:
        log(f"NOTE: roots {gaps} have no RDM file -- the DMRG run may be partial")

    section("Energies")
    e_states = None
    e_src = None

    if args.energies and os.path.isdir(args.energies):
        found, _kind = find_energy_source(args.energies, args.energies)
        if found is None:
            raise FileNotFoundError(
                f"--energies is a directory with no energy file inside: "
                f"{args.energies} (looked for dmrg.e, dmrg_states*.npz, "
                "energies.txt)"
            )
        log(f"--energies is a directory -- using {found}")
        args.energies = found

    if args.energies:
        if os.path.basename(args.energies) == "dmrg.out":
            raise ValueError(
                "dmrg.out holds per-sweep energies, including warmup and "
                "non-converged passes, so the converged values cannot be "
                "identified reliably. Point --energies at dmrg.e (binary, beside "
                "the RDMs) or a states npz instead; dmrg.out is used only as a "
                "consistency check."
            )
        if os.path.basename(args.energies) == "dmrg.e":
            e_states = read_dmrg_e(args.energies)
            e_src = args.energies
            log(f"Energies     : {args.energies}  ({e_states.size} values, dmrg.e)")
        else:
            e_states = load_energies(args.energies)
            e_src = args.energies
            log(f"Energies     : {args.energies}  ({e_states.size} values, given)")
    else:
        found, kind = find_energy_source(rdm_dir, run_dir)
        if found is None:
            log("No energy file found (looked for dmrg.e beside the RDMs, then "
                "dmrg_states*.npz / energies.txt).")
            log("  Pass --energies to get excitation energies downstream.")
        elif kind == "dmrg.e":
            e_states = read_dmrg_e(found)
            e_src = found
            if e_states is None:
                log(f"{found} is empty -- pass --energies instead.")
            else:
                log(f"Energies     : {found}  ({e_states.size} values, dmrg.e, "
                    "auto-discovered)")
        else:
            e_states = load_energies(found)
            e_src = found
            log(f"Energies     : {found}  ({e_states.size} values, "
                "auto-discovered)")

    if e_states is not None:
        log(f"  {np.round(e_states, 6).tolist()}")
        # dmrg.out is only ever a cross-check on the numbers above.
        out = None
        for d in (run_dir, rdm_dir, os.path.dirname(rdm_dir or "") or None):
            if d and os.path.exists(os.path.join(d, "dmrg.out")):
                out = os.path.join(d, "dmrg.out")
                break
        if out:
            check_against_dmrg_out(out, e_states)
        else:
            log("  consistency: no dmrg.out found to cross-check against")

    if e_states is not None and e_states.size < len(roots):
        log(f"WARNING: {e_states.size} energies for {len(roots)} roots; "
            "excitation energies will be incomplete downstream")

    section("Molecule inputs for compute_observables")
    search = [d for d in (top_dir, run_dir, rdm_dir) if d]
    geom = args.geom
    if geom:
        if not os.path.exists(geom):
            raise FileNotFoundError(f"--geom not found: {geom}")
        log(f"Geometry     : {geom}  [given]")
    else:
        geom, multi = find_geometry(*search)
        if geom:
            log(f"Geometry     : {geom}  [found]")
            if multi:
                log(f"  NOTE: several .xyz present ({', '.join(multi)}); using "
                    "the first. Pass --geom to choose.")
        else:
            log("Geometry     : not found -- pass --geom, or give compute_observables its own "
                "--geom")

    mo = args.mo
    if mo:
        if not os.path.exists(mo):
            raise FileNotFoundError(f"--mo not found: {mo}")
        log(f"MO coeff     : {mo}  [given]")
    else:
        mo = find_mo_file(*search)
        if mo:
            log(f"MO coeff     : {mo}  [found]")
        else:
            log("MO coeff     : not found -- pass --mo, or give compute_observables its own --mo")

    basis = args.basis
    if basis:
        log(f"Basis        : {basis}  [given]")
    else:
        basis = guess_basis(top_dir, run_dir, mo, geom)
        if basis:
            log(f"Basis        : {basis}  [inferred from the folder name]")
            log("  VERIFY this matches the basis your DMRG run used -- it is a "
                "name guess, not a value read from the output.")
        else:
            log("Basis        : not determined -- pass --basis, or give compute_observables its "
                "own --basis")

    if args.check:
        section("Check only")
        log("--check: inputs located, stopping before any file is written")
        return 0

    section("Contract RDMs")
    reader = (
        (lambda p: contract_twopdm(p, args.n_orbs, args.n_elec))
        if args.rdm_kind == "twopdm"
        else (lambda p: read_onepdm(p, args.n_orbs))
    )
    dm1_roots = {}
    for iroot in roots:
        path = os.path.join(rdm_dir, f"spatial_{args.rdm_kind}.{iroot}.{iroot}.txt")
        size_mb = os.path.getsize(path) / 1e6
        dm1 = reader(path)
        tr = np.trace(dm1)
        flag = ""
        if args.n_elec is not None and abs(tr - args.n_elec) > 0.1:
            flag = f"  <-- WARNING, expected {args.n_elec}"
        log(f"  root {iroot}: {size_mb:8.1f} MB  trace={tr:.6f}{flag}")
        dm1_roots[iroot] = dm1

    if any(
        args.n_elec is not None and abs(np.trace(d) - args.n_elec) > 0.1
        for d in dm1_roots.values()
    ):
        log("")
        log("Trace mismatch usually means --n-elec does not match the DMRG run,")
        log("or that these are spin-resolved rather than spatial RDMs.")

    section("Write bundle")
    rdm_path = f"{args.out_prefix}_rdm1.npz"
    np.savez(rdm_path, **{f"dm1_root{i}": dm for i, dm in dm1_roots.items()})
    log(f"1-RDMs   -> {rdm_path}  ({os.path.getsize(rdm_path) / 1e6:.2f} MB, "
        f"{len(dm1_roots)} roots)")

    states_path = None
    if e_states is not None:
        states_path = f"{args.out_prefix}_states.npz"
        delta_eV = (e_states - e_states[0]) * 27.211386245988
        np.savez(states_path, e_states=e_states, delta_eV=delta_eV)
        log(f"Energies -> {states_path}")

    cmd = [
        "python compute_observables.py",
        f"--rdm-npz {rdm_path}",
        f"--n-elec {args.n_elec}",
        f"--n-orbs {args.n_orbs}",
    ]
    if states_path:
        cmd.append(f"--energies {states_path}")
    if geom:
        cmd.append(f"--geom {geom}")
    if basis:
        cmd.append(f"--basis {basis}")
    if mo:
        cmd.append(f"--mo {mo}")
    cmd_str = " \\\n    ".join(cmd)

    missing = [n for n, v in (("--geom", geom), ("--basis", basis), ("--mo", mo))
               if not v]

    section("Next step")
    if missing:
        log("Run the observables script on this bundle (supply the inputs that")
        log("could not be resolved -- see the note below):")
    else:
        log("Run the observables script on this bundle -- every input below was")
        log("resolved from the run folder, so no built-in defaults are used:")
    log("")
    log(f"    {cmd_str}")
    if states_path is None:
        log("")
        log("No energy file was written, so excitation energies and wavelengths")
        log("will be reported as zero. Add --energies to compute_observables to fill them in.")
    if missing:
        log("")
        log(f"Could not resolve {', '.join(missing)} from the run folder -- the "
            "command above omits them,")
        log("so compute_observables would fall back to its built-in example defaults. Supply "
            "them explicitly.")

    manifest_path = f"{args.out_prefix}_manifest.txt"
    with open(manifest_path, "w") as fh:
        fh.write(f"# {os.path.basename(sys.argv[0])}\n")
        fh.write(f"# rdm_dir={rdm_dir} kind={args.rdm_kind}\n")
        fh.write(f"# fcidump={fcidump}\n")
        fh.write(f"# geom={geom} basis={basis} mo={mo}\n")
        fh.write(f"# CAS({args.n_elec}e,{args.n_orbs}o) roots={roots}\n")
        fh.write("\n".join(_report) + "\n")
    log("")
    log(f"Manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, FileNotFoundError) as exc:
        sys.stderr.write(f"\nERROR: {exc}\n")
        sys.exit(2)
