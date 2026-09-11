#!/usr/bin/env python
"""
DMRG excited-state observables from DMRG output (dipoles + charge transfer)
==========================================================================
Post-processing step that turns the output of a state-averaged DMRG calculation
into per-state observables. It runs no SCF, no MP2 and no DMRG sweeps: you point
it at a finished DMRG run directory and it computes, per electronic root:

  * excitation energies (eV, nm)
  * permanent dipole moments, mu = mu_nuclear + mu_frozen-core + mu_CAS (Debye)
  * dipole shift vs the ground state, |dmu| = |mu(Sn) - mu(S0)|
  * optional charge-transfer analysis: Mulliken electron counts per user-defined
    atom fragment, and delta-q vs S0

Written for block2 / StackBlock output (as produced through pyscf-dmrgscf), but
any DMRG code works if it can dump spatial 1- or 2-RDMs, or you can hand the
1-RDMs over in a .npz.

--------------------------------------------------------------------------------
Install
--------------------------------------------------------------------------------
Only numpy and pyscf; pyscf is used purely as an AO-integral library here.

    python3 -m venv venv && source venv/bin/activate
    pip install "pyscf>=2.3" numpy

    # conda alternative
    conda create -n dmrg-obs python=3.11 numpy && conda activate dmrg-obs
    pip install "pyscf>=2.3"

Tested with pyscf 2.14 / numpy 2.5 on Python 3.11-3.13. No MPI, no block2 and no
compiled dependencies beyond pyscf's own wheels. Runtime is seconds to a few
minutes, dominated by the AO integrals; memory stays well under a few GB.

--------------------------------------------------------------------------------
Quick start
--------------------------------------------------------------------------------
Every argument has a default matching the shared example (system1 / PD1-PD2,
CAS(50e,50o), 10 roots, def2-TZVP, windowed-MP2 orbitals), so if you keep the
shipped file layout the whole thing reduces to:

    python 030_dmrg_observables.py

which expects, relative to the current directory:

    dmrg_scratch/                  your DMRG run (RDMs found recursively,
                                   typically in node0/; FCIDUMP read if present)
    system1.xyz                    the geometry
    win_mp2_natorbs_def2-TZVP_win_mp2.npz    the CAS orbitals
    dmrg_states*.npz               state energies (auto-discovered; also accepts
                                   energies.txt, or omit and dE reads 0)

If your DMRG code writes somewhere else, override just the pieces that moved:

    python 030_dmrg_observables.py --dmrg-dir /path/to/run --energies e.txt

    # 1-RDMs straight from your own npz, nothing else on disk to discover
    python 030_dmrg_observables.py --rdm-npz my_rdms.npz --energies e.txt

    # a different system: give its geometry, active space and fragments
    python 030_dmrg_observables.py --geom system2.xyz --n-elec 24 --n-orbs 24 \
        --ct-fragments "A:0-120;B:121-248"

Run with --check first on a new machine: it resolves and prints every input,
validates them, and stops before doing any integral work.

Defaults in force (all overridable):
    --geom system1.xyz            --basis def2-tzvp
    --mo win_mp2_natorbs_def2-TZVP_win_mp2.npz
    --dmrg-dir dmrg_scratch       --rdm-kind twopdm
    --n-elec 50  --n-orbs 50  --n-roots 10   (FCIDUMP, --rdm-npz and the RDM
                                              files on disk all take precedence
                                              over these)
    --ct-fragments  the PD1/PD2 split, applied only to a 170-atom geometry
    --out-prefix dmrg_obs

--------------------------------------------------------------------------------
Inputs
--------------------------------------------------------------------------------
From your DMRG run (--dmrg-dir, or point at pieces explicitly):

  RDMs, one per root, in the active space. Any one of:
    * block2 spatial 2-RDM text, the default:
        <dir>/spatial_twopdm.<i>.<i>.txt, columns "i k l j value"
        (pyscf-dmrgscf convention; the 1-RDM is obtained by contracting
         1-RDM[k,i] = sum_j 2-RDM[i,j,k,k] / (n_elec - 1))
    * block2 spatial 1-RDM text, with --rdm-kind onepdm:
        <dir>/spatial_onepdm.<i>.<i>.txt, columns "i j value"
      Cheaper and preferred if your run wrote them: no 2-RDM contraction.
    * your own .npz, with --rdm-npz: either a stacked array
      [n_roots, n_orbs, n_orbs] under key 'dm1' / 'rdm1' / 'dm1_roots', or
      per-root keys 'dm1_root0', 'dm1_root1', ...
      Spin-summed (total) 1-RDMs in the active space, trace = n_elec.

  --rdm-dir  overrides discovery and names the RDM directory directly.

  --energies total energies in Hartree, one per root: a comma-separated list, a
             text file (one value per line), or a .npz with key 'e_states'.
             Only differences are used, so any constant offset is fine. Omit it
             and the excitation-energy columns simply read 0.

Defining the molecule -- needed for the AO integrals, which are what make a
dipole or a Mulliken charge meaningful and are not contained in DMRG output:

  --geom   the .xyz used for the DMRG run (line 1 atom count, line 2 comment,
           then "symbol x y z" in Angstrom)
  --basis  the same basis as that run, e.g. def2-tzvp. It must match, or the
           core and dipole integrals describe a different molecule.
  --mo     the CAS orbital basis the RDMs are expressed in: the full [nao, nmo]
           MO coefficient matrix, from a .npz with key 'mo_coeff' (e.g.
           win_mp2_natorbs_*.npz) or a pickled pyscf CASCI/CASSCF object or
           dict. Active orbitals are assumed contiguous, columns
           [ncore, ncore + n_orbs) with ncore = nelectron//2 - n_elec//2, the
           pyscf sort_mo / AVAS convention -- override with --ncore if your
           code orders them differently.

--------------------------------------------------------------------------------
Charge-transfer fragments
--------------------------------------------------------------------------------
    --ct-fragments "PD1:0-10,22-93,167,168;PD2:11-21,94-166,169"

0-based atom indices in .xyz order (the numbering VMD shows); ranges inclusive.
Atoms you leave out are collected into a fragment called "rest".

--------------------------------------------------------------------------------
Outputs
--------------------------------------------------------------------------------
Tables are printed to stdout, and:

  <prefix>_dipole.npz    e_states, delta_eV, mu_nuc_au, mu_core_au,
                         mu_au_root<i>, mu_debye_root<i>, dm1_root<i>
  <prefix>_ct.npz        frag_names, q_frag_root<i>, dq_frag_root<i>
                         (only with --ct-fragments)
  <prefix>_summary.txt   the printed tables, for the record

--------------------------------------------------------------------------------
Notes and pitfalls
--------------------------------------------------------------------------------
* Fragment indices are specific to one geometry and do NOT carry over between
  systems. The built-in PD1/PD2 default is therefore applied only when the
  geometry has exactly 170 atoms; for anything else the CT analysis is skipped
  with a message until you pass --ct-fragments (or --no-ct to silence it). If
  "rest" comes back holding a large number of electrons, your lists are
  incomplete for the geometry you actually passed.
* Mulliken fragment counts include only the CAS electrons -- not core density,
  not nuclear charge -- so they measure how the active-space density
  redistributes. Read delta-q rather than the absolute counts.
* The dipole is the full permanent dipole (nuclear + frozen core + CAS) with the
  origin at the Cartesian origin, hence origin-independent for a neutral
  molecule. |dmu| is the quantity to compare against experiment.
* A 1-RDM trace far from n_elec means the RDM files, --n-elec and --n-orbs do
  not belong to each other; the script warns per root and continues.
* Only diagonal (i,i) RDMs are read, so these are permanent properties.
  Transition dipoles and oscillator strengths need off-diagonal transition RDMs
  and are out of scope.
* Passing negative energies inline needs "=" so argparse does not read the
  leading minus as a new flag:  --energies="-4350.11,-4350.00"
  (a file or .npz avoids the issue entirely).
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

EV = 27.211386245988
NM_EV = 1239.84193
AU2DEBYE = 2.541746473
BOHR = 0.52917721092

# Defaults describing the shared example: system1, the PD1/PD2 chlorophyll
# special pair (170 atoms), CAS(50e,50o) over 10 roots in def2-TZVP with
# windowed-MP2 natural orbitals. Every one of these is overridable on the
# command line; they exist so a run on the shipped example needs no flags.
DEF_GEOM = "system1.xyz"
DEF_BASIS = "def2-tzvp"
DEF_MO = "win_mp2_natorbs_def2-TZVP_win_mp2.npz"
DEF_N_ELEC = 50
DEF_N_ORBS = 50
DEF_N_ROOTS = 10
DEF_DMRG_DIR = "dmrg_scratch"
DEF_RDM_KIND = "twopdm"
DEF_OUT_PREFIX = "dmrg_obs"

# PD1/PD2 split of system1.xyz: 85 atoms each, covering all 170 (no 'rest').
# 0-based indices in .xyz order. Meaningless for any other geometry.
DEF_CT_FRAGMENTS = "PD1:0-10,22-93,167-168;PD2:11-21,94-166,169"
DEF_CT_NATOM = 170

_t0 = time.time()
_report = []


def log(msg=""):
    print(f"[+{time.time() - _t0:7.1f}s]  {msg}", flush=True)


def emit(msg=""):
    """Print and keep for the summary file."""
    print(msg, flush=True)
    _report.append(msg)


def section(title):
    log("=" * 65)
    log(f"  {title}")
    log("=" * 65)


def parse_index_spec(spec):
    """'0-10,22,25-27' -> [0..10, 22, 25, 26, 27]; inclusive ranges."""
    out = []
    for chunk in spec.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk[1:]:
            lo, _, hi = chunk.partition("-")
            lo, hi = int(lo), int(hi)
            if hi < lo:
                raise ValueError(f"descending range '{chunk}' in fragment spec")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    return out


def parse_ct_fragments(spec):
    """'PD1:0-84;PD2:85-169' -> {'PD1': [...], 'PD2': [...]}"""
    frags = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, sep, idxs = part.partition(":")
        if not sep:
            raise ValueError(f"fragment '{part}' is missing a ':' separator")
        name = name.strip()
        if name in frags:
            raise ValueError(f"duplicate fragment name '{name}'")
        frags[name] = parse_index_spec(idxs)
    if not frags:
        raise ValueError("--ct-fragments parsed to nothing")
    return frags


def read_xyz(path):
    """Return (pyscf geometry string, natom, composition dict)."""
    with open(path) as fh:
        lines = [ln.strip() for ln in fh]
    try:
        natom = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError(f"first line of {path} is not an atom count: {lines[0]!r}")
    if len(lines) < natom + 2:
        raise ValueError(f"{path}: header says {natom} atoms but file has {len(lines)} lines")

    composition, geom = {}, []
    for n in range(2, natom + 2):
        parts = lines[n].split()
        if len(parts) < 4:
            raise ValueError(f"{path} line {n + 1}: expected 'symbol x y z'")
        sym = parts[0]
        try:
            x, y, z = (float(v) for v in parts[1:4])
        except ValueError:
            raise ValueError(f"{path} line {n + 1}: non-numeric coordinates")
        composition[sym] = composition.get(sym, 0) + 1
        geom.append(f"{sym}  {x:.9f}  {y:.9f}  {z:.9f}")
    return "\n".join(geom), natom, composition


def load_mo_coeff(path):
    """Full [nao, nmo] MO coefficients from .npz or a pickled CASCI/dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"--mo file not found: {path}")

    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as d:
            for key in ("mo_coeff", "mo_coeffs", "mo"):
                if key in d:
                    return np.asarray(d[key], dtype=float), key
            raise KeyError(f"{path} has no 'mo_coeff' key (found: {list(d.keys())})")

    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if hasattr(obj, "mo_coeff"):
        return np.asarray(obj.mo_coeff, dtype=float), "pickle:mo_coeff"
    if isinstance(obj, dict) and "mo_coeff" in obj:
        return np.asarray(obj["mo_coeff"], dtype=float), "pickle:dict[mo_coeff]"
    raise KeyError(f"{path} contains no usable mo_coeff")


def load_energies(spec, n_roots):
    """Total energies (Ha) from a comma list, a text file, or an npz."""
    if spec is None:
        return None
    if os.path.exists(spec):
        if spec.endswith(".npz"):
            with np.load(spec, allow_pickle=False) as d:
                for key in ("e_states", "e_tot", "energies"):
                    if key in d:
                        e = np.atleast_1d(np.asarray(d[key], dtype=float))
                        break
                else:
                    raise KeyError(f"{spec} has no 'e_states' key (found: {list(d.keys())})")
        else:
            with open(spec) as fh:
                e = np.array([float(ln.split()[0]) for ln in fh if ln.strip()], dtype=float)
    else:
        e = np.array([float(v) for v in spec.split(",") if v.strip()], dtype=float)

    if e.size < n_roots:
        raise ValueError(f"got {e.size} energies but {n_roots} roots requested")
    return e


def read_twopdm_text(path, n_orbs, n_elec):
    """block2 spatial 2-RDM text -> spin-summed 1-RDM [n_orbs, n_orbs]."""
    twopdm = np.zeros((n_orbs,) * 4)
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
    if n_elec < 2:
        raise ValueError("2-RDM contraction needs n_elec >= 2; use --rdm-kind onepdm")
    return np.einsum("ikjj->ki", twopdm) / (n_elec - 1)


def read_onepdm_text(path, n_orbs):
    """block2 spatial 1-RDM text -> 1-RDM [n_orbs, n_orbs]."""
    dm1 = np.zeros((n_orbs, n_orbs))
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
    if not np.allclose(dm1, dm1.T, atol=1e-6):
        dm1 = 0.5 * (dm1 + dm1.T)
    return dm1


def load_rdms_from_npz(path, n_orbs, n_roots):
    """Per-root 1-RDMs from a user npz: stacked array or dm1_root<i> keys."""
    with np.load(path, allow_pickle=False) as d:
        keys = list(d.keys())
        for key in ("dm1", "rdm1", "dm1_roots", "rdms"):
            if key in d:
                arr = np.atleast_3d(np.asarray(d[key], dtype=float))
                if arr.shape[-2:] != (n_orbs, n_orbs):
                    raise ValueError(
                        f"{path}['{key}'] has shape {arr.shape}, "
                        f"expected (n_roots, {n_orbs}, {n_orbs})"
                    )
                return {i: arr[i] for i in range(min(len(arr), n_roots))}

        found = {}
        for i in range(n_roots):
            for key in (f"dm1_root{i}", f"rdm1_root{i}", f"dm1_{i}"):
                if key in d:
                    found[i] = np.asarray(d[key], dtype=float)
                    break
        if not found:
            raise KeyError(f"{path}: no 1-RDM arrays found (keys: {keys})")
        for i, dm in found.items():
            if dm.shape != (n_orbs, n_orbs):
                raise ValueError(
                    f"{path}['dm1_root{i}'] has shape {dm.shape}, expected ({n_orbs}, {n_orbs})"
                )
        return found


def load_rdms(args):
    """Collect {root: 1-RDM} from whichever source was requested."""
    if args.rdm_npz:
        log(f"1-RDM source: {args.rdm_npz}")
        dms = load_rdms_from_npz(args.rdm_npz, args.n_orbs, args.n_roots)
    else:
        rdm_dir = args.rdm_dir
        if not os.path.isdir(rdm_dir):
            raise FileNotFoundError(f"--rdm-dir not a directory: {rdm_dir}")
        stem = "spatial_onepdm" if args.rdm_kind == "onepdm" else "spatial_twopdm"
        log(f"1-RDM source: {rdm_dir}/{stem}.<i>.<i>.txt")
        dms = {}
        for iroot in range(args.n_roots):
            path = os.path.join(rdm_dir, f"{stem}.{iroot}.{iroot}.txt")
            if not os.path.exists(path):
                log(f"  root {iroot}: {os.path.basename(path)} NOT FOUND -- skipping")
                continue
            if args.rdm_kind == "onepdm":
                dms[iroot] = read_onepdm_text(path, args.n_orbs)
            else:
                dms[iroot] = read_twopdm_text(path, args.n_orbs, args.n_elec)

    for iroot in sorted(dms):
        tr = np.trace(dms[iroot])
        flag = "" if abs(tr - args.n_elec) < 0.1 else "   <-- WARNING: trace error"
        log(f"  root {iroot}: 1-RDM trace={tr:.6f}  (expect {args.n_elec}){flag}")
    return dms


def find_dmrg_dir(root):
    """Locate the block2 run dir (has FCIDUMP / dmrg.conf / dmrg.out) under root."""
    markers = ("FCIDUMP", "dmrg.conf", "dmrg.out")
    if any(os.path.exists(os.path.join(root, m)) for m in markers):
        return root
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if any(m in files for m in markers):
            return base
    return None


def find_rdm_dir(dmrg_dir, kind):
    """Locate the dir holding spatial_*pdm.<i>.<i>.txt (usually node0/)."""
    stem = f"spatial_{kind}"
    for cand in (os.path.join(dmrg_dir, "node0"), dmrg_dir):
        if os.path.isdir(cand) and any(f.startswith(stem) for f in os.listdir(cand)):
            return cand
    for base, dirs, files in os.walk(dmrg_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if any(f.startswith(stem) for f in files):
            return base
    return None


def count_npz_roots(path):
    """Number of roots in a prepared bundle npz, from its keys alone.

    The bundle written by 029_prepare_dmrg_inputs.py is self-describing, so
    --n-roots need not be supplied (nor guessed from a default) when it is used.
    """
    try:
        with np.load(path) as d:
            keys = list(d.keys())
            for key in ("dm1", "rdm1", "dm1_roots", "rdms"):
                if key in keys:
                    arr = d[key]
                    if arr.ndim == 3:
                        return int(arr.shape[0])
                    if arr.ndim == 2:
                        return 1
            n = sum(1 for k in keys if k.startswith("dm1_root") and k[len("dm1_root") :].isdigit())
            return n or None
    except (OSError, ValueError, KeyError):
        return None


def count_rdm_roots(rdm_dir, kind):
    """Highest contiguous root index with a diagonal RDM file present."""
    n = 0
    while os.path.exists(os.path.join(rdm_dir, f"spatial_{kind}.{n}.{n}.txt")):
        n += 1
    return n


def find_energies(*dirs):
    """First dmrg_states*.npz / e_states*.npz / energies.txt found in dirs."""
    import glob as _glob

    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for pat in ("dmrg_states*.npz", "e_states*.npz", "energies.txt", "energies.dat"):
            hits = sorted(_glob.glob(os.path.join(d, pat)))
            if hits:
                return hits[0]
    return None


def read_fcidump_header(path):
    """Return (norb, nelec) from a FCIDUMP header, or (None, None)."""
    norb = nelec = None
    try:
        with open(path) as fh:
            for line in fh:
                up = line.upper()
                for key, setter in (("NORB", "norb"), ("NELEC", "nelec")):
                    if key + "=" in up.replace(" ", ""):
                        for part in up.replace("&FCI", "").split(","):
                            pk = part.replace(" ", "")
                            if pk.startswith(key + "="):
                                val = int(pk.split("=")[1])
                                if setter == "norb":
                                    norb = val
                                else:
                                    nelec = val
                if "&END" in up or "/" == line.strip():
                    break
    except (OSError, ValueError) as exc:
        log(f"  could not parse FCIDUMP header {path}: {exc}")
    return norb, nelec


def build_parser():
    p = argparse.ArgumentParser(
        description="DMRG excited-state observables (dipoles, CT) from 1-RDMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the module docstring for install instructions and examples.",
    )
    p.add_argument(
        "--geom", default=DEF_GEOM, help=f"geometry .xyz in Angstrom (default: {DEF_GEOM})"
    )
    p.add_argument(
        "--basis",
        default=DEF_BASIS,
        help=f"basis set, must match the DMRG run (default: {DEF_BASIS})",
    )
    p.add_argument(
        "--mo", default=DEF_MO, help=f"MO coefficients, .npz or .pkl (default: {DEF_MO})"
    )
    p.add_argument(
        "--n-elec",
        type=int,
        default=None,
        help=f"active electrons (default: FCIDUMP, else {DEF_N_ELEC})",
    )
    p.add_argument(
        "--n-orbs",
        type=int,
        default=None,
        help=f"active orbitals (default: FCIDUMP, else {DEF_N_ORBS})",
    )
    p.add_argument(
        "--n-roots",
        type=int,
        default=None,
        help="number of roots (default: from --energies, --rdm-npz, "
        f"or the RDM files present, else {DEF_N_ROOTS})",
    )
    p.add_argument(
        "--ncore",
        type=int,
        default=None,
        help="frozen-core orbital count (default: nelectron//2 - n_elec//2)",
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--dmrg-dir",
        default=None,
        help="DMRG run directory; RDM files, FCIDUMP and node0/ are "
        f"auto-discovered inside it (default: {DEF_DMRG_DIR})",
    )
    src.add_argument(
        "--rdm-dir", default=None, help="exact dir holding spatial_*pdm.<i>.<i>.txt files"
    )
    src.add_argument(
        "--rdm-npz", default=None, help="npz with per-root 1-RDMs, instead of the text files"
    )
    p.add_argument(
        "--rdm-kind",
        choices=("twopdm", "onepdm"),
        default=DEF_RDM_KIND,
        help=f"block2 RDM file type (default: {DEF_RDM_KIND})",
    )

    p.add_argument(
        "--energies",
        default=None,
        help="total energies in Ha: comma list, text file, or .npz "
        "(default: any dmrg_states*.npz / energies.txt next to "
        "the RDMs)",
    )
    p.add_argument(
        "--ct-fragments",
        default=None,
        help="'PD1:0-10,...;PD2:11-21,...' 0-based xyz atom indices "
        "(default: the PD1/PD2 split, applied only when the "
        f"geometry has {DEF_CT_NATOM} atoms)",
    )
    p.add_argument(
        "--no-ct", action="store_true", help="skip the charge-transfer analysis entirely"
    )
    p.add_argument("--charge", type=int, default=0, help="molecular charge (default: 0)")
    p.add_argument("--spin", type=int, default=0, help="2S, unpaired electrons (default: 0)")
    p.add_argument(
        "--max-memory", type=int, default=4000, help="pyscf max_memory in MB (default: 4000)"
    )
    p.add_argument("--out-prefix", default="dmrg_obs", help="output file prefix")
    p.add_argument(
        "--check", action="store_true", help="validate inputs and print the setup, then stop"
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    section("DMRG output")
    if not (args.dmrg_dir or args.rdm_dir or args.rdm_npz):
        args.dmrg_dir = DEF_DMRG_DIR
        log(f"No RDM source given -- trying default --dmrg-dir {DEF_DMRG_DIR}")
        if not os.path.isdir(args.dmrg_dir):
            raise FileNotFoundError(
                f"default DMRG directory '{DEF_DMRG_DIR}' not found in {os.getcwd()}. "
                "Pass --dmrg-dir <your DMRG run dir>, or --rdm-npz <file.npz> if "
                "your code writes 1-RDMs to an npz."
            )

    resolved_rdm_dir = args.rdm_dir
    if args.dmrg_dir:
        if not os.path.isdir(args.dmrg_dir):
            raise FileNotFoundError(f"--dmrg-dir not a directory: {args.dmrg_dir}")
        run_dir = find_dmrg_dir(args.dmrg_dir) or args.dmrg_dir
        log(f"DMRG run dir : {run_dir}")
        resolved_rdm_dir = find_rdm_dir(args.dmrg_dir, args.rdm_kind)
        if resolved_rdm_dir is None:
            raise FileNotFoundError(
                f"no spatial_{args.rdm_kind}.<i>.<i>.txt found under "
                f"{args.dmrg_dir}. Check --rdm-kind, or point --rdm-dir at the "
                f"directory holding them (block2 usually writes them to node0/)."
            )
        log(f"RDM dir      : {resolved_rdm_dir}")

        fcidump = os.path.join(run_dir, "FCIDUMP")
        if os.path.exists(fcidump):
            norb, nelec = read_fcidump_header(fcidump)
            if norb and args.n_orbs is None:
                args.n_orbs = norb
                log(f"n_orbs       : {norb}   (from FCIDUMP)")
            if nelec and args.n_elec is None:
                args.n_elec = nelec
                log(f"n_elec       : {nelec}   (from FCIDUMP)")
    elif args.rdm_dir:
        log(f"RDM dir      : {args.rdm_dir}")
    elif args.rdm_npz:
        log(f"RDM npz      : {args.rdm_npz}")
    args.rdm_dir = resolved_rdm_dir

    if args.n_orbs is None:
        args.n_orbs = DEF_N_ORBS
        log(f"n_orbs       : {args.n_orbs}   (default, no FCIDUMP)")
    if args.n_elec is None:
        args.n_elec = DEF_N_ELEC
        log(f"n_elec       : {args.n_elec}   (default, no FCIDUMP)")
    if args.n_elec <= 0 or args.n_orbs <= 0:
        raise ValueError("--n-elec and --n-orbs must be positive")
    if args.n_elec > 2 * args.n_orbs:
        raise ValueError(
            f"CAS({args.n_elec}e,{args.n_orbs}o) is impossible: "
            f"{args.n_elec} electrons need at least {-(-args.n_elec // 2)} orbitals"
        )

    section("Setup")
    geom_str, natom, composition = read_xyz(args.geom)
    log(f"Geometry     : {args.geom}")
    log(f"Atoms parsed : {natom}")
    log("Composition  : " + "  ".join(f"{k}{v}" for k, v in sorted(composition.items())))
    log(f"Basis        : {args.basis}")

    if args.energies is None:
        found = find_energies(
            args.rdm_dir,
            args.dmrg_dir,
            os.path.dirname(args.rdm_dir or "") or None,
            os.getcwd(),
        )
        if found:
            args.energies = found
            log(f"Energies     : {found}   (auto-discovered)")

    e_all = load_energies(args.energies, 1) if args.energies else None
    if args.n_roots is None:
        if e_all is not None:
            args.n_roots = len(e_all)
        elif args.rdm_npz:
            args.n_roots = count_npz_roots(args.rdm_npz) or 1
            log(f"n_roots      : {args.n_roots}   (from --rdm-npz contents)")
        elif args.rdm_dir:
            args.n_roots = count_rdm_roots(args.rdm_dir, args.rdm_kind) or 1
            log(f"n_roots      : {args.n_roots}   (from RDM files present)")
        else:
            args.n_roots = DEF_N_ROOTS
            log(f"n_roots      : {args.n_roots}   (built-in default -- pass --n-roots to override)")
    e_states = e_all[: args.n_roots] if e_all is not None else None
    if e_states is not None and len(e_states) < args.n_roots:
        raise ValueError(f"got {len(e_states)} energies but --n-roots={args.n_roots}")
    log(f"Active space : CAS({args.n_elec}e,{args.n_orbs}o)  n_roots={args.n_roots}")

    mo_coeff, mo_src = load_mo_coeff(args.mo)
    log(f"MO coeff     : {args.mo}  [{mo_src}]  shape={mo_coeff.shape}")

    ct_spec = args.ct_fragments
    if args.no_ct:
        ct_spec = None
    elif ct_spec is None:
        # The PD1/PD2 default describes system1 only. Applying it to a different
        # geometry silently dumps most of the density into 'rest', so gate it on
        # the atom count matching.
        if natom == DEF_CT_NATOM:
            ct_spec = DEF_CT_FRAGMENTS
        else:
            log(
                f"CT fragments : skipped -- geometry has {natom} atoms, the "
                f"built-in PD1/PD2 split describes {DEF_CT_NATOM}. "
                "Pass --ct-fragments to define your own."
            )

    frags = parse_ct_fragments(ct_spec) if ct_spec else None
    if frags:
        src = "given" if args.ct_fragments else "PD1/PD2 default"
        log(
            f"CT fragments : [{src}] " + ", ".join(f"{k}({len(v)} atoms)" for k, v in frags.items())
        )
        assigned = [i for idxs in frags.values() for i in idxs]
        if len(assigned) != len(set(assigned)):
            raise ValueError("--ct-fragments assigns an atom to two fragments")
        out_of_range = [i for i in assigned if not 0 <= i < natom]
        if out_of_range:
            raise ValueError(
                f"--ct-fragments atom indices outside 0..{natom - 1}: {out_of_range[:10]}"
            )
        n_rest = natom - len(set(assigned))
        if n_rest:
            log(f"               {n_rest} unassigned atoms -> fragment 'rest'")

    from pyscf import gto

    mol = gto.Mole()
    mol.atom = geom_str
    mol.basis = args.basis
    mol.charge = args.charge
    mol.spin = args.spin
    mol.symmetry = False
    mol.verbose = 0
    mol.max_memory = args.max_memory
    mol.build()
    log(f"Molecule     : {mol.nao} AOs, {mol.nelectron} electrons")

    if mo_coeff.shape[0] != mol.nao:
        raise ValueError(
            f"--mo has {mo_coeff.shape[0]} AO rows but this geometry/basis gives "
            f"{mol.nao} AOs -- wrong basis or wrong geometry?"
        )

    ncore = args.ncore
    if ncore is None:
        ncore = mol.nelectron // 2 - args.n_elec // 2
    if ncore < 0 or ncore + args.n_orbs > mo_coeff.shape[1]:
        raise ValueError(
            f"active window [{ncore}, {ncore + args.n_orbs}) does not fit in "
            f"{mo_coeff.shape[1]} MOs"
        )
    log(f"Core orbitals: {ncore}   active MO columns {ncore}..{ncore + args.n_orbs - 1}")

    if args.check:
        log("--check: inputs valid, stopping before integrals")
        return 0

    section("1-RDMs")
    dm1_roots = load_rdms(args)
    if not dm1_roots:
        log("ERROR: no 1-RDMs could be loaded -- nothing to do")
        return 1
    if len(dm1_roots) < args.n_roots:
        log(f"NOTE: {len(dm1_roots)} of {args.n_roots} roots had RDMs; reporting only those")
    if e_states is None:
        log("No --energies given: excitation energies will show as 0")
        e_states = np.zeros(args.n_roots)
    delta_eV = (e_states - e_states[0]) * EV

    section("Permanent dipoles")
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore : ncore + args.n_orbs]

    with mol.with_common_orig([0, 0, 0]):
        dip_ao = mol.intor("int1e_r", comp=3)

    mu_nuc = np.einsum(
        "a,ax->x",
        np.array([mol.atom_charge(i) for i in range(mol.natm)]),
        mol.atom_coords(),
    )
    mu_core = -np.einsum("xpq,pq->x", dip_ao, 2.0 * mo_core @ mo_core.T)
    dip_cas = np.einsum("xpq,pi,qj->xij", dip_ao, mo_cas, mo_cas)

    mu_states = {
        i: mu_nuc + mu_core + -np.einsum("xij,ij->x", dip_cas, dm) for i, dm in dm1_roots.items()
    }
    log(f"mu_nuclear (Debye)   : {np.round(mu_nuc * AU2DEBYE, 4).tolist()}")
    log(f"mu_frozen-core (D)   : {np.round(mu_core * AU2DEBYE, 4).tolist()}")

    emit("\nPermanent dipole per state (Debye):")
    emit(
        f"\n  {'St':>3}  {'dE(eV)':>8}  {'lam(nm)':>8}"
        f"  {'mu_x':>9}  {'mu_y':>9}  {'mu_z':>9}  {'|mu|(D)':>9}  {'|dmu|(D)':>9}"
    )
    emit(f"  {'-' * 85}")
    mu_ref = mu_states.get(0)
    for iroot in sorted(mu_states):
        de = delta_eV[iroot]
        nm = f"{NM_EV / de:8.1f}" if de > 0.01 else "       -"
        mu_D = mu_states[iroot] * AU2DEBYE
        dmu = (mu_states[iroot] - mu_ref) * AU2DEBYE if mu_ref is not None else np.full(3, np.nan)
        emit(
            f"  S{iroot:<2}  {de:>8.4f}  {nm}"
            f"  {mu_D[0]:>9.4f}  {mu_D[1]:>9.4f}  {mu_D[2]:>9.4f}"
            f"  {np.linalg.norm(mu_D):>9.4f}  {np.linalg.norm(dmu):>9.4f}"
        )
    if mu_ref is None:
        emit("\n  NOTE: root 0 absent, |dmu| undefined")

    dip_path = f"{args.out_prefix}_dipole.npz"
    np.savez(
        dip_path,
        e_states=e_states,
        delta_eV=delta_eV,
        mu_nuc_au=mu_nuc,
        mu_core_au=mu_core,
        **{f"mu_au_root{i}": mu for i, mu in mu_states.items()},
        **{f"mu_debye_root{i}": mu * AU2DEBYE for i, mu in mu_states.items()},
        **{f"dm1_root{i}": dm for i, dm in dm1_roots.items()},
    )
    log(f"Dipole results -> {dip_path}")

    if frags:
        section("Mulliken fragment charges (CT)")
        S = mol.intor_symmetric("int1e_ovlp")
        ao_atom = [lbl[0] for lbl in mol.ao_labels(fmt=False)]

        atom2frag = {a: name for name, idxs in frags.items() for a in idxs}
        names = list(frags) + (["rest"] if len(atom2frag) < natom else [])

        q_roots = {}
        for iroot, dm1 in dm1_roots.items():
            q_ao = np.einsum("ij,ji->i", S, mo_cas @ dm1 @ mo_cas.T)
            q = dict.fromkeys(names, 0.0)
            for i, atom in enumerate(ao_atom):
                q[atom2frag.get(atom, "rest")] += q_ao[i]
            q_roots[iroot] = q

        emit("\nMulliken fragment electron counts (CAS contribution):")
        hdr = f"  {'St':>3}  {'dE(eV)':>8}  " + "  ".join(f"{n:>10}" for n in names)
        emit(hdr)
        emit("  " + "-" * (len(hdr) - 2))
        for iroot in sorted(q_roots):
            vals = "  ".join(f"{q_roots[iroot][n]:>10.4f}" for n in names)
            emit(f"  S{iroot:<2}  {delta_eV[iroot]:>8.4f}  {vals}")

        if 0 in q_roots:
            emit("\n  Delta-q vs S0 (positive = gained electrons):")
            hdr2 = f"  {'St':>3}  {'dE(eV)':>8}  " + "  ".join(f"d{n:>9}" for n in names)
            emit(hdr2)
            emit("  " + "-" * (len(hdr2) - 2))
            for iroot in sorted(q_roots):
                if iroot == 0:
                    continue
                dvals = "  ".join(f"{q_roots[iroot][n] - q_roots[0][n]:>+10.4f}" for n in names)
                emit(f"  S{iroot:<2}  {delta_eV[iroot]:>8.4f}  {dvals}")
        else:
            emit("\n  NOTE: root 0 absent, delta-q not computed")

        ct_path = f"{args.out_prefix}_ct.npz"
        np.savez(
            ct_path,
            frag_names=np.array(names),
            e_states=e_states,
            delta_eV=delta_eV,
            **{f"q_frag_root{i}": np.array([q[n] for n in names]) for i, q in q_roots.items()},
            **(
                {
                    f"dq_frag_root{i}": np.array([q[n] - q_roots[0][n] for n in names])
                    for i, q in q_roots.items()
                }
                if 0 in q_roots
                else {}
            ),
        )
        log(f"CT results -> {ct_path}")

    summary_path = f"{args.out_prefix}_summary.txt"
    with open(summary_path, "w") as fh:
        fh.write(f"# {os.path.basename(sys.argv[0])}\n")
        fh.write(f"# geom={args.geom} basis={args.basis} mo={args.mo}\n")
        fh.write(f"# CAS({args.n_elec}e,{args.n_orbs}o) roots_used={len(dm1_roots)}\n")
        fh.write("\n".join(_report) + "\n")
    log(f"Summary -> {summary_path}")
    log(f"Total wall time: {time.time() - _t0:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, FileNotFoundError) as exc:
        sys.stderr.write(f"\nERROR: {exc}\n")
        sys.exit(2)
