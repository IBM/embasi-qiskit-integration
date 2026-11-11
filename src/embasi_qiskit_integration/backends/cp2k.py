# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""CP2K backend (``--qm_code cp2k``): CP2K's own active space, run and imported.

**Licence boundary.**  CP2K is GPL-2.0-or-later; this package is Apache-2.0.  CP2K is
therefore only ever *run as an external program* (``cp2k.ssmp``/``cp2k.psmp``): this
module writes its input *text*, runs it, and reads the files it writes.  No CP2K code
(nor its data files: basis sets and pseudopotentials are passed by path) is imported,
linked or vendored.

**What CP2K can supply** (checked against its source and real runs).  CP2K has a
Gaussian basis, but no ASI implementation and no runtime AO-matrix API: it cannot take
an AO potential or density matrix during its SCF, nor add the ``mu P_B`` projector, and
EmbASI has no adapter for it.  So
:func:`build_cp2k_backend` refuses this package's projection embedding.  What CP2K *does*
have is a native active-space module (``&ACTIVE_SPACE``): it builds an active-space
Hamiltonian over its own Kohn-Sham orbitals and writes it as an FCIDUMP, which
:func:`run_active_space` produces and ``embasi-qiskit-integration import_fcidump`` hands
to our FCI/SQD solvers.

That is a *different* embedding from ours: the active orbitals are CP2K's (canonical KS
orbitals by default), and the one-body term is CP2K's inactive Fock operator.  Its
energies are not comparable with a projection-embedding run.

Input pitfalls this module handles (each one aborted a real CP2K run):

* ``&ACTIVE_SPACE`` moved between versions -- under ``&DFT &PRINT`` in CP2K 7.1, under
  ``&DFT`` later.  :func:`active_space_section` asks the installed executable
  (``--xml``) instead of guessing from the version number;
* the active space needs virtual orbitals, so ``ADDED_MOS`` must cover them (otherwise
  ``CPASSERT`` in ``qs_active_space_methods``);
* **CP2K 7.1 drops every negative two-electron integral**: its ERI screen is the signed
  ``erint > EPS_INTEGRAL`` (later releases use ``ABS``).  Which integrals are negative
  depends on arbitrary orbital phases, so the FCIDUMP was silently wrong and its FCI
  energy even varied with the OpenMP thread count (water CAS(4e,4o): -17.3218 or
  -17.4807 Ha against the phase-invariant -17.4884 Ha).  :class:`ActiveSpaceSpec` writes
  a negative ``EPS_INTEGRAL``, and :func:`dropped_negative_integrals` flags a file that
  still shows the symptom;
* the ``&ACTIVE_SPACE`` keywords themselves changed: up to 2023.x ``INACTIVE_ELECTRONS``
  exists and, under UKS, ``ACTIVE_ORBITALS``/``INACTIVE_ELECTRONS`` are *per-spin lists*;
  from 2024.1 ``INACTIVE_ELECTRONS`` is gone (the inactive count is total minus active)
  and ``ACTIVE_ORBITALS`` takes one integer; ``ISOLATED_SYSTEM`` is gone from 2026.1
  (``&ERI PERIODICITY`` replaces it).  CP2K's parser aborts on an unknown keyword, so
  :func:`active_space_layout` reads the keyword set from the same ``--xml`` description
  and only keywords the installed CP2K has are written.  CP2K 7.1's UKS active space
  still aborts ("Spin option not yet available"): its module is restricted-only.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embasi_qiskit_integration.backends.registry import QMBackend, register_backend

#: Environment variable naming the directory with CP2K's ``BASIS_MOLOPT``/``GTH_POTENTIALS``.
DATA_ENV = "CP2K_DATA_DIR"
#: Environment variable naming the CP2K executable (default: first on PATH).
EXE_ENV = "CP2K_EXE"
_EXE_CANDIDATES = ("cp2k.psmp", "cp2k.ssmp", "cp2k.popt", "cp2k.sopt", "cp2k")


def find_executable() -> str:
    """The CP2K executable: ``$CP2K_EXE``, else the first ``cp2k.*`` on PATH."""
    exe = os.environ.get(EXE_ENV)
    if exe:
        return exe
    for name in _EXE_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(f"no CP2K executable: put cp2k.ssmp/psmp on PATH or set ${EXE_ENV}")


# --------------------------------------------------------------------------- #
# Where &ACTIVE_SPACE lives in the installed CP2K
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActiveSpaceLayout:
    """Where ``&ACTIVE_SPACE`` lives in the installed CP2K, and which keywords it takes.

    ``keywords`` / ``eri_keywords`` are the keyword names of ``&ACTIVE_SPACE`` and of its
    ``&ERI`` subsection; ``orbitals_per_spin`` says whether ``ACTIVE_ORBITALS`` takes a
    list (one value per spin, up to 2023.x) or a single integer (2024.1 on).
    """

    section: tuple[str, ...]
    keywords: frozenset[str]
    eri_keywords: frozenset[str] = frozenset()
    orbitals_per_spin: bool = True
    #: ``(major, minor)`` from the description's ``CP2K_VERSION``; ``None`` if absent.
    version: tuple[int, int] | None = None


def _parse_version(text: str | None) -> tuple[int, int] | None:
    """``"CP2K version 7.1"`` -> ``(7, 1)``; ``"CP2K version 2024.1"`` -> ``(2024, 1)``."""
    # Anchored on "version": the "2" of "CP2K" is not the version number.
    m = re.search(r"version\s+(\d+)(?:\.(\d+))?", text or "", flags=re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2) or 0)) if m else None


#: CP2K 7.1-2023.x: what this module wrote before it asked the executable.
LEGACY_KEYWORDS = frozenset(
    {"ACTIVE_ELECTRONS", "INACTIVE_ELECTRONS", "ACTIVE_ORBITALS", "ISOLATED_SYSTEM"}
)


def _keyword_names(section: ET.Element) -> frozenset[str]:
    return frozenset(
        (name.text or "").strip()
        for keyword in section.findall("KEYWORD")
        for name in keyword.findall("NAME")
    )


def active_space_layout_in_xml(xml_path: str | Path) -> ActiveSpaceLayout:
    """The :class:`ActiveSpaceLayout` of a ``cp2k --xml`` input description."""

    def walk(node: ET.Element, path: tuple[str, ...]) -> tuple[ET.Element, tuple] | None:
        for child in node.findall("SECTION"):
            name = (child.findtext("NAME") or "").strip()
            if name == "ACTIVE_SPACE":
                return child, path
            found = walk(child, (*path, name))
            if found is not None:
                return found
        return None

    root = ET.parse(xml_path).getroot()
    found = walk(root, ())
    if found is None:
        raise ValueError(f"{xml_path}: this CP2K has no ACTIVE_SPACE section")
    node, path = found
    keywords = _keyword_names(node)
    eri = next(
        (c for c in node.findall("SECTION") if (c.findtext("NAME") or "").strip() == "ERI"),
        None,
    )
    orbitals_per_spin = True
    for keyword in node.findall("KEYWORD"):
        if any((n.text or "").strip() == "ACTIVE_ORBITALS" for n in keyword.findall("NAME")):
            n_var = keyword.findtext("DATA_TYPE/N_VAR") or keyword.findtext("N_VAR")
            orbitals_per_spin = n_var is None or int(n_var) != 1
    return ActiveSpaceLayout(
        section=path,
        # A description without keyword entries (a hand-written stub) keeps the legacy set.
        keywords=keywords or LEGACY_KEYWORDS,
        eri_keywords=_keyword_names(eri) if eri is not None else frozenset(),
        orbitals_per_spin=orbitals_per_spin,
        version=_parse_version(root.findtext("CP2K_VERSION")),
    )


def active_space_section_in_xml(xml_path: str | Path) -> tuple[str, ...]:
    """The section path of ``ACTIVE_SPACE`` in a ``cp2k --xml`` input description.

    Returns e.g. ``("FORCE_EVAL", "DFT", "PRINT")`` (CP2K 7.1) or ``("FORCE_EVAL",
    "DFT")`` (later releases): the sections enclosing it, outermost first.
    """
    return active_space_layout_in_xml(xml_path).section


def active_space_layout(executable: str | None = None) -> ActiveSpaceLayout:
    """Ask the installed CP2K for its ``&ACTIVE_SPACE`` layout (runs ``<exe> --xml``)."""
    exe = executable or find_executable()
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([exe, "--xml"], cwd=tmp, check=True, capture_output=True, timeout=300)
        return active_space_layout_in_xml(Path(tmp) / "cp2k_input.xml")


def active_space_section(executable: str | None = None) -> tuple[str, ...]:
    """Ask the installed CP2K where ``&ACTIVE_SPACE`` goes (runs ``<exe> --xml``)."""
    return active_space_layout(executable).section


# --------------------------------------------------------------------------- #
# Input generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActiveSpaceSpec:
    """A CP2K ``&ACTIVE_SPACE`` request.  Electron counts are *valence* electrons."""

    n_active_electrons: int
    n_active_orbitals: int
    n_valence_electrons: int
    multiplicity: int = 1
    eri_method: str = "FULL_GPW"
    # Keep *every* two-electron integral.  CP2K 7.1 screens with the signed test
    # `erint > EPS_INTEGRAL`, which silently drops all negative integrals (their sign is
    # the arbitrary orbital phase, so which ones vanish even changes with OpenMP thread
    # count); later releases compare ABS(erint).  A negative threshold keeps them all on
    # both, at the cost of also writing exact zeros -- negligible for an active space.
    eps_integral: float = -1.0e10

    @property
    def unrestricted(self) -> bool:
        return self.multiplicity != 1

    def spin_split(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """``((n_act_alpha, n_act_beta), (n_inact_alpha, n_inact_beta))``.

        The unpaired electrons (``multiplicity - 1``) are all active; the inactive
        electrons fill doubly occupied orbitals, so they split evenly.
        """
        n_unpaired = self.multiplicity - 1
        n_inactive = self.n_valence_electrons - self.n_active_electrons
        if n_inactive < 0 or n_inactive % 2:
            raise ValueError(
                f"{self.n_valence_electrons} valence electrons with {self.n_active_electrons} "
                "active leave an odd or negative inactive count; inactive orbitals are "
                "doubly occupied"
            )
        if (self.n_active_electrons - n_unpaired) % 2 or self.n_active_electrons < n_unpaired:
            raise ValueError(
                f"{self.n_active_electrons} active electrons cannot hold multiplicity "
                f"{self.multiplicity}"
            )
        n_b = (self.n_active_electrons - n_unpaired) // 2
        return (n_b + n_unpaired, n_b), (n_inactive // 2, n_inactive // 2)


def valence_from_potential(potential: str) -> int:
    """Valence electrons of a GTH potential name: ``GTH-PBE-q6`` -> 6."""
    m = re.search(r"-q(\d+)$", potential)
    if not m:
        raise ValueError(f"cannot read the valence charge from potential {potential!r}")
    return int(m.group(1))


def active_space_input(
    symbols: Sequence[str],
    positions: Sequence[Sequence[float]],
    *,
    spec: ActiveSpaceSpec,
    basis: dict[str, str],
    potential: dict[str, str],
    data_dir: str | Path,
    cell: float = 10.0,
    xc: str = "PBE",
    cutoff: float = 300.0,
    section: tuple[str, ...] = ("FORCE_EVAL", "DFT"),
    project: str = "as",
    fcidump: str = "active.fcidump",
    added_mos: int | None = None,
    charge: int = 0,
    layout: ActiveSpaceLayout | None = None,
) -> str:
    """CP2K input text for an isolated-molecule SCF followed by an active-space FCIDUMP.

    ``positions`` are in Angstrom inside a cubic ``cell`` (Angstrom) with no periodicity
    (Martyna-Tuckerman Poisson solver).  ``layout`` (from :func:`active_space_layout`)
    says where ``&ACTIVE_SPACE`` goes and which of its keywords the installed CP2K has;
    without it, ``section`` places the block and the 7.1-2023.x keywords are written.
    """
    if layout is None:
        layout = ActiveSpaceLayout(section=tuple(section), keywords=LEGACY_KEYWORDS)
    section = layout.section
    if tuple(section[:2]) != ("FORCE_EVAL", "DFT") or len(section) not in (2, 3):
        raise ValueError(f"unsupported ACTIVE_SPACE location {section}")
    under_print = len(section) == 3
    if under_print and section[2] != "PRINT":
        raise ValueError(f"unsupported ACTIVE_SPACE location {section}")
    (act_a, act_b), (inact_a, inact_b) = spec.spin_split()
    n_virtual = spec.n_active_orbitals - min(act_a, act_b)
    data_dir = Path(data_dir)

    keywords = layout.keywords
    per_spin = spec.unrestricted and layout.orbitals_per_spin
    n_orb = spec.n_active_orbitals
    active = f"ACTIVE_ELECTRONS {spec.n_active_electrons}\n"
    if "INACTIVE_ELECTRONS" in keywords:
        inactive = f"{inact_a} {inact_b}" if per_spin else f"{inact_a + inact_b}"
        active += f"INACTIVE_ELECTRONS {inactive}\n"
    active += f"ACTIVE_ORBITALS {f'{n_orb} {n_orb}' if per_spin else n_orb}\n"
    if "ISOLATED_SYSTEM" in keywords:
        active += "ISOLATED_SYSTEM T\n"
    spin = f"    UKS\n    MULTIPLICITY {spec.multiplicity}\n" if spec.unrestricted else ""
    # Where ISOLATED_SYSTEM is gone, the ERI periodicity says the same (0 0 0 is its
    # default; written explicitly so the isolated treatment never rests on a default).
    periodicity = (
        "  PERIODICITY 0 0 0\n"
        if "ISOLATED_SYSTEM" not in keywords and "PERIODICITY" in layout.eri_keywords
        else ""
    )
    block = (
        "&ACTIVE_SPACE\n" + active + f"&ERI\n  METHOD {spec.eri_method}\n"
        f"  EPS_INTEGRAL {spec.eps_integral:.1E}\n" + periodicity + "&END ERI\n"
        f"&FCIDUMP ON\n  FILENAME ={fcidump}\n&END FCIDUMP\n"
        "&END ACTIVE_SPACE\n"
    )
    indent = "      " if under_print else "    "
    block = "".join(indent + line + "\n" for line in block.splitlines())
    if under_print:
        block = "    &PRINT\n" + block + "    &END PRINT\n"

    coords = "".join(
        f"      {s} {x:.10f} {y:.10f} {z:.10f}\n"
        for s, (x, y, z) in zip(symbols, positions, strict=True)
    )
    kinds = "".join(
        f"    &KIND {el}\n      BASIS_SET {basis[el]}\n      POTENTIAL {potential[el]}\n"
        "    &END KIND\n"
        for el in sorted(set(symbols))
    )
    return (
        "&GLOBAL\n"
        f"  PROJECT {project}\n  RUN_TYPE ENERGY\n  PRINT_LEVEL LOW\n"
        "&END GLOBAL\n"
        "&FORCE_EVAL\n  METHOD QS\n  &DFT\n"
        f"    BASIS_SET_FILE_NAME {data_dir / 'BASIS_MOLOPT'}\n"
        f"    POTENTIAL_FILE_NAME {data_dir / 'GTH_POTENTIALS'}\n"
        + spin
        + (f"    CHARGE {int(charge)}\n" if charge else "")
        + f"    &MGRID\n      CUTOFF {cutoff}\n    &END MGRID\n"
        "    &POISSON\n      PERIODIC NONE\n      PSOLVER MT\n    &END POISSON\n"
        "    &SCF\n      EPS_SCF 1.0E-8\n      MAX_SCF 100\n"
        f"      ADDED_MOS {added_mos if added_mos is not None else max(n_virtual, 1)}\n"
        "    &END SCF\n"
        f"    &XC\n      &XC_FUNCTIONAL {xc}\n      &END XC_FUNCTIONAL\n    &END XC\n"
        + block
        + "  &END DFT\n"
        "  &SUBSYS\n"
        f"    &CELL\n      ABC {cell} {cell} {cell}\n      PERIODIC NONE\n    &END CELL\n"
        "    &COORD\n" + coords + "    &END COORD\n" + kinds + "  &END SUBSYS\n"
        "&END FORCE_EVAL\n"
    )


# --------------------------------------------------------------------------- #
# Running it, and reading what it wrote
# --------------------------------------------------------------------------- #
def read_total_energy(output: str | Path) -> float:
    """CP2K's final ``ENERGY| Total FORCE_EVAL`` value (Hartree).

    Every spelling CP2K has used: ``... energy (a.u.):  <value>`` (7.1),
    ``... energy [a.u.]:  <value>`` (8.1-2024.1) and ``... energy [hartree]  <value>``
    (no colon, 2025.1 on).  Anchored on the unit so neighbouring ``gap`` / ``free
    energy`` lines never match.
    """
    hits = re.findall(
        r"ENERGY\|\s+Total FORCE_EVAL\s*\(\s*\w+\s*\)\s+energy\s*"
        r"(?:\(a\.u\.\)\s*:|\[a\.u\.\]\s*:|\[hartree\])\s*"
        r"(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)",
        Path(output).read_text(),
        flags=re.IGNORECASE,
    )
    if not hits:
        raise ValueError(f"{output} has no 'ENERGY| Total FORCE_EVAL' line")
    return float(hits[-1].replace("D", "E").replace("d", "e"))


def abort_message(output: str | Path) -> str | None:
    """CP2K's ``[ABORT]`` message from an output file, or ``None`` if it did not abort.

    CP2K frames it in ASCII art: the text sits on the two lines after ``[ABORT]``.
    """
    output = Path(output)
    lines = output.read_text().splitlines() if output.exists() else []
    for i, line in enumerate(lines):
        if "[ABORT]" in line:
            parts = []
            for raw in lines[i + 1 : i + 3]:
                text = raw.strip().strip("*").replace("\\___/", " ").replace("|", " ")
                parts.append(" ".join(text.split()))
            return " ".join(p for p in parts if p) or "CP2K aborted"
    return None


def dropped_negative_integrals(h2: Any, *, tol: float = 1e-6, min_couplings: int = 8) -> bool:
    """True if ``h2`` looks like CP2K 7.1's signed screening threw its negative ERIs away.

    Coulomb ``(pp|qq)`` and exchange ``(pq|pq)``-type integrals are never negative; every
    other ``(pq|rs)`` takes the sign of arbitrary orbital phases, so a genuine set with
    several such couplings essentially always contains negative ones.  ``min_couplings``
    keeps tiny spaces, where all-positive can happen by chance, out of the alarm.
    """
    import numpy as np

    h2 = np.asarray(h2)
    n = h2.shape[0]
    # Broadcast index grids instead of four materialised norb^4 int arrays.
    p, q, r, s = np.ogrid[:n, :n, :n, :n]
    coulomb = (p == q) & (r == s)
    exchange = ((p == r) & (q == s)) | ((p == s) & (q == r))
    other = ~(coulomb | exchange) & (np.abs(h2) > tol)
    return bool(other.sum() >= min_couplings and not (h2[other] < 0).any())


@dataclass(frozen=True)
class ActiveSpaceRun:
    fcidump: Path
    output: Path
    total_energy: float  # CP2K's SCF total (Hartree), for cross-checks


def run_active_space(
    atoms: Any,
    spec: ActiveSpaceSpec,
    *,
    workdir: str | Path,
    basis: dict[str, str],
    potential: dict[str, str],
    data_dir: str | Path | None = None,
    executable: str | None = None,
    cell: float = 10.0,
    xc: str = "PBE",
    cutoff: float = 300.0,
    threads: int = 1,
    charge: int = 0,
) -> ActiveSpaceRun:
    """Run CP2K on ``atoms`` (an ASE ``Atoms``) and return its active-space FCIDUMP.

    The FCIDUMP is the input of ``embasi-qiskit-integration import_fcidump``.  Molecules
    are centred in the cubic cell.  ``charge`` is the total charge; ``spec``'s valence
    electron count must already account for it.
    """
    exe = executable or find_executable()
    data = Path(data_dir or os.environ.get(DATA_ENV) or "")
    if not (data / "BASIS_MOLOPT").is_file() or not (data / "GTH_POTENTIALS").is_file():
        raise RuntimeError(
            f"CP2K basis/potential files not found in {data!s}: set ${DATA_ENV} to the "
            "directory holding BASIS_MOLOPT and GTH_POTENTIALS"
        )
    layout = active_space_layout(exe)
    if spec.unrestricted and layout.version is not None and layout.version < (8, 0):
        # Refused before running: CP2K 7.x's active-space module is restricted-only.  It
        # either aborts after the UKS SCF ("Spin option not yet available") or -- about
        # one run in seven here -- segfaults at the same point, so running it proves
        # nothing and the outcome is not even deterministic.
        raise RuntimeError(
            f"CP2K {layout.version[0]}.{layout.version[1]} cannot build an unrestricted "
            "active space (its active-space module is restricted-only: 'Spin option not "
            "yet available'); use a newer CP2K for a UHF FCIDUMP"
        )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    pos = atoms.get_positions()
    pos = pos - pos.mean(axis=0) + cell / 2.0
    text = active_space_input(
        atoms.get_chemical_symbols(),
        pos,
        spec=spec,
        basis=basis,
        potential=potential,
        data_dir=data,
        cell=cell,
        xc=xc,
        cutoff=cutoff,
        layout=layout,
        charge=charge,
    )
    inp, out = workdir / "as.inp", workdir / "as.out"
    fcidump = workdir / "active.fcidump"
    # Never let a previous run's FCIDUMP pass for this one's.
    fcidump.unlink(missing_ok=True)
    inp.write_text(text)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    with open(out, "w") as handle:
        proc = subprocess.run(
            [exe, "-i", inp.name], cwd=workdir, stdout=handle, stderr=subprocess.STDOUT, env=env
        )
    reason = abort_message(out)
    if reason is not None:
        raise RuntimeError(f"CP2K aborted ({out}): {reason}")
    if proc.returncode != 0:
        if proc.returncode < 0:  # killed by a signal (subprocess reports -N)
            raise RuntimeError(
                f"CP2K crashed (signal {-proc.returncode}, exited with status "
                f"{proc.returncode}); see {out}"
            )
        raise RuntimeError(f"CP2K exited with status {proc.returncode}; see {out}")
    if not fcidump.is_file():
        raise RuntimeError(f"CP2K finished but wrote no FCIDUMP; see {out}")
    from embasi_qiskit_integration.hamiltonian import fcidump as _fcidump

    if dropped_negative_integrals(_fcidump.read(fcidump).h2):
        raise RuntimeError(
            f"{fcidump} holds off-diagonal two-electron integrals but no negative ones: "
            "CP2K dropped them (signed EPS_INTEGRAL screening, CP2K <= 7.1).  This input "
            "sets a negative EPS_INTEGRAL to prevent it; check the CP2K version."
        )
    return ActiveSpaceRun(fcidump=fcidump, output=out, total_energy=read_total_energy(out))


# --------------------------------------------------------------------------- #
# The registered backend: an explicit, explained refusal
# --------------------------------------------------------------------------- #
@register_backend("cp2k")
def build_cp2k_backend(atoms: Any, charge: int, cfg: Any) -> QMBackend:
    """Refuse: CP2K cannot drive this package's projection embedding."""
    raise NotImplementedError(
        "--qm_code cp2k: CP2K has no ASI implementation or runtime AO-matrix API -- it "
        "cannot take an AO potential or density matrix during its SCF, nor the mu*P_B "
        "projector -- and EmbASI has no adapter for it, so it cannot drive this package's "
        "projection embedding.  Use CP2K's own active space instead: "
        "backends.cp2k.run_active_space (or a CP2K &ACTIVE_SPACE &FCIDUMP run), then "
        "`embasi-qiskit-integration import_fcidump <FCIDUMP> <job>` and `solve <job>`."
    )
