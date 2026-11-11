# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""CPMD backend (``--qm_code cpmd``): readers for CPMD's Wannier and Kohn-Sham output.

**Licence.**  CPMD is MIT-licensed (since 2022), compatible with this package's
Apache-2.0.  Two parts of the CPMD distribution are *not*: the Gromos QM/MM sources
(proprietary) and the MiMiC coupling library (LGPL-3.0+).  Neither is used.  CPMD is run
as its own executable (``cpmd.x``; there is no ASE calculator for it) and the files it
writes are read here.

**What CPMD can supply** (checked against its 4.3 source and a real run; see
``BACKEND_DECOUPLING_PLAN.md``).  CPMD is a plane-wave pseudopotential code: like
Quantum ESPRESSO it has **no AO basis**, so none of the AO-basis quantities this
package's projection embedding needs exist inside it, and it can only add a *local*
external potential (``EXTERNAL POTENTIAL``), not the nonlocal ``mu P_B`` projector.
Unlike QE it has **no native active-space route** either -- no ERIs, no FCIDUMP, no
WEST-like downfolding.  What it does have is Wannier localisation:

* ``WANNIER_HAM`` (``&CPMD WANNIER DOS`` in a ``PROPERTIES`` run with ``&PROP
  LOCALIZE``): the Kohn-Sham matrix over the *occupied* Wannier states, in eV;
* ``WANNIER_CENTER`` / ``WC_SPREAD``: centres (bohr, wrapped into the box) and spreads;
* ``IONS+CENTERS.xyz``: the atoms and the *unwrapped* centres, in Angstrom;
* the KS eigenvalues printed by a ``KOHN-SHAM ENERGIES`` run.

CPMD appends every localisation to these files (``fo_app``), so each reader returns the
*last* block, sized by the ``WC_SPREAD`` header.  These are the inputs of any future
localized-orbital route (Wannier orbitals -> our own integrals -> FCIDUMP); note that
``WANNIER_HAM`` is a Kohn-Sham operator (it already contains Hartree and xc), not a
core Hamiltonian, and spans the occupied space only.

:func:`build_cpmd_backend` declares no capabilities and refuses ``--qm_code cpmd`` with
those reasons.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embasi_qiskit_integration.backends.registry import QMBackend, register_backend

# CODATA 2018, as ASE's units.
_HARTREE_EV = 27.211386245988


def _floats(text: str) -> list[float]:
    return [float(x.replace("D", "E").replace("d", "e")) for x in text.split()]


@dataclass(frozen=True)
class SpreadBlock:
    step: int
    spreads: np.ndarray  # (nstate,), bohr^2


def read_wc_spread(path: str | Path) -> list[SpreadBlock]:
    """Every block of ``WC_SPREAD``: a ``step nstate`` header, then ``nstate`` spreads."""
    tokens = Path(path).read_text().split()
    blocks: list[SpreadBlock] = []
    i = 0
    while i < len(tokens):
        step, n = int(tokens[i]), int(tokens[i + 1])
        values = np.array(_floats(" ".join(tokens[i + 2 : i + 2 + n])))
        if values.size != n:
            raise ValueError(f"{path}: truncated block at step {step}")
        blocks.append(SpreadBlock(step, values))
        i += 2 + n
    if not blocks:
        raise ValueError(f"{path} holds no WC_SPREAD block")
    return blocks


@dataclass(frozen=True)
class CenterBlock:
    step: int
    centers: np.ndarray  # (nstate, 3), bohr, wrapped into the cell as CPMD writes them
    spreads: np.ndarray  # (nstate,)


def read_wannier_centers(path: str | Path, nstate: int) -> list[CenterBlock]:
    """Every ``nstate``-row block of ``WANNIER_CENTER`` (``step x y z spread`` per row)."""
    rows = [line.split() for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len(rows) % nstate:
        raise ValueError(f"{path} has {len(rows)} rows, not a multiple of nstate={nstate}")
    blocks = []
    for b in range(len(rows) // nstate):
        chunk = rows[b * nstate : (b + 1) * nstate]
        data = np.array([[float(x) for x in r[1:5]] for r in chunk])
        blocks.append(CenterBlock(int(chunk[0][0]), data[:, :3], data[:, 3]))
    return blocks


def read_wannier_ham(path: str | Path, nstate: int) -> np.ndarray:
    """The last ``nstate x nstate`` Kohn-Sham matrix of ``WANNIER_HAM``, in Hartree.

    CPMD writes it list-directed (rows may wrap across lines), in eV, and appends a
    block per localisation; all numbers are read and the final block kept.
    """
    values = _floats(Path(path).read_text())
    size = nstate * nstate
    if not values or len(values) % size:
        raise ValueError(f"{path} has {len(values)} values, not a multiple of {nstate}^2")
    h = np.array(values[-size:]).reshape(nstate, nstate) / _HARTREE_EV
    asym = float(np.abs(h - h.T).max())
    if asym > 1e-6:
        raise ValueError(f"{path}: the Wannier Hamiltonian is not symmetric ({asym:.1e} Ha)")
    return 0.5 * (h + h.T)


@dataclass(frozen=True)
class IonsAndCenters:
    symbols: list[str]
    positions: np.ndarray  # (natm, 3), Angstrom
    centers: np.ndarray  # (nstate, 3), Angstrom, unwrapped


def read_ions_centers_xyz(path: str | Path) -> IonsAndCenters:
    """The last frame of ``IONS+CENTERS.xyz`` (atoms, then the centres as ``X``)."""
    lines = Path(path).read_text().splitlines()
    frames: list[list[list[str]]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        frames.append([lines[j].split() for j in range(i + 2, i + 2 + n)])
        i += 2 + n
    if not frames:
        raise ValueError(f"{path} holds no frame")
    frame = frames[-1]
    atoms = [r for r in frame if r[0] != "X"]
    centers = [r for r in frame if r[0] == "X"]
    return IonsAndCenters(
        symbols=[r[0] for r in atoms],
        positions=np.array([[float(x) for x in r[1:4]] for r in atoms]),
        centers=np.array([[float(x) for x in r[1:4]] for r in centers]),
    )


@dataclass(frozen=True)
class WannierSet:
    """One CPMD Wannier localisation, cross-checked between its output files."""

    hamiltonian: np.ndarray  # (nstate, nstate), Hartree; KS operator over occupied WFs
    spreads: np.ndarray  # (nstate,), bohr^2
    centers: np.ndarray  # (nstate, 3), Angstrom, unwrapped
    symbols: list[str]
    positions: np.ndarray  # (natm, 3), Angstrom

    @property
    def nstate(self) -> int:
        return int(self.spreads.size)

    def nearest_atoms(self) -> np.ndarray:
        """For each Wannier centre, the index of the closest atom (fragment assignment)."""
        d = np.linalg.norm(self.centers[:, None, :] - self.positions[None, :, :], axis=2)
        return np.argmin(d, axis=1)

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the Wannier Hamiltonian: the occupied KS eigenvalues (Hartree)."""
        return np.linalg.eigvalsh(self.hamiltonian)


def read_wannier(directory: str | Path) -> WannierSet:
    """Read and cross-check the Wannier output a CPMD ``LOCALIZE`` run left in ``directory``.

    The state count comes from ``WC_SPREAD``'s last header; ``WANNIER_CENTER``'s last
    block must carry the same spreads, and ``IONS+CENTERS.xyz`` the same number of
    centres -- three files CPMD writes separately, so a mismatch means a stale or
    truncated file rather than a parse quirk.
    """
    directory = Path(directory)
    spread = read_wc_spread(directory / "WC_SPREAD")[-1]
    n = spread.spreads.size
    centers = read_wannier_centers(directory / "WANNIER_CENTER", n)[-1]
    if not np.allclose(centers.spreads, spread.spreads, atol=1e-5):
        raise ValueError(f"{directory}: WANNIER_CENTER and WC_SPREAD disagree on the last spreads")
    xyz = read_ions_centers_xyz(directory / "IONS+CENTERS.xyz")
    if xyz.centers.shape != (n, 3):
        raise ValueError(f"{directory}: IONS+CENTERS.xyz has {len(xyz.centers)} centres, not {n}")
    ham = read_wannier_ham(directory / "WANNIER_HAM", n)
    return WannierSet(ham, spread.spreads, xyz.centers, xyz.symbols, xyz.positions)


def read_ks_eigenvalues(output: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(eigenvalues [Hartree], occupations)`` from the last ``EIGENVALUES(EV)`` block.

    CPMD prints it from a ``KOHN-SHAM ENERGIES`` run (``index value occupation``, two
    states per line).
    """
    lines = Path(output).read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if "EIGENVALUES(EV) AND OCCUPATION" in line]
    if not starts:
        raise ValueError(f"{output} has no 'EIGENVALUES(EV) AND OCCUPATION' block")
    entries: dict[int, tuple[float, float]] = {}
    for line in lines[starts[-1] + 1 :]:
        parts = line.split()
        if not parts or len(parts) % 3 or not parts[0].isdigit():
            break
        for k in range(0, len(parts), 3):
            entries[int(parts[k])] = (float(parts[k + 1]), float(parts[k + 2]))
    idx = sorted(entries)
    if idx != list(range(1, len(idx) + 1)):
        raise ValueError(f"{output}: eigenvalue indices are not contiguous")
    eig = np.array([entries[i][0] for i in idx]) / _HARTREE_EV
    occ = np.array([entries[i][1] for i in idx])
    return eig, occ


# --------------------------------------------------------------------------- #
# The registered backend: an explicit, explained refusal
# --------------------------------------------------------------------------- #
@register_backend("cpmd")
def build_cpmd_backend(atoms: Any, charge: int, cfg: Any) -> QMBackend:
    """Refuse: CPMD cannot host this package's AO-basis projection embedding."""
    raise NotImplementedError(
        "--qm_code cpmd: CPMD is a plane-wave code with no AO basis; it can add only a "
        "local external potential, not the nonlocal mu*P_B projector, and EmbASI has no "
        "adapter for it, so it cannot drive this package's projection embedding.  Unlike "
        "Quantum ESPRESSO it also has no native active-space route (no ERIs, no FCIDUMP). "
        "Its Wannier output (WANNIER_HAM, centres, spreads) is readable with "
        "backends.cpmd.read_wannier, the starting point of a localized-orbital route."
    )
