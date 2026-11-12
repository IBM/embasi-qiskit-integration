# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""CPMD backend (``--qm_code cpmd``): readers for CPMD's Wannier and Kohn-Sham output.

**Licence.**  CPMD is MIT-licensed (since 2022), compatible with this package's
Apache-2.0.  Two parts of the CPMD distribution are *not*: the Gromos QM/MM sources
(proprietary) and the MiMiC coupling library (LGPL-3.0+).  Neither is used.  CPMD is run
as its own executable (``cpmd.x``; there is no ASE calculator for it) and the files it
writes are read here.

**What CPMD can supply** (checked against its 4.3 source and a real run).  CPMD is a
plane-wave pseudopotential code: like Quantum ESPRESSO it has **no AO basis**, so none of the AO-basis quantities this
package's projection embedding needs exist inside it, and it can only add a *local*
external potential (``EXTERNAL POTENTIAL``), not the nonlocal ``mu P_B`` projector.
Unlike QE it has **no native active-space route** either -- no ERIs, no FCIDUMP, no
WEST-like downfolding.  What it does have is Wannier localisation:

* ``WANNIER_HAM`` (``&CPMD WANNIER DOS`` in a ``PROPERTIES`` run with ``&PROP
  LOCALIZE``): the Kohn-Sham matrix over the *occupied* Wannier states, in eV times
  ``f/2`` (eV for a closed shell, half that for LSD; :func:`read_wannier_ham`);
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


def read_occupations(output: str | Path) -> np.ndarray:
    """Per-state occupations ``f`` from a CPMD output (its ``OCCUPATION`` header block).

    Closed shell: one ``OCCUPATION`` list; LSD: ``ALPHA OCCUPATION`` then ``BETA
    OCCUPATION``, returned concatenated (alpha first, CPMD's state order).
    """
    lines = Path(output).read_text().splitlines()
    n = next(
        (int(ln.split(":")[1]) for ln in lines if ln.strip().startswith("NUMBER OF STATES:")),
        None,
    )
    if n is None:
        raise ValueError(f"{output}: no 'NUMBER OF STATES:' line; is this CPMD output?")
    occ: list[float] = []
    for i, ln in enumerate(lines):
        if ln.strip() not in ("OCCUPATION", "ALPHA OCCUPATION", "BETA OCCUPATION"):
            continue
        for row in lines[i + 1 :]:
            try:
                occ.extend(float(x) for x in row.split())
            except ValueError:
                break
            if not row.split():
                break
        if len(occ) >= n:
            break
    if len(occ) != n:
        raise ValueError(f"{output}: read {len(occ)} occupations for {n} states")
    total = next(
        (float(ln.split(":")[1]) for ln in lines if ln.strip().startswith("NUMBER OF ELECTRONS:")),
        None,
    )
    if total is not None and abs(sum(occ) - total) > 1e-4:
        # The F5.1 print hides a fraction near an integer (1.96 shows as 2.0); the
        # electron count, printed to 5 decimals, does not.
        raise ValueError(
            f"{output}: the printed occupations sum to {sum(occ)} but NUMBER OF "
            f"ELECTRONS is {total}; they are rounded, pass the exact occupations"
        )
    if any(f != round(f) for f in occ):
        # Printed as F5.1: 2/3 shows as 0.7, which would scale WANNIER_HAM by 5% too much.
        raise ValueError(
            f"{output}: fractional occupations {occ} are printed to one decimal only; "
            "pass the exact occupations to read_wannier_ham instead"
        )
    return np.asarray(occ)


def read_wannier_ham(path: str | Path, nstate: int, occupations: Any) -> np.ndarray:
    """The last ``nstate x nstate`` Kohn-Sham matrix of ``WANNIER_HAM``, in Hartree.

    CPMD writes it list-directed (rows may wrap across lines) and appends a block per
    localisation; all numbers are read and the final block kept.  Each entry is
    ``-Ry <c0_i|c2_j>`` with the gradient ``c2_j`` scaled by the occupation ``f_j/2``,
    i.e. ``f_j/2 * H_ij`` in eV: plain eV only for doubly occupied states.  The file
    cannot tell an LSD run (``f = 1``, read as ``H/2`` yet still symmetric) from a closed
    shell, so ``occupations`` (:func:`read_occupations`, or ``[2.0] * nstate``) are
    required; all-2 occupations on a matrix with LSD's uncoupled alpha/beta blocks are
    refused as a contradiction.
    """
    values = _floats(Path(path).read_text())
    size = nstate * nstate
    if not values or len(values) % size:
        raise ValueError(f"{path} has {len(values)} values, not a multiple of {nstate}^2")
    raw = np.array(values[-size:]).reshape(nstate, nstate)
    f = np.asarray(occupations, dtype=float).reshape(-1)
    if f.shape != (nstate,) or np.any(f <= 0):
        raise ValueError(f"need {nstate} positive occupations, got {f.tolist()}")
    split = next(
        (k for k in range(1, nstate) if not raw[:k, k:].any() and not raw[k:, :k].any()),
        None,
    )
    if split is not None and np.all(f == 2.0):
        # LSD stores alpha then beta with an exactly zero cross block (at any split, since
        # an open shell has nsup != nsdown); read with f=2 it would come out as H/2.
        raise ValueError(
            f"{path}: states 1-{split} and {split + 1}-{nstate} do not couple at all, as "
            "the alpha/beta blocks of an LSD run (f=1), yet the occupations say closed "
            "shell; pass the run's occupations=read_occupations(<CPMD output>)"
        )
    h = raw / (0.5 * f[None, :] * _HARTREE_EV)
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


def read_wannier(
    directory: str | Path, *, output: str | Path | None = None, occupations: Any = None
) -> WannierSet:
    """Read and cross-check the Wannier output a CPMD ``LOCALIZE`` run left in ``directory``.

    The state count comes from ``WC_SPREAD``'s last header; ``WANNIER_CENTER``'s last
    block must carry the same spreads, and ``IONS+CENTERS.xyz`` the same number of
    centres -- three files CPMD writes separately, so a mismatch means a stale or
    truncated file rather than a parse quirk.  ``output`` is that run's CPMD output,
    whose occupations scale ``WANNIER_HAM`` (required for LSD; see
    :func:`read_wannier_ham`); ``occupations`` gives them directly (fractional ones,
    or ``[2.0] * nstate`` for a closed shell).  One of the two is required.  For LSD the matrix is block diagonal, alpha first.
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
    if (output is None) == (occupations is None):
        raise ValueError(
            "give the run's CPMD output (output=) or its occupations (occupations=), "
            "exactly one: WANNIER_HAM's scale depends on them (see read_wannier_ham)"
        )
    occ = occupations if output is None else read_occupations(output)
    ham = read_wannier_ham(directory / "WANNIER_HAM", n, occ)
    return WannierSet(ham, spread.spreads, xyz.centers, xyz.symbols, xyz.positions)


def read_ks_eigenvalues(output: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """``(eigenvalues [Hartree], occupations)`` from the last ``EIGENVALUES(EV)`` block.

    CPMD prints it from a ``KOHN-SHAM ENERGIES`` run (``index value occupation``, two
    states per line; ``wreigen`` in ``wrener_utils``).  A closed-shell run gives ``(n,)``
    arrays; a spin-polarised (LSD) run, whose block is split into ``ALPHA STATES:`` and
    ``BETA STATES:``, gives ``(2, n)`` arrays, alpha first.  A state CPMD flags as not
    converged (``NC`` glued to the value) is read, with a warning.  Multi-k-point output
    is refused (molecules are Gamma-only).
    """
    import re
    import warnings

    lines = Path(output).read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if "EIGENVALUES(EV) AND OCCUPATION" in line]
    if not starts:
        raise ValueError(f"{output} has no 'EIGENVALUES(EV) AND OCCUPATION' block")
    entry = re.compile(r"(\d+)\s+(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)(NC)?\s+(-?\d+\.\d+)")
    channels: list[dict[int, tuple[float, float]]] = [{}]
    labelled = False
    not_converged: list[int] = []
    for line in lines[starts[-1] + 1 :]:
        text = line.strip()
        if text.startswith("K POINT"):
            raise ValueError(f"{output}: k-point resolved eigenvalues are not supported")
        if text in ("ALPHA STATES:", "BETA STATES:"):
            if text == "BETA STATES:":
                channels.append({})
            labelled = True
            continue
        hits = entry.findall(line)
        if not hits:  # the CHEMICAL POTENTIAL line (or anything else) ends the block
            break
        for index, value, nc, occupation in hits:
            channels[-1][int(index)] = (float(value.replace("D", "E")), float(occupation))
            if nc:
                not_converged.append(int(index))
    if labelled and len(channels) != 2:
        raise ValueError(f"{output}: an LSD block needs both ALPHA and BETA STATES")
    eig, occ = [], []
    for states in channels:
        idx = sorted(states)
        if not idx or idx != list(range(1, len(idx) + 1)):
            raise ValueError(f"{output}: eigenvalue indices are missing or not contiguous")
        eig.append(np.array([states[i][0] for i in idx]) / _HARTREE_EV)
        occ.append(np.array([states[i][1] for i in idx]))
    if not_converged:
        warnings.warn(
            f"{output}: CPMD flags states {sorted(set(not_converged))} as not converged (NC)",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(channels) == 1:
        return eig[0], occ[0]
    if eig[0].size != eig[1].size:
        raise ValueError(f"{output}: alpha and beta list a different number of states")
    return np.stack(eig), np.stack(occ)


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
