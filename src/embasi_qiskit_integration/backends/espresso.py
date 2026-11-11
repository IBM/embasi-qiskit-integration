# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Quantum ESPRESSO backend (``--qm_code espresso``): readers for QE/WEST output.

**Licence boundary.**  Quantum ESPRESSO (GPL-2.0), WEST and westpy (both GPL-3.0) are
only ever *run as external programs* and their *output files* parsed by the readers
here, written from the documented data layout.  No QE/WEST/westpy code is imported,
linked or vendored -- in particular not westpy, whose QDET reader covers the same file.

**What QE can supply** (checked against QE 7.6 runs).  QE is a plane-wave code: it
has **no AO basis**, so none of the AO-basis quantities this package's projection
embedding needs (``S``, ``hcore``, ``veff`` at an AO density matrix, ``J``/``K``, AO
ERIs) exist inside it -- only post-SCF *projections* onto pseudo-atomic or Wannier
functions, which span the computed bands only and carry a completeness error.  QE also
has no hook to add a nonlocal operator such as ``mu P_B`` to its Hamiltonian.  So:

* :func:`build_espresso_backend` (``--qm_code espresso``) declares **no** capabilities
  and refuses, naming the reason; it is registered so the choice is explicit and the
  refusal is a clear message rather than "unknown code";
* QE's own active-space route is supported instead: WEST/QDET writes an effective
  Hamiltonian over a set of Kohn-Sham bands, which :func:`read_wfreq_json` turns into an
  :class:`~embasi_qiskit_integration.contract.EmbeddedHamiltonian` for our FCI/SQD solvers
  (``embasi-qiskit-integration import_fcidump`` covers WEST's FCIDUMP output, and
  :func:`write_job_from_wfreq` the JSON);
* :func:`read_atomic_proj` reads ``projwfc.x``'s ``atomic_proj.xml`` (band projections
  onto the pseudo-atomic functions, and their overlap), the starting point for any
  future localized-basis route.

QDET is a *different* embedding (a screened interaction ``W`` with a double-counting
correction, active space in KS bands at Gamma): its energies are not comparable with
this package's projection embedding on the same system.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embasi_qiskit_integration.backends.registry import QMBackend, register_backend
from embasi_qiskit_integration.contract import EmbeddedHamiltonian

# WEST writes the QDET matrix elements in eV.  CODATA 2018, as ASE's `units.Hartree`.
_HARTREE_EV = 27.211386245988


# --------------------------------------------------------------------------- #
# WEST / QDET: wfreq.json -> EmbeddedHamiltonian
# --------------------------------------------------------------------------- #
def _spin_key(spin: int) -> str:
    """WEST's per-spin key: ``K000001`` for spin 0, ``K000002`` for spin 1."""
    return f"K{spin + 1:06d}"


def _pair_index(n: int) -> list[tuple[int, int]]:
    """WEST's pair order: upper triangle ``i <= j``, row-major."""
    return [(i, j) for i in range(n) for j in range(i, n)]


def _unpack_h1(packed: Any, n: int) -> np.ndarray:
    pairs = _pair_index(n)
    values = np.asarray(packed, dtype=float)
    if values.shape != (len(pairs),):
        raise ValueError(f"h1e has {values.size} entries, expected {len(pairs)} for {n} bands")
    h = np.zeros((n, n))
    for (i, j), v in zip(pairs, values, strict=True):
        h[i, j] = h[j, i] = v
    return h


def _unpack_eri(rows: dict[str, Any], n: int) -> np.ndarray:
    """``(pq|rs)`` from WEST's ``pair<p>`` rows (each a row of the npair x npair table)."""
    pairs = _pair_index(n)
    eri = np.zeros((n,) * 4)
    for p, (i, j) in enumerate(pairs):
        row = np.asarray(rows[f"pair{p + 1:09d}"], dtype=float)
        if row.shape != (len(pairs),):
            raise ValueError(f"eri row pair{p + 1:09d} has {row.size} entries")
        for q, (k, m) in enumerate(pairs):
            v = row[q]
            for a, b in ((i, j), (j, i)):
                for c, d in ((k, m), (m, k)):
                    eri[a, b, c, d] = v
    return eri


@dataclass(frozen=True)
class QDETHamiltonian:
    """What a ``wfreq.json`` describes, before choosing how to hand it to a solver."""

    bands: tuple[int, ...]  # the KS bands (qp_bands) spanning the active space
    h1: tuple[np.ndarray, ...]  # per spin, Hartree
    eri: dict[tuple[int, int], np.ndarray]  # (s1, s2) -> (pq|rs), Hartree
    occupation: np.ndarray  # (nspin, nband)
    two_body: str  # which interaction was read ("eri_w" = partially screened W)

    @property
    def nspin(self) -> int:
        return len(self.h1)

    @property
    def nelec(self) -> tuple[int, int]:
        """``(n_alpha, n_beta)`` from the DFT occupations of the active bands.

        Rounded, not truncated: an occupation sum of 2.9999999 is three electrons.
        """
        occ = self.occupation
        if self.nspin == 1:
            total = int(round(float(occ.sum())))
            if total % 2:
                raise ValueError(
                    f"spin-restricted QDET active space holds an odd {total} electrons; "
                    "pass nelec explicitly"
                )
            return total // 2, total // 2
        return int(round(float(occ[0].sum()))), int(round(float(occ[1].sum())))


def read_wfreq_json(path: str | Path, *, two_body: str = "eri_w") -> QDETHamiltonian:
    """Parse WEST's ``wfreq.json`` (``wfreq_calculation`` with ``H``, i.e. QDET).

    ``two_body`` picks the interaction: ``"eri_w"`` (default) is QDET's partially
    screened ``W``; with ``l_qdet_verbose`` WEST also writes other variants (e.g. the
    bare Coulomb), selected by their key.  Matrix elements are converted eV -> Hartree.
    """
    data = json.loads(Path(path).read_text())
    try:
        nspin = int(data["system"]["electron"]["nspin"])
        bands = tuple(int(b) for b in data["input"]["wfreq_control"]["qp_bands"][0])
        qdet = data["qdet"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"{path} is not a QDET wfreq.json (needs system.electron.nspin, "
            "input.wfreq_control.qp_bands and a qdet section)"
        ) from exc
    if two_body not in qdet:
        raise ValueError(
            f"{path} has no qdet.{two_body}; available: "
            f"{', '.join(k for k in qdet if k.startswith('eri'))}"
        )
    n = len(bands)
    h1 = tuple(_unpack_h1(qdet["h1e"][_spin_key(s)], n) / _HARTREE_EV for s in range(nspin))
    eri = {
        (s1, s2): _unpack_eri(qdet[two_body][_spin_key(s1)][_spin_key(s2)], n) / _HARTREE_EV
        for s1 in range(nspin)
        for s2 in range(nspin)
    }
    occupation = np.array(
        [data["output"]["Q"][_spin_key(s)]["occupation"] for s in range(nspin)], dtype=float
    )
    if occupation.shape != (nspin, n):
        raise ValueError(f"occupation has shape {occupation.shape}, expected {(nspin, n)}")
    return QDETHamiltonian(bands, h1, eri, occupation, two_body)


def qdet_to_embedded_hamiltonian(
    qdet: QDETHamiltonian, *, nelec: tuple[int, int] | None = None
) -> EmbeddedHamiltonian:
    """An :class:`EmbeddedHamiltonian` for our solvers.

    QDET's effective Hamiltonian has no constant term (``e_core = 0``): its eigenvalues
    are excitation-energy references, not total energies.  A spin-polarised file keeps
    its per-spin blocks (``h1a/h1b``, ``(aa, ab, bb)``), consumed exactly by ``--solver
    fci``.
    """
    meta = {
        "source": "WEST/QDET wfreq.json",
        "qp_bands": list(qdet.bands),
        "two_body": qdet.two_body,
        "note": "QDET (screened W + double counting); not projection embedding",
    }
    sector: tuple[int, int] = (int(nelec[0]), int(nelec[1])) if nelec is not None else qdet.nelec
    if qdet.nspin == 1:
        return EmbeddedHamiltonian(
            h1=qdet.h1[0], h2=qdet.eri[(0, 0)], e_core=0.0, nelec=sector, meta=meta
        )
    h1a, h1b = qdet.h1
    aa, ab, bb = qdet.eri[(0, 0)], qdet.eri[(0, 1)], qdet.eri[(1, 1)]
    return EmbeddedHamiltonian(
        h1=0.5 * (h1a + h1b),
        h2=aa,
        e_core=0.0,
        nelec=sector,
        h1a=h1a,
        h1b=h1b,
        h2_spin=(aa, ab, bb),
        meta=meta,
    )


def write_job_from_wfreq(
    path: str | Path,
    job_dir: str | Path,
    *,
    two_body: str = "eri_w",
    nelec: tuple[int, int] | None = None,
) -> Path:
    """``wfreq.json`` -> a job directory ``embasi-qiskit-integration solve`` runs."""
    import warnings

    from embasi_qiskit_integration import ipc

    ham = qdet_to_embedded_hamiltonian(read_wfreq_json(path, two_body=two_body), nelec=nelec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # spin blocks travel in the sidecar
        return ipc.write_job(ham, job_dir)


# --------------------------------------------------------------------------- #
# projwfc.x: atomic_proj.xml
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AtomicProjections:
    """``projwfc.x`` output at one k-point: KS bands projected on pseudo-atomic functions."""

    eigenvalues: np.ndarray  # (nspin, nbnd), Hartree
    projections: np.ndarray  # (nspin, n_atomic_wfc, nbnd): <phi_a | psi_n>
    overlaps: np.ndarray | None  # (nspin, n_atomic_wfc, n_atomic_wfc), if written
    fermi_energy: float  # Hartree
    n_electrons: float

    def projectability(self, n_occ: int | None = None) -> np.ndarray:
        """Per band, the weight captured by the atomic basis (``1`` = fully represented).

        With the overlap, ``p_n = <psi_n|phi> S^{-1} <phi|psi_n>``; without it the
        orthonormal-basis estimate ``sum_a |<phi_a|psi_n>|^2``.  Below 1 is the
        completeness error any projected AO-like operator built from these bands carries.
        """
        p = self.projections
        if self.overlaps is not None:
            weights = np.stack(
                [
                    np.einsum("an,ab,bn->n", p[s].conj(), np.linalg.inv(self.overlaps[s]), p[s])
                    for s in range(p.shape[0])
                ]
            ).real
        else:
            weights = np.sum(np.abs(p) ** 2, axis=1)
        return weights if n_occ is None else weights[:, :n_occ]


def _floats(text: str | None) -> np.ndarray:
    return np.array([float(x) for x in (text or "").split()], dtype=float)


def _complex(text: str | None) -> np.ndarray:
    v = _floats(text)
    return v[0::2] + 1j * v[1::2]


def read_atomic_proj(path: str | Path) -> AtomicProjections:
    """Parse ``atomic_proj.xml`` (QE >= 6.x layout: ``HEADER``/``EIGENSTATES``/``OVERLAPS``).

    Only single-k-point (Gamma) files are accepted -- the molecular case.  Energies are
    converted Rydberg -> Hartree.
    """
    root = ET.parse(path).getroot()
    head = root.find("HEADER")
    if head is None:
        raise ValueError(f"{path} has no HEADER; is it a projwfc.x atomic_proj.xml?")
    nbnd = int(head.attrib["NUMBER_OF_BANDS"])
    nk = int(head.attrib["NUMBER_OF_K-POINTS"])
    nspin = int(head.attrib["NUMBER_OF_SPIN_COMPONENTS"])
    nwfc = int(head.attrib["NUMBER_OF_ATOMIC_WFC"])
    if nk != nspin and nk != 1:
        # QE stores LSDA spin channels as separate k-points.
        raise ValueError(f"{path} has {nk} k-points; only Gamma is supported")
    states = root.find("EIGENSTATES")
    if states is None:
        raise ValueError(f"{path} has no EIGENSTATES")
    eig_blocks = states.findall("E")
    proj_blocks = states.findall("PROJS")
    if len(eig_blocks) != nspin or len(proj_blocks) != nspin:
        raise ValueError(f"{path}: expected {nspin} spin block(s) of E/PROJS")
    eigen = np.stack([_floats(e.text) for e in eig_blocks]) / 2.0  # Ry -> Ha
    proj = np.zeros((nspin, nwfc, nbnd), dtype=complex)
    for s, block in enumerate(proj_blocks):
        wfcs = block.findall("ATOMIC_WFC")
        if len(wfcs) != nwfc:
            raise ValueError(f"{path}: {len(wfcs)} ATOMIC_WFC in spin {s}, expected {nwfc}")
        for wfc in wfcs:
            proj[s, int(wfc.attrib["index"]) - 1] = _complex(wfc.text)
    overlaps = None
    ovl = root.find("OVERLAPS")
    if ovl is not None:
        blocks = ovl.findall("OVPS")
        overlaps = np.stack([_complex(b.text).reshape(nwfc, nwfc) for b in blocks])
    return AtomicProjections(
        eigenvalues=eigen,
        projections=proj,
        overlaps=overlaps,
        fermi_energy=float(head.attrib.get("FERMI_ENERGY", "nan")) / 2.0,
        n_electrons=float(head.attrib["NUMBER_OF_ELECTRONS"]),
    )


# --------------------------------------------------------------------------- #
# The registered backend: an explicit, explained refusal
# --------------------------------------------------------------------------- #
@register_backend("espresso")
def build_espresso_backend(atoms: Any, charge: int, cfg: Any) -> QMBackend:
    """Refuse: QE cannot host this package's AO-basis projection embedding."""
    raise NotImplementedError(
        "--qm_code espresso: Quantum ESPRESSO is a plane-wave code with no AO basis and no "
        "hook for the nonlocal mu*P_B projector, so it cannot drive this package's "
        "projection embedding (there is no EmbASI adapter for it either).  Use QE's own "
        "active-space route instead: run WEST/QDET, then "
        "`embasi-qiskit-integration import_fcidump <FCIDUMP> <job>` (or "
        "backends.espresso.write_job_from_wfreq for wfreq.json) and `solve <job>`."
    )
