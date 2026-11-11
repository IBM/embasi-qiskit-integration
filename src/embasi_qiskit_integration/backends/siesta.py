# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SIESTA backend (``--qm_code siesta``): file-based, DFT-in-DFT only.

**Licence boundary.**  SIESTA is GPL-3.0; this package is Apache-2.0.  It is therefore
only ever *run as an external program* (``siesta``, through ASE's file-based
calculator) and its *output files* are read.  No SIESTA code is imported, linked or
vendored.  The plain-text ``.ORB_INDX`` is parsed here directly; the sparse binary
matrices (``.HSX``, ``.DM``) are read through ``sisl`` (MPL-2.0), an optional extra.

**What SIESTA can supply** (checked against its source and a real run):

* overlap, the full Kohn-Sham ``H`` and the density matrix, in its numerical
  atomic-orbital (NAO) basis -- so ``hcore = H[0]`` and ``veff = H[dm] - H[0]`` follow
  from Hamiltonian evaluations, as for any :mod:`~.callback` backend;
* the AO-to-atom map and per-AO ``(n, l, m, zeta, polarisation)`` labels;
* **no exact exchange** in any release (the hybrid-functional branch, MR !518, is not
  merged), so no ``veff_hf`` / ``get_k``;
* **no two-electron integrals**, so no ``eri``.

The backend therefore declares ``hcore`` and ``energy_elec`` only.  That is enough for
the DFT-in-DFT path; the WF-in-DFT downfold refuses up front for lack of ``eri`` and
``exact_exchange``.

**Status.**  EmbASI (whose changes are its maintainers') has no ``QMCodeAdapter`` for
ASE's ``Siesta`` calculator yet, so ``ProjectionEmbedding`` cannot drive SIESTA.
:func:`build_siesta_backend` checks for one and refuses with that reason until it exists;
everything on this side -- calculator construction, readers, the capability declaration,
the evaluator wiring -- is in place and starts working unchanged once EmbASI registers
the adapter.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embasi_qiskit_integration.backends.callback import CallbackIntegrals, LayerEvaluator
from embasi_qiskit_integration.backends.registry import QMBackend, register_backend

#: Environment variable naming the directory with the ``<Element>.psml``/``.psf`` files.
PSEUDO_ENV = "SIESTA_PP_PATH"
#: Environment variable with the command that runs SIESTA (ASE's own convention).
COMMAND_ENV = "ASE_SIESTA_COMMAND"

_L_LETTER = "spdfghi"


@dataclass(frozen=True)
class SiestaOrbital:
    """One row of a SIESTA ``.ORB_INDX`` file (unit-cell orbitals only)."""

    index: int  # 0-based orbital index
    atom: int  # 0-based atom index
    species: str
    n: int
    l: int  # noqa: E741 - the angular-momentum quantum number
    m: int
    zeta: int
    polarisation: bool
    symbol: str  # SIESTA's own m label ("py", "Pdxy", ...)

    @property
    def shell(self) -> str:
        """``"2p"``, or ``"3d*"`` for a polarisation function (never a valence shell)."""
        return f"{self.n}{_L_LETTER[self.l]}{'*' if self.polarisation else ''}"


def read_orb_indx(path: str | Path) -> list[SiestaOrbital]:
    """Parse a SIESTA ``.ORB_INDX`` file (written with ``WriteOrbIndx true``).

    Only the unit-cell orbitals are returned (``isc == 0 0 0``), which is the AO basis of
    a Gamma-point molecular calculation.
    """
    out: list[SiestaOrbital] = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            # io ia is spec iao n l m z p sym rc isc(3) iuo
            if len(parts) < 15 or not parts[0].isdigit():
                continue
            isc = tuple(int(x) for x in parts[12:15])
            if isc != (0, 0, 0):
                continue
            out.append(
                SiestaOrbital(
                    index=int(parts[0]) - 1,
                    atom=int(parts[1]) - 1,
                    species=parts[3],
                    n=int(parts[5]),
                    l=int(parts[6]),
                    m=int(parts[7]),
                    zeta=int(parts[8]),
                    polarisation=parts[9].upper().startswith("T"),
                    symbol=parts[10],
                )
            )
    if not out:
        raise ValueError(f"{path} holds no unit-cell orbitals; is it a SIESTA .ORB_INDX?")
    out.sort(key=lambda o: o.index)
    if [o.index for o in out] != list(range(len(out))):
        raise ValueError(f"{path}: orbital indices are not contiguous")
    return out


def _element(species: str) -> str:
    """The element of a SIESTA species label: ASE names them ``O.1``, ``Fe.2``, ..."""
    return species.split(".", 1)[0]


def ao_labels_from_orbitals(
    orbitals: Sequence[SiestaOrbital],
) -> list[tuple[int, str, str, str]]:
    """``(atom, element, shell, m)`` per AO, in the protocol's label convention.

    The element drops ASE's ``.N`` species suffix, so ``"O 2p"`` matches; polarisation
    functions carry a ``*`` (``"3d*"``) and never match a valence label.
    """
    return [(o.atom, _element(o.species), o.shell, o.symbol) for o in orbitals]


def read_matrices(label: str | Path) -> dict[str, np.ndarray]:
    """``S``, ``H`` (per spin) and ``DM`` (per spin) at Gamma from ``<label>.HSX`` / ``.DM``.

    Uses ``sisl`` (install the ``siesta`` extra).  Matrices are dense ``(nao, nao)`` in
    Hartree (sisl reports H in eV); spin-polarised runs give two channels.
    """
    try:
        import sisl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "reading SIESTA's binary matrices needs sisl: pip install "
            "'embasi-qiskit-integration[siesta]'"
        ) from exc
    from ase.units import Hartree

    label = Path(label)
    hsx = sisl.get_sile(str(label.with_suffix(".HSX")))
    ham = hsx.read_hamiltonian()
    dm = sisl.get_sile(str(label.with_suffix(".DM"))).read_density_matrix(geometry=ham.geometry)
    n_spin = 2 if ham.spin.is_polarized else 1
    s = np.asarray(ham.Sk(format="array")).real
    # sisl takes `spin=` only on a polarised matrix; an unpolarised one has one channel.
    spins: list[dict[str, int]] = [{"spin": i} for i in range(n_spin)] if n_spin == 2 else [{}]
    # What sisl returns is NOT the Kohn-Sham H: each channel comes back as H_s - Ef_s*S
    # (sisl shifts HSX v1/v2 files to the Fermi level on read, and with Spin.Fix SIESTA
    # stores each channel against its *own* Fermi level, while the HSX file carries only
    # one).  Add each channel's Fermi level back, so `hcore = H[0]`, `veff = H[dm] - H[0]`
    # and the alpha/beta splitting carry no density-dependent Ef*S (checked on real runs:
    # the generalised eigenvalues then equal SIESTA's .EIG per spin to 1e-7 eV).
    fermi = _fermi_levels(hsx, label, n_spin)
    h = tuple(
        (np.asarray(ham.Hk(format="array", **k)).real + ef * s) / Hartree
        for k, ef in zip(spins, fermi, strict=True)
    )
    d = tuple(np.asarray(dm.Dk(format="array", **k)).real for k in spins)
    return {"overlap": s, "hamiltonian": np.stack(h), "density": np.stack(d)}


def _fermi_levels(hsx: Any, label: Path, n_spin: int) -> tuple[float, ...]:
    """Per spin channel, the Fermi level (eV) sisl subtracted from H when reading ``hsx``.

    A v0 HSX file (SIESTA < 5) has no Fermi level and is not shifted (zeros).  Otherwise
    the file's single Ef applies to an unpolarised run, or to a spin-polarised one with a
    common Fermi level.  A ``Spin.Fix`` run has one Fermi level per spin, printed only in
    ``<label>.out`` (``Fermi_up`` / ``Fermi_dn``); without that file the beta channel
    cannot be corrected, so a polarised run then raises rather than return a wrong H.
    """
    import re
    import warnings

    if getattr(hsx, "version", 0) == 0:
        return (0.0,) * n_spin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sisl's own "no usable Ef" warning (it uses 0)
        common = hsx.read_fermi_level()
    common = 0.0 if common is None else float(common)
    if n_spin == 1:
        return (common,)
    out = label.with_suffix(".out")
    if not out.is_file():
        raise FileNotFoundError(
            f"{out} is needed to read a spin-polarised SIESTA Hamiltonian: its per-spin "
            "Fermi levels (Fermi_up/Fermi_dn) are not stored in the .HSX file"
        )
    text = out.read_text(errors="replace")
    number = r"(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)"
    up = re.findall(r"Fermi_up\s*=\s*" + number, text)
    dn = re.findall(r"Fermi_dn\s*=\s*" + number, text)
    if up and dn:
        return (float(up[-1]), float(dn[-1]))
    return (common, common)  # free spin: one Fermi level for both channels


def basis_atoms_from_orbitals(orbitals: Sequence[SiestaOrbital]) -> np.ndarray:
    return np.array([o.atom for o in orbitals], dtype=int)


def probe_basis(atoms: Any, cfg: Any, *, directory: str | Path) -> list[SiestaOrbital]:
    """The NAO basis SIESTA builds for ``atoms``, from a zero-SCF-step run.

    A SIESTA basis is fixed by the species, the basis-size setting and the
    pseudopotentials, so one run with ``MaxSCFIterations 0`` (about a second) writes the
    ``.ORB_INDX`` the AO-to-atom map and labels are read from -- before any embedding
    evaluation, so the backend can be sized up front.
    """
    calc = siesta_calculator(cfg, xc=cfg.xc_ll, spin_polarised=False, directory=str(directory))
    calc.parameters["fdf_arguments"] = {
        **calc.parameters.get("fdf_arguments", {}),
        "MaxSCFIterations": 0,
        # ASE's own key spelling: it writes `SCFMustConverge True` itself, so a dotted
        # variant would leave two conflicting keys in the fdf.
        "SCFMustConverge": False,
        "SaveHS": False,
        "Write.DM": False,
    }
    probe = atoms.copy()
    probe.calc = calc
    try:
        probe.get_potential_energy()
    except Exception:  # noqa: BLE001 - an unconverged 0-step run may still have written it
        pass
    hits = sorted(Path(directory).glob("*.ORB_INDX"))
    if not hits:
        raise RuntimeError(f"SIESTA basis probe in {directory} wrote no .ORB_INDX")
    return read_orb_indx(hits[0])


# --------------------------------------------------------------------------- #
# Calculator construction and the registered backend
# --------------------------------------------------------------------------- #
def _embasi_supports(calc: Any) -> bool:
    """Whether EmbASI has a ``QMCodeAdapter`` for ``calc`` (a calculator or its class name)."""
    try:
        from embasi.qmcode_adapters import implemented_calculators
    except (ImportError, RuntimeError, OSError):  # EmbASI / its MPI not available
        return False
    name = calc if isinstance(calc, str) else type(calc).__name__
    return name in implemented_calculators


#: Vacuum (Angstrom) between periodic images of a molecule given without a cell.
MOLECULE_VACUUM = 10.0


def ensure_vacuum_cell(atoms: Any, vacuum: float = MOLECULE_VACUUM) -> None:
    """Give a cell-less molecule an orthorhombic box with ``vacuum`` between its images.

    SIESTA is always periodic.  Without ``LatticeVectors`` (ASE writes none for a zero
    cell) it picks an automatic cell that only keeps the orbitals of neighbouring images
    from overlapping -- about 6.4 x 8.0 x 6.5 Angstrom for water, not vacuum -- so the
    images interact through Hartree and xc.  Set in place, positions untouched, because
    EmbASI copies these very atoms into every layer; an explicit cell is left alone.
    """
    if np.any(np.asarray(atoms.cell)):
        return
    extent = np.ptp(np.asarray(atoms.get_positions(), dtype=float), axis=0)
    atoms.set_cell(np.diag(extent + float(vacuum)))


def siesta_calculator(
    cfg: Any, *, xc: str, spin_polarised: bool, directory: str, total_spin: int = 0
) -> Any:
    """ASE's file-based ``Siesta`` calculator for one embedding level.

    A spin-polarised run pins ``Spin.Total = total_spin`` (``n_alpha - n_beta``, the
    workflow's ``--spin``) with ``Spin.Fix``.  Without it SIESTA starts from a
    non-magnetic guess and an open shell converges to the *unpolarised* solution
    (checked: OH doublet gives N_alpha = N_beta = 3.5 and H_alpha == H_beta).

    Basis, mesh cutoff and pseudopotential path come from the workflow settings
    (``siesta_basis``, ``siesta_mesh_cutoff``, ``SIESTA_PP_PATH``).  Matrix output
    (``SaveHS``, ``Write.DM``, ``WriteOrbIndx``) is always on: the readers above need it.
    """
    from ase.calculators.siesta import Siesta

    pseudo = os.environ.get(PSEUDO_ENV)
    if not pseudo:
        raise RuntimeError(
            f"--qm_code siesta needs ${PSEUDO_ENV}: the directory holding the "
            "<Element>.psml (or .psf) pseudopotentials"
        )
    command = os.environ.get(COMMAND_ENV) or (
        f"{shutil.which('siesta')} < PREFIX.fdf > PREFIX.out" if shutil.which("siesta") else None
    )
    if command is None:
        raise RuntimeError(
            f"--qm_code siesta needs the siesta executable on PATH, or ${COMMAND_ENV}"
        )
    functional = {"PBE": ("GGA", "PBE"), "LDA": ("LDA", "PZ"), "PZ": ("LDA", "PZ")}.get(xc.upper())
    if functional is None:
        raise ValueError(
            f"--qm_code siesta: xc {xc!r} is not available (no exact exchange in a SIESTA "
            "release; use a semilocal functional such as PBE)"
        )
    return Siesta(
        command=command,
        directory=directory,
        xc=functional,
        basis_set=getattr(cfg, "siesta_basis", "DZP"),
        mesh_cutoff=float(getattr(cfg, "siesta_mesh_cutoff", 200.0)) * 13.605693122994,  # Ry->eV
        spin="collinear" if spin_polarised else "non-polarized",
        pseudo_path=pseudo,
        fdf_arguments={
            "SaveHS": True,
            "Write.DM": True,
            "WriteOrbIndx": True,
            **({"Spin.Fix": True, "Spin.Total": float(total_spin)} if spin_polarised else {}),
            # --scf_max_cycle is honoured by every backend.
            **(
                {"MaxSCFIterations": int(cfg.scf_max_cycle)}
                if getattr(cfg, "scf_max_cycle", None) is not None
                else {}
            ),
        },
    )


@register_backend(
    "siesta",
    backend_only_settings={"siesta_basis": "DZP", "siesta_mesh_cutoff": 200.0},
)
def build_siesta_backend(atoms: Any, charge: int, cfg: Any) -> QMBackend:
    """SIESTA calculators plus a DFT-only :class:`CallbackIntegrals` over them."""
    if str(cfg.xc_hl).strip().upper() == "HF":
        raise NotImplementedError(
            "--qm_code siesta supports DFT-in-DFT only: SIESTA has no exact exchange and "
            "no two-electron integrals, so the WF-in-DFT downfold (xc_hl=HF) cannot be "
            "built.  Use a KS --xc_hl, or --qm_code pyscf."
        )
    if charge != 0:
        raise NotImplementedError("--qm_code siesta: charged systems are not wired yet")
    # `--basis` names a Gaussian basis; SIESTA's NAO basis is `--siesta_basis`.  Refuse a
    # non-default one rather than silently run DZP in its place.
    if str(getattr(cfg, "basis", "sto-3g")).lower() != "sto-3g":
        raise ValueError(
            f"--basis {cfg.basis!r} is a Gaussian basis and has no effect with --qm_code "
            "siesta; set SIESTA's numerical-orbital basis with --siesta_basis (SZ/DZ/DZP...)"
        )
    # Before the calculators: building them needs $SIESTA_PP_PATH and the executable,
    # and a user without SIESTA should hear the real reason first.
    if not _embasi_supports("Siesta"):
        raise NotImplementedError(
            "--qm_code siesta: EmbASI has no QMCodeAdapter for ASE's Siesta calculator, "
            "so ProjectionEmbedding cannot drive SIESTA yet (EmbASI's registry lists "
            "only its supported codes).  This backend is ready on our side and works "
            "unchanged once EmbASI registers an adapter for 'Siesta'."
        )
    ensure_vacuum_cell(atoms)
    spin_polarised = bool(cfg.spin != 0 or cfg.unrestricted)
    workdir = Path(getattr(cfg, "siesta_workdir", None) or "siesta_calc")
    calc_ll, calc_hl = (
        siesta_calculator(
            cfg,
            xc=xc,
            spin_polarised=spin_polarised,
            directory=str(workdir / level),
            total_spin=int(cfg.spin),
        )
        for xc, level in ((cfg.xc_ll, "ll"), (cfg.xc_hl, "hl"))
    )
    n_spin = 2 if spin_polarised else 1
    ev_ll = LayerEvaluator(atoms, calc_ll, n_spin=n_spin, label=str(workdir / "eval_ll"))
    ev_hl = LayerEvaluator(atoms, calc_hl, n_spin=n_spin, label=str(workdir / "eval_hl"))
    orbitals = probe_basis(atoms, cfg, directory=workdir / "basis_probe")
    symbols = atoms.get_chemical_symbols()
    ints = CallbackIntegrals(
        ev_ll,
        ev_hl,
        None,  # no exact exchange in SIESTA
        nao=len(orbitals),
        basis_atoms=basis_atoms_from_orbitals(orbitals),
        ao_labels=ao_labels_from_orbitals(orbitals),
        describe={
            "basis": f"siesta:{getattr(cfg, 'siesta_basis', 'DZP')}",
            "natm": len(atoms),
            "charge": int(charge),
            "spin": int(cfg.spin),
            "atom_coords": np.asarray(atoms.get_positions(), dtype=float),
            "atom_symbols": symbols,
            "xc_hl": str(cfg.xc_hl),
        },
        ll_spin_resolved=spin_polarised,
        eri_provider=None,  # no two-electron integrals in SIESTA
        cheap_evaluations=False,  # every evaluation is a full siesta run
    )
    return QMBackend(name="siesta", calc_ll=calc_ll, calc_hl=calc_hl, integrals=ints)
