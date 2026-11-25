# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SIESTA backend (``--qm_code siesta``): WF-in-DFT driven through ASI.

**Scope.**  SIESTA is the low-level PBE environment; the high level is the
external correlated solver.  The AO two-electron integrals come from a
SIESTA-QCOMP NetCDF file written by the low-level evaluator
(``QComp.Workflow DROP`` + ``QComp.Protocol FILE``, ``HFX.TruncateDM False``),
and exact exchange is computed directly from those ERIs (``J - K/2``), so
SIESTA needs no pure-HF mean field.  DFT-in-DFT with SIESTA is deliberately
not supported; use ``--qm_code pyscf`` for that reference path.

**Licence boundary.**  SIESTA is GPL-3.0; this package is Apache-2.0.  Running it
through ASI loads a separately built ``libsiesta.so`` into this process via
``ctypes`` (that is how EmbASI drives every ASI code), so the deployment must be
reviewed against the GPL boundary before it is distributed.  No SIESTA source is
vendored here; the matrices come through EmbASI's ASI callbacks.  The file-based
``.ORB_INDX`` reader is still available for deployments that prefer an external
``siesta`` executable for the basis probe.

**What SIESTA can supply** (checked against the ``qcomp-5.4-asi`` branch's source
and a real build):

* overlap, the full Kohn-Sham ``H`` and the density matrix, in its numerical
  atomic-orbital (NAO) basis -- so ``hcore = H[0]`` and ``veff = H[dm] - H[0]`` follow
  from Hamiltonian evaluations, as for any :mod:`~.callback` backend;
* the AO-to-atom map from ASI's type-101 basis descriptors (atom index only; no
  per-AO shell labels, so ``ao_labels`` is unavailable);
* no pure-HF mean field: the branch supports hybrid XC (PBE0/HSE06, <=50% exchange)
  but not the 100% exchange a ``veff_hf`` probe would need;
* **AO ERIs**, when built with ``SIESTA_WITH_QCOMP=ON`` (hybrid-XC/libint), via the
  NetCDF file read by :class:`~._siesta_qcomp.SiestaEriProvider`.

The backend declares ``hcore``/``energy_elec``/``eri``/``exact_exchange``/
``arbitrary_dm_veff``.  The A-fragment footing is the supersystem frame:
EmbASI's ``9c21cac`` sets ``ghosts=0`` on every layer, so the default
:meth:`CallbackIntegrals.layer_footing` is the right identity.

**Status.**  EmbASI (whose changes are its maintainers') has no ``QMCodeAdapter`` for
SIESTA, and its generic ASI extraction expects three Hamiltonian components per SCF
iteration while SIESTA exports one total ``H`` per step.  This module installs a
self-contained, removable shim (:mod:`~._embasi_compat`) that supplies the adapter
and works around the known EmbASI issues.  It is bypassed automatically once EmbASI
registers its own ``"Siesta"`` adapter.

TODO(embasi-siesta-shim): remove the shim install below and the
``write_inputfiles`` alias once EmbASI ships a Siesta adapter and calls
``write_input`` for non-Aims calculators.  The ASI calculator construction,
``probe_basis_asi`` and the QCOMP ERI provider are permanent and stay.

**ASI constraints** (enforced from ``Src/asi_interface/asi_libcalls.F90``): one MPI
rank, unpolarized Gamma-point sampling, no auxiliary supercell, diagonalization and
DM/H mixing only, and at least one SCF iteration.  ``--spin``/``--unrestricted`` and
multi-rank runs are refused here rather than aborting inside Fortran.
"""

from __future__ import annotations

import copy
import functools
import importlib.metadata
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embasi_qiskit_integration.backends._siesta_qcomp import QCOMP_ERI_NAME, SiestaEriProvider
from embasi_qiskit_integration.backends.callback import CallbackIntegrals, LayerEvaluator
from embasi_qiskit_integration.backends.registry import QMBackend, register_backend

#: Environment variable naming the directory with the ``<Element>.psml``/``.psf`` files.
PSEUDO_ENV = "SIESTA_PP_PATH"
#: Environment variable with the command that runs SIESTA (ASE's own convention).
COMMAND_ENV = "ASE_SIESTA_COMMAND"
#: Environment variable naming the ASI implementation (``libsiesta.so``).
ASI_ENV = "ASI_LIB_PATH"
#: First asi4py release with SIESTA flavour 3 (ASI main, MR !112).
MIN_ASI4PY = (1, 4, 10)

_L_LETTER = "spdfghi"


def _asi4py_version() -> tuple[int, ...] | None:
    """Installed ``asi4py`` version as an integer tuple, or None if absent."""
    try:
        raw = importlib.metadata.version("asi4py")
    except importlib.metadata.PackageNotFoundError:
        return None
    parts: list[int] = []
    for piece in raw.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def require_asi_support() -> str:
    """Return ``ASI_LIB_PATH`` after checking the wrapper understands flavour 3.

    ``asi4py`` <= 1.4.9 maps only flavours 0-2 and raises ``KeyError: 3`` inside
    ``ASIlib.init`` (which swallows it, leaving a half-initialised object).  Fail
    here with the install instruction instead.
    """
    lib_path = os.environ.get(ASI_ENV)
    if not lib_path:
        raise RuntimeError(
            f"--qm_code siesta needs ${ASI_ENV}: the ASI-enabled SIESTA shared "
            "library (built with SIESTA_WITH_ASI=ON and BUILD_SHARED_LIBS=ON). "
            "EmbASI drives SIESTA in-process through ASI; the plain `siesta` "
            "executable is not enough."
        )
    if not Path(lib_path).is_file():
        raise RuntimeError(f"${ASI_ENV} points to {lib_path!r}, which is not a file")
    version = _asi4py_version()
    if version is None:
        raise RuntimeError(
            "asi4py is not installed; SIESTA ASI flavour 3 needs asi4py >= 1.4.10 "
            "(install the `siesta` extra or "
            "`pip install 'asi4py @ git+https://gitlab.com/pvst/asi.git@303a9d7#subdirectory=pyasi'`)"
        )
    if version < MIN_ASI4PY:
        have = ".".join(str(part) for part in version)
        raise RuntimeError(
            f"asi4py {have} has no SIESTA flavour 3 (KeyError inside ASIlib.init); "
            "install asi4py >= 1.4.10 from ASI main: "
            "`pip install 'asi4py @ git+https://gitlab.com/pvst/asi.git@303a9d7#subdirectory=pyasi'`"
        )
    return lib_path


def _mpi_size() -> int:
    """``MPI.COMM_WORLD`` size, or 1 when MPI is unusable (single process)."""
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError, OSError):
        return 1
    return int(MPI.COMM_WORLD.Get_size())


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
    """Parse a SIESTA ``.ORB_INDX`` file (written with ``WriteOrbitalIndex true``).

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


def basis_atoms_from_orbitals(orbitals: Sequence[SiestaOrbital]) -> np.ndarray:
    return np.array([o.atom for o in orbitals], dtype=int)


def probe_basis(atoms: Any, cfg: Any, *, directory: str | Path) -> list[SiestaOrbital]:
    """The NAO basis SIESTA builds for ``atoms``, from a zero-SCF-step run.

    A SIESTA basis is fixed by the species, the basis-size setting and the
    pseudopotentials, so one run with ``MaxSCFIterations 0`` (about a second) writes the
    ``.ORB_INDX`` the AO-to-atom map and labels are read from -- before any embedding
    evaluation, so the backend can be sized up front.

    This path needs a ``siesta`` executable.  :func:`probe_basis_asi`, used by the
    ASI backend, does not; it exists for deployments that still have the binary.
    """
    calc = siesta_calculator(cfg, xc=cfg.xc_ll, directory=str(directory), max_scf_cycles=0)
    calc.parameters["fdf_arguments"] = {
        **calc.parameters.get("fdf_arguments", {}),
        "MaxSCFIterations": 0,
        # ASE's own key spelling: it writes `SCFMustConverge True` itself, so a dotted
        # variant would leave two conflicting keys in the fdf.
        "SCFMustConverge": False,
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


def _write_inputfiles(
    self: Any, atoms: Any, properties: Any = None, system_changes: Any = None
) -> Any:
    """``write_inputfiles`` bridge for EmbASI's ASI initializer (shim).

    TODO(embasi-siesta-shim): delete once EmbASI's ``calc_initializer``
    (atoms_embedding_asi.py:155) falls back to ``write_input`` for non-Aims
    calculators.  ASE's ``Siesta`` calculator only has ``write_input``, and it
    unconditionally adds ``DM.UseSaveDM True`` unless the species change (see
    ``Siesta.write_input``); a non-empty ``system_changes`` keeps our
    ``DM.UseSaveDM False`` so a stale ``.DM`` cannot override the density
    EmbASI imports.  This lives on a dynamically named ``Siesta`` subclass so
    EmbASI's ``type(calc).__name__`` dispatch still finds the adapter and
    ``deepcopy`` of its runtime calculator rebinds ``self`` correctly.
    """
    return self.write_input(
        atoms,
        properties=properties,
        system_changes=["numbers"] if system_changes is None else system_changes,
    )


def siesta_calculator(
    cfg: Any,
    *,
    xc: str,
    directory: str = ".",
    max_scf_cycles: int | None = None,
    require_asi: bool = False,
    qcomp_eris: bool = False,
) -> Any:
    """ASE's ``Siesta`` calculator for one ASI-driven embedding level.

    The calculator is primarily an **input writer**: EmbASI's ASI path calls
    ``write_inputfiles`` in its own work directory and then runs ``libsiesta.so``
    in-process.  ASI therefore needs no ``siesta`` executable; only the file-based
    basis probe does, and :func:`probe_basis_asi` replaces it when ASI is present.

    ``directory`` must stay ``"."`` for ASI: the ASE calculator writes where
    ``directory`` points, while ``ASIlib`` chdirs into its own work dir and calls
    ``ASI_init("siesta.fdf")`` there.  A per-level absolute path would put the
    fdf where SIESTA never looks.

    SIESTA's ASI flavour is unpolarized Gamma-point only; ``Spin.Fix`` is
    deliberately not written.  ``DM.UseSaveDM`` is off so a stale ``.DM`` in a
    reused work directory cannot override the density EmbASI imports.

    With ``qcomp_eris``, add the fdf options that make SIESTA-QCOMP write the AO
    ERIs to ``qcomp_eris.nc`` in the ASI work directory.  ``HFX.TruncateDM False``
    keeps the file independent of the current density (so even the zero-density
    reference evaluation stores all integrals); the tolerance filters only tiny
    values.  The file is read by :class:`~._siesta_qcomp.SiestaEriProvider`.
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
        if not require_asi:
            raise RuntimeError(
                f"--qm_code siesta needs the siesta executable on PATH, or ${COMMAND_ENV}"
            )
        # Never executed on the ASI path; keeps ASE's FileIOCalculator complete.
        command = "siesta < PREFIX.fdf > PREFIX.out"
    functional = {"PBE": ("GGA", "PBE"), "LDA": ("LDA", "PZ"), "PZ": ("LDA", "PZ")}.get(xc.upper())
    if functional is None:
        raise ValueError(
            f"--qm_code siesta: xc {xc!r} is not available (no pure-HF mean field; "
            "use a semilocal functional such as PBE)"
        )
    fdf_arguments: dict[str, Any] = {
        # Correct SIESTA key: SIESTA writes .ORB_INDX unless this is false.
        "WriteOrbitalIndex": True,
        "DM.UseSaveDM": False,
        "SCFMustConverge": True,
    }
    if max_scf_cycles is not None:
        fdf_arguments["MaxSCFIterations"] = int(max_scf_cycles)
    if qcomp_eris:
        fdf_arguments.update(
            {
                "QComp.Workflow": "DROP",
                "QComp.Protocol": "FILE",
                "QComp.DataAddress": QCOMP_ERI_NAME,
                # Store every ERI regardless of the current DM (DM_max = 1); otherwise
                # the zero-density reference evaluation would write an empty file.
                "HFX.TruncateDM": False,
                "HFX.StoreERIsTolerance": 1e-12,
                # QComp-only runs (eris_wanted and not hfx_wanted) force QCOMP's
                # short-range erfc(omega*r)/r operator.  With the default omega=0.11
                # the exported values are NOT full Coulomb and cannot be used as the
                # FCI h2 tensor.  A tiny omega makes erfc(omega*r) ~ 1, i.e. full 1/r.
                # Verified against a PySCF primitive-GTO reference: QCOMP self-ERIs
                # then equal the full-Coulomb values to float32 precision.
                "HFX.Omega": 1.0e-8,
            }
        )
    # TODO(embasi-siesta-shim): drop the dynamic subclass once EmbASI calls
    # write_input directly; the class name must stay "Siesta" for its adapter
    # dispatch, and the method must live on the class so deepcopy rebinds it.
    calculator_class = type("Siesta", (Siesta,), {"write_inputfiles": _write_inputfiles})
    calc = calculator_class(
        command=command,
        directory=directory,
        xc=functional,
        basis_set=getattr(cfg, "siesta_basis", "DZP"),
        mesh_cutoff=float(getattr(cfg, "siesta_mesh_cutoff", 200.0)) * 13.605693122994,  # Ry->eV
        spin="non-polarized",
        pseudo_path=pseudo,
        # Symlinks default to True on POSIX; some shared/container filesystems
        # reject them, and SIESTA then dies with "Pseudopotential file not found".
        symlink_pseudos=False,
        fdf_arguments=fdf_arguments,
    )
    return calc


def probe_basis_asi(
    atoms: Any,
    calc: Any,
    *,
    directory: str | Path,
    lib_path: str | None = None,
) -> tuple[int, np.ndarray]:
    """Size the NAO basis from ASI, without running an SCF or an executable.

    ``ASI_init`` reads the fdf and sets up species/orbitals; ``ASI_get_basis_size``
    and the type-101 basis descriptors are then valid before ``ASI_run``.  This
    replaces the zero-SCF executable probe for ASI-only deployments -- and, unlike
    ``.ORB_INDX``, needs no ``siesta`` binary.

    Returns:
        ``(n_basis, basis_atoms)`` where ``basis_atoms`` maps each AO to its atom.
    """
    lib_path = lib_path or require_asi_support()
    from asi4py.asecalc import ASI_ASE_calculator

    work = Path(directory)
    work.mkdir(parents=True, exist_ok=True)
    # The initializer runs after ASIlib chdirs into ``work``; ASE must write there.
    # A copy keeps the probe's one-step, non-converging settings out of the
    # calculators EmbASI later deep-copies for real evaluations.
    probe = copy.deepcopy(calc)
    probe.directory = "."
    probe.parameters["fdf_arguments"] = {
        **probe.parameters.get("fdf_arguments", {}),
        "MaxSCFIterations": 1,
        "SCFMustConverge": False,
    }

    def write_input(asi: Any) -> None:
        probe.write_inputfiles(asi.atoms, properties=["energy"])

    asi_calc = ASI_ASE_calculator(lib_path, write_input, None, atoms, work_dir=str(work))
    try:
        # Real SIESTA's validated lifecycle is init -> run -> finalize.  An
        # init-only probe left its FFT state inconsistent and the next ASI_init
        # segfaulted in gpfa; one SCF step makes this a normal cycle.
        asi_calc.get_potential_energy()
        asi = asi_calc.asi
        try:
            n_basis = int(asi.n_basis)
            basis_atoms = np.asarray(asi.basis_atoms, dtype=int)
        except KeyError as exc:  # asi4py <= 1.4.9: no flavour-3 dispatch
            raise RuntimeError(
                "asi4py has no SIESTA flavour 3; install asi4py >= 1.4.10 from ASI main "
                "(see the `siesta` extra)"
            ) from exc
    finally:
        asi_calc.close()

    if n_basis <= 0:
        raise RuntimeError(
            f"ASI reported {n_basis} basis functions after initialising from {work}; "
            "check the pseudopotentials and the generated siesta.fdf"
        )
    if basis_atoms.shape != (n_basis,):
        raise RuntimeError(
            f"ASI basis descriptors gave shape {basis_atoms.shape}, expected ({n_basis},)"
        )
    if np.any((basis_atoms < 0) | (basis_atoms >= len(atoms))):
        raise RuntimeError(f"ASI basis descriptors map beyond the {len(atoms)} atoms")
    return n_basis, basis_atoms


@functools.lru_cache(maxsize=None)
def _valence_charge(element: str, pseudo_dir: str) -> float:
    """Pseudopotential valence charge (``z-pseudo``/``Z_valence``) for an element."""
    directory = Path(pseudo_dir)
    psml = directory / f"{element}.psml"
    if psml.is_file():
        try:
            root = ET.parse(psml).getroot()
        except ET.ParseError as exc:
            raise RuntimeError(f"cannot parse {psml}") from exc
        for node in root.iter():
            value = node.attrib.get("z-pseudo")
            if value is None:
                continue
            try:
                charge = float(value)
            except ValueError:
                continue
            if charge > 0:
                return charge
        raise RuntimeError(f"{psml} has no numeric z-pseudo valence charge")
    psf = directory / f"{element}.psf"
    if psf.is_file():
        # PSF header: the fourth line ends with Z_valence.  Accept D exponents.
        lines = psf.read_text(errors="replace").splitlines()
        candidates = ([lines[3]] if len(lines) > 3 else []) + lines[:8]
        for line in candidates:
            match = re.search(r"([-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?)\s*$", line)
            if match is None:
                continue
            try:
                charge = float(match.group(1).replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            if charge > 0:
                return charge
        raise RuntimeError(f"{psf} has no Z_valence in its header")
    raise RuntimeError(
        f"no pseudopotential for {element} in ${PSEUDO_ENV}={pseudo_dir} "
        f"(looked for {element}.psml and {element}.psf)"
    )


def atomic_reference_energy(layer: Any) -> float:
    """SIESTA's neutral-atom reference ``Ena`` (Hartree) for a completed layer.

    This is the fallback when the shim cannot record the layer's input density;
    production uses :func:`same_density_reference_energy`, which evaluates the
    full bare functional at that density.

    TODO(embasi-siesta-shim): replace both helpers with an ASI/EmbASI-exported
    reference energy once the interface exposes one; parsing ``total.log`` is
    the only route available today.
    """
    outdir = getattr(layer, "outdir", None)
    if not outdir:
        return 0.0
    log = Path(str(outdir)) / "total.log"
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return 0.0
    match = re.search(r"siesta:\s*Ena\s*=\s*([-+0-9.Ee]+)", text)
    if match is None:
        return 0.0
    return float(match.group(1)) / 27.211384500  # SIESTA prints Ena in eV


def _log_energy(log: Path, key: str) -> float:
    """Read one ``siesta: <key> = <value>`` energy from ``total.log``, in Ha."""
    text = log.read_text(errors="replace")
    match = re.search(rf"siesta:\s*{key}\s*=\s*([-+0-9.Ee]+)", text)
    if match is None:
        raise RuntimeError(f"{log}: no 'siesta: {key}' energy")
    return float(match.group(1)) / 27.211384500  # SIESTA prints eV


def _layer_input_density(layer: Any) -> np.ndarray | None:
    """The spin-summed input density recorded by the shim, if available."""
    raw = getattr(layer, "_siesta_input_density", None)
    if raw is None:
        return None
    density = np.asarray(raw, dtype=float)
    if density.ndim != 2:
        return None
    return density


def same_density_reference_energy(layer: Any, integrals: Any) -> float:
    """Align the bare FCI total with EmbASI's low-level A-layer total.

    ``E_low(A)`` is SIESTA's DFT total evaluated at the localized input density
    ``D_A`` (EmbASI's ``ev_corr_total_energy``); the FCI is a bare
    ``hcore + full ERIs + E_nuc`` quantity.  Subtracting their difference puts
    both on the same energy zero, so the pseudopotential/double-counting
    reference cancels.

    SIESTA's ``Exc`` in ``total.log`` is evaluated at the input density of the
    non-SCF run, matching the other terms.  Falls back to the ``Ena``-only
    reference when the shim could not record the density (e.g. a non-SIESTA
    test layer).
    """
    density = _layer_input_density(layer)
    if density is None:
        return atomic_reference_energy(layer)
    outdir = getattr(layer, "outdir", None)
    if not outdir:
        return atomic_reference_energy(layer)
    log = Path(str(outdir)) / "total.log"
    try:
        exc = _log_energy(log, "Exc")
    except (OSError, RuntimeError):
        return atomic_reference_energy(layer)
    # QCOMP writes full-Coulomb ERIs in production (HFX.Omega=1e-8); the
    # Hartree term is therefore the physical 0.5 Tr[D J(D)].
    j, _k = integrals.get_jk(density)
    bare = (
        float(np.einsum("ij,ji->", density, integrals.hcore()))
        + 0.5 * float(np.einsum("ij,ji->", density, j))
        + exc
        + float(integrals.energy_nuc())
    )
    e_low_ev = getattr(layer, "ev_corr_total_energy", None)
    if e_low_ev is None:
        e_low_ev = getattr(layer, "total_energy", None)
    if e_low_ev is None:
        return atomic_reference_energy(layer)
    return float(bare - float(e_low_ev) / 27.211384500)


def nuclear_repulsion(atoms: Any) -> float:
    """Bare ion-ion repulsion (Hartree) from the pseudopotential valence charges.

    SIESTA's ASI energy is an SCF total, not the bare ``E_nuc`` that the WF energy
    assembly needs; a zero-density SCF cannot supply it either (SIESTA's Fermi
    filling aborts with no electrons).  The ionic charges are the pseudopotentials'
    ``z-pseudo`` (PSML) / ``Z_valence`` (PSF), the same charges in the one-electron
    Hamiltonian.  Distances use the ASE Bohr constant.
    """
    pseudo_dir = os.environ.get(PSEUDO_ENV)
    if not pseudo_dir:
        raise RuntimeError(f"computing E_nuc needs ${PSEUDO_ENV}")
    from ase import units

    charges = np.array(
        [_valence_charge(symbol, pseudo_dir) for symbol in atoms.get_chemical_symbols()],
        dtype=float,
    )
    positions = np.asarray(atoms.get_positions(), dtype=float) / units.Bohr
    total = 0.0
    for i in range(len(charges)):
        for j in range(i):
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance == 0.0:
                raise ValueError("two atoms coincide; E_nuc is undefined")
            total += charges[i] * charges[j] / distance
    return float(total)


@register_backend(
    "siesta",
    backend_only_settings={"siesta_basis": "DZP", "siesta_mesh_cutoff": 200.0},
)
def build_siesta_backend(atoms: Any, charge: int, cfg: Any) -> QMBackend:
    """SIESTA **WF-in-DFT** backend: PBE low level, external solver high level.

    The high level is the correlated solver; the two-electron integrals come from
    a SIESTA-QCOMP NetCDF file written by the low-level calculators.  SIESTA never
    runs a pure-HF mean field.  DFT-in-DFT with SIESTA is not supported here (use
    ``--qm_code pyscf`` for that reference path).

    ``calc_hl`` is a semilocal placeholder: the default WF path
    (``ProjectionEmbeddingAdapter.run_low_level``) never runs ``A_HL``;
    ``--relax_hf`` is therefore refused.
    """
    if str(cfg.xc_hl).strip().upper() != "HF":
        raise NotImplementedError(
            "--qm_code siesta supports WF-in-DFT only: set --xc_hl HF (the low level "
            "is --xc_ll, e.g. PBE).  For a DFT-in-DFT reference use --qm_code pyscf."
        )
    if getattr(cfg, "relax_hf", False):
        raise NotImplementedError(
            "--relax_hf with --qm_code siesta is not supported: it runs an embedded "
            "SCF on the high-level calculator, and SIESTA has no pure-HF functional "
            "(its hybrid XC is capped at 50% exchange).  Leave --relax_hf off."
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
    # ASI_init rejects these inside Fortran with an opaque `error stop`; refuse
    # them here while we can still name the workflow flag.
    if cfg.spin != 0 or cfg.unrestricted:
        raise NotImplementedError(
            "--qm_code siesta supports unpolarized Gamma-point sampling only: SIESTA's "
            "ASI interface aborts for any spin-polarized run (asi_libcalls.F90). "
            "Use --spin 0, or a spin-capable backend."
        )
    if _mpi_size() > 1:
        raise NotImplementedError(
            "SIESTA's ASI interface currently requires a single MPI rank (ASI_init "
            "aborts when MPI_COMM_WORLD has more than one process). Run this "
            "embedding without mpirun, or use a different --qm_code."
        )
    lib_path = require_asi_support()
    # TODO(embasi-siesta-shim): delete this import/install block (and the
    # `_embasi_supports` check) once EmbASI registers its own "Siesta" adapter;
    # then rely directly on that adapter.
    from embasi_qiskit_integration.backends._embasi_compat import install_siesta_adapter

    try:
        install_siesta_adapter()
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "--qm_code siesta needs EmbASI (the `embed` extra) and its ASI wrapper"
        ) from exc
    if not _embasi_supports("Siesta"):
        raise NotImplementedError(
            "--qm_code siesta: EmbASI has no QMCodeAdapter for SIESTA and the "
            "compatibility shim could not be installed; check the EmbASI/asi4py install."
        )
    ensure_vacuum_cell(atoms)
    workdir = Path(getattr(cfg, "siesta_workdir", None) or "siesta_calc")
    calc_ll = siesta_calculator(
        cfg,
        xc=cfg.xc_ll,
        directory=".",
        max_scf_cycles=getattr(cfg, "scf_max_cycle", None),
        require_asi=True,
        qcomp_eris=True,
    )
    # Placeholder for EmbASI's A_HL layer; the WF path never runs it.
    calc_hl = siesta_calculator(
        cfg,
        xc=cfg.xc_ll,
        directory=".",
        max_scf_cycles=getattr(cfg, "scf_max_cycle", None),
        require_asi=True,
    )
    ev_ll = LayerEvaluator(atoms, calc_ll, n_spin=1, label=str(workdir / "eval_ll"))
    ev_hl = LayerEvaluator(atoms, calc_hl, n_spin=1, label=str(workdir / "eval_hl"))
    n_basis, basis_atoms = probe_basis_asi(
        atoms, calc_ll, directory=workdir / "basis_probe", lib_path=lib_path
    )
    # The provider reads the file the low-level evaluator writes (same label).
    eri_provider = SiestaEriProvider(workdir / "eval_ll" / QCOMP_ERI_NAME, n_ao=n_basis)
    symbols = atoms.get_chemical_symbols()
    ints = CallbackIntegrals(
        ev_ll,
        ev_hl,
        None,  # no pure-HF mean field in SIESTA; ERIs supply exact exchange
        nao=n_basis,
        basis_atoms=basis_atoms,
        ao_labels=None,  # ASI type-101 descriptors carry only the atom index
        describe={
            "basis": f"siesta:{getattr(cfg, 'siesta_basis', 'DZP')}",
            "natm": len(atoms),
            "charge": int(charge),
            "spin": 0,
            "atom_coords": np.asarray(atoms.get_positions(), dtype=float),
            "atom_symbols": symbols,
            "xc_hl": "HF",
        },
        ll_spin_resolved=False,
        eri_provider=eri_provider,
        cheap_evaluations=False,  # every evaluation is a full SIESTA run
        # The WF downfold needs veff_ll(dm) at arbitrary densities; SIESTA can
        # evaluate that (one ASI run), it is just not cheap.
        arbitrary_dm_veff=True,
        # ASI's energy is an SCF total; E_nuc comes from the pseudopotential charges.
        nuclear_energy=lambda: nuclear_repulsion(atoms),
        # Put the bare-Hamiltonian FCI on the same energy zero as EmbASI's
        # low-level A-layer total (which carries the full pseudopotential /
        # double-counting reference).  ``ints`` is late-bound.
        high_level_reference=lambda layer: same_density_reference_energy(layer, ints),
    )
    return QMBackend(name="siesta", calc_ll=calc_ll, calc_hl=calc_hl, integrals=ints)
