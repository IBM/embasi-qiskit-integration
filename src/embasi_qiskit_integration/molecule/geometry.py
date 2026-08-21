# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Geometry input: ``.xyz`` file -> PySCF ``Mole`` (or ASE ``Atoms``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyscf import gto

logger = logging.getLogger(__name__)

Chem: Any
RDLogger: Any

try:
    from rdkit import Chem, RDLogger
except ImportError:  # optional 'chem' extra
    Chem = None
    RDLogger = None
else:
    # RDKit prints SMILES-parser chatter to stderr by default; silence unless the
    # caller re-enables it (see _rdkit_stderr_for).
    RDLogger.DisableLog("rdApp.*")


def _parse_metadata_charge(metadata: dict[str, str]) -> int | None:
    """Return the ``charge=<int>`` value from an XYZ metadata dict, or None.

    Accepts signed integers: ``0``, ``2``, ``-1``, ``+3``. Non-integer or
    unparseable values are logged as a warning and treated as absent (returns
    None), so a typo in the metadata does not silently mis-set the charge.
    """
    raw = metadata.get("charge", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "XYZ metadata has 'charge=%r' but it is not a valid integer; ignoring it.", raw
        )
        return None


def _rdkit_stderr_for(smiles: str) -> None:
    """Re-parse the SMILES with RDKit's error stream re-enabled so that its
    C-level diagnostic prints to the terminal.

    ``Chem.MolFromSmiles`` returns ``None`` on failure rather than raising.
    RDKit writes the diagnostic ("Explicit valence for atom ... is greater
    than permitted", etc.) from C++ at file-descriptor level, which is why we
    can't cleanly capture it into a Python string without an ``os.dup2()``
    juggle. We just re-enable the RDKit error logger and let the message
    print naturally right before our own ``logger.warning(...)`` -- users
    reading the log will see both.
    """
    RDLogger.EnableLog("rdApp.error")
    try:
        Chem.MolFromSmiles(smiles)
    except Exception:  # RDKit occasionally raises instead of returning None
        pass
    finally:
        RDLogger.DisableLog("rdApp.*")


def parse_xyz_atoms_and_metadata(xyz_file: str | Path) -> tuple[str, dict[str, str]]:
    """Parse an .xyz file into a PySCF-ready atom string and a metadata dict.

    The .xyz format expected::

        <n_atoms>                                    (line 0)
        key1=value1; key2=value2; ...                (line 1, semicolon-separated)
        <symbol> <x> <y> <z>                         (lines 2..n_atoms+1)

    Args:
        xyz_file: path to an .xyz file.

    Returns:
        Tuple ``(atom_str, metadata)``:

          - ``atom_str`` is the ``"C x y z\\nH x y z\\n..."`` string PySCF's
            ``gto.Mole.atom`` accepts, with 9-decimal-place coordinates.
          - ``metadata`` is a ``dict[str, str]`` of the key=value pairs found on
            the second line. Empty dict if the second line carries no ``=``
            signs. Never raises for a malformed comment line.

    Raises:
        ValueError: file too short for its declared atom count, or any atom
            line has fewer than 4 fields (symbol + 3 coordinates), or any
            coordinate cannot be parsed as float.
    """
    xyz_file = Path(xyz_file)
    with xyz_file.open() as fh:
        lines = [ln.rstrip() for ln in fh]
    if not lines:
        raise ValueError(f"{xyz_file}: file is empty")

    try:
        n_atoms = int(lines[0].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{xyz_file}: first line is not an atom count: {lines[0]!r}") from exc

    if len(lines) < n_atoms + 2:
        raise ValueError(
            f"{xyz_file}: header says {n_atoms} atoms but file has only {len(lines)} lines"
        )

    # Metadata line: ';'-separated key=value pairs, tolerant of missing/extra parts.
    metadata: dict[str, str] = {}
    comment = lines[1] if len(lines) > 1 else ""
    for chunk in comment.split(";"):
        chunk = chunk.strip()
        if "=" not in chunk:
            continue
        key, _, val = chunk.partition("=")
        metadata[key.strip()] = val.strip()

    # Atom block.
    out = []
    for i, ln in enumerate(lines[2 : 2 + n_atoms], start=2):
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"{xyz_file}: bad atom line {i}: {ln!r}")
        sym = parts[0]
        try:
            x, y, z = (float(v) for v in parts[1:4])
        except ValueError as exc:
            raise ValueError(f"{xyz_file}: non-numeric coords on line {i}: {ln!r}") from exc
        out.append(f"{sym}  {x:.9f}  {y:.9f}  {z:.9f}")
    atom_str = "\n".join(out)
    return atom_str, metadata


def build_pyscf_mol(
    atom_str: str,
    basis: str | dict[str, str],
    charge: int = 0,
    spin: int = 0,
    max_memory: int | None = None,
    verbose: int = 4,
    ecp: dict[str, str] | None = None,
) -> Any:
    """Build a :class:`pyscf.gto.Mole` from an atom string.

    Small wrapper that centralises the PySCF ``Mole.build()`` boilerplate for the
    geometry consumers -- :class:`~embasi_qiskit_integration.embedding.EmbeddingWorkflow`
    and ``scripts/make_test_data.py`` -- so neither reinvents it.

    Args:
        atom_str: the ``"<sym> <x> <y> <z>\\n..."`` block from
            :func:`parse_xyz_atoms_and_metadata`. Coordinates are Angstrom.
        basis: PySCF basis specification. String (``"cc-pvdz"``) or per-element
            dict (``{"C": "cc-pvtz", "H": "cc-pvdz"}``).
        charge: net molecular charge (default 0). Typically obtained from
            :func:`charge_from_metadata`.
        spin: ``2*S`` = number of unpaired electrons (default 0 = singlet).
        max_memory: PySCF ``mol.max_memory`` in MB. ``None`` (default) leaves
            PySCF's own default in place; raise it for large-basis MP2/ao2mo runs.
        verbose: PySCF verbosity level (default 4 = standard SCF output).
        ecp: optional per-element ECP dict (e.g. ``{"I": "cc-pVTZ-PP"}``).

    Returns:
        A built :class:`pyscf.gto.Mole`. Symmetry is disabled (PySCF's symmetry
        detection can drop cheaply-computed integrals when a molecule is nearly
        symmetric); enable it explicitly at the call site if you need it.
    """
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = int(charge)
    mol.spin = int(spin)
    mol.symmetry = False
    mol.verbose = verbose
    if max_memory is not None:
        mol.max_memory = max_memory
    if ecp is not None:
        mol.ecp = ecp
    mol.build()
    return mol


def charge_from_metadata(metadata: dict[str, str]) -> tuple[int, str | None]:
    """Return ``(charge, smiles)`` for an XYZ metadata dict.

    Precedence and fallback rules (all traced in the log):

      1. SMILES parses AND ``charge=`` present:
           - if they agree -> use them
           - if they disagree -> ``ValueError``: the XYZ metadata is
             self-contradictory and refusing is safer than silently picking
             one.
      2. SMILES parses AND no ``charge=``: use the sum of formal charges of
         the SMILES.
      3. SMILES fails to parse AND ``charge=`` present: use the metadata
         charge, log a warning that the SMILES was ignored.
      4. SMILES fails to parse AND no ``charge=``: assume neutral (0), warn.
      5. No SMILES AND ``charge=`` present: use the metadata charge.
      6. No SMILES AND no ``charge=``: assume neutral (0).

    Charge from SMILES is computed as the sum of atomic formal charges via
    RDKit (``Chem.GetFormalCharge``). This handles ionic groups (``[O-]``,
    ``[NH3+]``, ``[n+]``), multi-charge brackets, zwitterions, and neutral
    molecules correctly.

    Non-standard SMILES (e.g. dative arrows ``->``, ``<-`` used by some
    organometallic notations) are rejected by RDKit. In that case case 3 or
    case 4 kicks in.

    When RDKit itself is not installed (it ships in the optional ``chem`` extra)
    a present SMILES cannot be read at all, so cases 3 and 4 apply for the same
    reason -- but the log says the cross-check was *skipped*, not that the SMILES
    failed to parse, since nothing has actually inspected it.

    Args:
        metadata: dict returned by :func:`parse_xyz_atoms_and_metadata`.

    Returns:
        Tuple ``(charge, smiles)``. ``smiles`` is ``None`` if no SMILES field
        was in the metadata; otherwise it is the original string.

    Raises:
        ValueError: SMILES parses cleanly AND metadata has a ``charge=`` field
            AND the two values disagree (case 1, mismatch).
    """
    smiles = metadata.get("smiles", "").strip() or None
    meta_charge = _parse_metadata_charge(metadata)

    if smiles is None:
        # Cases 5 & 6: no SMILES.
        if meta_charge is not None:
            logger.info("charge=%+d from XYZ metadata (no SMILES present)", meta_charge)
            return meta_charge, None
        return 0, None

    # SMILES present but RDKit absent: fall back exactly as for an unparseable
    # SMILES (cases 3 and 4), while being precise that nothing read it.
    if Chem is None:
        if meta_charge is not None:
            logger.warning(
                "rdkit is not installed (optional 'chem' extra), so SMILES %r could not be "
                "cross-checked; using charge=%+d from XYZ metadata.",
                smiles,
                meta_charge,
            )
            return meta_charge, smiles
        logger.warning(
            "rdkit is not installed (optional 'chem' extra), so SMILES %r could not be read, "
            "and XYZ metadata has no charge= field; assuming neutral (charge=0). Install the "
            "'chem' extra or add an explicit charge= to the XYZ.",
            smiles,
        )
        return 0, smiles

    # SMILES present. Try to parse.
    rd_mol = Chem.MolFromSmiles(smiles)
    if rd_mol is None:
        # Print RDKit's diagnostic on stderr so the user can see WHY it failed.
        _rdkit_stderr_for(smiles)
        if meta_charge is not None:
            # Case 3: fall back to metadata charge.
            logger.warning(
                "RDKit could not parse SMILES %r; using charge=%+d from XYZ metadata instead. "
                "See the RDKit error printed above for the reason.",
                smiles,
                meta_charge,
            )
            return meta_charge, smiles
        # Case 4: neutral fallback.
        logger.warning(
            "RDKit could not parse SMILES %r and XYZ metadata has no charge= field; assuming "
            "neutral (charge=0). See the RDKit error printed above for the reason.",
            smiles,
        )
        return 0, smiles

    smi_charge = int(Chem.GetFormalCharge(rd_mol))
    if meta_charge is None:
        # Case 2: SMILES-only.
        logger.debug("charge=%+d from SMILES %r (no charge= in metadata)", smi_charge, smiles)
        return smi_charge, smiles

    # Case 1: both present. Cross-check.
    if smi_charge != meta_charge:
        raise ValueError(
            f"XYZ metadata is inconsistent: SMILES {smiles!r} implies "
            f"charge={smi_charge:+d} but the metadata field says "
            f"charge={meta_charge:+d}. Fix the XYZ (drop one of the two, or "
            "correct whichever is wrong) before continuing."
        )
    logger.info(
        "charge=%+d confirmed by both SMILES and XYZ metadata (consistency check passed)",
        smi_charge,
    )
    return smi_charge, smiles


def read_xyz(
    xyz_file: str | Path,
    charge_override: int | None = None,
) -> tuple[str, int, str | None]:
    """Parse an ``.xyz`` and resolve its charge: ``(atom_str, charge, smiles)``.

    The single front door for reading a geometry file. Composes
    :func:`parse_xyz_atoms_and_metadata` with :func:`charge_from_metadata`,
    applies ``charge_override``, and logs one provenance line -- so every
    consumer resolves a given file's charge identically and reports it the same
    way, whether it goes on to build a PySCF ``Mole``
    (:func:`read_mol_from_xyz`), an ASE ``Atoms``
    (:func:`read_atoms_from_xyz`), or something else.

    Args:
        xyz_file: path to an ``.xyz`` file.
        charge_override: use this charge instead of the metadata-derived one.

    Returns:
        Tuple ``(atom_str, charge, smiles)``. ``atom_str`` is the Angstrom
        ``"<sym> <x> <y> <z>\\n..."`` block; ``smiles`` is ``None`` when the file
        carries no ``smiles=`` field.

    Raises:
        ValueError: malformed ``.xyz`` (see :func:`parse_xyz_atoms_and_metadata`)
            or a SMILES/``charge=`` contradiction (see
            :func:`charge_from_metadata`).
    """
    atom_str, metadata = parse_xyz_atoms_and_metadata(xyz_file)
    if charge_override is not None:
        charge = charge_override
        smiles = metadata.get("smiles") or None
        source = "override"
    else:
        charge, smiles = charge_from_metadata(metadata)
        source = "XYZ metadata"
        if smiles is None:
            source = "no-SMILES default (neutral)"
        if Chem is None:
            source = "XYZ metadata (SMILES present but rdkit missing)"

    logger.info(
        "[%s] charge=%+d via %s%s",
        Path(xyz_file).name,
        charge,
        source,
        f"  smiles={smiles}" if smiles else "",
    )
    return atom_str, charge, smiles


def split_atom_str(atom_str: str) -> tuple[list[str], list[tuple[float, float, float]]]:
    """Split an atom-string block into ``(symbols, positions)``.

    Positions stay in Angstrom, the unit :func:`parse_xyz_atoms_and_metadata`
    emits and the one ASE expects. This is deliberately *not* routed through a
    built ``Mole``: ``Mole.atom_coords()`` returns **Bohr**, so going via PySCF
    would silently shrink every geometry by a factor of ~1.89.
    """
    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    for line in atom_str.splitlines():
        parts = line.split()
        if not parts:
            continue
        symbols.append(parts[0])
        x, y, z = (float(v) for v in parts[1:4])
        positions.append((x, y, z))
    return symbols, positions


def read_atoms_from_xyz(
    xyz_file: str | Path,
    charge_override: int | None = None,
) -> tuple[Any, int, str | None]:
    """One-shot: ``.xyz`` file -> ``(ase.Atoms, charge, smiles)``.

    The ASE counterpart of :func:`read_mol_from_xyz`, for consumers that need an
    ``ase.Atoms`` rather than a PySCF ``Mole`` -- notably EmbASI's
    ``ProjectionEmbedding``, which takes ASE objects. Charge derivation is shared
    with the PySCF path via :func:`charge_from_metadata`, so both entry points
    agree about a given file by construction.

    ``ase`` is imported lazily (it ships in the ``embed`` extra) so that importing
    this module, and the whole ``.xyz`` -> ``Mole`` path, needs only PySCF.

    Args:
        xyz_file: path to an ``.xyz`` file.
        charge_override: use this charge instead of the metadata-derived one.

    Returns:
        Tuple ``(atoms, charge, smiles)``. ``smiles`` is ``None`` when the file
        carries no ``smiles=`` field.

    Raises:
        ImportError: if ``ase`` is not installed (install the ``embed`` extra).
        ValueError: for a malformed ``.xyz`` (see
            :func:`parse_xyz_atoms_and_metadata`) or a SMILES/``charge=``
            contradiction (see :func:`charge_from_metadata`).
    """
    try:
        from ase import Atoms
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "read_atoms_from_xyz needs 'ase', which ships in the 'embed' extra "
            "(uv pip install -e '.[embed]'). For a PySCF Mole instead, use "
            "read_mol_from_xyz, which needs no extra."
        ) from exc

    atom_str, charge, smiles = read_xyz(xyz_file, charge_override=charge_override)
    symbols, positions = split_atom_str(atom_str)
    return Atoms(symbols=symbols, positions=positions), charge, smiles


def read_mol_from_xyz(
    xyz_file: str | Path,
    basis: str | dict[str, str] = "cc-pvdz",
    spin: int = 0,
    max_memory: int | None = None,
    charge_override: int | None = None,
    verbose: int = 4,
    ecp: dict[str, str] | None = None,
) -> Any:
    """One-shot: .xyz file -> :class:`pyscf.gto.Mole`, charge from SMILES metadata.

    Composes :func:`read_xyz` (parse + charge resolution, shared with
    :func:`read_atoms_from_xyz`) with :func:`build_pyscf_mol`.

    Args:
        xyz_file: path to an .xyz file (may or may not carry SMILES metadata).
        basis: PySCF basis (default ``"cc-pvdz"``).
        spin: ``2*S`` (default 0 = singlet).
        max_memory: PySCF ``mol.max_memory`` in MB; ``None`` keeps PySCF's default.
        charge_override: if given, use this charge instead of the one derived
            from the SMILES metadata. Useful for testing or for files whose
            metadata is missing/wrong.
        verbose: PySCF verbosity.
        ecp: optional per-element ECP dict.

    Returns:
        A built :class:`pyscf.gto.Mole`.
    """
    atom_str, charge, _smiles = read_xyz(xyz_file, charge_override=charge_override)
    return build_pyscf_mol(
        atom_str=atom_str,
        basis=basis,
        charge=charge,
        spin=spin,
        max_memory=max_memory,
        verbose=verbose,
        ecp=ecp,
    )
