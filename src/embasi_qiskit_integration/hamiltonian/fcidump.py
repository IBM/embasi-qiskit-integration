# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""FCIDUMP + ``.npz`` sidecar read/write for :class:`EmbeddedHamiltonian`.

FCIDUMP is a lossy carrier for our contract: it stores ``h1``/``h2``/``e_core``,
the total electron count and ``MS2``, but not the free-form ``meta`` provenance.
We therefore write a ``.npz`` sidecar with the same stem that carries ``e_core``,
``nelec`` and ``meta`` authoritatively; the FCIDUMP remains the source of truth
for the integrals themselves.

``MS2`` is written **signed** (``n_alpha - n_beta``), so the header alone pins the spin
sector unambiguously.  Note that PySCF's ``write_head`` derives an *unsigned* ``MS2``
when handed an ``(na, nb)`` tuple, so :func:`write` passes the total electron count plus
an explicit signed ``ms`` instead.

A sidecar-less read of a file written *elsewhere* may still carry an unsigned ``MS2``, in
which case a positive value is genuinely ambiguous; :func:`_nelec_from_header` reads it as
alpha-rich and warns, rather than refusing every open-shell file that lacks a sidecar.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def _sidecar_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".npz")


def write(
    ham: EmbeddedHamiltonian, path: str | Path, *, signed_ms2: bool = True, uhf: bool = False
) -> None:
    """Write ``ham`` to a FCIDUMP file at ``path`` plus a ``.npz`` sidecar.

    The integrals go into the FCIDUMP; ``e_core``, ``nelec`` and ``meta`` go
    into ``<stem>.npz`` so metadata survives the round-trip losslessly.

    Args:
        signed_ms2: write ``MS2 = n_alpha - n_beta`` (default), which pins the spin
            sector so the header alone round-trips exactly.  Set ``False`` for the
            unsigned ``|n_alpha - n_beta|`` the wider ecosystem expects -- notably
            ``qiskit_fermions``' ``FCIDump.from_file``, whose Rust header parse
            *panics* (not raises) on a negative value.  The internal temp file in
            :mod:`~embasi_qiskit_integration.circuit_generator.operator` therefore
            writes unsigned; that path takes ``nelec`` from the live ``ham`` and never
            re-reads it from the header, so no spin information is lost.
        uhf: write the integrals as a ``UHF=1`` file (:func:`write_uhf`), so a
            spin-dependent Hamiltonian stays exact even without its sidecar.

    The sidecar records the FCIDUMP's SHA-256, so :func:`read` refuses a sidecar that
    belongs to another file.
    """
    import hashlib

    from pyscf.tools import fcidump as pyscf_fcidump

    path = Path(path)
    norb = ham.norb
    na, nb = ham.nelec
    if ham.is_spin_dependent and not uhf:
        warnings.warn(
            "this Hamiltonian carries a spin-dependent (h1a, h1b) pair, which a "
            "restricted FCIDUMP cannot represent; the FCIDUMP gets the spin-averaged h1 "
            "(the .npz sidecar keeps the spin blocks, so read() round-trips exactly). "
            "Use write_uhf() for a UHF=1 file other programs can read.",
            stacklevel=2,
        )
    # The FCIDUMP appears last and whole (a watcher waits for it), after its sidecar.
    tmp = path.with_name(path.name + ".tmp")
    try:
        if uhf:
            write_uhf(ham, tmp)
        else:
            pyscf_fcidump.from_integrals(
                str(tmp),
                ham.h1,
                ham.h2,
                norb,
                na + nb,
                nuc=ham.e_core,
                ms=(na - nb) if signed_ms2 else abs(na - nb),
            )

        meta_json = json.dumps(ham.meta, default=_json_default)
        arrays: dict[str, Any] = {
            "e_core": np.asarray(ham.e_core, dtype=float),
            "nelec": np.asarray(ham.nelec, dtype=int),
            "meta_json": np.asarray(meta_json),
            "fcidump_sha256": np.asarray(hashlib.sha256(tmp.read_bytes()).hexdigest()),
            **({} if uhf else _spin_sidecar_fields(ham)),
        }
        np.savez(_sidecar_path(path), **arrays)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _check_sidecar(
    path: Path, sidecar: Path, nelec: tuple, header: dict[str, Any], digest: str | None
) -> None:
    """A sidecar left over from another Hamiltonian must not be paired silently: it must
    carry this FCIDUMP's SHA-256 (sidecars from before the hash: its electron count)."""
    import hashlib

    if digest is not None and digest != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(
            f"{sidecar.name} was written for a different {path.name} (content hash "
            "mismatch): the sidecar belongs to another file"
        )
    if sum(nelec) != int(header["NELEC"]):
        raise ValueError(
            f"{sidecar.name} has {sum(nelec)} electrons but {path.name} has "
            f"NELEC={header['NELEC']}: the sidecar belongs to another file"
        )


def _spin_sidecar_fields(ham: EmbeddedHamiltonian) -> dict[str, np.ndarray]:
    """The spin-resolved blocks for the sidecar (empty for a restricted Hamiltonian)."""
    out: dict[str, np.ndarray] = {}
    if ham.h1a is not None and ham.h1b is not None:
        out["h1a"] = np.asarray(ham.h1a, dtype=float)
        out["h1b"] = np.asarray(ham.h1b, dtype=float)
    if ham.h2_spin is not None:
        for name, block in zip(("h2_aa", "h2_ab", "h2_bb"), ham.h2_spin, strict=True):
            out[name] = np.asarray(block, dtype=float)
    return out


def read(
    path: str | Path,
    *,
    nelec: tuple[int, int] | None = None,
    alpha_index: str | None = None,
) -> EmbeddedHamiltonian:
    """Read an :class:`EmbeddedHamiltonian` from a FCIDUMP + ``.npz`` sidecar.

    If the sidecar is absent, ``nelec`` is reconstructed from the FCIDUMP
    ``NELEC``/``MS2`` header and ``e_core`` from ``ECORE``; ``meta`` is empty.  An
    explicit ``nelec = (n_alpha, n_beta)`` replaces the header's spin split (its sum must
    equal ``NELEC``), for a writer whose ``MS2`` convention is unknown.  ``alpha_index``
    is passed to :func:`read_uhf` (interleaved UHF files only).
    """
    path = Path(path)
    header = _read_header(path)
    if _is_uhf(header):
        return read_uhf(path, nelec=nelec, alpha_index=alpha_index)
    # Our own parser rather than PySCF's, which caps the header at 10 lines (a wrapped
    # ORBSYM of a large space) and rejects Fortran D exponents.
    norb = int(header["NORB"])
    h1 = np.zeros((norb, norb))
    h2 = np.zeros((norb,) * 4)
    e_core = float(header.get("ECORE", 0.0))  # a 0 0 0 0 line, if any, overrides it
    rows = _integral_lines(path)
    for v, i, j, k, m in rows:
        if i == j == k == m == 0:
            e_core = v
        elif j == k == m == 0:
            continue  # an orbital energy ("eps i 0 0 0"), not a matrix element
        elif k == m == 0:
            h1[i - 1, j - 1] = h1[j - 1, i - 1] = v
        else:
            _fill_eri(h2, i - 1, j - 1, k - 1, m - 1, v, same_spin=True)
    if not h1.any() and not any(r[1:] == (0, 0, 0, 0) for r in rows):
        raise ValueError(
            f"{path.name}: no one-electron integrals and no core line -- a truncated file?"
        )

    sidecar = _sidecar_path(path)
    spin: dict[str, Any] = {}
    override = _nelec_override(nelec, int(header["NELEC"]), path)
    if sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as npz:
            e_core = float(npz["e_core"])
            nelec = tuple(int(x) for x in npz["nelec"])
            meta = json.loads(str(npz["meta_json"]))
            digest = str(npz["fcidump_sha256"]) if "fcidump_sha256" in npz.files else None
            if "h1a" in npz.files:
                spin["h1a"], spin["h1b"] = np.asarray(npz["h1a"]), np.asarray(npz["h1b"])
            if "h2_aa" in npz.files:
                spin["h2_spin"] = tuple(np.asarray(npz[k]) for k in ("h2_aa", "h2_ab", "h2_bb"))
        _check_sidecar(path, sidecar, nelec, header, digest)
    else:
        nelec = override or _nelec_from_header(
            int(header["NELEC"]), int(header.get("MS2", 0)), path
        )
        meta = {}
    nelec = override or nelec

    nelec_pair: tuple[int, int] = (int(nelec[0]), int(nelec[1]))

    return EmbeddedHamiltonian(h1=h1, h2=h2, e_core=e_core, nelec=nelec_pair, meta=meta, **spin)


#: Energy units a foreign FCIDUMP may be written in, as the factor to Hartree.
#: WEST writes its QDET FCIDUMP in Rydberg (its internal unit, unconverted).
ENERGY_UNIT_TO_HARTREE = {"hartree": 1.0, "rydberg": 0.5}


def to_hartree(ham: EmbeddedHamiltonian, unit: str) -> EmbeddedHamiltonian:
    """``ham`` with every integral and the core energy converted from ``unit`` to Hartree.

    The FCIDUMP format carries no unit, and a file written in Rydberg (WEST/QDET) reads
    as a valid Hamiltonian twice too large; the caller has to say which it is.
    """
    try:
        factor = ENERGY_UNIT_TO_HARTREE[unit.lower()]
    except KeyError:
        raise ValueError(
            f"unknown energy unit {unit!r}; expected one of {sorted(ENERGY_UNIT_TO_HARTREE)}"
        ) from None
    if factor == 1.0:
        return ham
    update: dict[str, Any] = {
        "h1": ham.h1 * factor,
        "h2": ham.h2 * factor,
        "e_core": float(ham.e_core) * factor,
        "meta": {**ham.meta, "energy_unit_converted_from": unit.lower()},
    }
    if ham.h1a is not None and ham.h1b is not None:
        update["h1a"], update["h1b"] = ham.h1a * factor, ham.h1b * factor
    if ham.h2_spin is not None:
        update["h2_spin"] = tuple(block * factor for block in ham.h2_spin)
    return ham.model_copy(update=update)


def _nelec_override(
    nelec: tuple[int, int] | None, nelec_total: int, path: Path
) -> tuple[int, int] | None:
    """A caller's ``(n_alpha, n_beta)``, checked against the header's ``NELEC``."""
    if nelec is None:
        return None
    na, nb = (int(x) for x in nelec)
    if na + nb != nelec_total or min(na, nb) < 0:
        raise ValueError(f"nelec={nelec} does not add up to {path.name}'s NELEC={nelec_total}")
    return (na, nb)


def _nelec_from_header(nelec_total: int, ms2: int, path: Path) -> tuple[int, int]:
    """``(NELEC, MS2)`` -> ``(n_alpha, n_beta)``, taking ``MS2`` as ``n_alpha - n_beta``.

    :func:`write` emits a **signed** ``MS2``, so a file from this module round-trips
    exactly through the header alone, with no sidecar needed.  ``MS2`` is taken at face
    value in both directions.

    The residual ambiguity is external: much of the ecosystem (PySCF's own
    ``write_head`` among them) writes ``MS2`` unsigned, and a positive value from such
    a writer could equally mean the beta-rich sector.  We cannot tell the two apart
    from the header, so a positive ``MS2`` is read as alpha-rich -- the near-universal
    convention -- and the caller is warned, since the alternative (refusing every
    open-shell file lacking a sidecar, including our own) is worse.  Pass the sidecar,
    or construct the Hamiltonian directly, when the producer is known to write
    unsigned and the system is beta-rich.
    """
    if ms2 > 0 and (nelec_total - ms2) % 2 and ms2 - 1 <= nelec_total:
        # CP2K 2023.1-2026.1 write the multiplicity 2S+1 as MS2 (a singlet says MS2=1);
        # as 2S it fails the parity check, as 2S+1 it is consistent.
        warnings.warn(
            f"{path.name}: MS2={ms2} has the wrong parity for NELEC={nelec_total}; reading "
            f"it as the multiplicity 2S+1 (CP2K 2023.1-2026.1 write it so), i.e. 2S={ms2 - 1}."
            " Pass nelec= if that is wrong.",
            stacklevel=3,
        )
        ms2 -= 1
    if ms2 == 0 and nelec_total % 2:
        # CP2K 7.1-2022.x hard-code MS2=0 for every file, odd electron counts included.
        raise ValueError(
            f"{path.name}: MS2=0 with an odd NELEC={nelec_total} gives no spin split; pass "
            "nelec=(n_alpha, n_beta) (import_fcidump --nelec_alpha/--nelec_beta)"
        )
    if ms2 == 0:
        # Closed shell (or an equal-spin open shell): unambiguous.
        return (nelec_total // 2, nelec_total - nelec_total // 2)

    if abs(ms2) > nelec_total or (nelec_total - ms2) % 2 != 0:
        raise ValueError(
            f"{path.name} has an inconsistent header: MS2={ms2} cannot be "
            f"n_alpha - n_beta for NELEC={nelec_total} (needs |MS2| <= NELEC and "
            "NELEC - MS2 even)."
        )

    na = (nelec_total + ms2) // 2
    if ms2 > 0:
        warnings.warn(
            f"{path.name} has MS2={ms2} and no .npz sidecar; reading it as the "
            f"alpha-rich sector ({na}, {nelec_total - na}). This module writes MS2 "
            "signed, so its own files are exact -- but a writer that emits MS2 "
            "unsigned (PySCF's from_integrals with an (na, nb) tuple, among others) "
            f"would produce the same header for ({nelec_total - na}, {na}). Pass the "
            "sidecar or build the EmbeddedHamiltonian directly if that is the case.",
            stacklevel=3,
        )
    return (na, nelec_total - na)


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# --------------------------------------------------------------------------- #
# Unrestricted (UHF=1) FCIDUMP
# --------------------------------------------------------------------------- #
# Two layouts exist in the wild and both are read:
#
# * spin-orbital indexing (CP2K's `&ACTIVE_SPACE &FCIDUMP` for UKS): beta orbitals are
#   numbered norb+1..2*norb, so every line's spin block is fixed by its indices and no
#   separators are needed; the core energy is the single `0 0 0 0` line;
# * separator blocks (Molpro-style): indices restart at 1 in every block and a
#   `0.0 0 0 0 0` line closes each of (aa|aa), (bb|bb), (aa|bb), h_a, h_b, followed by
#   the core energy.
#
# `write_uhf` emits the first (self-describing) form.


def _read_header(path: Path) -> dict[str, Any]:
    """The ``&FCI ... /`` namelist as a dict (``ORBSYM`` dropped; see :func:`_namelist_value`)."""
    import re

    lines = []
    with open(path) as handle:
        for line in handle:
            lines.append(line)
            if _ends_namelist(line):
                break
    body = " ".join(lines)
    body = re.sub(r"&FCI\b", " ", body, flags=re.IGNORECASE)
    body = re.sub(r"(&END|\$END)", " ", body, flags=re.IGNORECASE).replace("/", " ")
    # Split on KEY= markers; each value runs to the next key.
    parts = re.split(r"([A-Za-z_][A-Za-z_0-9]*)\s*=", body)
    header: dict[str, Any] = {}
    for key, value in zip(parts[1::2], parts[2::2], strict=False):
        key = key.upper()
        tokens = value.replace(",", " ").split()
        if key == "ORBSYM" or not tokens:
            continue
        header[key] = _namelist_value(tokens[0])
    if "NORB" not in header or "NELEC" not in header:
        raise ValueError(f"{path.name}: no NORB/NELEC in an &FCI header; is it a FCIDUMP?")
    return header


def _namelist_value(token: str) -> Any:
    """A Fortran namelist scalar: int, float (incl. ``1.0D0``), logical, or the raw text.

    Logicals (``.TRUE.``/``.FALSE.``/``T``/``F``, as NECI/Molpro write ``UHF=``/``TREL=``)
    become 1/0, so a flag reads the same whichever spelling a writer used; anything else
    stays a string rather than aborting the read.
    """
    upper = token.strip().upper()
    if upper in (".TRUE.", "T", ".T.", "TRUE"):
        return 1
    if upper in (".FALSE.", "F", ".F.", "FALSE"):
        return 0
    number = token.replace("D", "E").replace("d", "e")
    try:
        return int(number)
    except ValueError:
        pass
    try:
        return float(number)
    except ValueError:
        return token


def _is_uhf(header: dict[str, Any]) -> bool:
    """Whether the header declares an unrestricted file (``UHF`` or ``IUHF``, any spelling)."""
    for key in ("UHF", "IUHF"):
        value = header.get(key)
        if isinstance(value, (int, float)) and value != 0:
            return True
    return False


def _ends_namelist(line: str) -> bool:
    # Anywhere in the line, as PySCF's reader (" &END," and "... ISYM=1 /" both close it).
    upper = line.upper()
    return "/" in upper or "&END" in upper or "$END" in upper


def _integral_lines(path: Path) -> list[tuple[float, int, int, int, int]]:
    rows: list[tuple[float, int, int, int, int]] = []
    in_header = True
    with open(path) as handle:
        for line in handle:
            if in_header:
                if _ends_namelist(line):
                    in_header = False
                continue
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 5:
                raise ValueError(f"{path.name}: malformed integral line {line.strip()!r}")
            i, j, k, m = (int(x) for x in parts[1:5])
            rows.append((float(parts[0].replace("D", "E").replace("d", "e")), i, j, k, m))
    if in_header:
        raise ValueError(f"{path.name}: the &FCI header is never closed (no '/' or '&END')")
    if not rows:
        raise ValueError(f"{path.name}: no integral lines after the header")
    return rows


def _fill_eri(eri: np.ndarray, p: int, q: int, r: int, s: int, v: float, *, same_spin: bool):
    """Fill every permutation of ``(pq|rs)`` a real Hamiltonian implies."""
    for a, b in ((p, q), (q, p)):
        for c, d in ((r, s), (s, r)):
            eri[a, b, c, d] = v
            if same_spin:
                eri[c, d, a, b] = v


def read_uhf(
    path: str | Path,
    *,
    nelec: tuple[int, int] | None = None,
    alpha_index: str | None = None,
) -> EmbeddedHamiltonian:
    """Read a ``UHF=1`` FCIDUMP into a spin-resolved :class:`EmbeddedHamiltonian`.

    ``h1``/``h2`` are filled with the spin average / the alpha block (the contract's
    restricted view); ``h1a``/``h1b`` and ``h2_spin = (aa, ab, bb)`` carry the exact
    unrestricted Hamiltonian, which :class:`~embasi_qiskit_integration.solvers.FCISolver`
    consumes through ``direct_uhf``.  A ``.npz`` sidecar, when present, still pins
    ``e_core``/``nelec``/``meta``.
    """
    path = Path(path)
    header = _read_header(path)
    norb = int(header["NORB"])
    rows = _integral_lines(path)
    spin_orbital = any(max(i, j, k, m) > norb for _v, i, j, k, m in rows)
    # NECI/HANDE/Psi4 `UHF=T`: NORB counts spin orbitals, alpha/beta interleaved (2p-1, 2p),
    # and only the core line has all-zero indices; the separated layout has >= 5 of them.
    interleaved = not spin_orbital and sum(1 for r in rows if not any(r[1:])) <= 1
    if interleaved:
        return _read_uhf_interleaved(path, header, rows, nelec, alpha_index)
    eri = {k: np.zeros((norb,) * 4) for k in ("aa", "ab", "bb")}
    h1 = {k: np.zeros((norb, norb)) for k in ("a", "b")}
    e_core = float(header.get("ECORE", 0.0))  # a 0 0 0 0 line, if any, overrides it

    if spin_orbital:
        for v, i, j, k, m in rows:
            if i == j == k == m == 0:
                e_core = v
                continue
            if j == k == m == 0:
                continue  # an orbital energy ("eps i 0 0 0"): informational, not a matrix element
            if k == 0 and m == 0:  # one-electron
                beta = i > norb
                if (j > norb) != beta:
                    raise ValueError(f"{path.name}: one-electron line mixes spins: {i} {j}")
                p, q = (i - norb - 1, j - norb - 1) if beta else (i - 1, j - 1)
                block = h1["b" if beta else "a"]
                block[p, q] = block[q, p] = v
                continue
            bra_b, ket_b = i > norb, k > norb
            if (j > norb) != bra_b or (m > norb) != ket_b:
                raise ValueError(f"{path.name}: a pair mixes spins: {i} {j} {k} {m}")
            p, q = (i - norb - 1, j - norb - 1) if bra_b else (i - 1, j - 1)
            r, s = (k - norb - 1, m - norb - 1) if ket_b else (k - 1, m - 1)
            if bra_b and ket_b:
                _fill_eri(eri["bb"], p, q, r, s, v, same_spin=True)
            elif not bra_b and not ket_b:
                _fill_eri(eri["aa"], p, q, r, s, v, same_spin=True)
            elif not bra_b and ket_b:
                _fill_eri(eri["ab"], p, q, r, s, v, same_spin=False)
            else:  # (bb|aa): transpose into (aa|bb)
                _fill_eri(eri["ab"], r, s, p, q, v, same_spin=False)
    else:
        order = ("aa", "bb", "ab", "a", "b", "core")
        section = 0
        for v, i, j, k, m in rows:
            if i == j == k == m == 0:
                if order[section] == "core":
                    e_core = v
                    break
                section += 1
                if order[section] == "core" and v != 0.0:
                    # Some writers omit the empty separator before the core line.
                    e_core = v
                    break
                continue
            if j == k == m == 0:
                continue  # an orbital energy ("eps i 0 0 0"), not part of any block
            kind = order[section]
            if kind in ("a", "b"):
                h1[kind][i - 1, j - 1] = h1[kind][j - 1, i - 1] = v
            else:
                _fill_eri(eri[kind], i - 1, j - 1, k - 1, m - 1, v, same_spin=kind != "ab")

    layout = "spin-orbital" if spin_orbital else "separated"
    return _uhf_hamiltonian(path, header, nelec, e_core, h1, eri, layout)


def _read_uhf_interleaved(path, header, rows, nelec, alpha_index) -> EmbeddedHamiltonian:
    """The NECI/HANDE/Psi4 ``UHF=T`` layout: ``NORB`` spin orbitals, ``2p-1``/``2p`` for
    spatial orbital ``p``.  Which of the two is alpha differs by writer -- Psi4 puts alpha
    on the odd index, NECI (``is_beta = btest(orb, 0)``) on the even one -- and the file
    does not say, so ``alpha_index`` (``"odd"``/``"even"``) must be given unless the
    alpha and beta blocks coincide, when the choice cannot change the Hamiltonian."""
    if alpha_index not in (None, "odd", "even"):
        raise ValueError(f"alpha_index must be 'odd' or 'even', got {alpha_index!r}")
    nso = int(header["NORB"])
    if nso % 2:
        raise ValueError(f"{path.name}: interleaved UHF layout needs an even NORB, got {nso}")
    n = nso // 2
    eri = {k: np.zeros((n,) * 4) for k in ("aa", "ab", "bb")}
    h1 = {k: np.zeros((n, n)) for k in ("a", "b")}
    e_core = float(header.get("ECORE", 0.0))
    for v, i, j, k, m in rows:
        if i == j == k == m == 0:
            e_core = v
            continue
        if j == k == m == 0:
            continue  # an orbital energy ("eps i 0 0 0")
        (p, sp), (q, sq) = divmod(i - 1, 2), divmod(j - 1, 2)
        if k == 0 and m == 0:
            if sp != sq:
                if v:
                    raise ValueError(f"{path.name}: one-electron line mixes spins: {i} {j}")
                continue
            block = h1["b" if sp else "a"]
            block[p, q] = block[q, p] = v
            continue
        (r, sr), (s, ss) = divmod(k - 1, 2), divmod(m - 1, 2)
        if sp != sq or sr != ss:
            if v:
                raise ValueError(f"{path.name}: a pair mixes spins: {i} {j} {k} {m}")
            continue
        if sp == sr:
            _fill_eri(eri["bb" if sp else "aa"], p, q, r, s, v, same_spin=True)
        elif sp == 0:
            _fill_eri(eri["ab"], p, q, r, s, v, same_spin=False)
        else:  # (bb|aa): transpose into (aa|bb)
            _fill_eri(eri["ab"], r, s, p, q, v, same_spin=False)
    # Read as odd = alpha; the other convention swaps the channels.
    swapped = (eri["ab"].transpose(2, 3, 0, 1), h1["b"], h1["a"], eri["bb"], eri["aa"])
    if alpha_index is None:
        diff = max(
            float(np.abs(h1["a"] - h1["b"]).max()),
            float(np.abs(eri["aa"] - eri["bb"]).max()),
            float(np.abs(eri["ab"] - swapped[0]).max()),
        )
        if diff > 1e-8:
            raise ValueError(
                f"{path.name}: interleaved UHF file with distinct alpha/beta blocks "
                f"(max difference {diff:.1e}); its writer decides which index is alpha -- "
                "pass alpha_index='odd' (Psi4) or 'even' (NECI/HANDE) "
                "(import_fcidump --uhf_alpha_index)"
            )
    elif alpha_index == "even":
        eri["ab"], h1["a"], h1["b"], eri["aa"], eri["bb"] = swapped
    return _uhf_hamiltonian(path, header, nelec, e_core, h1, eri, "interleaved")


def _uhf_hamiltonian(path, header, nelec, e_core, h1, eri, layout) -> EmbeddedHamiltonian:
    """Assemble a read UHF file, with the sidecar / ``nelec`` precedence of :func:`read`."""
    sidecar = _sidecar_path(path)
    override = _nelec_override(nelec, int(header["NELEC"]), path)
    if sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as npz:
            e_core = float(npz["e_core"])
            nelec = tuple(int(x) for x in npz["nelec"])
            meta = json.loads(str(npz["meta_json"]))
            digest = str(npz["fcidump_sha256"]) if "fcidump_sha256" in npz.files else None
        _check_sidecar(path, sidecar, nelec, header, digest)
    else:
        nelec = override or _nelec_from_header(
            int(header["NELEC"]), int(header.get("MS2", 0)), path
        )
        meta = {}
    nelec = override or nelec
    meta = {**meta, "fcidump_uhf_layout": layout}
    return EmbeddedHamiltonian(
        h1=0.5 * (h1["a"] + h1["b"]),
        h2=eri["aa"],
        e_core=e_core,
        nelec=(int(nelec[0]), int(nelec[1])),
        h1a=h1["a"],
        h1b=h1["b"],
        h2_spin=(eri["aa"], eri["ab"], eri["bb"]),
        meta=meta,
    )


def write_uhf(ham: EmbeddedHamiltonian, path: str | Path, *, tol: float = 1e-14) -> None:
    """Write a ``UHF=1`` FCIDUMP in the spin-orbital layout (see :func:`read_uhf`).

    A restricted Hamiltonian is written with identical alpha and beta blocks.  ``MS2`` is
    signed (``n_alpha - n_beta``), as in :func:`write`.
    """
    path = Path(path)
    n = ham.norb
    na, nb = ham.nelec
    h1a = np.asarray(ham.h1a if ham.h1a is not None else ham.h1)
    h1b = np.asarray(ham.h1b if ham.h1b is not None else ham.h1)
    aa, ab, bb = ham.h2_spin if ham.h2_spin is not None else (ham.h2, ham.h2, ham.h2)
    lines = [
        f"&FCI NORB={n},NELEC={na + nb},MS2={na - nb},",
        "  ORBSYM=" + "1," * n,
        "  ISYM=1,",
        "  UHF=1,",
        " /",
    ]

    def eri_lines(eri, bra_off, ket_off, same_spin):
        for i in range(n):
            for j in range(i + 1):
                for k in range(n):
                    for m in range(k + 1):
                        if same_spin and (i * (i + 1) // 2 + j) < (k * (k + 1) // 2 + m):
                            continue
                        v = float(eri[i, j, k, m])
                        if abs(v) > tol:
                            lines.append(
                                f"{v: .16E} {i + 1 + bra_off:4d} {j + 1 + bra_off:4d} "
                                f"{k + 1 + ket_off:4d} {m + 1 + ket_off:4d}"
                            )

    eri_lines(np.asarray(aa), 0, 0, True)
    eri_lines(np.asarray(ab), 0, n, False)
    eri_lines(np.asarray(bb), n, n, True)
    for h, off in ((h1a, 0), (h1b, n)):
        for i in range(n):
            for j in range(i + 1):
                if abs(h[i, j]) > tol:
                    lines.append(f"{h[i, j]: .16E} {i + 1 + off:4d} {j + 1 + off:4d}    0    0")
    lines.append(f"{ham.e_core: .16E}    0    0    0    0")
    path.write_text("\n".join(lines) + "\n")
