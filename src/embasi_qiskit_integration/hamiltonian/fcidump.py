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


def write(ham: EmbeddedHamiltonian, path: str | Path, *, signed_ms2: bool = True) -> None:
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
    """
    from pyscf.tools import fcidump as pyscf_fcidump

    path = Path(path)
    norb = ham.norb
    na, nb = ham.nelec
    if ham.is_spin_dependent:
        warnings.warn(
            "this Hamiltonian carries a spin-dependent (h1a, h1b) pair, which a "
            "restricted FCIDUMP cannot represent; the FCIDUMP gets the spin-averaged h1 "
            "(the .npz sidecar keeps the spin blocks, so read() round-trips exactly). "
            "Use write_uhf() for a UHF=1 file other programs can read.",
            stacklevel=2,
        )
    pyscf_fcidump.from_integrals(
        str(path),
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
        **_spin_sidecar_fields(ham),
    }
    np.savez(_sidecar_path(path), **arrays)


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


def read(path: str | Path) -> EmbeddedHamiltonian:
    """Read an :class:`EmbeddedHamiltonian` from a FCIDUMP + ``.npz`` sidecar.

    If the sidecar is absent, ``nelec`` is reconstructed from the FCIDUMP
    ``NELEC``/``MS2`` header and ``e_core`` from ``ECORE``; ``meta`` is empty.
    """
    from pyscf import ao2mo
    from pyscf.tools import fcidump as pyscf_fcidump

    path = Path(path)
    header = _read_header(path)
    if int(header.get("UHF", 0)) == 1:
        return read_uhf(path)
    data = pyscf_fcidump.read(str(path))
    norb = int(data["NORB"])
    h1 = np.asarray(data["H1"], dtype=float)
    # pyscf packs H2 in 4-fold/8-fold triangular form; restore the full tensor.
    h2 = ao2mo.restore(1, np.asarray(data["H2"], dtype=float), norb)

    sidecar = _sidecar_path(path)
    spin: dict[str, Any] = {}
    if sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as npz:
            e_core = float(npz["e_core"])
            nelec = tuple(int(x) for x in npz["nelec"])
            meta = json.loads(str(npz["meta_json"]))
            if "h1a" in npz.files:
                spin["h1a"], spin["h1b"] = np.asarray(npz["h1a"]), np.asarray(npz["h1b"])
            if "h2_aa" in npz.files:
                spin["h2_spin"] = tuple(np.asarray(npz[k]) for k in ("h2_aa", "h2_ab", "h2_bb"))
    else:
        e_core = float(data.get("ECORE", 0.0))
        nelec = _nelec_from_header(int(data["NELEC"]), int(data.get("MS2", 0)), path)
        meta = {}

    nelec_pair: tuple[int, int] = (int(nelec[0]), int(nelec[1]))

    return EmbeddedHamiltonian(h1=h1, h2=h2, e_core=e_core, nelec=nelec_pair, meta=meta, **spin)


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
    """The ``&FCI ... /`` namelist as a dict (``ORBSYM`` dropped; values as int/float)."""
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
        token = tokens[0].replace("D", "E").replace("d", "e")
        try:
            header[key] = int(token)
        except ValueError:
            header[key] = float(token)
    return header


def _ends_namelist(line: str) -> bool:
    stripped = line.strip().upper()
    return stripped.endswith(("/", "&END", "$END")) or stripped in ("/", "&END", "$END")


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
            if len(parts) < 5:
                continue
            i, j, k, m = (int(x) for x in parts[1:5])
            rows.append((float(parts[0].replace("D", "E").replace("d", "e")), i, j, k, m))
    return rows


def _fill_eri(eri: np.ndarray, p: int, q: int, r: int, s: int, v: float, *, same_spin: bool):
    """Fill every permutation of ``(pq|rs)`` a real Hamiltonian implies."""
    for a, b in ((p, q), (q, p)):
        for c, d in ((r, s), (s, r)):
            eri[a, b, c, d] = v
            if same_spin:
                eri[c, d, a, b] = v


def read_uhf(path: str | Path) -> EmbeddedHamiltonian:
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
    eri = {k: np.zeros((norb,) * 4) for k in ("aa", "ab", "bb")}
    h1 = {k: np.zeros((norb, norb)) for k in ("a", "b")}
    e_core = 0.0

    spin_orbital = any(max(i, j, k, m) > norb for _v, i, j, k, m in rows)
    if spin_orbital:
        for v, i, j, k, m in rows:
            if i == j == k == m == 0:
                e_core = v
                continue
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
            kind = order[section]
            if kind in ("a", "b"):
                h1[kind][i - 1, j - 1] = h1[kind][j - 1, i - 1] = v
            else:
                _fill_eri(eri[kind], i - 1, j - 1, k - 1, m - 1, v, same_spin=kind != "ab")

    sidecar = _sidecar_path(path)
    if sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as npz:
            e_core = float(npz["e_core"])
            nelec = tuple(int(x) for x in npz["nelec"])
            meta = json.loads(str(npz["meta_json"]))
    else:
        nelec = _nelec_from_header(int(header["NELEC"]), int(header.get("MS2", 0)), path)
        meta = {}
    meta = {**meta, "fcidump_uhf_layout": "spin-orbital" if spin_orbital else "separated"}
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
