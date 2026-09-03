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
    np.savez(
        _sidecar_path(path),
        e_core=np.asarray(ham.e_core, dtype=float),
        nelec=np.asarray(ham.nelec, dtype=int),
        meta_json=np.asarray(meta_json),
    )


def read(path: str | Path) -> EmbeddedHamiltonian:
    """Read an :class:`EmbeddedHamiltonian` from a FCIDUMP + ``.npz`` sidecar.

    If the sidecar is absent, ``nelec`` is reconstructed from the FCIDUMP
    ``NELEC``/``MS2`` header and ``e_core`` from ``ECORE``; ``meta`` is empty.
    """
    from pyscf import ao2mo
    from pyscf.tools import fcidump as pyscf_fcidump

    path = Path(path)
    data = pyscf_fcidump.read(str(path))
    norb = int(data["NORB"])
    h1 = np.asarray(data["H1"], dtype=float)
    # pyscf packs H2 in 4-fold/8-fold triangular form; restore the full tensor.
    h2 = ao2mo.restore(1, np.asarray(data["H2"], dtype=float), norb)

    sidecar = _sidecar_path(path)
    if sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as npz:
            e_core = float(npz["e_core"])
            nelec = tuple(int(x) for x in npz["nelec"])
            meta = json.loads(str(npz["meta_json"]))
    else:
        e_core = float(data.get("ECORE", 0.0))
        nelec = _nelec_from_header(int(data["NELEC"]), int(data.get("MS2", 0)), path)
        meta = {}

    nelec_pair: tuple[int, int] = (int(nelec[0]), int(nelec[1]))

    return EmbeddedHamiltonian(h1=h1, h2=h2, e_core=e_core, nelec=nelec_pair, meta=meta)


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
