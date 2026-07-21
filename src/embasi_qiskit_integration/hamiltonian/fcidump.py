# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""FCIDUMP + ``.npz`` sidecar read/write for :class:`EmbeddedHamiltonian`.

FCIDUMP is a lossy carrier for our contract: it stores ``h1``/``h2``/``e_core``
and the *total* electron count, but not the ``(n_alpha, n_beta)`` split
unambiguously in general, nor the free-form ``meta`` provenance. We therefore
write a ``.npz`` sidecar with the same stem that carries ``e_core``, ``nelec``
and ``meta`` authoritatively; the FCIDUMP remains the source of truth for the
integrals themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def _sidecar_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".npz")


def write(ham: EmbeddedHamiltonian, path: str | Path) -> None:
    """Write ``ham`` to a FCIDUMP file at ``path`` plus a ``.npz`` sidecar.

    The integrals go into the FCIDUMP; ``e_core``, ``nelec`` and ``meta`` go
    into ``<stem>.npz`` so metadata survives the round-trip losslessly.
    """
    from pyscf.tools import fcidump as pyscf_fcidump

    path = Path(path)
    norb = ham.norb
    na, nb = ham.nelec
    # ms = n_alpha - n_beta (2*S_z); pyscf writes this as the MS2 header field.
    ms = na - nb
    pyscf_fcidump.from_integrals(
        str(path),
        ham.h1,
        ham.h2,
        norb,
        (na, nb),
        nuc=ham.e_core,
        ms=ms,
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
        nelec = _nelec_from_header(int(data["NELEC"]), int(data.get("MS2", 0)))
        meta = {}

    nelec_pair: tuple[int, int] = (int(nelec[0]), int(nelec[1]))

    return EmbeddedHamiltonian(h1=h1, h2=h2, e_core=e_core, nelec=nelec_pair, meta=meta)


def _nelec_from_header(nelec_total: int, ms2: int) -> tuple[int, int]:
    """(NELEC, MS2=n_a-n_b) -> (n_alpha, n_beta)."""
    na = (nelec_total + ms2) // 2
    nb = nelec_total - na
    return (na, nb)


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
