# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""FCIDUMP + ``.npz`` sidecar read/write for :class:`EmbeddedHamiltonian`.

FCIDUMP is a lossy carrier for our contract: it stores ``h1``/``h2``/``e_core``,
the total electron count and ``MS2``, but not the free-form ``meta`` provenance.
We therefore write a ``.npz`` sidecar with the same stem that carries ``e_core``,
``nelec`` and ``meta`` authoritatively; the FCIDUMP remains the source of truth
for the integrals themselves.

``MS2`` is written unsigned, as the ecosystem expects, so it pins ``|n_alpha -
n_beta|`` but not which spin is in excess. A sidecar-less read therefore recovers
the split exactly for a closed shell and *refuses to guess* for an open one,
rather than silently returning the alpha-rich reading (see
:func:`_nelec_from_header`).
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
    pyscf_fcidump.from_integrals(
        str(path),
        ham.h1,
        ham.h2,
        norb,
        (na, nb),
        nuc=ham.e_core,
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
    """``(NELEC, MS2)`` -> ``(n_alpha, n_beta)``, taking MS2 as ``n_alpha - n_beta``.

    ``MS2`` is conventionally written unsigned (see :func:`write`), so a non-zero
    value is genuinely ambiguous: ``MS2=1`` with ``NELEC=3`` fits both ``(2, 1)``
    and ``(1, 2)``. Rather than silently assume the alpha-rich reading -- which
    would return the wrong spin sector for half of all open-shell inputs, with a
    plausible-looking energy and nothing to flag it -- this raises and points at the
    sidecar that records the split authoritatively.

    A negative ``MS2`` is accepted (some writers do emit one) and taken at face
    value, since it is then unambiguous.
    """
    if ms2 == 0:
        # Closed shell (or an equal-spin open shell): unambiguous.
        return (nelec_total // 2, nelec_total - nelec_total // 2)

    if ms2 < 0:
        na = (nelec_total + ms2) // 2
        return (na, nelec_total - na)

    alpha_rich = (nelec_total + ms2) // 2
    raise ValueError(
        f"{path.name} has MS2={ms2} (NELEC={nelec_total}) and no .npz sidecar, so the "
        f"(n_alpha, n_beta) split is ambiguous: both ({alpha_rich}, "
        f"{nelec_total - alpha_rich}) and ({nelec_total - alpha_rich}, {alpha_rich}) "
        "match this header, because MS2 is written unsigned. Provide the sidecar "
        f"({_sidecar_path(path).name}, written by this module's `write`), or construct "
        "the EmbeddedHamiltonian directly with the intended nelec."
    )


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
