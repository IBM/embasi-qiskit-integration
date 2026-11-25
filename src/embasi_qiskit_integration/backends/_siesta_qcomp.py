# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SIESTA-QCOMP NetCDF ERIs for the WF-in-DFT path.

SIESTA's ASI interface exports S/H/DM but no two-electron integrals, so the
projection-embedding downfold cannot be built.  SIESTA-QCOMP can write the AO
ERIs to a NetCDF file (``QComp.Workflow DROP`` + ``QComp.Protocol FILE``): the
``eri_indx``/``eri_vals`` pair, 1-based AO indices and values in Hartree, stored
for the symmetry-unique quartets.  This provider expands them to the full
chemists'-notation tensor and supplies the pieces
:class:`~embasi_qiskit_integration.backends.callback.CallbackIntegrals` needs
(``eri_mo``, ``eri_mo_mixed``, ``jk``).

The QCOMP build must include ``SIESTA_WITH_QCOMP=ON`` and hybrid-XC/libint
support.  The fdf options that produce the file live in
:func:`embasi_qiskit_integration.backends.siesta.siesta_calculator` when
``qcomp_eris=True``.

This module is SIESTA-specific but not an EmbASI adapter shim; it is the
backend's ERI provider.  It can be replaced by a richer sparse implementation
later without touching the adapter.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

#: File name written by ``QComp.DataAddress`` in every ASI work directory.
QCOMP_ERI_NAME = "qcomp_eris.nc"

#: Above this many float64 elements, refuse the dense expansion (512 MB).
MAX_DENSE_ELEMENTS = 64_000_000


def _quartet_permutations() -> tuple[tuple[int, int, int, int], ...]:
    """The eight permutations of a chemists'-notation quartet ``(pq|rs)``."""
    out: set[tuple[int, int, int, int]] = set()
    for swap_01 in (False, True):
        for swap_23 in (False, True):
            for swap_pairs in (False, True):
                quartet = [0, 1, 2, 3]
                if swap_01:
                    quartet[0], quartet[1] = quartet[1], quartet[0]
                if swap_23:
                    quartet[2], quartet[3] = quartet[3], quartet[2]
                if swap_pairs:
                    quartet = [quartet[2], quartet[3], quartet[0], quartet[1]]
                out.add(tuple(quartet))
    return tuple(sorted(out))


_PERMUTATIONS = _quartet_permutations()


class SiestaEriProvider:
    """AO ERIs read from a SIESTA-QCOMP NetCDF file.

    Args:
        path: the file written by ``QComp.DataAddress``.
        n_ao: expected number of AO basis functions; checked against the file.
    """

    def __init__(self, path: str | Path, *, n_ao: int | None = None):
        self.path = Path(path)
        self.n_ao = None if n_ao is None else int(n_ao)
        self._indices: np.ndarray | None = None
        self._values: np.ndarray | None = None
        self._signature: tuple[int, int] | None = None
        self._dense: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    _DUMP = r"""
import sys
import numpy as np
import h5py
src, dst = sys.argv[1], sys.argv[2]
with h5py.File(src, "r") as handle:
    indices = np.asarray(handle["eri_indx"][...])
    values = np.asarray(handle["eri_vals"][...])
    unit = handle["eri_vals"].attrs.get("unit", "Hartree")
np.savez(dst, eri_indx=indices, eri_vals=values, unit=np.asarray(unit))
"""

    @staticmethod
    def _read_h5py(path: Path) -> tuple[np.ndarray, np.ndarray, Any]:
        import h5py

        with h5py.File(path, "r") as handle:
            indices = np.asarray(handle["eri_indx"][...], dtype=np.int64)
            values = np.asarray(handle["eri_vals"][...], dtype=float).reshape(-1)
            unit = handle["eri_vals"].attrs.get("unit", b"Hartree")
        return indices, values, unit

    @classmethod
    def _read_subprocess(cls, path: Path) -> tuple[np.ndarray, np.ndarray, Any]:
        """Read in a clean Python process when h5py clashes with SIESTA's HDF5.

        ``libsiesta`` links NetCDF/HDF5; a wheel h5py built against another HDF5
        then fails at import (``Not a datatype``).  The helper process has no
        libsiesta loaded, so its bundled HDF5 can read the file normally.
        """
        with tempfile.TemporaryDirectory(prefix="siesta-eris.") as tmp:
            out = Path(tmp) / "eris.npz"
            process = subprocess.run(
                [sys.executable, "-c", cls._DUMP, str(path), str(out)],
                capture_output=True,
                text=True,
            )
            if process.returncode != 0:
                raise RuntimeError(process.stderr.strip() or "QCOMP ERI helper failed")
            with np.load(out, allow_pickle=False) as data:
                unit = data["unit"].item()
                if isinstance(unit, bytes):
                    unit = unit.decode()
                return (
                    np.asarray(data["eri_indx"], dtype=np.int64),
                    np.asarray(data["eri_vals"], dtype=float).reshape(-1),
                    unit,
                )

    def _load(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"no SIESTA-QCOMP ERI file at {self.path}; build SIESTA with "
                "SIESTA_WITH_QCOMP=ON (hybrid-XC/libint) and set "
                "QComp.Workflow DROP / QComp.Protocol FILE"
            ) from exc
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._signature:
            return

        try:
            indices, values, unit = self._read_h5py(self.path)
        except Exception:  # noqa: BLE001 - HDF5 ABI clash or a missing h5py
            try:
                indices, values, unit = self._read_subprocess(self.path)
            except Exception as exc:  # pragma: no cover - dependency hint
                raise RuntimeError(
                    "could not read the SIESTA-QCOMP ERI file; install h5py "
                    "(the `siesta` extra) and check the file is intact"
                ) from exc
        if isinstance(unit, bytes):
            unit = unit.decode()
        if str(unit).strip().lower() not in {"hartree", "ha"}:
            raise ValueError(f"{self.path}: eri_vals is in {unit!r}, expected Hartree")

        # NetCDF4/HDF5 presents Fortran (4, N) as C (N, 4); tolerate both.
        if indices.ndim != 2:
            raise ValueError(f"{self.path}: eri_indx must be 2-D, got {indices.shape}")
        if indices.shape[1] != 4:
            if indices.shape[0] == 4:
                indices = indices.T
            else:
                raise ValueError(
                    f"{self.path}: eri_indx must have a quartet axis, got {indices.shape}"
                )
        if indices.shape[0] != values.size:
            raise ValueError(
                f"{self.path}: eri_indx has {indices.shape[0]} quartets but "
                f"eri_vals has {values.size}"
            )
        if values.size == 0:
            raise ValueError(
                f"{self.path} holds no ERIs; set HFX.TruncateDM False and a small "
                "HFX.StoreERIsTolerance in the SIESTA input"
            )

        # Fortran writes 1-based AO indices.
        indices = indices - 1
        if indices.min() < 0:
            raise ValueError(f"{self.path}: eri_indx contains a non-positive AO index")
        n_ao = int(indices.max()) + 1
        if self.n_ao is None:
            self.n_ao = n_ao
        elif n_ao > self.n_ao:
            raise ValueError(
                f"{self.path}: eri_indx references AO {n_ao} but the basis has "
                f"{self.n_ao} functions"
            )
        self._indices = indices
        self._values = values
        self._signature = signature
        self._dense = None

    @property
    def n_eri(self) -> int:
        self._load()
        assert self._values is not None
        return int(self._values.size)

    # ------------------------------------------------------------------ #
    # Full tensor
    # ------------------------------------------------------------------ #
    def dense(self) -> np.ndarray:
        """Chemists'-notation ``(pq|rs)`` tensor, symmetrised by the 8 permutations."""
        self._load()
        if self._dense is not None:
            return self._dense
        assert self._indices is not None and self._values is not None and self.n_ao
        n = self.n_ao
        if n**4 > MAX_DENSE_ELEMENTS:
            raise RuntimeError(
                f"the {n}-AO dense ERI tensor is too large for this provider "
                f"({n**4} elements > {MAX_DENSE_ELEMENTS}); use a smaller active space "
                "or a sparse ERI implementation"
            )
        tensor = np.zeros((n, n, n, n), dtype=float)
        for (p, q, r, s), value in zip(self._indices, self._values, strict=True):
            quartet = (int(p), int(q), int(r), int(s))
            for permutation in _PERMUTATIONS:
                a, b, c, d = (quartet[i] for i in permutation)
                tensor[a, b, c, d] = value
        self._dense = tensor
        return tensor

    def eri_ao(self) -> np.ndarray:
        return self.dense().copy()

    # ------------------------------------------------------------------ #
    # AO contractions
    # ------------------------------------------------------------------ #
    def jk(self, dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(J[dm], K[dm])`` in the AO basis from the stored quartets.

        ``J[p,q] = sum_{rs} (pq|rs) dm[r,s]`` and
        ``K[p,q] = sum_{rs} (pr|qs) dm[r,s]``.  The loop is over the
        symmetry-unique quartets, so it does not allocate the dense tensor.
        """
        dm = np.asarray(dm, dtype=float)
        self._load()
        assert self._indices is not None and self._values is not None and self.n_ao
        if dm.shape != (self.n_ao, self.n_ao):
            raise ValueError(f"density is {dm.shape}, expected {(self.n_ao, self.n_ao)}")
        j = np.zeros_like(dm)
        k = np.zeros_like(dm)
        for (p, q, r, s), value in zip(self._indices, self._values, strict=True):
            quartet = (int(p), int(q), int(r), int(s))
            # Repeated AO indices make several symmetry permutations the same
            # tensor element; the dense expansion assigns it once, so the
            # contraction must count each distinct element once as well.
            seen: set[tuple[int, int, int, int]] = set()
            for permutation in _PERMUTATIONS:
                a, b, c, d = (quartet[i] for i in permutation)
                if (a, b, c, d) in seen:
                    continue
                seen.add((a, b, c, d))
                j[a, b] += value * dm[c, d]
                k[a, c] += value * dm[b, d]
        return j, k

    def eri_mo(self, mo_coeff: np.ndarray) -> np.ndarray:
        coeff = np.asarray(mo_coeff, dtype=float)
        if coeff.ndim != 2 or coeff.shape[0] != self.n_ao:
            raise ValueError(f"mo_coeff is {coeff.shape}, expected ({self.n_ao}, n_mo)")
        return np.einsum(
            "pqrs,pi,qj,rk,sl->ijkl", self.dense(), coeff, coeff, coeff, coeff, optimize=True
        )

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        coeff_a = np.asarray(mo_a, dtype=float)
        coeff_b = np.asarray(mo_b, dtype=float)
        if coeff_a.shape[0] != self.n_ao or coeff_b.shape[0] != self.n_ao:
            raise ValueError(
                f"mo_a/mo_b must have {self.n_ao} rows; got {coeff_a.shape}/{coeff_b.shape}"
            )
        return np.einsum(
            "pqrs,pi,qj,rk,sl->ijkl",
            self.dense(),
            coeff_a,
            coeff_a,
            coeff_b,
            coeff_b,
            optimize=True,
        )
