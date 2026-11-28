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

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

#: File name written by ``QComp.DataAddress`` in every ASI work directory.
QCOMP_ERI_NAME = "qcomp_eris.nc"

#: Above this many float64 elements, refuse the dense expansion (1.2 GB).
#: A 12-atom DZP molecule has ~100 AOs (10^8 elements), so the old 512 MB cap
#: rejected systems that fit comfortably; the transform below keeps only one
#: intermediate of the active-space change of basis, not a second full tensor.
MAX_DENSE_ELEMENTS = 150_000_000


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
        self._weights: np.ndarray | None = None

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
        # QCOMP normally stores one orientation per symmetry orbit, but real files
        # can contain the same orbit under several of the eight permutations (the
        # SZ butyronitrile file has ~7.5% such rows).  `dense()` assigns each orbit
        # once (last write wins); the sparse contractions sum every stored row, so
        # leaving duplicates would over-count them.  Canonicalise by the minimum
        # permutation code and keep the last occurrence, matching dense semantics.
        index_dtype = indices.dtype
        key = np.full(indices.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
        for permutation in _PERMUTATIONS:
            p, q, r, s = (indices[:, i] for i in permutation)
            code = (((p * n_ao + q) * n_ao + r) * n_ao + s).astype(np.int64)
            np.minimum(key, code, out=key)
        order = np.argsort(key, kind="stable")
        sorted_keys = key[order]
        keep_last = np.empty(key.size, dtype=bool)
        keep_last[-1] = True
        keep_last[:-1] = sorted_keys[1:] != sorted_keys[:-1]
        keep = np.sort(order[keep_last])
        indices = indices[keep].astype(index_dtype, copy=False)
        values = values[keep]
        self._indices = indices
        self._values = values
        self._signature = signature
        self._dense = None
        self._weights = None

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
        # Assign each permutation as one fancy-index operation; the explicit
        # quartet loop cost ~10^8 Python iterations on a 100-AO DZP molecule.
        for permutation in _PERMUTATIONS:
            p, q, r, s = (self._indices[:, i] for i in permutation)
            tensor[p, q, r, s] = self._values
        self._dense = tensor
        return tensor

    def eri_ao(self) -> np.ndarray:
        return self.dense().copy()

    # ------------------------------------------------------------------ #
    # AO contractions
    # ------------------------------------------------------------------ #
    @staticmethod
    def _permutation_weights(indices: np.ndarray) -> np.ndarray:
        """Weight each of the eight permutations for repeated AO indices.

        ``_PERMUTATIONS`` swaps the first pair, the second pair, and the two pairs.
        Equality can make any of those swaps a no-op; the resulting image
        multiplicity is computed from all four possible pair coincidences (plus
        both first/second swaps together), with the all-equal quartet special-cased
        to eight.  Scaling by ``1/multiplicity`` makes the vectorised sum equal the
        dense expansion, which writes each distinct tensor element once.
        """
        a, b, c, d = indices.T
        all_equal = (a == b) & (b == c) & (c == d)
        frequency = (
            1
            + (a == b).astype(np.int8)
            + (c == d).astype(np.int8)
            + ((a == c) & (b == d)).astype(np.int8)
            + ((a == d) & (b == c)).astype(np.int8)
            + ((a == b) & (c == d)).astype(np.int8)
        )
        frequency = np.where(all_equal, 8, frequency)
        return 1.0 / frequency

    def _quartet_weights(self) -> np.ndarray:
        self._load()
        assert self._indices is not None
        if getattr(self, "_weights", None) is None:
            self._weights = self._permutation_weights(self._indices)
        return self._weights

    def jk(self, dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(J[dm], K[dm])`` in the AO basis from the stored quartets.

        ``J[p,q] = sum_{rs} (pq|rs) dm[r,s]`` and
        ``K[p,q] = sum_{rs} (pr|qs) dm[r,s]``.  This is the same eight-permutation
        expansion as :meth:`dense`, evaluated with ``bincount`` in chunks so a
        100-AO DZP file never needs the full ``n_ao^4`` tensor.
        """
        dm = np.asarray(dm, dtype=float)
        self._load()
        assert self._indices is not None and self._values is not None and self.n_ao
        if dm.shape != (self.n_ao, self.n_ao):
            raise ValueError(f"density is {dm.shape}, expected {(self.n_ao, self.n_ao)}")
        n = self.n_ao
        j = np.zeros((n, n))
        k = np.zeros((n, n))
        weights = self._quartet_weights()
        # Bound the temporary quartet arrays; DZP has ~12M stored quartets.
        chunk = 2_000_000
        for start in range(0, self._values.size, chunk):
            stop = min(start + chunk, self._values.size)
            idx = self._indices[start:stop]
            values = self._values[start:stop]
            weighted = values * weights[start:stop]
            for permutation in _PERMUTATIONS:
                p, q, r, s = (idx[:, i] for i in permutation)
                j += np.bincount(
                    p * n + q,
                    weights=weighted * dm[r, s],
                    minlength=n * n,
                ).reshape(n, n)
                k += np.bincount(
                    p * n + r,
                    weights=weighted * dm[q, s],
                    minlength=n * n,
                ).reshape(n, n)
        return j, k

    def _sparse_transform(
        self, ca: np.ndarray, cb: np.ndarray, cc: np.ndarray, cd: np.ndarray
    ) -> np.ndarray:
        """Active-space transform over the stored quartets, never forming ``n_ao^4``.

        The right pair is transformed first into an ``(n_ao, n_ao, n_mo, n_mo)``
        intermediate (~1M entries at 100 AOs and 10 active orbitals), then the
        left pair is contracted.  Each stored quartet is expanded through the eight
        symmetry permutations with the same repeated-index weights as :meth:`jk`.
        Chunks bound the ``chunk x n_mo^2`` temporary to a few tens of MB.
        """
        self._load()
        assert self._indices is not None and self._values is not None and self.n_ao
        n = self.n_ao
        nmo_c, nmo_d = cc.shape[1], cd.shape[1]
        if ca.shape[1] != cb.shape[1]:
            raise ValueError("the two left coefficient blocks must have the same width")
        nmo_a = ca.shape[1]
        # right intermediate: [p, q, k, l]
        right = np.zeros((n * n, nmo_c, nmo_d), dtype=float)
        kl = np.arange(nmo_c)[:, None] * nmo_d + np.arange(nmo_d)[None, :]
        weights = self._quartet_weights()
        # ~64 MB of chunk x n_mo^2 doubles, independent of n_ao.
        chunk = max(1, min(200_000, 8_000_000 // max(1, nmo_c * nmo_d)))
        for start in range(0, self._values.size, chunk):
            stop = min(start + chunk, self._values.size)
            idx = self._indices[start:stop]
            weighted = self._values[start:stop] * weights[start:stop]
            for permutation in _PERMUTATIONS:
                p, q, r, s = (idx[:, i] for i in permutation)
                pair = p * n + q
                # (chunk, n_mo_c, n_mo_d), one right-pair transform per quartet.
                block = weighted[:, None, None] * cc[r][:, :, None] * cd[s][:, None, :]
                code = pair[:, None, None] * (nmo_c * nmo_d) + kl[None, :, :]
                right += np.bincount(
                    code.ravel(), weights=block.ravel(), minlength=n * n * nmo_c * nmo_d
                ).reshape(n * n, nmo_c, nmo_d)
        # left pair: [p, q, k, l] -> [i, j, k, l]
        right = right.reshape(n, n, nmo_c, nmo_d)
        out = np.einsum("pqkl,pi,qj->ijkl", right, ca, cb, optimize=True)
        if out.shape[:2] != (nmo_a, cb.shape[1]):
            raise RuntimeError("internal sparse-transform shape mismatch")
        return np.ascontiguousarray(out)

    def _transform_four(self, ca: np.ndarray, cb: np.ndarray, cc: np.ndarray, cd: np.ndarray):
        """Active-space change of basis.

        The sparse streaming path is for the large-AO regime that motivated it;
        within the dense cap the dense tensor plus a vectorised einsum is much
        faster (10-100x) and its memory is bounded by the provider's 150M-element
        guard.  ``EQI_SIESTA_SPARSE_ERI=1`` forces sparse, ``=0`` forces dense.
        """
        mode = os.environ.get("EQI_SIESTA_SPARSE_ERI", "auto").strip().lower()
        if mode in {"1", "true", "on", "yes"}:
            return self._sparse_transform(ca, cb, cc, cd)
        if mode in {"0", "false", "off", "no"}:
            tensor = np.einsum("pqrs,pi->iqrs", self.dense(), ca, optimize=True)
            tensor = np.einsum("iqrs,qj->ijrs", tensor, cb, optimize=True)
            tensor = np.einsum("ijrs,rk->ijks", tensor, cc, optimize=True)
            return np.einsum("ijks,sl->ijkl", tensor, cd, optimize=True)
        if self.n_ao**4 <= 20_000_000:
            tensor = np.einsum("pqrs,pi->iqrs", self.dense(), ca, optimize=True)
            tensor = np.einsum("iqrs,qj->ijrs", tensor, cb, optimize=True)
            tensor = np.einsum("ijrs,rk->ijks", tensor, cc, optimize=True)
            return np.einsum("ijks,sl->ijkl", tensor, cd, optimize=True)
        return self._sparse_transform(ca, cb, cc, cd)

    def eri_mo(self, mo_coeff: np.ndarray) -> np.ndarray:
        coeff = np.asarray(mo_coeff, dtype=float)
        if coeff.ndim != 2 or coeff.shape[0] != self.n_ao:
            raise ValueError(f"mo_coeff is {coeff.shape}, expected ({self.n_ao}, n_mo)")
        return self._transform_four(coeff, coeff, coeff, coeff)

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        coeff_a = np.asarray(mo_a, dtype=float)
        coeff_b = np.asarray(mo_b, dtype=float)
        if coeff_a.ndim != 2 or coeff_b.ndim != 2:
            raise ValueError(f"mo_a/mo_b must be 2-D; got {coeff_a.shape}/{coeff_b.shape}")
        if coeff_a.shape[0] != self.n_ao or coeff_b.shape[0] != self.n_ao:
            raise ValueError(
                f"mo_a/mo_b must have {self.n_ao} rows; got {coeff_a.shape}/{coeff_b.shape}"
            )
        return self._transform_four(coeff_a, coeff_a, coeff_b, coeff_b)

    def eri_mo_block(
        self,
        mo_a: np.ndarray,
        mo_b: np.ndarray,
        mo_c: np.ndarray,
        mo_d: np.ndarray,
    ) -> np.ndarray:
        """Chemist-notation ``(p_a q_b | r_c s_d)`` for four independent blocks.

        The sparse quartet transform requires the *two left* blocks to have equal
        width, which is not true of the mixed ``(ia|jb)`` block the MP2 ranking
        needs (one occupied and one virtual on each side).  When the direct
        transform is not rectangularly compatible, transform the right pair with an
        identity left pair -- always allowed -- and contract the left pair into the
        resulting ``(n_ao, n_ao, n_c, n_d)`` intermediate.  For the sizes the
        ranking uses (2 occupied x 86 virtual on a 100-AO basis) that intermediate
        is ~1.7M doubles, not the fourth power of the subsystem.
        """
        ca = np.asarray(mo_a, dtype=float)
        cb = np.asarray(mo_b, dtype=float)
        cc = np.asarray(mo_c, dtype=float)
        cd = np.asarray(mo_d, dtype=float)
        for name, block in (("mo_a", ca), ("mo_b", cb), ("mo_c", cc), ("mo_d", cd)):
            if block.ndim != 2 or block.shape[0] != self.n_ao:
                raise ValueError(f"{name} must be 2-D with {self.n_ao} rows; got {block.shape}")
        self._load()
        assert self.n_ao is not None
        if ca.shape[1] == cb.shape[1] and cc.shape[1] == cd.shape[1]:
            return self._transform_four(ca, cb, cc, cd)
        identity = np.eye(self.n_ao)
        right = self._transform_four(identity, identity, cc, cd)
        return np.ascontiguousarray(np.einsum("pqkl,pi,qj->ijkl", right, ca, cb, optimize=True))
