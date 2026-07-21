# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Core data contracts exchanged between the embedding side and the solvers.

``EmbeddedHamiltonian`` carries an active-space Hamiltonian (one- and two-body
integrals with the embedding potential already folded into ``h1``, plus the
scalar ``e_core`` offset). ``SolverResult`` carries the solver output (energy
and reduced density matrices) fed back into the embedding energy expression.

Both are frozen Pydantic models. They hold numpy arrays (``arbitrary_types_
allowed``), coerce list-like inputs to arrays on the way in, and validate
shapes/hermiticity via model validators.
"""

from __future__ import annotations

import warnings

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EmbeddedHamiltonian(BaseModel):
    """Active-space Hamiltonian ready for a many-body solver.

    Attributes:
        h1: One-body integrals, shape ``(norb, norb)``. Hermitian. The embedding
            potential is already folded in.
        h2: Two-body integrals, shape ``(norb, norb, norb, norb)`` in *chemists'*
            notation ``(pq|rs)``.
        e_core: Scalar offset (nuclear repulsion + environment/frozen-core).
        nelec: ``(n_alpha, n_beta)`` electron counts in the active space.
        meta: Free-form provenance (localisation, sha, reference energies, ...).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    h1: np.ndarray
    h2: np.ndarray
    e_core: float
    nelec: tuple[int, int]
    meta: dict = Field(default_factory=dict)

    @field_validator("h1", "h2", mode="before")
    @classmethod
    def _as_float_array(cls, value: object) -> np.ndarray:
        return np.asarray(value, dtype=float)

    @model_validator(mode="after")
    def _validate_shapes(self) -> "EmbeddedHamiltonian":
        h1, h2 = self.h1, self.h2
        if h1.ndim != 2 or h1.shape[0] != h1.shape[1]:
            raise ValueError(f"h1 must be square (norb, norb); got shape {h1.shape}")
        norb = h1.shape[0]
        if h2.shape != (norb, norb, norb, norb):
            raise ValueError(
                f"h2 must have shape {(norb,) * 4}; got {h2.shape} (norb inferred from h1)"
            )
        na, nb = self.nelec
        if na < 0 or nb < 0 or na > norb or nb > norb:
            raise ValueError(f"nelec {self.nelec} out of range for norb={norb}")

        # h1 hermiticity is a hard requirement.
        if not np.allclose(h1, h1.conj().T, atol=1e-8):
            raise ValueError("h1 is not Hermitian to atol=1e-8")

        # 8-fold permutational symmetry of h2 is only warned about (integrals
        # coming out of embedding may carry small asymmetries we tolerate).
        if not _has_eightfold_symmetry(h2, atol=1e-6):
            warnings.warn(
                "h2 does not satisfy 8-fold permutational symmetry to atol=1e-6",
                stacklevel=2,
            )
        return self

    @property
    def norb(self) -> int:
        return int(self.h1.shape[0])


class SolverResult(BaseModel):
    """Output of an active-space solve.

    Attributes:
        energy: Total energy = electronic + ``e_core``.
        rdm1: Spin-summed one-particle RDM, shape ``(norb, norb)``.
        rdm2: Optional two-particle RDM, shape ``(norb,)*4``.
        diagnostics: Free-form solver metadata (versions, seed, iterations, ...).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    energy: float
    rdm1: np.ndarray
    rdm2: np.ndarray | None = None
    diagnostics: dict = Field(default_factory=dict)

    @field_validator("rdm1", "rdm2", mode="before")
    @classmethod
    def _as_array(cls, value: object) -> object:
        if value is None:
            return None
        return np.asarray(value)


def _has_eightfold_symmetry(h2: np.ndarray, atol: float) -> bool:
    """Check the real 8-fold symmetry of chemists'-notation integrals (pq|rs).

    (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr) = (rs|pq) = (sr|pq) = (rs|qp) = (sr|qp)
    """
    checks = (
        h2.transpose(1, 0, 2, 3),  # (qp|rs)
        h2.transpose(0, 1, 3, 2),  # (pq|sr)
        h2.transpose(2, 3, 0, 1),  # (rs|pq)
    )
    return all(np.allclose(h2, c, atol=atol) for c in checks)
