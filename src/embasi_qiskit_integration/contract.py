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

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EmbeddedHamiltonian(BaseModel):
    """Active-space Hamiltonian ready for a many-body solver.

    Attributes:
        h1: One-body integrals, shape ``(norb, norb)``. Hermitian. The embedding
            potential is already folded in. On an open shell this is the
            **spin-averaged** operator, kept required so every existing consumer
            (SQD, FCIDUMP) keeps working unchanged.
        h1a: Optional alpha one-body integrals, shape ``(norb, norb)``, Hermitian.
            Given with ``h1b`` for a genuine spin-dependent downfold, where the two
            spin channels see different mean fields. Only a solver that can consume
            the pair uses it (``FCISolver`` -> ``pyscf.fci.direct_uhf``); the others
            fall back to ``h1`` and say so. ``None`` means spin-restricted.
        h1b: Optional beta one-body integrals. Must accompany ``h1a``.
        h2_spin: Optional ``(h2_aa, h2_ab, h2_bb)`` two-body triple, each
            ``(norb,)*4`` in chemists' notation, for a downfold whose two spin channels
            use *different* orbital sets. ``h2`` stays required and remains the
            alpha-only tensor every other consumer reads. Only ``h2_aa`` and ``h2_bb``
            carry full 8-fold permutational symmetry; the mixed ``h2_ab`` block is built
            from two different orbital sets, so it is symmetric only under
            ``(pq|rs) == (qp|rs) == (pq|sr)`` and is *not* checked for the 8-fold
            property. Requires ``h1a``/``h1b``.
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
    h1a: np.ndarray | None = None
    h1b: np.ndarray | None = None
    h2_spin: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    meta: dict = Field(default_factory=dict)

    @field_validator("h1", "h2", mode="before")
    @classmethod
    def _as_float_array(cls, value: object) -> np.ndarray:
        return np.asarray(value, dtype=float)

    @field_validator("h1a", "h1b", mode="before")
    @classmethod
    def _as_optional_float_array(cls, value: object) -> np.ndarray | None:
        return None if value is None else np.asarray(value, dtype=float)

    @field_validator("h2_spin", mode="before")
    @classmethod
    def _as_optional_eri_triple(cls, value: Any) -> Any:
        if value is None:
            return None
        arrays = tuple(np.asarray(v, dtype=float) for v in value)
        if len(arrays) != 3:
            raise ValueError(
                f"h2_spin must be a (h2_aa, h2_ab, h2_bb) triple; got {len(arrays)} arrays"
            )
        return arrays

    @property
    def is_spin_dependent(self) -> bool:
        """True when a genuine ``(h1a, h1b)`` pair is available."""
        return self.h1a is not None

    @property
    def has_spin_dependent_eri(self) -> bool:
        """True when a genuine ``(aa, ab, bb)`` two-body triple is available."""
        return self.h2_spin is not None

    @model_validator(mode="after")
    def _validate_shapes(self) -> EmbeddedHamiltonian:
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

        # The spin-dependent pair is all-or-nothing, same rule as SolverResult's
        # rdm1a/rdm1b: half a pair is a plumbing slip, and a solver picking the
        # present half would silently use one spin's operator for both channels.
        if (self.h1a is None) != (self.h1b is None):
            raise ValueError("h1a and h1b must be given together, or neither")
        h1a, h1b = self.h1a, self.h1b
        if h1a is not None and h1b is not None:
            # Bound to locals so the None-narrowing survives into the loop (a tuple of
            # the attributes would widen the element type back to `ndarray | None`).
            for name, arr in (("h1a", h1a), ("h1b", h1b)):
                if arr.shape != (norb, norb):
                    raise ValueError(
                        f"{name} must have shape {(norb, norb)} to match h1; got {arr.shape}"
                    )
                if not np.allclose(arr, arr.conj().T, atol=1e-8):
                    raise ValueError(f"{name} is not Hermitian to atol=1e-8")

        h2_spin = self.h2_spin
        if h2_spin is not None:
            if h1a is None:
                raise ValueError(
                    "h2_spin needs h1a/h1b: a spin-dependent two-body tensor with a "
                    "spin-averaged one-body operator is not a coherent Hamiltonian"
                )
            for name, arr in zip(("h2_aa", "h2_ab", "h2_bb"), h2_spin, strict=True):
                if arr.shape != (norb,) * 4:
                    raise ValueError(f"{name} must have shape {(norb,) * 4}; got {arr.shape}")
            # Only the same-spin blocks have the 8-fold property; h2_ab is built from
            # two different orbital sets, so (pq|rs) != (rs|pq) there by construction.
            for name, arr in (("h2_aa", h2_spin[0]), ("h2_bb", h2_spin[2])):
                if not _has_eightfold_symmetry(arr, atol=1e-6):
                    warnings.warn(
                        f"{name} does not satisfy 8-fold permutational symmetry to atol=1e-6",
                        stacklevel=2,
                    )
            ab = h2_spin[1]
            if not np.allclose(ab, ab.transpose(1, 0, 2, 3), atol=1e-6) or not np.allclose(
                ab, ab.transpose(0, 1, 3, 2), atol=1e-6
            ):
                warnings.warn(
                    "h2_ab is not symmetric under (pq|rs) == (qp|rs) == (pq|sr) to "
                    "atol=1e-6; the mixed-spin block looks malformed",
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
        rdm1a: Optional alpha one-particle RDM, shape ``(norb, norb)``. Required for
            unrestricted density feedback, where a spin-summed ``rdm1`` cannot
            represent ``gamma_alpha != gamma_beta``. When given with ``rdm1b``, the two
            must sum to ``rdm1``.
        rdm1b: Optional beta one-particle RDM, shape ``(norb, norb)``.
        diagnostics: Free-form solver metadata (versions, seed, iterations, ...).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    energy: float
    rdm1: np.ndarray
    rdm2: np.ndarray | None = None
    rdm1a: np.ndarray | None = None
    rdm1b: np.ndarray | None = None
    diagnostics: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_spin_rdms(self) -> SolverResult:
        """A spin-resolved pair must be complete and consistent with ``rdm1``.

        Half a pair is almost certainly a plumbing slip, and a pair that does not sum to
        the spin-summed RDM would feed two different densities into the same outer loop.
        """
        if (self.rdm1a is None) != (self.rdm1b is None):
            raise ValueError("rdm1a and rdm1b must be given together, or neither")
        if self.rdm1a is None:
            return self
        if not np.allclose(self.rdm1a + self.rdm1b, self.rdm1, atol=1e-8):
            worst = float(np.abs(self.rdm1a + self.rdm1b - self.rdm1).max())
            raise ValueError(
                f"rdm1a + rdm1b != rdm1 (max deviation {worst:.2e}); the spin-resolved "
                "and spin-summed densities disagree"
            )
        return self

    @property
    def is_spin_resolved(self) -> bool:
        """True when the alpha/beta RDM pair is available."""
        return self.rdm1a is not None

    def check_spin_sector(self, nelec: tuple[int, int], *, atol: float = 1e-6) -> float:
        """Verify ``trace(rdm1a) - trace(rdm1b)`` equals ``n_alpha - n_beta``.

        :meth:`check_particle_number` sums the pair, so it validates the *total* only and
        cannot tell ``(2, 2)`` from ``(3, 1)`` -- a solver that returned the wrong spin
        sector at the right total passes it. This closes that gap, and is the only check
        that can catch a swapped alpha/beta block once the RDMs are formed.

        Returns:
            The signed deviation ``(tr_a - tr_b) - (n_alpha - n_beta)``, for logging.

        Raises:
            ValueError: if no spin-resolved pair is present, or the deviation exceeds
                ``atol``.
        """
        if self.rdm1a is None:
            raise ValueError(
                "check_spin_sector needs the rdm1a/rdm1b pair; this SolverResult carries "
                "only a spin-summed rdm1"
            )
        observed = float(np.trace(self.rdm1a)) - float(np.trace(self.rdm1b))
        expected = float(nelec[0] - nelec[1])
        deviation = observed - expected
        if abs(deviation) > atol:
            raise ValueError(
                f"trace(rdm1a) - trace(rdm1b) = {observed:.6f} but nelec={nelec} implies "
                f"{expected:g} (off by {deviation:+.2e}, tolerance {atol:g}). The solver "
                "returned the wrong spin sector; check the alpha/beta bit layout."
            )
        return deviation

    @field_validator("rdm1", "rdm2", "rdm1a", "rdm1b", mode="before")
    @classmethod
    def _as_array(cls, value: object) -> object:
        if value is None:
            return None
        return np.asarray(value)

    def check_particle_number(self, nelec: tuple[int, int] | int, *, atol: float = 1e-6) -> float:
        """Verify ``trace(rdm1)`` equals the electron count; return the deviation.

        ``trace`` of a spin-summed one-particle RDM *is* the particle number, so a
        mismatch means the solver returned an RDM for a different sector than the
        Hamiltonian describes -- a wrong-but-plausible result. This matters beyond
        one solve: the RDM is fed back into the next embedding cycle's density, so an
        unchecked error propagates through the outer loop while every printed energy
        still looks reasonable.

        ``atol`` is loose enough to absorb the sampling noise of an SQD solve while
        still catching a whole-electron discrepancy.

        Args:
            nelec: the expected count, as ``(n_alpha, n_beta)`` or a total.
            atol: absolute tolerance on ``|trace(rdm1) - expected|``.

        Returns:
            The signed deviation ``trace(rdm1) - expected``, for logging.

        Raises:
            ValueError: if the deviation exceeds ``atol``, or ``rdm1`` is not square.
        """
        expected = float(sum(nelec)) if isinstance(nelec, tuple) else float(nelec)
        rdm1 = np.asarray(self.rdm1)
        if rdm1.ndim != 2 or rdm1.shape[0] != rdm1.shape[1]:
            raise ValueError(f"rdm1 must be a square matrix, got shape {rdm1.shape}")

        trace = float(np.trace(rdm1))
        deviation = trace - expected
        if abs(deviation) > atol:
            raise ValueError(
                f"trace(rdm1) = {trace:.6f} but the active space holds {expected:g} "
                f"electrons (off by {deviation:+.2e}, tolerance {atol:g}). The solver "
                "returned an RDM for the wrong particle-number sector; feeding it back "
                "would corrupt the embedding density. Check the solver's bitstring "
                "convention and its postselection."
            )
        return deviation


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
