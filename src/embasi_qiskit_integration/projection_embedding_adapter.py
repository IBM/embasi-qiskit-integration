# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Real EmbASI glue: ``ProjectionEmbedding`` -> ``EmbeddedHamiltonian``.

Design in one line
------------------
``ProjectionEmbedding.construct_embedded_fock()`` returns a *Fock* matrix; a
correlated solver (FCI/SQD) needs the *one-body* embedded Hamiltonian plus the
bare ERIs.  The adapter strips the high-level mean field back off, rebuilds an
orthonormal orbital set from the embedded Fock, and downfolds CASCI-style.

    F_emb  =  h_core + v_emb[γ^A, γ^B] + P_B  +  G_HL[γ^A]        (paper Eq. 4-5)
    h_emb  =  F_emb - G_HL[γ^A]              <- what the solver's h1 comes from

Two distinct effective potentials
---------------------------------
``G_HL`` above must match whatever ``calc_base_hl`` is (KS or HF), because it is
undoing a subtraction.  The *downfolding* of inactive orbitals into ``e_core``
and ``h1`` must instead always use Hartree-Fock (J - K/2), because the object
handed to the solver is a bare electronic Hamiltonian.  Hence two methods on the
integral backend: ``veff_hl`` and ``veff_hf``.  Conflating them silently
subtracts an xc potential where exchange is required.

Requirement on the EmbASI side
------------------------------
``projection="level-shift"``.  Huzinaga (Eq. 7) is not exportable as a constant
offset because P_B depends on the high-level Fock inside the SCF.

What this adapter reads out of EmbASI (and what would be cleaner upstream)
-------------------------------------------------------------------------
The adapter is self-contained: it makes the workflow run end-to-end against the
current EmbASI by reading its internals directly.  A few of those reads reach
into implementation details that would be better as stable public API; those
carry an inline ``TODO(embasi-api)``.

Handled adapter-side (noted here for future upstream cleanup):

* **Low-level energies.**  ``construct_embedded_fock()`` stores E_L[γ^A+γ^B] and
  E_L[γ^A] as ``subsys_AB_lowlvl_scftotalen`` / ``subsys_A_lowlvl_totalen`` (eV).
  :meth:`_low_level_energies` reads and converts them.  Cleaner upstream: expose
  them under stable names in Hartree.
* **Complex dtype.**  SPADE returns real restricted data (densities, Fock, and
  MO coefficients) as ``complex128``; :meth:`_as_ao_matrix` and
  :meth:`_as_ao_by_mo` assert a negligible imaginary part and take ``.real``.
  Left complex, the MO coefficients push PySCF's ``ao2mo`` onto its relativistic
  (spinor) path.  Cleaner upstream: return a real dtype for the restricted
  single-k case.
* **h1 Hermiticity.**  ``C^T (h_emb+veff_in) C`` is Hermitian in exact
  arithmetic but the rotation accumulates round-off that grows with the
  active-space size; :meth:`embedded_hamiltonian` asserts the asymmetry is
  round-off-scale (< 1e-6) and symmetrizes it, so the contract's strict 1e-8
  check passes for large bases.
* **Level-shift mu.**  Cross-checked against ``p.mu_val`` in
  :meth:`run_low_level`.
* **Density feedback shape.**  ``construct_embedded_fock(dmab_in=...)`` expects a
  ``SpinKpointArray``, not the plain ``(nao, nao)`` density the outer loop feeds
  back; :meth:`_as_spin_kpoint_array` wraps it (n_spin=1, n_kpoints=1) so the
  multi-cycle loop runs against real EmbASI.  Cleaner upstream: accept a plain
  array, or expose the wrapper as public API.

Genuine upstream asks that remain (each marked ``TODO(embasi-api)`` inline):

* **v_emb / P_B as readable attributes.**  We reconstruct both by subtraction
  (correct, but silently fragile if F_emb assembly changes; ``mu`` is now
  cross-checked against ``mu_val``).  *Ask:* expose them.  See :attr:`p_b`,
  :attr:`h_emb`.
* **Retained-AO index map under basis truncation** (paper Sec. 2.4).  Needed to
  *assert* that ``F_emb``, ``S``, and ``mo_coeffs_A_LL`` share a basis rather
  than trust it.  See :meth:`run_low_level`.
* **RI-LVL three-index ERI export.**  Without it FHI-aims cannot drive the
  WF-in-DFT half and the workflow stays PySCF-only.  *Ask:* export the
  ``V^{-1/2}``-contracted ``M^P_{pq}`` so ``(pq|rs) = sum_P M^P_pq M^P_rs``.
  See :class:`FHIaimsIntegrals`.
* **Huzinaga as a constant offset when high/low xc match** (paper Sec. 2.1).
  ``H^{AB}_H`` then collapses to the supersystem low-level Hamiltonian, making
  ``P_B`` a constant matrix -- exportable after all, in exactly the WF-in-DFT
  regime.  See :meth:`__init__`.

Open-shell / unrestricted embedding is a deliberately-deferred package rewrite
(separate alpha/beta orbital sets, an ``(h1a, h1b)`` pair, spin-symmetric
downfold, and SQD spin handling); see :meth:`_as_ao_by_mo`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.linalg as sla

from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult

# Environment orbitals come out of the generalized eigenproblem at ~mu * N_B.
_ENV_EIGENVALUE_FLOOR = 1.0e3

# EmbASI works internally in eV.  Its Hartree->eV factor is a bare literal
# 27.211384500 repeated inline throughout embasi/{embedding,atoms_embedding_asi,
# qmcode_adapters}.py -- it is never bound to an importable symbol, so we mirror
# the exact literal here.  It MUST equal EmbASI's value (not CODATA 27.211386...)
# so that dividing its eV energies back to Hartree exactly inverts what EmbASI
# multiplied; a different constant would reintroduce a ~1e-7 relative error.
_EMBASI_HA2EV = 27.211384500  # keep in sync with embasi's inline factor
_EV2HA = 1.0 / _EMBASI_HA2EV

# EmbASI's SPADE eigensolve returns real restricted data carried in a complex128
# dtype.  A nonzero imaginary part above this is a genuine error (multi-k / open
# shell / a bug), not round-off -- so we assert rather than silently discard it.
_IMAG_TOL = 1.0e-9


# --------------------------------------------------------------------------- #
# AO integrals: the quantities EmbASI/ASI does not (yet) hand you
# --------------------------------------------------------------------------- #
class AOIntegrals(Protocol):
    """AO-basis quantities the adapter needs beyond ProjectionEmbedding."""

    def overlap(self) -> np.ndarray: ...
    def hcore(self) -> np.ndarray: ...

    def veff_hl(self, dm: np.ndarray) -> np.ndarray:
        """Effective potential of the *high-level* calculator, at a 2-occupancy dm.

        Must match ``calc_base_hl``: it exists only to undo the mean-field
        contribution EmbASI folded into ``F_emb``.
        """

    def veff_hf(self, dm: np.ndarray) -> np.ndarray:
        """Hartree-Fock effective potential J[dm] - K[dm]/2, at a 2-occupancy dm.

        Used for the inactive-core downfold, independently of the high level.
        """

    def eri_mo(self, mo_coeff: np.ndarray) -> np.ndarray:
        """Chemist-notation (pq|rs), shape (nmo,)*4, for the given MO block."""

    def energy_nuc(self) -> float: ...


class PySCFIntegrals:
    """Backend for the ``examples/PySCF`` path of the qm-code-adapter branch.

    ``density_fit`` selects the two-electron transform in :meth:`eri_mo`:

    - ``False`` (default): the exact dense ``ao2mo`` N^5 transform.  Bit-for-bit
      what the workflow produced before this option existed.
    - ``True`` / an auxbasis string: a density-fitted ``(pq|rs)`` via
      ``pyscf.df.DF``.  This is an *approximation* (the DF/RI error, typically
      1e-3-1e-4 Ha on valence integrals), traded for O(N^3) storage so the active
      space can outgrow the few dozen orbitals the dense (nmo)^4 tensor allows.
      A string is passed straight through as the fitting auxbasis (e.g.
      ``"cc-pvtz-ri"``); ``True`` lets PySCF pick the default aux for ``mol.basis``.
    """

    def __init__(self, mf_hl, *, density_fit: bool | str = False):
        self.mf = mf_hl                 # the same object passed to calc_base_hl
        self.mol = mf_hl.mol
        self._hf = mf_hl.mol.RHF()      # integral engine only; never kernel()'d
        self._density_fit = density_fit
        self._df = None                 # built lazily on first eri_mo call

    def overlap(self) -> np.ndarray:
        return np.asarray(self.mol.intor("int1e_ovlp"))

    def hcore(self) -> np.ndarray:
        return np.asarray(self.mf.get_hcore())

    def veff_hl(self, dm) -> np.ndarray:
        return np.asarray(self.mf.get_veff(self.mol, np.asarray(dm)))

    def veff_hf(self, dm) -> np.ndarray:
        return np.asarray(self._hf.get_veff(self.mol, np.asarray(dm)))

    def eri_mo(self, mo_coeff) -> np.ndarray:
        """Chemist-notation ``(pq|rs)`` over the given MO block, shape (nmo,)*4.

        Dense by default; density-fitted when the backend was constructed with
        ``density_fit`` (see the class docstring).  Both return a full unpacked
        tensor -- the DF path only changes how it is *built*, not its shape, so
        the downfold and the solver are oblivious to the choice.
        """
        from pyscf import ao2mo

        nmo = mo_coeff.shape[1]
        if not self._density_fit:
            return ao2mo.restore(1, ao2mo.kernel(self.mol, mo_coeff), nmo)

        # Density-fitted transform: (pq|rs) ~= sum_P (pq|P)(P|rs), O(N^3) storage.
        if self._df is None:
            from pyscf import df

            self._df = df.DF(self.mol)
            if isinstance(self._density_fit, str):
                self._df.auxbasis = self._density_fit
            self._df.build()
        return ao2mo.restore(1, self._df.ao2mo(mo_coeff), nmo)

    def energy_nuc(self) -> float:
        return float(self.mol.energy_nuc())


class FHIaimsIntegrals:
    """FHI-aims path.

    TODO(embasi-api): ASK EmbASI/ASI for an RI-LVL three-index ERI export
    (the V^-1/2-contracted M^P_{pq}, so (pq|rs) = sum_P M^P_pq M^P_rs).  SYMPTOM:
    ASI exports density/overlap/Hamiltonian but not the (truncated) ERIs, so this
    backend cannot be built at all -- FHI-aims can drive only the DFT-in-DFT part
    of the workflow and WF-in-DFT stays PySCF-only.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError("RI-LVL ERI export not available through ASI yet")


# --------------------------------------------------------------------------- #
# Orbital bookkeeping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmbeddedOrbitals:
    """Orthonormal orbitals of subsystem A in the (possibly truncated) AO basis."""

    coeff: np.ndarray        # (nao, nmo_A) -- environment columns already dropped
    energy: np.ndarray       # (nmo_A,)
    n_occ: int               # doubly occupied orbitals of A
    inactive: np.ndarray     # column indices frozen into e_core
    active: np.ndarray       # column indices handed to the solver

    @property
    def c_inactive(self) -> np.ndarray:
        return self.coeff[:, self.inactive]

    @property
    def c_active(self) -> np.ndarray:
        return self.coeff[:, self.active]

    @property
    def n_active_orbitals(self) -> int:
        return len(self.active)

    @property
    def n_active_electrons(self) -> int:
        """Implied by the partition; never supplied by the caller."""
        return 2 * (self.n_occ - len(self.inactive))

    def __str__(self) -> str:
        return (
            f"CAS({self.n_active_orbitals}o, {self.n_active_electrons}e) "
            f"from {self.n_occ} occupied + {self.coeff.shape[1] - self.n_occ} "
            f"virtual orbitals on subsystem A"
        )


@dataclass(frozen=True)
class ProjectionEnergy:
    """Term-by-term breakdown of paper Eq. 8."""

    e_low_total: float       # E_L[γ^A + γ^B]
    e_low_A: float           # E_L[γ^A]
    e_high_A: float          # E_H[Ψ̃^A], embedding potential removed
    correction: float        # tr[(γ̃^A - γ^A) v_emb]
    projector_leak: float    # tr[γ̃^A P_B], a numerical zero when clean

    @property
    def total(self) -> float:
        return self.e_low_total - self.e_low_A + self.e_high_A + self.correction


# Selector signature: (coeff, energy, n_occ) -> active column indices
Selector = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class ProjectionEmbeddingAdapter:
    """Wraps a live ``embasi.embedding.ProjectionEmbedding``."""

    def __init__(
        self,
        projection,                       # embasi.embedding.ProjectionEmbedding
        integrals: AOIntegrals,
        *,
        mu: float = 1.0e6,                # level-shift parameter, paper Eq. 6
        env_eigenvalue_floor: float = _ENV_EIGENVALUE_FLOOR,
    ):
        if getattr(projection, "projection", None) != "level-shift":
            # TODO(embasi-api): ASK EmbASI to export Huzinaga as a constant
            # offset in the same-functional case.  SYMPTOM: any projection other
            # than level-shift is rejected here, so Huzinaga-in-DFT cannot be
            # solved externally -- even though the paper (Sec. 2.1) shows
            # H^AB_H collapses to the supersystem low-level Hamiltonian when the
            # high/low xc match, making P_B constant and exportable in exactly the
            # WF-in-DFT regime.
            raise ValueError(
                "only projection='level-shift' can be exported to an external "
                "solver; Huzinaga needs the high-level Fock inside the SCF"
            )
        self.p = projection
        self.ints = integrals
        self.mu = mu
        self._floor = env_eigenvalue_floor

        self._dm_a = None      # γ^A  (localized, low level), AO, 2-occupancy
        self._dm_b = None      # γ^B  (environment, frozen), AO
        self._fock = None      # F_emb
        self._s = None

    # ---------------- low-level embedding ---------------- #
    def run_low_level(self, dm_ab_in: np.ndarray | None = None) -> None:
        """Drive EmbASI: supersystem SCF, SPADE/PM localisation, F_emb."""
        # TODO(embasi-api): ASK EmbASI to expose the retained-AO index array
        # under basis truncation (paper Sec. 2.4, threshold tau).  SYMPTOM: with
        # truncation on, every matrix below must return in the *same* truncated
        # AO ordering and mo_coeffs_A_LL be sliced to match; without the index
        # map the adapter can only *trust* that ints.overlap() and F_emb share a
        # basis, not assert it -- a mismatch would corrupt every downstream
        # contraction silently.
        if dm_ab_in is None:
            dm_a, dm_b, fock = self.p.construct_embedded_fock()
        else:
            # EmbASI's construct_embedded_fock wants dmab_in as a
            # SpinKpointArray, not the plain (nao, nao) density the outer loop
            # feeds back -- wrap it (mirror of the _as_ao_matrix squeeze on the
            # way out) so the multi-cycle loop runs against real EmbASI.
            dm_a, dm_b, fock = self.p.construct_embedded_fock(
                dmab_in=self._as_spin_kpoint_array(dm_ab_in)
            )

        # EmbASI returns SpinKpointArray objects (leading (nspin, nkpt) axes)
        # holding real restricted data in a complex128 dtype; _as_ao_matrix
        # squeezes the length-1 leading axes and drops the (asserted-negligible)
        # imaginary part, so everything downstream (einsum, sla.eigh, veff) sees
        # a bare 2-occupancy real (nao, nao) matrix.
        self._dm_a = self._as_ao_matrix(dm_a)
        self._dm_b = self._as_ao_matrix(dm_b)
        self._fock = self._as_ao_matrix(fock)
        self._s = self.ints.overlap()
        self._validate_densities()

        # Cross-check our level-shift mu against the value EmbASI used inside the
        # SCF.  P_B is reconstructed on our side as mu * S gamma^B S, so a silent
        # mismatch here would corrupt the projector without any other symptom.
        # (v_emb / P_B themselves are still recovered by subtraction -- see the
        # TODO(embasi-api) on `p_b` / `h_emb`.)
        mu_embasi = getattr(self.p, "mu_val", None)
        if mu_embasi is not None and not np.isclose(
            float(mu_embasi), self.mu, rtol=1e-9, atol=0.0
        ):
            raise ValueError(
                f"level-shift mu mismatch: adapter mu={self.mu:g} but EmbASI "
                f"used mu_val={float(mu_embasi):g}; the reconstructed P_B would "
                "not match the projector inside F_emb"
            )

    @staticmethod
    def _as_ao_matrix(m) -> np.ndarray:
        """Squeeze a closed-shell EmbASI (nspin, nkpt, nao, nao) block to (nao, nao).

        The returned ``SpinKpointArray`` carries leading spin/k-point axes of
        length 1 for the restricted single-k case handled here.  A leading axis
        of length > 1 means unrestricted or multi-k, which the restricted
        downfold cannot represent.

        EmbASI's SPADE eigensolve returns this real, restricted data in a
        ``complex128`` dtype.  We take the real part -- but only after asserting
        the imaginary part is numerically zero, so a genuinely complex block
        (multi-k, open shell, or an upstream bug) is caught loudly instead of
        being silently truncated to its real projection.
        """
        m = np.asarray(m)
        while m.ndim > 2:
            if m.shape[0] != 1:
                raise NotImplementedError(
                    "open-shell / multi-k embedding not wired up "
                    f"(leading axis has length {m.shape[0]}, expected 1)"
                )
            m = m[0]
        if np.iscomplexobj(m):
            max_imag = float(np.abs(m.imag).max()) if m.size else 0.0
            if max_imag > _IMAG_TOL:
                raise ValueError(
                    f"embedded matrix has a non-negligible imaginary part "
                    f"(max |imag| = {max_imag:.2e} > {_IMAG_TOL:.0e}); the "
                    "restricted single-k downfold assumes real data"
                )
            m = m.real
        return np.ascontiguousarray(m)

    @staticmethod
    def _as_spin_kpoint_array(m: np.ndarray):
        """Wrap a plain ``(nao, nao)`` density in EmbASI's ``SpinKpointArray``.

        This is the inverse of :meth:`_as_ao_matrix`: on the way *out* EmbASI
        hands back a ``SpinKpointArray`` that we squeeze to a bare matrix; on the
        way *back in* (density feedback, ``dmab_in``) EmbASI expects the same
        wrapper again, not an ndarray.  We wrap for the restricted single-k case
        (``n_spin=1, n_kpoints=1``); the wrapper is what both EmbASI consumers of
        ``density_matrix_in`` need -- ``qmcode_adapters.run_scf`` indexes it as
        ``[0, 0]`` and ``atoms_embedding_asi.run`` uses its ``@`` / ``trace``.

        When EmbASI is not installed (the default test suite runs the outer loop
        against a PySCF-backed mock, no EmbASI), fall back to a ``(1, 1, nao,
        nao)`` ndarray, which is ``[0, 0]``-indexable in the same way -- enough
        for the mock, which only unwraps.  A real embedding run always has
        EmbASI, so the ndarray fallback never reaches its ``@`` / ``trace`` path.
        """
        block = np.ascontiguousarray(np.asarray(m))
        try:
            from embasi.ks_array import SpinKpointArray
        except ImportError:
            return block[np.newaxis, np.newaxis, :, :]
        return SpinKpointArray({(0, 0): block}, n_spin=1, n_kpoints=1)

    # ---------------- MO coefficients ---------------- #
    @property
    def mo_a_ll(self) -> np.ndarray:
        """Localized *occupied* MOs of A, low level, normalized to (nao, nocc).

        These come back from EmbASI as ``mo_coeffs_A_LL``.  Two things they are
        NOT: (a) a complete orbital set -- there are no virtuals; (b) canonical
        -- they are SPADE/Pipek-Mezey rotations, so they diagonalize no Fock
        operator and their "orbital energies" are meaningless.  Use them for the
        span and the electron count only.
        """
        return self._as_ao_by_mo(self.p.mo_coeffs_A_LL)

    @property
    def mo_b_ll(self) -> np.ndarray:
        return self._as_ao_by_mo(self.p.mo_coeffs_B_LL)

    def _as_ao_by_mo(self, c) -> np.ndarray:
        """Fix layout: ASI/Fortran may hand back (nmo, nao) or a leading spin axis."""
        c = np.asarray(c)
        if c.ndim == 3:                       # (nspin, ., .) -- closed shell only
            # Open-shell embedding is a deferred package rewrite (see the
            # module docstring): everything downstream assumes a 2-occupancy
            # density and a spin-restricted downfold, so supporting open shells
            # means separate alpha/beta orbital sets, an (h1a, h1b) pair, and
            # SQD's spin-symmetry handling on the solver side.
            if c.shape[0] != 1:
                raise NotImplementedError("open-shell embedding not wired up")
            c = c[0]
        if np.iscomplexobj(c):
            # Same real-in-complex dtype as the density/Fock blocks (see
            # _as_ao_matrix): SPADE carries real restricted MO coefficients in a
            # complex128 array.  Drop the imaginary part -- but only after
            # asserting it is negligible, so a genuinely complex block is caught
            # loudly.  Left complex, these coefficients push the downstream
            # ao2mo onto PySCF's relativistic (spinor) path, which mis-broadcasts.
            max_imag = float(np.abs(c.imag).max()) if c.size else 0.0
            if max_imag > _IMAG_TOL:
                raise ValueError(
                    f"MO coefficients have a non-negligible imaginary part "
                    f"(max |imag| = {max_imag:.2e} > {_IMAG_TOL:.0e}); the "
                    "restricted single-k downfold assumes real data"
                )
            c = c.real
        nao = self._s.shape[0]
        if c.shape[0] != nao and c.shape[1] == nao:
            c = c.T
        # Definitive check rather than a guess: C^T S C == 1
        gram = c.T @ self._s @ c
        if not np.allclose(gram, np.eye(c.shape[1]), atol=1e-6):
            raise ValueError(
                f"MO coefficients are not S-orthonormal (max deviation "
                f"{np.abs(gram - np.eye(c.shape[1])).max():.2e}); check layout, "
                "basis-truncation index map, or normalisation convention"
            )
        return np.ascontiguousarray(c)

    # ---------------- embedding operators ---------------- #
    @property
    def p_b(self) -> np.ndarray:
        """Level-shift projector, paper Eq. 6.  Reconstructed, not exported."""
        # TODO(embasi-api): ASK EmbASI to expose P_B directly.  SYMPTOM: we
        # rebuild it as mu * S gamma^B S; mu itself is now cross-checked against
        # p.mu_val in run_low_level, but the assembly still assumes EmbASI's own
        # P_B has exactly this form -- exposing it would remove the assumption.
        return self.mu * (self._s @ self._dm_b @ self._s)

    @property
    def h_emb(self) -> np.ndarray:
        """h_core + v_emb + P_B: the one-body operator the solver must see.

        TODO(embasi-api): ASK EmbASI to expose ``v_emb`` (and ideally
        ``P_B``) directly.  SYMPTOM: this reconstruction assumes ``F_emb`` was
        built at ``dm_a`` with the calculator wrapped by ``integrals.veff_hl``;
        it is correct today but any change to how EmbASI assembles ``F_emb``
        breaks it silently rather than loudly.
        """
        return self._fock - self.ints.veff_hl(self._dm_a)

    @property
    def v_emb(self) -> np.ndarray:
        """Eq. 3, recovered by subtraction.  Needed for the energy correction."""
        return self.h_emb - self.ints.hcore() - self.p_b

    # ---------------- orbital construction ---------------- #
    def _eigh_subsystem_a(self) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalize F_emb inside span(A) instead of the full AO basis.

        span(B) is spanned by the localized environment orbitals ``mo_b_ll``, so
        its S-orthogonal complement *is* span(A).  Build an S-orthonormal basis
        ``Q`` of that complement (dim ``nao - n_occ_B``), solve the small
        standard eigenproblem ``(Q^T F_emb Q) u = eps u``, and map back
        ``C = Q u``.  The returned orbitals are S-orthonormal by construction and
        the level shift no longer appears (B is gone), so no ``eps < floor`` cut
        is needed.  For a large environment this replaces an O(nao^3) generalized
        solve with one on the much smaller A block.

        Falls back to nothing: if ``mo_b_ll`` is unavailable the caller uses the
        full-basis path via ``restrict_to_a=False``.
        """
        s = self._s
        c_b = self.mo_b_ll                    # (nao, n_occ_B), S-orthonormal
        # Project span(B) out in the S-metric: P = I - c_b c_b^T S.
        proj = np.eye(s.shape[0]) - c_b @ (c_b.T @ s)
        # S-orthonormal basis of R^nao (columns of L^-T with L L^T = S), deflated.
        chol = np.linalg.cholesky(s)
        x_all = sla.solve_triangular(chol.T, np.eye(s.shape[0]), lower=False)
        y = proj @ x_all
        # Re-S-orthonormalize the deflated set and drop the null space (=span B).
        gram = y.T @ s @ y
        w, u = np.linalg.eigh(gram)
        nonzero = w > 1e-8
        q = y @ u[:, nonzero] @ np.diag(1.0 / np.sqrt(w[nonzero]))
        # Small standard eigenproblem in the A basis (Q^T S Q = I by construction).
        fs = q.T @ self._fock @ q
        eps, cc = np.linalg.eigh(fs)
        return eps, q @ cc

    def build_orbitals(
        self,
        *,
        n_frozen_occ: int = 0,
        n_virtual: int | None = None,
        selector: Selector | None = None,
        restrict_to_a: bool = True,
    ) -> EmbeddedOrbitals:
        """Orbitals of subsystem A from the generalized problem F_emb C = S C eps.

        The default is the **full** subsystem-A space: every orbital surviving
        the level shift.  That is the WF-in-DFT problem the paper actually
        solves; projection-based embedding has no notion of an active space.
        The electron count is inferred from ``mo_coeffs_A_LL`` and is never a
        caller argument -- N_act = 2 * (n_occ^A - n_frozen).

        ``n_frozen_occ`` and ``n_virtual`` exist only to fit a solver budget
        (qubit count, determinant count), which is a property of the solver and
        not of the embedding.  Nothing in ``ProjectionEmbedding`` knows it.

        Rationale for diagonalizing F_emb rather than reusing the localized
        orbitals: ``mo_coeffs_A_LL`` spans only the occupied space of A, so it
        cannot supply virtuals.  Diagonalizing gives occupied *and* virtual in
        one shot, and the level shift pushes every environment orbital to
        ~mu*N_B, making the A/B split a threshold test rather than a heuristic.

        ``restrict_to_a`` (default) deflates the environment before the eigensolve
        (see :meth:`_eigh_subsystem_a`): span(B) is known from ``mo_coeffs_B_LL``,
        so we project it out and diagonalize F_emb in its S-orthogonal complement
        -- exactly span(A) -- instead of over the full AO basis.  For the paper's
        large-environment systems this is the whole cost win.  Set ``False`` to
        recover the plain full-basis ``sla.eigh(F_emb, S)``; the two agree to
        solver precision (asserted in the tests), so this flag is a pure
        performance knob, never a physics one.
        """
        if restrict_to_a:
            eps, c = self._eigh_subsystem_a()
        else:
            eps, c = sla.eigh(self._fock, self._s)
            keep = eps < self._floor          # level shift removes subsystem B
            eps, c = eps[keep], c[:, keep]

        n_occ = self.mo_a_ll.shape[1]         # inferred, never passed in
        self._validate_span(c[:, :n_occ])

        n_virt_total = c.shape[1] - n_occ
        n_virt = n_virt_total if n_virtual is None else min(n_virtual, n_virt_total)
        if not 0 <= n_frozen_occ < n_occ:
            raise ValueError(
                f"n_frozen_occ={n_frozen_occ} outside [0, {n_occ}) for subsystem A"
            )

        if selector is not None:
            # Hook for AVAS / MP2-NOON / concentric-localisation selection.  A
            # concentric-localisation selector (Claudino & Mayhall, same lineage
            # as SPADE) would order virtuals into shells by overlap with the A
            # fragment and cut on the singular-value gap
            # Δσ_i^2 = σ_i^2 - σ_{i+1}^2, exactly as the paper picks the occupied
            # A space, making n_virtual inferable up to a tolerance.
            active = np.asarray(selector(c, eps, n_occ), dtype=int)
        else:
            active = np.arange(n_frozen_occ, n_occ + n_virt)

        inactive = np.array(
            [i for i in range(n_occ) if i not in set(active.tolist())], dtype=int
        )
        return EmbeddedOrbitals(
            coeff=c, energy=eps, n_occ=n_occ, inactive=inactive, active=active
        )

    # ---------------- downfolding ---------------- #
    def embedded_hamiltonian(self, orbitals: EmbeddedOrbitals) -> EmbeddedHamiltonian:
        """CASCI-style downfold of h_emb + bare ERIs onto the active space."""
        h_emb = self.h_emb
        c_in, c_act = orbitals.c_inactive, orbitals.c_active

        dm_in = 2.0 * (c_in @ c_in.T)
        veff_in = self.ints.veff_hf(dm_in)      # HF, regardless of the high level

        h1 = c_act.T @ (h_emb + veff_in) @ c_act
        # (h_emb + veff_in) is a symmetric operator and c_act is real, so h1 is
        # Hermitian in exact arithmetic.  The C^T M C rotation accumulates
        # round-off that grows with the active-space size (~1e-10 at 10 orbitals,
        # ~1e-7 at 112), which trips the contract's strict Hermiticity check.
        # Assert the asymmetry is at round-off scale -- so a genuinely
        # non-Hermitian input still fails loudly -- then symmetrize it away.
        asym = float(np.abs(h1 - h1.T).max())
        if asym > 1e-6:
            raise ValueError(
                f"h1 is non-Hermitian well beyond round-off (max |h1 - h1^T| = "
                f"{asym:.2e}); the embedded operator or orbitals are inconsistent"
            )
        h1 = 0.5 * (h1 + h1.T)
        h2 = self.ints.eri_mo(c_act)
        e_core = self.ints.energy_nuc() + np.einsum(
            "ij,ji->", dm_in, h_emb + 0.5 * veff_in
        )

        n_alpha = n_beta = orbitals.n_active_electrons // 2

        # P_B must be invisible inside the active space; if it is not, the A/B
        # separation is broken and everything above is meaningless.
        leak = float(np.abs(c_act.T @ self.p_b @ c_act).max())
        if leak > 1e-6:
            raise ValueError(
                f"active orbitals leak into subsystem B (|P_B| = {leak:.2e})"
            )

        return EmbeddedHamiltonian(
            h1=h1,
            h2=h2,
            e_core=float(e_core),
            nelec=(n_alpha, n_beta),
            meta={
                "source": "embasi-projection-embedding",
                "n_active_orbitals": int(h1.shape[0]),
                "projection": "level-shift",
                "mu": self.mu,
                "p_b_leak": leak,
            },
        )

    # ---------------- energy assembly ---------------- #
    def projection_energy(
        self, result: SolverResult, orbitals: EmbeddedOrbitals
    ) -> ProjectionEnergy:
        """Paper Eq. 8, undoing the embedding potential the solver already saw.

            E_solver = E_H[γ̃^A] + tr[γ̃^A (v_emb + P_B)]

        so E_H[Ψ̃^A] is recovered by subtraction, and the Eq. 8 correction term
        tr[(γ̃^A - γ^A) v_emb] is reported separately.  Summing the two shows the
        γ̃^A v_emb contributions cancel, leaving -tr[γ^A v_emb] against the raw
        solver energy -- a useful cross-check on the bookkeeping.
        """
        v_emb, p_b = self.v_emb, self.p_b
        dm_hl = self.rdm1_ao(result.rdm1, orbitals)

        leak = float(np.einsum("ij,ji->", dm_hl, p_b))
        e_high_a = float(result.energy) - np.einsum("ij,ji->", dm_hl, v_emb) - leak
        correction = float(np.einsum("ij,ji->", dm_hl - self._dm_a, v_emb))

        e_low_ab, e_low_a = self._low_level_energies()
        return ProjectionEnergy(
            e_low_total=e_low_ab,
            e_low_A=e_low_a,
            e_high_A=float(e_high_a),
            correction=correction,
            projector_leak=leak,
        )

    def _low_level_energies(self) -> tuple[float, float]:
        """(E_low(AB), E_low(A)) in Hartree, read from EmbASI internals.

        ``construct_embedded_fock()`` (via ``construct_embedding_potential``)
        already computes both low-level total energies as a side effect and
        stores them on the projection object, in eV:

          * ``subsys_AB_lowlvl_scftotalen``  = E_L[gamma^A + gamma^B]
          * ``subsys_A_lowlvl_totalen``      = E_L[gamma^A]

        EmbASI does not (yet) surface these under the paper's ``e_low_*`` names
        or in Hartree, so we read the internal attributes and convert with
        EmbASI's own eV factor (:data:`_EV2HA`).  If either is missing -- a
        future upstream rename -- we raise loudly rather than assemble a wrong
        energy silently.  The values can be complex-zero (same SPADE dtype as the
        matrices), so take ``.real``.
        """
        try:
            e_ab = self.p.subsys_AB_lowlvl_scftotalen
            e_a = self.p.subsys_A_lowlvl_totalen
        except AttributeError as exc:  # pragma: no cover - upstream rename guard
            raise AttributeError(
                "EmbASI did not expose the low-level energies "
                "'subsys_AB_lowlvl_scftotalen' / 'subsys_A_lowlvl_totalen' after "
                "construct_embedded_fock(); the projection energy cannot be "
                "assembled (upstream attribute renamed?)"
            ) from exc
        return float(np.real(e_ab)) * _EV2HA, float(np.real(e_a)) * _EV2HA

    # ---------------- density feedback ---------------- #
    def rdm1_ao(
        self, rdm1_active: np.ndarray, orbitals: EmbeddedOrbitals
    ) -> np.ndarray:
        """Back-transform the correlated (spin-summed) 1-RDM to the AO basis."""
        c_in, c_act = orbitals.c_inactive, orbitals.c_active
        return 2.0 * (c_in @ c_in.T) + c_act @ np.asarray(rdm1_active) @ c_act.T

    def feedback(self, rdm1_active: np.ndarray, orbitals: EmbeddedOrbitals) -> None:
        """Advance the embedding by one step from the correlated density.

        This is the single-step, *undamped* primitive (γ -> F(γ) with no
        mixing); the convergence loop that drives it (|Δρ| / |ΔE| criteria, cycle
        cap, density mixing, and the SQD reseed policy) lives in
        ``scripts/embedding_workflow.py::EmbeddingWorkflow._run_outer_loop``,
        because that is where solver construction, logging, and the MPI broadcast
        already are.  That loop applies its ``mix_alpha`` damping by calling
        :meth:`run_low_level` with a pre-mixed density directly, so this method
        is the bare (undamped) step callers can use when they do their own mixing.

        The fed-back total density (γ̃^A from the correlated solve, plus the
        frozen environment γ^B) is wrapped into EmbASI's ``SpinKpointArray`` by
        :meth:`run_low_level` before it reaches ``construct_embedded_fock``.
        """
        dm_hl = self.rdm1_ao(rdm1_active, orbitals)
        self.run_low_level(dm_ab_in=dm_hl + self._dm_b)

    # ---------------- checks ---------------- #
    def _validate_densities(self) -> None:
        n_a = np.einsum("ij,ji->", self._dm_a, self._s)
        n_b = np.einsum("ij,ji->", self._dm_b, self._s)
        if not np.isclose(round(n_a), n_a, atol=1e-6):
            raise ValueError(f"tr(γ^A S) = {n_a:.6f} is not integral")
        if not np.isclose(round(n_b), n_b, atol=1e-6):
            raise ValueError(f"tr(γ^B S) = {n_b:.6f} is not integral")

    def _validate_span(self, c_occ: np.ndarray) -> None:
        """Occupied block of F_emb must span the same space as mo_coeffs_A_LL."""
        overlap = self.mo_a_ll.T @ self._s @ c_occ
        sv = np.linalg.svd(overlap, compute_uv=False)
        if sv.min() < 1.0 - 1e-6:
            raise ValueError(
                f"occupied A space mismatch (min singular value {sv.min():.6f}); "
                "the localisation and the embedded Fock disagree"
            )
