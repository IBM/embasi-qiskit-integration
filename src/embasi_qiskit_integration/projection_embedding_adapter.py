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

"The paper" (Eq. 2, 4-6, 8, 19; §2.2; Fig. 3B) is the EmbASI framework paper:
G. Bramley, P. Stishenko, O. van Vuren, V. Blum, A. J. Logsdail, "A General
Pythonic Framework for DFT-in-DFT and WF-in-DFT Embedding", ChemRxiv (2025),
preprint, doi:10.26434/chemrxiv-2025-c23jf.  The projection / level-shift scheme
it implements originates with Manby, Stella, Goodpaster & Miller III, J. Chem.
Theory Comput. 8, 2564 (2012), doi:10.1021/ct300544e.

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

Upstream gaps that remain (each marked ``TODO(embasi-api)`` inline):

* **v_emb / P_B are not readable attributes.**  We reconstruct both by subtraction
  (correct, but silently fragile if F_emb assembly changes; ``mu`` is now
  cross-checked against ``mu_val``).  Exposing them directly would remove the
  reconstruction.  See :attr:`p_b`, :attr:`h_emb`.
* **No retained-AO index map under basis truncation** (paper Sec. 2.4).  Without
  it the adapter can only *trust*, not *assert*, that ``F_emb``, ``S``, and
  ``mo_coeffs_A_LL`` share a basis.  See :meth:`run_low_level`.
* **No RI-LVL three-index ERI export.**  ASI exports density/overlap/Hamiltonian
  but not the (truncated) ERIs -- the ``V^{-1/2}``-contracted ``M^P_{pq}`` with
  ``(pq|rs) = sum_P M^P_pq M^P_rs`` -- so FHI-aims cannot drive the WF-in-DFT half
  and the workflow stays PySCF-only.  See :class:`FHIaimsIntegrals`.
* **No A-fragment footing (h_core^A / E_nuc^A).**  The WF-path nuclear-frame rebase
  reaches through ``A_LL.atoms.calc.mol`` to a PySCF ``Mole`` because EmbASI exposes
  no backend-agnostic A-fragment one-electron operator / nuclear repulsion; the
  rebase is therefore PySCF-only.  See :meth:`_a_fragment_footing`.
* **Huzinaga as a constant offset when high/low xc match** (paper Sec. 2.1).
  ``H^{AB}_H`` then collapses to the supersystem low-level Hamiltonian, making
  ``P_B`` a constant matrix -- exportable after all, in exactly the WF-in-DFT
  regime.  See :meth:`__init__`.

Open-shell / unrestricted embedding
-----------------------------------

The *quantum* half is open-shell correct today (FCIDUMP -> circuit -> SQD -> energy and
spin-resolved RDMs, at any ``(n_alpha, n_beta)``).  The *embedding* half is not, and
finishing it **requires upstream EmbASI changes** -- it is not a local refactor.  Sites
are marked ``TODO(open-shell, needs EmbASI)`` inline; the blockers, verified against the
installed EmbASI:

1. **Per-spin occupied counts are computed but never exposed.**
   ``spade_localisation`` loops over ``ispin`` and derives
   ``max_occ_state = count_nonzero(occ_mat[ispin, ikpt])`` per channel, but that stays a
   local.  ``rot_evecs_occ_a`` is allocated as ``evecs.copy()`` (full MO width) and only
   its ``[:, spade_ncores:max_occ_state]`` columns are written, so the occupied count
   **cannot** be recovered from the returned array's shape.  Without it
   :attr:`EmbeddedOrbitals.n_occ_b` has no source and the downfold falls back to the
   restricted ``n_alpha = n_beta``.  See :meth:`build_orbitals`.
2. **No per-spin density ingest.**  :meth:`_as_spin_kpoint_array` hardcodes
   ``n_spin=1``, and EmbASI's only consumer indexes ``[0, 0]``
   (``qmcode_adapters.py:782``), so the ``n_spins > 1`` feedback path is unexercised
   upstream.  See :meth:`_as_spin_kpoint_array` and ``embedding.py``'s step 5.
3. **EmbASI's own open-shell TODOs.**  ``spade_localisation.py:116`` and ``:157`` carry
   ``# @TODOSPIN: Need to redefine occupancies`` on the density assembly that branches on
   ``n_spins == 1`` for the factor-2 occupancy; ``embedding.py`` carries
   ``# TODO: @SPIN AND K-POINT LOOP`` on the truncation path.

Until (1) and (2) land upstream, drive open shell through a FCIDUMP
(:func:`~embasi_qiskit_integration.hamiltonian.fcidump.read` carries
``(n_alpha, n_beta)`` exactly).  ``unrestricted=True`` raises rather than silently
returning a closed-shell answer.  Also still needed on this side once upstream lands: an
``(h1a, h1b)`` pair for a genuinely spin-dependent downfold, and validation of the
assembled open-shell energy against a UKS reference.  See :meth:`_as_ao_by_mo`.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

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

    def get_k(self, dm: np.ndarray) -> np.ndarray:
        """Bare HF exchange operator K[dm] (no J, no 0.5 factor), at a 2-occupancy dm.

        Used only by the APC active-space ranking (:meth:`ProjectionEmbeddingAdapter.
        build_orbitals_apc_concentric`), which needs K's diagonal separately from the
        combined ``veff_hf`` (King & Gagliardi, *J. Chem. Theory Comput.* **2021**, 17,
        7. eq. 18: ``(ia|ia) ~ 0.5 K_aa``).
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
        self.mf = mf_hl  # the same object passed to calc_base_hl
        self.mol = mf_hl.mol
        self._hf = mf_hl.mol.RHF()  # integral engine only; never kernel()'d
        self._density_fit = density_fit
        # A pyscf.df.DF, built lazily on the first eri_mo call. PySCF is untyped,
        # so this is Any rather than a precise DF type.
        self._df: Any = None

    def overlap(self) -> np.ndarray:
        return np.asarray(self.mol.intor("int1e_ovlp"))

    def hcore(self) -> np.ndarray:
        return np.asarray(self.mf.get_hcore())

    def veff_hl(self, dm) -> np.ndarray:
        return np.asarray(self.mf.get_veff(self.mol, np.asarray(dm)))

    def veff_hf(self, dm) -> np.ndarray:
        return np.asarray(self._hf.get_veff(self.mol, np.asarray(dm)))

    def get_k(self, dm) -> np.ndarray:
        return np.asarray(self._hf.get_k(self.mol, np.asarray(dm)))

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

    TODO(embasi-api): EmbASI/ASI does not export an RI-LVL three-index ERI tensor
    (the V^-1/2-contracted M^P_{pq}, so (pq|rs) = sum_P M^P_pq M^P_rs).  It exports
    density/overlap/Hamiltonian but not the (truncated) ERIs, so this backend
    cannot be built at all -- FHI-aims can drive only the DFT-in-DFT part of the
    workflow and WF-in-DFT stays PySCF-only.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError("RI-LVL ERI export not available through ASI yet")


# --------------------------------------------------------------------------- #
# Orbital bookkeeping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmbeddedOrbitals:
    """Orthonormal orbitals of subsystem A in the (possibly truncated) AO basis."""

    coeff: np.ndarray  # (nao, nmo_A) -- environment columns already dropped
    energy: np.ndarray  # (nmo_A,)
    n_occ: int  # occupied orbitals of A (doubly occupied when n_occ_b is None)
    inactive: np.ndarray  # column indices frozen into e_core
    active: np.ndarray  # column indices handed to the solver
    # Open shell only: the beta occupied count, when it differs from ``n_occ`` (which
    # is then the *alpha* count).  ``None`` (default) keeps the restricted reading --
    # ``n_occ`` doubly-occupied orbitals -- so every existing construction is unchanged.
    n_occ_b: int | None = None

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
    def is_open_shell(self) -> bool:
        """True when alpha and beta carry different occupied counts."""
        return self.n_occ_b is not None and self.n_occ_b != self.n_occ

    @property
    def n_active_electrons(self) -> int:
        """Total active electrons, implied by the partition; never supplied by the caller."""
        na, nb = self.n_active_electrons_spin
        return na + nb

    @property
    def n_active_electrons_spin(self) -> tuple[int, int]:
        """``(n_alpha, n_beta)`` in the active space.

        The restricted case (``n_occ_b is None``) splits a doubly-occupied count evenly,
        reproducing the old ``2 * (n_occ - n_inactive)`` total exactly. Open shell takes
        the two counts as given -- this is what replaces the ``// 2`` halving that could
        not express an open shell.

        ``inactive`` columns are frozen into ``e_core`` and are doubly occupied in both
        readings, so they subtract from each channel alike.
        """
        n_frozen = len(self.inactive)
        n_act_a = self.n_occ - n_frozen
        if self.n_occ_b is None:
            return (n_act_a, n_act_a)
        return (n_act_a, self.n_occ_b - n_frozen)

    def __str__(self) -> str:
        spin = ""
        if self.is_open_shell:
            na, nb = self.n_active_electrons_spin
            spin = f" [alpha={na}, beta={nb}]"
        return (
            f"CAS({self.n_active_orbitals}o, {self.n_active_electrons}e){spin} "
            f"from {self.n_occ} occupied + {self.coeff.shape[1] - self.n_occ} "
            f"virtual orbitals on subsystem A"
        )


def projection_energy_from_state(
    state: dict[str, Any],
    *,
    solver_energy: float,
    rdm1_ao: np.ndarray,
) -> "ProjectionEnergy":
    """Assemble paper Eq. 8 from an :meth:`ProjectionEmbeddingAdapter.export_state`
    snapshot, with no live EmbASI.

    :meth:`ProjectionEmbeddingAdapter.projection_energy` needs the live
    ``ProjectionEmbedding`` for two things only -- the ghosted-subsystem-A footing and
    EmbASI's low-level energies -- and ``export_state`` carries both.  So an
    out-of-process consumer can assemble the *same* energy from the snapshot, which is
    what keeps the footing shift, the Eq. 8 correction and the projector leak in one
    implementation instead of two.  Reimplementing them downstream is exactly how a
    ~4 Ha footing error gets reintroduced silently: the outer loop still converges on
    the density while carrying the offset in the energy.

    Args:
        state: an ``export_state`` mapping (or an ``np.load``ed ``.npz`` of one).
        solver_energy: the correlated solver's total energy for the embedded fragment.
        rdm1_ao: the solver 1-RDM lifted to the AO basis (see
            :meth:`ProjectionEmbeddingAdapter.rdm1_ao`).

    Returns:
        The same :class:`ProjectionEnergy` breakdown the in-process path returns.
    """

    def _array(key: str) -> np.ndarray:
        if key not in state:
            raise KeyError(
                f"state snapshot has no {key!r}; it must come from export_state() "
                f"(keys present: {sorted(state)})"
            )
        return np.asarray(state[key], dtype=float)

    dm_hl = np.asarray(rdm1_ao, dtype=float)
    p_b = _array("p_b")
    # The adapter's own `v_emb` (`h_emb - hcore - P_B`), exported directly -- NOT
    # EmbASI's `_v_emb_embasi`, which follows a different convention (it omits
    # subsystem A's nuclear-electron term).  Mixing the two is a silent energy error.
    v_emb = _array("v_emb")

    leak = float(np.einsum("ij,ji->", dm_hl, p_b))
    e_high_a = float(solver_energy) - np.einsum("ij,ji->", dm_hl, v_emb) - leak
    correction = float(np.einsum("ij,ji->", dm_hl - _array("dm_a_init"), v_emb))

    footing_shift = float(
        (float(_array("enuc_full")) - float(_array("enuc_a")))
        + np.einsum("ij,ji->", dm_hl, _array("hcore_full") - _array("hcore_a"))
    )
    e_high_a -= footing_shift

    return ProjectionEnergy(
        e_low_total=float(_array("e_low_total")),
        e_low_A=float(_array("e_low_a")),
        e_high_A=float(e_high_a),
        correction=correction,
        projector_leak=leak,
        footing_shift=footing_shift,
    )


@dataclass(frozen=True)
class ProjectionEnergy:
    """Term-by-term breakdown of paper Eq. 8."""

    e_low_total: float  # E_L[γ^A + γ^B]
    e_low_A: float  # E_L[γ^A]
    e_high_A: float  # E_H[Ψ̃^A], embedding pot. removed, rebased to E_low(A)'s footing
    correction: float  # tr[(γ̃^A - γ^A) v_emb]
    projector_leak: float  # tr[γ̃^A P_B], a numerical zero when clean
    footing_shift: float = 0.0  # nuclei/hcore rebasing applied to e_high_A (see below)

    @property
    def total(self) -> float:
        return self.e_low_total - self.e_low_A + self.e_high_A + self.correction


# Selector signature: (coeff, energy, n_occ) -> active column indices.  A Selector
# only PICKS canonical columns; it cannot rotate the virtual block.
Selector = Callable[[np.ndarray, np.ndarray, int], np.ndarray]

# VirtualLocalizer signature: (coeff, n_occ) -> (coeff_localised, sigma2).  Unlike a
# Selector, a localiser ROTATES the virtual block of ``coeff`` (occupied block and
# any non-A columns untouched) so that rotated virtual ``i`` carries fragment weight
# ``sigma2[i]`` (descending), and hands ``build_orbitals`` the per-virtual weight its
# gap cut runs on.  ``sigma2`` has length ``coeff.shape[1] - n_occ``.  The concrete
# SPADE localiser is ``selectors.spade_virtual_selector``.
VirtualLocalizer = Callable[[np.ndarray, int], "tuple[np.ndarray, np.ndarray]"]


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class ProjectionEmbeddingAdapter:
    """Wraps a live ``embasi.embedding.ProjectionEmbedding``."""

    def __init__(
        self,
        projection,  # embasi.embedding.ProjectionEmbedding
        integrals: AOIntegrals,
        *,
        mu: float = 1.0e6,  # level-shift parameter, paper Eq. 6
        env_eigenvalue_floor: float = _ENV_EIGENVALUE_FLOOR,
        unrestricted: bool = False,
    ):
        self.unrestricted = unrestricted
        projection_kind = getattr(projection, "projection", None)
        if projection_kind != "level-shift":
            # P_B and v_emb are now read directly from construct_embedding_potential
            # (see run_low_level), so the export no longer reconstructs P_B from the
            # level-shift closed form mu*S*gamma^B*S -- the earlier hard guard was
            # protecting that reconstruction, not the physics.  A non-level-shift
            # projection whose construct_embedding_potential yields a usable
            # (gamma^A, gamma^B, S, v_emb, P_B) can therefore flow through.  The
            # paper (Sec. 2.1) shows H^AB_H collapses to the supersystem low-level
            # Hamiltonian when the high/low xc match, making P_B constant and
            # exportable in exactly the WF-in-DFT regime.  Correctness for such a
            # projection is UNVERIFIED in-repo (the only regression case, methanol,
            # is level-shift), so this is a warning rather than a silent pass.
            warnings.warn(
                f"projection={projection_kind!r} is untested for external export; "
                "only 'level-shift' has an in-repo regression case. P_B/v_emb are "
                "read directly from construct_embedding_potential, so the export no "
                "longer depends on the level-shift closed form -- but verify the "
                "assembled energy against a known reference before trusting it.",
                stacklevel=2,
            )
        self.p = projection
        self.ints = integrals
        self.mu = mu
        self._floor = env_eigenvalue_floor

        # Populated by run_low_level(); None until then, hence the optional type.
        self._dm_a: np.ndarray | None = None  # γ^A (localized, low level), AO, 2-occupancy
        self._dm_a_init: np.ndarray | None = None  # γ^A (localized, low level), AO, 2-occupancy
        self._dm_b: np.ndarray | None = None  # γ^B (environment, frozen), AO
        self._fock: np.ndarray | None = None  # F_emb
        self._s: np.ndarray | None = None
        self._p_b: np.ndarray | None = None  # P_B, read from construct_embedding_potential
        # EmbASI's v_emb (= H^AB - H^A); on EmbASI's footing, i.e. it does NOT
        # carry subsystem A's nuclear-electron term (that sits in
        # A_LL.hamiltonian_estat_plus_xc) -- see the v_emb property.
        self._v_emb_embasi: np.ndarray | None = None

    # ---------------- low-level embedding ---------------- #
    def run_low_level_a_only(
        self, dma_in: np.ndarray | None = None, dmb_in: np.ndarray | None = None
    ) -> None:

        wrapped_dma_in = None if dma_in is None else self._as_spin_kpoint_array(dma_in)

        self.p.A_LL.run_noscf(dm_in=wrapped_dma_in)

        # EmbASI returns SpinKpointArray objects (leading (nspin, nkpt) axes)
        # holding real restricted data in a complex128 dtype; _as_ao_matrix
        # squeezes the length-1 leading axes and drops the (asserted-negligible)
        # imaginary part, so everything downstream (einsum, sla.eigh, veff) sees
        # a bare 2-occupancy real (nao, nao) matrix.
        self._dm_a = dma_in
        self._dm_b = dmb_in
        self._s = self.ints.overlap()

        self._v_emb_embasi = self._v_emb_embasi
        self._p_b = self._p_b

        self._assemble_fock_a_only()

    def _assemble_fock_a_only(self) -> None:
        """Assemble ``F_emb`` from the A_LL one-electron blocks plus ``v_emb``/``P_B``.

        Split out of :meth:`run_low_level_a_only` so a *restored* adapter (one whose
        ``v_emb``/``P_B`` came from :meth:`restore_state` rather than from a prior
        ``run_low_level`` in the same process) reassembles the Fock through exactly
        the same arithmetic.  Sharing the code is the point: an out-of-process
        embedding loop must not get a second, subtly different downfold.

        Assembles it exactly as EmbASI's ``construct_embedded_fock`` does, from the
        A_LL one-electron blocks plus the exported ``v_emb`` and ``P_B``.  Reading
        these off the same A_LL object EmbASI integrated keeps the downfold
        bit-identical to the previous ``construct_embedded_fock()`` path.
        """
        h_kin_a = self._as_ao_total(self.p.A_LL.hamiltonian_kinetic)
        h_estat_xc_a = self._as_ao_total(self.p.A_LL.hamiltonian_estat_plus_xc)
        self._fock = h_kin_a + h_estat_xc_a + self._v_emb_embasi + self._p_b
        self._validate_densities()

        # Cross-check our level-shift mu against the value EmbASI used inside the
        # SCF.  We now read P_B directly, but self.mu still parameterises the
        # adapter (e.g. the p_b property fallback and meta), so a silent mismatch
        # would be confusing; keep the loud check.
        mu_embasi = getattr(self.p, "mu_val", None)
        if mu_embasi is not None and not np.isclose(float(mu_embasi), self.mu, rtol=1e-9, atol=0.0):
            raise ValueError(
                f"level-shift mu mismatch: adapter mu={self.mu:g} but EmbASI "
                f"used mu_val={float(mu_embasi):g}; the exported P_B was built "
                f"with mu_val, not the adapter's mu"
            )

    def run_low_level(
        self,
        dma_in: np.ndarray | None = None,
        dmb_in: np.ndarray | None = None,
        a_nmos: int | None = None,
    ) -> None:
        """Drive EmbASI: supersystem SCF, SPADE/PM localisation, F_emb."""
        # TODO(embasi-api): EmbASI does not expose the retained-AO index array
        # under basis truncation (paper Sec. 2.4, threshold tau).  With
        # truncation on, every matrix below must return in the *same* truncated
        # AO ordering and mo_coeffs_A_LL be sliced to match; without the index
        # map the adapter can only *trust* that ints.overlap() and F_emb share a
        # basis, not assert it -- a mismatch would corrupt every downstream
        # contraction silently.
        # Drive EmbASI through ``construct_embedding_potential``, which returns the
        # embedding potential ``v_emb`` and the projector ``P_B`` *directly* (rather
        # than only the assembled ``F_emb`` from ``construct_embedded_fock``).  We no
        # longer reconstruct ``P_B`` as ``mu * S gamma^B S`` by subtraction -- we read
        # EmbASI's own projector.  ``F_emb`` is then assembled exactly as EmbASI's
        # ``construct_embedded_fock`` does it, so the downfold is unchanged:
        #     F_emb = h_kin^A + h_estat_xc^A + v_emb + P_B
        # (see embasi.embedding.ProjectionEmbedding.construct_embedded_fock).
        wrapped_dma_in = None if dma_in is None else self._as_spin_kpoint_array(dma_in)
        wrapped_dmb_in = None if dmb_in is None else self._as_spin_kpoint_array(dmb_in)
        # EmbASI wants dmab_in as a SpinKpointArray, not the plain (nao, nao)
        # density the outer loop feeds back -- wrap it (mirror of the _as_ao_matrix
        # squeeze on the way out) so the multi-cycle loop runs against real EmbASI.
        dm_a, dm_b, _overlap, v_emb_embasi, p_b_embasi = self.p.construct_embedding_potential(
            dma_in=wrapped_dma_in, dmb_in=wrapped_dmb_in, a_nspade_mos=a_nmos
        )

        # EmbASI returns SpinKpointArray objects (leading (nspin, nkpt) axes)
        # holding real restricted data in a complex128 dtype; _as_ao_matrix
        # squeezes the length-1 leading axes and drops the (asserted-negligible)
        # imaginary part, so everything downstream (einsum, sla.eigh, veff) sees
        # a bare 2-occupancy real (nao, nao) matrix.
        #
        # TODO(open-shell, needs EmbASI): `_as_ao_total` SUMS the spin axis, so an
        # unrestricted run collapses the pair here (blocker (2)); keeping it would need
        # somewhere to store the halves plus per-channel Fock/veff consumers.  Note
        # `dm_a`/`dm_b` are subsystems A and B, NOT alpha/beta -- each carries its own
        # spin axis, so an unrestricted run has four blocks.
        self._dm_a = self._as_ao_total(dm_a)
        if self._dm_a_init is None:
            self._dm_a_init = self._as_ao_total(dm_a)
        self._dm_b = self._as_ao_total(dm_b)
        self._s = self.ints.overlap()

        # v_emb / P_B read straight from EmbASI (no longer reconstructed by
        # subtraction).  P_B may be None for projection modes that build it inside
        # the SCF (huzinaga-sc); the __init__ guard already rejects those, so a
        # None here is an upstream contract change and we fail loudly.
        if p_b_embasi is None:
            raise ValueError(
                "construct_embedding_potential returned P_B=None; only "
                "projection='level-shift' (constant projector) can be exported"
            )
        self._p_b = self._as_ao_total(p_b_embasi)
        self._v_emb_embasi = self._as_ao_total(v_emb_embasi)

        self._assemble_fock_a_only()

    # ---------------- cross-process state seam ---------------- #
    # An out-of-process embedding loop (one round per OS process, e.g. a workflow
    # engine driving `max_cycles=1` repeatedly) cannot carry a live adapter across a
    # round boundary: it holds a live ProjectionEmbedding and two PySCF KS objects,
    # none of them picklable.  What it *can* carry is the handful of arrays the
    # downfold needs.  These two methods are that seam.
    #
    # Deliberately not property setters: the arrays are only meaningful as a *set*
    # (F_emb is assembled from v_emb + P_B against a specific basis and geometry), so
    # pairing a snapshot with a mismatched adapter must be refused, not silently
    # assembled into a wrong energy.  A fingerprint is what makes that refusal possible.

    _STATE_ARRAYS = ("_dm_a", "_dm_a_init", "_dm_b", "_fock", "_s", "_p_b", "_v_emb_embasi")

    def state_fingerprint(self) -> dict[str, Any]:
        """Identify the adapter a snapshot may be restored into.

        Everything here changes the *meaning* of the exported arrays: a different
        basis or geometry makes them a different-sized (or same-sized but wrong)
        operator, and a different projection kind or spin treatment changes how
        ``P_B``/``v_emb`` were built.  Compared as a whole by :meth:`restore_state`.
        """
        mol = getattr(self.ints, "mol", None)
        mf = getattr(self.ints, "mf", None)
        fingerprint: dict[str, Any] = {
            "nao": int(np.asarray(self.ints.overlap()).shape[-1]),
            "projection": str(getattr(self.p, "projection", None)),
            "unrestricted": bool(self.unrestricted),
            "mu": float(self.mu),
            # The high-level functional and the DF setting both change what veff_hl
            # subtracts, so a snapshot paired across a change in either assembles a
            # wrong energy while every array keeps its shape.  Verified live: without
            # these an xc_hl mismatch was accepted.
            "xc_hl": str(getattr(mf, "xc", None)),
            "density_fit": str(getattr(self.ints, "_density_fit", None)),
        }
        if mol is not None:
            # Round coordinates before hashing: PySCF stores them in bohr as floats,
            # and a re-read geometry can differ in the last bit without being a
            # different molecule.
            coords = np.round(np.asarray(mol.atom_coords(), dtype=float), 8)
            fingerprint["basis"] = str(mol.basis)
            fingerprint["natm"] = int(mol.natm)
            fingerprint["charge"] = int(mol.charge)
            fingerprint["spin"] = int(mol.spin)
            fingerprint["geometry_hash"] = hashlib.sha256(
                np.ascontiguousarray(coords).tobytes()
                + str([mol.atom_symbol(i) for i in range(mol.natm)]).encode()
            ).hexdigest()[:16]
        return fingerprint

    def low_level_diagnostics(self) -> dict[str, Any]:
        """What can be said about the low-level SCF's convergence, honestly.

        An out-of-process outer loop needs this per round: an unconverged low-level
        SCF is a fixed bias in a single shot, but in a *k*-round loop it is a
        per-round term that a plain ``|ΔE|`` stop criterion cannot distinguish from
        genuine slow convergence.  Surfacing it means an unconverged round is visible
        in the round's diagnostics rather than mistaken for a plateau.

        **EmbASI exposes no convergence flag for its own subsystem SCF** (``A_LL``
        carries only ``no_scf``), and its low-level SCF is observed not to converge
        even on a 6-atom sto-3g monomer -- it prints ``SCF not converged.`` and
        continues.  So this reports what is actually available rather than
        synthesising a flag: the two low-level energies the energy assembly reads, and
        the PySCF high-level mean field's own convergence state.  ``a_ll_no_scf``
        distinguishes a genuine SCF from the ``run_noscf`` path
        ``run_low_level_a_only`` uses.

        Returns:
            A JSON-serialisable mapping; every value may be ``None`` when the
            underlying object does not expose it.
        """
        mf = getattr(self.ints, "mf", None)
        diagnostics: dict[str, Any] = {
            "a_ll_no_scf": getattr(self.p.A_LL, "no_scf", None),
            "mf_hl_converged": getattr(mf, "converged", None),
            "mf_hl_conv_tol": getattr(mf, "conv_tol", None),
            "mf_hl_max_cycle": getattr(mf, "max_cycle", None),
        }
        try:
            e_low_a, e_low_total = self._low_level_energies()
            diagnostics["e_low_A"] = e_low_a
            diagnostics["e_low_total"] = e_low_total
        except Exception:  # noqa: BLE001 -- diagnostics must never break a run
            diagnostics["e_low_A"] = None
            diagnostics["e_low_total"] = None
        return diagnostics

    def export_state(self) -> dict[str, Any]:
        """Snapshot the low-level state needed to reassemble ``F_emb`` elsewhere.

        Call after :meth:`run_low_level` (the arrays are ``None`` before it).  The
        returned mapping is plain arrays and scalars, so it serialises with
        ``np.savez`` and survives a process boundary.

        ``_dm_a`` is in the snapshot for three reasons beyond re-deriving the Fock:
        an out-of-process outer loop needs it as the DIIS input vector, as the
        ``prev_dm_a`` of its ``max|Δγ^A|`` diagnostic, and -- crucially -- it must be
        the density the *previous* cycle fed in, not the one this round's fresh
        supersystem SCF just wrote.  A consumer that reads the post-SCF value instead
        builds its DIIS residual against the wrong reference, which is silent.

        ``_dm_a_init`` matters for a subtler reason: :meth:`projection_energy` computes
        the Eq. 8 correction as ``tr[(γ̃^A - γ^A_init) v_emb]`` against it, and it is
        set once, on the first ``run_low_level`` ever.  A fresh process re-runs the SCF
        and would reset it to *this* round's density, silently zeroing the drift the
        correction is meant to measure.  So it is carried from round 0, not recomputed.
        """
        missing = [n for n in self._STATE_ARRAYS if getattr(self, n) is None]
        if missing:
            raise RuntimeError(
                f"export_state() before run_low_level(): {missing} are unset. The "
                f"snapshot is only meaningful after the supersystem SCF has run."
            )
        state: dict[str, Any] = {
            name.lstrip("_"): np.asarray(getattr(self, name)) for name in self._STATE_ARRAYS
        }
        # The ghosted-subsystem-A footing and the two low-level energies are the only
        # things `projection_energy` reads off the *live* ProjectionEmbedding
        # (`_a_fragment_footing` reaches through `A_LL.atoms.calc.mol`;
        # `_low_level_energies` reads EmbASI's own scalars).  Both reduce to one array
        # and three scalars, so exporting them is what lets a separate process assemble
        # the energy from a snapshot alone -- no live EmbASI, no second SCF.
        hcore_a, enuc_a = self._a_fragment_footing()
        e_low_ab, e_low_a = self._low_level_energies()
        # `v_emb` (and `h_emb`) are exported as arrays rather than left to be rebuilt
        # downstream: the adapter's `v_emb` is `h_emb - hcore - P_B` and `h_emb` needs
        # `veff_hl(gamma^A)`, i.e. a live PySCF mean field.  It also uses a *different
        # convention* from EmbASI's exported `_v_emb_embasi` (which omits subsystem A's
        # nuclear-electron term), so a consumer reconstructing it from the snapshot is
        # one convention slip away from a silently wrong energy.
        state["v_emb"] = np.asarray(self.v_emb)
        state["h_emb"] = np.asarray(self.h_emb)
        state["hcore_a"] = np.asarray(hcore_a)
        state["hcore_full"] = np.asarray(self.ints.hcore())
        state["enuc_a"] = np.asarray(float(enuc_a))
        state["enuc_full"] = np.asarray(float(self.ints.energy_nuc()))
        state["e_low_total"] = np.asarray(e_low_ab)
        state["e_low_a"] = np.asarray(e_low_a)
        state["fingerprint"] = self.state_fingerprint()
        return state

    def restore_state(self, state: dict[str, Any], *, strict: bool = True) -> None:
        """Install a snapshot from :meth:`export_state` into this adapter.

        **Call this *after* a fresh :meth:`run_low_level`, not instead of one.** The
        A_LL one-electron blocks (``hamiltonian_kinetic``,
        ``hamiltonian_estat_plus_xc``) live in EmbASI proper and are only populated
        once EmbASI has integrated the subsystem; they depend on geometry and basis,
        not on cycle history, so re-running the SCF is how they get there.  Restoring
        then overwrites that fresh SCF's ``v_emb``/``P_B``/densities with the carried
        ones, and :meth:`_assemble_fock_a_only` rebuilds ``F_emb`` from the pair.
        Restoring into an adapter that never ran will raise from the assembly.

        Args:
            state: an :meth:`export_state` mapping (or an ``np.load``ed ``.npz`` of one).
            strict: verify the fingerprint against this adapter and raise on mismatch.
                Only pass ``False`` when a *deliberate* re-pairing is intended; a
                mismatched basis or geometry otherwise assembles a wrong ``F_emb``
                and reports a plausible, wrong energy.

        Raises:
            KeyError: the snapshot is missing an array.
            ValueError: ``strict`` and the fingerprint disagrees with this adapter.
        """
        arrays = {}
        for name in self._STATE_ARRAYS:
            key = name.lstrip("_")
            if key not in state:
                raise KeyError(f"state snapshot has no {key!r}; keys: {sorted(state)}")
            arrays[name] = np.asarray(state[key], dtype=float)

        if strict:
            self._check_fingerprint(state)

        # nao is checked even when not strict: a shape mismatch is not a judgement
        # call, it is an array that cannot be contracted with this adapter's.
        nao = int(np.asarray(self.ints.overlap()).shape[-1])
        for name, arr in arrays.items():
            if arr.shape[-2:] != (nao, nao):
                raise ValueError(
                    f"snapshot {name.lstrip('_')!r} has shape {arr.shape}, but this "
                    f"adapter's basis has nao={nao}; refusing to assemble F_emb from "
                    f"a different basis"
                )

        for name, arr in arrays.items():
            setattr(self, name, arr)
        self._assemble_fock_a_only()

    def _check_fingerprint(self, state: dict[str, Any]) -> None:
        """Raise if ``state``'s fingerprint disagrees with this adapter's."""
        stored = state.get("fingerprint")
        if stored is None:
            raise ValueError(
                "state snapshot carries no 'fingerprint', so it cannot be verified "
                "against this adapter; pass strict=False only if the pairing is "
                "deliberate and independently checked"
            )
        # np.savez round-trips a dict through a 0-d object array; unwrap it.
        if isinstance(stored, np.ndarray):
            stored = stored.item()
        current = self.state_fingerprint()
        differing = {
            k: (stored.get(k), current.get(k))
            for k in sorted(set(stored) | set(current))
            if stored.get(k) != current.get(k)
        }
        if differing:
            detail = ", ".join(
                f"{k}: snapshot={s!r} adapter={c!r}" for k, (s, c) in differing.items()
            )
            raise ValueError(
                f"state snapshot does not match this adapter ({detail}); restoring it "
                f"would assemble F_emb from a different system and report a plausible "
                f"but wrong energy"
            )

    # ---------------- low-level state accessors ---------------- #
    # ``_dm_a``/``_dm_b``/``_fock``/``_s`` are only populated by run_low_level().
    # These accessors turn "used before the embedding ran" from an AttributeError
    # on None deep inside a contraction into one clear message, and give the type
    # checker the non-optional arrays the numerics below require.

    def _require(self, value: np.ndarray | None, name: str) -> np.ndarray:
        """Return ``value``, or explain that :meth:`run_low_level` has not run."""
        if value is None:
            raise RuntimeError(
                f"{name} is not available yet: call run_low_level() before using "
                "the embedding operators, orbitals, or energies"
            )
        return value

    @property
    def _s_arr(self) -> np.ndarray:
        """The AO overlap matrix S (requires :meth:`run_low_level`)."""
        return self._require(self._s, "the AO overlap matrix")

    @property
    def _dm_a_arr(self) -> np.ndarray:
        """The localized subsystem-A density γ^A (requires :meth:`run_low_level`)."""
        return self._require(self._dm_a, "the subsystem-A density")

    @property
    def _dm_a_arr_init(self) -> np.ndarray:
        """The localized subsystem-A density γ^A (requires :meth:`run_low_level`)."""
        return self._require(self._dm_a_init, "the subsystem-A density")

    @property
    def _dm_b_arr(self) -> np.ndarray:
        """The frozen environment density γ^B (requires :meth:`run_low_level`)."""
        return self._require(self._dm_b, "the environment density")

    @property
    def _fock_arr(self) -> np.ndarray:
        """The embedded Fock matrix F_emb (requires :meth:`run_low_level`)."""
        return self._require(self._fock, "the embedded Fock matrix")

    def _as_ao_total(self, m) -> np.ndarray:
        """Spin-summed ``(nao, nao)`` block, accepting a length-2 spin axis when opted in.

        Thin wrapper over :meth:`_as_ao_matrix` (which stays strictly restricted, and is
        pinned as such by ``tests/test_adapter_helpers.py``): with ``unrestricted=True``
        a genuine ``(2, ...)`` spin axis is summed into the total the density and Fock
        consumers want, rather than refused.  Per-channel access is
        :meth:`_as_ao_matrix_spin`.
        """
        arr = np.asarray(m)
        if self.unrestricted:
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] == 2:
                a, b = self._as_ao_matrix_spin(arr)
                return np.ascontiguousarray(a + b)
        return self._as_ao_matrix(arr)

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
                    f"(leading axis has length {m.shape[0]}, expected 1). "
                    "Pass unrestricted=True to the adapter for a length-2 spin axis."
                )
            m = m[0]
        # Real-part extraction is shared with the per-spin path; see _real.
        return ProjectionEmbeddingAdapter._real(m)

    def _as_ao_matrix_spin(self, m) -> tuple[np.ndarray, np.ndarray]:
        """Split an EmbASI ``(nspin, nkpt, nao, nao)`` block into ``(alpha, beta)``.

        The spin-resolved counterpart of :meth:`_as_ao_matrix`. A length-1 spin axis
        (restricted data) is returned as two halves of the doubly-occupied total, so a
        caller written against this always sees a consistent ``alpha + beta == total``.

        Raises:
            NotImplementedError: unless ``unrestricted=True``, or if the spin axis is
                neither 1 nor 2 long.
        """
        if not self.unrestricted:
            raise NotImplementedError("per-spin blocks require unrestricted=True on the adapter")
        arr = np.asarray(m)
        while arr.ndim > 4:  # drop any extra leading axis EmbASI may add
            arr = arr[0]
        if arr.ndim == 4:  # (nspin, nkpt, nao, nao) -> single k-point
            if arr.shape[1] != 1:
                raise NotImplementedError(f"multi-k not wired up (n_kpoints={arr.shape[1]})")
            arr = arr[:, 0]
        if arr.ndim == 2:  # already squeezed: restricted total
            total = self._real(arr)
            return 0.5 * total, 0.5 * total
        if arr.ndim != 3:
            raise ValueError(f"expected a (nspin, nao, nao) block, got shape {arr.shape}")
        if arr.shape[0] == 1:
            total = self._real(arr[0])
            return 0.5 * total, 0.5 * total
        if arr.shape[0] != 2:
            raise NotImplementedError(f"spin axis has length {arr.shape[0]}, expected 1 or 2")
        return self._real(arr[0]), self._real(arr[1])

    @staticmethod
    def _real(m: np.ndarray) -> np.ndarray:
        """Drop a numerically-zero imaginary part, loudly if it is not zero.

        Same contract as the complex handling in :meth:`_as_ao_matrix`, factored out so
        the per-spin path applies it identically.
        """
        m = np.asarray(m)
        if np.iscomplexobj(m):
            max_imag = float(np.abs(m.imag).max()) if m.size else 0.0
            if max_imag > _IMAG_TOL:
                raise ValueError(
                    f"matrix has a non-negligible imaginary part (max |imag| = "
                    f"{max_imag:.2e} > {_IMAG_TOL:.0e})"
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
        # TODO(open-shell, needs EmbASI): `n_spin=1` is hardcoded -- the write-side half
        # of blocker (2).  An unrestricted loop wants `SpinKpointArray({(0, 0): a,
        # (1, 0): b}, n_spin=2, ...)`, but verify the key convention first: EmbASI's only
        # reader indexes `[0, 0]`, so the `n_spin=2` path is unexercised upstream.
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
        if c.ndim == 3 and c.shape[0] == 2 and getattr(self, "unrestricted", False):
            # Spin-resolved caller asked for one set through the restricted accessor:
            # the alpha channel is the conventional representative.  Per-channel access
            # is _as_ao_by_mo_spin.
            c = c[0]
        if c.ndim == 3:  # (nspin, ., .) -- closed shell only
            # Open-shell embedding is a deferred package rewrite (see the
            # module docstring): everything downstream assumes a 2-occupancy
            # density and a spin-restricted downfold, so supporting open shells
            # means separate alpha/beta orbital sets, an (h1a, h1b) pair, and
            # SQD's spin-symmetry handling on the solver side.
            if c.shape[0] != 1:
                raise NotImplementedError(
                    "open-shell embedding not wired up (spin axis has length "
                    f"{c.shape[0]}); pass unrestricted=True to the adapter"
                )
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
        nao = self._s_arr.shape[0]
        if c.shape[0] != nao and c.shape[1] == nao:
            c = c.T
        # Definitive check rather than a guess: C^T S C == 1
        gram = c.T @ self._s_arr @ c
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
        """Level-shift projector, paper Eq. 6.

        Read directly off EmbASI's ``construct_embedding_potential`` in
        :meth:`run_low_level` (``levelshift_projector(gamma^B, S, mu_val)``
        upstream), rather than reconstructed as ``mu * S gamma^B S``.  Reading it
        removes the assumption that EmbASI's projector has exactly that closed form
        -- the cross-check on ``mu_val`` in :meth:`run_low_level` now guards only
        that ``self.mu`` and the exported projector agree.
        """
        return self._require(self._p_b, "the level-shift projector P_B")

    @property
    def h_emb(self) -> np.ndarray:
        """h_core + v_emb + P_B: the one-body operator the solver must see.

        ``F_emb - veff_hl(gamma^A)`` strips the high-level mean field EmbASI folded
        into ``F_emb`` back off, leaving the one-body operator on the adapter's
        PySCF ``h_core`` footing (the footing the downfold and the bare-electronic
        solver Hamiltonian use).  ``F_emb`` itself is now assembled in
        :meth:`run_low_level` from EmbASI's *exported* ``v_emb`` and ``P_B`` (plus
        the ``A_LL`` one-electron blocks), so this is no longer the inverse of an
        opaque ``construct_embedded_fock`` -- the projector it subtracts back out
        via :attr:`p_b` is EmbASI's own exported ``P_B``, not a reconstruction.
        """
        return self._fock_arr - self.ints.veff_hl(self._dm_a_arr)

    @property
    def v_emb(self) -> np.ndarray:
        """Eq. 3 embedding potential, on the adapter's PySCF ``h_core`` footing.

        NOT the same object as EmbASI's exported ``v_emb`` (:attr:`_v_emb_embasi`).
        EmbASI's ``v_emb = H^AB - H^A`` deliberately omits subsystem A's
        nuclear-electron term (it lives in ``A_LL.hamiltonian_estat_plus_xc``), a
        convention inherited from FHI-aims where the Hartree and nuclear-electron
        pieces are bundled.  The adapter's ``v_emb`` is instead the potential
        *relative to the PySCF ``h_core``* the solver Hamiltonian is built on, so it
        is recovered here as ``h_emb - h_core^PySCF - P_B``.  The two conventions
        differ by exactly that ``h_ne^A`` bookkeeping, which is why the exported
        ``_v_emb_embasi`` is used only to *assemble* ``F_emb`` (alongside the
        matching ``A_LL`` one-electron blocks) and never substituted here.
        """
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
        s = self._s_arr
        c_b = self.mo_b_ll  # (nao, n_occ_B), S-orthonormal
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
        fs = q.T @ self._fock_arr @ q
        eps, cc = np.linalg.eigh(fs)
        return eps, q @ cc

    def build_orbitals(
        self,
        *,
        n_frozen_occ: int = 0,
        n_virtual: int | None = None,
        selector: Selector | None = None,
        virtual_localizer: VirtualLocalizer | None = None,
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

        ``virtual_localizer`` (mutually exclusive with ``selector``) is the SPADE
        alternative to a column-index :data:`Selector`.  A localiser *rotates* the
        virtual block -- the kept orbitals become linear combinations of the
        canonical virtuals, not columns of ``C`` -- so it cannot be expressed as an
        index selector.  We apply it here (``selectors.spade_virtual_selector``
        builds one): rotate the virtual block in place, cut on the returned σ² gap
        (the same ``_gap_cut`` the Mulliken :func:`mulliken_selector` uses), and
        keep the occupied block untouched.  Rotating *within* the A-virtual block
        preserves S-orthonormality and keeps ``P_B`` invisible in the active space
        (``span`` is unchanged), so the downfold and energy assembly -- which treat
        the active columns as an opaque S-orthonormal spanning set, never as
        canonical MOs -- are unaffected.  The rotated virtuals are no longer F_emb
        eigenvectors, so their ``energy`` entries are set to ``NaN`` (nothing reads a
        virtual eigenvalue; a Fock eigenvalue would be a lie for a rotated orbital).
        """
        if selector is not None and virtual_localizer is not None:
            raise ValueError(
                "pass either selector or virtual_localizer, not both: a localiser "
                "rotates and cuts the virtual block itself"
            )
        if restrict_to_a:
            eps, c = self._eigh_subsystem_a()
        else:
            eps, c = sla.eigh(self._fock_arr, self._s_arr)
            keep = eps < self._floor  # level shift removes subsystem B
            eps, c = eps[keep], c[:, keep]

        n_occ = self.mo_a_ll.shape[1]  # inferred, never passed in

        # TODO(open-shell, needs EmbASI): this line is blocker (1) in the module
        # docstring -- `_as_ao_by_mo` keeps only the alpha channel, so one `n_occ` is all
        # that survives and `EmbeddedOrbitals` falls back to `n_alpha = n_beta`.  Once
        # EmbASI exposes the per-spin occupied counts, read them here and pass `n_occ_b`
        # into every `EmbeddedOrbitals(...)` built below.

        n_virt_total = c.shape[1] - n_occ
        n_virt = n_virt_total if n_virtual is None else min(n_virtual, n_virt_total)
        if not 0 <= n_frozen_occ < n_occ:
            raise ValueError(f"n_frozen_occ={n_frozen_occ} outside [0, {n_occ}) for subsystem A")

        if virtual_localizer is not None:
            # SPADE: rotate the virtual block so each rotated virtual carries a
            # definite fragment weight sigma2, then cut on the sigma2 gap exactly as
            # the Mulliken path cuts on population.  c is replaced by the rotated
            # coefficients; rotated-virtual eigenvalues are meaningless (set NaN).
            from embasi_qiskit_integration.selectors import _gap_cut

            c, sigma2 = virtual_localizer(c, n_occ)
            eps = eps.copy()
            eps[n_occ:] = np.nan
            keep = _gap_cut(
                sigma2,
                gap_tol=getattr(virtual_localizer, "gap_tol", 1.0e-3),
                max_virtual=getattr(virtual_localizer, "max_virtual", None),
                min_virtual=getattr(virtual_localizer, "min_virtual", 0),
            )
            active = np.arange(n_frozen_occ, n_occ + keep)
        elif selector is not None:
            # Index-selector hook for AVAS / MP2-NOON / Mulliken concentric-cut
            # selection: it PICKS canonical columns and returns their indices (it
            # cannot rotate the virtual block -- that is the virtual_localizer path
            # above, which is the true SPADE singular-value-gap cut).
            active = np.asarray(selector(c, eps, n_occ), dtype=int)
            if n_frozen_occ:
                active = active[active >= n_frozen_occ]
                if active.size == 0:
                    raise ValueError(
                        f"n_frozen_occ={n_frozen_occ} froze every orbital the selector "
                        "kept, leaving an empty active space"
                    )
        else:
            active = np.arange(n_frozen_occ, n_occ + n_virt)

        # from pyscf.tools import cubegen
        # for occ_idx in np.arange(0,n_occ):
        #    print(occ_idx)
        #    cubegen.orbital(self.p.A_LL.atoms.calc.method.mol, f'orbital_occ_{occ_idx}.cube', c[:, occ_idx])
        #
        # for virt_idx in np.arange(n_occ,n_occ+n_virt):
        #    print(virt_idx)
        #    cubegen.orbital(self.p.A_LL.atoms.calc.method.mol, f'orbital_virt_{virt_idx}.cube', c[:, virt_idx])

        inactive = np.array([i for i in range(n_occ) if i not in set(active.tolist())], dtype=int)
        return EmbeddedOrbitals(coeff=c, energy=eps, n_occ=n_occ, inactive=inactive, active=active)

    def build_orbitals_apc_concentric(
        self,
        *,
        fragment_ao: np.ndarray,
        n_shells: int = 0,
        max_size: int | tuple[int, int],
        fixed: bool = False,
        restrict_to_a: bool = True,
    ) -> EmbeddedOrbitals:
        """Concentric localization for locality, then APC to rank and truncate.

        Two-stage active-space construction (King & Gagliardi, *J. Chem. Theory
        Comput.* **2021**, 17, 7387, doi:10.1021/acs.jctc.1c00037):

        1. :func:`~embasi_qiskit_integration.selectors.concentric_localization_selector`
           rotates the virtual block into fragment-coupled shells -- this answers
           *which virtuals are spatially/electronically relevant to the embedded
           region*, the question the level-shift embedding leaves open (it removes
           subsystem B entirely, but says nothing about which of subsystem A's
           virtuals matter for the active atoms specifically). Its own shell count
           (``n_shells``) sets the candidate pool; it is called uncapped
           (``max_virtual=None``) since APC, not CL's own cap, does the truncation.
        2. APC (:func:`~embasi_qiskit_integration.selectors.apc_pair_coefficients`
           / ``apc_orbital_entropies`` / ``apc_active_space``) then ranks *every*
           candidate orbital -- occupied and virtual together -- by an approximate
           multiconfigurational pair-coefficient entropy, and truncates to
           ``max_size``. Unlike every other selector in this package, APC can drop
           occupied candidates -- it supersedes ``n_frozen_occ`` for this active
           space; occupied freezing is a ranking outcome, not a caller-set count.

        CL's kept virtuals are rotated, not Fock eigenvectors, so ``ε_a`` in APC's
        eq. 19 is recomputed as the expectation value ``diag(C^T F_emb C)`` in
        the *current* (possibly rotated) basis rather than read off
        ``EmbeddedOrbitals.energy`` (which CL already sets to ``NaN`` for exactly
        this reason -- see :meth:`build_orbitals`). The exchange diagonal is
        computed the same way against ``self.ints.get_k`` at the full-A 2-occupancy
        density, mirroring the inactive-core downfold's ``dm_in`` in
        :meth:`embedded_hamiltonian`.

        Outer-loop stability: this selection is recomputed every self-consistency
        cycle from the current ``F_emb`` (see ``EmbeddingWorkflow._run_outer_loop``).
        ``fixed=True`` (with a ``(nelec, norb)`` ``max_size``) pins the selection to
        exactly that size every cycle; the default dynamic drop-until-budget
        (``fixed=False``) can flip which near-tied orbital is dropped as the Fock
        matrix drifts cycle to cycle, changing the active-space size -- and hence
        the qubit count -- mid-run. Use ``fixed=True`` for anything feeding
        ``_run_outer_loop`` with ``max_cycles > 1``.

        Args:
            fragment_ao: AO indices of the active-fragment atoms (as
                :func:`~embasi_qiskit_integration.selectors.fragment_ao_indices`).
            n_shells: CL shell expansions after shell 0 (the locality pre-filter's
                own accuracy knob; see :func:`~embasi_qiskit_integration.selectors.
                concentric_localization_selector`).
            max_size: APC's active-space budget, forwarded to
                :func:`~embasi_qiskit_integration.selectors.apc_active_space`: an
                orbital-count ceiling (int; ``Chooser``'s convention, not an
                ``N_CSF`` count), or a ``(nelec, norb)`` target (required if
                ``fixed=True``).
            fixed: pin the exact ``(nelec, norb)`` rather than dynamically dropping
                (see the stability note above).
            restrict_to_a: forwarded to the CL stage's :meth:`build_orbitals` call
                (see its docstring) -- a pure performance knob, ``False`` only
                needed to exercise this against a stub adapter without live EmbASI.

        Returns:
            The APC-selected :class:`EmbeddedOrbitals`.
        """
        from embasi_qiskit_integration.selectors import (
            apc_active_space,
            apc_orbital_entropies,
            apc_pair_coefficients,
            concentric_localization_selector,
            somo_occupation_pattern,
        )

        cl = concentric_localization_selector(
            self._s_arr, fragment_ao, self._fock_arr, n_shells=n_shells
        )
        orbitals = self.build_orbitals(
            n_frozen_occ=0, virtual_localizer=cl, restrict_to_a=restrict_to_a
        )

        c_full = orbitals.coeff  # (nao, n_A): every subsystem-A orbital, occ + virt
        n_orb = c_full.shape[1]
        n_occ = orbitals.n_occ

        dm_a = 2.0 * c_full[:, :n_occ] @ c_full[:, :n_occ].T
        k_ao = self.ints.get_k(dm_a)
        f_diag = np.einsum("pi,pq,qi->i", c_full, self._fock_arr, c_full)
        k_diag = np.einsum("pi,pq,qi->i", c_full, k_ao, c_full)

        # CL's own (uncapped) shell pool is the candidate set APC ranks within; every
        # orbital outside it (CL's kernel remainder -- spatially irrelevant to the
        # fragment) gets a sentinel entropy far below the real range so Chooser
        # always drops it before any genuine candidate.
        cand_occ = orbitals.active[orbitals.active < n_occ]
        cand_virt = orbitals.active[orbitals.active >= n_occ]
        c_pairs = apc_pair_coefficients(f_diag[cand_occ], f_diag[cand_virt], k_diag[cand_virt])
        s_occ, s_virt = apc_orbital_entropies(c_pairs)

        # from pyscf.tools import cubegen
        # for occ_idx in np.arange(0,n_occ):
        #    print(occ_idx)
        #    cubegen.orbital(self.p.A_LL.atoms.calc.method.mol, f'orbital_occ_{occ_idx}.cube', c_full[:, occ_idx])

        # for virt_idx in np.arange(n_occ,n_orb):
        #    print(virt_idx)
        #    cubegen.orbital(self.p.A_LL.atoms.calc.method.mol, f'orbital_virt_{virt_idx}.cube', c_full[:, virt_idx])

        entropies = np.full(n_orb, -1.0e18)
        entropies[cand_occ], entropies[cand_virt] = s_occ, s_virt

        # TODO(open-shell, needs EmbASI): `somo_occupation_pattern` is POSITIONAL, and
        # this method has just run concentric localization, which rotates within blocks.
        # Plain UHF already breaks the assumption: for OH/sto-3g the projected occupation
        # is [2,2,2,1,2,0] -- the SOMO is at index 3, not atop the occupied block -- so
        # APC would protect the wrong orbital.  Latent only because nothing sets
        # `n_occ_b`.  Replace with the rotation-invariant projection once available:
        #
        #     dm_a, dm_b = selectors.per_spin_ao_rdm1(<unrestricted low-level mf>)
        #     occ_pattern = selectors.column_occupation_from_density(
        #         c_full, dm_a, dm_b, self._s_arr)
        #
        # which also needs the per-spin subsystem-A densities (blocker (2)).
        if orbitals.is_open_shell:
            occ_pattern = somo_occupation_pattern(n_orb, n_occ, orbitals.n_occ_b)
        else:
            occ_pattern = np.where(np.arange(n_orb) < n_occ, 2, 0)
        active = apc_active_space(occ_pattern, entropies, max_size, fixed=fixed)
        print(f"ACTIVE SPACE: {active}")
        inactive = np.array([i for i in range(n_occ) if i not in set(active.tolist())], dtype=int)
        return EmbeddedOrbitals(
            coeff=c_full,
            energy=orbitals.energy,
            n_occ=n_occ,
            inactive=inactive,
            active=active,
            # Carry the beta count through selection; dropping it here would silently
            # demote an open-shell partition back to the restricted reading.
            n_occ_b=orbitals.n_occ_b,
        )

    # ---------------- downfolding ---------------- #
    def embedded_hamiltonian(self, orbitals: EmbeddedOrbitals) -> EmbeddedHamiltonian:
        """CASCI-style downfold of h_emb + bare ERIs onto the active space."""
        if self.unrestricted and not orbitals.is_open_shell:
            raise NotImplementedError(
                "unrestricted=True but these orbitals carry a single occupied count "
                "(n_occ_b is None), so the downfold would silently produce the "
                "restricted n_alpha = n_beta split.  EmbASI does not yet expose the "
                "per-spin occupied counts (see this module's docstring, blocker (1)).  "
                "Until then, drive open shell through a FCIDUMP: read the Hamiltonian "
                "with hamiltonian.fcidump.read, which carries (n_alpha, n_beta) exactly."
            )
        h_emb = self.h_emb
        c_in, c_act = orbitals.c_inactive, orbitals.c_active

        dm_in = 2.0 * (c_in @ c_in.T)
        veff_in = self.ints.veff_hf(dm_in)  # HF, regardless of the high level

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
        e_core = self.ints.energy_nuc() + np.einsum("ij,ji->", dm_in, h_emb + 0.5 * veff_in)

        n_alpha, n_beta = orbitals.n_active_electrons_spin

        # P_B must be invisible inside the active space; if it is not, the A/B
        # separation is broken and everything above is meaningless.
        leak = float(np.abs(c_act.T @ self.p_b @ c_act).max())
        if leak > 1e-6:
            raise ValueError(f"active orbitals leak into subsystem B (|P_B| = {leak:.2e})")

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

        **Footing rebasing.**  The subtraction ``E_low(AB) - E_low(A) + E_high(A)``
        only telescopes if ``E_high(A)`` and ``E_low(A)`` reference the *same*
        one-electron operator.  They do not by default: ``E_high(A)`` inherits the
        **full-supersystem** ``h_core`` / ``energy_nuc`` from the downfolded
        Hamiltonian's ``e_core`` (all nuclei), whereas EmbASI computes
        ``subsys_A_lowlvl_totalen`` on its **ghosted subsystem-A** ``mol`` (only A's
        nuclei; the environment atoms are basis ghosts).  Left uncorrected the two
        A-terms sit ~4 Ha apart and the paper's Eq. 8 cancellation fails -- and
        the outer loop still converges on the *density* while carrying that offset
        in the *energy*, so density convergence alone never reveals it.  The shift
        between the two references is a density-linear one-body quantity,
        ``Δ = (E_nuc^full - E_nuc^A) + tr[γ̃^A (h_core^full - h_core^A)]`` (verified
        density-independent in the 2-electron part -- same basis, same ERIs), so we
        subtract it from ``e_high_A`` to land it on ``E_low(A)``'s footing.  What
        remains after the shift is the genuine high-vs-low functional difference on
        the fragment (WF-in-DFT), not the nuclear-frame artefact.
        """
        v_emb, p_b = self.v_emb, self.p_b
        dm_hl = self.rdm1_ao(result.rdm1, orbitals)

        leak = float(np.einsum("ij,ji->", dm_hl, p_b))
        e_high_a = float(result.energy) - np.einsum("ij,ji->", dm_hl, v_emb) - leak
        correction = float(np.einsum("ij,ji->", dm_hl - self._dm_a_arr_init, v_emb))

        # Rebase e_high_A onto E_low(A)'s (ghosted subsystem-A) nuclear footing.
        hcore_a, enuc_a = self._a_fragment_footing()
        footing_shift = float(
            (self.ints.energy_nuc() - enuc_a)
            + np.einsum("ij,ji->", dm_hl, self.ints.hcore() - hcore_a)
        )
        e_high_a -= footing_shift

        e_low_ab, e_low_a = self._low_level_energies()
        return ProjectionEnergy(
            e_low_total=e_low_ab,
            e_low_A=e_low_a,
            e_high_A=float(e_high_a),
            correction=correction,
            projector_leak=leak,
            footing_shift=footing_shift,
        )

    # ---------------- DFT-in-DFT (reference path, no solver) ---------------- #
    def dft_in_dft_energy(self) -> ProjectionEnergy:
        """Paper **Eq. 2** DFT-in-DFT total, read straight from EmbASI's ``run()``.

        This is a *separate path* from the WF-in-DFT downfold (:meth:`build_orbitals`
        -> :meth:`embedded_hamiltonian` -> solver -> :meth:`projection_energy`).  A
        hybrid or pure high-level functional (PBE0, PBE, ...) is a **density**
        functional, not a wavefunction method: Eq. 2 evaluates ``E_H[γ̃^A]`` over the
        *full* occupied space of A at the embedded density, with no active space, no
        virtual budget, no frozen core, no bare-electronic downfold, and no solver.
        Routing a hybrid xc through the WF path (Eq. 8) is a category error whose
        signature is a large non-monotonic dependence on the virtual budget; this
        method exists so the driver can send hybrid/pure-DFT high levels here instead.

        EmbASI already ships the complete DFT-in-DFT total.  ``ProjectionEmbedding.
        run()`` runs the supersystem SCF, ``A_LL.run_noscf``, and the embedded
        high-level KS SCF ``A_HL.run_emb_scf``, then assembles (all eV, its
        ``total_energy_corr="1storder"`` branch)::

            DFT_AinB_total_energy = subsys_AB_lowlvl_scftotalen   # E_low(AB)
                                  - subsys_A_lowlvl_totalen       # E_low(A)
                                  + subsys_A_highlvl_totalen      # E_high(A), KS
                                  + order_1_embedding_corr        # tr[(γ̃^A-γ^A) v_emb]
                                  + PB_corr                       # tr[P_B γ̃^A]

        so we run it once and re-slice those terms into the :class:`ProjectionEnergy`
        breakdown (:meth:`_native_dft_in_dft_energy`), rather than re-solving an
        embedded KS SCF in PySCF alongside it.  Two consequences of using EmbASI's
        native total:

        * **No footing rebase.**  ``DFT_AinB_total_energy`` telescopes on EmbASI's own
          internally-consistent frame -- ``subsys_A_highlvl_totalen`` and
          ``subsys_A_lowlvl_totalen`` are both on the ghosted subsystem-A ``mol`` --
          so the ``E_high(A)``-onto-``E_low(A)`` nuclear rebase the WF path needs
          (:meth:`_a_fragment_footing`) does **not** apply here; ``footing_shift`` is
          exactly ``0``.
        * **PB projector term is included.**  ``PB_corr = tr[P_B γ̃^A]`` is folded into
          ``e_high_A`` (it is the projector-leak energy of the A high-level term) and
          also reported as ``projector_leak``.  The WF ``projection_energy`` subtracts
          this leak because its solver energy *carried* the embedding potential; the
          DFT-in-DFT total *adds* it because EmbASI's assembly includes it.

        Runs ``self.p.run()`` and NOT ``run_low_level()`` -- ``run()`` re-invokes
        ``construct_embedding_potential`` internally, so calling both would double the
        supersystem SCF.  The driver branches on :meth:`EmbeddingWorkflow._is_dft_in_dft`
        *before* the low-level call to guarantee exactly one fires.

        Returns the same :class:`ProjectionEnergy` breakdown as the WF path, so the
        driver and the dissociation-energy bracket consume both identically.
        """
        self.p.run()  # supersystem SCF + A_LL noscf + A_HL embedded KS SCF + assembly
        return self._native_dft_in_dft_energy()

    def _native_dft_in_dft_energy(self) -> ProjectionEnergy:
        """Re-slice EmbASI's native ``DFT_AinB`` terms into a :class:`ProjectionEnergy`.

        All source terms are eV on the projection object after :meth:`p.run`; we
        convert with EmbASI's own factor (:data:`_EV2HA`) and preserve the identity
        ``.total == DFT_AinB_total_energy * _EV2HA`` exactly (a linear re-slice of
        EmbASI's own sum, not a recompute) -- see :meth:`dft_in_dft_energy`.  Missing
        attributes raise loudly (upstream rename) rather than assemble a wrong energy.
        """
        p = self.p
        e_low_ab, e_low_a = self._low_level_energies()  # reuse the eV->Ha reader
        try:
            e_high_a_ks = float(np.real(p.subsys_A_highlvl_totalen)) * _EV2HA
            correction = float(np.real(p.order_1_embedding_corr)) * _EV2HA
            pb_corr = float(np.real(p.PB_corr)) * _EV2HA
        except AttributeError as exc:  # pragma: no cover - upstream rename guard
            raise AttributeError(
                "EmbASI did not expose the DFT-in-DFT high-level terms "
                "'subsys_A_highlvl_totalen' / 'order_1_embedding_corr' / 'PB_corr' "
                "after run(); the DFT-in-DFT energy cannot be assembled (upstream "
                "attribute renamed, or run() did not take the 1st-order branch?)"
            ) from exc
        return ProjectionEnergy(
            e_low_total=e_low_ab,
            e_low_A=e_low_a,
            e_high_A=e_high_a_ks + pb_corr,  # PB projector term belongs with A high-level
            correction=correction,
            projector_leak=pb_corr,  # reported only; already inside e_high_A above
            footing_shift=0.0,  # native frame is self-consistent; no rebase
        )

    def _a_fragment_footing(self) -> tuple[np.ndarray, float]:
        """(h_core^A, E_nuc^A) of EmbASI's *ghosted subsystem-A* reference.

        These are the one-electron operator and nuclear repulsion EmbASI used to
        build ``subsys_A_lowlvl_totalen`` (:meth:`_low_level_energies`): its
        ``A_LL`` layer runs on a ``mol`` where only subsystem-A atoms carry nuclei
        and the environment atoms are basis *ghosts* (basis functions, no charge).
        ``E_high(A)`` must be rebased onto exactly this footing before the Eq. 8
        subtraction (see :meth:`projection_energy`).  This is the **WF-in-DFT path
        only**: the DFT-in-DFT path reads EmbASI's native ``DFT_AinB_total_energy``,
        which telescopes on EmbASI's own frame and needs no rebase (:meth:`dft_in_dft_energy`).

        We read the operators *directly* off EmbASI's ``A_LL`` ``mol`` (the same
        object it integrated), not by reconstructing them from a subtraction of
        exposed matrices -- ``A_LL.hamiltonian_estat_plus_xc`` bundles the nuclear
        attraction together with Coulomb+xc, so h_core^A is not separable from
        what EmbASI exports.  ``int1e_kin + int1e_nuc`` on that ``mol`` reproduces
        ``subsys_A_lowlvl_totalen`` to ~1e-14 Ha (asserted in the tests), and its
        overlap matches ``self._s`` bit-for-bit (same basis, same AO ordering).

        TODO(embasi-api): this reach-through is physically unavoidable for the WF
        path against today's EmbASI.  The rebase needs ``h_core^A`` (a matrix) and the
        scalar ``E_nuc^A``, and EmbASI exposes neither cleanly nor lets them be
        reconstructed: ``hamiltonian_estat_plus_xc`` fuses nuclear attraction with the
        density-dependent Coulomb+xc, and ``subsys_A_lowlvl_totalen`` is one scalar
        (carrying ``tr[γ^A h_core^A]``, not the ``tr[γ̃ h_core^A]`` this contracts).
        ``subsys_A_highlvl_totalen`` cannot substitute either -- on the WF path the
        high level is HF + an external correlated solver, so EmbASI's value is only
        the *mean-field* HF-in-DFT reference (and is the *embedded* energy, carrying
        ``v_emb``/``P_B``), not the bare correlated ``E_high(A)`` the assembly needs.
        Absent an EmbASI-native A-fragment footing, we reach through
        ``A_LL.atoms.calc.mol`` to a PySCF ``Mole`` -- fine for PySCF, but there is no
        equivalent for the FHI-aims/ASI backend, so the WF-path rebase is PySCF-only.
        A backend-agnostic A-fragment ``h_core`` / ``energy_nuc`` on EmbASI would
        remove the reach-through entirely.
        """
        try:
            mol_a = self.p.A_LL.atoms.calc.mol
            hcore_a = np.asarray(mol_a.intor("int1e_kin") + mol_a.intor("int1e_nuc"))
            enuc_a = float(mol_a.energy_nuc())
        except AttributeError as exc:  # pragma: no cover - non-PySCF / upstream change
            raise AttributeError(
                "cannot reach EmbASI's subsystem-A (ghosted) mol via "
                "A_LL.atoms.calc.mol to rebase E_high(A) onto E_low(A)'s nuclear "
                "footing; the projection energy would be ~4 Ha off without it. "
                "Expose subsys_A_highlvl_totalen or the A-fragment "
                "h_core/energy_nuc upstream for a backend-agnostic fix."
            ) from exc
        if hcore_a.shape != self._s_arr.shape:
            raise ValueError(
                f"subsystem-A h_core is {hcore_a.shape} but the supersystem basis "
                f"is {self._s_arr.shape}; the ghosted A mol and F_emb are on different "
                "bases (basis truncation not yet mapped, see run_low_level TODO)"
            )
        return hcore_a, enuc_a

    def _low_level_energies(self) -> tuple[float, float]:
        """(E_low(AB), E_low(A)) in Hartree, read from EmbASI internals.

        ``construct_embedding_potential`` (called by :meth:`run_low_level` on the
        WF path, and internally by ``ProjectionEmbedding.run()`` on the DFT-in-DFT
        path) already computes both low-level total energies as a side effect and
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
                "construct_embedding_potential(); the projection energy cannot be "
                "assembled (upstream attribute renamed?)"
            ) from exc
        return float(np.real(e_ab)) * _EV2HA, float(np.real(e_a)) * _EV2HA

    # ---------------- density feedback ---------------- #
    def rdm1_ao(self, rdm1_active: np.ndarray, orbitals: EmbeddedOrbitals) -> np.ndarray:
        """Back-transform the correlated (spin-summed) 1-RDM to the AO basis.

        ``inactive`` columns are doubly occupied in both the restricted and the
        unrestricted reading (they are frozen into ``e_core``), so the ``2.0`` factor on
        the core block is correct either way; only the active block carries the
        correlated occupation.
        """
        c_in, c_act = orbitals.c_inactive, orbitals.c_active
        return 2.0 * (c_in @ c_in.T) + c_act @ np.asarray(rdm1_active) @ c_act.T

    def rdm1_ao_spin(
        self,
        rdm1_active_a: np.ndarray,
        rdm1_active_b: np.ndarray,
        orbitals: EmbeddedOrbitals,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Back-transform a spin-resolved active 1-RDM pair to AO alpha/beta densities.

        The unrestricted counterpart of :meth:`rdm1_ao`, for the density feedback an
        open-shell outer loop needs: a spin-summed total cannot represent
        ``gamma_alpha != gamma_beta``, so the loop must carry the pair.

        The frozen core contributes one electron per spin channel (``c_in @ c_in.T``,
        not ``2.0 *``), so that the two channels sum back to the same total
        :meth:`rdm1_ao` produces.

        Returns:
            ``(dm_alpha, dm_beta)`` in the AO basis.
        """
        c_in, c_act = orbitals.c_inactive, orbitals.c_active
        core = c_in @ c_in.T  # one electron per channel
        dm_a = core + c_act @ np.asarray(rdm1_active_a) @ c_act.T
        dm_b = core + c_act @ np.asarray(rdm1_active_b) @ c_act.T
        return dm_a, dm_b

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
        self.run_low_level(dm_ab_in=dm_hl + self._dm_b_arr)

    # ---------------- checks ---------------- #
    def _validate_densities(self) -> None:
        n_a = np.einsum("ij,ji->", self._dm_a_arr, self._s_arr)
        n_b = np.einsum("ij,ji->", self._dm_b_arr, self._s_arr)
        if not np.isclose(round(n_a), n_a, atol=1e-6):
            raise ValueError(f"tr(γ^A S) = {n_a:.6f} is not integral")
        if not np.isclose(round(n_b), n_b, atol=1e-6):
            raise ValueError(f"tr(γ^B S) = {n_b:.6f} is not integral")

    def _validate_span(self, c_occ: np.ndarray) -> None:
        """Occupied block of F_emb must span the same space as mo_coeffs_A_LL."""
        overlap = self.mo_a_ll.T @ self._s_arr @ c_occ
        sv = np.linalg.svd(overlap, compute_uv=False)
        if sv.min() < 1.0 - 1e-6:
            raise ValueError(
                f"occupied A space mismatch (min singular value {sv.min():.6f}); "
                "the localisation and the embedded Fock disagree"
            )
