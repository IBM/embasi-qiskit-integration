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
  **Largely moot since EmbASI's ``9c21cac``**, which sets ``ghosts=0`` on every
  ``set_layer`` call and carries the fragment via the total charge instead: ``A_LL`` now
  integrates on the full supersystem frame, the same one ``e_core`` uses, so the rebase
  self-neutralises (measured ``footing_shift ~ 8e-16``, was ~3.8 Ha).  It is kept as the
  *identity* that makes the two frames agree -- costing nothing when they already do, and
  catching a future divergence -- so the reach-through above still exists and is still
  PySCF-only, it just no longer changes any number.  See ``port.md``.
* **Huzinaga as a constant offset when high/low xc match** (paper Sec. 2.1).
  ``H^{AB}_H`` then collapses to the supersystem low-level Hamiltonian, making
  ``P_B`` a constant matrix -- exportable after all, in exactly the WF-in-DFT
  regime.  See :meth:`__init__`.

Open-shell / unrestricted embedding
-----------------------------------

The *quantum* half is open-shell correct (FCIDUMP -> circuit -> SQD -> energy and
spin-resolved RDMs, at any ``(n_alpha, n_beta)``).  The *embedding* half now works too,
on a per-spin path -- and it needed **no upstream EmbASI change**: everything required was
already exposed by the spin-polarised PySCF support on ``qm-code-adapter``.

**What works** (validated on a live OH-radical-in-water doublet, sto-3g):

* **Spin sector.**  ``A_spin`` off the live ``ProjectionEmbedding`` gives the beta
  occupied count (:meth:`_n_occ_b_from_embasi`), so :attr:`EmbeddedOrbitals.n_occ_b` has a
  real source.  Measured ``A_spin = 1`` / ``B_spin = 0``; sector ``(5, 4)``.
* **Per-spin operators.**  :meth:`_as_ao_pair` keeps the ``(alpha, beta)`` blocks that
  :meth:`_as_ao_total` sums, and :meth:`_assemble_fock_spin` builds a per-channel
  ``F_emb``.  The spin-summed copies are retained unchanged for the restricted path.
* **Per-spin downfold.**  :meth:`build_orbitals_spin` diagonalizes each channel in *its
  own* span(A) (:meth:`_eigh_subsystem_a_spin`) and
  :meth:`embedded_hamiltonian_spin` emits the ``(h1a, h1b)`` pair
  :class:`~embasi_qiskit_integration.contract.EmbeddedHamiltonian` accepts.
  ``FCISolver`` consumes it via ``pyscf.fci.direct_uhf``.
* **Per-spin lift-back.**  Because the two channels are diagonalized in *different*
  spans, everything that lifts an active-space quantity to AO has to be told which
  channel it is lifting: :meth:`rdm1_ao_spin` and :meth:`projection_energy` both take
  beta's own orbital set.  Getting this wrong is silent -- the electron counts, the spin
  sector, the density symmetry and the footing shift all stay exact while the beta AO
  density is simply the wrong matrix (measured 0.998 off, and 0.0779 Ha / 48.9 kcal/mol
  on the reported total).

**Why the downfold must be per spin, not spin-summed.**  EmbASI's SPADE partitions the
two channels independently, so the A/B separation holds *per channel only*: measured
``span(A_alpha)`` vs ``span(B_alpha)`` = 8.6e-15 (orthogonal) but vs ``span(B_beta)`` =
2.7e-04.  A spin-summed ``P_B`` therefore annihilates neither channel's A orbitals --
the projector-leak check fires at 3.9e-02, and no reduction of ``P_B`` (alpha, beta, sum
or mean) avoids it.  Each per-spin ``P_B`` annihilates its own channel to ~3e-16, and the
per-spin downfold leaks ~1.5e-10.  So a spin-restricted downfold is not merely lower
quality on an open shell, it is ill-defined; :meth:`embedded_hamiltonian` refuses it with
an error saying so.

**What is spin-resolved end to end:** the sector, ``v_emb``/``P_B``/``F_emb``, the MO
sets, the one-body operator (``h1a``/``h1b``), the two-body integrals
(``h2_spin`` = ``(aa|aa)``, ``(aa|bb)``, ``(bb|bb)``, the mixed block via
:meth:`PySCFIntegrals.eri_mo_mixed`), the solver (``pyscf.fci.direct_uhf``) and the
density feedback (:meth:`_as_spin_kpoint_pair` at ``n_spin=2``, which is the shape
EmbASI's PySCF adapter reads as two genuine channels).  ``EmbeddingWorkflow`` reaches it
with ``spin_downfold=True``; measured end to end on the OH radical: ``nelec=(5, 4)``,
per-spin leak ~2e-10, ``E_solver = -53.522493`` Ha.

**Known limits of the per-spin path** (none of them silent):

* **Localisers work per channel; index selectors do not.**  ``spade`` /
  ``concentric-cl`` are applied to each channel's own virtual block.  ``mulliken`` and
  ``apc-concentric`` rank columns against a single spin-summed Fock and have no
  per-channel form, so ``EmbeddingWorkflow`` refuses them with ``spin_downfold``.
* **The active-orbital count is reconciled across the channels, the electron counts are
  not.**  ``build_orbitals_spin`` equalises ``norb`` -- that is the only thing
  ``fci.direct_uhf`` requires, since it takes one orbital dimension with an asymmetric
  ``(n_alpha, n_beta)`` -- and derives each channel's *virtual* count from its own
  occupied count, so the channel with fewer electrons takes more virtuals.  ``n_virtual``
  is consequently a ceiling on the active space, counted above the widest channel's active
  occupied block, **not** a per-channel virtual count: an open shell has
  ``n_occ_alpha != n_occ_beta``, so one number applied to both would leave the two active
  spaces differing by exactly ``A_spin``.  ``n_frozen_occ`` stays per channel (a shared
  *count* does not freeze corresponding orbitals) and is validated against the narrower
  channel, with a warning when a frozen pair's ``|<a_i|S|b_i>|`` drops below
  ``_FROZEN_OVERLAP_TOL``.
* **Only FCI consumes the pair.**  SQD's ``diagonalize_fermionic_hamiltonian`` and the
  FCIDUMP format each take a single one-body tensor; both fall back to the spin-averaged
  ``h1`` and warn.  ``h2_spin`` is likewise FCI-only.  This is an upstream
  ``qiskit-addon-sqd`` limitation (no UHF entry point, and its ``sci_solver`` hook
  receives the tensor already substituted), not something fixable here -- see
  ``port.md``.
* **The energy terms are contracted per channel, and the totals derived from them.**
  Every density-linear term in :meth:`projection_energy` contracts each channel against
  *its own* ``v_emb``/``P_B`` (:meth:`v_emb_spin`, ``_p_b_spin``) and sums, because that
  is what :meth:`embedded_hamiltonian_spin` folded into ``h1_s``.  The spin-summed
  ``correction`` / ``projector_leak`` are then **derived** from the pair, so
  ``sum(correction_spin) == correction`` still holds exactly -- by construction now,
  rather than by sharing one operator.  Contracting the spin-summed density against the
  spin-summed operators instead is wrong twice over, since SPADE partitions the spins
  separately and ``span(A_alpha)`` is not S-orthogonal to ``span(B_beta)``: the projector
  picks up cross terms scaled by ``mu`` (``+2.4e4`` on ``data/22.inp``, a field documented
  as a numerical zero, dragging ``total`` to -24288 Ha on a ~-207 Ha system) and the
  ``v_emb`` term removes something the solver never added (+54.0 Ha).  Each channel is
  referenced to its own round-0 density (``_dm_a_spin_init``), not half the spin-summed
  one -- ``gamma^A`` is polarised, so halving mis-attributes reference density.
  ``e_high_A`` still has no per-spin form: it carries the solver's total energy, which the
  solver does not decompose.  :meth:`export_state` exports the spin-summed arrays, the
  per-spin ones, and the adapter's own per-channel ``v_emb`` pair
  (``v_emb_spin_adapter``) that :func:`projection_energy_from_state` needs.
* **The frozen core is folded unrestricted.**  At ``n_frozen_occ > 0`` each channel sees
  ``J[d_a + d_b] - K[d_sigma]`` (:meth:`AOIntegrals.veff_uhf`), not the restricted
  ``J - K/2``: Coulomb is a functional of the total core density but exchange couples
  like spins only, and SPADE freezes different orbitals per channel.  Inert at
  ``n_frozen_occ=0``.
* **The outer loop damps the channels, converges on the total.**  When the feedback is
  spin-resolved, DIIS and the linear mixing act on the stacked ``(alpha, beta)`` pair, so
  each channel's own residual is damped; a single coefficient set is shared, since the
  two channels have one fixed point.  The ``max|Δγ^A|`` diagnostic still reports the
  spin-summed total.

**Still open upstream** (not blocking the above): ``spade_localisation.py`` carries
``# @TODOSPIN: Need to redefine occupancies`` at ``:116`` and ``:196`` on the density
assembly that branches on ``n_spins == 1`` for the factor-2 occupancy, and
``# TODO: @SPIN AND K-POINT LOOP`` appears at several sites in ``embedding.py`` -- only
relevant if k-points ever matter.  ``A_spin``/``B_spin`` also have no ``__init__``
default (they are assigned only inside ``construct_embedding_potential``), which is why
:meth:`_n_occ_b_from_embasi` reads them through ``getattr``.
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

# `n_frozen_occ` freezes the lowest k orbitals of EACH spin channel, but SPADE orders
# them per spin, so the two cores denote one shared set only while they overlap.  Below
# this, `build_orbitals_spin` warns: measured on the stretched C-N geometry the pairs run
# 1.000 / 1.000 / 0.982 / 0.951 and then fall off a cliff to 1e-04, so the boundary is
# sharp and a threshold anywhere in (0.95, 1) separates "corresponding" from "unrelated".
# 0.9 is deliberately permissive -- it flags the cliff, not ordinary polarisation.
_FROZEN_OVERLAP_TOL = 0.9


# A density argument to the `veff_*` accessors: either the spin-summed AO matrix or a
# genuine `(alpha, beta)` pair.  The pair matters for a KS low level, whose xc is
# nonlinear in the spin densities -- see `ProjectionEmbeddingAdapter._dm_a_for_veff`.
DensityArg = np.ndarray | tuple[np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# AO integrals: the quantities EmbASI/ASI does not (yet) hand you
# --------------------------------------------------------------------------- #
class AOIntegrals(Protocol):
    """AO-basis quantities the adapter needs beyond ProjectionEmbedding."""

    def overlap(self) -> np.ndarray: ...
    def hcore(self) -> np.ndarray: ...

    def veff_hl(self, dm: DensityArg) -> np.ndarray:
        """Effective potential of the *high-level* calculator, at a 2-occupancy dm.

        Must match ``calc_base_hl``.  Retained for callers that want the
        high-level mean field explicitly; the :attr:`h_emb` inverse uses
        :meth:`veff_ll` instead -- see that method.

        ``dm`` may be an ``(alpha, beta)`` pair; see :meth:`veff_ll` for why that
        matters and what the return must be in that case.
        """

    def veff_ll(self, dm: DensityArg) -> np.ndarray:
        """Effective potential of the *low-level* calculator, at a 2-occupancy dm.

        Must match ``calc_base_ll``: it exists only to undo the mean-field
        contribution EmbASI folded into ``F_emb``.  ``F_emb`` is assembled from
        the **A_LL** one-electron blocks (see
        :meth:`ProjectionEmbeddingAdapter._assemble_fock_a_only`), so the mean
        field baked into it is the *low-level* one, and that is what
        :attr:`ProjectionEmbeddingAdapter.h_emb` must subtract back off.

        ``dm`` is either a spin-summed ``(nao, nao)`` matrix or an
        ``(alpha, beta)`` pair -- the adapter passes the pair whenever it has one
        (:attr:`ProjectionEmbeddingAdapter._dm_a_for_veff`), because an
        unrestricted KS ``get_veff`` handed a summed matrix silently substitutes
        ``d/2`` for both channels and loses the spin polarisation.  The **return**
        is always a single ``(nao, nao)`` operator: a spin-resolved backend must
        average its two channels, not sum them (a potential is per-electron).
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

    def veff_uhf(self, dm_a: np.ndarray, dm_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Unrestricted HF mean field of a spin-resolved 1-occupancy density pair.

        Returns ``(v_alpha, v_beta, e_two_body)`` where
        ``v_sigma = J[dm_a + dm_b] - K[dm_sigma]`` and
        ``e_two_body = 0.5 tr[d J[d]] - 0.5 (tr[dm_a K[dm_a]] + tr[dm_b K[dm_b]])``
        with ``d = dm_a + dm_b``.

        This is the frozen-core fold the *per-spin* downfold needs, and it is **not**
        obtainable from :meth:`veff_hf`.  ``veff_hf`` is the restricted ``J - K/2``:
        correct only when the two channels share an orbital set, because Coulomb is a
        functional of the total density but **exchange couples like spins only**.
        Using the spin-averaged form for two genuinely different cores understates the
        channel splitting -- measured (ERI-only ground truth, 6-31g, valence-like cores
        overlapping by 0.34) ``max|v_avg - v_alpha| = 1.09 Ha`` and **+0.78 Ha
        (+492 kcal/mol)** on the CAS total.  It vanishes only as the two cores coincide,
        which is why a deep-1s frozen core (overlap 0.9999998) hides it at 0.02
        kcal/mol.

        The energy is returned alongside the potentials rather than left to the caller
        because the exchange self-interaction does not factor out of
        ``0.5 tr[d v]``: that expression is only valid for the restricted ``J - K/2``.
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

    ``mf_ll`` is the same object handed to ``calc_base_ll``, and backs
    :meth:`veff_ll` -- the mean field :attr:`ProjectionEmbeddingAdapter.h_emb`
    subtracts back off ``F_emb``.  It defaults to ``mf_hl`` so a caller that only
    has one mean field (the stub adapters in the tests, and the ``xc_hl == xc_ll``
    case where the two coincide anyway) still constructs; pass the real low-level
    object whenever the two levels differ, or ``h_emb`` strips the wrong veff.
    """

    def __init__(self, mf_hl, mf_ll=None, *, density_fit: bool | str = False):
        self.mf = mf_hl  # the same object passed to calc_base_hl
        self.mf_ll = mf_hl if mf_ll is None else mf_ll  # the one passed to calc_base_ll
        self.mol = mf_hl.mol
        # Integral engine only; never kernel()'d.  `scf.hf.RHF` explicitly, NOT
        # `mol.RHF()`: on a `spin != 0` Mole the latter dispatches to **ROHF**, whose
        # `get_veff`/`get_k` return a `(2, nao, nao)` spin pair.  The downfold and the
        # APC ranking both contract those against `(nao, nao)` blocks, so an open-shell
        # run died with "operands could not be broadcast together with shapes (2,n,n)
        # (n,n,2)".  `scf.hf.RHF` is the restricted class regardless of `mol.spin`, and
        # is bit-identical to `mol.RHF()` whenever the Mole is closed shell -- verified,
        # so this changes nothing for restricted runs.
        from pyscf import scf as _pyscf_scf

        self._hf = _pyscf_scf.hf.RHF(mf_hl.mol)
        self._density_fit = density_fit
        # A pyscf.df.DF, built lazily on the first eri_mo call. PySCF is untyped,
        # so this is Any rather than a precise DF type.
        self._df: Any = None

    def overlap(self) -> np.ndarray:
        return np.asarray(self.mol.intor("int1e_ovlp"))

    def hcore(self) -> np.ndarray:
        return np.asarray(self.mf.get_hcore())

    @staticmethod
    def _spin_average(v: np.ndarray) -> np.ndarray:
        """Collapse a ``(2, nao, nao)`` veff pair to the restricted equivalent.

        A UKS/UHF/ROHF ``get_veff`` returns one matrix per spin channel.  The
        spin-restricted downfold this adapter performs wants a single ``(nao, nao)``
        operator, and the correct reduction is the **mean**, not the sum: ``veff`` is a
        *potential* (a per-electron operator), unlike a density matrix.  Verified on a
        closed shell, where ``0.5 * (v_alpha + v_beta)`` reproduces the restricted
        ``get_veff`` exactly while the sum is twice too large -- routing this through
        ``_as_ao_total`` (which sums, correctly, for densities) would silently double
        the mean field that ``h_emb`` subtracts back off.
        """
        v = np.asarray(v)
        if v.ndim == 3 and v.shape[0] == 2:
            return 0.5 * (v[0] + v[1])
        return v

    @staticmethod
    def _as_veff_arg(mf, dm: DensityArg) -> np.ndarray:
        """Shape a density argument for ``mf.get_veff``, collapsing a pair if ``mf`` is restricted.

        An unrestricted ``get_veff`` wants ``(2, nao, nao)`` and reads the two channels
        as genuine alpha/beta; a *restricted* one wants ``(nao, nao)`` and would read a
        3-D array as a batch of densities.  So the pair is only passed through when the
        mean field can interpret it, and summed otherwise -- which is exactly right,
        since a restricted functional depends on the total density alone.
        """
        d = np.asarray(dm[0]) + np.asarray(dm[1]) if isinstance(dm, tuple) else np.asarray(dm)
        if not isinstance(dm, tuple):
            return d
        from pyscf import scf as _scf

        if isinstance(mf, (_scf.uhf.UHF, _scf.rohf.ROHF)):
            return np.stack([np.asarray(dm[0]), np.asarray(dm[1])])
        return d

    def veff_hl(self, dm: DensityArg) -> np.ndarray:
        return self._spin_average(self.mf.get_veff(self.mol, self._as_veff_arg(self.mf, dm)))

    def veff_ll(self, dm: DensityArg) -> np.ndarray:
        return self._spin_average(self.mf_ll.get_veff(self.mol, self._as_veff_arg(self.mf_ll, dm)))

    def veff_hf(self, dm) -> np.ndarray:
        # `_hf` is the restricted engine (see __init__), so this is normally already
        # (nao, nao); the reduction guards a spin-resolved `dm` argument, which puts
        # even RHF.get_veff on its two-channel branch.
        return self._spin_average(self._hf.get_veff(self.mol, np.asarray(dm)))

    def get_k(self, dm) -> np.ndarray:
        return self._spin_average(self._hf.get_k(self.mol, np.asarray(dm)))

    def veff_uhf(self, dm_a, dm_b) -> tuple[np.ndarray, np.ndarray, float]:
        """``(v_alpha, v_beta, e_two_body)`` for a spin-resolved core; see the protocol.

        ``self._hf`` is the restricted *integral engine* (never kernel()'d), and
        ``get_j``/``get_k`` on it are plain contractions of the AO ERIs with whatever
        density they are handed -- verified against a direct ``int2e`` contraction.  No
        spin averaging happens anywhere here; that is what :meth:`veff_hf` does, and what
        this method exists to avoid.

        Each channel's exchange is fetched in its own call rather than as one stacked
        ``(2, nao, nao)`` density: the stacked form gives identical numbers (checked) but
        makes PySCF log ``Incompatible dm dimension. Treat dm as RHF density matrix.`` on
        every evaluation, which is noise on a per-cycle code path.

        Like :meth:`veff_hf` and :meth:`get_k`, this uses the exact AO ERIs even when
        the backend was built with ``density_fit``; the DF approximation is applied only
        in :meth:`eri_mo`/:meth:`eri_mo_mixed`, matching the pre-existing behaviour of
        the restricted fold rather than introducing a second convention.
        """
        d_a = np.asarray(dm_a)
        d_b = np.asarray(dm_b)
        d_tot = d_a + d_b
        j_tot = np.asarray(self._hf.get_j(self.mol, d_tot))
        k_a = np.asarray(self._hf.get_k(self.mol, d_a))
        k_b = np.asarray(self._hf.get_k(self.mol, d_b))
        e_two_body = float(
            0.5 * np.einsum("ij,ji->", d_tot, j_tot)
            - 0.5 * (np.einsum("ij,ji->", d_a, k_a) + np.einsum("ij,ji->", d_b, k_b))
        )
        return j_tot - k_a, j_tot - k_b, e_two_body

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

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        """Mixed-spin ``(p_a q_a | r_b s_b)``, shape ``(nmo,)*4``.

        The ``(aa|bb)`` block of an unrestricted downfold, where the first index pair
        comes from the alpha orbitals and the second from the beta ones.  Needs the
        four-orbital-set ``ao2mo.general``, not ``ao2mo.kernel``: the two pairs are
        different orbital sets, so the result has only 4-fold symmetry
        (``(pq|rs) == (qp|rs) == (pq|sr)``) and **not** the 8-fold property -- which is
        why :class:`~embasi_qiskit_integration.contract.EmbeddedHamiltonian` checks it
        separately.
        """
        from pyscf import ao2mo

        nmo = mo_a.shape[1]
        if mo_b.shape[1] != nmo:
            raise ValueError(
                f"alpha and beta blocks must share a dimension; got {nmo} and {mo_b.shape[1]}"
            )
        if not self._density_fit:
            out = ao2mo.general(self.mol, (mo_a, mo_a, mo_b, mo_b), compact=False)
            return np.asarray(out).reshape(nmo, nmo, nmo, nmo)
        if self._df is None:
            from pyscf import df

            self._df = df.DF(self.mol)
            if isinstance(self._density_fit, str):
                self._df.auxbasis = self._density_fit
            self._df.build()
        out = self._df.ao2mo((mo_a, mo_a, mo_b, mo_b), compact=False)
        return np.asarray(out).reshape(nmo, nmo, nmo, nmo)

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
    rdm1_ao_spin: tuple[np.ndarray, np.ndarray] | None = None,
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
        rdm1_ao_spin: on an **open-shell** snapshot, the ``(alpha, beta)`` AO densities
            from :meth:`ProjectionEmbeddingAdapter.rdm1_ao_spin` -- each channel lifted
            through its *own* active space.  Required to reproduce the in-process energy
            there: each channel must be contracted against its own ``v_emb``/``P_B``,
            because SPADE partitions the spins separately, so the spin-summed operators
            carry cross-channel terms the solver never saw (see
            :meth:`ProjectionEmbeddingAdapter.projection_energy`).  ``None`` on a
            restricted snapshot, where the spin-summed form is exact.

    Returns:
        The same :class:`ProjectionEnergy` breakdown the in-process path returns.

    Raises:
        ValueError: the snapshot carries per-spin arrays (so it is open-shell) but
            ``rdm1_ao_spin`` was not passed.  Assembling it spin-summed would silently
            report a wrong energy -- on a stretched C-N bond, -24288 Ha against a ~-207 Ha
            system -- so this refuses rather than guessing.
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

    # An open-shell snapshot is identifiable by the per-spin arrays `export_state` adds
    # only when they exist.  Without the matching per-spin density there is nothing to
    # contract per channel, and the spin-summed fallback is not merely approximate here --
    # it is off by `mu` times the cross-channel overlap.  Refuse instead.
    is_open_shell = "p_b_spin" in state and "v_emb_spin" in state
    if is_open_shell and rdm1_ao_spin is None:
        raise ValueError(
            "this snapshot is open-shell (it carries per-spin P_B / v_emb), so "
            "projection_energy_from_state needs rdm1_ao_spin=(dm_alpha, dm_beta) with "
            "each channel lifted through its own active space. Assembling it from the "
            "spin-summed density would contract across channels whose spans are not "
            "S-orthogonal and report a plausible but badly wrong energy."
        )

    leak_spin: tuple[float, float] | None = None
    correction_spin: tuple[float, float] | None = None
    if rdm1_ao_spin is not None:
        # Mirrors the two conditions in `ProjectionEmbeddingAdapter.projection_energy`, so
        # the two implementations agree bit-for-bit rather than only to within the
        # summation-order difference between `tr[d_a P] + tr[d_b P]` and `tr[(d_a+d_b) P]`
        # (which `mu` amplifies to ~1e-10 in the projector, on a quantity that is a
        # numerical zero either way).
        #
        # Open shell -> each channel against its OWN operators, the rule the in-process
        # path documents.  `v_emb_spin_adapter` is the ADAPTER's per-channel `v_emb` pair
        # (`h_emb_s - hcore - P_B_s`); EmbASI's `v_emb_spin` in the same snapshot follows
        # the other convention and is NOT interchangeable with it.
        #
        # Restricted snapshot with a spin-resolved density -> the channels share one span,
        # so the spin-summed operators are exact and are what the live path uses too.
        dm_a_hl = np.asarray(rdm1_ao_spin[0], dtype=float)
        dm_b_hl = np.asarray(rdm1_ao_spin[1], dtype=float)
        if is_open_shell:
            p_b_pair = _array("p_b_spin")
            v_emb_pair = _array("v_emb_spin_adapter")
            init_a, init_b = _array("dm_a_spin_init")
            p_b_a, p_b_b = p_b_pair[0], p_b_pair[1]
            v_emb_a, v_emb_b = v_emb_pair[0], v_emb_pair[1]
        else:
            p_b_a = p_b_b = p_b
            v_emb_a = v_emb_b = v_emb
            if "dm_a_spin_init" in state:
                init_a, init_b = _array("dm_a_spin_init")
            else:
                init_a = init_b = 0.5 * _array("dm_a_init")
        leak_a = float(np.einsum("ij,ji->", dm_a_hl, p_b_a))
        leak_b = float(np.einsum("ij,ji->", dm_b_hl, p_b_b))
        corr_a = float(np.einsum("ij,ji->", dm_a_hl - init_a, v_emb_a))
        corr_b = float(np.einsum("ij,ji->", dm_b_hl - init_b, v_emb_b))
        leak_spin = (leak_a, leak_b)
        correction_spin = (corr_a, corr_b)
        leak = leak_a + leak_b
        correction = corr_a + corr_b
        v_emb_term = float(
            np.einsum("ij,ji->", dm_a_hl, v_emb_a) + np.einsum("ij,ji->", dm_b_hl, v_emb_b)
        )
    else:
        leak = float(np.einsum("ij,ji->", dm_hl, p_b))
        correction = float(np.einsum("ij,ji->", dm_hl - _array("dm_a_init"), v_emb))
        v_emb_term = float(np.einsum("ij,ji->", dm_hl, v_emb))

    e_high_a = float(solver_energy) - v_emb_term - leak

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
        correction_spin=correction_spin,
        projector_leak_spin=leak_spin,
    )


@dataclass(frozen=True)
class ProjectionEnergy:
    """Term-by-term breakdown of paper Eq. 8."""

    e_low_total: float  # E_L[γ^A + γ^B]
    e_low_A: float  # E_L[γ^A]
    e_high_A: float  # E_H[Ψ̃^A], embedding pot. removed, rebased to E_low(A)'s footing
    correction: float  # tr[(γ̃^A - γ^A) v_emb]
    # tr[γ̃^A P_B], a numerical zero when clean.  On a per-spin downfold this is the sum
    # of the per-channel traces (each density against its OWN projector), not a
    # contraction of the spin-summed density against the spin-summed projector -- the
    # latter includes cross-channel terms scaled by mu and is *not* a numerical zero.
    projector_leak: float
    footing_shift: float = 0.0  # nuclei/hcore rebasing applied to e_high_A (see below)
    # Per-spin split of the two density-linear terms, when the solver returned a
    # spin-resolved RDM and the adapter has a per-spin v_emb/P_B.  ``None`` on any
    # restricted path.  These are a *decomposition* of the spin-summed fields above, not
    # an alternative to them: each pair sums to its total (asserted in the tests), so
    # ``total`` is unchanged and remains the authoritative number.
    correction_spin: tuple[float, float] | None = None
    projector_leak_spin: tuple[float, float] | None = None

    @property
    def is_spin_resolved(self) -> bool:
        """True when the per-spin breakdown is available."""
        return self.correction_spin is not None

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
        # Populated only by relax_active_hf(); the relaxed embedded-HF Fock on A_HL
        # and its converged density.  None means "relaxation was never requested".
        self._fock_relaxed: np.ndarray | None = None
        self._dm_a_relaxed: np.ndarray | None = None
        self._fock_relaxed_spin: tuple[np.ndarray, np.ndarray] | None = None
        self._s: np.ndarray | None = None
        self._p_b: np.ndarray | None = None  # P_B, read from construct_embedding_potential
        # Per-spin counterparts, populated only by run_low_level on an unrestricted run.
        # None means "no spin-resolved information available", which every consumer of
        # these branches on rather than inspecting shapes.
        self._p_b_spin: tuple[np.ndarray, np.ndarray] | None = None
        self._v_emb_spin: tuple[np.ndarray, np.ndarray] | None = None
        self._dm_a_spin: tuple[np.ndarray, np.ndarray] | None = None
        # The (alpha, beta) halves of `_dm_a_init`, captured on the FIRST run_low_level
        # only (like `_dm_a_init` itself) and never refreshed -- `_dm_a_spin` tracks the
        # *current* cycle, so it cannot serve as the Eq. 8 reference.  Without this pair
        # the per-spin correction has to halve the spin-summed reference, which is wrong
        # whenever gamma^A is polarised: see `projection_energy`.
        self._dm_a_spin_init: tuple[np.ndarray, np.ndarray] | None = None
        self._dm_b_spin: tuple[np.ndarray, np.ndarray] | None = None
        self._fock_spin: tuple[np.ndarray, np.ndarray] | None = None
        # EmbASI's v_emb (= H^AB - H^A); on EmbASI's footing, i.e. it does NOT
        # carry subsystem A's nuclear-electron term (that sits in
        # A_LL.hamiltonian_estat_plus_xc) -- see the v_emb property.
        self._v_emb_embasi: np.ndarray | None = None

    # ---------------- low-level embedding ---------------- #
    def run_low_level_a_only(
        self,
        dma_in: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        dmb_in: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Re-run only the A layer at a fed-back density (the outer loop's inner step).

        ``dma_in``/``dmb_in`` accept a spin-summed ``(nao, nao)`` density or an
        ``(alpha, beta)`` tuple.  A pair is handed to EmbASI at ``n_spin=2`` and its
        channels are kept in ``_dm_a_spin``; the spin-summed total is stored in
        ``_dm_a`` either way, since that is what the Fock assembly, the energy and the
        export all read.
        """
        wrapped_dma_in = self._wrap_density(dma_in)

        self.p.A_LL.run_noscf(dm_in=wrapped_dma_in)

        # EmbASI returns SpinKpointArray objects (leading (nspin, nkpt) axes)
        # holding real restricted data in a complex128 dtype; _as_ao_matrix
        # squeezes the length-1 leading axes and drops the (asserted-negligible)
        # imaginary part, so everything downstream (einsum, sla.eigh, veff) sees
        # a bare 2-occupancy real (nao, nao) matrix.
        #
        # A fed-back pair is split: the total goes to `_dm_a` (every existing consumer),
        # the channels to `_dm_a_spin` (the per-spin downfold).  Storing the tuple in
        # `_dm_a` would break every einsum downstream.
        if isinstance(dma_in, tuple):
            self._dm_a_spin = (np.asarray(dma_in[0]), np.asarray(dma_in[1]))
            self._dm_a = self._dm_a_spin[0] + self._dm_a_spin[1]
        else:
            self._dm_a = dma_in
        if isinstance(dmb_in, tuple):
            self._dm_b_spin = (np.asarray(dmb_in[0]), np.asarray(dmb_in[1]))
            self._dm_b = self._dm_b_spin[0] + self._dm_b_spin[1]
        else:
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
        # Rebuild the per-spin Fock from the same (unchanged) v_emb/P_B pair whenever one
        # is available -- `A_LL`'s per-spin one-electron blocks have just been refreshed
        # by `run_noscf`, and the embedding potential is frozen for the A-only step.  This
        # is what lets a multi-cycle per-spin outer loop survive past cycle 1.  When no
        # pair is available (`restore_state`, or a restricted run) it clears to None, so
        # consumers still see an honest "no spin-resolved information" rather than a
        # stale pair from a previous cycle.
        self._assemble_fock_spin()
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

    def _assemble_fock_spin(self) -> None:
        """Assemble a per-spin ``F_emb`` pair, mirroring :meth:`_assemble_fock_a_only`.

        Same arithmetic as the spin-summed assembly -- ``h_kin + h_estat_xc + v_emb +
        P_B`` -- but per channel, off ``A_LL``'s own per-spin one-electron blocks.
        ``hamiltonian_estat_plus_xc`` genuinely differs between the channels (measured
        1.5e-01 on an OH radical): that difference *is* the spin polarisation a
        restricted downfold throws away.

        Sets ``self._fock_spin`` to ``None`` on a restricted run (or when EmbASI handed
        back no spin axis), so every consumer can branch on a single attribute.
        """
        if self._p_b_spin is None or self._v_emb_spin is None:
            self._fock_spin = None
            return
        kin = self._as_ao_pair(self.p.A_LL.hamiltonian_kinetic)
        estat = self._as_ao_pair(self.p.A_LL.hamiltonian_estat_plus_xc)
        if kin is None or estat is None:
            self._fock_spin = None
            return
        self._fock_spin = tuple(  # type: ignore[assignment]
            kin[s] + estat[s] + self._v_emb_spin[s] + self._p_b_spin[s] for s in (0, 1)
        )

    def _wrap_density(self, dm):
        """Wrap a density for EmbASI: an ``(alpha, beta)`` pair at ``n_spin=2``, else 1.

        Lets the outer loop feed back either a spin-summed total (restricted, unchanged)
        or a genuine per-spin pair without the caller knowing which wrapper EmbASI needs.
        """
        if dm is None:
            return None
        if isinstance(dm, tuple):
            if len(dm) != 2:
                raise ValueError(f"a density pair must be (alpha, beta); got {len(dm)}")
            return self._as_spin_kpoint_pair(dm[0], dm[1])
        return self._as_spin_kpoint_array(dm)

    def run_low_level(
        self,
        dma_in: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        dmb_in: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        a_nmos: int | None = None,
    ) -> None:
        """Drive EmbASI: supersystem SCF, SPADE/PM localisation, F_emb.

        ``dma_in``/``dmb_in`` accept either a spin-summed ``(nao, nao)`` density or an
        ``(alpha, beta)`` tuple; the pair is handed to EmbASI at ``n_spin=2`` so an
        unrestricted outer loop can feed back genuine spin resolution.
        """
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
        # EmbASI wants these as SpinKpointArrays, not the plain (nao, nao) density the
        # outer loop feeds back -- wrap them (mirror of the _as_ao_matrix squeeze on the
        # way out) so the multi-cycle loop runs against real EmbASI.  A *pair* is wrapped
        # at n_spin=2, which EmbASI reads as two genuine channels; a single matrix keeps
        # the restricted n_spin=1 shape.
        wrapped_dma_in = self._wrap_density(dma_in)
        wrapped_dmb_in = self._wrap_density(dmb_in)
        dm_a, dm_b, _overlap, v_emb_embasi, p_b_embasi = self.p.construct_embedding_potential(
            dma_in=wrapped_dma_in, dmb_in=wrapped_dmb_in, a_nspade_mos=a_nmos
        )

        # EmbASI returns SpinKpointArray objects (leading (nspin, nkpt) axes)
        # holding real restricted data in a complex128 dtype; _as_ao_matrix
        # squeezes the length-1 leading axes and drops the (asserted-negligible)
        # imaginary part, so everything downstream (einsum, sla.eigh, veff) sees
        # a bare 2-occupancy real (nao, nao) matrix.
        #
        # `_as_ao_total` SUMS the spin axis here, deliberately: these spin-summed
        # copies are what the restricted path, the energy assembly and the export all
        # read.  The per-spin halves are kept ALONGSIDE them just below
        # (`_dm_a_spin`/`_dm_b_spin` via `_as_ao_pair`), which is what the per-spin
        # downfold uses -- so nothing is lost.  Note `dm_a`/`dm_b` are subsystems A and
        # B, NOT alpha/beta: each carries its own spin axis, so an unrestricted run has
        # four blocks.
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

        # Per-spin blocks kept ALONGSIDE the spin-summed ones above (which the
        # restricted path and the energy assembly still use unchanged).  These are what
        # a spin-dependent downfold needs; each is None on a restricted run.
        self._p_b_spin = self._as_ao_pair(p_b_embasi)
        self._v_emb_spin = self._as_ao_pair(v_emb_embasi)
        self._dm_a_spin = self._as_ao_pair(dm_a)
        self._dm_b_spin = self._as_ao_pair(dm_b)
        # Captured once, on the first run_low_level ever -- the per-spin counterpart of
        # `_dm_a_init` above, and set under the same condition so the two always describe
        # the same cycle.  `_dm_a_spin` is refreshed every cycle and must not be used as
        # the reference (see `projection_energy`'s per-spin split).
        if self._dm_a_spin_init is None and self._dm_a_spin is not None:
            self._dm_a_spin_init = (self._dm_a_spin[0].copy(), self._dm_a_spin[1].copy())

        # `_assemble_fock_a_only` also rebuilds the per-spin pair (see its comment).
        self._assemble_fock_a_only()

    def relax_active_hf(self) -> None:
        """Converge subsystem A's embedded-HF reference on ``A_HL``, in the frozen
        ``v_emb``/``P_B`` potential :meth:`run_low_level` already built.

        ``self._fock`` (from :meth:`run_low_level`) is the *low-level* (``xc_ll``)
        ``A_LL`` Fock plus the embedding potential, diagonalized once and never fed
        back into itself -- so Brillouin's theorem does not hold for the orbitals it
        returns, and the reference handed to FCI/SQD carries an orbital-relaxation
        error that a formally-exact active-space solve cannot recover (the solver
        optimizes the CI coefficients, not the orbitals).

        For ``xc_hl="HF"`` this adapter's ``A_HL`` calculator IS the intended
        high-level reference, and EmbASI already owns a converged embedded SCF for
        it -- ``ProjectionEmbedding.freeze_and_thaw``'s final post-processing step
        (``embasi/embedding.py:790-794``), invoked here as a standalone call on the
        frozen potential :meth:`run_low_level` already produced.  ``run_emb_scf``
        adds ``emb_pot + proj_pot`` into the Fock every cycle and delegates
        DIIS/damping to the backend's own SCF, so no separate loop is needed here.

        Sets ``self._fock_relaxed`` / ``self._dm_a_relaxed``; ``self._fock`` /
        ``self._dm_a`` (still read by :attr:`h_emb`, :attr:`v_emb`, the export and
        the DFT-in-DFT path) are untouched.  ``h_emb = h_core + v_emb + P_B`` is an
        identity of the embedding potential rather than of whichever Fock was
        diagonalized, so it needs no relaxed counterpart.  Call
        ``build_orbitals(use_relaxed=True)`` (or
        ``build_orbitals_apc_concentric(use_relaxed=True)``) afterwards to
        diagonalize this instead of the low-level Fock.
        """
        self.p.A_HL.input_fragment_nelectrons = self.p.A_pop
        if getattr(self.p, "A_spin", None) is not None:
            # Target the right sector: without this the embedded SCF fills alpha/beta by
            # aufbau on the *total* count and can converge to the wrong spin state.
            self.p.A_HL.input_fragment_spin = self.p.A_spin
        # Hand EmbASI the per-spin density/potential/projector when there is one -- the
        # single-channel wrapper would feed a spin-summed density into an unrestricted SCF
        # and silently relax against the wrong reference.  `_wrap_density` picks n_spin=2
        # for a pair and n_spin=1 for a matrix.
        spin_pairs = (
            self._dm_a_spin is not None
            and self._v_emb_spin is not None
            and self._p_b_spin is not None
        )
        if spin_pairs:
            dm_in = self._wrap_density(self._dm_a_spin)
            emb_pot = self._wrap_density(self._v_emb_spin)
            proj_pot = self._wrap_density(self._p_b_spin)
        else:
            dm_in = self._wrap_density(self._dm_a_arr)
            emb_pot = self._wrap_density(self._require(self._v_emb_embasi, "v_emb"))
            proj_pot = self._wrap_density(self.p_b)
        self.p.A_HL.run_emb_scf(dm_in=dm_in, emb_pot=emb_pot, proj_pot=proj_pot)

        # Same assembly as _assemble_fock_a_only's self._fock, but off A_HL's
        # converged one-electron blocks instead of A_LL's frozen ones (mirrors
        # ProjectionEmbedding.construct_embedded_fock, embasi/embedding.py:828).
        h_kin_a = self._as_ao_total(self.p.A_HL.hamiltonian_kinetic)
        h_estat_xc_a = self._as_ao_total(self.p.A_HL.hamiltonian_estat_plus_xc)
        self._fock_relaxed = h_kin_a + h_estat_xc_a + self._v_emb_embasi + self._p_b
        self._dm_a_relaxed = self._as_ao_total(self.p.A_HL.density_matrices_out)

        # Relaxed per-spin Fock, so `build_orbitals_spin` can use the relaxed reference
        # too.  Same assembly as `_assemble_fock_spin`, off A_HL's per-spin blocks.
        kin = self._as_ao_pair(self.p.A_HL.hamiltonian_kinetic)
        estat = self._as_ao_pair(self.p.A_HL.hamiltonian_estat_plus_xc)
        v_spin, p_spin = self._v_emb_spin, self._p_b_spin
        if kin is not None and estat is not None and v_spin is not None and p_spin is not None:
            # Locals so the None-narrowing survives into the comprehension.
            self._fock_relaxed_spin = (
                kin[0] + estat[0] + v_spin[0] + p_spin[0],
                kin[1] + estat[1] + v_spin[1] + p_spin[1],
            )
        else:
            self._fock_relaxed_spin = None

    _STATE_ARRAYS = ("_dm_a", "_dm_a_init", "_dm_b", "_fock", "_s", "_p_b", "_v_emb_embasi")
    # Per-spin counterparts, exported as a SEPARATE optional set: they are `None` on any
    # restricted run, so requiring them (as `_STATE_ARRAYS` does) would break every
    # closed-shell export.  Each is stored flattened to `(2, nao, nao)`.
    _STATE_SPIN_ARRAYS = (
        "_p_b_spin",
        "_v_emb_spin",
        "_dm_a_spin",
        "_dm_b_spin",
        "_fock_spin",
        # Carried for the same reason `_dm_a_init` is (see `export_state`): a fresh
        # process re-runs the SCF, and without the round-0 pair the per-spin correction
        # silently falls back to halving a polarised reference.
        "_dm_a_spin_init",
    )

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
            "xc_hl": str(getattr(mf, "xc", None)),
            "density_fit": str(getattr(self.ints, "_density_fit", None)),
            # Subsystem A's own spin, as the SPADE partition assigned it.  Neither
            # `unrestricted` (a bool) nor `mol.spin` (the *supersystem*'s) captures it,
            # so without this a doublet snapshot and a singlet adapter are
            # indistinguishable: same basis, same geometry, same array shapes, different
            # meaning for every exported matrix.  None when EmbASI exposed no A_spin.
            "a_spin": (
                None if (a := getattr(getattr(self, "p", None), "A_spin", None)) is None else int(a)
            ),
        }
        # WHICH orbitals are subsystem A, not just how many basis functions there are.
        # `a_nmos` (CLI `--a_nmos`, forwarded as `a_nspade_mos`) moves the SPADE cut, so
        # two runs on an identical molecule/basis/mu/xc partition differently and produce
        # snapshots that are otherwise fingerprint-identical: same `nao`, same shapes,
        # integral `tr(gamma^A S)` and `tr(gamma^B S)` on both sides, and a clean
        # projector leak after the cross-restore -- every existing guard passes while the
        # energy moves ~1.4 Ha (859 kcal/mol, measured on H6/sto-3g at A=2 vs A=1).
        # The electron count in A is the cheapest faithful observable of the cut: it is an
        # integer by construction, needs no MO layout assumption, and differs exactly when
        # the partition does.
        try:
            n_a = float(np.einsum("ij,ji->", self._dm_a_arr, np.asarray(self.ints.overlap())))
            fingerprint["a_nelec"] = int(round(n_a))
        except (AttributeError, ValueError):
            # Before `run_low_level` there is no density to measure; a fingerprint taken
            # then simply omits the field, and two such fingerprints still compare equal.
            fingerprint["a_nelec"] = None
        if mol is not None:
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
            e_low_total, e_low_a = self._low_level_energies()
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
        hcore_a, enuc_a = self._a_fragment_footing()
        e_low_ab, e_low_a = self._low_level_energies()
        state["v_emb"] = np.asarray(self.v_emb)
        state["h_emb"] = np.asarray(self.h_emb)
        state["hcore_a"] = np.asarray(hcore_a)
        state["hcore_full"] = np.asarray(self.ints.hcore())
        state["enuc_a"] = np.asarray(float(enuc_a))
        state["enuc_full"] = np.asarray(float(self.ints.energy_nuc()))
        state["e_low_total"] = np.asarray(e_low_ab)
        state["e_low_a"] = np.asarray(e_low_a)
        # Per-spin state, when there is any.  Without this an out-of-process open-shell
        # loop silently degrades to the spin-summed arrays -- it would restore, assemble a
        # plausible restricted Fock and report a wrong energy with nothing to flag it.
        for name in self._STATE_SPIN_ARRAYS:
            pair = getattr(self, name, None)
            if pair is not None:
                state[name.lstrip("_")] = np.asarray(np.stack(pair))
        # The ADAPTER's per-channel `v_emb` pair (`h_emb_s - hcore - P_B_s`), which
        # `projection_energy` contracts each channel against.  Exported under its own key
        # because `v_emb_spin` above is EmbASI's pair, on the other convention (it omits
        # subsystem A's nuclear-electron term); the two are not interchangeable, and
        # `projection_energy_from_state` needs this one to reproduce the live energy.
        if self._fock_spin is not None and self._p_b_spin is not None:
            state["v_emb_spin_adapter"] = np.asarray(
                np.stack((self.v_emb_spin(0), self.v_emb_spin(1)))
            )
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

        # Restore the per-spin pairs when the snapshot carries them, and clear them when
        # it does not -- so a restricted snapshot cannot leave a previous run's spin state
        # standing, and an open-shell snapshot does not silently lose it.
        for name in self._STATE_SPIN_ARRAYS:
            key = name.lstrip("_")
            if key in state:
                stacked = np.asarray(state[key], dtype=float)
                if stacked.shape != (2, nao, nao):
                    raise ValueError(
                        f"snapshot {key!r} has shape {stacked.shape}, expected "
                        f"{(2, nao, nao)}; refusing to assemble from a mismatched basis"
                    )
                setattr(self, name, (stacked[0], stacked[1]))
            else:
                setattr(self, name, None)

        # The relaxed embedded-HF reference is NOT part of the snapshot, and after a
        # restore it is worse than absent: `relax_active_hf` converges it "in the frozen
        # v_emb/P_B potential run_low_level already built", and `restore_state` has just
        # replaced v_emb/P_B/the densities with the SENDER's.  So a surviving
        # `_fock_relaxed` was converged against a potential this adapter no longer holds --
        # the frozen-potential invariant that justifies reusing it in-process is exactly
        # what a restore breaks.  Measured: `max|_fock_relaxed - _fock| = 0.137` with
        # orbital energies 0.982 Ha apart, consumed without complaint by
        # `build_orbitals(use_relaxed=True)`.
        #
        # Clearing is the same rule `_STATE_SPIN_ARRAYS` follows just above ("a restricted
        # snapshot cannot leave a previous run's spin state standing"), and it fails loudly
        # rather than silently: `_fock_relaxed_arr` raises via `_require`, and
        # `build_orbitals_spin(use_relaxed=True)` already refuses a `None` pair.  Call
        # `relax_active_hf()` again after restoring to rebuild it in the restored potential.
        self._fock_relaxed = None
        self._dm_a_relaxed = None
        self._fock_relaxed_spin = None

        self._assemble_fock_a_only()
        # `_assemble_fock_a_only` rebuilds `_fock_spin` from the restored v_emb/P_B pair
        # when one is present, so a restored open-shell adapter can downfold per spin.
        if "fock_spin" in state and self._fock_spin is None:
            raise ValueError(
                "the snapshot carried a per-spin Fock but it could not be reassembled; "
                "the per-spin v_emb/P_B or A_LL blocks are missing, so a per-spin "
                "downfold would silently fall back to the spin-summed one"
            )

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
    def _dm_a_for_veff(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """gamma^A as the argument to ``veff_ll``/``veff_hl``: the PAIR when there is one.

        The reduction of a ``(2, nao, nao)`` veff *return* is the mean
        (:meth:`PySCFIntegrals._spin_average`); this is the separate question of what
        goes **in**.  Handing an unrestricted ``get_veff`` the spin-summed
        ``_dm_a_arr`` is not a harmless simplification: PySCF sees a 2-D array, warns
        "Incompatible dm dimension. Treat dm as RHF density matrix.", and silently
        substitutes ``d/2`` for *both* channels -- i.e. it throws the spin
        polarisation away and reconstructs a fictitious unpolarised density.

        For an **HF/HFhybrid-exchange** low level that substitution is exactly
        harmless, which is why it went unnoticed: ``veff_s = J[d_a + d_b] - K[d_s]``
        is linear in the densities, so the mean is
        ``J[d_tot] - 0.5(K[d_a] + K[d_b]) == J[d_tot] - 0.5 K[d_tot]`` either way
        (verified to 3e-14 on a quintet Fe and triplet CH2).

        For a **KS** low level it is not, because the xc functional is nonlinear in
        the spin densities: ``e_xc[d_a, d_b] != e_xc[d/2, d/2]`` whenever the channels
        differ.  Measured on triplet CH2 at ``xc_ll=PBE`` -- this repo's default --
        ``max|veff_ll| `` differs by 0.0219 Ha and ``tr[gamma^A Delta veff_ll]`` by
        0.0652 Ha (40.9 kcal/mol), entering ``h_emb``, every ``h_emb_s``, and hence
        every downfolded Hamiltonian and ``v_emb``.

        Restricted runs are bit-identical: ``_dm_a_spin`` is ``None`` there, so this
        returns the same matrix as before.  On a closed shell the pair and the summed
        input agree anyway (checked: both match a restricted ``RKS.get_veff`` to the
        SCF residual), so the correction is confined to genuinely polarised channels.
        """
        # `getattr`: the stub adapters in the tests are built via `object.__new__` and
        # never run `__init__`, so the attribute may not exist at all (the same reason
        # `projection_energy` reads `_dm_a_spin_init` this way).
        pair = getattr(self, "_dm_a_spin", None)
        if pair is not None:
            return pair
        return self._dm_a_arr

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

    @property
    def _fock_relaxed_arr(self) -> np.ndarray:
        """The relaxed embedded-HF Fock on A_HL (requires :meth:`relax_active_hf`)."""
        return self._require(
            self._fock_relaxed,
            "the relaxed embedded-HF Fock matrix (call relax_active_hf() first)",
        )

    def _spin_is_unknown(self) -> bool:
        """True when EmbASI exposed no ``A_spin`` at all.

        Distinguishes the two reasons :meth:`_n_occ_b_from_embasi` returns ``None``: a
        reported ``2S == 0`` (a real singlet -- the restricted reading is exact) from a
        missing attribute (no evidence either way -- the restricted reading is a guess).
        Only the latter should block the downfold.
        """
        return getattr(getattr(self, "p", None), "A_spin", None) is None

    def _n_occ_b_from_embasi(self, n_occ: int) -> int | None:
        """Beta occupied count of subsystem A, from EmbASI's ``A_spin``.

        EmbASI sets ``A_spin = round(A_pop_alpha - A_pop_beta)`` inside
        ``construct_embedding_potential`` (which :meth:`run_low_level` already calls),
        so it is populated by the time orbitals are built.  Its SPADE forces the beta
        cutoff from the alpha partition, so subsystem A carries the whole supersystem
        spin and B nets to zero *by construction* -- which is why this is a sounder
        source than the per-spin occupied counts ``spade_localisation`` still does not
        return (``rot_evecs_occ_a`` comes back at full MO width, so the count is not
        recoverable from the array shape).

        ``n_occ`` is the *alpha* count: :meth:`_as_ao_by_mo` keeps the alpha channel as
        the representative, so ``mo_a_ll.shape[1]`` counts alpha orbitals.  Hence
        ``n_beta = n_occ - A_spin``.

        Returns ``None`` -- the restricted reading, ``n_occ`` doubly-occupied orbitals --
        when ``A_spin`` is unavailable (an older EmbASI, or a read before
        ``construct_embedding_potential``) or is ``0``.  A ``2S == 0`` unrestricted
        singlet has ``n_alpha == n_beta``, which the restricted reading represents
        exactly, so refusing it would reject a well-posed run.

        Raises:
            ValueError: if the implied beta count falls outside ``[0, n_occ]``.  That is
                a partition/MO disagreement, and passing it through would hand the
                solver a valid-looking wrong spin sector.
        """
        # EmbASI declares no __init__ default for A_spin (it is only assigned inside
        # construct_embedding_potential), so a read before that call raises
        # AttributeError; getattr also buys tolerance for a pre-spin EmbASI for free.
        # The outer getattr covers a stub adapter built without `p` at all (the selector
        # tests construct one via object.__new__, bypassing __init__).
        a_spin = getattr(getattr(self, "p", None), "A_spin", None)
        if a_spin is None:
            return None
        a_spin = int(a_spin)
        if a_spin == 0:
            return None
        n_occ_b = n_occ - a_spin
        if not 0 <= n_occ_b <= n_occ:
            raise ValueError(
                f"EmbASI reports A_spin={a_spin} for subsystem A, implying n_beta="
                f"{n_occ_b} against n_alpha={n_occ}; that is outside [0, {n_occ}] and "
                "means the SPADE partition and the MO coefficients disagree. Refusing "
                "to hand the solver a wrong spin sector."
            )
        return n_occ_b

    def _as_ao_pair(self, m) -> tuple[np.ndarray, np.ndarray] | None:
        """``(alpha, beta)`` blocks when ``m`` genuinely carries two spin channels.

        The counterpart to :meth:`_as_ao_total`, which sums them.  Returns ``None`` for
        a restricted block (or a restricted adapter), so a caller can branch on "is
        there per-spin information here" without inspecting shapes itself.

        Keeping the pair is what makes an open-shell downfold possible at all: EmbASI's
        SPADE partitions the two channels independently, so ``span(A_alpha)`` is
        S-orthogonal to ``span(B_alpha)`` but **not** to ``span(B_beta)`` (measured
        8.6e-15 vs 2.7e-04 on an OH radical).  A spin-summed ``P_B`` therefore does not
        annihilate either channel's A orbitals, while each per-spin ``P_B`` annihilates
        its own to ~3e-16.
        """
        if not self.unrestricted:
            return None
        arr = np.asarray(m)
        while arr.ndim > 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] == 2:
            a, b = self._as_ao_matrix_spin(arr)
            return np.ascontiguousarray(a), np.ascontiguousarray(b)
        return None

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
        # Restricted wrapper: one channel at key (0, 0).  The per-spin counterpart is
        # `_as_spin_kpoint_pair`, which EmbASI reads as a genuine pair.
        return SpinKpointArray({(0, 0): block}, n_spin=1, n_kpoints=1)

    @staticmethod
    def _as_spin_kpoint_pair(dm_a: np.ndarray, dm_b: np.ndarray):
        """Wrap an ``(alpha, beta)`` density pair as an ``n_spin=2`` ``SpinKpointArray``.

        The write-side counterpart of :meth:`_as_spin_kpoint_array`.  EmbASI's PySCF
        adapter reads ``density_matrix_in[0, 0]`` **and** ``[1, 0]`` as a genuine pair
        when its mean field is UHF/ROHF-shaped (``qmcode_adapters``: ``n_spins = 2 if
        isinstance(mf, (UHF, ROHF))``), so this is the shape an unrestricted feedback
        must hand back -- a spin-summed total in a one-channel wrapper silently discards
        the polarisation the solver just computed.

        Falls back to a ``(2, 1, nao, nao)`` ndarray without EmbASI, indexable as
        ``[0, 0]`` / ``[1, 0]`` in the same way, for the mock-driven tests.
        """
        a = np.ascontiguousarray(np.asarray(dm_a))
        b = np.ascontiguousarray(np.asarray(dm_b))
        if a.shape != b.shape:
            raise ValueError(
                f"alpha and beta densities must share a shape; got {a.shape} and {b.shape}"
            )
        try:
            from embasi.ks_array import SpinKpointArray
        except ImportError:
            return np.stack([a, b])[:, np.newaxis, :, :]
        return SpinKpointArray({(0, 0): a, (1, 0): b}, n_spin=2, n_kpoints=1)

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

    def _mo_spin(self, container, ispin: int) -> np.ndarray:
        """One spin channel's MO coefficients, S-orthonormality checked.

        The per-spin counterpart of :meth:`_as_ao_by_mo`.  The channels have different
        widths on an open shell (measured ``(13, 5)`` alpha vs ``(13, 4)`` beta), so each
        must be taken out of the ``SpinKpointArray`` by index rather than converted as a
        block -- see :meth:`_spin_block`.
        """
        c = self._real(np.asarray(container[ispin, 0]))
        nao = self._s_arr.shape[0]
        if c.shape[0] != nao and c.shape[1] == nao:
            c = c.T
        gram = c.T @ self._s_arr @ c
        if not np.allclose(gram, np.eye(c.shape[1]), atol=1e-6):
            raise ValueError(
                f"spin-{ispin} MO coefficients are not S-orthonormal (max deviation "
                f"{np.abs(gram - np.eye(c.shape[1])).max():.2e})"
            )
        return np.ascontiguousarray(c)

    def _eigh_subsystem_a_spin(
        self, ispin: int, *, use_relaxed: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalize spin ``ispin``'s ``F_emb`` in **that spin's own** span(A).

        The whole point of the per-spin path.  ``_eigh_subsystem_a`` builds its
        S-orthogonal complement from the alpha ``mo_b_ll`` and diagonalizes a
        spin-summed Fock; here both come from the same channel, so the resulting
        orbitals are annihilated by *that channel's* ``P_B``.  Measured leak: ~1.5e-10,
        against ~2e-02 for the mixed-channel version.
        """
        s = self._s_arr
        c_b = self._mo_spin(self.p.mo_coeffs_B_LL, ispin)
        pair = self._fock_relaxed_spin if use_relaxed else self._fock_spin
        fock = self._require(
            None if pair is None else pair[ispin],
            f"the spin-{ispin} embedded Fock (needs an unrestricted run_low_level)",
        )
        proj = np.eye(s.shape[0]) - c_b @ (c_b.T @ s)
        chol = np.linalg.cholesky(s)
        x_all = sla.solve_triangular(chol.T, np.eye(s.shape[0]), lower=False)
        y = proj @ x_all
        gram = y.T @ s @ y
        w, u = np.linalg.eigh(gram)
        nonzero = w > 1e-8
        q = y @ u[:, nonzero] @ np.diag(1.0 / np.sqrt(w[nonzero]))
        eps, cc = np.linalg.eigh(q.T @ fock @ q)
        return eps, q @ cc

    def _spin_block(self, c):
        """Reduce a (possibly ragged) spin-resolved MO container to one channel.

        EmbASI hands MO coefficients back as a ``SpinKpointArray`` keyed by
        ``(ispin, ikpt)``.  On an open shell the two spin blocks have different
        widths (alpha and beta carry different occupied counts), so the container is
        not a rectangular array and must be indexed rather than converted.

        Returns the alpha channel when ``unrestricted`` and a length-2 spin axis is
        present -- the conventional representative for the spin-*restricted* downfold
        this adapter still performs; the beta width is recovered separately from
        EmbASI's ``A_spin`` (see :meth:`_n_occ_b_from_embasi`).  Anything else is
        passed through untouched for the existing code paths to handle.
        """
        if not getattr(self, "unrestricted", False):
            return c
        # NB `SpinKpointArray` stores the count as `n_spins` (plural) even though its
        # constructor keyword is `n_spin`; accept either so a rename upstream cannot
        # silently turn this back into the ragged-array crash.
        n_spin = getattr(c, "n_spins", None)
        if n_spin is None:
            n_spin = getattr(c, "n_spin", None)
        if n_spin == 2:
            try:
                return c[0, 0]
            except (TypeError, IndexError, KeyError):  # pragma: no cover - layout guard
                return c
        return c

    def _as_ao_by_mo(self, c) -> np.ndarray:
        """Fix layout: ASI/Fortran may hand back (nmo, nao) or a leading spin axis."""
        # Take the spin channel BEFORE np.asarray.  On a live open shell EmbASI's
        # per-spin MO blocks have DIFFERENT widths -- measured (13, 5) alpha vs
        # (13, 4) beta for an OH radical in sto-3g, since SPADE slices each channel
        # at its own occupied count -- so the pair is a *ragged* nested sequence and
        # `np.asarray` on it raises "inhomogeneous shape" before any spin handling
        # below can run.  `_spin_block` indexes the SpinKpointArray instead, which is
        # width-agnostic.  Without this, `mo_a_ll` cannot be read at all on a doublet
        # and `build_orbitals` dies before it reaches the spin sector logic.
        c = self._spin_block(c)
        c = np.asarray(c)
        if c.ndim == 3 and c.shape[0] == 2 and getattr(self, "unrestricted", False):
            # Spin-resolved caller asked for one set through the restricted accessor:
            # the alpha channel is the conventional representative.  (Reached only for
            # an equal-width pair; a ragged one was already reduced by `_spin_block`.)
            # Per-channel access is `_mo_spin`, used by the per-spin downfold.
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

        ``F_emb - veff_ll(gamma^A)`` strips the mean field EmbASI folded into
        ``F_emb`` back off, leaving the one-body operator on the adapter's PySCF
        ``h_core`` footing (the footing the downfold and the bare-electronic
        solver Hamiltonian use).  ``F_emb`` itself is now assembled in
        :meth:`run_low_level` from EmbASI's *exported* ``v_emb`` and ``P_B`` (plus
        the ``A_LL`` one-electron blocks), so this is no longer the inverse of an
        opaque ``construct_embedded_fock`` -- the projector it subtracts back out
        via :attr:`p_b` is EmbASI's own exported ``P_B``, not a reconstruction.

        **It is the LOW-level veff, not the high-level one.**  The mean field baked
        into ``F_emb`` comes from ``A_LL.hamiltonian_estat_plus_xc`` -- the
        ``xc_ll`` calculator's -- because that is what
        :meth:`_assemble_fock_a_only` (mirroring EmbASI's
        ``construct_embedded_fock``) adds.  Subtracting ``veff_hl`` instead leaves
        the residual ``veff_ll - veff_hl`` sitting in ``h_emb``, hence in
        ``v_emb`` and in every downfolded Hamiltonian: a silent one-body error
        that vanishes only in the ``xc_hl == xc_ll`` case (where the two veffs
        coincide) and so survives any test that does not vary the two levels
        independently.
        """
        return self._fock_arr - self.ints.veff_ll(self._dm_a_for_veff)

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

    def v_emb_spin(self, ispin: int) -> np.ndarray:
        """One spin channel's embedded one-body potential, on the PySCF ``h_core`` footing.

        The per-channel analogue of :attr:`v_emb`, derived the same way: that channel's
        ``F_emb`` minus the mean field, ``h_core`` and that channel's ``P_B``.

        **Not an additive decomposition of :attr:`v_emb`.**  ``h_emb`` subtracts
        ``h_core`` and ``veff_ll`` *once* to build the spin-summed operator, so summing
        two channels subtracts them twice: measured
        ``|v_emb - (v_a + v_b)| ~ 33 Ha`` on an OH radical (24.2 Ha on the stretched
        butyronitrile in ``data/22.inp``).

        That is a property of ``v_emb``, **not** a reason to avoid this pair.  Each
        channel is the right operator to contract against *that channel's* density, which
        is precisely what :meth:`embedded_hamiltonian_spin` folds into ``h1_s`` and
        therefore what :meth:`projection_energy` must subtract back off.  Verified:
        ``h_core + v_emb_spin(s) + P_B_s`` reproduces the ``h_emb_s`` the downfold used to
        ``0.0`` — while the same reconstruction with a *per-channel* ``veff_ll`` is 4.18 Ha
        out, which is why the spin-summed ``veff_ll`` above is the correct convention here.

        The spin-summed ``v_emb`` is the wrong operator for a per-spin density: it removes
        a quantity the solver never added (+54.0 Ha on ``data/22.inp``).  See
        :meth:`projection_energy`, which now derives its spin-summed ``correction`` /
        ``projector_leak`` *from* the channels rather than contracting across them.
        """
        fock = self._require(
            None if self._fock_spin is None else self._fock_spin[ispin],
            f"the spin-{ispin} embedded Fock (needs an unrestricted run_low_level)",
        )
        p_b_s = self._require(
            None if self._p_b_spin is None else self._p_b_spin[ispin],
            f"the spin-{ispin} projector P_B",
        )
        h_emb_s = fock - self.ints.veff_ll(self._dm_a_for_veff)
        return h_emb_s - self.ints.hcore() - p_b_s

    # ---------------- orbital construction ---------------- #
    def _eigh_subsystem_a(self, fock: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Diagonalize ``fock`` (default F_emb) inside span(A), not the full AO basis.

        span(B) is spanned by the localized environment orbitals ``mo_b_ll``, so
        its S-orthogonal complement *is* span(A).  Build an S-orthonormal basis
        ``Q`` of that complement (dim ``nao - n_occ_B``), solve the small
        standard eigenproblem ``(Q^T fock Q) u = eps u``, and map back
        ``C = Q u``.  The returned orbitals are S-orthonormal by construction and
        the level shift no longer appears (B is gone), so no ``eps < floor`` cut
        is needed.  For a large environment this replaces an O(nao^3) generalized
        solve with one on the much smaller A block.

        ``fock`` defaults to :attr:`_fock_arr` (the low-level F_emb); pass
        :attr:`_fock_relaxed_arr` to diagonalize the relaxed embedded-HF reference
        instead (see :meth:`relax_active_hf`).  Only ``mo_b_ll``/``_s_arr`` build
        ``Q``, so the S-orthogonal complement of span(B) is unaffected by which
        Fock is subsequently diagonalized within it.

        Falls back to nothing: if ``mo_b_ll`` is unavailable the caller uses the
        full-basis path via ``restrict_to_a=False``.
        """
        fock = self._fock_arr if fock is None else fock
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
        fs = q.T @ fock @ q
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
        use_relaxed: bool = False,
    ) -> EmbeddedOrbitals:
        """Orbitals of subsystem A from the generalized problem F_emb C = S C eps.

        ``use_relaxed=True`` diagonalizes the relaxed embedded-HF Fock from
        :meth:`relax_active_hf` (:attr:`_fock_relaxed_arr`) instead of the one-shot
        low-level F_emb (:attr:`_fock_arr`).  The downfold itself
        (:meth:`embedded_hamiltonian`) is unaffected by the choice: it reads
        :attr:`h_emb`, the ``h_core + v_emb + P_B`` identity of the embedding
        potential, not ``F_emb`` directly, so it needs no matching flag.

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
        fock = self._fock_relaxed_arr if use_relaxed else self._fock_arr
        if restrict_to_a:
            eps, c = self._eigh_subsystem_a(fock)
        else:
            eps, c = sla.eigh(fock, self._s_arr)
            keep = eps < self._floor  # level shift removes subsystem B
            eps, c = eps[keep], c[:, keep]

        n_occ = self.mo_a_ll.shape[1]  # inferred, never passed in (the ALPHA count)

        # Beta occupied count from EmbASI's A_spin, or None for the restricted reading.
        # `_as_ao_by_mo` still keeps only the alpha channel, so the downfold below stays
        # spin-RESTRICTED (one h1, one MO set) -- this fixes the reported *sector*, not
        # the orbitals, so the result is ROHF-like rather than UKS-quality.
        n_occ_b = self._n_occ_b_from_embasi(n_occ)

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
        return EmbeddedOrbitals(
            coeff=c,
            energy=eps,
            n_occ=n_occ,
            inactive=inactive,
            active=active,
            n_occ_b=n_occ_b,
        )

    def _occupation_pattern(
        self, c_full: np.ndarray, n_orb: int, n_occ: int, n_occ_b: int | None
    ) -> np.ndarray:
        """``{0, 1, 2}`` occupation per column of ``c_full``, rotation-invariantly.

        Projects the per-spin subsystem-A densities onto the columns
        (:func:`~embasi_qiskit_integration.selectors.column_occupation_from_density`)
        rather than assuming "the first ``n_occ`` columns are doubly occupied".  That
        positional rule holds only for canonical orbitals in energy order, which
        concentric localization has just destroyed.

        Falls back to the positional
        :func:`~embasi_qiskit_integration.selectors.somo_occupation_pattern` when no
        per-spin density is available (``run_low_level`` on a restricted adapter), which
        reproduces the previous behaviour exactly -- there is no rotation-invariant
        answer to give without a spin-resolved density, and the fallback is at least
        explicit about which reading it used.
        """
        from embasi_qiskit_integration.selectors import (
            column_occupation_from_density,
            somo_occupation_pattern,
        )

        if self._dm_a_spin is not None:
            dm_a, dm_b = self._dm_a_spin
            pattern = column_occupation_from_density(c_full, dm_a, dm_b, self._s_arr)
            # Cross-check against the count EmbASI reported: a disagreement means the
            # projection and the SPADE partition see different spin sectors, which
            # would hand APC a plausible but wrong set of orbitals to protect.
            n_a = int((pattern >= 1).sum())
            n_b = int((pattern == 2).sum())
            if (n_a, n_b) != (n_occ, n_occ_b):
                warnings.warn(
                    f"density-projected occupation implies (n_alpha, n_beta) = "
                    f"({n_a}, {n_b}) but the partition reports ({n_occ}, {n_occ_b}); "
                    "using the projected pattern (it is rotation-invariant) but the "
                    "two should agree -- check the SPADE partition.",
                    stacklevel=3,
                )
            return pattern
        if n_occ_b is None:
            # `is_open_shell` gates this method's only caller, so n_occ_b is set there;
            # be explicit rather than passing None into a positional int.
            raise ValueError(
                "an occupation pattern needs the beta count; n_occ_b is None, which "
                "means these orbitals are not open shell"
            )
        return somo_occupation_pattern(n_orb, n_occ, n_occ_b)

    def build_orbitals_apc_concentric(
        self,
        *,
        fragment_ao: np.ndarray,
        n_shells: int = 0,
        max_size: int | tuple[int, int],
        fixed: bool = False,
        restrict_to_a: bool = True,
        use_relaxed: bool = False,
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
            use_relaxed: diagonalize the relaxed embedded-HF Fock from
                :meth:`relax_active_hf` instead of the one-shot low-level F_emb.
                Forwarded to the CL stage's :meth:`build_orbitals` call *and* used
                for the ``f_diag``/APC-ranking Fock below, so CL and APC always
                agree on which Fock produced the candidates they rank.

        Returns:
            The APC-selected :class:`EmbeddedOrbitals`.
        """
        from embasi_qiskit_integration.selectors import (
            apc_active_space,
            apc_orbital_entropies,
            apc_pair_coefficients,
            concentric_localization_selector,
        )

        fock = self._fock_relaxed_arr if use_relaxed else self._fock_arr
        cl = concentric_localization_selector(self._s_arr, fragment_ao, fock, n_shells=n_shells)
        orbitals = self.build_orbitals(
            n_frozen_occ=0,
            virtual_localizer=cl,
            restrict_to_a=restrict_to_a,
            use_relaxed=use_relaxed,
        )

        c_full = orbitals.coeff  # (nao, n_A): every subsystem-A orbital, occ + virt
        n_orb = c_full.shape[1]
        n_occ = orbitals.n_occ

        dm_a = 2.0 * c_full[:, :n_occ] @ c_full[:, :n_occ].T
        k_ao = self.ints.get_k(dm_a)
        f_diag = np.einsum("pi,pq,qi->i", c_full, fock, c_full)
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

        # Which columns hold the unpaired electrons must be asked of the DENSITY, not
        # of column position: this method has just run concentric localization, which
        # rotates within blocks and so destroys the energy ordering that
        # `somo_occupation_pattern`'s positional rule assumes.  For OH/sto-3g the
        # projected occupation is [2, 2, 2, 1, 2, 0] -- the SOMO sits at index 3, not
        # atop the occupied block -- so the positional reading would hand APC the wrong
        # orbital to protect.  `column_occupation_from_density` is rotation-invariant.
        if orbitals.is_open_shell:
            occ_pattern = self._occupation_pattern(c_full, n_orb, n_occ, orbitals.n_occ_b)
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
        # Key the refusal on the *reported* spin, not on `unrestricted` alone.  EmbASI's
        # `A_spin` now supplies the beta count (see `_n_occ_b_from_embasi`), so an
        # unrestricted run whose partition reports `2S == 0` is a genuine singlet:
        # `n_alpha == n_beta`, which the restricted downfold represents exactly, and
        # refusing it would reject a well-posed run.  What must still be refused is an
        # unrestricted run whose beta count went *missing* -- there the restricted split
        # is a silent guess at the sector rather than a faithful representation of it.
        if self.unrestricted and orbitals.n_occ_b is None and self._spin_is_unknown():
            raise NotImplementedError(
                "unrestricted=True but these orbitals carry no beta occupied count and "
                "EmbASI exposed no A_spin, so the downfold would silently produce the "
                "restricted n_alpha = n_beta split without any evidence that is the "
                "right sector.  Either use an EmbASI that sets A_spin inside "
                "construct_embedding_potential, or drive open shell through a FCIDUMP: "
                "read the Hamiltonian with hamiltonian.fcidump.read, which carries "
                "(n_alpha, n_beta) exactly."
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
            msg = f"active orbitals leak into subsystem B (|P_B| = {leak:.2e})"
            if self.unrestricted:
                # Measured on an OH radical in water: span(A_alpha) is S-orthogonal to
                # span(B_alpha) to ~9e-15 but to span(B_beta) only to ~3e-4, because
                # SPADE partitions each spin channel independently.  A spin-RESTRICTED
                # downfold takes the alpha MOs (`_as_ao_by_mo` keeps alpha) against a
                # spin-summed P_B, mixing the channels -- so the leak is structural
                # here, not a tolerance to relax, and no reduction of P_B (alpha, beta,
                # sum or mean) removes it: all give the same magnitude.
                msg += (
                    ".  This is an unrestricted run, where the A/B separation holds "
                    "PER SPIN CHANNEL only: EmbASI's SPADE partitions alpha and beta "
                    "independently, so alpha subsystem-A orbitals are not orthogonal "
                    "to the beta environment.  A spin-restricted downfold (one MO set, "
                    "one P_B) therefore cannot be made leak-free on an open shell -- it "
                    "needs per-spin MO sets and a per-spin projector (Stage 2, the "
                    "remaining items).  Until then, drive open shell through a FCIDUMP: "
                    "hamiltonian.fcidump.read carries (n_alpha, n_beta) exactly."
                )
            raise ValueError(msg)

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

    def build_orbitals_spin(
        self,
        *,
        n_frozen_occ: int = 0,
        n_virtual: int | None = None,
        virtual_localizer: VirtualLocalizer | None = None,
        use_relaxed: bool = False,
    ) -> tuple[EmbeddedOrbitals, EmbeddedOrbitals]:
        """``(alpha, beta)`` orbital sets, each from its own channel's ``F_emb``.

        The per-spin counterpart of :meth:`build_orbitals`.  Each channel is
        diagonalized in *its own* span(A) (see :meth:`_eigh_subsystem_a_spin`), which is
        what makes the pair usable: the mixed-channel alternative leaks into subsystem B
        by ~2e-02 and is refused by :meth:`embedded_hamiltonian`'s projector check.

        Both sets are returned with the *restricted* reading of ``n_occ`` (each channel's
        own occupied count, ``n_occ_b=None``), because each set now describes exactly one
        spin: the ``(n_alpha, n_beta)`` sector lives in the pairing, not inside either
        member.  Selectors are deliberately not accepted here -- an independent cut per
        channel could keep different orbital counts, and the downfold needs a common
        active-space dimension.

        **The two channels are reconciled to a common active-orbital count.**  An open
        shell has ``n_occ_alpha != n_occ_beta`` (by ``A_spin``), so applying one
        ``n_virtual`` to both leaves ``n_occ - n_frozen_occ + n_virtual`` differing by
        exactly ``A_spin`` and :meth:`embedded_hamiltonian_spin` then refuses the pair.
        What the solver actually requires is a single ``norb`` -- ``fci.direct_uhf`` takes
        one orbital dimension with an *asymmetric* ``(n_alpha, n_beta)`` sector, which is
        the whole point of an unrestricted solve.  So ``norb`` is equalised and each
        channel's **virtual** count is derived from its own occupied count: the channel
        with fewer electrons simply takes more virtuals.  ``n_virtual`` is therefore a
        ceiling on the *active space*, not a per-channel virtual count -- which is the
        only reading under which it can mean the same thing for both spins.

        Args:
            n_frozen_occ: occupied orbitals frozen into ``e_core``, **per channel**.
                Validated against ``min(n_occ_alpha, n_occ_beta)`` up front, so an
                out-of-range value is reported once against the binding limit rather than
                when the loop happens to reach the narrower channel.  Deliberately *not*
                reconciled: freezing the same *count* in both channels does not freeze
                corresponding orbitals, and a mismatched pair is warned about (see below)
                rather than silently accepted.
            n_virtual: ceiling on the common active-orbital count, expressed as virtuals
                above the *widest* channel's active occupied block.  ``None`` takes the
                largest space both channels can support.
            virtual_localizer: rotates and cuts **each channel's own** virtual block
                (its own sigma^2 ordering, since each was diagonalized in its own
                span(A)), before the two channels are reconciled to a common ``norb``.
                The per-channel cut therefore bounds what the reconciliation may promise.
            use_relaxed: diagonalize the relaxed per-spin Fock.

        Warns:
            UserWarning: a frozen pair's overlap ``|<a_i|S|b_i>|`` falls below
                ``_FROZEN_OVERLAP_TOL``.  The two channels' frozen orbitals correspond
                only while they overlap: measured on the stretched C-N geometry the first
                pairs run 1.000 / 1.000 / 0.982 / 0.951 and then collapse to 1e-04, so
                freezing five in each channel freezes *physically different* orbitals in
                the fifth slot.  The count matching cannot detect that; the overlap can.
        """
        if self._fock_spin is None:
            raise ValueError(
                "no per-spin embedded Fock available; build_orbitals_spin needs an "
                "unrestricted run_low_level() against an EmbASI that returns a spin "
                "axis (adapter unrestricted=%r)" % (self.unrestricted,)
            )
        if use_relaxed and self._fock_relaxed_spin is None:
            raise ValueError(
                "use_relaxed=True needs a relaxed per-spin Fock; call relax_active_hf() "
                "on an unrestricted adapter first"
            )

        # Pass 1: each channel's own eigenbasis and counts.  Nothing is cut yet -- the cut
        # needs both channels' numbers, which is why this cannot stay a single loop.
        eps_c: list[tuple[np.ndarray, np.ndarray]] = []
        n_occs: list[int] = []
        n_virt_totals: list[int] = []
        for ispin in (0, 1):
            eps, c = self._eigh_subsystem_a_spin(ispin, use_relaxed=use_relaxed)
            n_occ = self._mo_spin(self.p.mo_coeffs_A_LL, ispin).shape[1]
            n_virt_total = c.shape[1] - n_occ
            if n_virt_total < 0:
                raise ValueError(
                    f"spin {ispin}: subsystem-A space has {c.shape[1]} orbitals but "
                    f"{n_occ} are occupied; the partition and the MOs disagree"
                )
            if virtual_localizer is not None:
                # Rotate THIS channel's virtual block and cut on its own sigma^2 gap,
                # exactly as `build_orbitals` does for the restricted path.  It has to
                # happen here, in pass 1: the cut changes how many virtuals the channel
                # can offer, which is an input to the reconciliation below -- applying it
                # afterwards would let `n_act` promise orbitals the localiser had removed.
                #
                # Each channel gets its own sigma^2 ordering because each was diagonalized
                # in its own span(A); reusing alpha's rotation for beta is the same
                # per-channel-basis error the rest of this method exists to avoid.
                from embasi_qiskit_integration.selectors import _gap_cut

                c, sigma2 = virtual_localizer(c, n_occ)
                eps = eps.copy()
                eps[n_occ:] = np.nan  # rotated-virtual eigenvalues are meaningless
                # Read the cut knobs immediately after THIS call: `concentric-cl` pins
                # `max_virtual`/`min_virtual` as mutable attributes on the closure, so a
                # single localiser object reused across channels is last-call-wins.
                kept = _gap_cut(
                    sigma2,
                    gap_tol=getattr(virtual_localizer, "gap_tol", 1.0e-3),
                    max_virtual=getattr(virtual_localizer, "max_virtual", None),
                    min_virtual=getattr(virtual_localizer, "min_virtual", 0),
                )
                n_virt_total = min(int(kept), n_virt_total)
            eps_c.append((eps, c))
            n_occs.append(n_occ)
            n_virt_totals.append(n_virt_total)

        # Validate `n_frozen_occ` against the BINDING channel, naming it.  Reporting this
        # up front is the difference between "n_frozen_occ=5 outside [0, 5) for spin 1"
        # (which reads as beta being at fault) and a message that says the shared knob is
        # capped by the channel with fewer electrons.
        n_occ_min = min(n_occs)
        if not 0 <= n_frozen_occ < n_occ_min:
            raise ValueError(
                f"n_frozen_occ={n_frozen_occ} outside [0, {n_occ_min}): subsystem A has "
                f"n_occ={n_occs[0]} alpha / {n_occs[1]} beta, and the frozen count applies "
                f"to both channels, so the smaller one binds. Reduce n_frozen_occ to at "
                f"most {n_occ_min - 1}."
            )

        # Reconcile to a common ACTIVE-ORBITAL count.  Each channel can support at most
        # `n_occ_s - n_frozen_occ + n_virt_total_s` active orbitals, so the smaller of the
        # two is the largest `norb` both can realise.  `n_virtual` then caps it, counted
        # above the WIDEST channel's active occupied block -- as a per-channel virtual
        # count it cannot mean the same thing for two channels with different occupancies,
        # which is exactly the bug this replaces.
        n_act_occ = [n - n_frozen_occ for n in n_occs]
        supported = [n_act_occ[s] + n_virt_totals[s] for s in (0, 1)]
        n_act = min(supported)
        if n_virtual is not None:
            n_act = min(n_act, max(n_act_occ) + n_virtual)
        if n_act < max(n_act_occ):
            # `norb` cannot cover both channels' active occupied blocks.  Two distinct
            # causes, and the message has to say which -- they need opposite fixes.
            #
            # `_eigh_subsystem_a_spin` returns span(A_s) = the S-orthogonal complement of
            # span(B_s), whose width is `nao - n_occ_B_s` and so differs per channel.  On
            # the real doublets here both come out equal (19 or 20 wide), but nothing
            # guarantees it: when the narrower span cannot hold the wider channel's
            # occupied block, no `n_virtual` helps and the partition itself is the limit.
            binding_span = min(supported) < max(n_act_occ)
            if binding_span:
                raise ValueError(
                    f"the two spin channels cannot share an active-orbital count: spin 0 "
                    f"supports {supported[0]} active orbitals and spin 1 supports "
                    f"{supported[1]}, but the wider channel's active occupied block alone "
                    f"needs {max(n_act_occ)} (n_occ={n_occs[0]}/{n_occs[1]} minus "
                    f"n_frozen_occ={n_frozen_occ}). The per-channel span(A) widths differ "
                    f"({n_occs[0] + n_virt_totals[0]}/{n_occs[1] + n_virt_totals[1]}), so "
                    f"this is a property of EmbASI's partition, not of n_virtual -- "
                    f"raising it cannot fix this. Freeze more occupied orbitals "
                    f"(n_frozen_occ > {n_frozen_occ}) to shrink the active occupied block."
                )
            raise ValueError(
                f"n_virtual={n_virtual} gives a {n_act}-orbital active space, smaller than "
                f"the active occupied block of one channel ({max(n_act_occ)}: n_occ="
                f"{n_occs[0]}/{n_occs[1]} minus n_frozen_occ={n_frozen_occ}); that would "
                f"drop occupied orbitals from both the core and the active space, losing "
                f"electrons. n_virtual counts above the widest channel's active occupied "
                f"block, so it must be >= 0 -- which it is here, meaning the cap is simply "
                f"below what the sector needs."
            )

        out = []
        for ispin in (0, 1):
            eps, c = eps_c[ispin]
            n_occ = n_occs[ispin]
            # Each channel's virtual count follows from the COMMON active size and its own
            # occupied count: fewer electrons -> more virtuals, same `norb`.
            n_virt = n_act - n_act_occ[ispin]
            active = np.arange(n_frozen_occ, n_occ + n_virt)
            inactive = np.arange(n_frozen_occ, dtype=int)
            out.append(
                EmbeddedOrbitals(coeff=c, energy=eps, n_occ=n_occ, inactive=inactive, active=active)
            )

        self._warn_on_mismatched_frozen_core(out[0], out[1])
        return out[0], out[1]

    def _warn_on_mismatched_frozen_core(
        self, alpha: EmbeddedOrbitals, beta: EmbeddedOrbitals
    ) -> None:
        """Warn when the two channels' frozen orbitals do not correspond.

        ``n_frozen_occ`` freezes the lowest *k* orbitals of each channel independently, so
        the two cores match only while ``|<a_i|S|b_i>|`` stays near 1.  When it does not,
        the pair is still dimensionally valid and every count-based check passes -- the
        electron count, the spin sector, the projector leak -- while ``e_core`` charges
        two physically different orbitals to one frozen slot.  Only the overlap sees it,
        which is why it is checked rather than inferred.

        A warning, not an error: the frozen core is folded *unrestricted*
        (:meth:`AOIntegrals.veff_uhf`), so a mismatched pair is handled correctly as two
        distinct densities.  What degrades is the *interpretation* -- "frozen core" implies
        a shared set -- and how well a fixed count truncates the two channels alike.
        """
        if not alpha.inactive.size:
            return
        s = self._s_arr
        c_a, c_b = alpha.c_inactive, beta.c_inactive
        overlaps = np.abs(np.diag(c_a.T @ s @ c_b))
        bad = [(i, float(o)) for i, o in enumerate(overlaps) if o < _FROZEN_OVERLAP_TOL]
        if bad:
            detail = ", ".join(f"i={i}: {o:.3g}" for i, o in bad)
            warnings.warn(
                f"the two spin channels' frozen orbitals do not correspond ({detail}; "
                f"|<a_i|S|b_i>| < {_FROZEN_OVERLAP_TOL}). n_frozen_occ freezes the lowest "
                f"{alpha.inactive.size} orbitals of EACH channel, and SPADE orders them "
                f"per spin, so a matching count does not mean matching orbitals. The core "
                f"is folded unrestricted so the energy is still assembled correctly, but "
                f"'frozen core' no longer denotes one shared set -- reduce n_frozen_occ to "
                f"stay inside the corresponding block.",
                stacklevel=3,
            )

    def embedded_hamiltonian_spin(
        self, orbitals: tuple[EmbeddedOrbitals, EmbeddedOrbitals]
    ) -> EmbeddedHamiltonian:
        """Downfold a per-spin orbital pair to an ``(h1a, h1b)`` Hamiltonian.

        Produces the spin-dependent one-body pair :class:`EmbeddedHamiltonian` now
        accepts, plus the spin-averaged ``h1`` that the consumers which cannot take a
        pair (SQD, FCIDUMP) fall back on.

        The two-body part is spin-resolved too: ``h2_spin`` carries ``(aa|aa)``,
        ``(aa|bb)`` and ``(bb|bb)`` over the two different orbital sets (the mixed block
        via :meth:`PySCFIntegrals.eri_mo_mixed`).  ``h2`` is still produced as the
        alpha-only tensor, because every consumer that cannot take the triple reads it.
        A backend without ``eri_mo_mixed`` yields ``h2_spin=None``, recorded in ``meta``
        as ``h2_spin_free`` rather than silently fabricated.

        With ``n_frozen_occ > 0`` the frozen core is the *sum* of the two channels'
        inactive densities, not twice either one: each channel freezes its own orbitals,
        so both the folded ``veff`` and ``e_core`` are built from
        ``c_in_a c_in_a^T + c_in_b c_in_b^T``.

        The frozen-core mean field is **unrestricted**: each channel sees
        ``J[d_a + d_b] - K[d_sigma]`` (:meth:`AOIntegrals.veff_uhf`).  Coulomb is a
        functional of the total core density, but exchange couples like spins only, so
        the restricted ``J - K/2`` is correct only when both channels freeze the same
        orbitals -- which SPADE's per-spin partition does not guarantee.  ``e_core``'s
        one-body part is likewise per channel (each core density against its own
        ``h_emb_s``, since the two channels see different embedded Focks), and its
        two-body part carries the matching UHF exchange self-interaction rather than
        ``0.5 tr[d veff]``, an identity that holds only for the restricted form.  A
        backend without ``veff_uhf`` falls back to the restricted fold and records
        ``veff_core_spin_free`` in ``meta``.  All of this is inert at
        ``n_frozen_occ=0``, where both cores are empty.
        """
        alpha, beta = orbitals
        if alpha.n_active_orbitals != beta.n_active_orbitals:
            raise ValueError(
                f"alpha and beta active spaces differ in size "
                f"({alpha.n_active_orbitals} vs {beta.n_active_orbitals}); the downfold "
                "needs a common orbital dimension"
            )
        if self._p_b_spin is None:
            raise ValueError("no per-spin P_B available; call run_low_level() unrestricted")

        # Frozen-core density, ONCE for the pair.  The two channels freeze *different*
        # orbitals (SPADE partitions each spin separately), so the core is
        # `c_in_a c_in_a^T + c_in_b c_in_b^T` -- one electron per channel.  The
        # restricted `2 * c_in c_in^T` is not a valid stand-in: it has the right trace,
        # so no electron-count check fires, but it is the wrong matrix wherever the two
        # cores differ.  That error is invisible for a deep 1s core (<a|S|b> ~ 0.999998,
        # ~0.4 kcal/mol on OH) and large once a frozen orbital is valence-like
        # (<a|S|b> ~ 0.96, ~40 kcal/mol on triplet CH2).  It also cancels identically at
        # `n_frozen_occ=0`, where both forms are zero.
        # Kept as separate halves as well as the sum: the one-body part of `e_core` below
        # contracts each channel's core against its own operator, which the summed
        # density alone cannot express.
        dm_core_a = alpha.c_inactive @ alpha.c_inactive.T
        dm_core_b = beta.c_inactive @ beta.c_inactive.T
        dm_in = dm_core_a + dm_core_b
        # The frozen-core mean field is UNRESTRICTED: `J` is a functional of the total
        # core density, but exchange couples like spins only, so each channel sees
        # `J[d_a + d_b] - K[d_sigma]` and the two differ.  The restricted `J - K/2`
        # (`veff_hf`) is correct only when both channels freeze the SAME orbitals; here
        # they do not, because SPADE partitions the spins separately.  Measured against an
        # ERI-only UHF ground truth with valence-like cores (overlap 0.34): the averaged
        # form is 1.09 Ha off per channel and +0.78 Ha (+492 kcal/mol) on the CAS total.
        # It is invisible for a deep 1s core (overlap 0.9999998, 0.02 kcal/mol), which is
        # why the pinned OH numbers barely move -- and why this had to be checked against
        # a rebuild that does not itself call `veff_hf`.
        #
        # `e_two_body` comes back from the same call rather than being recomputed as
        # `0.5 tr[d veff]`: that identity holds for the restricted `J - K/2` only, since
        # the exchange self-interaction does not factor out once the two channels have
        # different K.
        if hasattr(self.ints, "veff_uhf"):
            veff_in_a, veff_in_b, e_core_two_body = self.ints.veff_uhf(dm_core_a, dm_core_b)
            veff_spin_free = False
        else:
            # A backend without the unrestricted fold (the stub integral classes in the
            # tests): keep the restricted form rather than fabricating one, and record it
            # in `meta` so a consumer can tell the fold was spin-averaged.  Identical to
            # the previous behaviour, and inert at `n_frozen_occ=0` where both cores are
            # empty and every form below is zero.
            veff_in_a = veff_in_b = self.ints.veff_hf(dm_in)
            e_core_two_body = 0.5 * float(np.einsum("ij,ji->", dm_in, veff_in_a))
            veff_spin_free = True
        veff_in_pair = (veff_in_a, veff_in_b)

        h1_pair, leaks, h_emb_pair = [], [], []
        for ispin, orb in ((0, alpha), (1, beta)):
            c_act = orb.c_active
            # Each channel's own projector must be invisible in its own active space.
            leaks.append(float(np.abs(c_act.T @ self._p_b_spin[ispin] @ c_act).max()))
            fock = self._fock_spin[ispin]  # type: ignore[index]
            # h_emb per channel: strip the low-level mean field, exactly as `h_emb` does
            # for the spin-summed Fock (see that property for why it is veff_ll).
            h_emb_s = fock - self.ints.veff_ll(self._dm_a_for_veff)
            # Kept for `e_core` below, which must charge each channel's frozen core to
            # its OWN one-body operator -- the same one its h1 is built from.
            h_emb_pair.append(h_emb_s)
            # ...and each channel's own frozen-core potential (see `veff_in_pair` above).
            h1_s = c_act.T @ (h_emb_s + veff_in_pair[ispin]) @ c_act
            asym = float(np.abs(h1_s - h1_s.T).max())
            if asym > 1e-6:
                raise ValueError(
                    f"spin-{ispin} h1 is non-Hermitian beyond round-off "
                    f"(max |h1 - h1^T| = {asym:.2e})"
                )
            h1_pair.append(0.5 * (h1_s + h1_s.T))

        worst_leak = max(leaks)
        if worst_leak > 1e-6:
            raise ValueError(
                f"active orbitals leak into subsystem B per spin (|P_B| = "
                f"{worst_leak:.2e}); each channel was diagonalized in its own span(A), "
                "so this is not the cross-spin mixing the restricted path hits -- check "
                "the partition"
            )

        h1a, h1b = h1_pair
        # Genuine (aa|aa), (aa|bb), (bb|bb) over the two DIFFERENT orbital sets -- the
        # two-body counterpart of the h1 pair.  `h2` (alpha-only) is still produced
        # because every consumer that cannot take the triple reads it.
        c_act_a, c_act_b = alpha.c_active, beta.c_active
        h2 = self.ints.eri_mo(c_act_a)
        h2_spin: tuple[np.ndarray, np.ndarray, np.ndarray] | None
        if hasattr(self.ints, "eri_mo_mixed"):
            h2_spin = (
                h2,  # (aa|aa) is exactly the alpha-only tensor
                self.ints.eri_mo_mixed(c_act_a, c_act_b),
                self.ints.eri_mo(c_act_b),
            )
        else:
            # A backend without the mixed transform (e.g. a stub): fall back to the
            # spin-free tensor rather than fabricating a triple, and record it.
            h2_spin = None
        # Frozen-core energy, per channel.  Each channel's core density is charged to
        # ITS OWN embedded one-body operator (the very `h_emb_s` its h1 is built from),
        # not to alpha's for both: the two channels see different Focks, so contracting
        # beta's core against `h_emb_a` charges it at the wrong potential.  The error is
        # exactly `tr[d_b (h_emb_a - h_emb_b)]` -- 0.0162 Ha (10.2 kcal/mol) on the OH
        # radical at n_frozen_occ=1, with the right electron count and spin sector, so
        # nothing else flags it.  The two-body term stays on the TOTAL core density
        # (`veff_in` is a functional of it) and is counted once, hence the single 0.5.
        # Vanishes identically at n_frozen_occ=0, where both cores are empty.
        e_core = float(
            self.ints.energy_nuc()
            + np.einsum("ij,ji->", dm_core_a, h_emb_pair[0])
            + np.einsum("ij,ji->", dm_core_b, h_emb_pair[1])
            + e_core_two_body
        )
        n_alpha = alpha.n_occ - alpha.inactive.size
        n_beta = beta.n_occ - beta.inactive.size
        return EmbeddedHamiltonian(
            h1=0.5 * (h1a + h1b),
            h2=h2,
            e_core=float(e_core),
            nelec=(n_alpha, n_beta),
            h1a=h1a,
            h1b=h1b,
            h2_spin=h2_spin,
            meta={
                "localisation": "spade",
                "projection": "level-shift",
                "mu": self.mu,
                "p_b_leak_per_spin": leaks,
                "spin_dependent": True,
                # Honest record of what is and is not spin-resolved here.
                "h2_spin_free": h2_spin is None,
                # True when the backend had no `veff_uhf` and the frozen core fell back
                # to the restricted `J - K/2` fold.  Always inert at `n_frozen_occ=0`.
                "veff_core_spin_free": veff_spin_free,
            },
        )

    # ---------------- energy assembly ---------------- #
    def projection_energy(
        self,
        result: SolverResult,
        orbitals: EmbeddedOrbitals,
        orbitals_b: EmbeddedOrbitals | None = None,
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

        **Per-spin downfold.**  ``orbitals_b`` is the beta channel's own orbital set
        (:meth:`build_orbitals_spin`'s second return value).  Pass it whenever the
        Hamiltonian was built by :meth:`embedded_hamiltonian_spin`: the two channels live
        in *different* spans, so the spin-summed ``rdm1`` cannot be lifted through one
        set.  Getting ``dm_hl`` wrong moves the reported total -- measured
        **0.0779 Ha (48.9 kcal/mol)** on the OH-radical doublet, with the electron count,
        the spin sector and the footing shift all still exact, which is why no existing
        check caught it.  Omit it on a restricted run (bit-identical to before).

        **Passing it also selects the per-channel contraction.** With ``orbitals_b`` every
        density-linear term is contracted channel-against-its-own-operator and the
        spin-summed values derived from the pair; without it the restricted spin-summed
        form is used.  So omitting ``orbitals_b`` on a per-spin downfold is not merely a
        lift error -- it also reintroduces the cross-channel projector term, which ``mu``
        scales to ``+2.4e4`` on ``data/22.inp``.  See the module docstring's
        *known limits* for the measurements.
        """
        v_emb, p_b = self.v_emb, self.p_b
        # Bound to locals so the None-narrowing below reaches `rdm1_ao_spin` (reading the
        # attributes back off `result` widens them to `ndarray | None` again).  Together
        # these are `result.is_spin_resolved`, which the contract validates as
        # all-or-nothing.
        rdm1a_in, rdm1b_in = result.rdm1a, result.rdm1b
        dm_a_hl: np.ndarray | None = None
        dm_b_hl: np.ndarray | None = None
        if rdm1a_in is not None and rdm1b_in is not None:
            # Lift each channel through its OWN active space.  Equivalent to
            # `rdm1_ao(result.rdm1, orbitals)` when both sets coincide (the restricted
            # case, where `orbitals_b is None`), but on a per-spin downfold the beta half
            # belongs in beta's span, not alpha's.
            dm_a_hl, dm_b_hl = self.rdm1_ao_spin(rdm1a_in, rdm1b_in, orbitals, orbitals_b)
        if orbitals_b is not None and dm_a_hl is not None and dm_b_hl is not None:
            dm_hl = dm_a_hl + dm_b_hl
        else:
            dm_hl = self.rdm1_ao(result.rdm1, orbitals)

        # Per-spin reference pair, needed by both the totals (below) and the split.
        # `_dm_a_spin_init` is the round-0 pair captured once in `run_low_level`; fall back
        # to halving the spin-summed reference only when there is none (a restricted
        # adapter, or a snapshot predating it), where the two channels are equal anyway
        # and halving is exact.  Read through `getattr` because the stub adapters in the
        # tests are built via `object.__new__` and never run `__init__`.
        spin_init = getattr(self, "_dm_a_spin_init", None)
        if spin_init is not None:
            init_a, init_b = spin_init
        else:
            init_a = init_b = 0.5 * self._dm_a_arr_init

        # On a per-spin downfold, contract EACH CHANNEL AGAINST ITS OWN OPERATORS and sum
        # -- do not contract the spin-summed density against the spin-summed ones.  The
        # rule: subtract what the downfold actually folded in, channel by channel.
        #
        # `embedded_hamiltonian_spin` builds `h1_s` from `h_emb_s = F_emb_s - veff_ll`,
        # i.e. each channel's own `v_emb_s` and `P_B_s` (verified: `hcore + v_emb_spin(s)
        # + P_B_s` reproduces that `h_emb_s` to 0.0).  So `result.energy` already contains
        # `sum_s tr[d_s v_emb_s]`, NOT `tr[dm v_emb]`, and it never contained a
        # cross-channel projector term at all.
        #
        # Using the spin-summed operators instead is wrong twice over, because SPADE
        # partitions each spin separately, so `span(A_alpha)` is S-orthogonal to
        # `span(B_alpha)` but *not* to `span(B_beta)`:
        #
        # * The projector picks up the cross terms `tr[d_alpha P_beta]`, which the level
        #   shift multiplies by `mu`.  Measured on data/22.inp (stretched C-N, 2.2 A):
        #   own terms -1.2e-11 / +1.2e-10, cross terms +2.4e+04 / +5.4e+02, so the
        #   spin-summed `leak` came out at +2.41e+04 -- a quantity documented as a
        #   numerical zero -- and dragged the reported total to -24288 Ha on a ~-207 Ha
        #   system.
        # * The `v_emb` term removes something the solver never added, worth +54.0 Ha
        #   there.
        #
        # Both together: `e_high_A` lands at -9.0917 against a closed-shell control on the
        # SAME geometry at -9.1139 (totals -207.2066 vs -207.2108, a triplet ~0.1 eV above
        # the singlet).  That two-path agreement is the evidence for this form.
        #
        # Two independent conditions, deliberately NOT collapsed into one:
        #
        # * `have_spin_operators` -- is there a per-spin `v_emb`/`P_B` at all?  Only then
        #   can anything be contracted per channel.  It is false on every restricted run,
        #   which is why those stay bit-identical.
        # * `dm_a_hl`/`dm_b_hl` -- did the solver return a spin-resolved RDM?  A restricted
        #   adapter can still get one (an unrestricted solver on a closed-shell downfold),
        #   and that case must keep reporting the split it always did, against the
        #   spin-summed operators, which are exact there because the two channels share one
        #   span.
        have_spin_operators = self._p_b_spin is not None and self._fock_spin is not None
        leak_spin: tuple[float, float] | None = None
        correction_spin: tuple[float, float] | None = None
        if dm_a_hl is not None and dm_b_hl is not None:
            if have_spin_operators:
                # Genuine per-spin downfold: each channel against its OWN operators.
                v_emb_a, v_emb_b = self.v_emb_spin(0), self.v_emb_spin(1)
                p_b_pair = self._p_b_spin
                assert p_b_pair is not None  # narrowed by `have_spin_operators`
                p_b_a, p_b_b = p_b_pair
            else:
                # Restricted operators, spin-resolved density: the channels share a span,
                # so the spin-summed operators are the right ones and the split is exact.
                v_emb_a = v_emb_b = v_emb
                p_b_a = p_b_b = p_b
            leak_a = float(np.einsum("ij,ji->", dm_a_hl, p_b_a))
            leak_b = float(np.einsum("ij,ji->", dm_b_hl, p_b_b))
            corr_a = float(np.einsum("ij,ji->", dm_a_hl - init_a, v_emb_a))
            corr_b = float(np.einsum("ij,ji->", dm_b_hl - init_b, v_emb_b))
            leak_spin = (leak_a, leak_b)
            correction_spin = (corr_a, corr_b)
            # The spin-summed values are DERIVED from the channels, not the reverse.
            # `sum(correction_spin) == correction` therefore still holds exactly -- by
            # construction now, rather than by sharing one operator as before.
            leak = leak_a + leak_b
            correction = corr_a + corr_b
            v_emb_term = float(
                np.einsum("ij,ji->", dm_a_hl, v_emb_a) + np.einsum("ij,ji->", dm_b_hl, v_emb_b)
            )
        else:
            leak = float(np.einsum("ij,ji->", dm_hl, p_b))
            correction = float(np.einsum("ij,ji->", dm_hl - self._dm_a_arr_init, v_emb))
            v_emb_term = float(np.einsum("ij,ji->", dm_hl, v_emb))

        e_high_a = float(result.energy) - v_emb_term - leak

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
            correction_spin=correction_spin,
            projector_leak_spin=leak_spin,
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
        orbitals_b: EmbeddedOrbitals | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Back-transform a spin-resolved active 1-RDM pair to AO alpha/beta densities.

        The unrestricted counterpart of :meth:`rdm1_ao`, for the density feedback an
        open-shell outer loop needs: a spin-summed total cannot represent
        ``gamma_alpha != gamma_beta``, so the loop must carry the pair.

        The frozen core contributes one electron per spin channel (``c_in @ c_in.T``,
        not ``2.0 *``), so that the two channels sum back to the same total
        :meth:`rdm1_ao` produces.

        ``orbitals_b`` is the **beta** channel's own orbital set, as returned second by
        :meth:`build_orbitals_spin`.  It is mandatory on a per-spin downfold and must be
        omitted on a restricted one:

        * On the per-spin path the two channels are diagonalized in *different* spans
          (each in its own span(A) -- see :meth:`_eigh_subsystem_a_spin`), so
          ``rdm1_active_b`` is expressed in **beta's** active orbitals.  Lifting it with
          alpha's ``c_active`` reads a beta-basis matrix as though it were alpha-basis:
          the result still has the right trace and the right electron count, so no
          sector or population check fires, but it is the wrong matrix.  Measured on the
          OH-radical doublet: ``max|dm_beta|`` wrong by **0.998** (a whole electron's
          worth of AO density) while ``N_beta`` stayed 4.0 to 1e-15 and the spin
          polarisation stayed exactly 1.
        * On the restricted path there is only one set, and passing ``None`` reuses it
          for both channels -- bit-identical to the previous behaviour.

        Returns:
            ``(dm_alpha, dm_beta)`` in the AO basis.
        """
        orb_b = orbitals if orbitals_b is None else orbitals_b
        # One electron per channel, each from its OWN span: `core` is not shared, because
        # the two channels freeze different orbitals (SPADE partitions the spins
        # independently).  At `n_frozen_occ=0` both blocks are empty and this is zero.
        c_in_a, c_act_a = orbitals.c_inactive, orbitals.c_active
        c_in_b, c_act_b = orb_b.c_inactive, orb_b.c_active
        dm_a = c_in_a @ c_in_a.T + c_act_a @ np.asarray(rdm1_active_a) @ c_act_a.T
        dm_b = c_in_b @ c_in_b.T + c_act_b @ np.asarray(rdm1_active_b) @ c_act_b.T
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

        The fed-back subsystem-A density (γ̃^A from the correlated solve) and the
        frozen environment γ^B go in as the *separate* arguments
        :meth:`run_low_level` takes, each wrapped into EmbASI's ``SpinKpointArray``
        there.  They are deliberately not pre-summed: ``run_low_level`` hands
        ``dma_in`` and ``dmb_in`` to ``construct_embedding_potential`` as the two
        subsystems it re-partitions against, so collapsing them into one total
        would feed the A layer the environment density as well.
        """
        dm_hl = self.rdm1_ao(rdm1_active, orbitals)
        self.run_low_level(dma_in=dm_hl, dmb_in=self._dm_b_arr)

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
