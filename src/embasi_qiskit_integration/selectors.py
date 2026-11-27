# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Active-space selectors for :meth:`ProjectionEmbeddingAdapter.build_orbitals`.

``build_orbitals`` accepts a ``Selector`` hook -- ``(coeff, energy, n_occ) ->
active column indices`` -- so the choice of which virtuals to correlate can be
made data-driven instead of guessed with ``--n_virtual``.

:func:`mulliken_selector` implements a single-shell concentric-localisation cut in
the lineage of SPADE (Claudino & Mayhall): the occupied subsystem-A space is fixed
by the embedding, and the virtuals are *ranked* by how strongly each couples to
the active fragment, then truncated at the largest gap in that ranking.  Because
``build_orbitals`` hands the selector the *canonical* orbitals of ``F_emb`` (the
occupied-virtual Fock coupling is therefore diagonal and useless as a ranking),
the coupling we score is the fragment population of each virtual -- a Mulliken
weight ``w_v = sum_{mu in fragment} (S C)_{mu v} C_{mu v}`` -- which is well
defined in the canonical basis.

The selector never touches the occupied block: subsystem A's occupied orbitals
are dictated by the level-shift embedding, and freezing occupied orbitals is a
separate knob (``n_frozen_occ``).  It only decides *how many virtuals* to keep.

:func:`concentric_localization_selector` is the *iterative* concentric algorithm
of Claudino et al. (*J. Chem. Theory Comput.* **2019**, 15, 6085), of which the
Mulliken/SPADE cuts above are the single-shell case.  Shell 0 projects the
virtuals onto the active-fragment AO basis and SVD-splits the fragment-*spanned*
virtuals from the kernel remainder; each subsequent shell SVDs the current span
against the current kernel *through the embedded Fock matrix* and appends the
newly-coupled virtuals.  The kept count therefore *grows* with the shell count and
is set by SVD rank, not a gap cut -- the shell structure itself is the truncation.
Like SPADE it rotates the virtual block, so it is a ``VirtualLocalizer``; unlike
SPADE it needs the Fock matrix as the shell-expansion metric.  The paper reports
the single-shell cut at ~1 kcal/mol; the extra shells are the accuracy knob.

:func:`spade_virtual_selector` is the localising variant.  A Mulliken weight of a
*canonical* virtual is only a good shell coordinate while the canonical virtuals
are themselves fragment-local; for a large environment they delocalise, the
fragment populations compress toward a common value, the gap flattens, and the cut
can admit non-local virtuals or refuse to truncate at all.  The SPADE construction
(Claudino & Mayhall, *J. Chem. Theory Comput.* **2019**, 15, 1053,
doi:10.1021/acs.jctc.9b00682) removes that dependence: it *rotates* the virtual
block so each rotated virtual carries a definite fragment weight (the squared
singular value ``σ²`` of the Löwdin-orthogonalised, fragment-restricted virtual
coefficients), then cuts on the ``σ²`` gap ``Δσ_i² = σ_i² − σ_{i+1}²``.  Because
the rotation is by orthogonal congruence, ``σ²`` is *basis-invariant* -- it does
not care whether the incoming canonical virtuals are localised -- so it is the
sustainable long-term cut.  A rotation is not expressible as a column-index
:class:`Selector`, so the localiser is a separate ``VirtualLocalizer`` hook on
:meth:`ProjectionEmbeddingAdapter.build_orbitals`; the gap cut itself is shared
(:func:`_gap_cut`) so both paths truncate identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embasi_qiskit_integration.backends.base import (
    CAP_MINAO_PROJECTOR,
    MissingCapabilityError,
    as_ao_basis,
    atom_ao_indices,
    require_capability,
)
from embasi_qiskit_integration.projection_embedding_adapter import (
    Selector,
    VirtualLocalizer,
)

__all__ = [
    "mulliken_selector",
    "per_fragment_mulliken_selector",
    "concentric_localization_selector",
    "spade_virtual_selector",
    "fragment_ao_indices",
    "fragment_ao_projector",
    "avas_ao_projector",
    "apc_pair_coefficients",
    "apc_orbital_entropies",
    "apc_active_space",
    "MP2Correlation",
    "mp2_orbital_correlation",
    "select_correlated_orbitals",
]


def _gap_cut(
    weights: np.ndarray,
    *,
    gap_tol: float,
    max_virtual: int | None,
    min_virtual: int,
) -> int:
    """Number of virtuals to keep given per-virtual fragment ``weights``.

    ``weights`` must already be sorted descending.  The cut is the *largest* gap
    ``weights[i] - weights[i+1]`` that clears ``gap_tol`` (keeping ``0..i``); if no
    gap qualifies every virtual is kept -- the cut refuses to truncate on noise.
    ``min_virtual``/``max_virtual`` then floor/cap the count.  Shared by
    :func:`mulliken_selector` (Mulliken weights) and :func:`spade_virtual_selector`
    (σ² weights) so both truncate identically.
    """
    n_virt = int(weights.shape[0])
    if n_virt == 0:
        return 0
    gaps = weights[:-1] - weights[1:]
    keep = n_virt  # default: keep everything
    significant = np.nonzero(gaps >= gap_tol)[0]
    if significant.size:
        # +1: a gap after index i keeps virtuals 0..i inclusive.
        keep = int(significant[np.argmax(gaps[significant])]) + 1
    keep = max(keep, min_virtual)
    if max_virtual is not None:
        keep = min(keep, max_virtual)
    return min(keep, n_virt)


def mulliken_selector(
    overlap: np.ndarray,
    fragment_ao: np.ndarray,
    *,
    gap_tol: float = 1.0e-3,
    max_virtual: int | None = None,
    min_virtual: int = 0,
) -> Selector:
    """Build a single-shell virtual-space selector that cuts on the fragment-coupling gap.

    The single-shell (shell-0) Mulliken cut; :func:`concentric_localization_selector`
    is the iterative generalisation.

    Args:
        overlap: AO overlap matrix ``S`` (``adapter._s``), shape (nao, nao).
        fragment_ao: AO indices belonging to the active fragment atoms.  These
            define the concentric reference the virtuals are ranked against.
        gap_tol: minimum ``w_i - w_{i+1}`` (in fragment-population units) that
            counts as a shell boundary.  The cut is taken at the *largest* gap
            (among those exceeding this) from the top, so the whole tightly-
            coupled shell is kept and the long tail of fragment-orthogonal
            virtuals is dropped.  Cutting at the first qualifying gap instead
            would slice off after the leading virtual whenever the top shell has
            any internal jitter above ``gap_tol``.  If no gap exceeds ``gap_tol``
            every virtual is kept (the selector refuses to truncate on noise).
        max_virtual: hard cap on the number of virtuals kept, applied after the
            gap cut (a solver-budget ceiling).  ``None`` for no cap.
        min_virtual: keep at least this many virtuals even if the first gap comes
            sooner (guards against an over-eager cut on a small system).

    Returns:
        A ``Selector`` closing over ``S`` and the fragment.  When called with the
        level-shift-kept ``(coeff, energy, n_occ)`` it returns
        ``[0, n_occ) `` (all occupied A) followed by the kept virtual indices.
    """
    s = np.asarray(overlap)
    frag = np.asarray(fragment_ao, dtype=int)

    def _select(coeff: np.ndarray, energy: np.ndarray, n_occ: int) -> np.ndarray:
        c_virt = coeff[:, n_occ:]
        n_virt = c_virt.shape[1]
        occupied = np.arange(n_occ)
        if n_virt == 0:
            return occupied

        # Fragment population of each virtual: w_v = sum_{mu in frag} (S C)_{mu v} C_{mu v}.
        sc = s @ c_virt
        w = np.einsum("mv,mv->v", sc[frag, :], c_virt[frag, :])

        order = np.argsort(-w)  # strongest fragment coupling first
        # Largest gap from the top that clears the tolerance sets the shell edge
        # (the *first* qualifying gap would slice off after the top virtual whenever
        # the leading shell has internal jitter above gap_tol; the largest gap is the
        # true shell boundary).  Shared with the SPADE cut via ``_gap_cut``.
        keep = _gap_cut(w[order], gap_tol=gap_tol, max_virtual=max_virtual, min_virtual=min_virtual)

        kept_virtuals = n_occ + np.sort(order[:keep])
        return np.concatenate([occupied, kept_virtuals]).astype(int)

    return _select


def per_fragment_mulliken_selector(
    overlap: np.ndarray,
    fragment_ao_groups: list[np.ndarray],
    *,
    gap_tol: float = 1.0e-3,
    max_virtual_per_fragment: int | None = None,
    min_virtual_per_fragment: int = 0,
) -> Selector:
    """Single-shell Mulliken cut run *independently per fragment*, then unioned.

    For an interaction of ``k`` fragments (e.g. both OH groups of a dimer), a
    single concentric cut anchored on the *union* ``OH-A ∪ OH-B`` cannot
    guarantee that the kept virtuals span each fragment's own tightly-coupled
    shell -- the gap analysis on the union can drop one fragment's shell entirely
    in favour of the other's, breaking additivity by construction.  This was
    shown directly by a span-containment test: at ``n_virtual=2`` the dimer's
    energy-ordered virtuals were a *different* 4-space than the union of the two
    monomers' 2-each (min principal-angle cosine ~0.1, i.e. containment failing),
    at *every* separation -- including 20 A where nothing overlaps, ruling out
    BSSE as the cause.

    The fix is to select per fragment: score each virtual by its Mulliken
    population on fragment ``g`` alone, take the concentric-gap cut for ``g``, and
    keep the *union* over ``g`` of the selected virtuals.  The dimer active-virtual
    span is then the direct sum of the per-fragment shells by construction, so
    ``span(C_virt_dimer) ⊇ span(C_virt_A) ⊕ span(C_virt_B)``.

    Args:
        overlap: AO overlap matrix ``S`` (``adapter._s``), shape (nao, nao).
        fragment_ao_groups: one AO-index array per physical fragment.  A single
            group reproduces :func:`mulliken_selector` exactly (verified by the
            length-1 delegation below), so this is a strict generalisation.
        gap_tol: shell-boundary tolerance, per fragment (see
            :func:`mulliken_selector`).
        max_virtual_per_fragment: cap on virtuals kept *for each fragment* (the
            per-fragment budget -- this is what additivity needs, in contrast to a
            single global cap that starves the multi-fragment leg).  ``None`` for
            no cap.
        min_virtual_per_fragment: floor on virtuals kept for each fragment.

    Returns:
        A ``Selector`` returning ``[0, n_occ)`` (all occupied A) followed by the
        sorted union of the per-fragment kept virtual indices.
    """
    groups = [np.asarray(g, dtype=int) for g in fragment_ao_groups]
    if len(groups) == 1:
        # Single fragment: identical to the plain single-shell Mulliken cut
        # (per-fragment cap == global cap when there is one fragment).
        return mulliken_selector(
            overlap,
            groups[0],
            gap_tol=gap_tol,
            max_virtual=max_virtual_per_fragment,
            min_virtual=min_virtual_per_fragment,
        )

    s = np.asarray(overlap)

    def _select(coeff: np.ndarray, energy: np.ndarray, n_occ: int) -> np.ndarray:
        c_virt = coeff[:, n_occ:]
        n_virt = c_virt.shape[1]
        occupied = np.arange(n_occ)
        if n_virt == 0:
            return occupied

        sc = s @ c_virt
        kept: set[int] = set()
        for frag in groups:
            # Fragment population of each virtual on THIS fragment only.
            w = np.einsum("mv,mv->v", sc[frag, :], c_virt[frag, :])
            order = np.argsort(-w)
            keep = _gap_cut(
                w[order],
                gap_tol=gap_tol,
                max_virtual=max_virtual_per_fragment,
                min_virtual=min_virtual_per_fragment,
            )
            kept.update(int(n_occ + v) for v in order[:keep])

        kept_virtuals = np.sort(np.fromiter(kept, dtype=int))
        return np.concatenate([occupied, kept_virtuals]).astype(int)

    return _select


def concentric_localization_selector(
    overlap: np.ndarray,
    fragment_ao: np.ndarray,
    fock: np.ndarray,
    *,
    n_shells: int = 0,
    min_virtual: int = 0,
    max_virtual: int | None = None,
    canonical: bool = False,
    shell0_size: int | None = None,
) -> VirtualLocalizer:
    """Build the *iterative* Concentric Localization (CL) virtual-space localiser.

    The full Claudino et al. algorithm (*J. Chem. Theory Comput.* **2019**, 15,
    6085), of which :func:`mulliken_selector`/:func:`spade_virtual_selector` are the
    single-shell case.  It **rotates** the virtual block (so it is a
    :data:`VirtualLocalizer`, not a :class:`Selector`) and grows the retained space
    one *shell* at a time:

    * **Shell 0** projects the virtuals onto the active-fragment AO basis
      (``proj = S[frag, :]``) and SVDs ``C_spanᵀ proj C_ker`` to split the
      fragment-*spanned* virtuals (SVD rank) from the orthogonal kernel remainder.
    * **Each further shell** SVDs the current span against the current kernel
      *through the embedded Fock matrix* (``proj = fock``) and appends the newly
      Fock-coupled virtuals to the span.

    The kept count therefore *grows* with ``n_shells`` and is fixed by SVD rank, not
    a gap in a weight vector -- the shell structure itself is the truncation.  To
    reuse :meth:`ProjectionEmbeddingAdapter.build_orbitals`' localiser branch
    unchanged, the returned callable rotates the *whole* virtual block into
    ``[kept shells | kernel remainder]``, returns a synthetic descending ``sigma2``
    (``1`` for kept, ``0`` for the remainder), and pins ``max_virtual == min_virtual
    == k`` (the kept count) on itself so the shared :func:`_gap_cut` returns exactly
    ``k`` regardless of the synthetic weights.

    S-orthonormality (the correctness gate the downfold relies on) is preserved
    because every retained/kernel column is the *incoming* S-orthonormal ``C_virt``
    rotated by an orthogonal SVD factor ``V`` (``(C_virt V)ᵀ S (C_virt V) = Vᵀ I V =
    I``); ``proj`` -- the non-symmetric ``S[frag,:]`` or the Fock -- only *chooses* the
    rotation, it is never multiplied into the returned coefficients.  The projected
    ``C'`` used to seed shell 0 is an SVD input only and is discarded.  A final
    assert (and, if round-off drifts it, a symmetric re-S-orthonormalisation) keeps
    the invariant exact.

    Args:
        overlap: AO overlap matrix ``S`` (``adapter._s``), shape (nao, nao).
        fragment_ao: AO indices of the active-fragment atoms (as
            :func:`mulliken_selector`).
        fock: embedded Fock matrix ``F_emb`` (``adapter._fock``), shape (nao, nao) --
            the shell-expansion metric for shells >= 1.  Unlike SPADE, CL needs it.
            A call may override it (``localiser(coeff, n_occ, fock=...)``): the
            per-spin path passes each channel's own ``F_emb``, so that channel's shells
            grow through its own Fock coupling rather than the spin-summed one.
        n_shells: number of Fock shell expansions after shell 0 (``0`` -> the
            fragment-spanned shell only, the single-shell case; the paper's
            convergence graphs suggest one cycle suffices in the working basis).
        min_virtual: floor on the kept count (forwarded through the pinned cap).
        max_virtual: hard ceiling on the kept count, applied *after* the shell
            structure sets ``k`` -- a solver-budget knob for small simulations,
            **not** part of the CL construction.  Because the rotated block is
            ordered most-local first (shell 0, then successive Fock shells), a cap
            of ``m < k`` keeps the ``m`` most fragment-local rotated virtuals and
            drops the outermost shell tail.  ``None`` (default) keeps the full
            shell-determined ``k``.  A cap below ``min_virtual`` wins (the ceiling
            is the harder constraint when the two conflict).
        canonical: pseudocanonicalize the kept shells -- diagonalize ``F_emb`` within
            their span and order them by ascending orbital energy -- so a
            ``max_virtual`` cap keeps the *lowest-energy* kept virtuals instead of the
            most fragment-local ones.  This is the post-processing of the reference
            projection-embedding implementation (Rossmannek et al., J. Phys. Chem. Lett.
            2023), whose active virtuals are the lowest canonical orbitals of the
            truncated span.  An orthogonal rotation of the kept columns, so
            S-orthonormality is untouched.
        shell0_size: keep exactly this many shell-0 columns instead of the SVD rank.
            The reference implementation keeps ``num_basis_functions`` (the fragment AO
            count) columns even when fewer singular values are non-zero, so its shell 0
            then includes null-space vectors whose choice is up to LAPACK.  Set it
            (with ``len(fragment_ao)``) only to reproduce that code; ``None`` (default)
            keeps the mathematically defined fragment-coupled span.

    Returns:
        A ``VirtualLocalizer`` closing over ``S``, the fragment, and ``F_emb``.
    """
    s = np.asarray(overlap, dtype=float)
    f = np.asarray(fock, dtype=float)
    frag = np.asarray(fragment_ao, dtype=int)

    def _span_kernel(c_span: np.ndarray, c_ker: np.ndarray, proj: np.ndarray):
        """SVD-split ``c_ker`` by its ``proj``-coupling to ``c_span``.

        Returns ``(cn, cn_ker)``: the columns of ``c_ker`` that couple to
        ``c_span`` through ``proj`` (SVD rank, singular value > tol), and the
        orthogonal remainder.  Both are ``c_ker`` rotated by the (orthogonal) right
        singular vectors, so both inherit ``c_ker``'s S-orthonormality.
        """
        if c_ker.shape[1] == 0 or c_span.shape[1] == 0:
            return c_ker[:, :0], c_ker
        m = c_span.T @ proj @ c_ker
        _u, svals, vt = np.linalg.svd(m)
        rank = int(np.sum(svals > 1.0e-10))
        v = vt.T  # right singular vectors: an orthogonal rotation of c_ker's columns
        return c_ker @ v[:, :rank], c_ker @ v[:, rank:]

    def _localise(
        coeff: np.ndarray, n_occ: int, fock: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        c_virt = coeff[:, n_occ:]
        n_virt = c_virt.shape[1]
        if n_virt == 0:
            return coeff.copy(), np.empty(0)
        f_shell = f if fock is None else np.asarray(fock, dtype=float)

        # Shell 0: project onto the fragment AO basis and split spanned vs kernel.
        # ``c_virt_prime`` is the projection SEED for the SVD only -- it is NOT
        # S-orthonormal and is never returned as coefficients.
        s_pbwb = s[frag, :]  # (n_frag, nao)
        s_inv = np.linalg.inv(s[np.ix_(frag, frag)])
        c_virt_prime = s_inv @ s_pbwb @ c_virt
        if shell0_size is None:
            c_kept, c_ker = _span_kernel(c_virt_prime, c_virt, s_pbwb)
        else:
            _u, _sv, vt = np.linalg.svd(c_virt_prime.T @ s_pbwb @ c_virt)
            c_kept, c_ker = c_virt @ vt.T[:, :shell0_size], c_virt @ vt.T[:, shell0_size:]

        # Shells 1..n_shells: grow the span through the Fock coupling to the kernel.
        for _ in range(n_shells):
            c_new, c_ker = _span_kernel(c_kept, c_ker, f_shell)
            if c_new.shape[1] == 0:
                break  # kernel exhausted / no further Fock coupling
            c_kept = np.hstack([c_kept, c_new])

        if canonical and c_kept.shape[1] > 1:
            # eigh returns ascending eigenvalues: the cap below then keeps the lowest.
            _w, u_kept = np.linalg.eigh(c_kept.T @ f_shell @ c_kept)
            c_kept = c_kept @ u_kept

        k = c_kept.shape[1]
        # Full rotated virtual block: kept shells first, then the kernel remainder,
        # so build_orbitals' active = [0, n_occ + k) picks exactly the kept shells.
        virt_rot = np.hstack([c_kept, c_ker])

        # Correctness gate: the retained + remainder columns must stay S-orthonormal
        # (the downfold treats the active columns as an opaque S-orthonormal set).
        gram = virt_rot.T @ s @ virt_rot
        if not np.allclose(gram, np.eye(virt_rot.shape[1]), atol=1.0e-8):
            # Symmetric (Löwdin) re-orthonormalisation against S, then re-check.  A
            # slip beyond round-off means the rotation was not orthogonal -- fail loud.
            w_g, u_g = np.linalg.eigh(gram)
            if np.any(w_g <= 1.0e-10):
                raise ValueError(
                    "concentric-localization virtual block lost rank under S "
                    "(non-orthogonal rotation); the span/kernel split is inconsistent"
                )
            virt_rot = virt_rot @ (u_g * (1.0 / np.sqrt(w_g))) @ u_g.T
            gram = virt_rot.T @ s @ virt_rot
            if not np.allclose(gram, np.eye(virt_rot.shape[1]), atol=1.0e-6):
                raise ValueError(
                    "concentric-localization virtual block is not S-orthonormal "
                    f"(max |CᵀSC - I| = {np.abs(gram - np.eye(gram.shape[0])).max():.2e})"
                )

        # Shell structure sets k; an optional solver-budget cap truncates it from
        # the most-local end (the rotated block is ordered shell-0 first).  The cap
        # is the harder constraint, so it also overrides the min_virtual floor.
        k_keep = k if max_virtual is None else min(k, max_virtual)

        c_loc = coeff.copy()
        c_loc[:, n_occ:] = virt_rot
        # Synthetic descending weight so _gap_cut's invariants hold; the pinned cap
        # below is what actually fixes the count at k_keep.
        sigma2 = np.concatenate([np.ones(k_keep), np.zeros(n_virt - k_keep)])
        _localise.max_virtual = k_keep  # type: ignore[attr-defined]
        _localise.min_virtual = k_keep  # type: ignore[attr-defined]
        return c_loc, sigma2

    # gap_tol < 1 so the synthetic 1->0 step is a valid boundary; max/min_virtual are
    # (re)pinned to the kept count k inside _localise, computed once coeff is known.
    _localise.gap_tol = 0.5  # type: ignore[attr-defined]
    _localise.max_virtual = None  # type: ignore[attr-defined]
    _localise.min_virtual = min_virtual  # type: ignore[attr-defined]
    # Tells callers they may pass a per-call ``fock`` (see the ``fock`` argument).
    _localise.accepts_fock = True  # type: ignore[attr-defined]
    return _localise


def spade_virtual_selector(
    overlap: np.ndarray,
    fragment_ao: np.ndarray,
    *,
    gap_tol: float = 1.0e-3,
    max_virtual: int | None = None,
    min_virtual: int = 0,
) -> VirtualLocalizer:
    """Build a SPADE virtual-space *localiser* that rotates then cuts on σ².

    Unlike :func:`mulliken_selector`, which ranks the *canonical* virtuals by
    their Mulliken fragment population, this **rotates** the virtual block so each
    rotated virtual carries a definite fragment weight, then cuts on the gap in
    those weights.  The rotation is the SPADE construction (Claudino & Mayhall,
    doi:10.1021/acs.jctc.9b00682): Löwdin-orthogonalise the virtuals, restrict to
    the fragment AO rows, and diagonalise the resulting Gram matrix.  Its
    eigenvalues ``σ² ∈ [0, 1]`` are the fragment weights (basis-invariant, since the
    rotation is an orthogonal congruence), and its eigenvectors are the rotation.

    The returned callable is a :data:`VirtualLocalizer`, not a :class:`Selector`:
    ``(coeff, n_occ) -> (coeff_localised, sigma2)``.  ``coeff_localised`` is ``coeff``
    with only the virtual block (columns ``n_occ:``) replaced by the σ²-ordered
    rotated virtuals; the occupied block and any non-A columns are untouched.
    :meth:`ProjectionEmbeddingAdapter.build_orbitals` applies the gap cut
    (:func:`_gap_cut`, shared with the Mulliken path) to ``sigma2`` and appends the
    kept rotated virtuals to the occupied block.

    Args:
        overlap: AO overlap matrix ``S`` (``adapter._s``), shape (nao, nao).  Used
            for the Löwdin orthogonalisation; at ``S = I`` σ² reduces to the plain
            fragment population ``Σ_{μ∈frag} C[μ,v]²`` (same weight the Mulliken cut
            scores), so the two paths coincide when the AO basis is orthonormal.
        fragment_ao: AO indices of the active-fragment atoms (as
            :func:`mulliken_selector`).
        gap_tol: minimum ``σ²_i − σ²_{i+1}`` counting as a shell boundary; the cut is
            at the largest qualifying gap.  No gap over the tolerance keeps every
            virtual (refuses to truncate on noise).
        max_virtual: hard cap after the gap cut (solver-budget ceiling).
        min_virtual: floor on the kept count.

    Returns:
        A ``VirtualLocalizer`` closing over ``S`` and the fragment.
    """
    s = np.asarray(overlap, dtype=float)
    frag = np.asarray(fragment_ao, dtype=int)
    # Löwdin S^{1/2} via the symmetric eigendecomposition of the SPD overlap.
    w_s, u_s = np.linalg.eigh(s)
    s_half = (u_s * np.sqrt(np.clip(w_s, 0.0, None))) @ u_s.T
    s_half = 0.5 * (s_half + s_half.T)

    def _localise(coeff: np.ndarray, n_occ: int) -> tuple[np.ndarray, np.ndarray]:
        c_virt = coeff[:, n_occ:]
        n_virt = c_virt.shape[1]
        if n_virt == 0:
            return coeff.copy(), np.empty(0)

        # SPADE weights: Gram of the Löwdin-orthogonalised virtuals restricted to
        # the fragment rows.  M is (n_virt, n_virt), PSD, eigenvalues σ² in [0, 1].
        a = (s_half @ c_virt)[frag, :]  # (n_frag, n_virt), Löwdin, fragment rows
        m = a.T @ a
        m = 0.5 * (m + m.T)
        evals, rot = np.linalg.eigh(m)  # ascending
        order = np.argsort(evals)[::-1]  # descending fragment weight
        sigma2 = np.clip(evals[order], 0.0, 1.0)
        v = rot[:, order]  # orthogonal rotation, σ²-ordered

        c_loc = coeff.copy()
        c_loc[:, n_occ:] = c_virt @ v  # rotated virtuals, still S-orthonormal
        return c_loc, sigma2

    # Carry the cut knobs on the localiser so build_orbitals can apply _gap_cut
    # identically to the Mulliken path (see ProjectionEmbeddingAdapter.build_orbitals).
    _localise.gap_tol = gap_tol  # type: ignore[attr-defined]
    _localise.max_virtual = max_virtual  # type: ignore[attr-defined]
    _localise.min_virtual = min_virtual  # type: ignore[attr-defined]
    return _localise


def fragment_ao_indices(basis, active_atoms) -> np.ndarray:
    """AO indices of the ``active_atoms`` in ``basis``.

    ``basis`` is a QM backend (:class:`~embasi_qiskit_integration.backends.base.AOBasis`)
    or, for existing callers, a PySCF ``Mole``.  ``active_atoms`` are indices into the
    *reordered* atom list the workflow hands the QM code (region-1 atoms first), so they
    line up with the basis.
    """
    basis = as_ao_basis(basis)
    direct = getattr(basis, "atom_ao_indices", None)
    if callable(direct):
        return direct(list(active_atoms))
    return atom_ao_indices(basis.ao_atom_map(), list(active_atoms))


# --------------------------------------------------------------------------- #
# AVAS (Atomic Valence Active Space)
# --------------------------------------------------------------------------- #
#
# Sayfutyarova, Sun, Chan & Knizia, J. Chem. Theory Comput. 2017, 13, 4063
# (doi:10.1021/acs.jctc.7b00128): project the occupied and virtual blocks separately
# onto a set of target valence AOs in a minimal reference basis, rotate each block to
# the eigenvectors of that projection, and keep the orbitals whose projected weight
# exceeds a threshold.  The construction follows PySCF's ``mcscf.avas`` (without IAOs);
# the rotation itself lives on the adapter
# (:meth:`ProjectionEmbeddingAdapter.build_orbitals_avas`), since it must stay inside
# span(A) and respect the embedding's occupied/virtual split.
#
# ``fragment_ao_projector`` is the same metric built from target AOs in the *working*
# basis instead of a MINAO reference.  It is what ``--selector ao-avas`` uses: the
# selection then depends only on fixed fragment AOs and fixed occupied/virtual counts,
# not on canonical orbital energies, and it is available on NAO/pseudopotential
# backends that expose neither ``valence_ao_projector`` nor per-AO shell labels.


def avas_ao_projector(basis, ao_labels, atoms=None, minao: str = "minao") -> np.ndarray:
    """AO-basis matrix ``M`` whose ``C^T M C`` is the AVAS projection of orbitals ``C``.

    ``M = S_21^T S_22^{-1} S_21``, with ``S_22`` the overlap of the target reference AOs
    and ``S_21`` their cross overlap with the system's basis: ``c^T M c`` is the squared
    norm of orbital ``c`` projected onto span(target AOs), in ``[0, 1]`` for an
    S-normalised ``c``.

    Args:
        basis: the QM backend (or a PySCF ``Mole``) the orbitals are expanded in.  It
            must provide ``valence_ao_projector`` (the ``minao_projector`` capability).
        ao_labels: AO labels selecting the target AOs in the reference basis
            (``"Fe 3d"``, ``"C 2p"``, ...).
        atoms: restrict the targets to the AOs on these atoms (indices into the
            system); ``None`` keeps every match.  The workflow passes the active
            fragment, so a label also present in the environment only targets the
            fragment's copy.
        minao: the minimal reference basis the labels are matched in (PySCF's AVAS
            default).

    Raises:
        ValueError: if no reference AO matches.
        MissingCapabilityError: if the backend cannot build a reference-basis projector.
    """
    basis = as_ao_basis(basis)
    require_capability(basis, CAP_MINAO_PROJECTOR, "the AVAS valence-AO projector")
    return basis.valence_ao_projector(list(ao_labels), atoms=atoms, minao=minao)


def _overlap_matrix(basis) -> np.ndarray:
    """AO overlap ``S`` from an AOIntegrals backend (the code-agnostic seam)."""
    method = getattr(basis, "overlap", None)
    if callable(method):
        return np.asarray(method(), dtype=float)
    raise TypeError(
        f"{type(basis).__name__} has no overlap(); a fragment-AO projector needs an "
        "AOIntegrals backend (e.g. adapter.ints), not a bare QM-code molecule object"
    )


def fragment_ao_projector(basis, ao_labels=None, atoms=None) -> np.ndarray:
    """Same-basis fragment-AO projector ``M = S B (B^T S B)^-1 B^T S``.

    The working-basis counterpart of :func:`avas_ao_projector`.  ``B`` selects target
    AOs *in the system's own basis* (no MINAO reference), so for an S-normalised
    orbital ``c`` the weight ``c^T M c`` is the squared S-norm of its S-orthogonal
    projection onto ``span(target AOs)`` and lies in ``[0, 1]``.  This is the metric
    :meth:`ProjectionEmbeddingAdapter.build_orbitals_avas` diagonalises inside the
    occupied and virtual blocks of subsystem A.

    Why a same-basis variant: the MINAO projector behind ``--selector avas`` needs
    ``valence_ao_projector``, which NAO/pseudopotential backends (SIESTA) do not
    provide.  Matching labels in the working basis needs only ``search_ao_label``.
    SIESTA supplies it lazily from the ``.ORB_INDX`` its low-level run writes; when
    a backend reports no shell labels at all, the fragment's AO *indices* from
    ``atom_ao_indices`` are still enough: pass ``ao_labels=None`` to target every AO
    of ``atoms``.  On a pseudopotential basis those are the valence functions, so
    the "all fragment AOs" fallback is a chemically sensible valence target rather
    than a core-inclusive one -- but it also includes every valence shell and
    polarisation function, so it is a different target space from a specific
    ``"N 2p"`` request.

    The target AOs need not lie exactly inside ``span(A)``.  ``M`` is only the
    *metric*; the selected orbitals are rotations of subsystem-A columns, so they
    stay S-orthogonal to ``P_B`` whatever the target AOs' small environment tails.

    Args:
        basis: an :class:`~embasi_qiskit_integration.backends.base.AOIntegrals`
            backend (it must expose ``overlap()``, ``atom_ao_indices`` and, for
            labels, ``search_ao_label``).  The workflow passes ``adapter.ints``; this
            is deliberately the backend seam rather than a bare QM-code molecule, so
            no PySCF-specific object leaks outside ``backends/``.
        ao_labels: AO labels matched in the working basis (``"N 2p"``, ``"C 2p"``).
            ``None`` targets every AO of the fragment ``atoms`` -- the portable
            choice for a backend without AO labels.
        atoms: fragment atom indices (indices into the basis' atom order).  ``None``
            uses every atom.

    Returns:
        ``(nao, nao)`` symmetric AO-basis metric ``M``.

    Raises:
        ValueError: no target AO, labels requested on a backend that reports none, or
            linearly dependent target AOs (``B^T S B`` singular).
        TypeError: the object is not an ``AOIntegrals`` backend (no ``overlap()``).
    """
    basis = as_ao_basis(basis)
    s = _overlap_matrix(basis)
    nao = int(basis.nao)
    if s.shape != (nao, nao):
        raise ValueError(f"overlap has shape {s.shape}, expected ({nao}, {nao})")

    frag = (
        np.arange(nao, dtype=int)
        if atoms is None
        else np.asarray(fragment_ao_indices(basis, atoms), dtype=int)
    )
    target = frag
    if ao_labels:
        search = getattr(basis, "search_ao_label", None)
        if not callable(search):
            raise ValueError(
                "ao_labels were requested but this backend reports no AO shell labels; "
                "omit ao_labels to target every AO of the fragment (the portable "
                "choice for SIESTA's pseudopotential/NAO basis)"
            )
        try:
            hits = np.asarray(search(list(ao_labels)), dtype=int)
        except MissingCapabilityError as exc:
            raise ValueError(
                "ao_labels were requested but this backend reports no AO shell labels "
                "(SIESTA's ASI basis descriptors carry only the atom index); omit "
                "ao_labels to target every AO of the fragment"
            ) from exc
        target = np.intersect1d(hits, frag)
        if target.size == 0:
            try:
                labels = basis.ao_labels()
                shells = sorted({" ".join(labels[i][1:3]) for i in frag})
                available = f"; the fragment's shells are: {', '.join(shells)}"
            except (MissingCapabilityError, AttributeError):
                available = ""
            raise ValueError(
                f"fragment-AO labels {list(ao_labels)} match no AO of the active "
                f"fragment{available}"
            )
    if target.size == 0:
        raise ValueError("the active fragment has no AOs to project onto")

    b = np.zeros((nao, target.size), dtype=float)
    b[target, np.arange(target.size)] = 1.0
    s22 = b.T @ s @ b
    w = np.linalg.eigvalsh(0.5 * (s22 + s22.T))
    if w.size == 0 or w.min() <= 1.0e-10 * max(float(w.max()), 1.0):
        raise ValueError(
            "the fragment-AO target set is linearly dependent under S "
            f"(overlap eigenvalues {np.array2string(w, precision=2)}); choose "
            "independent AOs/labels"
        )
    m = s @ b @ np.linalg.solve(s22, b.T @ s)
    return 0.5 * (m + m.T)


# --------------------------------------------------------------------------- #
# APC (Approximate Pair Coefficient) ranking
# --------------------------------------------------------------------------- #
#
# King & Gagliardi, J. Chem. Theory Comput. 2021, 17, 7387 (doi:10.1021/acs.jctc.1c00037):
# a cheap surrogate for the DMRG single-orbital entropy (AutoCAS), built from an
# analytic two-configuration model of each occupied-virtual pair. Unlike the
# gap-cut selectors above, APC ranks the OCCUPIED block alongside the virtuals
# and truncates to a fixed active-space budget rather than a gap in the ranking
# metric -- so it uses PySCF's own ``mcscf.apc.Chooser`` for the truncation (the
# ranked-orbital drop-until-budget procedure the paper describes) instead of
# this module's ``_gap_cut``.
#
# The pair-entropy model below covers doubly-occupied/virtual pairs. PySCF's
# ``apc.APC`` handles singly-occupied orbitals (ROHF/UHF) by assigning them a
# synthetic max-entropy value so they are never dropped; ``apc_active_space``
# forwards a ``1`` in ``occ_pattern`` to ``Chooser``, which applies exactly that.
# ``somo_occupation_pattern`` builds such a pattern from per-spin counts.


def apc_pair_coefficients(
    f_diag_occ: np.ndarray, f_diag_virt: np.ndarray, k_diag_virt: np.ndarray
) -> np.ndarray:
    """Eq. 19: analytic pair coefficient ``c_ia`` for every (occupied, virtual) pair.

    Each doubly-occupied/virtual pair is modelled as an independent two-configuration
    CI problem; ``c_ia`` is that problem's exact ground-state mixing coefficient,
    approximated from one-electron quantities alone (eq. 15-18): ``Delta_ia = f_a -
    f_i`` (Koopmans-like orbital energy gap) and ``(ia|ia) ~ 0.5 K_aa`` (the exchange
    integral, approximated from the virtual's own exchange diagonal only).

    Args:
        f_diag_occ: ``(n_occ,)`` Fock expectation value of each candidate occupied
            orbital -- an eigenvalue for a canonical orbital, or ``diag(C^T F C)``
            for a rotated one (e.g. after :func:`concentric_localization_selector`).
        f_diag_virt: ``(n_virt,)`` ditto for the candidate virtuals.
        k_diag_virt: ``(n_virt,)`` exchange-operator diagonal of the candidate
            virtuals, ``diag(C^T K C)``.

    Returns:
        ``(n_occ, n_virt)`` matrix of pair coefficients.
    """
    k12 = 0.5 * np.asarray(k_diag_virt)[None, :]
    delta = np.asarray(f_diag_virt)[None, :] - np.asarray(f_diag_occ)[:, None]
    return -k12 / (delta + np.sqrt(k12**2 + delta**2))


def apc_orbital_entropies(c_pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 12/13: approximate single-orbital entropy from the pair-coefficient matrix.

    Each occupied orbital's entropy is computed from its row (summed over all
    candidate virtuals it pairs with); each virtual's from its column (summed over
    all candidate occupieds). A pair coefficient of 0 contributes 0 entropy for
    either orbital, so an orbital that pairs with nothing ranks lowest -- the
    intended behaviour for :func:`apc_active_space`'s truncation.

    Args:
        c_pairs: ``(n_occ, n_virt)`` pair coefficients, as returned by
            :func:`apc_pair_coefficients`.

    Returns:
        ``(s_occ, s_virt)``: entropies of shape ``(n_occ,)`` and ``(n_virt,)``.
    """
    c2 = np.asarray(c_pairs) ** 2

    def _entropy(sum_c2: np.ndarray) -> np.ndarray:
        norm2 = 1.0 / (1.0 + sum_c2)
        excited2 = sum_c2 / (1.0 + sum_c2)
        excited_term = np.where(excited2 > 0, excited2 * np.log(excited2), 0.0)
        return -norm2 * np.log(norm2) - excited_term

    return _entropy(c2.sum(axis=1)), _entropy(c2.sum(axis=0))


def somo_occupation_pattern(n_orb: int, n_occ_a: int, n_occ_b: int) -> np.ndarray:
    """Build an APC ``occ_pattern`` with ``2``/``1``/``0`` from per-spin occupied counts.

    Doubly occupied below ``min(n_occ_a, n_occ_b)``, singly occupied (a SOMO) between the
    two counts, empty above ``max``. This is what lets APC rank an open-shell candidate
    set: ``Chooser`` gives every ``1`` a synthetic max entropy, so the singly-occupied
    orbitals carrying the spin are always retained.

    Args:
        n_orb: total candidate orbitals.
        n_occ_a: alpha occupied count.
        n_occ_b: beta occupied count.

    Returns:
        ``(n_orb,)`` int array of occupations.
    """
    if min(n_occ_a, n_occ_b) < 0 or max(n_occ_a, n_occ_b) > n_orb:
        raise ValueError(f"occupied counts ({n_occ_a}, {n_occ_b}) do not fit n_orb={n_orb}")
    lo, hi = sorted((int(n_occ_a), int(n_occ_b)))
    pattern = np.zeros(n_orb, dtype=int)
    pattern[:lo] = 2
    pattern[lo:hi] = 1
    return pattern


def column_occupation_from_density(
    mo: np.ndarray, dm_a: np.ndarray, dm_b: np.ndarray, s_ao: np.ndarray
) -> np.ndarray:
    """Per-column occupation in ``{0, 1, 2}`` by projecting a spin density onto ``mo``.

    Why this rather than a positional rule: "columns below ``n_occ`` are doubly occupied"
    is only true for *canonical* orbitals in energy order. Any procedure that rotates
    within blocks -- AVAS, concentric localization, SPADE, the APC path in
    :meth:`~embasi_qiskit_integration.projection_embedding_adapter.ProjectionEmbeddingAdapter.build_orbitals_apc_concentric`
    -- destroys that ordering while leaving the span intact, so the positional reading
    silently mislabels which orbitals hold the unpaired electrons. Projecting the density
    asks each column directly.

    Args:
        mo: ``(nao, ncol)`` AO expansion of the columns to characterise.
        dm_a: alpha AO-basis 1-RDM ``(nao, nao)``.
        dm_b: beta AO-basis 1-RDM ``(nao, nao)``.
        s_ao: AO overlap ``(nao, nao)``.

    Returns:
        ``(ncol,)`` int array of occupations: ``2`` (both channels), ``1`` (a SOMO),
        ``0`` (empty). Feed it straight to :func:`apc_active_space`, or derive per-spin
        counts with :func:`spin_counts_from_occupation`.
    """
    mo = np.asarray(mo, dtype=float)
    sm = np.asarray(s_ao, dtype=float) @ mo
    na = np.einsum("ai,ab,bi->i", sm, np.asarray(dm_a, dtype=float), sm)
    nb = np.einsum("ai,ab,bi->i", sm, np.asarray(dm_b, dtype=float), sm)
    return ((na >= 0.5).astype(int) + (nb >= 0.5).astype(int)).astype(int)


def spin_counts_from_occupation(occ_pattern: np.ndarray) -> tuple[int, int]:
    """``(n_alpha, n_beta)`` from a ``{0, 1, 2}`` occupation pattern.

    The inverse of :func:`somo_occupation_pattern`: an alpha electron in every
    ``occ >= 1``, a beta electron in every ``occ == 2``. Use it to turn
    :func:`column_occupation_from_density`'s output into the ``n_occ``/``n_occ_b`` pair
    :class:`~embasi_qiskit_integration.projection_embedding_adapter.EmbeddedOrbitals`
    wants.
    """
    occ = np.asarray(occ_pattern)
    return int(np.count_nonzero(occ >= 1)), int(np.count_nonzero(occ >= 2))


def apc_active_space(
    occ_pattern: np.ndarray,
    entropies: np.ndarray,
    max_size: int | tuple[int, int],
    *,
    fixed: bool = False,
) -> np.ndarray:
    """Rank-and-truncate orbitals to ``max_size`` via PySCF's ``Chooser``.

    Delegates the truncation itself to ``pyscf.mcscf.apc.Chooser`` -- the
    ranked-orbital procedure of the APC paper (repeatedly drop the lowest-entropy
    orbital, respecting a minimum-reasonability floor of >=1 active electron and
    not every active orbital doubly occupied) -- rather than this module's
    gap-based ``_gap_cut``, since APC's truncation is budget-driven, not a gap in
    the ranking metric.

    ``Chooser`` also wants the orbital coefficients themselves (to build its own
    ``casorbs`` return value), and asserts they are square -- ``(n_mo, n_mo)``, the
    normal SCF convention. This adapter's ``EmbeddedOrbitals.coeff`` is instead
    ``(nao, n_A)`` with ``n_A < nao`` whenever there is a real environment (i.e.
    essentially always, once ``restrict_to_a`` has deflated subsystem B out), so it
    is never square and would trip that assertion. Since only ``active_idx`` is
    used here (:meth:`ProjectionEmbeddingAdapter.build_orbitals_apc_concentric`
    rebuilds ``EmbeddedOrbitals`` from its own ``coeff``, never from ``casorbs``),
    an identity placeholder of the right size satisfies ``Chooser`` without
    needing real coefficients at all.

    To rank only a subset of candidates (e.g. a concentric-localization pool),
    give every orbital outside that subset an entropy far below the real range
    (e.g. a large negative sentinel) so ``Chooser`` always drops them first.

    Args:
        occ_pattern: ``(n_orb,)`` occupation of each orbital -- ``2`` (doubly
            occupied), ``1`` (singly occupied, ROHF/UHF) or ``0`` (empty). PySCF's
            ``Chooser`` handles a singly-occupied entry by assigning it a synthetic
            max-entropy value, so a SOMO is always kept; see
            :func:`apc_orbital_entropies` for how the entropies themselves are built.
        entropies: ``(n_orb,)`` importance ranking from :func:`apc_orbital_entropies`
            (occupied and virtual entropies scattered back to matching positions).
        max_size: ``Chooser``'s size constraint -- an int ceiling on the *number of
            active orbitals* (not an N_CSF count, despite the paper's ranked-orbital
            framing -- this is ``pyscf.mcscf.apc.Chooser``'s own convention), or a
            fixed ``(nelec, norb)`` target when ``fixed=True`` (``Chooser`` requires
            a tuple in that mode).
        fixed: if True, select exactly the highest-entropy ``(nelec, norb)`` rather
            than dynamically dropping until the orbital-count budget is met.
            Recommended when this feeds a self-consistency loop (see the outer-loop
            stability note in
            :meth:`ProjectionEmbeddingAdapter.build_orbitals_apc_concentric`):
            dynamic dropping can flip which orbital is dropped near a budget
            boundary as the Fock matrix drifts cycle to cycle, changing the qubit
            count mid-run; a fixed target does not.

    Returns:
        Sorted indices, positional into ``occ_pattern``/``entropies``, of the
        selected active space.
    """
    from pyscf.mcscf.apc import Chooser

    occ_pattern = np.asarray(occ_pattern)
    placeholder_orbs = np.eye(occ_pattern.size)
    chooser = Chooser(
        placeholder_orbs,
        occ_pattern,
        np.asarray(entropies),
        max_size=max_size,
        fixed=fixed,
        verbose=0,
    )
    _, _, _, active_idx = chooser.kernel()
    return np.asarray(sorted(active_idx), dtype=int)


# --------------------------------------------------------------------------- #
# MP2 correlation ranking (the ``correlation-avas`` virtual selector)
# --------------------------------------------------------------------------- #
#
# The AVAS target-AO weight is a *one-electron* metric.  On SIESTA's DZP basis it
# saturates: several high-energy N-localised pi* virtuals reach weight 1.000 while a
# C-localised pi* pair, only ~0.04 Ha above the occupied pi, carries weight
# 0.887/0.892.  A weight-only cut therefore keeps the wrong pair, and no
# one-electron ranking can tell them apart.  The missing information is the
# two-electron coupling and the downfolded excitation denominator, both of which
# this module already receives from the ERI provider and ``h1``.
#
# The ranking below is closed-shell, *semicanonical* MP2 on the embedded one-body
# operator:
#
#   f = h1 + 2J[d_occ] - K[d_occ]           (reference Fock, active occupieds only)
#   f_oo -> diag(e_o), f_vv -> diag(e_v)    (separate rotations, MP2 invariant)
#   T_ij^ab = (ia|jb) / (e_i + e_j - e_a - e_b)
#   E_corr  = sum_ijab (2(ia|jb) - (ib|ja)) T_ij^ab
#
# Semicanonicalisation is essential rather than cosmetic: the F_emb eigenvectors
# that AVAS ranks are *not* eigenstates of f, and the virtual-virtual block of f
# is far from diagonal (measured max off-diagonal 5.9 Ha on butyronitrile 2.5 A).
# Using the raw F_emb eigenvalues as denominators would be a different, and much
# worse, approximation.
#
# The per-orbital *marginal correlation* splits E_corr symmetrically over its two
# indices,
#
#   m_a = 1/2 [ sum_ijb E_ijab + sum_ijb E_ijba ],
#
# so sum_a m_a = E_corr.  It is the ranking metric: a virtual whose excitation
# has a large coupling and a small denominator ranks far below one that is only
# high-weight.  MP2 natural occupations are returned as a diagnostic but are not
# used for ranking -- near-degenerate denominators can inflate a NOON while its
# net energy contribution cancels (seen at butyronitrile 0.8 A).
#
# Only the *virtual* block is ranked here.  Ranking the occupied block by the
# same marginal metric was tested against the archived butyronitrile dump and is
# not safe: with all seven subsystem-A occupieds as the reference it ranks a
# near-degenerate mixture of the two deepest core orbitals (marginal -0.22 Ha)
# above the pi pair (-0.072 Ha), because core-valence mixing dominates the
# occupied Fock rotations.  The chemically anchored AVAS target-AO weight is kept
# for the occupied block, and MP2 only replaces the saturated virtual cut.


@dataclass(frozen=True)
class MP2Correlation:
    """Semicanonical MP2 decomposition of a closed-shell reference.

    All arrays are in the *semicanonical* basis: the returned rotations map the
    caller's occupied/virtual blocks into the eigenvectors of the reference Fock
    (columns of ``occupied_rotation`` / ``virtual_rotation``).

    Attributes:
        occupied_rotation: ``(n_occ, n_occ)`` rotation diagonalizing ``f_oo``.
        occupied_energies: ``(n_occ,)`` semicanonical occupied Fock eigenvalues.
        virtual_rotation: ``(n_virt, n_virt)`` rotation diagonalizing ``f_vv``.
        virtual_energies: ``(n_virt,)`` semicanonical virtual Fock eigenvalues.
        occupied_natural_occupations: ``(n_occ,)`` MP2 1-RDM diagonal, occupied block
            (reference occupation 2 per spatial orbital plus the MP2 correction).
        virtual_natural_occupations: ``(n_virt,)`` MP2 1-RDM diagonal, virtual block.
        occupied_marginal_correlation: ``(n_occ,)`` symmetric split of ``e_corr`` over
            the occupied indices (Ha; negative for normal correlation).
        virtual_marginal_correlation: ``(n_virt,)`` the same over virtual indices;
            the selector ranks these ascending (most negative first).
        pair_energies: ``(n_occ, n_occ, n_virt, n_virt)`` per-excitation contribution
            to ``e_corr`` in the semicanonical basis.
        e_corr: total MP2 correlation energy (Ha).
        n_skipped: number of amplitudes whose denominator fell below
            ``denominator_floor`` and were excluded.
    """

    occupied_rotation: np.ndarray
    occupied_energies: np.ndarray
    virtual_rotation: np.ndarray
    virtual_energies: np.ndarray
    occupied_natural_occupations: np.ndarray
    virtual_natural_occupations: np.ndarray
    occupied_marginal_correlation: np.ndarray
    virtual_marginal_correlation: np.ndarray
    pair_energies: np.ndarray
    e_corr: float
    n_skipped: int


def _symmetric(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return 0.5 * (a + a.T)


def mp2_orbital_correlation(
    h1_oo: np.ndarray,
    h1_vv: np.ndarray,
    eri_ovov: np.ndarray,
    eri_oooo: np.ndarray,
    eri_vvoo: np.ndarray,
    *,
    denominator_floor: float = 1.0e-8,
    build_fock: bool = True,
) -> MP2Correlation:
    """Closed-shell semicanonical MP2 from the embedded one- and two-body blocks.

    The reference determinant doubly occupies every column of the occupied block
    and leaves the virtual block empty.  ``h1_oo``/``h1_vv`` are the downfolded
    one-body operator ``C^T (h_emb + veff_hf(dm_inactive)) C`` in the *same*
    occupied/virtual bases as the ERI blocks.  The frozen core, if any, is already
    folded into ``h1`` and is not part of the reference determinant used here (it
    never is: the MP2 reference only fills the active occupieds).

    Args:
        h1_oo: ``(n_occ, n_occ)`` one-body block, active occupieds.
        h1_vv: ``(n_virt, n_virt)`` one-body block, candidate virtuals.
        eri_ovov: ``(n_occ, n_virt, n_occ, n_virt)`` chemists' ``(ia|jb)``.
        eri_oooo: ``(n_occ, n_occ, n_occ, n_occ)`` chemists' ``(ij|kl)``.
        eri_vvoo: ``(n_virt, n_virt, n_occ, n_occ)`` chemists' ``(ab|ij)``.
        denominator_floor: amplitudes with ``|D| <= floor`` are dropped (counted
            in ``n_skipped``) instead of dividing by a near-zero gap.  The default
            only guards exact degeneracies; the ranking is unchanged for measured
            gaps down to 1.5e-3 Ha.
        build_fock: build the MP2 reference Fock ``h1 + 2J - K`` and diagonalize it
            within the two blocks (the production path).  Set ``False`` only for a
            ranking sanity check on a backend that exposes just a diagonal one-body
            spectrum (the archived H2 dumps): the supplied ``h1`` diagonals are then
            used directly as the denominators and the blocks are not rotated.  The
            production selector never uses this mode.

    Returns:
        :class:`MP2Correlation`; see its attributes for the shapes and sign
        conventions.

    Raises:
        ValueError: an input block has the wrong shape.
    """
    h1_oo = _symmetric(h1_oo)
    h1_vv = _symmetric(h1_vv)
    eri_ovov = np.asarray(eri_ovov, dtype=float)
    eri_oooo = np.asarray(eri_oooo, dtype=float)
    eri_vvoo = np.asarray(eri_vvoo, dtype=float)
    n_occ = h1_oo.shape[0]
    n_virt = h1_vv.shape[0]
    if h1_oo.shape != (n_occ, n_occ) or h1_vv.shape != (n_virt, n_virt):
        raise ValueError("h1_oo and h1_vv must be square occupied/virtual blocks")
    if eri_ovov.shape != (n_occ, n_virt, n_occ, n_virt):
        raise ValueError(
            f"eri_ovov has shape {eri_ovov.shape}, expected {(n_occ, n_virt, n_occ, n_virt)}"
        )
    if eri_oooo.shape != (n_occ,) * 4:
        raise ValueError(f"eri_oooo has shape {eri_oooo.shape}, expected {(n_occ,) * 4}")
    if eri_vvoo.shape != (n_virt, n_virt, n_occ, n_occ):
        raise ValueError(
            f"eri_vvoo has shape {eri_vvoo.shape}, expected {(n_virt, n_virt, n_occ, n_occ)}"
        )
    if n_occ == 0 or n_virt == 0:
        return MP2Correlation(
            occupied_rotation=np.eye(n_occ),
            occupied_energies=np.empty(0),
            virtual_rotation=np.eye(n_virt),
            virtual_energies=np.empty(0),
            occupied_natural_occupations=np.full(n_occ, 2.0),
            virtual_natural_occupations=np.empty(0),
            occupied_marginal_correlation=np.empty(0),
            virtual_marginal_correlation=np.empty(0),
            pair_energies=np.empty((n_occ, n_occ, n_virt, n_virt)),
            e_corr=0.0,
            n_skipped=0,
        )

    if build_fock:
        # Reference Fock in the caller's basis.  Only the active occupieds contribute
        # to the mean field: the frozen core is already folded into h1 by the downfold.
        j_oo = np.einsum("ijkk->ij", eri_oooo)
        k_oo = np.einsum("ikjk->ij", eri_oooo)
        f_oo = _symmetric(h1_oo + 2.0 * j_oo - k_oo)
        j_vv = np.einsum("abii->ab", eri_vvoo)
        k_vv = np.einsum("iaib->ab", eri_ovov)
        f_vv = _symmetric(h1_vv + 2.0 * j_vv - k_vv)
        e_o, u_o = np.linalg.eigh(f_oo)
        e_v, u_v = np.linalg.eigh(f_vv)
    else:
        # Ranking-only: the caller swears the blocks are already diagonal.
        e_o, u_o = np.diag(h1_oo).copy(), np.eye(n_occ)
        e_v, u_v = np.diag(h1_vv).copy(), np.eye(n_virt)

    # Rotate (ia|jb) -> (pa|qb) in the semicanonical basis.  ``u_o`` has the old
    # index as its row and the new (eigenvector) index as its column, so the
    # contraction is ``u_o[i, p]``, not ``u_o[p, i]``: the latter rotates by the
    # transpose, which for an orthogonal matrix is the inverse rotation and gives
    # the wrong correlation ranking.
    tsc = np.einsum("iajb,ip,jq,ac,bd->pcqd", eri_ovov, u_o, u_o, u_v, u_v, optimize=True)
    # x_ijab = (ia|jb); y_ijab = (ib|ja).
    x = np.transpose(tsc, (0, 2, 1, 3))
    y = np.transpose(tsc, (0, 2, 3, 1))
    denominator = (
        e_o[:, None, None, None]
        + e_o[None, :, None, None]
        - e_v[None, None, :, None]
        - e_v[None, None, None, :]
    )
    good = np.abs(denominator) > float(denominator_floor)
    amplitudes = np.zeros_like(x)
    np.divide(x, denominator, out=amplitudes, where=good)

    pair_energies = (2.0 * x - y) * amplitudes
    e_corr = float(pair_energies.sum())

    # Unrelaxed MP2 1-RDM (PySCF's conventions: T_ij^ab = (ia|jb)/D, spatial
    # occupations, reference 2 per occupied).
    virt_noon = 2.0 * np.einsum("ijca,ijca->a", amplitudes, amplitudes) - np.einsum(
        "ijca,ijac->a", amplitudes, amplitudes
    )
    occ_correction = 2.0 * np.einsum("ijab,ijab->i", amplitudes, amplitudes) - np.einsum(
        "ijab,ijba->i", amplitudes, amplitudes
    )
    occ_noon = 2.0 - occ_correction

    occ_marginal = 0.5 * (pair_energies.sum(axis=(1, 2, 3)) + pair_energies.sum(axis=(0, 2, 3)))
    virt_marginal = 0.5 * (pair_energies.sum(axis=(0, 1, 3)) + pair_energies.sum(axis=(0, 1, 2)))

    return MP2Correlation(
        occupied_rotation=u_o,
        occupied_energies=e_o,
        virtual_rotation=u_v,
        virtual_energies=e_v,
        occupied_natural_occupations=occ_noon,
        virtual_natural_occupations=virt_noon,
        occupied_marginal_correlation=occ_marginal,
        virtual_marginal_correlation=virt_marginal,
        pair_energies=pair_energies,
        e_corr=e_corr,
        n_skipped=int(np.count_nonzero(~good)),
    )


def select_correlated_orbitals(
    correlation: MP2Correlation, *, n_virtual: int, n_occupied: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the most strongly correlating occupied/virtual semicanonical orbitals.

    Ranks by *marginal correlation energy*, ascending (most negative first): the
    virtual whose excitations contribute most correlation energy is selected,
    regardless of its AVAS weight or F_emb energy.  Ties keep the incoming order.

    Occupied ranking is available but off by default (``n_occupied=0``): the
    archived butyronitrile test shows it prefers a core-valence mixture over the
    pi pair, so callers that want it must opt in and validate the chemistry.

    Args:
        correlation: result of :func:`mp2_orbital_correlation`.
        n_virtual: number of virtuals the CAS budget needs.
        n_occupied: number of occupieds to rank (0 keeps the caller's selection).

    Returns:
        ``(occupied_indices, virtual_indices)`` into the semicanonical rotations,
        each sorted by increasing marginal correlation (most negative first).

    Raises:
        ValueError: more orbitals were requested than the blocks hold.
    """
    n_v = int(n_virtual)
    n_o = int(n_occupied)
    if not 0 <= n_v <= correlation.virtual_marginal_correlation.size:
        raise ValueError(
            f"n_virtual={n_v} does not fit the "
            f"{correlation.virtual_marginal_correlation.size} candidate virtuals"
        )
    if not 0 <= n_o <= correlation.occupied_marginal_correlation.size:
        raise ValueError(
            f"n_occupied={n_o} does not fit the "
            f"{correlation.occupied_marginal_correlation.size} candidate occupieds"
        )
    virt = np.argsort(correlation.virtual_marginal_correlation, kind="stable")[:n_v]
    occ = (
        np.argsort(correlation.occupied_marginal_correlation, kind="stable")[:n_o]
        if n_o
        else np.empty(0, dtype=int)
    )
    return occ.astype(int), virt.astype(int)
