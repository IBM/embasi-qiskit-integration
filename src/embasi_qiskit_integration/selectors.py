# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Active-space selectors for :meth:`ProjectionEmbeddingAdapter.build_orbitals`.

``build_orbitals`` accepts a ``Selector`` hook -- ``(coeff, energy, n_occ) ->
active column indices`` -- so the choice of which virtuals to correlate can be
made data-driven instead of guessed with ``--n_virtual``.

:func:`concentric_selector` implements a concentric-localisation cut in the
lineage of SPADE (Claudino & Mayhall): the occupied subsystem-A space is fixed
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

import numpy as np

from embasi_qiskit_integration.projection_embedding_adapter import (
    Selector,
    VirtualLocalizer,
)

__all__ = [
    "concentric_selector",
    "per_fragment_concentric_selector",
    "spade_virtual_selector",
    "fragment_ao_indices",
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
    :func:`concentric_selector` (Mulliken weights) and :func:`spade_virtual_selector`
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


def concentric_selector(
    overlap: np.ndarray,
    fragment_ao: np.ndarray,
    *,
    gap_tol: float = 1.0e-3,
    max_virtual: int | None = None,
    min_virtual: int = 0,
) -> Selector:
    """Build a virtual-space selector that cuts on the fragment-coupling gap.

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


def per_fragment_concentric_selector(
    overlap: np.ndarray,
    fragment_ao_groups: list[np.ndarray],
    *,
    gap_tol: float = 1.0e-3,
    max_virtual_per_fragment: int | None = None,
    min_virtual_per_fragment: int = 0,
) -> Selector:
    """Concentric cut run *independently per fragment*, then unioned.

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
            group reproduces :func:`concentric_selector` exactly (verified by the
            length-1 delegation below), so this is a strict generalisation.
        gap_tol: shell-boundary tolerance, per fragment (see
            :func:`concentric_selector`).
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
        # Single fragment: identical to the plain concentric cut (per-fragment
        # cap == global cap when there is one fragment).
        return concentric_selector(
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


def spade_virtual_selector(
    overlap: np.ndarray,
    fragment_ao: np.ndarray,
    *,
    gap_tol: float = 1.0e-3,
    max_virtual: int | None = None,
    min_virtual: int = 0,
) -> VirtualLocalizer:
    """Build a SPADE virtual-space *localiser* that rotates then cuts on σ².

    Unlike :func:`concentric_selector`, which ranks the *canonical* virtuals by
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
            :func:`concentric_selector`).
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


def fragment_ao_indices(mol, active_atoms) -> np.ndarray:
    """AO indices of the ``active_atoms`` in ``mol`` (PySCF ``aoslice_by_atom``).

    ``active_atoms`` are indices into the *reordered* atom list the workflow
    hands to ``pyscf.M`` (region-1 atoms first), so they line up with ``mol``.
    """
    aoslice = mol.aoslice_by_atom()
    ranges = [np.arange(aoslice[a, 2], aoslice[a, 3]) for a in active_atoms]
    return np.hstack(ranges).astype(int) if ranges else np.empty(0, dtype=int)
