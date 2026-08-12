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
"""

from __future__ import annotations

import numpy as np

from embasi_qiskit_integration.projection_embedding_adapter import Selector


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
        w_sorted = w[order]

        # Largest gap from the top that clears the tolerance sets the shell edge.
        # Cutting at the *first* qualifying gap slices off after the top virtual
        # whenever the leading shell has any internal jitter above gap_tol (a
        # dense shell of near-degenerate fragment weights), silently discarding
        # the rest of that shell; the *largest* gap is the true shell boundary.
        gaps = w_sorted[:-1] - w_sorted[1:]
        keep = n_virt  # default: keep everything
        significant = np.nonzero(gaps >= gap_tol)[0]
        if significant.size:
            # +1: a gap after index i keeps virtuals 0..i inclusive.
            edge = int(significant[np.argmax(gaps[significant])])
            keep = edge + 1

        keep = max(keep, min_virtual)
        if max_virtual is not None:
            keep = min(keep, max_virtual)
        keep = min(keep, n_virt)

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
            w_sorted = w[order]
            gaps = w_sorted[:-1] - w_sorted[1:]
            keep = n_virt
            significant = np.nonzero(gaps >= gap_tol)[0]
            if significant.size:
                edge = int(significant[np.argmax(gaps[significant])])
                keep = edge + 1
            keep = max(keep, min_virtual_per_fragment)
            if max_virtual_per_fragment is not None:
                keep = min(keep, max_virtual_per_fragment)
            keep = min(keep, n_virt)
            kept.update(int(n_occ + v) for v in order[:keep])

        kept_virtuals = np.sort(np.fromiter(kept, dtype=int))
        return np.concatenate([occupied, kept_virtuals]).astype(int)

    return _select


def fragment_ao_indices(mol, active_atoms) -> np.ndarray:
    """AO indices of the ``active_atoms`` in ``mol`` (PySCF ``aoslice_by_atom``).

    ``active_atoms`` are indices into the *reordered* atom list the workflow
    hands to ``pyscf.M`` (region-1 atoms first), so they line up with ``mol``.
    """
    aoslice = mol.aoslice_by_atom()
    ranges = [np.arange(aoslice[a, 2], aoslice[a, 3]) for a in active_atoms]
    return np.hstack(ranges).astype(int) if ranges else np.empty(0, dtype=int)
