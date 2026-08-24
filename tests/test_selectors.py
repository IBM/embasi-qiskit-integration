# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for :mod:`embasi_qiskit_integration.selectors`.

Pure numpy -- no EmbASI, no PySCF -- so these run in the default suite.  The
selector's contract is: given canonical subsystem-A orbitals ``(coeff, energy,
n_occ)`` and the AO overlap ``S``, return the active column indices as
``[0, n_occ)`` (all occupied A) followed by the virtuals whose *fragment
population* survives the gap/cap cut.

We build synthetic orbitals directly, so the fragment population of each virtual
is known by construction and the cut is checked against a value we set, not a
value the physics happens to produce.
"""

from __future__ import annotations

import numpy as np

from embasi_qiskit_integration.selectors import (
    _gap_cut,
    concentric_selector,
    fragment_ao_indices,
    spade_virtual_selector,
)


def _orthonormal(n: int, seed: int = 0) -> np.ndarray:
    """A random orthonormal (nao, nao) matrix via QR -- columns are MO coeffs."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def test_orthonormal_basis_ranks_virtuals_by_fragment_population():
    # S = I so fragment population of virtual v is sum_{mu in frag} C[mu, v]^2:
    # exactly the norm of the virtual restricted to the fragment AOs.  Build the
    # coeff matrix so that population is a clean, strictly-decreasing ladder.
    nao = 6
    n_occ = 2
    frag = np.array([0, 1])  # two fragment AOs

    coeff = np.zeros((nao, nao))
    # Occupied block: arbitrary but orthonormal within their own 2D subspace.
    coeff[2, 0] = 1.0
    coeff[3, 1] = 1.0
    # Virtuals (columns 2..5): fragment weight = C[0,v]^2 + C[1,v]^2 on AO 0/1,
    # remainder on non-fragment AOs so each column stays unit norm.
    #  v2: w=1.00   v3: w=0.64   v4: w=0.04   v5: w=0.00
    ladder = [1.00, 0.64, 0.04, 0.00]
    filler_ao = [4, 5, 4, 5]
    for j, (w, fao) in enumerate(zip(ladder, filler_ao)):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        coeff[filler_ao[j], v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    sel = concentric_selector(s, frag, gap_tol=0.1)
    active = sel(coeff, energy, n_occ)

    # Occupied block is always kept, in order.
    assert list(active[:n_occ]) == [0, 1]
    # The ladder is 1.00, 0.64, 0.04, 0.00, so gaps are 0.36, 0.60, 0.04.  The
    # *largest* gap >= gap_tol is after index 1 (0.60), so the tight shell is
    # v2 and v3 (the first-gap rule would have kept just v2).
    assert set(active[n_occ:]) == {2, 3}


def test_gap_cut_keeps_the_tight_shell():
    # A clear two-shell structure: two strongly-coupled virtuals (~0.9) then a
    # long weak tail (~0.05).  The big gap sits between the shells; the selector
    # must keep exactly the strong shell.
    nao = 8
    n_occ = 2
    frag = np.array([0])
    coeff = np.zeros((nao, nao))
    coeff[6, 0] = 1.0
    coeff[7, 1] = 1.0
    weights = [0.92, 0.88, 0.06, 0.05, 0.04, 0.03]  # 2 strong, 4 weak
    for j, w in enumerate(weights):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        # spread the remainder onto a distinct non-fragment AO per column
        coeff[2 + (j % 4), v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    sel = concentric_selector(s, frag, gap_tol=0.1)
    active = sel(coeff, energy, n_occ)
    kept_virt = set(active[n_occ:])
    # Only the two strong virtuals clear the gap.
    assert kept_virt == {2, 3}


def test_max_virtual_caps_after_the_gap():
    nao = 7
    n_occ = 1
    frag = np.array([0])
    coeff = np.zeros((nao, nao))
    coeff[6, 0] = 1.0
    weights = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # smooth ramp, no clear gap
    for j, w in enumerate(weights):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        coeff[1 + (j % 5), v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    # gap_tol high enough that nothing cuts on the gap; cap does the work.
    sel = concentric_selector(s, frag, gap_tol=0.5, max_virtual=2)
    active = sel(coeff, energy, n_occ)
    assert active[0] == 0  # occupied kept
    assert len(active) - n_occ == 2  # capped at 2 virtuals
    # the two kept are the two strongest-coupled (columns 1 and 2).
    assert set(active[n_occ:]) == {1, 2}


def test_no_significant_gap_keeps_everything():
    # A smooth ramp with no gap >= gap_tol: the selector must refuse to truncate
    # on noise and keep every virtual.
    nao = 6
    n_occ = 1
    frag = np.array([0])
    coeff = np.zeros((nao, nao))
    coeff[5, 0] = 1.0
    weights = [0.60, 0.58, 0.56, 0.54, 0.52]
    for j, w in enumerate(weights):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        coeff[1 + (j % 4), v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    sel = concentric_selector(s, frag, gap_tol=0.1)  # gaps are all 0.02
    active = sel(coeff, energy, n_occ)
    assert len(active) == nao  # all occ + all virt
    assert set(active[n_occ:]) == set(range(n_occ, nao))


def test_min_virtual_floor():
    # Even when the first gap would keep just one virtual, min_virtual keeps more.
    nao = 6
    n_occ = 1
    frag = np.array([0])
    coeff = np.zeros((nao, nao))
    coeff[5, 0] = 1.0
    weights = [0.9, 0.2, 0.15, 0.1, 0.05]  # big gap after the first virtual
    for j, w in enumerate(weights):
        v = n_occ + j
        coeff[0, v] = np.sqrt(w)
        coeff[1 + (j % 4), v] = np.sqrt(1.0 - w)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    sel = concentric_selector(s, frag, gap_tol=0.1, min_virtual=3)
    active = sel(coeff, energy, n_occ)
    assert len(active) - n_occ == 3
    # the three strongest by fragment weight (columns 1, 2, 3).
    assert set(active[n_occ:]) == {1, 2, 3}


def test_no_virtuals_returns_occupied_only():
    nao = 3
    n_occ = 3  # fully occupied, no virtuals
    frag = np.array([0])
    coeff = _orthonormal(nao)
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)
    sel = concentric_selector(s, frag)
    active = sel(coeff, energy, n_occ)
    assert list(active) == [0, 1, 2]


def test_nonorthogonal_overlap_uses_S_weighted_population():
    # With a non-identity S the population is w_v = sum_{mu in frag}(S C)_{mu v}
    # C_{mu v}, not sum C^2.  Verify the selector uses the S-weighted form by
    # comparing against a hand-computed ranking.
    nao = 4
    n_occ = 1
    frag = np.array([0, 1])
    rng = np.random.default_rng(3)
    a = rng.standard_normal((nao, nao))
    s = a @ a.T + nao * np.eye(nao)  # SPD overlap
    # S-orthonormalize a random set of columns: C^T S C = I.
    x = rng.standard_normal((nao, nao))
    # Lowdin-style: C = L^{-T} Q  with S = L L^T is fussy; instead use eig.
    w, u = np.linalg.eigh(s)
    s_inv_half = u @ np.diag(1.0 / np.sqrt(w)) @ u.T
    q, _ = np.linalg.qr(x)
    coeff = s_inv_half @ q  # C^T S C = Q^T Q = I
    energy = np.arange(nao, dtype=float)

    # Hand-compute the fragment population ranking (same form the selector uses).
    c_virt = coeff[:, n_occ:]
    sc = s @ c_virt
    pop = np.einsum("mv,mv->v", sc[frag, :], c_virt[frag, :])
    expected_order = np.argsort(-pop)

    sel = concentric_selector(s, frag, gap_tol=1e9, max_virtual=1)  # keep top-1
    active = sel(coeff, energy, n_occ)
    kept = active[n_occ:]
    assert kept.tolist() == [n_occ + int(expected_order[0])]


def test_fragment_ao_indices_from_mock_mol():
    # fragment_ao_indices only needs mol.aoslice_by_atom(); fake it.
    class _Mol:
        def aoslice_by_atom(self):
            # columns 2 and 3 are (ao_start, ao_end) per atom.
            return np.array(
                [
                    [0, 0, 0, 2],  # atom 0 -> AOs 0,1
                    [0, 0, 2, 5],  # atom 1 -> AOs 2,3,4
                    [0, 0, 5, 6],  # atom 2 -> AO 5
                ]
            )

    idx = fragment_ao_indices(_Mol(), [0, 2])
    assert idx.tolist() == [0, 1, 5]

    empty = fragment_ao_indices(_Mol(), [])
    assert empty.size == 0


# ----- n_frozen_occ must survive a selector ---------------------------------- #


def _stub_adapter(nao: int, n_occ: int, seed: int = 3):
    """A ``ProjectionEmbeddingAdapter`` with only what ``build_orbitals`` reads.

    ``build_orbitals`` needs the Fock/overlap matrices, the level-shift floor and
    ``n_occ``; everything else on the adapter belongs to the live EmbASI embedding.
    Subclassing to override the read-only properties keeps this independent of
    EmbASI, which the full fixtures require.
    """
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nao, nao))
    fock = a + a.T
    overlap = np.eye(nao)

    class _Stub(ProjectionEmbeddingAdapter):
        _s_arr = property(lambda self: overlap)
        _fock_arr = property(lambda self: fock)
        mo_a_ll = property(lambda self: np.zeros((nao, n_occ)))

        def _validate_span(self, coeff):  # needs the real embedding
            return None

    adapter = object.__new__(_Stub)
    adapter._floor = np.inf  # keep every orbital
    return adapter, overlap


def test_n_frozen_occ_is_honoured_when_a_selector_is_set():
    """A selector must not silently discard the occupied freeze.

    Regression: ``n_frozen_occ`` was validated and then dropped whenever a selector
    was passed, because the selector branch replaced the ``arange(n_frozen_occ, ...)``
    that applied it. Selectors choose *virtuals* and return the occupied block
    untouched by contract, so the freeze has to be applied after them.

    The symptom was quiet: ``e_core`` and ``nelec`` stay mutually consistent, so
    energies looked fine while the active space — and hence the qubit count the flag
    exists to bound — was larger than requested.
    """
    adapter, overlap = _stub_adapter(nao=8, n_occ=3)
    selector = concentric_selector(overlap, np.arange(4), max_virtual=2)

    unfrozen = adapter.build_orbitals(n_frozen_occ=0, selector=selector, restrict_to_a=False)
    frozen = adapter.build_orbitals(n_frozen_occ=2, selector=selector, restrict_to_a=False)

    # The freeze removes exactly the two lowest occupied orbitals...
    assert unfrozen.inactive.tolist() == []
    assert frozen.inactive.tolist() == [0, 1]
    # ...which is 4 fewer active electrons and 2 fewer active orbitals.
    assert frozen.n_active_electrons == unfrozen.n_active_electrons - 4
    assert frozen.n_active_orbitals == unfrozen.n_active_orbitals - 2
    # The selector's virtual choice is untouched by the freeze.
    assert [i for i in frozen.active if i >= 3] == [i for i in unfrozen.active if i >= 3]


def test_freeze_that_empties_the_active_space_raises():
    """Freezing everything the selector kept must fail loudly, not yield nothing."""
    import pytest

    adapter, _ = _stub_adapter(nao=8, n_occ=3)

    with pytest.raises(ValueError, match="froze every orbital"):
        adapter.build_orbitals(
            n_frozen_occ=2,
            selector=lambda coeff, energy, n_occ: np.array([0, 1]),
            restrict_to_a=False,
        )


# ----- SPADE virtual-space localiser ----------------------------------------- #


def _apply_localizer(loc, coeff, n_occ):
    """Mimic ``build_orbitals``: rotate + gap-cut, returning (kept_count, c_loc)."""
    c_loc, sigma2 = loc(coeff, n_occ)
    keep = _gap_cut(
        sigma2,
        gap_tol=loc.gap_tol,
        max_virtual=loc.max_virtual,
        min_virtual=loc.min_virtual,
    )
    return keep, c_loc, sigma2


def test_spade_sigma2_reduces_to_fragment_population_at_identity_overlap():
    """At ``S = I`` the σ² of each rotated virtual is its own fragment population.

    The Löwdin step is the identity when ``S = I``, so σ² collapses to the plain
    ``Σ_{μ∈frag} C[μ,v]²`` the Mulliken cut scores -- but now of the *rotated*
    virtuals.  Verified directly against the rotated coefficients.
    """
    nao, n_occ = 6, 2
    frag = np.array([0, 1])
    coeff = _orthonormal(nao, seed=1)

    loc = spade_virtual_selector(np.eye(nao), frag, gap_tol=0.1)
    c_loc, sigma2 = loc(coeff, n_occ)

    rotated_pop = (c_loc[frag, n_occ:] ** 2).sum(axis=0)
    assert np.allclose(sigma2, rotated_pop)
    # σ² is a probability weight: within [0, 1] and sorted descending.
    assert np.all(sigma2 >= -1e-12) and np.all(sigma2 <= 1 + 1e-12)
    assert np.all(np.diff(sigma2) <= 1e-12)


def test_spade_rotation_preserves_orthonormality_and_span():
    """A within-A virtual rotation keeps S-orthonormality and the virtual span.

    These are the two invariants the downfold relies on (the active columns are an
    opaque S-orthonormal spanning set, and P_B stays invisible because span is
    unchanged).  A non-identity SPD overlap makes the Löwdin step non-trivial, so
    this also exercises the ``S != I`` path.
    """
    nao, n_occ = 7, 3
    frag = np.array([0, 1, 2])
    # An SPD overlap that is not the identity: S = I + small symmetric perturbation.
    rng = np.random.default_rng(11)
    b = rng.normal(size=(nao, nao)) * 0.1
    s = np.eye(nao) + b @ b.T
    # S-orthonormal coefficients: C = S^{-1/2} Q for orthonormal Q (so CᵀSC = I).
    w, u = np.linalg.eigh(s)
    s_minus_half = (u / np.sqrt(w)) @ u.T
    q = _orthonormal(nao, seed=2)
    coeff = s_minus_half @ q
    assert np.allclose(coeff.T @ s @ coeff, np.eye(nao), atol=1e-10)

    loc = spade_virtual_selector(s, frag, gap_tol=0.1)
    c_loc, sigma2 = loc(coeff, n_occ)

    virt = c_loc[:, n_occ:]
    # S-orthonormality of the rotated virtuals.
    assert np.allclose(virt.T @ s @ virt, np.eye(nao - n_occ), atol=1e-10)
    # Occupied block untouched.
    assert np.allclose(c_loc[:, :n_occ], coeff[:, :n_occ])
    # Span of the virtual block unchanged (all principal cosines with the original
    # canonical virtual block are 1) -- so no P_B leak is introduced.
    cos = np.linalg.svd(coeff[:, n_occ:].T @ s @ virt, compute_uv=False)
    assert np.allclose(cos, 1.0, atol=1e-10)
    # σ² ∈ [0, 1], descending.
    assert np.all(sigma2 >= -1e-12) and np.all(sigma2 <= 1 + 1e-12)


def test_spade_recovers_local_shell_the_mulliken_cut_misses():
    """The decisive case: delocalised canonical virtuals hide a local shell.

    Build a 2-dimensional fragment-*local* virtual shell (living entirely on the
    fragment AOs) plus two environment virtuals (zero fragment population), then mix
    all four by a random orthogonal rotation.  The resulting *canonical* virtuals
    each carry a similar, jittery fragment population -- the Mulliken cut sees a flat
    spectrum, finds its largest gap in the wrong place, and cannot isolate the true
    2-dim shell.  SPADE unmixes the rotation: σ² is exactly ``[1, 1, 0, 0]``, so the
    gap cut keeps precisely the two local virtuals, and the kept rotated block spans
    the known local subspace exactly.  This is the improvement the localiser exists
    for.
    """
    nao, n_occ = 6, 2
    frag = np.array([0, 1, 2])  # local shell (dim 2) lives on AO 0, 1

    c_virt = np.zeros((nao, 4))
    c_virt[0, 0] = 1.0
    c_virt[1, 1] = 1.0  # local shell: full fragment population
    c_virt[4, 2] = 1.0
    c_virt[5, 3] = 1.0  # environment: zero fragment population
    rot, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((4, 4)))
    c_virt_canonical = c_virt @ rot  # delocalise: flatten the fragment populations

    coeff = np.zeros((nao, nao))
    coeff[2, 0] = 1.0
    coeff[3, 1] = 1.0  # occupied block
    coeff[:, n_occ:] = c_virt_canonical
    s = np.eye(nao)
    energy = np.arange(nao, dtype=float)

    # The Mulliken cut on the canonical virtuals does NOT isolate the 2-dim shell.
    mulliken = concentric_selector(s, frag, gap_tol=0.1)
    kept_mulliken = [int(i - n_occ) for i in mulliken(coeff, energy, n_occ)[n_occ:]]
    assert len(kept_mulliken) != 2, (
        f"the Mulliken-on-canonical cut kept {len(kept_mulliken)} virtuals -- the "
        "delocalised construction was meant to defeat it (kept != 2)"
    )

    # SPADE recovers exactly the 2-dim local shell.
    loc = spade_virtual_selector(s, frag, gap_tol=0.1)
    keep, c_loc, sigma2 = _apply_localizer(loc, coeff, n_occ)
    assert keep == 2, f"SPADE kept {keep} virtuals, expected the 2-dim local shell"
    assert np.allclose(np.sort(sigma2)[::-1][:2], 1.0, atol=1e-9)
    assert np.allclose(sigma2[2:], 0.0, atol=1e-9)

    # The kept rotated virtuals span the true local subspace (AO 0, 1) exactly.
    kept_block = c_loc[:, n_occ : n_occ + keep]
    local_true = np.zeros((nao, 2))
    local_true[0, 0] = 1.0
    local_true[1, 1] = 1.0
    cos = np.linalg.svd(local_true.T @ kept_block, compute_uv=False)
    assert np.allclose(cos, 1.0, atol=1e-9), (
        f"kept SPADE virtuals do not span the local shell (principal cosines {cos})"
    )


def test_spade_no_gap_keeps_every_virtual():
    """A flat σ² spectrum (no gap over tolerance) refuses to truncate.

    Same safe-degrade contract as the Mulliken cut: when no σ² gap clears the
    tolerance, keep every virtual rather than slice on noise.
    """
    nao, n_occ = 6, 2
    frag = np.array([0, 1])
    # Four virtuals sharing the fragment evenly -> σ² near-degenerate, no clean gap.
    c_virt = np.zeros((nao, 4))
    for j in range(4):
        c_virt[0, j] = 0.5
        c_virt[1, j] = 0.5
        c_virt[2 + j, j] = np.sqrt(0.5)
    # Orthonormalise so the block is a legitimate S=I virtual set.
    q, _ = np.linalg.qr(c_virt)
    coeff = np.zeros((nao, nao))
    coeff[2, 0] = 1.0  # occupied (overwritten below by orthonormal fill)
    coeff = _orthonormal(nao, seed=5)
    coeff[:, n_occ:] = q[:, : nao - n_occ]

    loc = spade_virtual_selector(np.eye(nao), frag, gap_tol=0.9)  # huge tol -> no gap
    keep, _c_loc, _sigma2 = _apply_localizer(loc, coeff, n_occ)
    assert keep == nao - n_occ  # kept everything


def test_spade_localizer_wired_through_build_orbitals():
    """``build_orbitals(virtual_localizer=...)`` rotates, cuts, and NaNs the eps.

    End-to-end through the adapter hook on a stub adapter (no EmbASI): the active
    space is occupied + the kept SPADE shell, rotated-virtual energies are NaN, and
    the active virtual columns are S-orthonormal.
    """
    nao, n_occ = 6, 2
    frag = np.array([0, 1, 2])
    adapter, overlap = _stub_adapter(nao=nao, n_occ=n_occ, seed=4)

    loc = spade_virtual_selector(overlap, frag, gap_tol=1e-4)
    orb = adapter.build_orbitals(virtual_localizer=loc, restrict_to_a=False)

    # Occupied block always kept and leading.
    assert list(orb.active[:n_occ]) == list(range(n_occ))
    # Rotated virtuals carry NaN energies (no meaningful Fock eigenvalue).
    assert np.all(np.isnan(orb.energy[n_occ:]))
    assert np.all(np.isfinite(orb.energy[:n_occ]))
    # Active virtual columns are S-orthonormal against the (identity) overlap.
    c_act_virt = orb.coeff[:, orb.active[orb.active >= n_occ]]
    gram = c_act_virt.T @ overlap @ c_act_virt
    assert np.allclose(gram, np.eye(c_act_virt.shape[1]), atol=1e-10)


def test_build_orbitals_rejects_selector_and_localizer_together():
    """Passing both a selector and a localiser is a hard error, not a silent pick."""
    import pytest

    nao, n_occ = 6, 2
    adapter, overlap = _stub_adapter(nao=nao, n_occ=n_occ)
    sel = concentric_selector(overlap, np.array([0, 1]))
    loc = spade_virtual_selector(overlap, np.array([0, 1]))

    with pytest.raises(ValueError, match="not both"):
        adapter.build_orbitals(selector=sel, virtual_localizer=loc, restrict_to_a=False)
