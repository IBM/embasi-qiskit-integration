# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy unit tests for adapter helpers that need no EmbASI/PySCF.

These run in the default suite.  They pin down ``_as_ao_matrix``'s contract for
the shape and dtype EmbASI actually hands back on the restricted single-k path:

* EmbASI's SPADE eigensolve returns real, restricted data in a ``complex128``
  dtype with a **zero** imaginary part.  ``_as_ao_matrix`` must drop the
  imaginary part -- but only after asserting it is negligible, so a genuinely
  complex block (multi-k / open shell / upstream bug) fails loudly rather than
  being silently truncated to its real projection.
* A leading spin/k axis of length 1 is squeezed; length > 1 is rejected.

They also pin ``_as_spin_kpoint_array`` -- the inverse used on the density
feedback path -- which must produce something EmbASI indexes as ``[0, 0]``.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.projection_embedding_adapter import (
    _IMAG_TOL,
    ProjectionEmbeddingAdapter,
)

_as_ao_matrix = ProjectionEmbeddingAdapter._as_ao_matrix
_as_spin_kpoint_array = ProjectionEmbeddingAdapter._as_spin_kpoint_array


def test_complex_zero_imag_returns_real_array():
    real = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = real.astype(np.complex128)  # real data carried in a complex dtype
    out = _as_ao_matrix(m)
    assert not np.iscomplexobj(out)
    np.testing.assert_array_equal(out, real)


def test_complex_imag_within_tol_is_dropped():
    # imag just under the tolerance is treated as round-off and discarded.
    real = np.eye(3)
    m = real.astype(np.complex128)
    m[0, 1] = 1.0 + 0.1 * _IMAG_TOL * 1j
    out = _as_ao_matrix(m)
    assert not np.iscomplexobj(out)
    assert out[0, 1] == pytest.approx(1.0)


def test_complex_imag_above_tol_raises():
    m = np.eye(2, dtype=np.complex128)
    m[1, 0] = 0.5 + 10.0 * _IMAG_TOL * 1j  # well above the tolerance
    with pytest.raises(ValueError, match="non-negligible imaginary part"):
        _as_ao_matrix(m)


def test_leading_unit_axis_is_squeezed():
    real = np.arange(9.0).reshape(3, 3)
    m = real[np.newaxis, :, :]  # (1, nao, nao) restricted single-k block
    out = _as_ao_matrix(m)
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out, real)


def test_two_leading_unit_axes_are_squeezed():
    real = np.arange(4.0).reshape(2, 2)
    m = real[np.newaxis, np.newaxis, :, :]  # (nspin=1, nkpt=1, nao, nao)
    out = _as_ao_matrix(m)
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out, real)


def test_leading_axis_gt_one_is_rejected():
    m = np.zeros((2, 3, 3))  # nspin=2 (unrestricted) -- not representable
    with pytest.raises(NotImplementedError, match="open-shell / multi-k"):
        _as_ao_matrix(m)


def test_result_is_contiguous():
    # A transposed (non-contiguous) view must come back C-contiguous.
    m = np.arange(9.0).reshape(3, 3).T
    out = _as_ao_matrix(m)
    assert out.flags["C_CONTIGUOUS"]


def _no_embasi_import(monkeypatch):
    """Force ``import embasi...`` to fail, exercising the ndarray fallback."""
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name.startswith("embasi"):
            raise ImportError("simulated: EmbASI not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_spin_kpoint_array_fallback_is_indexable(monkeypatch):
    """Without EmbASI, the wrapper is a (1, 1, nao, nao) ndarray, [0,0]-indexable.

    The mock outer loop (and real EmbASI) unwrap ``dma_in[0, 0]`` / ``dmb_in[0, 0]``
    to the plain density block, so the fallback must reproduce that indexing exactly.
    """
    _no_embasi_import(monkeypatch)
    dm = np.arange(16.0).reshape(4, 4)
    wrapped = _as_spin_kpoint_array(dm)
    assert isinstance(wrapped, np.ndarray)
    assert wrapped.shape == (1, 1, 4, 4)
    np.testing.assert_array_equal(wrapped[0, 0], dm)


def test_spin_kpoint_array_roundtrips_through_as_ao_matrix(monkeypatch):
    """wrap -> _as_ao_matrix squeezes back to the original (nao, nao) block."""
    _no_embasi_import(monkeypatch)
    dm = np.arange(9.0).reshape(3, 3)
    back = _as_ao_matrix(_as_spin_kpoint_array(dm))
    assert back.shape == (3, 3)
    np.testing.assert_array_equal(back, dm)


# --------------------------------------------------------------------------- #
# Subsystem-A spin sector, from EmbASI's ``A_spin``  (Stage 1)
# --------------------------------------------------------------------------- #
_n_occ_b = ProjectionEmbeddingAdapter._n_occ_b_from_embasi
_spin_unknown = ProjectionEmbeddingAdapter._spin_is_unknown


class _SpinStub:
    """Minimal adapter carrying only what the spin helpers read."""

    def __init__(self, a_spin=...):
        if a_spin is not ...:
            self.p = SimpleNamespace(A_spin=a_spin)
        else:
            self.p = SimpleNamespace()  # EmbASI without spin support


def test_a_spin_gives_the_beta_count():
    """``n_beta = n_alpha - A_spin``: a doublet on 5 alpha orbitals is (5, 4)."""
    assert _n_occ_b(_SpinStub(a_spin=1), 5) == 4
    # A triplet takes two off the beta channel.
    assert _n_occ_b(_SpinStub(a_spin=2), 5) == 3


def test_a_spin_zero_falls_back_to_the_restricted_reading():
    """An unrestricted *singlet* is well-posed and must not be refused.

    ``2S == 0`` means ``n_alpha == n_beta``, which the restricted reading (``n_occ``
    doubly-occupied orbitals) represents exactly -- so ``None`` here is the right
    answer, not a failure.  ``_spin_is_unknown`` must still report ``False``: the
    spin is *known* to be zero, which is what lets the downfold proceed.
    """
    assert _n_occ_b(_SpinStub(a_spin=0), 5) is None
    assert _spin_unknown(_SpinStub(a_spin=0)) is False


def test_missing_a_spin_is_distinguishable_from_a_reported_zero():
    """No ``A_spin`` at all -> ``None``, but flagged unknown so the downfold refuses.

    Both cases return ``None`` from ``_n_occ_b_from_embasi``; only this one is a
    guess at the sector rather than a faithful representation of it.
    """
    assert _n_occ_b(_SpinStub(), 5) is None
    assert _spin_unknown(_SpinStub()) is True


def test_a_spin_none_is_treated_as_absent():
    """EmbASI has no ``__init__`` default for ``A_spin``; a literal None means absent."""
    assert _n_occ_b(_SpinStub(a_spin=None), 5) is None
    assert _spin_unknown(_SpinStub(a_spin=None)) is True


def test_a_spin_without_p_attribute_is_tolerated():
    """A stub adapter built via ``object.__new__`` has no ``p`` at all."""

    class _NoP:
        pass

    assert _n_occ_b(_NoP(), 4) is None
    assert _spin_unknown(_NoP()) is True


@pytest.mark.parametrize("a_spin", [6, -1, 99])
def test_impossible_a_spin_raises_rather_than_guessing(a_spin):
    """A beta count outside ``[0, n_alpha]`` is a partition/MO disagreement.

    Silently clamping (or passing it through) would hand the solver a
    valid-looking wrong spin sector, which no downstream check can catch once the
    active space is built -- so it must raise here.
    """
    with pytest.raises(ValueError, match="outside"):
        _n_occ_b(_SpinStub(a_spin=a_spin), 5)


def test_a_spin_equal_to_n_occ_is_allowed():
    """A fully spin-polarised fragment (every beta emptied) is a boundary, not an error."""
    assert _n_occ_b(_SpinStub(a_spin=5), 5) == 0


def test_state_fingerprint_records_subsystem_a_spin():
    """A doublet snapshot must not restore into a singlet adapter.

    Neither ``unrestricted`` (a bool) nor ``mol.spin`` (the *supersystem*'s) captures
    what the SPADE partition assigned to subsystem A, so without ``a_spin`` in the
    fingerprint the two are indistinguishable: same basis, same geometry, same array
    shapes, different meaning for every exported matrix.  ``restore_state`` compares
    the fingerprint as a whole, so it is enough that the value differs.
    """

    class _Ints:
        mol = None
        mf = None
        _density_fit = False

        def overlap(self):
            return np.eye(3)

    def _fp(a_spin):
        adapter = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
        adapter.ints = _Ints()
        adapter.unrestricted = True
        adapter.mu = 1.0e6
        adapter.p = SimpleNamespace(projection="level-shift", A_spin=a_spin)
        return adapter.state_fingerprint()

    doublet, singlet = _fp(1), _fp(0)
    assert doublet["a_spin"] == 1
    assert singlet["a_spin"] == 0
    assert doublet != singlet


def test_state_fingerprint_a_spin_is_none_without_embasi_support():
    """A pre-spin EmbASI records ``None`` rather than raising."""

    class _Ints:
        mol = None
        mf = None
        _density_fit = False

        def overlap(self):
            return np.eye(2)

    adapter = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    adapter.ints = _Ints()
    adapter.unrestricted = False
    adapter.mu = 1.0e6
    adapter.p = SimpleNamespace(projection="level-shift")

    assert adapter.state_fingerprint()["a_spin"] is None


# --------------------------------------------------------------------------- #
# ragged per-spin MO blocks  (live open shell)
# --------------------------------------------------------------------------- #
class _RaggedSpinKpointArray:
    """Stand-in for EmbASI's ``SpinKpointArray`` with unequal per-spin widths.

    Measured against real EmbASI on an OH radical in sto-3g: subsystem A comes back
    with ``(13, 5)`` alpha and ``(13, 4)`` beta blocks, because SPADE slices each spin
    channel at its own occupied count.  Indexable by ``(ispin, ikpt)``; deliberately
    NOT convertible with ``np.asarray``, which is the whole point.
    """

    n_spins = 2
    n_kpoints = 1

    def __init__(self, alpha, beta):
        self._d = {(0, 0): alpha, (1, 0): beta}

    def __getitem__(self, key):
        return self._d[key]

    def __array__(self, *args, **kwargs):
        raise ValueError(
            "setting an array element with a sequence. The requested array has an "
            "inhomogeneous shape after 2 dimensions."
        )


def test_ragged_per_spin_mo_blocks_reduce_to_alpha():
    """A live open shell must not crash on ``np.asarray`` of unequal spin widths.

    ``_as_ao_by_mo`` used to call ``np.asarray(c)`` first, which raises
    "inhomogeneous shape" on EmbASI's ragged pair -- so ``mo_a_ll`` could not be read
    at all on a doublet and ``build_orbitals`` died before reaching the spin-sector
    logic.  ``_spin_block`` now takes the alpha channel *before* the conversion.
    """
    nao, n_alpha, n_beta = 6, 4, 3
    rng = np.random.default_rng(3)
    # S-orthonormal columns so _as_ao_by_mo's C^T S C == 1 check passes.
    alpha = np.linalg.qr(rng.standard_normal((nao, nao)))[0][:, :n_alpha]
    beta = np.linalg.qr(rng.standard_normal((nao, nao)))[0][:, :n_beta]

    adapter = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    adapter.unrestricted = True
    adapter._s = np.eye(nao)

    out = adapter._as_ao_by_mo(_RaggedSpinKpointArray(alpha, beta))
    assert out.shape == (nao, n_alpha), "must reduce to the alpha channel"
    np.testing.assert_allclose(out, alpha)


def test_spin_block_passes_through_when_restricted():
    """A restricted adapter must not index the container at all."""
    adapter = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    adapter.unrestricted = False
    sentinel = _RaggedSpinKpointArray(np.zeros((3, 2)), np.zeros((3, 1)))
    assert adapter._spin_block(sentinel) is sentinel


def test_spin_block_accepts_either_spin_count_attribute():
    """``SpinKpointArray`` stores ``n_spins``; guard against an upstream rename.

    Its constructor keyword is ``n_spin`` (singular) while the stored attribute is
    ``n_spins`` (plural) -- reading the wrong one silently reinstates the
    ragged-array crash, which is how this bug survived its first fix.
    """
    alpha, beta = np.zeros((4, 2)), np.zeros((4, 1))

    adapter = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    adapter.unrestricted = True

    plural = _RaggedSpinKpointArray(alpha, beta)
    assert adapter._spin_block(plural) is alpha

    singular = _RaggedSpinKpointArray(alpha, beta)
    del type(singular).n_spins  # simulate the singular-only spelling
    try:
        singular.n_spin = 2
        assert adapter._spin_block(singular) is alpha
    finally:
        type(singular).n_spins = 2
