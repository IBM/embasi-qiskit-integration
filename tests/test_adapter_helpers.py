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
