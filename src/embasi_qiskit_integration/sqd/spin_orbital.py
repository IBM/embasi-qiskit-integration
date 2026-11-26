# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Spin-resolved selected-CI diagonalization for SQD.

The stock ``qiskit_addon_sqd`` path takes one spatial ``h1``/``h2`` pair, so a
spin-dependent embedding Hamiltonian would otherwise be solved with its
spin-averaged one-body operator and alpha-only two-body tensor (ROHF-like).
This module encodes the exact ``(h1a, h1b, h2_aa, h2_ab, h2_bb)`` tensors as a flat
spin-orbital Hamiltonian over ``2*norb`` orbitals and hands it to torch-sci's
spinless engine.  It only changes the representation: the addon's
configuration-recovery loop, subspace sampling, and carryover are unchanged.

Orbitals ``[0, norb)`` are alpha and ``[norb, 2*norb)`` beta.  An addon
``(strs_a, strs_b)`` batch maps to spin-orbital bitmasks
``strs_a | (strs_b << norb)``.  torch-sci therefore sees the determinant space
as a generic open-shell ``N = n_alpha + n_beta`` problem, with no spin symmetry
assumed.

Limitations
-----------
``spin_sq`` projection is not available on torch-sci's spinless path.  The
dense ``(2*norb)^4`` ERI costs 16x one ``n^4`` block (5.3x the
``(aa, ab, bb)`` triple), so large active spaces need a sparse representation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def build_2n_eri(h2_spin: tuple[np.ndarray, np.ndarray, np.ndarray], norb: int) -> np.ndarray:
    """Pack ``(aa, ab, bb)`` into one spin-orbital ERI over ``2*norb`` orbitals."""
    aa, ab, bb = (np.asarray(x, dtype=float) for x in h2_spin)
    n2 = 2 * norb
    eri = np.zeros((n2, n2, n2, n2))
    eri[:norb, :norb, :norb, :norb] = aa
    eri[norb:, norb:, norb:, norb:] = bb
    eri[:norb, :norb, norb:, norb:] = ab
    eri[norb:, norb:, :norb, :norb] = ab.transpose(2, 3, 0, 1)
    return eri


def build_2n_one_body(h1a: np.ndarray, h1b: np.ndarray, norb: int) -> np.ndarray:
    """Pack ``(h1a, h1b)`` as a block-diagonal ``2*norb`` matrix."""
    n2 = 2 * norb
    h1 = np.zeros((n2, n2))
    h1[:norb, :norb] = np.asarray(h1a, dtype=float)
    h1[norb:, norb:] = np.asarray(h1b, dtype=float)
    return h1


def energy_from_spin_blocks(
    h1a: np.ndarray,
    h1b: np.ndarray,
    h2_spin: tuple[np.ndarray, np.ndarray, np.ndarray],
    rdm1s: tuple[np.ndarray, np.ndarray],
    dm2s: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """``<H>`` from spin-resolved RDMs in PySCF's block convention."""
    aa, ab, bb = dm2s
    e1 = np.einsum("pq,qp->", h1a, rdm1s[0]) + np.einsum("pq,qp->", h1b, rdm1s[1])
    e2 = (
        0.5 * np.einsum("pqrs,pqrs->", h2_spin[0], aa)
        + 0.5 * np.einsum("pqrs,pqrs->", h2_spin[2], bb)
        + np.einsum("pqrs,pqrs->", h2_spin[1], ab)
    )
    return float(e1 + e2)


def _validate_spin_hamiltonian(ham: EmbeddedHamiltonian) -> None:
    if not ham.is_spin_dependent:
        raise ValueError(
            "a spin-resolved solver needs a spin-dependent Hamiltonian "
            "(h1a/h1b); this one is spin-restricted"
        )


def _spin_eri_triple(ham: EmbeddedHamiltonian):
    """The exact ``(aa, ab, bb)`` triple, or the spin-free fallback in all slots.

    Mirrors ``FCISolver``: an ``h2_spin=None`` downfold gets ``(h2, h2, h2)``
    and is only exact when both channels share an orbital set.
    """
    if ham.h2_spin is not None:
        return tuple(np.asarray(x, dtype=float) for x in ham.h2_spin)
    h2 = np.asarray(ham.h2, dtype=float)
    return (h2, h2, h2)


def _spin_strings(strs_a, strs_b, norb: int) -> np.ndarray:
    """Spin-orbital bitmasks, alpha bits low and beta bits shifted by ``norb``."""
    a = np.asarray(strs_a)
    b = np.asarray(strs_b)
    if 2 * norb <= 62:
        return (a.astype(np.int64)[None, :] | (b.astype(np.int64)[:, None] << norb)).ravel()
    return np.array([int(aa) | (int(bb) << norb) for bb in b for aa in a], dtype=object)


def _amplitudes_from_flat(flat_strs, amplitudes, strs_a, strs_b, norb: int) -> np.ndarray:
    """Map a flat spin-orbital eigenvector back to spatial ``(na, nb)`` amplitudes."""
    so = _spin_strings(strs_a, strs_b, norb)
    flat = np.asarray(flat_strs)
    amps = np.asarray(amplitudes).ravel()
    if flat.shape[0] != amps.shape[0]:
        raise ValueError(
            f"torch-sci state has {flat.shape[0]} strings but {amps.shape[0]} amplitudes"
        )
    if flat.dtype == object or so.dtype == object:
        pos = {int(s): i for i, s in enumerate(flat)}
        try:
            vals = np.array([amps[pos[int(s)]] for s in so])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"torch-sci dropped determinant {exc.args[0]} from the returned basis"
            ) from exc
    else:
        flat = flat.astype(np.int64, copy=False)
        order = np.argsort(flat)
        fs = flat[order]
        fa = amps[order]
        idx = np.searchsorted(fs, so)
        if np.any(idx >= len(fs)) or not np.array_equal(fs[idx], so):
            raise ValueError(
                "torch-sci returned a basis that is not a superset of the "
                f"requested subspace ({len(so)} determinants)"
            )
        vals = fa[idx]
    return np.ascontiguousarray(vals.reshape(len(strs_b), len(strs_a)).T)


def _spatial_rdms(amplitudes, strs_a, strs_b, norb, nelec):
    """Addon-convention RDMs from spatial CI amplitudes."""
    from qiskit_addon_sqd.fermion import SCIState

    # PySCF's selected_ci C kernels read the CI vector through a raw pointer and
    # do not honour non-contiguous strides; force C order here too.
    amps = np.ascontiguousarray(
        np.asarray(amplitudes, dtype=float).reshape(len(strs_a), len(strs_b))
    )
    norm = float(np.linalg.norm(amps))
    if norm > 0.0:
        amps = amps / norm
    state = SCIState(
        amplitudes=amps,
        ci_strs_a=np.asarray(strs_a, dtype=np.int64),
        ci_strs_b=np.asarray(strs_b, dtype=np.int64),
        norb=norb,
        nelec=nelec,
    )
    rdm1s = tuple(np.asarray(x, dtype=float) for x in state.rdm(rank=1, spin_summed=False))
    dm2s = tuple(np.asarray(x, dtype=float) for x in state.rdm(rank=2, spin_summed=False))
    rdm2 = dm2s[0] + dm2s[2] + dm2s[1] + dm2s[1].transpose(2, 3, 0, 1)
    return state, rdm1s, dm2s, rdm2


def make_torch_spin_orbital_solver(
    ham: EmbeddedHamiltonian,
    *,
    spin_sq: float | None = None,
    torch_sci_kwargs: dict | None = None,
    verbose: bool = False,
):
    """Return an ``sci_solver`` callable using torch-sci's spinless engine.

    torch-sci's flat spin-orbital path is selected by passing a scalar particle
    number.  It sees the same ``2*norb`` Hamiltonian and the addon's CI strings
    mapped to spin-orbital bitmasks.
    """
    _validate_spin_hamiltonian(ham)
    if spin_sq is not None:
        raise NotImplementedError(
            "torch-sci's spinless path has no alpha/beta split, so spin_sq is "
            "unsupported by the engine: pass spin_sq=None."
        )

    norb = int(ham.norb)
    nelec = tuple(int(x) for x in ham.nelec)
    n_particles = int(sum(nelec))
    h1_2n = build_2n_one_body(
        np.asarray(ham.h1a, dtype=float), np.asarray(ham.h1b, dtype=float), norb
    )
    eri_2n = build_2n_eri(_spin_eri_triple(ham), norb)

    # Keep the same macOS OpenMP discipline as the stock torch-sci path.
    from embasi_qiskit_integration._openmp import preload_torch_openmp

    if not preload_torch_openmp():
        import torch

        torch.set_num_threads(1)
    import torch

    kwargs: dict[str, Any] = dict(torch_sci_kwargs or {})
    mode = kwargs.pop("mode", None)
    if mode not in (None, "tpb"):
        raise NotImplementedError(
            "the spin-orbital torch-sci path uses its flat spinless engine; "
            f"torch_sci_mode={mode!r} is not supported"
        )
    kwargs.setdefault("apply_ab_projection", False)
    kwargs.setdefault("compute_full_rdm", False)
    device = kwargs.get("device")
    if device is None and not kwargs.get("devices"):
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        kwargs["device"] = device
    if str(device).startswith("mps"):
        kwargs.setdefault("dtype", torch.float32)
        kwargs.setdefault("davidson_tolerance", 1e-5)
    else:
        kwargs.setdefault("davidson_tolerance", 1e-8)
    kwargs.setdefault("verbose", verbose)

    from torch_sci_solver.qiskit import make_torch_sci_solver

    solver = make_torch_sci_solver(**kwargs)

    def solve_batch(
        ci_strings,
        one_body_tensor=None,
        two_body_tensor=None,
        norb_=None,
        nelec_=None,
        **call_kwargs,
    ):
        from qiskit_addon_sqd.fermion import SCIResult

        if norb_ is not None and int(norb_) != norb:
            raise ValueError(f"norb mismatch: solver built for {norb}, called with {norb_}")
        if nelec_ is not None and tuple(int(x) for x in nelec_) != nelec:
            raise ValueError(
                f"nelec mismatch: solver built for {nelec}, called with {tuple(nelec_)}"
            )
        flat_list = [_spin_strings(sa, sb, norb) for sa, sb in ci_strings]
        # Scalar nelec selects torch-sci's flat spin-orbital dispatch.
        eigen_list = solver(flat_list, h1_2n, eri_2n, 2 * norb, n_particles)
        results = []
        for (strs_a, strs_b), eigen in zip(ci_strings, eigen_list):
            state_result = eigen.states[0]
            if getattr(state_result, "state", "converged") != "converged":
                warnings.warn(
                    f"torch-sci spinless Davidson did not converge for a "
                    f"{len(strs_a)}x{len(strs_b)} subspace (nelec={nelec}); "
                    "the energy and RDMs are the best iterate",
                    RuntimeWarning,
                    stacklevel=3,
                )
            amplitudes = _amplitudes_from_flat(
                state_result.sci_state.ci_strs,
                state_result.sci_state.amplitudes,
                strs_a,
                strs_b,
                norb,
            )
            state, rdm1s, _dm2s, rdm2 = _spatial_rdms(amplitudes, strs_a, strs_b, norb, nelec)
            results.append(
                SCIResult(
                    energy=float(state_result.energy),
                    sci_state=state,
                    orbital_occupancies=(np.diagonal(rdm1s[0]), np.diagonal(rdm1s[1])),
                    rdm1=rdm1s[0] + rdm1s[1],
                    rdm2=rdm2,
                )
            )
        return results

    return solve_batch


__all__ = [
    "build_2n_eri",
    "build_2n_one_body",
    "energy_from_spin_blocks",
    "make_torch_spin_orbital_solver",
]
