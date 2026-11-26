# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The WF-in-DFT protocol of the reference projection-embedding implementation.

Rossmannek et al., J. Phys. Chem. Lett. 2023 (arXiv:2302.03052), code at
github.com/mrossinek/qiskit-nature-projection-embedding.  It differs from this
package's default WF-in-DFT energy (``ProjectionEmbeddingAdapter.projection_energy``)
in two places, and the two give measurably different curves for a KS low level:

* **Variational SCF-in-SCF.**  Subsystem A's HF density minimizes

      E[γ_A] = E_HF[γ_A] + E_LL[γ_A + γ_B] - E_LL[γ_A] + E_nuc

  with the embedding potential ``F_LL[γ_A + γ_B] - F_LL[γ_A]`` rebuilt from the
  current ``γ_A`` every iteration.  The default path instead keeps ``v_emb`` frozen at
  the low-level density and adds the first-order ``tr[(γ_HL - γ_LL) v_emb]``, which
  grows large at stretched bonds.
* **Correlation on top.**  The total is ``E[γ_A] + (E_solver - E_ref)``: the active-space
  correlation in the canonical orbitals of the converged embedded Fock matrix, the
  active virtuals being the lowest canonical concentric-localization virtuals.

For an HF low level both reduce to the default path (the SPADE density of A is
already self-consistent).  :func:`reference_wf_in_dft_hamiltonian` returns the
active-space Hamiltonian with ``e_core`` chosen so that ``E_solver + e_core`` is the
protocol's total, so FCI and SQD both run through it unchanged.  Single-shot only:
the protocol has no density feedback.

Stabilization
-------------
The fixed-point map evaluated by :func:`variational_hf_in_dft` is ``γ_A -> G[γ_A]``,
the occupied projector of the embedded Fock built at ``γ_A``.  Undamped iteration of
``G`` is a faithful reproduction of the paper's reference code, but it is not a
contractive SCF: on a file-driven backend (SIESTA through ASI) a single evaluation
costs a full external run and carries grid/NAO integration noise, and PySCF's DIIS
can cycle on these Fock matrices instead of converging.  On butyronitrile geometry
08 the legacy loop settles into a three-density cycle (the SIESTA evaluation seeded
with the same density reproduces its energy to all printed digits, while DIIS keeps
proposing a different Fock), so no number of extra evaluations reaches the density
criterion.

``stabilization="auto"`` (the default) keeps the legacy path bit-for-bit for cheap,
in-process backends, and switches to safeguarded density mixing + Anderson/DIIS for a
backend that declares expensive evaluations (``cheap_evaluations=False``, as the
callback backend does) or as soon as the legacy path is observed to cycle, diverge
or stagnate.  A *single* residual spike is not a divergence: DIIS's first
extrapolation normally overshoots (0.35 -> 1.43 on butyronitrile 2.5 A) and recovers
on the next step, so the fallback needs two consecutive spikes.  ``"legacy"``
reproduces the old behaviour exactly, ``"mixing"`` and
``"anderson"`` start stabilized, and ``"none"`` is the undamped, no-DIIS map.  Every
evaluation of one density is cached for the duration of the SCF, and the low-level
potential and energy at a density are requested together, so a stabilized iteration
costs two ASI runs instead of the legacy path's three or four.  For the same reason
``converge_on="auto"`` (also the default) uses the density criterion alone on a
file-driven backend, whose one-step energy floor is above the energy threshold.

The variational energy is shifted by ``-ints.high_level_reference_energy(A_LL)`` when
the backend exposes it (SIESTA).  That same-density offset aligns the bare high-level
Hamiltonian's zero with the low-level A-layer total at *one* geometry; it is geometry
dependent (measured 5.76, 8.01 and 8.71 Ha at butyronitrile 0.8, 1.3 and 2.5 A in SZ)
and it does not put the reference total on the single-shot path's absolute reference
(a ~27 Ha pseudopotential gap remains at 0.8 A).  Relative energies along a scan must
therefore use the raw value, which is exposed as
:attr:`ReferenceProtocolInfo.energy_offset`.  PySCF exposes no such method and is
unchanged.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, cast

import numpy as np
import scipy.linalg as sla

from embasi_qiskit_integration.backends.base import (
    CAP_ARBITRARY_DM_VEFF,
    CAP_ENERGY_ELEC,
    CAP_ERI,
    CAP_EXACT_EXCHANGE,
    CAP_HCORE,
    require_capability,
)
from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.selectors import concentric_localization_selector

logger = logging.getLogger(__name__)

Stabilization = Literal["auto", "legacy", "mixing", "anderson", "none"]

#: Density-matrix elements are compared with this floor when choosing a fallback.
_PSD_FLOOR = 1e-10


@dataclass(frozen=True)
class SCFIteration:
    """One accepted SCF-in-SCF step, for convergence diagnostics.

    Attributes:
        iteration: 1-based iteration count.
        energy: ``E[γ_A]`` at the accepted density (Ha).
        delta_energy: energy change from the previous accepted density (Ha).
        dDmax: ``max |γ_A(new) - γ_A(old)|`` of the accepted step (the density
            criterion used by the reference implementation).
        residual: ``max |G[γ_A] - γ_A|`` at the accepted density -- the remaining
            fixed-point error, which is what actually has to vanish.
        step_norm: Frobenius norm of the accepted density step.
        mode: ``"legacy"``, ``"none"``, ``"mixing"`` or ``"anderson"``.
        mix_alpha: damping used by the step (``None`` for an undamped legacy/DIIS step).
        converged: whether this step satisfied the configured convergence criteria
            (see ``converge_on``; by default the paper's energy+density criterion for
            an in-process backend, density alone for a file-driven one).
    """

    iteration: int
    energy: float
    delta_energy: float
    dDmax: float
    residual: float
    step_norm: float
    mode: str
    mix_alpha: float | None = None
    converged: bool = False


@dataclass(frozen=True)
class ReferenceProtocolInfo:
    """What the protocol converged to, for the run log.

    ``e_variational`` is on the backend's total-energy reference (the ``energy_offset``
    below is already included).  That offset is a *same-density* quantity and therefore
    geometry dependent on a pseudopotential code, so relative energies along a scan
    must use ``e_variational - energy_offset`` (the raw variational energy) and not the
    shifted value.
    """

    e_variational: float  # E[γ_A] at convergence, on the backend's total-energy reference (Ha)
    e_reference: float  # active-space HF-determinant energy, E_ref + e_core = E[γ_A]
    n_iterations: int
    n_virtual_kept: int  # concentric-localization virtuals before the n_virtual cap
    active_orbital_energies: tuple[float, ...] = ()  # diag of the embedded Fock (Ha)
    converged: bool = True
    scf_history: tuple[SCFIteration, ...] = ()
    #: Constant added to every reported energy: ``-high_level_reference_energy(A_LL)``.
    #: Subtract it from ``e_variational`` for the raw (one consistent reference)
    #: variational energy.  Zero for a backend without a same-density reference.
    energy_offset: float = 0.0


class SCFConvergenceError(RuntimeError):
    """The variational HF-in-DFT SCF did not converge.

    A :class:`RuntimeError` subclass so existing callers that only catch
    ``RuntimeError`` keep working.  The diagnostic fields let a caller inspect the
    last iterate instead of silently continuing from a non-converged density.
    """

    def __init__(
        self,
        message: str,
        *,
        history: tuple[SCFIteration, ...] = (),
        last_density: np.ndarray | None = None,
        last_fock: np.ndarray | None = None,
        last_energy: float | None = None,
    ):
        super().__init__(message)
        self.history = history
        self.last_density = last_density
        self.last_fock = last_fock
        self.last_energy = last_energy


@dataclass
class _SCFPoint:
    """One evaluated density and everything a fixed-point step needs from it."""

    dm: np.ndarray
    fock: np.ndarray  # raw embedded Fock at dm (DIIS/Anderson applied separately)
    response: np.ndarray  # occupied projector of fock: G[dm]
    residual: np.ndarray  # G[dm] - dm
    energy: float
    residual_norm: float


def _max_abs(matrix: np.ndarray) -> float:
    return float(np.abs(matrix).max()) if matrix.size else 0.0


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _density_key(density: np.ndarray) -> bytes:
    return np.ascontiguousarray(density, dtype=float).tobytes()


def _external_evaluations(ints: Any) -> bool:
    """Whether one ``veff``/energy request is a full external QM-code run.

    ``CallbackIntegrals`` evaluates a file-driven code (SIESTA, FHI-aims) and
    declares ``cheap_evaluations=False``; it currently stores that flag privately as
    ``_cheap``.  Read the public spelling first so a future backend can expose it
    without touching this module.  ``PySCFIntegrals`` has neither and stays on the
    legacy path, which preserves the paper's PySCF trajectory bit-for-bit.
    """
    cheap = getattr(ints, "cheap_evaluations", None)
    if cheap is None:
        cheap = getattr(ints, "_cheap", None)
    return cheap is False


def _resolve_converge_on(
    ints: Any,
    converge_on: Literal["auto", "energy_and_density", "density"],
) -> Literal["energy_and_density", "density"]:
    """Resolve ``"auto"`` to the criterion the backend can actually satisfy.

    An in-process backend keeps the paper's energy+density criterion.  A file-driven
    one (``cheap_evaluations=False``) gets the density criterion alone: its one-step
    A-only energy carries a numerical floor above ``conv_tol``, so the energy-change
    test never passes and the SCF would run to ``max_iter`` despite a converged
    density.
    """
    if converge_on != "auto":
        return converge_on
    return "density" if _external_evaluations(ints) else "energy_and_density"


def _external_density_tol(ints: Any) -> float | None:
    """The practical density threshold for a file-driven backend, or ``None``.

    The ASI one-step density residual reaches a floor above the in-process 1e-6 at
    stretched geometries; 1e-4 Ha in the fixed-point residual is still far below the
    energy resolution of the cross-code comparison.
    """
    return 1e-4 if _external_evaluations(ints) else None


def _commutator(fock: np.ndarray, dm: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    return fock @ dm @ overlap - overlap @ dm @ fock


def _anderson_candidate(
    points: list[_SCFPoint],
    alpha: float,
    *,
    regularization: float = 1e-6,
    max_coefficient: float = 10.0,
) -> tuple[np.ndarray, float] | None:
    """Damped type-II Anderson/Pulay step from the last fixed-point residuals.

    Minimizes ``||sum_i c_i r_i||`` subject to ``sum_i c_i = 1`` (regularized for
    linearly dependent residual histories), applies the optimal linear combination
    to the density iterates and takes a damped residual correction.  Returns the
    candidate and its predicted ``max |residual|``, or ``None`` when the subspace
    system is singular or the coefficients are unusable.  The caller accepts the
    candidate only when it predicts a real reduction; otherwise plain damped mixing
    (which cannot be worse than the current residual) is used.
    """
    if len(points) < 2:
        return None
    residuals = np.stack([point.residual.ravel() for point in points])
    gram = residuals @ residuals.T
    n = len(points)
    diagonal = np.diag(gram)
    scale = float(diagonal.mean()) if diagonal.size else 0.0
    gram[np.diag_indices(n)] += regularization * max(scale, _PSD_FLOOR) + _PSD_FLOOR

    # KKT system for min c^T B c s.t. 1^T c = 1.
    system = np.zeros((n + 1, n + 1))
    system[:n, :n] = gram
    system[:n, n] = 1.0
    system[n, :n] = 1.0
    rhs = np.zeros(n + 1)
    rhs[n] = 1.0
    try:
        coefficients = np.linalg.solve(system, rhs)[:n]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coefficients)):
        return None
    total = float(coefficients.sum())
    if abs(total) < 1e-12:
        return None
    coefficients = coefficients / total
    if np.abs(coefficients).max() > max_coefficient:
        coefficients = np.clip(coefficients, -max_coefficient, max_coefficient)
        total = float(coefficients.sum())
        if abs(total) < 1e-12:
            return None
        coefficients = coefficients / total

    dm = np.zeros_like(points[0].dm)
    residual = np.zeros_like(dm)
    for coefficient, point in zip(coefficients, points):
        dm += coefficient * point.dm
        residual += coefficient * point.residual
    return dm + alpha * residual, _max_abs(residual)


def _safeguard_candidate(
    candidate: np.ndarray,
    current: _SCFPoint,
    alpha: float,
    overlap: np.ndarray,
    n_occ_a: int,
    *,
    trust_factor: float,
    max_step_norm: float | None,
) -> np.ndarray:
    """Keep an Anderson step physical and inside the mixing trust region.

    Falls back to the plain damped step ``γ + α(G[γ] - γ)`` whenever the candidate
    is non-finite, outside ``trust_factor`` times the damped-step size, or not a
    positive-semidefinite density with the correct electron count.  The damped step
    itself is a convex combination of two PSD densities, so it is always safe.
    """
    damped = current.dm + alpha * current.residual
    candidate = _symmetrize(np.asarray(candidate, dtype=float))
    if not np.all(np.isfinite(candidate)):
        return damped

    step = candidate - current.dm
    step_size = _max_abs(step)
    damped_size = _max_abs(alpha * current.residual)
    limit = trust_factor * damped_size
    if max_step_norm is not None:
        limit = max_step_norm if limit == 0.0 else min(limit, float(max_step_norm))
    if limit > 0.0 and step_size > limit:
        candidate = current.dm + step * (limit / step_size)

    if candidate.size:
        lowest = float(np.linalg.eigvalsh(candidate).min())
        tolerance = -_PSD_FLOOR * max(1.0, _max_abs(candidate))
        if lowest < tolerance:
            return damped
    expected_trace = 2.0 * n_occ_a
    trace = float(np.einsum("ij,ji->", overlap, candidate))
    if abs(trace - expected_trace) > 1e-6 * max(1.0, abs(expected_trace)):
        return damped
    return candidate


def _validate_stabilization(stabilization: str) -> Stabilization:
    allowed = {"auto", "legacy", "mixing", "anderson", "none"}
    value = str(stabilization).strip().lower()
    if value not in allowed:
        raise ValueError(f"stabilization must be one of {sorted(allowed)}, got {stabilization!r}")
    return cast(Stabilization, value)


def variational_hf_in_dft(
    ints,
    dm_a,
    dm_b,
    p_b,
    n_occ_a,
    *,
    max_iter=200,
    conv_tol=1e-10,
    density_tol=1e-6,
    converge_on: Literal["auto", "energy_and_density", "density"] = "auto",
    energy_offset: float = 0.0,
    stabilization: Stabilization = "auto",
    mix_alpha: float | None = None,
    anderson: bool = True,
    history_size: int = 8,
    diis_space: int = 6,
    anderson_trust: float = 2.0,
    adaptive_damping: bool = True,
    max_step_norm: float | None = None,
    cycle_tol: float = 1e-3,
    stall_window: int = 6,
    rollback_factor: float = 4.0,
    allow_unconverged: bool = False,
    history: list[SCFIteration] | None = None,
    log: Callable[[str], Any] | None = None,
):
    """Minimize ``E[γ_A]`` (module docstring) over A's closed-shell HF density.

    ``dm_a``/``dm_b`` are spin-summed AO densities (``dm_b`` stays fixed), ``p_b`` the
    level-shift projector that keeps A orthogonal to B.  Returns ``(dm_a, fock, energy,
    n_iterations)``; ``fock`` is the converged embedded Fock matrix including ``p_b``.

    Args:
        max_iter: maximum number of accepted density updates.
        conv_tol: energy-change threshold (Ha).
        density_tol: density threshold.  The legacy path applies it to
            ``max |γ_new - γ_old|`` (the reference implementation's criterion); the
            stabilized paths apply it to the fixed-point residual
            ``max |G[γ] - γ|``, which damping cannot make small spuriously.
        converge_on: ``"auto"`` (default), ``"energy_and_density"`` (the paper's
            criterion) or ``"density"``.  ``"auto"`` resolves to
            ``"energy_and_density"`` for an in-process backend and to ``"density"``
            for a file-driven one (``cheap_evaluations=False``): the latter's energy
            at the A-only density is not the functional consistent with its Fock
            (the SIESTA ASI one-step evaluation) and carries a numerical floor above
            ``conv_tol``, so the energy-change criterion is unreachable.  The
            density criterion is then the reliable one and the energy change is
            still recorded in the history.
        energy_offset: constant added to ``E[γ_A]`` and every history energy (Ha).
            It does not enter the SCF step, the density criterion or ``ΔE`` between
            steps; it puts the reported variational energy on a backend's
            total-energy reference (see :func:`reference_wf_in_dft_hamiltonian`).
        stabilization: ``"auto"`` (legacy for in-process backends, safeguarded
            mixing/Anderson for expensive ones and on detected cycling),
            ``"legacy"`` (exactly the old DIIS loop), ``"mixing"`` (damped fixed
            point), ``"anderson"`` (damped Anderson/DIIS with rollback), or
            ``"none"`` (undamped map without DIIS).
        mix_alpha: damping for the stabilized paths.  ``None`` starts at 0.4.
            ``anderson`` adapts it from the observed fixed-point mode (0.15-0.9)
            when ``adaptive_damping`` is on, and halves it on stalls; ``mixing``
            keeps it fixed.
        anderson: accelerate a stabilized path with Anderson/Pulay mixing.
        history_size: number of residual/density pairs in the Anderson subspace.
        diis_space: subspace size of the legacy PySCF DIIS (default 6, PySCF's own).
        anderson_trust: maximum size of an Anderson step relative to the plain
            damped step (the trust region).
        adaptive_damping: in a stabilized path, estimate the dominant fixed-point
            eigenvalue from the last two residuals and move ``mix_alpha`` towards
            the damping that minimizes it (0.15-0.9).  This raises alpha for a
            contracting map and lowers it for an oscillating one; on geometry 08 it
            measured a positive mode and moved alpha 0.4 -> 0.65, giving a steadier
            asymptotic contraction than fixed 0.4.  ``mixing`` keeps its fixed
            alpha; the safeguard reductions still apply everywhere.
        max_step_norm: optional absolute cap on ``max |Δγ|`` per accepted step.
        cycle_tol: two densities closer than this are treated as the same iterate
            when looking for a cycle.
        stall_window: iterations without residual improvement before damping is
            reduced / the legacy path is abandoned.
        rollback_factor: residual growth over the best seen value that triggers a
            rollback to the best density.
        allow_unconverged: return the last iterate with a warning instead of raising
            :class:`SCFConvergenceError`.  Off by default: a non-converged density is
            never returned silently.
        history: optional list filled with one :class:`SCFIteration` per accepted
            step (diagnostics).
        log: optional ``str -> None`` sink for the per-iteration diagnostic; defaults
            to this module's logger at INFO level.

    Raises:
        SCFConvergenceError: if the criteria are not met and ``allow_unconverged``
            is false (including non-finite energies/densities).
    """
    for capability in (CAP_HCORE, CAP_EXACT_EXCHANGE, CAP_ARBITRARY_DM_VEFF, CAP_ENERGY_ELEC):
        require_capability(ints, capability, "the variational HF-in-DFT SCF-in-SCF")

    h = np.asarray(ints.hcore(), dtype=float)
    s = np.asarray(ints.overlap(), dtype=float)
    e_nuc = float(ints.energy_nuc())
    dm_b = np.asarray(dm_b, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    n_occ_a = int(n_occ_a)
    if n_occ_a <= 0:
        raise ValueError(f"n_occ_a must be positive, got {n_occ_a}")
    if n_occ_a > s.shape[0]:
        raise ValueError(f"n_occ_a={n_occ_a} exceeds the {s.shape[0]} AO basis functions")

    stabilization = _validate_stabilization(stabilization)
    if max_iter < 1:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if conv_tol <= 0.0 or density_tol <= 0.0:
        raise ValueError("conv_tol and density_tol must be positive")
    if converge_on not in {"auto", "energy_and_density", "density"}:
        raise ValueError(
            f"converge_on must be 'auto', 'energy_and_density' or 'density', got {converge_on!r}"
        )
    if not np.isfinite(energy_offset):
        raise ValueError(f"energy_offset must be finite, got {energy_offset!r}")
    if history_size < 2:
        raise ValueError(f"history_size must be at least 2, got {history_size}")
    if anderson_trust <= 0.0:
        raise ValueError(f"anderson_trust must be positive, got {anderson_trust}")
    if max_step_norm is not None and max_step_norm <= 0.0:
        raise ValueError(f"max_step_norm must be positive, got {max_step_norm}")
    if mix_alpha is not None and stabilization != "legacy":
        if not 0.0 < float(mix_alpha) <= 1.0:
            raise ValueError(f"mix_alpha must be in (0, 1], got {mix_alpha}")

    legacy_fallback = stabilization == "auto"
    external_evaluations = _external_evaluations(ints)
    if stabilization == "auto":
        mode = "anderson" if external_evaluations else "legacy"
    else:
        mode = stabilization
    # A file-driven backend's one-step energy has a numerical floor above conv_tol,
    # so "energy_and_density" can never be satisfied there.  This is the same policy
    # as stabilization="auto": keep the in-process path exactly as it was, and use
    # the reliable criterion for an external code.
    effective_converge_on = _resolve_converge_on(ints, converge_on)

    def emit(message: str) -> None:
        if log is not None:
            log(message)
        else:
            logger.info(message)

    # One cache for the whole SCF: a repeat of an already-evaluated density (a
    # damping fixed point, a rollback, DIIS proposing the same step twice) must not
    # cost another external run.  Bounded so a long SCF on a large AO basis cannot
    # grow without limit; the current, best and Anderson densities stay referenced
    # directly by the loop.
    cache: dict[bytes, _SCFPoint] = {}
    cache_limit = max(24, 2 * max(history_size, 1) + 8)

    def evaluate(dm: np.ndarray) -> _SCFPoint:
        dm = np.asarray(dm, dtype=float)
        key = _density_key(dm)
        cached = cache.get(key)
        if cached is not None:
            return cached

        # Pair each potential with its energy at the *same* density so the callback
        # backend's one-entry evaluation cache is hit: veff_ll(d) and
        # energy_elec_ll(d) then share one ASI run.  This halves the cost of an
        # iteration relative to the old ordering and changes no value.
        dm_total = dm + dm_b
        fock_ll_total = h + ints.veff_ll(dm_total)
        e_ll_total = float(ints.energy_elec_ll(dm_total))
        fock_ll_a = h + ints.veff_ll(dm)
        e_ll_a = float(ints.energy_elec_ll(dm))
        veff_hf = ints.veff_hf(dm)

        fock = h + veff_hf + (fock_ll_total - fock_ll_a) + p_b
        e_hf = float(np.einsum("ij,ji->", h, dm) + 0.5 * np.einsum("ij,ji->", veff_hf, dm))
        energy = float(e_hf + e_ll_total - e_ll_a + e_nuc + energy_offset)
        try:
            _eps, coefficients = sla.eigh(fock, s)
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise SCFConvergenceError(
                f"the embedded Fock matrix could not be diagonalized at iteration "
                f"{len(cache) + 1}: {exc}"
            ) from exc
        occupied = coefficients[:, :n_occ_a]
        response = 2.0 * occupied @ occupied.T
        residual = response - dm
        point = _SCFPoint(
            dm=dm,
            fock=fock,
            response=response,
            residual=residual,
            energy=energy,
            residual_norm=_max_abs(residual),
        )
        if len(cache) >= cache_limit:
            cache.pop(next(iter(cache)))
        cache[key] = point
        return point

    def fail(
        message: str,
        point: _SCFPoint | None,
        records: list[SCFIteration],
    ) -> SCFConvergenceError:
        detail = ""
        if records:
            last = records[-1]
            detail = (
                f"; last step {last.iteration}: dE={last.delta_energy:.3e} Ha, "
                f"dDmax={last.dDmax:.3e}, residual={last.residual:.3e}, "
                f"mode={last.mode}"
            )
        return SCFConvergenceError(
            message + detail,
            history=tuple(records),
            last_density=None if point is None else point.dm,
            last_fock=None if point is None else point.fock,
            last_energy=None if point is None else point.energy,
        )

    dm = np.asarray(dm_a, dtype=float)
    if dm.shape != s.shape:
        raise ValueError(f"dm_a is {dm.shape}, expected {s.shape}")
    if dm_b.shape != s.shape:
        raise ValueError(f"dm_b is {dm_b.shape}, expected {s.shape}")
    if p_b.shape != s.shape:
        raise ValueError(f"p_b is {p_b.shape}, expected {s.shape}")
    if not np.all(np.isfinite(dm)):
        raise SCFConvergenceError("the initial subsystem-A density is not finite")
    current = evaluate(dm)
    e_old = current.energy
    records: list[SCFIteration] = []
    if history is not None:
        history.clear()

    diis = None
    if mode == "legacy":
        from pyscf import lib as pyscf_lib

        diis = pyscf_lib.diis.DIIS()
        diis.space = max(2, int(diis_space))

    alpha: float | None = None
    if mode in {"mixing", "anderson"}:
        alpha = 0.4 if mix_alpha is None else float(mix_alpha)
        if mode == "mixing":
            anderson = False

    anderson_points: list[_SCFPoint] = []
    best_point = current
    best_iteration = 0
    last_improvement = 0
    stall = 0
    # Consecutive legacy-DIIS steps whose residual exceeds the divergence threshold.
    # One such step is the normal DIIS startup overshoot, not a divergence: on
    # butyronitrile 2.5 A (PySCF, STO-3G) the very first extrapolation raises the
    # residual from 0.35 to 1.43 and the next one returns to 0.045, after which DIIS
    # converges in 35 iterations.  Abandoning the path on that single step (the old
    # behaviour) left the stabilized loop crawling for hundreds of iterations.
    divergence_streak = 0
    last_used_anderson = False
    # A short history of accepted densities is enough to recognise the 2-/3-cycle
    # that a badly conditioned fixed-point map falls into.
    cycle_densities: list[np.ndarray] = [current.dm]
    cycle_window = max(stall_window, 6)

    for iteration in range(1, max_iter + 1):
        # ------------------------------------------------------------------ #
        # Propose the next density.
        # ------------------------------------------------------------------ #
        used_anderson = False
        if mode == "legacy":
            fock_trial = current.fock
            if diis is not None:
                fock_trial = diis.update(
                    current.fock, xerr=_commutator(current.fock, current.dm, s)
                )
            try:
                _eps, coefficients = sla.eigh(fock_trial, s)
            except (np.linalg.LinAlgError, ValueError) as exc:
                raise fail(
                    f"the DIIS-extrapolated Fock matrix could not be diagonalized: {exc}",
                    current,
                    records,
                ) from exc
            occupied = coefficients[:, :n_occ_a]
            dm_new = 2.0 * occupied @ occupied.T
            step_alpha = None
        elif mode == "none":
            dm_new = current.response
            step_alpha = None
        else:
            if not anderson_points or anderson_points[-1].dm is not current.dm:
                anderson_points.append(current)
                if len(anderson_points) > max(history_size, 2):
                    anderson_points.pop(0)
            assert alpha is not None
            if (
                adaptive_damping
                and mode == "anderson"
                and not last_used_anderson
                and len(anderson_points) >= 2
            ):
                # The last accepted step was the plain damped map, so the ratio of
                # consecutive residuals estimates its dominant eigenvalue
                # lambda_eff = 1 - alpha (1 - lambda_raw).  Damping is optimal for
                # the observed mode at alpha / (1 - lambda_eff); move halfway and
                # clip.  A positive (contracting) mode raises alpha towards the
                # undamped iteration; a negative (oscillating) one lowers it.
                previous = anderson_points[-2]
                latest = anderson_points[-1]
                denominator = float(np.einsum("ij,ij->", previous.residual, previous.residual))
                if denominator > _PSD_FLOOR and previous.residual_norm > 10.0 * density_tol:
                    lambda_eff = (
                        float(np.einsum("ij,ij->", latest.residual, previous.residual))
                        / denominator
                    )
                    if np.isfinite(lambda_eff) and -2.0 < lambda_eff < 0.98:
                        target = np.clip(alpha / (1.0 - lambda_eff), 0.15, 0.9)
                        adjusted = float(np.clip(0.5 * alpha + 0.5 * target, 0.15, 0.9))
                        if abs(adjusted - alpha) > 0.03:
                            emit(
                                f"   damping adjusted {alpha:.3f} -> {adjusted:.3f} "
                                f"(observed mode {lambda_eff:+.3f})"
                            )
                            alpha = adjusted
            candidate = None
            if anderson and len(anderson_points) >= 2:
                accelerated = _anderson_candidate(anderson_points, alpha)
                # Use the extrapolation only if the residual combination predicts a
                # real reduction; an uninformative subspace must not replace the
                # safe damped step.
                if accelerated is not None and accelerated[1] < 0.9 * current.residual_norm:
                    candidate, _predicted = accelerated
                    used_anderson = True
            if candidate is None:
                candidate = current.dm + alpha * current.residual
            dm_new = _safeguard_candidate(
                candidate,
                current,
                alpha,
                s,
                n_occ_a,
                trust_factor=anderson_trust,
                max_step_norm=max_step_norm,
            )
            step_alpha = alpha

        # ------------------------------------------------------------------ #
        # Evaluate and check convergence.
        # ------------------------------------------------------------------ #
        new = evaluate(dm_new)
        dDmax = _max_abs(dm_new - current.dm)
        step_norm = float(np.linalg.norm(dm_new - current.dm))
        delta_energy = new.energy - e_old

        if not np.isfinite(new.energy) or not np.all(np.isfinite(new.dm)):
            raise fail(
                "the variational HF-in-DFT SCF produced a non-finite density or energy "
                "at iteration "
                f"{iteration}",
                new,
                records,
            )

        if mode == "legacy":
            density_ok = dDmax < density_tol
        else:
            # Damping can make the accepted step small while the fixed-point
            # residual is still large; require the residual itself to vanish.
            density_ok = new.residual_norm < density_tol
        energy_ok = abs(delta_energy) < conv_tol
        converged = density_ok and (energy_ok or effective_converge_on == "density")

        record = SCFIteration(
            iteration=iteration,
            energy=new.energy,
            delta_energy=delta_energy,
            dDmax=dDmax,
            residual=new.residual_norm,
            step_norm=step_norm,
            mode=("anderson" if used_anderson else ("mixing" if mode == "anderson" else mode)),
            mix_alpha=step_alpha,
            converged=converged,
        )
        records.append(record)
        if history is not None:
            history.append(record)
        emit(
            f"SCF-in-SCF iter {iteration:3d}: E={new.energy:.10f} Ha "
            f"dE={delta_energy:+.3e} dDmax={dDmax:.3e} "
            f"residual={new.residual_norm:.3e} step={step_norm:.3e} "
            f"mode={record.mode}" + ("" if step_alpha is None else f" alpha={step_alpha:.3f}")
        )

        if new.residual_norm < best_point.residual_norm:
            best_point = new
            best_iteration = iteration
            last_improvement = iteration

        if converged:
            return new.dm, new.fock, new.energy, iteration

        last_used_anderson = used_anderson
        current = new
        e_old = new.energy
        cycle_densities.append(new.dm)
        if len(cycle_densities) > cycle_window + 1:
            cycle_densities.pop(0)

        if mode in {"mixing", "anderson"}:
            # -------------------------------------------------------------- #
            # Adaptive damping / rollback.
            # -------------------------------------------------------------- #
            assert alpha is not None
            if (
                new.residual_norm > rollback_factor * max(best_point.residual_norm, _PSD_FLOOR)
                and new.residual_norm > 100.0 * density_tol
                and best_point is not new
            ):
                alpha = max(alpha * 0.5, 0.05)
                anderson_points = [best_point]
                stall = 0
                current = best_point
                e_old = best_point.energy
                cycle_densities = [best_point.dm]
                emit(
                    f"   rollback to iteration {best_iteration} "
                    f"(residual {best_point.residual_norm:.3e}); alpha -> {alpha:.3f}"
                )
                continue
            if new.residual_norm >= best_point.residual_norm * 1.05:
                stall += 1
                if stall >= max(2, stall_window // 2):
                    alpha = max(alpha * 0.5, 0.05)
                    anderson_points = [current]
                    stall = 0
                    emit(f"   damping reduced to alpha={alpha:.3f} after a stall")
            else:
                stall = 0

        elif legacy_fallback:
            # -------------------------------------------------------------- #
            # Detect a legacy DIIS cycle / divergence / stagnation and fall
            # back to safeguarded mixing+Anderson from the best density.
            # -------------------------------------------------------------- #
            earlier = cycle_densities[:-1]
            cycle = (
                len(earlier) >= 2
                and current.residual_norm > 100.0 * density_tol
                and _max_abs(current.dm - earlier[-1]) > cycle_tol
                and any(_max_abs(current.dm - old) < cycle_tol for old in earlier[:-1])
            )
            stagnated = (
                iteration - last_improvement >= stall_window
                and current.residual_norm > 100.0 * density_tol
            )
            diverged_now = (
                current.residual_norm > rollback_factor * max(best_point.residual_norm, _PSD_FLOOR)
                and current.residual_norm > 100.0 * density_tol
            )
            divergence_streak = divergence_streak + 1 if diverged_now else 0
            # Require two consecutive divergent steps: a single one is the DIIS
            # startup overshoot (see `divergence_streak` above).
            diverged = divergence_streak >= 2
            if cycle or stagnated or diverged:
                reason = "cycle" if cycle else ("stagnation" if stagnated else "divergence")
                mode = "anderson"
                alpha = 0.4 if mix_alpha is None else float(mix_alpha)
                diis = None
                current = best_point
                e_old = best_point.energy
                anderson_points = [best_point]
                cycle_densities = [best_point.dm]
                stall = 0
                emit(
                    f"   legacy DIIS {reason} detected; switching to safeguarded "
                    f"mixing+Anderson at iteration {best_iteration} "
                    f"(alpha={alpha:.3f}, residual={best_point.residual_norm:.3e})"
                )

    message = f"variational HF-in-DFT SCF for subsystem A did not converge in {max_iter} iterations"
    if allow_unconverged:
        warnings.warn(
            message + "; returning the last density (allow_unconverged=True)",
            stacklevel=2,
        )
        emit("   " + message)
        return current.dm, current.fock, current.energy, max_iter
    raise fail(message, current, records)


def reference_wf_in_dft_hamiltonian(
    adapter,
    fragment_ao,
    *,
    n_frozen_occ: int,
    n_virtual: int,
    n_shells: int = 0,
    scf_options: Mapping[str, Any] | None = None,
) -> tuple[EmbeddedHamiltonian, ReferenceProtocolInfo]:
    """The protocol's active-space Hamiltonian, after ``adapter.run_low_level()``.

    ``fragment_ao`` are subsystem A's AO indices (the concentric-localization basis).
    The active space is A's occupied orbitals ``n_frozen_occ:`` plus the ``n_virtual``
    lowest canonical virtuals of the kept shells (``n_shells`` after shell 0).

    ``scf_options`` is passed to :func:`variational_hf_in_dft` (for example
    ``{"max_iter": 50, "stabilization": "anderson", "mix_alpha": 0.4, "log": print}``).
    With no options the defaults stabilize a file-driven backend automatically and
    leave the PySCF path unchanged.

    A backend may expose ``high_level_reference_energy(layer)`` -- SIESTA does -- the
    same-density offset between the bare high-level Hamiltonian and its low-level
    A-layer total (pseudopotential double-counting and the like).  It is subtracted
    from ``E[γ_A]`` (and hence from ``e_core``) exactly as
    :meth:`ProjectionEmbeddingAdapter.projection_energy` subtracts it from
    ``E_high(A)``.  Without it the protocol's total is off by tens of hartree on a
    pseudopotential code; PySCF has no such method, so its numbers are unchanged.

    The offset is a *same-density* quantity, evaluated at the initial A density, and on
    SIESTA it changes by a few hartree across a bond stretch (measured 5.76, 8.01 and
    8.71 Ha at butyronitrile 0.8, 1.3 and 2.5 A in SZ).  It is a same-density alignment
    of the bare high-level Hamiltonian with the low-level A-layer total at *one*
    geometry, not a scan-comparable reference: applying it to ``E[γ_A]`` shifts a
    relative energy by the offset's geometry dependence (up to ~1850 kcal/mol over the
    0.8-2.5 A range).  The applied value is reported as
    :attr:`ReferenceProtocolInfo.energy_offset`;
    ``info.e_variational - info.energy_offset`` is the raw variational energy on one
    consistent reference and is what relative energies must use.

    A file-driven backend also gets ``density_tol=1e-4`` unless ``scf_options``
    overrides it.  Its ASI one-step density residual reaches a practical floor above
    the 1e-6 default at stretched geometries; the energy at the accepted density is
    still converged far below the resolution of the cross-code comparison.
    """
    ints = adapter.ints
    require_capability(ints, CAP_ERI, "the reference WF-in-DFT active-space Hamiltonian")
    s = np.asarray(ints.overlap())
    dm_a0, dm_b = np.asarray(adapter._dm_a), np.asarray(adapter._dm_b)
    p_b = np.asarray(adapter.p_b)
    n_occ_a = int(round(np.einsum("ij,ji->", dm_a0, s) / 2.0))

    history: list[SCFIteration] = []
    options = dict(scf_options or {})
    if "history" in options:
        raise ValueError("pass SCF diagnostics through scf_options; 'history' is reserved")
    if "energy_offset" in options:
        raise ValueError("energy_offset is derived from the backend, not a scf_option")
    if "density_tol" not in options:
        density_floor = _external_density_tol(ints)
        if density_floor is not None:
            options["density_tol"] = density_floor
    reference = getattr(ints, "high_level_reference_energy", None)
    if callable(reference):
        layer = getattr(getattr(adapter, "p", None), "A_LL", None)
        if layer is None:
            raise ValueError(
                "the backend exposes high_level_reference_energy but the adapter has "
                "no p.A_LL layer to evaluate it on"
            )
        energy_offset = -float(reference(layer))
        logger.info(
            "reference WF-in-DFT: applying same-density reference energy offset "
            "%.6f Ha to E[gamma_A]",
            energy_offset,
        )
    else:
        energy_offset = 0.0
    dm_a, f, e_var, n_iter = variational_hf_in_dft(
        ints,
        dm_a0,
        dm_b,
        p_b,
        n_occ_a,
        history=history,
        energy_offset=energy_offset,
        **options,
    )

    # Canonical orbitals of the embedded Fock matrix.  B's occupied orbitals sit at
    # ~mu (the projector), so keep only A's occupied block and the genuine virtuals.
    eps, c = sla.eigh(f, s)
    n_occ_b = int(round(np.einsum("ij,ji->", dm_b, s) / 2.0))
    n_virt = c.shape[1] - n_occ_a - n_occ_b
    coeff = np.hstack([c[:, :n_occ_a], c[:, n_occ_a : n_occ_a + n_virt]])
    if n_virt and eps[n_occ_a + n_virt - 1] > 0.5 * float(np.abs(p_b).max()):
        raise RuntimeError("projector-shifted orbitals leaked into the virtual block")

    localise = concentric_localization_selector(
        s,
        np.asarray(fragment_ao),
        f,
        n_shells=n_shells,
        max_virtual=n_virtual,
        canonical=True,
        # The reference code keeps len(fragment_ao) shell-0 columns even past the SVD rank
        # (its active virtuals then depend on LAPACK's null-space basis); matched here.
        shell0_size=len(fragment_ao),
    )
    coeff_loc, sigma2 = localise(coeff, n_occ_a)
    n_kept = int(np.sum(sigma2 > 0.5))
    if n_kept < n_virtual:
        raise ValueError(f"only {n_kept} concentric virtuals kept; n_virtual={n_virtual}")
    c_act = np.hstack(
        [coeff_loc[:, n_frozen_occ:n_occ_a], coeff_loc[:, n_occ_a : n_occ_a + n_virtual]]
    )
    n_act_occ = n_occ_a - n_frozen_occ

    # One-body operator seen by the active electrons: everything in F except their own
    # mean field, i.e. h + G_HF[core] + v_emb + P_B.
    c_aocc = coeff_loc[:, n_frozen_occ:n_occ_a]
    h1 = c_act.T @ (f - ints.veff_hf(2.0 * c_aocc @ c_aocc.T)) @ c_act
    h2 = ints.eri_mo(c_act)
    occ = range(n_act_occ)
    e_ref = float(
        2.0 * sum(h1[i, i] for i in occ)
        + sum(2.0 * h2[i, i, j, j] - h2[i, j, j, i] for i in occ for j in occ)
    )
    ham = EmbeddedHamiltonian(
        h1=h1,
        h2=h2,
        e_core=e_var - e_ref,
        nelec=(n_act_occ, n_act_occ),
        meta={"protocol": "reference-wf-in-dft"},
    )
    energies = tuple(float(e) for e in np.diag(c_act.T @ f @ c_act))
    converged = bool(history and history[-1].converged)
    return ham, ReferenceProtocolInfo(
        e_variational=e_var,
        e_reference=e_ref,
        n_iterations=n_iter,
        n_virtual_kept=n_kept,
        active_orbital_energies=energies,
        converged=converged,
        scf_history=tuple(history),
        energy_offset=energy_offset,
    )
