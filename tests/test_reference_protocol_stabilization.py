# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Stabilization tests for the reference WF-in-DFT SCF-in-SCF.

The fake ``ints`` below is a deterministic 2x2 closed-shell problem with

    F(γ) = h + V_hf(γ),   V_hf(γ) = [[0, k], [k, 0]],   k = K tanh(γ_01),

so the occupied projector map on the density off-diagonal is

    γ_01 -> G(γ_01) = -2 K tanh(γ_01) / sqrt(1 + 4 K^2 tanh(γ_01)^2).

For ``K = 1.5`` the fixed point at zero has ``G' = -3``: plain iteration runs into
a stable two-density cycle and never satisfies the density criterion.  Damped
mixing and safeguarded Anderson/DIIS converge to it, which is the behaviour the
SIESTA ASI backend needs.  The fake has no external dependencies and runs in
milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from embasi_qiskit_integration.backends.base import (
    CAP_ARBITRARY_DM_VEFF,
    CAP_ENERGY_ELEC,
    CAP_EXACT_EXCHANGE,
    CAP_HCORE,
)
from embasi_qiskit_integration.reference_protocol import (
    SCFConvergenceError,
    SCFIteration,
    _external_evaluations,
    variational_hf_in_dft,
)

_CAPABILITIES = frozenset(
    {CAP_HCORE, CAP_EXACT_EXCHANGE, CAP_ARBITRARY_DM_VEFF, CAP_ENERGY_ELEC}
)


class OscillatingIntegrals:
    """A 2-orbital closed-shell fake with an expansive off-diagonal map.

    ``cheap`` controls the ``cheap_evaluations`` marker: ``False`` makes
    ``stabilization="auto"`` treat it like the callback (SIESTA/FHI-aims) backend
    and start stabilized; ``True`` or unset keeps the legacy path.
    """

    capabilities = _CAPABILITIES

    def __init__(self, *, K: float = 1.5, cheap: bool | None = None, contractive: bool = False):
        self.K = float(K)
        self.contractive = bool(contractive)
        self.veff_calls = 0
        self.energy_calls = 0
        self.veff_hf_calls = 0
        if cheap is not None:
            self.cheap_evaluations = bool(cheap)

    def overlap(self) -> np.ndarray:
        return np.eye(2)

    def hcore(self) -> np.ndarray:
        return np.diag([0.0, 1.0])

    def energy_nuc(self) -> float:
        return 0.0

    def veff_ll(self, dm: np.ndarray) -> np.ndarray:
        self.veff_calls += 1
        return np.zeros((2, 2))

    def energy_elec_ll(self, dm: np.ndarray) -> float:
        self.energy_calls += 1
        return 0.0

    def veff_hf(self, dm: np.ndarray) -> np.ndarray:
        self.veff_hf_calls += 1
        sign = -1.0 if self.contractive else 1.0
        k = sign * self.K * np.tanh(float(dm[0, 1]))
        return np.array([[0.0, k], [k, 0.0]])


def _density(offdiagonal: float) -> np.ndarray:
    """The 2x2 one-occupied-orbital density with the requested off-diagonal."""
    root = np.sqrt(max(0.0, 1.0 - offdiagonal**2))
    c = np.array(
        [
            np.sqrt((1.0 + root) / 2.0),
            np.sign(offdiagonal) * np.sqrt((1.0 - root) / 2.0),
        ]
    )
    return 2.0 * np.outer(c, c)


def _fixed_point_residual(ints: OscillatingIntegrals, dm: np.ndarray, n_occ: int = 1) -> float:
    """Independent ``max |G(dm) - dm|`` for the fake (S = I, dm_b = p_b = 0)."""
    fock = ints.hcore() + ints.veff_hf(dm)
    _eps, c = sla.eigh(fock, np.eye(2))
    response = 2.0 * c[:, :n_occ] @ c[:, :n_occ].T
    return float(np.abs(response - dm).max())


def _run(ints, **kwargs):
    return variational_hf_in_dft(
        ints,
        _density(0.5),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        1,
        **kwargs,
    )


def test_plain_fixed_point_cycles_and_raises():
    ints = OscillatingIntegrals()
    history: list[SCFIteration] = []
    with pytest.raises(SCFConvergenceError) as excinfo:
        _run(ints, stabilization="none", max_iter=25, history=history)

    error = excinfo.value
    assert isinstance(error, RuntimeError)
    assert "did not converge in 25 iterations" in str(error)
    assert "dDmax" in str(error) and "residual" in str(error)
    assert len(history) == 25
    assert not any(record.converged for record in history)
    assert history[-1].residual > 1e-2  # a genuine cycle, not slow convergence
    assert history[-1].dDmax > 1e-2
    assert tuple(error.history) == tuple(history)
    assert error.last_density is not None and error.last_fock is not None
    assert error.last_energy == pytest.approx(history[-1].energy)


def test_damped_mixing_converges_where_the_plain_map_cycles():
    ints = OscillatingIntegrals()
    history: list[SCFIteration] = []
    dm, fock, energy, iterations = _run(
        ints, stabilization="mixing", max_iter=120, history=history
    )

    assert iterations < 120
    assert history[-1].converged
    assert history[-1].residual < 1e-6
    assert abs(history[-1].delta_energy) < 1e-10
    assert history[-1].mode == "mixing"
    assert history[-1].mix_alpha == pytest.approx(0.4)
    assert _fixed_point_residual(ints, dm) < 1e-6
    assert np.trace(dm) == pytest.approx(2.0, abs=1e-10)
    assert np.linalg.eigvalsh(dm).min() > -1e-10
    np.testing.assert_allclose(fock, fock.T, atol=1e-12)
    assert np.isfinite(energy)


def test_adaptive_damping_raises_alpha_on_a_contracting_map():
    """A monotone map can take a much larger step than the conservative 0.4."""
    ints = OscillatingIntegrals(contractive=True)
    history: list[SCFIteration] = []
    messages: list[str] = []
    dm, _fock, _energy, iterations = _run(
        ints,
        stabilization="anderson",
        adaptive_damping=True,
        max_iter=150,
        history=history,
        log=messages.append,
    )

    assert iterations < 150
    assert any("damping adjusted" in message for message in messages)
    assert any(record.mix_alpha == pytest.approx(0.65) for record in history)
    assert history[-1].converged
    assert _fixed_point_residual(ints, dm) < 1e-6


def test_converge_on_density_is_an_explicit_opt_in():
    """Density-only convergence must be requested, never silent."""
    ints = OscillatingIntegrals(contractive=True)
    with pytest.raises(SCFConvergenceError):
        _run(ints, stabilization="mixing", max_iter=40, conv_tol=1e-300)

    history: list[SCFIteration] = []
    dm, _fock, _energy, iterations = _run(
        ints,
        stabilization="mixing",
        max_iter=40,
        conv_tol=1e-300,
        converge_on="density",
        history=history,
    )

    assert iterations < 40
    assert history[-1].converged
    assert history[-1].residual < 1e-6
    # The energy change is real but far above the strict (unreachable) threshold.
    assert abs(history[-1].delta_energy) > 1e-10
    assert _fixed_point_residual(ints, dm) < 1e-6


def test_auto_starts_stabilized_for_an_expensive_backend():
    ints = OscillatingIntegrals(cheap=False)
    history: list[SCFIteration] = []
    messages: list[str] = []
    dm, _fock, _energy, iterations = _run(
        ints,
        stabilization="auto",
        max_iter=120,
        history=history,
        log=messages.append,
    )

    assert iterations < 120
    assert all(record.mode != "legacy" for record in history)
    assert history[-1].converged
    assert _fixed_point_residual(ints, dm) < 1e-6
    assert any("SCF-in-SCF iter" in message for message in messages)


def test_auto_falls_back_when_legacy_diis_cycles(monkeypatch):
    # Make PySCF's DIIS a no-op so the legacy phase is the undamped fixed-point
    # map, which cycles; the auto path must notice and switch to stabilization.
    from pyscf import lib

    monkeypatch.setattr(lib.diis.DIIS, "update", lambda self, x, xerr=None: x)

    ints = OscillatingIntegrals(cheap=True)
    history: list[SCFIteration] = []
    messages: list[str] = []
    dm, _fock, _energy, iterations = _run(
        ints,
        stabilization="auto",
        max_iter=120,
        history=history,
        log=messages.append,
    )

    assert iterations < 120
    assert any(record.mode == "legacy" for record in history)
    assert any(record.mode != "legacy" for record in history)
    assert any("switching to safeguarded" in message for message in messages)
    assert history[-1].converged
    assert _fixed_point_residual(ints, dm) < 1e-6


def test_legacy_path_matches_the_original_reference_loop():
    """The default PySCF route must be the old algorithm, not just close to it."""
    ints = OscillatingIntegrals()
    expected = _original_loop(ints, _density(0.5), np.zeros((2, 2)), np.zeros((2, 2)), 1)

    dm, fock, energy, iterations = _run(ints, stabilization="legacy", max_iter=100)

    np.testing.assert_array_equal(dm, expected[0])
    np.testing.assert_array_equal(fock, expected[1])
    assert energy == expected[2]
    assert iterations == expected[3]


def test_allow_unconverged_warns_and_returns_the_last_density():
    history: list[SCFIteration] = []
    with pytest.warns(UserWarning, match="allow_unconverged=True"):
        dm, fock, energy, iterations = _run(
            OscillatingIntegrals(),
            stabilization="none",
            max_iter=3,
            allow_unconverged=True,
            history=history,
        )

    assert iterations == 3
    assert dm.shape == (2, 2) and fock.shape == (2, 2)
    assert np.isfinite(energy)
    assert history and not history[-1].converged


def test_iteration_diagnostics_are_emitted_and_finite():
    messages: list[str] = []
    history: list[SCFIteration] = []
    _run(
        OscillatingIntegrals(),
        stabilization="mixing",
        max_iter=120,
        history=history,
        log=messages.append,
    )

    assert messages
    assert all(
        ("E=" in message and "dE=" in message and "dDmax=" in message)
        and "residual=" in message
        and "step=" in message
        and "mode=" in message
        for message in messages
    )
    for record in history:
        assert record.iteration >= 1
        assert np.isfinite(record.energy)
        assert np.isfinite(record.delta_energy)
        assert np.isfinite(record.dDmax)
        assert np.isfinite(record.residual)
        assert np.isfinite(record.step_norm)
    assert [record.iteration for record in history] == list(range(1, len(history) + 1))


def test_potential_and_energy_share_one_evaluation_per_density():
    """Each density is evaluated once, with veff and energy paired."""
    ints = OscillatingIntegrals()
    _dm, _fock, _energy, iterations = _run(
        ints, stabilization="mixing", max_iter=120
    )

    # Initial density plus one new density per accepted iteration.
    evaluations = iterations + 1
    assert ints.veff_calls == 2 * evaluations
    assert ints.energy_calls == 2 * evaluations
    # veff_hf is paired with its density too, not recomputed in the energy.
    assert ints.veff_hf_calls == evaluations


def test_auto_detects_the_callback_backend_cost_marker():
    """The real SIESTA/FHI-aims route declares its evaluations expensive."""
    from embasi_qiskit_integration.backends.callback import CallbackIntegrals, Evaluation

    def evaluator(dm):
        density = np.asarray(dm, dtype=float)
        if density.ndim == 1:
            density = np.diag(density)
        return Evaluation((density.copy(),), np.eye(2), float(np.trace(density)))

    expensive = CallbackIntegrals(
        evaluator, evaluator, None, nao=2, cheap_evaluations=False
    )
    cheap = CallbackIntegrals(evaluator, evaluator, None, nao=2, cheap_evaluations=True)
    assert _external_evaluations(expensive) is True
    assert _external_evaluations(cheap) is False
    assert _external_evaluations(OscillatingIntegrals()) is False


def test_invalid_knobs_fail_loudly():
    with pytest.raises(ValueError, match="stabilization must be one of"):
        _run(OscillatingIntegrals(), stabilization="bogus")
    with pytest.raises(ValueError, match="mix_alpha"):
        _run(OscillatingIntegrals(), stabilization="mixing", mix_alpha=0.0)
    with pytest.raises(ValueError, match="mix_alpha"):
        _run(OscillatingIntegrals(), stabilization="anderson", mix_alpha=1.5)
    with pytest.raises(ValueError, match="converge_on"):
        _run(OscillatingIntegrals(), converge_on="energy")


def _original_loop(
    ints, dm_a, dm_b, p_b, n_occ_a, *, max_iter=100, conv_tol=1e-10, density_tol=1e-6
):
    """The pre-stabilization function, verbatim, used as the preservation oracle."""
    from pyscf import lib

    h = np.asarray(ints.hcore())
    s = np.asarray(ints.overlap())
    e_nuc = ints.energy_nuc()

    def fock_ll(dm):
        return h + ints.veff_ll(dm)

    def energy(dm):
        e_hf = np.einsum("ij,ji->", h, dm) + 0.5 * np.einsum("ij,ji->", ints.veff_hf(dm), dm)
        e_ll_tot = ints.energy_elec_ll(dm + dm_b)
        e_ll_a = ints.energy_elec_ll(dm)
        return float(e_hf + e_ll_tot - e_ll_a + e_nuc)

    def fock(dm):
        return h + ints.veff_hf(dm) + (fock_ll(dm + dm_b) - fock_ll(dm)) + p_b

    diis = lib.diis.DIIS()
    dm, e_old = np.asarray(dm_a), energy(np.asarray(dm_a))
    for iteration in range(1, max_iter + 1):
        f = fock(dm)
        f = diis.update(f, xerr=f @ dm @ s - s @ dm @ f)
        _eps, c = sla.eigh(f, s)
        dm_new = 2.0 * c[:, :n_occ_a] @ c[:, :n_occ_a].T
        e_new = energy(dm_new)
        converged = abs(e_new - e_old) < conv_tol and np.abs(dm_new - dm).max() < density_tol
        dm, e_old = dm_new, e_new
        if converged:
            return dm, fock(dm), e_new, iteration
    raise RuntimeError("did not converge")
