# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Outer self-consistency loop (``EmbeddingWorkflow._run_outer_loop``).

The loop logic is package-owned; the only thing blocking it against *real*
EmbASI is that ``construct_embedded_fock(dmab_in=...)`` wants a
``SpinKpointArray`` rather than the plain ``(nao, nao)`` density the loop feeds
back.  Here we exercise the loop end-to-end against a mock that closes that gap
*on the mock side only* -- it accepts a plain ``(nao, nao)`` density and exposes
low-level energies under EmbASI's own attribute names -- so the loop's control
flow (cycle count, convergence criteria, reseed policy) is tested on honest
adapter numbers without EmbASI.

The mock is a real PySCF RHF partitioned into A/B subsystems, identical in
spirit to ``tests/_mpi_workflow_mock.py`` but with a working ``feedback`` path:
``construct_embedded_fock(dmab_in=...)`` rebuilds ``F_emb`` from the fed-back
density, so successive cycles genuinely move (and, because the high-level solver
here is FCI on the full A space, converge back to the same fixed point).

NOTE: the surrogate low-level energies below (``subsys_A_lowlvl_totalen`` /
``subsys_AB_lowlvl_scftotalen``, the names the adapter reads) are the only
invented low-level energies in the suite, and they live entirely inside this
test mock -- never in the adapter.  They are constants, so they contribute a
fixed offset to ``ProjectionEnergy.total`` and do not affect the convergence
behaviour under test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

# Imported after the importorskip above, so the module skips cleanly without
# pyscf rather than failing at import time -- hence the E402 waivers.
from embasi_qiskit_integration.embedding import EmbeddingWorkflow
from embasi_qiskit_integration.projection_embedding_adapter import (
    ProjectionEmbeddingAdapter,
    PySCFIntegrals,
)
from embasi_qiskit_integration.solvers import FCISolver


class _FeedbackMockEmbedding:
    """Partitioned RHF standing in for ProjectionEmbedding, with feedback.

    ``construct_embedded_fock`` accepts an optional ``dmab_in`` (plain ndarray,
    the shape the adapter's ``feedback`` produces).  Without it, it returns the
    reference partition; with it, it rebuilds ``F_emb`` from the supplied total
    density so the outer loop's second cycle actually differs from the first.

    Exposes surrogate low-level energies under EmbASI's own attribute names
    (``subsys_A_lowlvl_totalen`` / ``subsys_AB_lowlvl_scftotalen``, in eV, the
    form the adapter's ``_low_level_energies`` reads) so ``projection_energy``
    runs.  Constants -> a fixed energy offset that does not affect the
    convergence behaviour under test.

    It also mimics EmbASI's ``A_LL.atoms.calc.mol`` reach-through that
    ``projection_energy`` uses to rebase ``E_high(A)`` onto ``E_low(A)``'s nuclear
    footing.  Real EmbASI runs the A layer on a *ghosted* subsystem-A ``mol``;
    this mock never ghosts -- A and B share the one supersystem ``mol`` -- so the
    A-fragment footing simply *is* the supersystem footing and the adapter's
    ``footing_shift`` comes out ~0.  Exposing the same ``mol`` here keeps that
    reach-through working (and the resulting fixed offset stable) without pulling
    a second, differently-nuclei molecule into the mock.
    """

    projection = "level-shift"
    mu_val = None  # set per-instance in __init__ to match the adapter's mu

    # Surrogate low-level energies (constants -> fixed energy offset).  EmbASI
    # stores these in eV; the adapter converts with its own eV->Ha factor.  The
    # numeric value is irrelevant to the loop's control flow.
    subsys_A_lowlvl_totalen = -1.0
    subsys_AB_lowlvl_scftotalen = -2.5

    def __init__(self, mol, mu: float, n_occ_a: int):
        from pyscf import scf

        mf = scf.RHF(mol).run()
        c = mf.mo_coeff
        occ = mf.mo_occ > 0
        c_occ = c[:, occ]
        n_occ = c_occ.shape[1]
        assert 0 < n_occ_a < n_occ

        self._mu = mu
        self.mu_val = mu  # adapter cross-checks this against its own mu
        self._s = mol.intor("int1e_ovlp")
        self._c = c
        self._eps0 = mf.mo_energy.copy()
        self._b_cols = np.arange(n_occ_a, n_occ)

        # Minimal A_LL.atoms.calc.mol reach-through for _a_fragment_footing.  No
        # ghosting in the mock: the A-fragment mol is the supersystem mol, so the
        # rebasing is a full-vs-full no-op (footing_shift ~ 0).
        self.A_LL = SimpleNamespace(atoms=SimpleNamespace(calc=SimpleNamespace(mol=mol)))

        c_a = c_occ[:, :n_occ_a]
        c_b = c_occ[:, n_occ_a:]
        self._ref_dm_a = 2.0 * (c_a @ c_a.T)
        self._dm_b = 2.0 * (c_b @ c_b.T)
        self.mo_coeffs_A_LL = c_a[np.newaxis, :, :]
        self.mo_coeffs_B_LL = c_b[np.newaxis, :, :]

    def _assemble_fock(self):
        """F_emb = S C diag(eps) C^T S with B occupied lifted by mu."""
        eps = self._eps0.copy()
        eps[self._b_cols] += self._mu
        sc = self._s @ self._c
        return sc @ np.diag(eps) @ sc.T

    def construct_embedded_fock(self, dmab_in=None):
        fock = self._assemble_fock()
        if dmab_in is None:
            dm_a = self._ref_dm_a
        else:
            # dmab_in arrives as EmbASI's SpinKpointArray (the adapter wraps the
            # fed-back (nao, nao) density in run_low_level); unwrap the single
            # spin/k-point block exactly as EmbASI's qmcode adapter does with
            # ``density_matrix_in[0, 0]``.
            dm_total = np.asarray(dmab_in[0, 0])
            # Fed-back total density minus the frozen environment = new γ^A.
            # (Real EmbASI re-localizes; the mock keeps the algebra honest by
            # just subtracting the frozen B block, which is exactly what the
            # adapter's feedback passed in.)
            dm_a = dm_total - self._dm_b
        return (
            dm_a[np.newaxis, :, :],
            self._dm_b[np.newaxis, :, :],
            fock[np.newaxis, :, :],
        )


def _build_adapter(n_occ_a: int = 1, mu: float = 1.0e6):
    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", basis="sto-3g")
    mf_hl = mol.RHF()
    mock = _FeedbackMockEmbedding(mol, mu=mu, n_occ_a=n_occ_a)
    return ProjectionEmbeddingAdapter(mock, PySCFIntegrals(mf_hl), mu=mu)


def _workflow(**overrides):
    """A minimally-configured EmbeddingWorkflow (no CLI parsing)."""
    return EmbeddingWorkflow(_cli_parse_args=False, **overrides)


def test_embedded_hamiltonian_symmetrizes_roundoff_asymmetry():
    """h1 round-off asymmetry is symmetrized; gross asymmetry fails loudly.

    ``h1 = C^T (h_emb + veff_in) C`` is Hermitian in exact arithmetic, but the
    rotation accumulates round-off that grows with the active-space size and
    would otherwise trip the contract's strict ``atol=1e-8`` Hermiticity check.
    We inject a controlled asymmetry into ``h_emb`` and check both branches.
    """
    adapter = _build_adapter()
    adapter.run_low_level()
    orbitals = adapter.build_orbitals(n_frozen_occ=0, n_virtual=None)

    # h_emb = _fock - veff_hl(dm_a); perturbing _fock (a plain attribute) skews
    # h_emb by the same amount without touching the properties.
    base_fock = adapter._fock.copy()
    nao = base_fock.shape[0]

    # Round-off-scale asymmetry (below the 1e-6 assert): accepted + symmetrized,
    # so the contract's stricter 1e-8 Hermiticity check passes.
    skew = np.zeros((nao, nao))
    skew[0, 1] = 1e-9
    adapter._fock = base_fock + skew
    try:
        ham = adapter.embedded_hamiltonian(orbitals)
        assert np.allclose(ham.h1, ham.h1.T, atol=1e-12)  # exactly symmetric now
    finally:
        adapter._fock = base_fock

    # Gross asymmetry (above 1e-6): must raise rather than be silently hidden.
    skew_big = np.zeros((nao, nao))
    skew_big[0, 1] = 1e-3
    adapter._fock = base_fock + skew_big
    try:
        with pytest.raises(ValueError, match="non-Hermitian well beyond round-off"):
            adapter.embedded_hamiltonian(orbitals)
    finally:
        adapter._fock = base_fock


def test_single_cycle_matches_direct_calls():
    """max_cycles=1 reproduces build->downfold->solve->energy exactly."""
    adapter = _build_adapter()
    adapter.run_low_level()

    # Direct one-pass reference.
    orbitals = adapter.build_orbitals(n_frozen_occ=0, n_virtual=None)
    ham = adapter.embedded_hamiltonian(orbitals)
    result = FCISolver().solve(ham)
    ref = adapter.projection_energy(result, orbitals)

    # Loop with a single cycle.
    adapter2 = _build_adapter()
    adapter2.run_low_level()
    wf = _workflow(solver="fci", max_cycles=1)
    energy = wf._run_outer_loop(adapter2, FCISolver(), None, rank=0, log=lambda *a, **k: None)
    assert energy.total == pytest.approx(ref.total, abs=1e-9)


def test_loop_converges_and_stops_early():
    """A multi-cycle run converges and returns before max_cycles."""
    adapter = _build_adapter()
    adapter.run_low_level()
    logs: list[str] = []
    wf = _workflow(solver="fci", max_cycles=8, e_tol=1e-6, rho_tol=1e-5)
    energy = wf._run_outer_loop(adapter, FCISolver(), None, rank=0, log=logs.append)
    assert energy is not None
    # FCI on the full A space is a fixed point: the density fed back equals the
    # one that produced it, so cycle 2 already matches cycle 1 and the loop stops.
    assert any("converged" in line for line in logs)


def test_converge_on_energy_stops_when_density_still_moving():
    """``converge_on='energy'`` stops on |ΔE| alone, ignoring a loose rho_tol.

    With an unreachably tight ``rho_tol`` the density criterion can never be
    met, so ``energy_and_density`` would run to ``max_cycles``; ``energy`` stops
    as soon as the total energy settles.  On the FCI fixed point |ΔE| hits 0 at
    cycle 2, so the energy criterion fires immediately.
    """
    logs_e: list[str] = []
    wf_e = _workflow(solver="fci", max_cycles=8, e_tol=1e-6, rho_tol=1e-30, converge_on="energy")
    adapter_e = _build_adapter()
    adapter_e.run_low_level()
    wf_e._run_outer_loop(adapter_e, FCISolver(), None, rank=0, log=logs_e.append)
    assert any("converged (|ΔE|)" in line for line in logs_e)

    # Same setup but requiring density too: the impossible rho_tol prevents the
    # early stop, so the loop exhausts max_cycles.
    logs_ed: list[str] = []
    wf_ed = _workflow(
        solver="fci",
        max_cycles=3,
        e_tol=1e-6,
        rho_tol=1e-30,
        converge_on="energy_and_density",
    )
    adapter_ed = _build_adapter()
    adapter_ed.run_low_level()
    wf_ed._run_outer_loop(adapter_ed, FCISolver(), None, rank=0, log=logs_ed.append)
    assert any("reached max_cycles" in line for line in logs_ed)
    assert not any("converged" in line for line in logs_ed)


def test_mix_alpha_damps_the_fed_back_density():
    """``mix_alpha`` linearly mixes the fed-back density across cycles.

    The mock records every ``dmab_in`` it is handed.  With ``mix_alpha=1.0`` the
    loop feeds the raw new total density (undamped); with ``mix_alpha=0.5`` the
    second cycle's fed density must be the average of the unmixed new density and
    the previous cycle's fed density -- i.e. strictly between them.
    """
    seen: list[np.ndarray] = []

    class _Recording(_FeedbackMockEmbedding):
        def construct_embedded_fock(self, dmab_in=None):
            if dmab_in is not None:
                seen.append(np.asarray(dmab_in[0, 0]).copy())
            return super().construct_embedded_fock(dmab_in=dmab_in)

    def build(alpha):
        mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", basis="sto-3g")
        mock = _Recording(mol, mu=1.0e6, n_occ_a=1)
        adapter = ProjectionEmbeddingAdapter(mock, PySCFIntegrals(mol.RHF()), mu=1.0e6)
        adapter.run_low_level()
        seen.clear()
        wf = _workflow(
            solver="fci", max_cycles=3, mix_alpha=alpha, e_tol=1e-30, rho_tol=1e-30
        )  # never stop early -> full 3 cycles
        wf._run_outer_loop(adapter, FCISolver(), None, rank=0, log=lambda *a, **k: None)
        return [m.copy() for m in seen]

    undamped = build(1.0)
    damped = build(0.5)

    # First fed density is identical (no previous to mix with yet).
    np.testing.assert_allclose(damped[0], undamped[0], atol=1e-12)
    # Second differs: damped = 0.5*new + 0.5*prev, strictly between the two.
    assert not np.allclose(damped[1], undamped[1], atol=1e-9)
    expected = 0.5 * undamped[1] + 0.5 * damped[0]
    np.testing.assert_allclose(damped[1], expected, atol=1e-10)


def test_reseed_policy_advances_fci_noop():
    """FCI has no seed, so _maybe_reseed is a no-op regardless of reseed_sqd."""
    adapter = _build_adapter()
    adapter.run_low_level()
    wf = _workflow(solver="fci", max_cycles=2, reseed_sqd=True)
    solver = FCISolver()
    # Should not raise nor grow a spurious attribute.
    wf._maybe_reseed(solver, cycle=1)
    assert not hasattr(solver, "seed")


def test_reseed_advances_seed_when_solver_has_one():
    """_maybe_reseed bumps a stochastic solver's seed under reseed_sqd."""
    wf = _workflow(solver="fci", reseed_sqd=True)

    class _StubSeeded:
        seed = 100

    stub = _StubSeeded()
    wf._maybe_reseed(stub, cycle=0)  # cycle 0 never changes the seed
    assert stub.seed == 100
    wf._maybe_reseed(stub, cycle=3)
    assert stub.seed == 103

    # With reseeding off, the seed is held (subspace carried over).
    wf_off = _workflow(solver="fci", reseed_sqd=False)
    stub2 = _StubSeeded()
    wf_off._maybe_reseed(stub2, cycle=3)
    assert stub2.seed == 100
