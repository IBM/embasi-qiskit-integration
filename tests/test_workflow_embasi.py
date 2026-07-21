# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""EmbASI workflow glue.

Two kinds of tests:
- ``FakeEmbASI`` stub tests (UNMARKED): exercise the extraction + energy-assembly
  + two-process handoff logic in CI without EmbASI/FHI-aims.
- ``@pytest.mark.embasi`` tests: run only with real EmbASI (EMBASI_AVAILABLE=1).
"""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.contract import SolverResult
from embasi_qiskit_integration.hamiltonian.extract import (
    embedded_hamiltonian_from_embasi,
)
from embasi_qiskit_integration.solvers import FCISolver
from embasi_qiskit_integration.workflow import (
    embedding_energy,
    feedback_rdm1,
    solve_embedding_in_process,
    solve_embedding_two_process,
)


class FakeEmbASI:
    """Minimal stand-in exposing the attributes extract.py reads.

    Carries random-but-consistent matrices so the glue logic (shapes, folding,
    solving, energy assembly) is fully CI-covered without EmbASI.
    """

    def __init__(self, nao: int, n_active: int, nelec_active: tuple[int, int], seed: int = 0):
        rng = np.random.default_rng(seed)
        # Orthonormal-ish coefficient matrix (AO x MO).
        q, _ = np.linalg.qr(rng.standard_normal((nao, nao)))
        self.mo_coeff = q
        # Symmetric one-body operators.
        k = rng.standard_normal((nao, nao))
        self._ham_kin = k + k.T
        f = rng.standard_normal((nao, nao))
        self._fock_embedding = f + f.T
        self.e_core = -12.3456
        self.n_active_alpha, self.n_active_beta = nelec_active
        self._n_active = n_active
        self.density_matrix_in = None  # settable; used by feedback_rdm1

    # Mirror the real EmbASI property names read by extract.py.
    @property
    def hamiltonian_kinetic(self):
        return self._ham_kin

    @property
    def fock_embedding_matrix(self):
        return self._fock_embedding


def test_extract_produces_valid_hamiltonian():
    emb = FakeEmbASI(nao=8, n_active=4, nelec_active=(2, 2), seed=1)
    ham = embedded_hamiltonian_from_embasi(emb, active_orbitals=[0, 1, 2, 3])
    assert ham.norb == 4
    assert ham.nelec == (2, 2)
    assert ham.h1.shape == (4, 4)
    assert np.allclose(ham.h1, ham.h1.T, atol=1e-8)  # h1 hermitian by construction
    assert ham.meta["source"] == "embasi"


def test_solve_embedding_in_process_runs_solver():
    emb = FakeEmbASI(nao=6, n_active=4, nelec_active=(2, 2), seed=2)
    result = solve_embedding_in_process(emb, [0, 1, 2, 3], FCISolver())
    assert isinstance(result, SolverResult)
    assert result.rdm1.shape == (4, 4)


def test_embedding_energy_assembly():
    res = SolverResult(energy=-5.0, rdm1=np.eye(4))
    e = embedding_energy(res, e_low_total=-100.0, e_low_A=-4.0, correction=0.1)
    # E = E_low(total) - E_low(A) + E_high(A) + correction
    assert e.total == pytest.approx(-100.0 - (-4.0) + (-5.0) + 0.1)
    assert e.e_high_A == -5.0


def test_feedback_rdm1_writes_density():
    emb = FakeEmbASI(nao=4, n_active=2, nelec_active=(1, 1), seed=3)
    rdm1 = np.array([[2.0, 0.0], [0.0, 0.0]])
    feedback_rdm1(emb, rdm1)
    assert np.allclose(emb.density_matrix_in, rdm1)


def test_two_process_handoff_reads_result(tmp_path):
    """Process A writes the job; a stubbed process B writes result.npz; A reads it."""
    from embasi_qiskit_integration import ipc

    emb = FakeEmbASI(nao=6, n_active=4, nelec_active=(2, 2), seed=4)

    # Simulate process B having already produced a result, so the wait returns
    # immediately (write job first, then the result).
    ham = embedded_hamiltonian_from_embasi(emb, [0, 1, 2, 3])
    ipc.write_job(ham, tmp_path)
    ipc.write_result(SolverResult(energy=-7.0, rdm1=np.eye(4)), tmp_path)

    result = solve_embedding_two_process(emb, [0, 1, 2, 3], tmp_path, timeout=5.0)
    assert result.energy == -7.0


def test_two_process_handoff_raises_on_error(tmp_path):
    from embasi_qiskit_integration import ipc

    emb = FakeEmbASI(nao=6, n_active=4, nelec_active=(2, 2), seed=5)
    ipc.write_error(RuntimeError("boom"), tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        solve_embedding_two_process(emb, [0, 1, 2, 3], tmp_path, timeout=5.0)


@pytest.mark.embasi
def test_real_embasi_extraction():
    """Smoke test against real EmbASI (skipped unless EMBASI_AVAILABLE=1)."""
    pytest.skip("real EmbASI extraction requires a completed FHI-aims embedding run")
