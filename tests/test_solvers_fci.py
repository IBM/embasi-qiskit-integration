# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""FCISolver on the frozen N2 data reproduces the stored reference."""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.solvers import FCISolver, cheap_ccsd_t2


@pytest.fixture
def n2_ham(data_dir):
    return fcidump.read(data_dir / "n2_8o10e.fcidump")


def test_fci_reproduces_reference(n2_ham):
    ref = n2_ham.meta["reference_fci_energy"]
    res = FCISolver().solve(n2_ham)
    assert abs(res.energy - ref) < 1e-8


def test_rdm1_trace_equals_nelec(n2_ham):
    res = FCISolver().solve(n2_ham)
    trace = np.trace(res.rdm1)
    assert abs(trace - sum(n2_ham.nelec)) < 1e-8


def test_rdm1_symmetric(n2_ham):
    res = FCISolver().solve(n2_ham)
    assert np.allclose(res.rdm1, res.rdm1.T, atol=1e-8)


def test_diagnostics_present(n2_ham):
    res = FCISolver().solve(n2_ham)
    assert res.diagnostics["solver"] == "pyscf-fci"


def test_cheap_ccsd_t2_shape(n2_ham):
    t2 = cheap_ccsd_t2(n2_ham)
    # Closed-shell RCCSD t2: (nocc, nocc, nvir, nvir) with nocc = n_alpha.
    nocc = n2_ham.nelec[0]
    nvir = n2_ham.norb - nocc
    assert t2.shape == (nocc, nocc, nvir, nvir)
