# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Optional IBM/sci-solvers subspace diagonalizers.

The factory-dispatch test only needs torch; the end-to-end test is skipped
unless the optional ``sci-solvers`` extra is installed:

    uv sync --extra sci-solvers
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.circuit_run.base import MockSampler
from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.solvers import FCISolver, SQDSolver
from embasi_qiskit_integration.sqd.driver import resolve_sci_solver


def test_solver_listing_puts_torch_sci_after_the_builtin_baseline():
    from embasi_qiskit_integration.sqd.driver import (
        QISKIT_ADDON_SOLVER,
        _available_solvers,
    )

    names = _available_solvers(("arpack", "sbd", "torch_sci"))
    assert names == (QISKIT_ADDON_SOLVER, "torch_sci", "arpack", "sbd")


def test_resolve_sci_solvers_through_the_factory(monkeypatch):
    pytest.importorskip("torch")
    created = {}
    received = []

    def fake_callable(*args, **kwargs):
        received.append(kwargs)
        return []

    def fake_create_solver(name, **kwargs):
        created[name] = kwargs
        return SimpleNamespace(sci_solver_callable=fake_callable)

    monkeypatch.setitem(
        sys.modules,
        "sci_solvers.energy_solvers.factory",
        SimpleNamespace(
            AVAILABLE_SOLVERS=("arpack", "torch_sci"),
            create_solver=fake_create_solver,
        ),
    )

    spin_pinned = resolve_sci_solver("torch_sci", 0.0, {"device": "cpu"})
    assert spin_pinned([], None, None, 1, (1, 1)) == []
    assert received[-1] == {"spin_sq": 0.0}
    assert created["torch_sci"] == {
        "device": "cpu",
        "apply_ab_projection": False,
        "davidson_tolerance": 1e-8,
        "symmetrize_spin": False,
        "compute_rdm2": True,
    }

    # Without an explicit device the resolver picks CUDA, MPS, or CPU; the
    # other torch-only options must not leak into another solver's constructor.
    assert resolve_sci_solver("arpack", None, {"device": "cpu"}) is fake_callable
    assert created["arpack"] == {"symmetrize_spin": False, "compute_rdm2": True}

    assert resolve_sci_solver("torch_sci", None, None) is fake_callable
    auto = created["torch_sci"]
    assert auto["device"] in {"cuda", "mps", "cpu"}
    if auto["device"] == "mps":
        import torch

        assert auto["dtype"] is torch.float32
        assert auto["davidson_tolerance"] == 1e-5
    else:
        assert auto["davidson_tolerance"] == 1e-8

    with pytest.raises(ValueError, match="unknown sci_solver"):
        resolve_sci_solver("nope", None, None)


def test_torch_sci_gdb_pair_mode_is_rejected_with_guidance():
    """The gdb engine's flat pair output is not usable by the addon's SQD loop."""
    pytest.importorskip("sci_solvers")
    with pytest.raises(NotImplementedError, match="gdb"):
        resolve_sci_solver("torch_sci", None, {"mode": "gdb", "device": "cpu"})


def test_sqdsolver_torch_sci_matches_fci(data_dir, monkeypatch):
    pytest.importorskip("torch_sci_solver")
    ham = fcidump.read(data_dir / "n2_8o10e.fcidump")
    sampler = MockSampler(data_dir / "mock_counts.json")
    fci = FCISolver().solve(ham)
    sqd = SQDSolver(
        sampler,
        shots=100_000,
        seed=24,
        sci_solver="torch_sci",
        torch_sci_kwargs={"device": "cpu", "apply_ab_projection": False},
    ).solve(ham)
    assert abs(sqd.energy - fci.energy) <= 2e-3
    assert sqd.diagnostics["sci_solver"] == "torch-sci-solver"
    assert sqd.rdm1.shape == (ham.norb, ham.norb)
    assert abs(np.trace(sqd.rdm1) - sum(ham.nelec)) < 1e-6
