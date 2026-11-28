# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Spin-resolved SQD diagonalizer: exactness against ``direct_uhf``.

These tests exercise the 2n spin-orbital encoding against PySCF's unrestricted
FCI on the same ``(h1a, h1b, h2_spin)`` tensors, both directly and through the
addon's configuration-recovery loop.  They skip when the optional
``sci-solvers`` extra is absent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pyscf.fci import cistring, direct_uhf

from embasi_qiskit_integration.circuit_run.base import MockSampler
from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.solvers import SQDSolver
from embasi_qiskit_integration.sqd.spin_orbital import (
    energy_from_spin_blocks,
    make_torch_spin_orbital_solver,
)

pytest.importorskip("torch_sci_solver")


def _random_hamiltonian(rng, norb, nelec):
    def sym():
        a = rng.normal(size=(norb, norb))
        return 0.5 * (a + a.T)

    def same_spin():
        a = rng.normal(size=(norb, norb, 3))
        a = 0.5 * (a + a.transpose(1, 0, 2))
        return np.einsum("pqm,rsm->pqrs", a, a)

    def mixed():
        a = rng.normal(size=(norb, norb, 3))
        a = 0.5 * (a + a.transpose(1, 0, 2))
        b = rng.normal(size=(norb, norb, 3))
        b = 0.5 * (b + b.transpose(1, 0, 2))
        return np.einsum("pqm,rsm->pqrs", a, b)

    h1a, h1b = sym(), sym()
    aa, ab, bb = same_spin(), mixed(), same_spin()
    return (
        EmbeddedHamiltonian(
            h1=0.5 * (h1a + h1b),
            h2=aa,
            e_core=0.0,
            nelec=nelec,
            h1a=h1a,
            h1b=h1b,
            h2_spin=(aa, ab, bb),
        ),
        (h1a, h1b),
        (aa, ab, bb),
    )


def _strings(norb, nelec):
    return (
        np.asarray(cistring.gen_strings4orblist(range(norb), nelec[0])),
        np.asarray(cistring.gen_strings4orblist(range(norb), nelec[1])),
    )


def _reference(ham):
    return direct_uhf.kernel(
        (np.asarray(ham.h1a), np.asarray(ham.h1b)), ham.h2_spin, ham.norb, ham.nelec
    )


def _full_counts(norb, nelec):
    a_all, b_all = _strings(norb, nelec)
    counts = {}
    for ai in a_all:
        for bi in b_all:
            alpha = "".join("1" if (int(ai) >> (norb - 1 - i)) & 1 else "0" for i in range(norb))
            beta = "".join("1" if (int(bi) >> (norb - 1 - i)) & 1 else "0" for i in range(norb))
            counts[beta + alpha] = 1
    return counts


def test_torch_spinless_full_space_matches_direct_uhf():
    rng = np.random.default_rng(5)
    for norb, nelec in [(2, (1, 1)), (3, (2, 1)), (4, (3, 2))]:
        ham, _h1, _eri = _random_hamiltonian(rng, norb, nelec)
        a, b = _strings(norb, nelec)
        solver = make_torch_spin_orbital_solver(ham, torch_sci_kwargs={"device": "cpu"})
        res = solver([(a, b)], None, None, norb, nelec)[0]
        e_ref, _ci = _reference(ham)
        assert res.energy == pytest.approx(e_ref, abs=1e-7)
        r1s = res.sci_state.rdm(rank=1, spin_summed=False)
        assert np.trace(r1s[0]) == pytest.approx(nelec[0], abs=1e-7)
        assert np.trace(r1s[1]) == pytest.approx(nelec[1], abs=1e-7)


def test_torch_spinless_strict_subspace_matches_projected_reference():
    rng = np.random.default_rng(99)
    norb, nelec = 4, (2, 2)
    ham, _h1, _eri = _random_hamiltonian(rng, norb, nelec)
    a_all, b_all = _strings(norb, nelec)
    sa = np.sort(rng.choice(a_all, size=3, replace=False))
    sb = np.sort(rng.choice(b_all, size=2, replace=False))

    # Project the exact UHF Hamiltonian onto the same Cartesian product.
    h2e = direct_uhf.absorb_h1e(
        (np.asarray(ham.h1a), np.asarray(ham.h1b)), ham.h2_spin, norb, nelec, 0.5
    )
    na, nb = len(a_all), len(b_all)
    H = np.empty((na * nb, na * nb))
    v = np.zeros((na, nb))
    for j in range(na * nb):
        v.flat[j] = 1.0
        H[:, j] = np.asarray(direct_uhf.contract_2e(h2e, v.copy(), norb, nelec)).ravel()
        v.flat[j] = 0.0
    apos = {int(s): i for i, s in enumerate(a_all)}
    bpos = {int(s): i for i, s in enumerate(b_all)}
    idx = np.array([apos[int(a)] * nb + bpos[int(b)] for b in sb for a in sa])
    e_ref = float(np.linalg.eigvalsh(0.5 * (H[np.ix_(idx, idx)] + H[np.ix_(idx, idx)].T))[0])

    solver = make_torch_spin_orbital_solver(ham, torch_sci_kwargs={"device": "cpu"})
    res = solver([(sa, sb)], None, None, norb, nelec)[0]
    assert res.energy == pytest.approx(e_ref, abs=1e-7)


def test_energy_from_spin_blocks_matches_direct_uhf_rdms():
    rng = np.random.default_rng(7)
    ham, _h1, _eri = _random_hamiltonian(rng, 3, (2, 1))
    e_ref, ci = _reference(ham)
    r1s, r2s = direct_uhf.make_rdm12s(ci, 3, ham.nelec)
    assert energy_from_spin_blocks(
        np.asarray(ham.h1a), np.asarray(ham.h1b), ham.h2_spin, r1s, r2s
    ) == pytest.approx(e_ref, abs=1e-10)


def test_sqd_loop_uses_spin_resolved_hamiltonian():
    from embasi_qiskit_integration.sqd.driver import run_sqd

    rng = np.random.default_rng(321)
    norb, nelec = 3, (2, 1)
    ham, _h1, _eri = _random_hamiltonian(rng, norb, nelec)
    e_ref, _ci = _reference(ham)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_sqd(
            ham,
            _full_counts(norb, nelec),
            sci_solver="torch_sci",
            samples_per_batch=50,
            num_batches=5,
            max_iterations=5,
            seed=3,
            symmetrize_spin=False,
            torch_sci_kwargs={"device": "cpu"},
        )
    assert res.diagnostics["spin_resolved_subspace"] is True
    assert res.diagnostics["spin_solver_backend"] == "torch_sci_spin"
    assert res.energy - ham.e_core == pytest.approx(e_ref, abs=1e-6)
    assert np.trace(res.rdm1) == pytest.approx(sum(nelec), abs=1e-6)


def test_sqdsolver_workflow_auto_selects_spin_resolved(tmp_path):
    import json

    rng = np.random.default_rng(17)
    norb, nelec = 3, (2, 1)
    ham, _h1, _eri = _random_hamiltonian(rng, norb, nelec)
    counts_path = tmp_path / "counts.json"
    counts_path.write_text(json.dumps(_full_counts(norb, nelec)))
    result = SQDSolver(
        MockSampler(counts_path),
        shots=100_000,
        seed=3,
        sci_solver="torch_sci",
        torch_sci_kwargs={"device": "cpu"},
    ).solve(ham)
    e_ref, _ci = _reference(ham)
    assert result.diagnostics["spin_resolved_subspace"] is True
    assert result.diagnostics["sci_solver"] == "torch-sci-spinless"
    assert result.energy - ham.e_core == pytest.approx(e_ref, abs=1e-6)


def test_unsupported_solver_is_blocked_for_spin_dependent_hamiltonian():
    from embasi_qiskit_integration.sqd.driver import run_sqd

    rng = np.random.default_rng(4)
    ham, _h1, _eri = _random_hamiltonian(rng, 3, (2, 1))
    with pytest.raises(NotImplementedError, match="does not support a spin-dependent"):
        run_sqd(ham, _full_counts(3, (2, 1)), sci_solver="qiskit_addon_sqd")


def test_spin_sq_is_rejected():
    rng = np.random.default_rng(1)
    ham, _h1, _eri = _random_hamiltonian(rng, 3, (2, 1))
    with pytest.raises(NotImplementedError, match="spin_sq"):
        make_torch_spin_orbital_solver(ham, spin_sq=0.75)


def test_spin_solver_rejects_restricted_hamiltonian():
    rng = np.random.default_rng(2)
    ham, _h1, _eri = _random_hamiltonian(rng, 3, (2, 1))
    restricted = EmbeddedHamiltonian(h1=ham.h1, h2=ham.h2, e_core=ham.e_core, nelec=ham.nelec)
    with pytest.raises(ValueError, match="spin-dependent"):
        make_torch_spin_orbital_solver(restricted)
