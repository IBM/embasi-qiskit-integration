# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""FCIDUMP + npz sidecar round-trip fidelity."""

from __future__ import annotations

import numpy as np

from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.hamiltonian import fcidump


def _random_ham(norb: int, nelec: tuple[int, int], rng: np.random.Generator, meta=None):
    a = rng.standard_normal((norb, norb))
    h1 = a + a.T
    b = rng.standard_normal((norb, norb, norb, norb))
    b = b + b.transpose(1, 0, 2, 3)
    b = b + b.transpose(0, 1, 3, 2)
    b = b + b.transpose(2, 3, 0, 1)
    return EmbeddedHamiltonian(h1=h1, h2=b, e_core=-3.14159, nelec=nelec, meta=meta or {})


def test_roundtrip_exact(tmp_path, rng):
    ham = _random_ham(4, (3, 2), rng, meta={"sha": "deadbeef", "note": "n2-like"})
    path = tmp_path / "job.fcidump"
    fcidump.write(ham, path)
    back = fcidump.read(path)

    assert np.max(np.abs(back.h1 - ham.h1)) < 1e-12
    assert np.max(np.abs(back.h2 - ham.h2)) < 1e-12
    assert abs(back.e_core - ham.e_core) < 1e-12
    assert back.nelec == ham.nelec
    assert back.meta == ham.meta


def test_sidecar_written(tmp_path, rng):
    ham = _random_ham(3, (2, 1), rng)
    path = tmp_path / "job.fcidump"
    fcidump.write(ham, path)
    assert (tmp_path / "job.npz").exists()


def test_read_without_sidecar(tmp_path, rng):
    """Without the sidecar, nelec/e_core are recovered from the header."""
    ham = _random_ham(3, (2, 1), rng)
    path = tmp_path / "job.fcidump"
    fcidump.write(ham, path)
    (tmp_path / "job.npz").unlink()

    back = fcidump.read(path)
    assert back.nelec == (2, 1)
    assert abs(back.e_core - ham.e_core) < 1e-12
    assert back.meta == {}


def test_closed_shell_roundtrip(tmp_path, rng):
    ham = _random_ham(5, (3, 3), rng, meta={"reference_energy": -108.1})
    path = tmp_path / "cs.fcidump"
    fcidump.write(ham, path)
    back = fcidump.read(path)
    assert back.nelec == (3, 3)
    assert np.max(np.abs(back.h2 - ham.h2)) < 1e-12
    assert back.meta["reference_energy"] == -108.1
