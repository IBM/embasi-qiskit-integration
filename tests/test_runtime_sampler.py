# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""RuntimeSampler flow, driven offline via a fake backend (no credentials).

These exercise the transpile-to-ISA + submit path using
``qiskit_ibm_runtime.fake_provider`` backends, so no network or IBM Quantum
account is needed. Skipped if the ``hardware`` extra is not installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit_ibm_runtime", reason="needs the 'hardware' extra")

from qiskit import QuantumCircuit  # noqa: E402

from embasi_qiskit_integration.sampling.runtime import RuntimeSampler  # noqa: E402


class _FakeService:
    """Minimal QiskitRuntimeService stand-in: hands back a fixed backend."""

    def __init__(self, backend):
        self._backend = backend
        self.least_busy_calls = 0
        self.backend_calls: list[str] = []

    def least_busy(self, min_num_qubits=None, **kwargs):
        self.least_busy_calls += 1
        return self._backend

    def backend(self, name, **kwargs):
        self.backend_calls.append(name)
        return self._backend


@pytest.fixture
def fake_backend():
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    return FakeManilaV2()


@pytest.fixture
def bell():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def test_sample_transpiles_and_runs_on_injected_backend(fake_backend, bell):
    sampler = RuntimeSampler(backend_obj=fake_backend)
    counts = sampler.sample(bell, shots=512)
    assert sum(counts.values()) == 512
    # Bell state on a noisy fake backend: dominated by 00/11.
    assert counts.get("00", 0) + counts.get("11", 0) > 512 * 0.7


def test_least_busy_used_when_no_backend_named(fake_backend, bell):
    service = _FakeService(fake_backend)
    sampler = RuntimeSampler(service=service)
    sampler.sample(bell, shots=256)
    assert service.least_busy_calls == 1
    assert service.backend_calls == []


def test_named_backend_looked_up(fake_backend, bell):
    service = _FakeService(fake_backend)
    sampler = RuntimeSampler(backend="ibm_fake", service=service)
    sampler.sample(bell, shots=256)
    assert service.backend_calls == ["ibm_fake"]
    assert service.least_busy_calls == 0


def test_backend_resolution_cached(fake_backend, bell):
    service = _FakeService(fake_backend)
    sampler = RuntimeSampler(service=service)
    sampler.sample(bell, shots=128)
    sampler.sample(bell, shots=128)
    # least_busy is only queried once; the backend is cached.
    assert service.least_busy_calls == 1
