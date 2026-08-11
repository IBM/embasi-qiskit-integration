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

from qiskit import QuantumCircuit

from embasi_qiskit_integration.circuit_run.runtime import RuntimeSampler


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


# ----- noise-aware layout selection ------------------------------------------ #


def test_default_run_records_hardware_provenance():
    """Every run records the layout it used and the backend's reported noise.

    Even with the measurement step off, a caller can tell which physical qubits the
    counts came from and what the device claimed about them.
    """
    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.circuit_run.runtime import RuntimeSampler

    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_all()

    sampler = RuntimeSampler(backend="FakeManilaV2")
    counts = sampler.run([circuit], 256)

    assert sum(counts[0].values()) == 256
    characterisation = sampler.hardware_characterisation
    # Not measured, since characterisation defaults off.
    assert characterisation["readout_characterisation"] is None
    # But the layout and the reported noise are recorded.
    assert len(characterisation["layout_virtual_to_physical"]) == 3
    assert characterisation["readout_info"]["readout_error"]
    assert characterisation["readout_info"]["two_qubit_gate"] in ("ecr", "cz", "cx")


def test_characterisation_is_refused_on_a_simulated_backend(caplog):
    """A Fake device's readout error is its noise model, not a measurement.

    Pruning on it would dress a fixed configuration up as a live characterisation
    and could reject qubits on a device that has no real error at all. The run must
    fall back to the default layout and still produce counts.
    """
    import logging

    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.circuit_run.runtime import RuntimeSampler

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    sampler = RuntimeSampler(backend="FakeManilaV2", enable_readout_characterisation=True)
    with caplog.at_level(logging.WARNING, logger="embasi_qiskit_integration.circuit_run.runtime"):
        counts = sampler.run([circuit], 128)

    assert sum(counts[0].values()) == 128
    assert sampler.hardware_characterisation["readout_characterisation"] is None
    messages = [record.getMessage() for record in caplog.records]
    assert any("simulated backend" in message for message in messages), messages


def test_characterisation_settings_reach_the_sampler():
    """The characterisation defaults, exposed and forwarded unchanged."""
    from embasi_qiskit_integration.circuit_run.runtime import RuntimeSampler

    default = RuntimeSampler()
    assert default.enable_readout_characterisation is False
    assert default.readout_error_threshold == 0.03
    assert default.n_rand_twirl == 300
    assert default.n_shots_per_twirl == 25
    assert default.hot_coupler_ps is False

    tuned = RuntimeSampler(
        enable_readout_characterisation=True,
        readout_error_threshold=0.05,
        n_rand_twirl=64,
        n_shots_per_twirl=10,
        hot_coupler_ps=True,
    )
    assert tuned.enable_readout_characterisation is True
    assert tuned.readout_error_threshold == 0.05
    assert tuned.n_rand_twirl == 64
    assert tuned.n_shots_per_twirl == 10
    assert tuned.hot_coupler_ps is True


def test_build_sampler_rejects_characterisation_for_local_samplers(data_dir):
    """Only a real device can be characterised, so asking elsewhere must raise."""
    from embasi_qiskit_integration.circuit_run import build_sampler

    with pytest.raises(ValueError, match="cannot characterise readout"):
        build_sampler("aer", enable_readout_characterisation=True)

    with pytest.raises(ValueError, match="cannot characterise readout"):
        build_sampler(
            "mock",
            counts=str(data_dir / "mock_counts.json"),
            enable_readout_characterisation=True,
        )

    # Disabled is the default and must stay accepted everywhere.
    assert build_sampler("aer").default_shots > 0


def test_pinned_pass_manager_places_circuits_on_the_requested_chain():
    """The transpile half: a chosen layout must actually be used.

    Selecting a good chain is pointless if the pass manager then relocates the
    circuit, so this pins that ``build_pinned_pass_manager`` honours the layout and
    respects the pruned coupling map.
    """
    from qiskit import QuantumCircuit
    from qiskit.transpiler import CouplingMap
    from qiskit_ibm_runtime import fake_provider

    from embasi_qiskit_integration.circuit_run.backend import (
        build_pinned_pass_manager,
        virtual_to_physical,
    )

    backend = fake_provider.FakeManilaV2()
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.measure_all()

    chain = [2, 3, 4]
    pass_manager = build_pinned_pass_manager(
        backend, CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4]]), chain
    )
    isa = pass_manager.run(circuit)

    assert sorted(virtual_to_physical(isa).values()) == chain


def test_read_noise_from_backend_is_json_serialisable():
    """The noise dict is persisted next to the counts, so it must serialise.

    Missing values are absent rather than ``None``, so a consumer never has to
    distinguish "unknown" from "known but null".
    """
    import json

    from qiskit_ibm_runtime import fake_provider

    from embasi_qiskit_integration.circuit_run.backend import read_noise_from_backend

    noise = read_noise_from_backend(fake_provider.FakeManilaV2())
    json.dumps(noise)  # must not raise

    assert len(noise["readout_error"]) == 5
    assert all(isinstance(value, float) for value in noise["readout_error"].values())
    assert all(value is not None for value in noise["p01"].values())
    assert "-" in next(iter(noise["two_qubit_errors"]))


def test_read_noise_tolerates_a_backend_without_properties():
    """A backend exposing no calibration must yield empty dicts, not raise."""
    from embasi_qiskit_integration.circuit_run.backend import read_noise_from_backend

    class _Bare:
        num_qubits = 3
        target = None

    noise = read_noise_from_backend(_Bare())
    assert noise == {
        "readout_error": {},
        "p01": {},
        "p10": {},
        "two_qubit_gate": None,
        "two_qubit_errors": {},
    }


def test_edges_along_layout_filters_to_the_used_qubits():
    from embasi_qiskit_integration.circuit_run.backend import edges_along_layout

    noise = {"two_qubit_errors": {"0-1": 0.01, "1-2": 0.02, "3-4": 0.03, "malformed": 0.9}}
    # Layout uses physical qubits 0, 1, 2 only.
    selected = edges_along_layout({0: 0, 1: 1, 2: 2}, noise)

    assert selected == {"0-1": 0.01, "1-2": 0.02}
