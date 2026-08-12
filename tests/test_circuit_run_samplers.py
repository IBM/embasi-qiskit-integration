# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Samplers: MockSampler determinism, Aer correctness, and the batch seam."""

from __future__ import annotations

import pytest

from embasi_qiskit_integration.circuit_run.base import BitstringSampler, MockSampler


def test_mock_sampler_replays_counts(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    counts = sampler.sample(circuit=None, shots=2000)
    assert counts == {"00": 983, "11": 1017}


def test_mock_sampler_deterministic(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    a = sampler.sample(circuit=None, shots=2000, seed=1)
    b = sampler.sample(circuit=None, shots=2000, seed=999)
    assert a == b  # seed and circuit are ignored — pure replay


def test_mock_sampler_rescales_preserving_total(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    counts = sampler.sample(circuit=None, shots=1000)
    assert sum(counts.values()) == 1000
    # proportions preserved to rounding
    assert abs(counts["11"] / 1000 - 1017 / 2000) < 1e-3


def test_mock_sampler_satisfies_protocol(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    assert isinstance(sampler, BitstringSampler)


# ----- the batch seam: run() and its single-circuit shim ------------------- #


def test_mock_sampler_run_returns_one_dict_per_circuit(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    out = sampler.run([None, None, None], shots=2000)
    assert len(out) == 3
    assert all(counts == {"00": 983, "11": 1017} for counts in out)


def test_mock_sampler_run_empty_batch(data_dir):
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    assert sampler.run([], shots=2000) == []


def test_sample_is_the_first_element_of_run(data_dir):
    """The single-circuit shim must agree with the batch method by construction."""
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    assert sampler.sample(circuit=None, shots=1500) == sampler.run([None], shots=1500)[0]


def test_mock_sampler_declares_it_needs_no_circuit(data_dir):
    """Lets callers skip circuit construction entirely for deterministic replay."""
    sampler = MockSampler(data_dir / "mock_counts_bell.json")
    assert sampler.requires_circuit is False


def test_merge_counts_sums_per_bitstring():
    from embasi_qiskit_integration.circuit_run import merge_counts

    merged = merge_counts([{"00": 3, "11": 2}, {"00": 1, "01": 5}, {}])
    assert merged == {"00": 4, "11": 2, "01": 5}
    assert sum(merged.values()) == 11


def test_merge_counts_of_nothing_is_empty():
    from embasi_qiskit_integration.circuit_run import merge_counts

    assert merge_counts([]) == {}


@pytest.mark.slow
def test_aer_run_samples_every_circuit_in_the_batch():
    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.circuit_run.aer import AerSampler

    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    zero = QuantumCircuit(2)
    zero.measure_all()

    out = AerSampler().run([bell, zero], shots=1000, seed=5)
    assert len(out) == 2
    assert set(out[0]) <= {"00", "11"}
    assert out[1] == {"00": 1000}  # the all-zero circuit is deterministic
    assert all(sum(counts.values()) == 1000 for counts in out)


@pytest.mark.slow
def test_aer_bell_circuit_only_00_and_11():
    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.circuit_run.aer import AerSampler

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    counts = AerSampler().sample(qc, shots=4000, seed=42)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 4000
    # both outcomes appear with roughly equal weight
    assert counts.get("00", 0) > 1000
    assert counts.get("11", 0) > 1000


@pytest.mark.slow
def test_aer_sampler_deterministic_with_seed():
    from qiskit import QuantumCircuit

    from embasi_qiskit_integration.circuit_run.aer import AerSampler

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    a = AerSampler().sample(qc, shots=2000, seed=7)
    b = AerSampler().sample(qc, shots=2000, seed=7)
    assert a == b


# ----- sampler options reach the runtime sampler ------------------------------ #


def test_build_sampler_forwards_options_to_runtime():
    """``options`` must reach ``RuntimeSampler``, which forwards them to SamplerV2.

    Regression: ``build_sampler`` had no ``options`` parameter, so the measurement
    twirling that ``RuntimeSampler``'s own docstring calls for on hardware was
    unreachable through the supported entry points (the CLI included) -- the
    mitigation was documented but not wired.
    """
    from embasi_qiskit_integration.circuit_run import build_sampler

    options = {"twirling": {"enable_measure": True}}
    sampler = build_sampler("runtime", options=options)
    assert sampler.options == options

    # Omitting them stays None, so SamplerV2 keeps its own defaults.
    assert build_sampler("runtime").options is None


def test_build_sampler_rejects_options_for_samplers_that_cannot_use_them(data_dir):
    """Options are IBM Runtime specific; accepting them elsewhere would mislead.

    Silently dropping a twirling request on the Aer or mock sampler would let a
    caller believe an error-suppression setting is active when nothing applies it.
    """
    from embasi_qiskit_integration.circuit_run import build_sampler

    with pytest.raises(ValueError, match="takes no sampler options"):
        build_sampler("aer", options={"twirling": {"enable_measure": True}})

    with pytest.raises(ValueError, match="takes no sampler options"):
        build_sampler(
            "mock",
            counts=str(data_dir / "mock_counts.json"),
            options={"twirling": {"enable_measure": True}},
        )


def test_cli_option_flags_build_valid_sampler_options():
    """The CLI flags must produce a dict ``SamplerOptions`` actually accepts.

    A malformed key would only surface at hardware submission, after the queue
    wait, so it is validated here against the real options model.
    """
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.options import SamplerOptions

    from embasi_qiskit_integration.cli import SolveCommand

    built = SolveCommand(directory="/tmp/unused")._sampler_options()
    # Measurement twirling defaults on: readout error is the channel SQD is most
    # sensitive to, and SamplerV2 leaves it off.
    assert built["twirling"]["enable_measure"] is True
    assert built["dynamical_decoupling"]["enable"] is False
    options = SamplerOptions(**built)
    assert options.twirling.enable_measure is True

    off = SolveCommand(directory="/tmp/unused", measure_twirling=False)._sampler_options()
    assert off["twirling"]["enable_measure"] is False


def test_cli_builds_no_options_for_non_runtime_samplers():
    """Only the runtime sampler takes options, so nothing is assembled otherwise."""
    from embasi_qiskit_integration.cli import SolveCommand

    for kind in ("aer", "mock"):
        assert SolveCommand(directory="/tmp/unused", sampler=kind)._sampler_options() is None


def test_cli_sampler_options_json_merges_one_level_deep():
    """``--sampler-options`` refines a block rather than replacing it wholesale.

    ``{"twirling": {"num_randomizations": 64}}`` must keep ``enable_measure`` from
    the flag; a shallow ``dict.update`` would silently drop it and disable twirling.
    """
    from embasi_qiskit_integration.cli import SolveCommand

    merged = SolveCommand(
        directory="/tmp/unused",
        sampler_options='{"twirling": {"num_randomizations": 64}}',
    )._sampler_options()
    assert merged["twirling"] == {"enable_measure": True, "num_randomizations": 64}

    # An explicit override still wins over the flag.
    overridden = SolveCommand(
        directory="/tmp/unused",
        sampler_options='{"twirling": {"enable_measure": false}}',
    )._sampler_options()
    assert overridden["twirling"]["enable_measure"] is False

    # And a key the flags don't cover is added.
    extra = SolveCommand(
        directory="/tmp/unused", sampler_options='{"default_shots": 99}'
    )._sampler_options()
    assert extra["default_shots"] == 99


def test_cli_sampler_options_rejects_malformed_json():
    """A typo in the JSON must be a usage error, not a traceback or a silent skip."""
    from embasi_qiskit_integration.cli import SolveCommand

    with pytest.raises(SystemExit, match="not valid JSON"):
        SolveCommand(directory="/tmp/unused", sampler_options="{oops")._sampler_options()

    with pytest.raises(SystemExit, match="must be a JSON object"):
        SolveCommand(directory="/tmp/unused", sampler_options="[1, 2]")._sampler_options()
