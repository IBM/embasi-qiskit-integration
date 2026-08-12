# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Readout characterisation: measure a device's per-qubit readout error."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np

from embasi_qiskit_integration.circuit_run.layout import (
    DEFAULT_READOUT_ERROR_THRESHOLD,
    rank_layouts,
    select_layout,
)

logger = logging.getLogger(__name__)

# Reference defaults for the twirl program.
DEFAULT_N_RAND_TWIRL = 300
DEFAULT_SHOTS_PER_TWIRL = 25


def samplomatic_available() -> bool:
    """True if the optional ``samplomatic`` package is importable."""
    try:
        return importlib.util.find_spec("samplomatic") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs only
        return False


def _require_samplomatic() -> None:
    if not samplomatic_available():
        raise RuntimeError(
            "readout characterisation requires the 'samplomatic' package; install it "
            "(pip install -e '.[hardware]') or disable characterisation."
        )


def append_ps_measurements(circuit: Any, n_qubits: int, xslow: Any) -> Any:
    """Append a hot-coupler postselection register and its re-measurements.

    Each data qubit is re-measured after an ``xslow`` pulse, into a parallel
    ``meas_ps`` register, so a shot can be postselected on the re-measurement
    having inverted the data measurement.

    Args:
        circuit: the transpiled circuit to extend.
        n_qubits: number of data qubits.
        xslow: the ``xslow`` gate used for the re-measurement.

    Returns:
        A new circuit carrying the extra register (the input is not mutated).
    """
    from qiskit.circuit import ClassicalRegister, Measure

    extended = deepcopy(circuit)

    ps_register = ClassicalRegister(n_qubits, name="meas_ps")
    extended.barrier()
    extended.add_register(ps_register)

    measurements = []
    for instruction, qargs, cargs in extended.data:
        if isinstance(instruction, Measure):
            measurements.append((qargs[0], cargs[0]))

    for qubit, _ in measurements:
        extended.append(xslow, [qubit._index])

    for qubit, clbit in measurements:
        if clbit._register.name == "meas":
            extended.measure(qubit._index, ps_register[clbit._index])

    return extended


def build_readout_program(
    backend: Any,
    qubit_layout: Sequence[int],
    n_rand: int = DEFAULT_N_RAND_TWIRL,
    shots_per_twirl: int = DEFAULT_SHOTS_PER_TWIRL,
    hot_coupler_ps: bool = False,
    xslow: Any = None,
) -> Any:
    """Build the ``samplomatic`` readout-twirl program for ``qubit_layout``.

    The circuit is measure-only: prepare ``|0...0>``, measure every qubit. Under
    measurement twirling each randomization flips a known random subset of the
    outcomes, so XOR-ing the flip mask back out (see
    :func:`extract_corrected_shots`) leaves exactly the readout errors.

    Args:
        backend: the IBM backend to transpile against.
        qubit_layout: physical qubits to characterise.
        n_rand: twirling randomizations.
        shots_per_twirl: shots per randomization.
        hot_coupler_ps: append ``xslow`` postselection re-measurements.
        xslow: the ``xslow`` gate; required when ``hot_coupler_ps`` is set.

    Returns:
        A ready-to-submit ``QuantumProgram``.
    """
    _require_samplomatic()

    from qiskit.circuit import ClassicalRegister, QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.quantum_program import QuantumProgram
    from samplomatic.builders import build
    from samplomatic.transpiler import generate_boxing_pass_manager

    qubit_layout = list(qubit_layout)
    n_qubits = len(qubit_layout)

    boxing_pass_manager = generate_boxing_pass_manager(
        enable_gates=True, enable_measures=True, inject_noise_targets="none"
    )
    # optimization_level=0 and a pinned initial_layout: this program must measure
    # exactly the requested physical qubits, so no pass may relocate them.
    isa_pass_manager = generate_preset_pass_manager(
        backend=backend, initial_layout=qubit_layout, optimization_level=0
    )

    circuit = QuantumCircuit(n_qubits)
    data_register = ClassicalRegister(n_qubits, name="meas")
    circuit.add_register(data_register)
    for index in range(n_qubits):
        circuit.measure(index, data_register[index])

    boxed = boxing_pass_manager.run(isa_pass_manager.run(circuit))
    template, samplex = build(boxed)

    if hot_coupler_ps:
        if xslow is None:
            raise ValueError("xslow gate must be provided when hot_coupler_ps=True")
        template = append_ps_measurements(template, n_qubits, xslow)

    program = QuantumProgram(shots=shots_per_twirl)
    program.append_samplex_item(template, samplex=samplex, samplex_arguments={}, shape=(n_rand, 1))
    return program


def run_readout_job(
    backend: Any,
    qubit_layout: Sequence[int],
    n_rand_twirl: int = DEFAULT_N_RAND_TWIRL,
    n_shots_per_twirl: int = DEFAULT_SHOTS_PER_TWIRL,
    hot_coupler_ps: bool = False,
) -> Any:
    """Submit the readout-twirl program and return its raw result.

    Submitted in *job mode* (``Executor(mode=backend)``), not inside a session --
    see the module docstring for why.
    """
    _require_samplomatic()

    from qiskit_ibm_runtime import Executor

    program = build_readout_program(
        backend=backend,
        qubit_layout=qubit_layout,
        n_rand=n_rand_twirl,
        shots_per_twirl=n_shots_per_twirl,
        hot_coupler_ps=hot_coupler_ps,
    )
    executor = Executor(mode=backend)
    return executor.run(program).result()


def extract_corrected_shots(
    job_result: Any,
    outputs: dict,
    flip_key_data: str = "measurement_flips.meas",
    hot_coupler_ps: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Undo the measurement twirl to recover corrected shots.

    Each randomization flipped a known subset of outcomes; XOR-ing that mask back
    out leaves only the genuine readout errors, so the mean of the result is the
    per-qubit error rate.

    **samplomatic reports its qubit order reversed relative to the circuit**, so
    the flip mask is reversed (``[:, :, ::-1]``) before the XOR. Getting that
    backwards would silently attribute every qubit's error to its mirror.

    Args:
        job_result: result of ``executor.run(program)``.
        outputs: the samplex outputs dict (``job_result[0]``).
        flip_key_data: key holding the data-measurement flip mask.
        hot_coupler_ps: also correct and return the postselection re-measurement.

    Returns:
        ``(data_shots, ps_shots_or_None)``, each ``(n_rand, shots, n_qubits)``.
    """
    data = job_result[0]
    flips_data = outputs[flip_key_data][:, :, ::-1]
    measured = (data["meas"] ^ flips_data).astype("int")

    if hot_coupler_ps:
        ps_shots = (data["meas_ps"] ^ flips_data).astype("int")
        return np.squeeze(measured), np.squeeze(ps_shots)
    return np.squeeze(measured), None


def select_readout_layout(
    job_result: Any,
    backend: Any,
    qubit_layout: Sequence[int],
    required_size: int,
    hot_coupler_ps: bool = False,
    readout_error_threshold: float = DEFAULT_READOUT_ERROR_THRESHOLD,
) -> dict[str, Any]:
    """Post-process a readout-twirl result into a ranked chain layout.

    Pure client-side work -- no job submission -- so it is safe to run however long
    it takes: extract corrected shots, prune the coupling map by measured readout
    error (with adaptive threshold relaxation), find viable chains, rank them.

    Raises:
        RuntimeError: if no chain of ``required_size`` exists even on the
            fully-unpruned topology, since there is then no layout to pin.
    """
    shots_trex, _ = extract_corrected_shots(
        job_result=job_result, outputs=job_result[0], hot_coupler_ps=hot_coupler_ps
    )

    pruned_map, possible_layouts, qubits_to_avoid, threshold_used = select_layout(
        shots_trex=shots_trex,
        coupling_map=backend.coupling_map,
        qubit_layout=qubit_layout,
        required_size=required_size,
        readout_error_threshold=readout_error_threshold,
    )
    if not possible_layouts:
        raise RuntimeError(
            f"No viable 1-D chain of size {required_size} on the device coupling map "
            f"even after relaxing the readout-error threshold to {threshold_used:g}"
        )

    ranked_layouts, layout_scores = rank_layouts(possible_layouts, shots_trex, qubit_layout)

    logger.info(
        "Readout characterisation: chose layout %s (joint readout fidelity %.4f) "
        "from %d candidate chain(s); avoided %d qubit(s) at threshold %g.",
        ranked_layouts[0],
        layout_scores[0],
        len(ranked_layouts),
        len(qubits_to_avoid),
        threshold_used,
    )

    return {
        "best_layout": [int(q) for q in ranked_layouts[0]],
        "layout_score": float(layout_scores[0]),
        "possible_layouts": [[int(q) for q in layout] for layout in ranked_layouts],
        "layout_scores": [float(score) for score in layout_scores],
        "qubits_to_avoid": [int(q) for q in qubits_to_avoid],
        "coupling_map_pruned": [[int(u), int(v)] for u, v in pruned_map.get_edges()],
        "readout_error_threshold_used": float(threshold_used),
    }


def characterise_readout(
    backend: Any,
    qubit_layout: Sequence[int],
    required_size: int,
    n_rand_twirl: int = DEFAULT_N_RAND_TWIRL,
    n_shots_per_twirl: int = DEFAULT_SHOTS_PER_TWIRL,
    hot_coupler_ps: bool = False,
    readout_error_threshold: float = DEFAULT_READOUT_ERROR_THRESHOLD,
) -> dict[str, Any]:
    """Run the readout-twirl program and select a 1-D chain layout.

    Compose helper over :func:`run_readout_job` and :func:`select_readout_layout`.
    The job runs in job mode, so by the time this returns nothing is left open and
    the layout post-processing has already happened client-side.
    """
    job_result = run_readout_job(
        backend=backend,
        qubit_layout=qubit_layout,
        n_rand_twirl=n_rand_twirl,
        n_shots_per_twirl=n_shots_per_twirl,
        hot_coupler_ps=hot_coupler_ps,
    )
    return select_readout_layout(
        job_result=job_result,
        backend=backend,
        qubit_layout=qubit_layout,
        required_size=required_size,
        hot_coupler_ps=hot_coupler_ps,
        readout_error_threshold=readout_error_threshold,
    )
