# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Readout metric computation from corrected shot arrays (pure numpy)."""

from __future__ import annotations

import numpy as np


def compute_manual_readout_errors(shots_trex: np.ndarray) -> np.ndarray:
    """Per-qubit readout error: the mean of the twirl-corrected shots.

    The characterisation program measures the all-zero state, so after the twirl
    correction every ``1`` in the array is a readout flip. The mean over
    randomizations and shots is therefore the per-qubit error probability
    directly -- this is the sole criterion
    :func:`~embasi_qiskit_integration.circuit_run.layout.select_layout` prunes on.

    Args:
        shots_trex: corrected shots, shape ``(n_rand, shots_per_twirl, n_qubits)``.

    Returns:
        Per-qubit error, shape ``(n_qubits,)``.
    """
    return np.mean(shots_trex, axis=(0, 1))


def compute_xslow_flip_fidelity(shots_trex: np.ndarray, shots_ps_trex: np.ndarray) -> np.ndarray:
    """Per-qubit fraction of shots whose ``xslow`` re-measurement inverted the data.

    Used with hot-coupler postselection: the re-measurement should read the
    complement of the data measurement, so agreement with that expectation is a
    fidelity.

    Args:
        shots_trex: corrected data shots, ``(n_rand, shots_per_twirl, n_qubits)``.
        shots_ps_trex: corrected postselection shots, same shape.

    Returns:
        Per-qubit fidelity, shape ``(n_qubits,)``.
    """
    data_comp = np.logical_not(shots_trex) == shots_ps_trex.astype(bool)
    return np.mean(data_comp, axis=(0, 1))


def compute_trex_fidelities(
    shots_trex: np.ndarray, n_qubits: int, ps_mask: np.ndarray | None = None
) -> np.ndarray:
    """Per-qubit TREX readout fidelity: the single-qubit ``Z`` expectation value.

    ``1`` is perfect readout, ``0`` fully depolarized.

    Args:
        shots_trex: corrected shots, ``(n_rand, shots_per_twirl, n_qubits)``.
        n_qubits: qubit count (the last axis of ``shots_trex``).
        ps_mask: optional boolean ``(n_rand, shots_per_twirl)`` mask; when given
            only the selected shots contribute (e.g. hot-coupler postselection).

    Returns:
        Per-qubit fidelity, shape ``(n_qubits,)``.
    """
    observables = ["I" * i + "Z" + "I" * (n_qubits - i - 1) for i in range(n_qubits)]
    return np.array(
        [
            _expectation_values(shots_trex, [observable], ps_mask=ps_mask)[0][0]
            for observable in observables
        ]
    )


def _expectation_values(
    bitstrings: np.ndarray,
    observables: list[str],
    signs: np.ndarray | None = None,
    gamma_factor: float = 1.0,
    ps_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expectation values (and standard errors) of diagonal observables.

    Only ``Z`` positions in each Pauli string contribute; a shot's contribution is
    the parity of the measured bits on those positions, flipped when the
    randomization carries a sign.

    Args:
        bitstrings: shape ``(n_settings, shots_per_setting, n_qubits)``.
        observables: Pauli strings, e.g. ``["IZZIII"]``.
        signs: optional per-setting boolean signs; ``None`` means all-positive.
        gamma_factor: PEC gamma factor for the circuit.
        ps_mask: optional boolean ``(n_settings, shots_per_setting)`` keep-mask.

    Returns:
        ``(values, standard_errors)``, each of length ``len(observables)``.
    """
    n_settings = bitstrings.shape[0]
    n_qubits = bitstrings.shape[2]
    values = np.zeros(len(observables))
    errors = np.zeros(len(observables))

    for index, observable in enumerate(observables):
        setting_signs = (
            np.zeros(n_settings, dtype=bool) if signs is None else np.asarray(signs).ravel()
        )
        z_mask = [observable[position] == "Z" for position in range(n_qubits)]
        relevant = bitstrings[:, :, z_mask]

        zero_count = 0
        one_count = 0
        for setting, outcomes in enumerate(relevant):
            for shot, outcome in enumerate(outcomes):
                if ps_mask is not None and not ps_mask[setting][shot]:
                    continue
                even_parity = np.count_nonzero(outcome) % 2 == 0
                if even_parity != bool(setting_signs[setting]):
                    zero_count += 1
                else:
                    one_count += 1

        total = zero_count + one_count
        if total > 0:
            values[index] = gamma_factor * (zero_count - one_count) / total
            samples = np.concatenate(
                [
                    np.ones(zero_count) * gamma_factor,
                    -gamma_factor * np.ones(one_count),
                ]
            )
            errors[index] = np.std(samples) / np.sqrt(total)

    return values, errors
