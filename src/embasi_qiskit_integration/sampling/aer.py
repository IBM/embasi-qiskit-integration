# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Aer-backed bitstring sampler (noiseless statevector simulation)."""

from __future__ import annotations


class AerSampler:
    """Thin wrapper over :class:`qiskit_aer.primitives.SamplerV2`.

    Seeds the simulator per-call so runs are reproducible. Returns the counts
    of the circuit's measurement register.
    """

    def __init__(self, *, default_shots: int = 100_000):
        self.default_shots = default_shots

    def sample(
        self, circuit, shots: int | None = None, *, seed: int | None = None
    ) -> dict[str, int]:
        from qiskit_aer.primitives import SamplerV2

        shots = self.default_shots if shots is None else shots
        sampler = SamplerV2(seed=seed)
        result = sampler.run([circuit], shots=shots).result()
        data = result[0].data
        bit_array = _single_register(data)
        return {str(k): int(v) for k, v in bit_array.get_counts().items()}


def _single_register(data_bin):
    """Return the one classical register in a SamplerV2 result DataBin.

    ``measure_all()`` produces a register named ``meas``; fall back to the sole
    register if a circuit used a differently-named one.
    """
    if hasattr(data_bin, "meas"):
        return data_bin.meas
    names = list(data_bin.keys())
    if len(names) != 1:
        raise ValueError(
            f"expected a single classical register, found {names!r}; "
            "measure into one register or a register named 'meas'"
        )
    return data_bin[names[0]]
