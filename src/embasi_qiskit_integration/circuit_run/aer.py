# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Aer-backed bitstring sampler (noiseless statevector simulation)."""

from __future__ import annotations

from embasi_qiskit_integration.circuit_run.base import SamplerMixin


class AerSampler(SamplerMixin):
    """Thin wrapper over :class:`qiskit_aer.primitives.SamplerV2`.

    Seeds the simulator per-call so runs are reproducible. Returns the counts of
    each circuit's measurement register.

    The simulation ``method`` is pinned (default ``"statevector"``) rather than
    left at Aer's ``"automatic"``: automatic picks a method from circuit structure,
    so the RNG consumption pattern -- and hence what a given ``seed`` reproduces --
    could drift as circuits or Aer's heuristics change.
    """

    def __init__(self, *, default_shots: int = 100_000, method: str = "statevector"):
        self.default_shots = default_shots
        self.method = method

    def run(
        self, circuits: list, shots: int | None = None, *, seed: int | None = None
    ) -> list[dict[str, int]]:
        """Sample every circuit in one job; one counts dict each, in input order.

        All circuits share a single seeded ``SamplerV2``, so one submission covers
        the whole batch.
        """
        from qiskit_aer.primitives import SamplerV2

        from embasi_qiskit_integration.circuit_run.counts import (
            counts_per_binding_from_pub_result,
        )

        shots = self.default_shots if shots is None else shots
        if not circuits:
            return []

        sampler = SamplerV2(seed=seed, options={"backend_options": {"method": self.method}})
        result = sampler.run(list(circuits), shots=shots).result()
        # One PUB per circuit; a circuit carries no parameter bindings here, so
        # each PUB yields exactly one counts dict.
        counts: list[dict[str, int]] = []
        for pub_result in result:
            counts.extend(counts_per_binding_from_pub_result(pub_result))
        return counts
