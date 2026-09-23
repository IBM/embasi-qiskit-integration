# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Aer-backed bitstring sampler (local simulation, method selectable)."""

from __future__ import annotations

from embasi_qiskit_integration.circuit_run.base import SamplerMixin


def available_aer_methods() -> tuple[str, ...]:
    """Simulation methods the installed Aer actually offers.

    Read from Aer rather than hardcoded, so the validation below tracks whatever is
    installed (the list differs with build options -- GPU builds add methods).
    """
    from qiskit_aer import AerSimulator

    return tuple(AerSimulator().available_methods())


class AerSampler(SamplerMixin):
    """Thin wrapper over :class:`qiskit_aer.primitives.SamplerV2`.

    Seeds the simulator per-call so runs are reproducible. Returns the counts of
    each circuit's measurement register.

    The simulation ``method`` is pinned (default ``"matrix_product_state"``) rather than
    left at Aer's ``"automatic"``, which picks from circuit structure so what a given
    ``seed`` reproduces could drift.  **Seeds do not reproduce across methods.**

    MPS trades exactness for memory and reaches wider registers than ``statevector``,
    but only while the bond dimension stays small; ``mps_max_bond_dimension`` caps it,
    at the cost of an approximation.  Uncapped it grows toward the exact state and can
    be slower than ``statevector`` on entangling circuits.
    """

    def __init__(
        self,
        *,
        default_shots: int = 100_000,
        method: str = "matrix_product_state",
        mps_max_bond_dimension: int | None = None,
        mps_truncation_threshold: float | None = None,
    ):
        supported = available_aer_methods()
        if method not in supported:
            raise ValueError(
                f"unknown Aer simulation method {method!r}; the installed Aer offers "
                f"{', '.join(supported)}. (A typo would otherwise fall through to "
                f"whatever Aer defaults to, silently simulating with the wrong method.)"
            )
        if mps_max_bond_dimension is not None:
            if method != "matrix_product_state":
                raise ValueError(
                    f"mps_max_bond_dimension is only meaningful for "
                    f"method='matrix_product_state', not {method!r}"
                )
            if mps_max_bond_dimension < 1:
                raise ValueError(
                    f"mps_max_bond_dimension must be >= 1; got {mps_max_bond_dimension}"
                )
        if mps_truncation_threshold is not None and method != "matrix_product_state":
            raise ValueError(
                f"mps_truncation_threshold is only meaningful for "
                f"method='matrix_product_state', not {method!r}"
            )
        self.default_shots = default_shots
        self.method = method
        self.mps_max_bond_dimension = mps_max_bond_dimension
        self.mps_truncation_threshold = mps_truncation_threshold

    @property
    def backend_options(self) -> dict:
        """The ``backend_options`` handed to ``SamplerV2``, method knobs included."""
        opts: dict = {"method": self.method}
        if self.mps_max_bond_dimension is not None:
            opts["matrix_product_state_max_bond_dimension"] = self.mps_max_bond_dimension
        if self.mps_truncation_threshold is not None:
            opts["matrix_product_state_truncation_threshold"] = self.mps_truncation_threshold
        return opts

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

        sampler = SamplerV2(seed=seed, options={"backend_options": self.backend_options})
        result = sampler.run(list(circuits), shots=shots).result()
        # One PUB per circuit; a circuit carries no parameter bindings here, so
        # each PUB yields exactly one counts dict.
        counts: list[dict[str, int]] = []
        for pub_result in result:
            counts.extend(counts_per_binding_from_pub_result(pub_result))
        return counts
