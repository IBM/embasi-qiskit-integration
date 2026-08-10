# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Circuit execution: circuits -> measured bitstring counts.

The execution half of the quantum path, kept separable from circuit generation.
Circuits arrive as live :class:`~qiskit.QuantumCircuit` objects rather than files
on disk, and counts are returned rather than written to a run directory.

The pipeline a caller drives is:

1. :func:`~.prep.resolve_initial_state` -- pick the reference determinant,
2. :func:`~.prep.compose_full_circuit` -- prepend it to the bare ansatz circuit,
3. ``sampler.run(circuits, shots)`` -- sample the batch,
4. :func:`~.counts.merge_counts` -- pool the ensemble into one distribution.

Modules:

- :mod:`.base` -- the :class:`BitstringSampler` protocol and ``MockSampler``.
- :mod:`.aer` / :mod:`.runtime` -- Aer and IBM Quantum Runtime samplers.
- :mod:`.prep` -- initial-state resolution and composition.
- :mod:`.counts` -- one shared ``SamplerV2``-result reader, plus count merging.
- :mod:`.backend` -- backend resolution and ISA transpilation.
"""

from __future__ import annotations

from embasi_qiskit_integration.circuit_run.base import (
    BitstringSampler,
    MockSampler,
    SamplerKind,
    SamplerMixin,
)
from embasi_qiskit_integration.circuit_run.counts import merge_counts
from embasi_qiskit_integration.circuit_run.prep import (
    bitstring_prep_circuit,
    compose_full_circuit,
    hf_prep_circuit,
    resolve_initial_state,
)

__all__ = [
    "BitstringSampler",
    "MockSampler",
    "SamplerKind",
    "SamplerMixin",
    "bitstring_prep_circuit",
    "build_sampler",
    "compose_full_circuit",
    "hf_prep_circuit",
    "merge_counts",
    "resolve_initial_state",
]


def build_sampler(
    kind: SamplerKind,
    *,
    counts: str | None = None,
    backend: str | None = None,
    optimization_level: int = 3,
    default_shots: int = 100_000,
):
    """Construct a sampler by name -- the shared ``--sampler`` dispatch.

    Args:
        kind: ``"aer"`` (local noiseless simulation), ``"mock"`` (replay frozen
            counts), or ``"runtime"`` (IBM Quantum hardware or a ``Fake*`` device).
        counts: path to the frozen counts JSON; required for ``"mock"``.
        backend: backend name for ``"runtime"``; ``None`` picks the least-busy.
        optimization_level: ISA-transpile level for ``"runtime"``.
        default_shots: shot budget used when a caller does not pass ``shots``.

    Raises:
        ValueError: for an unknown ``kind``, or ``"mock"`` without ``counts``.
    """
    if kind == "mock":
        if not counts:
            raise ValueError("sampler 'mock' requires a counts file path")
        return MockSampler(counts)

    if kind == "aer":
        from embasi_qiskit_integration.circuit_run.aer import AerSampler

        return AerSampler(default_shots=default_shots)

    if kind == "runtime":
        from embasi_qiskit_integration.circuit_run.runtime import RuntimeSampler

        return RuntimeSampler(
            backend=backend,
            optimization_level=optimization_level,
            default_shots=default_shots,
        )

    raise ValueError(f"unknown sampler {kind!r}; expected one of 'aer', 'mock', 'runtime'")
