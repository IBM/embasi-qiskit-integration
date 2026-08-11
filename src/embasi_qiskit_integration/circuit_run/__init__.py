# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Circuit execution: circuits -> measured bitstring counts.

The execution half of the quantum path, kept separable from circuit generation.
Circuits arrive as live :class:`~qiskit.QuantumCircuit` objects rather than files
on disk, and counts are returned rather than written to a run directory.
"""

from __future__ import annotations

from embasi_qiskit_integration.circuit_run.base import (
    BitstringSampler,
    MockSampler,
    SamplerKind,
    SamplerMixin,
)
from embasi_qiskit_integration.circuit_run.counts import merge_counts
from embasi_qiskit_integration.circuit_run.permutation import (
    unpermute_bitstrings,
    unpermute_counts,
    unpermute_counts_list,
)
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
    "characterise_readout",
    "compose_full_circuit",
    "find_lines",
    "hf_prep_circuit",
    "merge_counts",
    "rank_layouts",
    "read_noise_from_backend",
    "resolve_initial_state",
    "select_layout",
    "unpermute_bitstrings",
    "unpermute_counts",
    "unpermute_counts_list",
]


def __getattr__(name: str):
    """Lazily expose the layout / characterisation helpers.

    These pull in qiskit (and, for characterisation, ``samplomatic``), so they are
    resolved on first access rather than at package import -- the classical and
    replay paths must stay free of those dependencies.
    """
    if name in ("find_lines", "rank_layouts", "select_layout"):
        from embasi_qiskit_integration.circuit_run import layout

        return getattr(layout, name)
    if name == "characterise_readout":
        from embasi_qiskit_integration.circuit_run.characterisation import characterise_readout

        return characterise_readout
    if name == "read_noise_from_backend":
        from embasi_qiskit_integration.circuit_run.backend import read_noise_from_backend

        return read_noise_from_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_sampler(
    kind: SamplerKind,
    *,
    counts: str | None = None,
    backend: str | None = None,
    optimization_level: int = 3,
    default_shots: int = 100_000,
    options=None,
    enable_readout_characterisation: bool = False,
    readout_error_threshold: float = 0.03,
    n_rand_twirl: int = 300,
    n_shots_per_twirl: int = 25,
    hot_coupler_ps: bool = False,
):
    """Construct a sampler by name -- the shared ``--sampler`` dispatch.

    Args:
        kind: ``"aer"`` (local noiseless simulation), ``"mock"`` (replay frozen
            counts), or ``"runtime"`` (IBM Quantum hardware or a ``Fake*`` device).
        counts: path to the frozen counts JSON; required for ``"mock"``.
        backend: backend name for ``"runtime"``; ``None`` picks the least-busy.
        optimization_level: ISA-transpile level for ``"runtime"``.
        default_shots: shot budget used when a caller does not pass ``shots``.
        options: ``SamplerOptions`` (or dict) forwarded to ``SamplerV2``; see
            :class:`~embasi_qiskit_integration.circuit_run.runtime.RuntimeSampler`
            for why hardware runs want ``{"twirling": {"enable_measure": True}}``.
            ``"runtime"`` only -- the local and replay samplers have no such
            options, so passing it with them is a ``ValueError`` rather than a
            silently ignored mitigation.
        enable_readout_characterisation: measure the device's per-qubit readout
            error before submitting and pin the best 1-D chain (``"runtime"``
            only; see
            :mod:`~embasi_qiskit_integration.circuit_run.characterisation`).
            Costs one extra short job. Off by default, and refused on simulated
            backends, whose "measured" error is just their noise model.
        readout_error_threshold: readout-error ceiling for a usable qubit when
            characterising; relaxed automatically if pruning leaves no chain.
        n_rand_twirl: twirling randomizations in the characterisation job.
        n_shots_per_twirl: shots per randomization.
        hot_coupler_ps: append ``xslow`` postselection re-measurements.

    Raises:
        ValueError: for an unknown ``kind``, ``"mock"`` without ``counts``, or
            ``options``/characterisation with a sampler that cannot use them.
    """
    if kind != "runtime":
        if options is not None:
            raise ValueError(
                f"sampler {kind!r} takes no sampler options (they are IBM Runtime "
                "SamplerV2 options); drop `options` or use --sampler runtime"
            )
        if enable_readout_characterisation:
            raise ValueError(
                f"sampler {kind!r} cannot characterise readout: it needs a real "
                "device to measure. Use --sampler runtime, or leave it disabled."
            )

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
            options=options,
            enable_readout_characterisation=enable_readout_characterisation,
            readout_error_threshold=readout_error_threshold,
            n_rand_twirl=n_rand_twirl,
            n_shots_per_twirl=n_shots_per_twirl,
            hot_coupler_ps=hot_coupler_ps,
        )

    raise ValueError(f"unknown sampler {kind!r}; expected one of 'aer', 'mock', 'runtime'")
