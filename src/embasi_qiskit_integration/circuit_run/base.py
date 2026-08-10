# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Bitstring sampler protocol and the deterministic ``MockSampler`` for CI.

A sampler turns measured circuits into ``{bitstring: count}`` dictionaries, which
the SQD driver consumes.

:meth:`BitstringSampler.run` is the primary seam: it takes a *list* of circuits
and returns one counts dict per circuit, so an ensemble ansatz (e.g. SqDRIFT
randomizations) is sampled in one submission rather than one call per circuit.
:meth:`BitstringSampler.sample` is the single-circuit convenience over it,
provided once by :class:`SamplerMixin`.

``MockSampler`` replays frozen counts so tests never touch a simulator, network,
or hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

SamplerKind = Literal["aer", "mock", "runtime"]


@runtime_checkable
class BitstringSampler(Protocol):
    """Anything that maps circuits + a shot budget to measurement counts."""

    def run(
        self, circuits: list, shots: int | None = None, *, seed: int | None = None
    ) -> list[dict[str, int]]:
        """Sample every circuit, returning one counts dict each, in input order."""
        ...

    def sample(self, circuit, shots: int, *, seed: int | None = None) -> dict[str, int]:
        """Sample a single circuit; the counts dict for it."""
        ...


class SamplerMixin:
    """Provides :meth:`sample` in terms of :meth:`run`.

    Every sampler implements the batch method and inherits the single-circuit
    one, so the two can never disagree.
    """

    def sample(self, circuit, shots: int | None = None, *, seed: int | None = None):
        """Sample one ``circuit``; equivalent to ``run([circuit], ...)[0]``."""
        return self.run([circuit], shots, seed=seed)[0]  # type: ignore[attr-defined]


class MockSampler(SamplerMixin):
    """Replays counts from a JSON file; ignores the circuits. For tests/CI.

    The JSON file maps ``{bitstring: count}``. Each requested circuit yields
    those counts verbatim (rescaled to ``shots`` if requested), so a single
    frozen file drives fully deterministic SQD tests.
    """

    # The circuits are ignored entirely, so callers can skip building them --
    # which keeps pure-CI runs free of the optional qiskit-fermions dependency.
    requires_circuit = False

    def __init__(self, counts_path: str | Path):
        self.counts_path = Path(counts_path)
        with self.counts_path.open() as fh:
            raw = json.load(fh)
        self._counts: dict[str, int] = {str(k): int(v) for k, v in raw.items()}

    def run(
        self, circuits: list, shots: int | None = None, *, seed: int | None = None
    ) -> list[dict[str, int]]:
        """Replay the frozen counts once per circuit, ignoring the circuits."""
        del seed  # deliberately ignored — deterministic replay
        return [self._replay(shots) for _ in circuits]

    def _replay(self, shots: int | None) -> dict[str, int]:
        """The frozen counts, rescaled to ``shots`` when one is requested."""
        total = sum(self._counts.values())
        if total == 0 or shots is None or shots == total:
            return dict(self._counts)
        # Rescale proportionally to the requested shot budget, preserving the
        # total exactly by handing any rounding remainder to the largest bin.
        scaled: dict[str, int] = {}
        for bitstring, count in self._counts.items():
            # round() on a float already returns an int; no cast needed.
            scaled[bitstring] = round(count * shots / total)
        drift = shots - sum(scaled.values())
        if drift != 0 and scaled:
            top = max(scaled, key=scaled.__getitem__)
            scaled[top] += drift
        return scaled
