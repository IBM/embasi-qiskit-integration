# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Bitstring sampler protocol and the deterministic ``MockSampler`` for CI.

A sampler turns a measured circuit into a ``{bitstring: count}`` dictionary.
The SQD driver consumes these counts. ``MockSampler`` replays frozen counts so
tests never touch a simulator, network, or hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class BitstringSampler(Protocol):
    """Anything that maps a circuit + shot budget to measurement counts."""

    def sample(self, circuit, shots: int, *, seed: int | None = None) -> dict[str, int]: ...


class MockSampler:
    """Replays counts from a JSON file; ignores the circuit. For tests/CI.

    The JSON file maps ``{bitstring: count}``. ``sample`` returns those counts
    verbatim (rescaled to ``shots`` if requested), so a single frozen file
    drives fully deterministic SQD tests.
    """

    def __init__(self, counts_path: str | Path):
        self.counts_path = Path(counts_path)
        with self.counts_path.open() as fh:
            raw = json.load(fh)
        self._counts: dict[str, int] = {str(k): int(v) for k, v in raw.items()}

    def sample(self, circuit, shots: int, *, seed: int | None = None) -> dict[str, int]:
        del circuit, seed  # deliberately ignored — deterministic replay
        total = sum(self._counts.values())
        if total == 0 or shots is None or shots == total:
            return dict(self._counts)
        # Rescale proportionally to the requested shot budget, preserving the
        # total exactly by handing any rounding remainder to the largest bin.
        scaled: dict[str, int] = {}
        for bitstring, count in self._counts.items():
            scaled[bitstring] = int(round(count * shots / total))
        drift = shots - sum(scaled.values())
        if drift != 0 and scaled:
            top = max(scaled, key=scaled.__getitem__)
            scaled[top] += drift
        return scaled
