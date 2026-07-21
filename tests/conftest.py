# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures and markers.

- ``embasi`` marked tests are skipped unless ``EMBASI_AVAILABLE=1``.
- ``rng_seed`` gives every test a deterministic seed.
- ``data_dir`` points at ``tests/data`` (frozen integrals / mock counts).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

EMBASI_AVAILABLE = os.environ.get("EMBASI_AVAILABLE") == "1"

# Frozen global seed so any accidental un-seeded randomness is still reproducible.
RNG_SEED = 20240607


def pytest_collection_modifyitems(config, items):
    """Skip ``embasi``-marked tests unless EMBASI_AVAILABLE=1."""
    if EMBASI_AVAILABLE:
        return
    skip_embasi = pytest.mark.skip(reason="requires EmbASI + FHI-aims (set EMBASI_AVAILABLE=1)")
    for item in items:
        if "embasi" in item.keywords:
            item.add_marker(skip_embasi)


@pytest.fixture
def rng_seed() -> int:
    return RNG_SEED


@pytest.fixture
def rng(rng_seed: int) -> np.random.Generator:
    return np.random.default_rng(rng_seed)


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"
