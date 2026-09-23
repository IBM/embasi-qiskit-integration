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


# The shipped default is `matrix_product_state`, but uncapped MPS is slower on the narrow
# entangling circuits the suite builds (~32 min vs ~4.5 min for the full run).  Tests
# exercise circuit and solver logic, not simulator scaling, so pin them to `statevector`.
TEST_AER_METHOD = "statevector"


@pytest.fixture(autouse=True)
def _statevector_aer_in_tests(request, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default the Aer sampler to ``statevector`` for the suite only.

    Opt out with ``@pytest.mark.shipped_aer_default`` -- the tests that assert what the
    *shipped* default is must see the real one, or they would pin this fixture instead.
    """
    if request.node.get_closest_marker("shipped_aer_default"):
        return
    try:
        from embasi_qiskit_integration.circuit_run import aer as _aer
    except ImportError:  # qiskit-aer absent: the classical/replay paths do not need it
        return

    monkeypatch.setattr(
        _aer.AerSampler.__init__,
        "__kwdefaults__",
        {**(_aer.AerSampler.__init__.__kwdefaults__ or {}), "method": TEST_AER_METHOD},
    )
    # Patching the resolved default covers `build_sampler` and the pydantic settings
    # classes too, since all of them end up constructing an `AerSampler`.
    import embasi_qiskit_integration.circuit_run as _cr

    monkeypatch.setattr(_cr, "_DEFAULT_AER_METHOD", TEST_AER_METHOD)
