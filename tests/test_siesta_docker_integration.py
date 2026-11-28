# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Full Docker-based SIESTA-QCOMP WF-in-DFT integration test.

Opt in explicitly; the default pytest run skips it because a cold run builds
the image's MPI HDF5/NetCDF stack, clones SIESTA, builds libint and SIESTA,
installs a Python venv and executes the real embedding (roughly 25-45 minutes).
Subsequent runs reuse the image and the named volume.

Run from the repository root::

    SIESTA_DOCKER_INTEGRATION=1 pytest -q tests/test_siesta_docker_integration.py

The test skips when Docker is unavailable, when its daemon cannot be reached,
or when ``CI`` is set.  See ``docker/siesta-asi/README.md`` for cache layout,
log location and reset instructions.
"""

from __future__ import annotations

import pytest

from _siesta_docker import run_integration, skip_reason

pytestmark = [pytest.mark.slow, pytest.mark.siesta_docker]


def test_siesta_qcomp_docker_integration():
    """Clone/build SIESTA from qcomp-5.4-asi and pass the real WF-in-DFT test."""
    reason = skip_reason()
    if reason is not None:
        pytest.skip(reason)

    result = run_integration()
    assert result.returncode == 0, (
        f"SIESTA-QCOMP Docker integration failed with exit status "
        f"{result.returncode} (image {result.image}, log {result.log_path}).\n"
        f"--- last log lines ---\n{result.tail()}"
    )
