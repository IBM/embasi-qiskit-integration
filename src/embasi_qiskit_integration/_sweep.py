# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Settings types for the SqDRIFT sweep axes (``evolution_time``, ``num_groups``).

The circuit generator fans out over ``time x num_groups`` and the solver pools every
circuit of that product into one SQD run.  The CLI
already parses ``--evolution_time 1,2,3`` (or ``'[1,2,3]'``, or a repeated flag) into a
list; the before-validator also accepts a bare scalar from an env var or a keyword
argument, so ``evolution_time=3.0`` keeps working and means ``[3.0]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

from pydantic import BeforeValidator


def _as_list(v: Any) -> Any:
    if isinstance(v, (str, bytes)) or not isinstance(v, Sequence):
        return [v]
    return v


FloatSweep = Annotated[list[float], BeforeValidator(_as_list)]
IntSweep = Annotated[list[int], BeforeValidator(_as_list)]
