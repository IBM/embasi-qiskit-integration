# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for ``sqdrift._build_chunk`` that fails on chosen draw seeds.

Imported by spawned worker processes, so it lives in its own module. The seeds to
fail come from the environment (inherited by spawned workers):
``EQI_TEST_FAIL_SEEDS="37,77"``, and ``EQI_TEST_FAIL_MODE`` ``segv`` (default; the
worker dies of a real SIGSEGV, as HiGHS does) or ``raise``. ``EQI_TEST_FAIL_SEEDS=all``
fails every draw. A circuit is labelled with the seed it was built from.
"""

from __future__ import annotations

import os
import signal

from embasi_qiskit_integration.circuit_generator.sqdrift import SqdriftBuildResult


def build_chunk(spec, combos, rand_range):
    spec_fail = os.environ.get("EQI_TEST_FAIL_SEEDS", "")
    fail = {int(s) for s in spec_fail.split(",") if s and s != "all"}
    result = SqdriftBuildResult(optimize_requested=spec.optimize)
    for evolution_time, n_groups in combos:
        for randomization in rand_range:
            seed = spec.seed + randomization
            if spec_fail == "all" or seed in fail:
                if os.environ.get("EQI_TEST_FAIL_MODE", "segv") == "segv":
                    os.kill(os.getpid(), signal.SIGSEGV)
                raise RuntimeError(f"seed {seed} failed")
            result.circuits.append(f"t{evolution_time:g}-g{n_groups}-s{seed}")
            result.permutations.append(None)
            result.randomization_indices.append(randomization)
    return result
