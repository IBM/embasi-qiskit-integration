# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""LUCJ ansatz — ON HOLD.

The project focus is the SqDRIFT ansatz (see :mod:`circuits.sqdrift`). The
LUCJ (local unitary cluster Jastrow) ansatz built from CCSD ``t2`` amplitudes
via ffsim (``UCJOpSpinBalanced.from_t_amplitudes`` + ``UCJOpSpinBalancedJW``)
is deferred. ``cheap_ccsd_t2`` in :mod:`solvers` already provides the ``t2``
seed this ansatz would consume when it is implemented.
"""

from __future__ import annotations


def build_lucj_circuit(*args, **kwargs):  # pragma: no cover - intentionally deferred
    raise NotImplementedError(
        "LUCJ is on hold; use build_sqdrift_circuits from circuits.sqdrift, or "
        "build_hf_circuit from circuits.hf for the reference-state baseline."
    )
