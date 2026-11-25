# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Integration library coupling EmbASI embedding with the Qiskit SQD stack."""

from embasi_qiskit_integration._openmp import preload_torch_openmp as _preload_torch_openmp

# Must run before anything imports qiskit-aer (see _openmp.py).
_preload_torch_openmp()

from embasi_qiskit_integration.contract import (  # noqa: E402
    EmbeddedHamiltonian,
    SolverResult,
)

__all__ = ["EmbeddedHamiltonian", "SolverResult"]
__version__ = "0.0.1"
