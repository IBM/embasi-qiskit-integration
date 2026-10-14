# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Integration library coupling EmbASI embedding with the Qiskit SQD stack."""

import embasi_qiskit_integration._pyscf_compat  # noqa: F401
from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult

__all__ = ["EmbeddedHamiltonian", "SolverResult"]
__version__ = "0.0.1"
