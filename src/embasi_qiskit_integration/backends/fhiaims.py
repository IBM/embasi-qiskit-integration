# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""FHI-aims backend placeholder."""

from __future__ import annotations


class FHIaimsIntegrals:
    """FHI-aims path.

    TODO(embasi-api): EmbASI/ASI does not export an RI-LVL three-index ERI tensor
    (the V^-1/2-contracted M^P_{pq}, so (pq|rs) = sum_P M^P_pq M^P_rs).  It exports
    density/overlap/Hamiltonian but not the (truncated) ERIs, so this backend
    cannot be built at all -- FHI-aims can drive only the DFT-in-DFT part of the
    workflow and WF-in-DFT stays PySCF-only.
    """

    def __init__(self, *_, **__):
        raise NotImplementedError("RI-LVL ERI export not available through ASI yet")
