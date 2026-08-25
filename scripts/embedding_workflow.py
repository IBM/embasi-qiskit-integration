# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Thin CLI entry point for the embedding workflow.

The workflow itself now lives in the package as
:class:`embasi_qiskit_integration.embedding.EmbeddingWorkflow` (importable, so
tests and the dissociation-energy driver use it directly rather than loading this
script by path).  This file just runs it::

    uv run python scripts/embedding_workflow.py                 # WF-in-DFT: HF-in-PBE + SQD (default)
    uv run python scripts/embedding_workflow.py --solver fci    # classical FCI reference for SQD
    uv run python scripts/embedding_workflow.py --xc_hl PBE0    # DFT-in-DFT (KS xc_hl; solver inert)

The defaults route the **WF-in-DFT** path (``xc_hl=HF`` -> HF mean field + an
active-space quantum solve), with the ``concentric-cl`` selector (iterative
Concentric Localization) keeping a nested, size-consistent active-virtual space.  A Kohn-Sham ``--xc_hl`` (PBE0/PBE) instead
routes DFT-in-DFT, where the solver/selector are inert.  See
``embasi_qiskit_integration/embedding.py`` for the full flow description and the
complete list of options.
"""

from __future__ import annotations

from embasi_qiskit_integration.embedding import EmbeddingWorkflow, main

__all__ = ["EmbeddingWorkflow", "main"]


if __name__ == "__main__":
    main()
