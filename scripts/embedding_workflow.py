# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Thin CLI entry point for the embedding workflow.

The workflow itself now lives in the package as
:class:`embasi_qiskit_integration.embedding.EmbeddingWorkflow` (importable, so
tests and the interaction-energy driver use it directly rather than loading this
script by path).  This file just runs it::

    uv run python scripts/embedding_workflow.py --solver fci --n-virtual 2

See ``embasi_qiskit_integration/embedding.py`` for the full flow description and
the complete list of options.
"""

from __future__ import annotations

from embasi_qiskit_integration.embedding import EmbeddingWorkflow, main

__all__ = ["EmbeddingWorkflow", "main"]


if __name__ == "__main__":
    main()
