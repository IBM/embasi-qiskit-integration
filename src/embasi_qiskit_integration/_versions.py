# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Provenance helper: versions of the pre-1.0 dependencies for diagnostics."""

from __future__ import annotations

import importlib.metadata as _md

_PACKAGES = (
    "qiskit",
    "qiskit-aer",
    "qiskit-addon-sqd",
    "qiskit-fermions",
    "qiskit-ibm-runtime",
    "ffsim",
    "pyscf",
    "numpy",
    "scipy",
)


def collect_versions() -> dict[str, str]:
    """Return ``{package: version}`` for every resolvable dependency.

    Missing/optional packages (e.g. ``qiskit-fermions``) are reported as
    ``"not installed"`` rather than omitted, so the provenance is explicit.
    """
    versions: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            versions[name] = _md.version(name)
        except _md.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
