# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Keep macOS on a single OpenMP runtime when qiskit-aer and torch coexist.

qiskit-aer's macOS wheel ships its own ``libomp.dylib`` and loads it when
qiskit-aer is imported.  torch's wheel ships another copy.  If Aer's copy is
loaded first with global symbols, torch's later-loaded runtime resolves some
internal calls (``__kmp_suspend_initialize_thread``) against Aer's copy; the
two runtimes then mix thread metadata and torch segfaults in its first
multi-threaded kernel.

Loading torch's ``libomp.dylib`` with ``RTLD_GLOBAL`` before qiskit-aer is
imported makes Aer reuse it instead of loading its bundled copy.  Locating the
dylibs through ``find_spec`` and ``RTLD_NOLOAD`` avoids importing either
package just for this.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from pathlib import Path

_preloaded = False
_preloaded_before_aer = False


def _package_dir(name: str) -> Path | None:
    """Directory of an installed package, without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent


def _is_loaded(path: Path) -> bool:
    """Whether an image at this exact path is already mapped (macOS)."""
    no_load = getattr(os, "RTLD_NOLOAD", 0)
    if not no_load:
        return False
    try:
        ctypes.CDLL(str(path), mode=no_load)
    except OSError:
        return False
    return True


def preload_torch_openmp() -> bool:
    """Make multi-threaded torch safe alongside qiskit-aer on macOS.

    Returns True when only one OpenMP runtime will exist: torch's copy was
    loaded before Aer's, torch is not installed, or Aer's bundled copy is not
    mapped.  Returns False when Aer's ``libomp.dylib`` is already loaded
    without torch's runtime, in which case torch must stay at one thread.
    """
    global _preloaded, _preloaded_before_aer
    if sys.platform != "darwin":
        return True
    if _preloaded:
        return _preloaded_before_aer

    aer_dir = _package_dir("qiskit_aer")
    aer_omp = None if aer_dir is None else aer_dir / "backends" / "libomp.dylib"
    aer_loaded = bool(aer_omp is not None and aer_omp.is_file() and _is_loaded(aer_omp))

    torch_dir = _package_dir("torch")
    if torch_dir is None:
        return True  # no torch: Aer's runtime is the only one that can exist

    torch_omp = None
    for name in ("libomp.dylib", "libiomp5.dylib"):
        candidate = torch_dir / "lib" / name
        if candidate.is_file():
            torch_omp = candidate
            break
    torch_loaded = bool(torch_omp is not None and _is_loaded(torch_omp))

    if aer_loaded:
        _preloaded = True
        _preloaded_before_aer = False
        return False
    if torch_omp is None:
        # torch is installed but we cannot identify its OpenMP runtime; be
        # conservative only if qiskit-aer is already in play.
        safe = "qiskit_aer" not in sys.modules
        _preloaded, _preloaded_before_aer = True, safe
        return safe
    if not torch_loaded:
        try:
            ctypes.CDLL(str(torch_omp), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            _preloaded, _preloaded_before_aer = True, False
            return False
    _preloaded = True
    _preloaded_before_aer = True
    return True
