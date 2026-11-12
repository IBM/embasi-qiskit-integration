# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Selecting a QM backend by name (``--qm_code``).

A backend is a builder::

    build(atoms, charge, cfg) -> QMBackend

that turns the reordered (region-1-first) ASE ``atoms`` and the workflow settings into
the two EmbASI calculators plus the matching :class:`~.base.AOIntegrals`.  It is the
only code that knows how to construct and configure a particular QM program.

A new code registers itself with
:func:`register_backend`; :func:`make_backend` refuses an unknown name with the list of
what is available, rather than falling back to PySCF.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from embasi_qiskit_integration.backends.base import AOIntegrals

#: The QM code used when none is named.
DEFAULT_QM_CODE = "pyscf"


@dataclass(frozen=True)
class QMBackend:
    """What a backend builder hands the embedding.

    ``calc_ll`` / ``calc_hl`` are the ASE calculators for EmbASI's ``calc_base_ll`` /
    ``calc_base_hl``; ``integrals`` must be built over the *same* calculators, so its
    ``veff_ll`` undoes exactly what EmbASI folded into ``F_emb``.
    """

    name: str
    calc_ll: Any
    calc_hl: Any
    integrals: AOIntegrals


BackendBuilder = Callable[[Any, int, Any], QMBackend]

_REGISTRY: dict[str, BackendBuilder] = {}

# Settings that only mean something to one backend, with the value that leaves them
# unset.  `make_backend` rejects a non-default value for any other backend, so a flag
# is never accepted and silently ignored.
# Each backend declares its own through `register_backend(backend_only_settings=...)`.
_BACKEND_ONLY_SETTINGS: dict[str, dict[str, Any]] = {}
# Backends that accept another backend's own settings (a variant of the same QM code).
_SETTINGS_FAMILY: dict[str, str] = {}


def register_backend(
    name: str,
    *,
    backend_only_settings: dict[str, Any] | None = None,
    settings_of: str | None = None,
) -> Callable[[BackendBuilder], BackendBuilder]:
    """Decorator registering ``build(atoms, charge, cfg) -> QMBackend`` under ``name``.

    ``backend_only_settings`` maps settings only this backend understands to their
    "unset" value; any other backend will refuse them when set.  ``settings_of`` names a
    backend whose own settings this one also accepts (a variant driving the same QM
    code, e.g. ``pyscf-callback`` of ``pyscf``).
    """

    def decorator(build: BackendBuilder) -> BackendBuilder:
        if name in _REGISTRY and _REGISTRY[name] is not build:
            raise ValueError(f"a QM backend named {name!r} is already registered")
        _REGISTRY[name] = build
        if backend_only_settings is not None:
            _BACKEND_ONLY_SETTINGS[name] = dict(backend_only_settings)
        if settings_of is not None:
            _SETTINGS_FAMILY[name] = settings_of
        return build

    return decorator


def _ensure_builtin_backends() -> None:
    # Imported lazily so that `backends` stays importable without PySCF/ASE on the path.
    import embasi_qiskit_integration.backends.pyscf  # noqa: F401
    import embasi_qiskit_integration.backends.siesta  # noqa: F401


def available_backends() -> list[str]:
    """Names accepted by ``--qm_code``."""
    _ensure_builtin_backends()
    return sorted(_REGISTRY)


def check_backend_settings(name: str, cfg: Any) -> None:
    """Raise if ``cfg`` sets an option that belongs to a backend other than ``name``."""
    _ensure_builtin_backends()  # so the built-ins' settings are registered
    for owner, settings in _BACKEND_ONLY_SETTINGS.items():
        if owner == name or _SETTINGS_FAMILY.get(name) == owner:
            continue
        for key, unset in settings.items():
            value = getattr(cfg, key, unset)
            if value != unset:
                raise ValueError(
                    f"--{key} is a {owner} option and has no effect with --qm_code {name}; drop it"
                )


def make_backend(name: str, atoms: Any, charge: int, cfg: Any) -> QMBackend:
    """Build the named QM backend for ``atoms`` (already in region-1-first order)."""
    _ensure_builtin_backends()
    if name not in _REGISTRY:
        raise NotImplementedError(
            f"--qm_code {name!r} is not available; implemented backends: "
            f"{', '.join(available_backends())}"
        )
    check_backend_settings(name, cfg)
    backend = _REGISTRY[name](atoms, charge, cfg)
    if backend.name != name:
        raise RuntimeError(f"backend {name!r} returned a {backend.name!r} backend")
    # A real backend must say what it can supply.  Without `capabilities` the adapter
    # treats the integrals as a test stand-in: capability gates let everything through
    # and missing protocol methods silently get PySCF's (or a spin-averaged) fallback.
    caps = getattr(backend.integrals, "capabilities", None)
    if not isinstance(caps, frozenset):
        raise TypeError(
            f"backend {name!r} returned integrals ({type(backend.integrals).__name__}) "
            f"without a `capabilities` frozenset; every registered backend must declare one"
        )
    return backend
