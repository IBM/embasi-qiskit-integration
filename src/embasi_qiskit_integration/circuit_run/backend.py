# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend resolution and ISA transpilation for hardware-backed sampling."""

from __future__ import annotations

from typing import Any


def require_runtime(what: str = "Backend resolution") -> None:
    """Raise a helpful ImportError if ``qiskit-ibm-runtime`` is not installed.

    Args:
        what: the capability named in the message, e.g. ``"RuntimeSampler"``.
    """
    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise ImportError(
            f"{what} requires qiskit-ibm-runtime; install the "
            "'hardware' extra: pip install -e '.[hardware]'"
        ) from exc


def resolve_backend(
    backend_name: str | None = None,
    *,
    service: Any = None,
    backend_obj: Any = None,
    min_num_qubits: int | None = None,
) -> tuple[Any, Any, bool]:
    """Resolve a backend to run on.

    Precedence: an already-resolved ``backend_obj`` wins; then a ``Fake*``
    name (resolved locally, no credentials); then a named real backend; then the
    least-busy operational backend with at least ``min_num_qubits`` qubits.

    Args:
        backend_name: backend to target, e.g. ``"ibm_kingston"`` or
            ``"FakeManilaV2"``. ``None`` selects the least-busy backend.
        service: an existing ``QiskitRuntimeService`` (else one is constructed
            from saved credentials / ``QISKIT_IBM_TOKEN``). Injectable for tests.
        backend_obj: an already-resolved backend; returned as-is.
        min_num_qubits: floor for least-busy selection.

    Returns:
        ``(backend, service_or_None, is_fake)``. ``service`` is ``None`` for a
        locally-resolved fake backend, since none was needed.
    """
    if backend_obj is not None:
        return backend_obj, service, _is_fake_backend(backend_obj)

    # Fake backends resolve locally and need no credentials -- the offline path.
    # Only consulted when no service was injected: an explicit service is the
    # caller saying "look names up here", and it may legitimately serve a backend
    # whose name happens to contain "fake".
    if service is None and backend_name is not None and _is_fake_provider_name(backend_name):
        require_runtime()
        from qiskit_ibm_runtime import fake_provider

        if not hasattr(fake_provider, backend_name):
            available = [
                name
                for name in dir(fake_provider)
                if name.startswith("Fake") and not name.startswith("Fake_")
            ]
            raise ValueError(
                f"Fake backend {backend_name!r} not found in "
                f"qiskit_ibm_runtime.fake_provider. Available (first 20): {available[:20]}"
            )
        return getattr(fake_provider, backend_name)(), None, True

    if service is None:
        require_runtime()
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()

    if backend_name is not None:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(min_num_qubits=min_num_qubits)
    return backend, service, _is_fake_backend(backend)


def prepare_isa(
    circuits: list,
    backend: Any,
    *,
    optimization_level: int = 3,
    seed_transpiler: int | None = None,
) -> list:
    """Transpile ``circuits`` to ``backend``'s ISA with one shared pass manager.

    Real backends only accept circuits in their own instruction set, so this must
    run before submission. Building the pass manager once and running the whole
    batch through it is what makes a large ensemble (e.g. many SqDRIFT
    randomizations) cheap to prepare.

    Args:
        circuits: circuits to transpile.
        backend: the transpilation target.
        optimization_level: preset transpiler level (0-3; 3 is most aggressive).
        seed_transpiler: seed for the stochastic passes, for reproducible layouts.

    Returns:
        The ISA circuits, in input order.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pass_manager = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend,
        seed_transpiler=seed_transpiler,
    )
    # ``PassManager.run`` accepts a list and preserves order.
    return list(pass_manager.run(circuits))


def _is_fake_provider_name(backend_name: str) -> bool:
    """True if ``backend_name`` names a ``fake_provider`` class (``FakeManilaV2``).

    Deliberately keyed on the ``Fake`` *prefix*, not a substring search: a real
    backend name is lowercase (``ibm_kingston``), and a loose ``"fake" in name``
    test would also capture names like ``ibm_fake`` that a caller means to look up
    through a service.
    """
    return backend_name.startswith("Fake")


def _is_fake_backend(backend: Any) -> bool:
    """True if ``backend`` is a local fake/simulated backend rather than hardware.

    Keyed on the class name so it holds for the whole ``fake_provider`` family
    without importing it (and without a ``qiskit-ibm-runtime`` dependency here).
    """
    return type(backend).__name__.startswith("Fake")
