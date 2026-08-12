# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend resolution and ISA transpilation for hardware-backed sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# Two-qubit basis gates tried in order when reading per-edge gate errors; the
# first one the target exposes wins (IBM devices differ across generations).
_TWO_QUBIT_GATE_CANDIDATES: tuple[str, ...] = ("ecr", "cz", "cx")


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


def build_pinned_pass_manager(
    backend: Any, coupling_map: Any, initial_layout: Sequence[int]
) -> Any:
    """Build a pass manager pinned to ``initial_layout`` over a pruned coupling map.

    This is the transpile half of noise-aware layout selection: once
    :mod:`.characterisation` has measured the device and picked a chain, the
    circuits must be placed on *that* chain rather than wherever the default layout
    passes would put them.

    Passes ``target=backend.target`` rather than ``backend=``. Supplying ``backend``
    together with ``coupling_map`` makes qiskit emit a ``UserWarning`` and discard
    the backend's gate durations and error rates; the target carries those while the
    pruned coupling map still restricts routing, so both survive warning-free.

    The post-optimization stage is replaced to match the reference:

    - ``FoldRzzAngle`` is a *correctness* requirement, not an optimization, on
      backends exposing fractional ``rzz``: the IBM ISA only accepts
      ``Rzz(theta)`` for ``theta`` in ``[0, pi/2]``, and an out-of-range angle is
      rejected at submission.
    - ``Optimize1qGatesDecomposition`` and ``RemoveIdentityEquivalent`` then clean
      up what folding leaves behind.
    """
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        Optimize1qGatesDecomposition,
        RemoveIdentityEquivalent,
    )
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

    target = getattr(backend, "target", None)
    pass_manager = generate_preset_pass_manager(
        target=target,
        coupling_map=coupling_map,
        initial_layout=list(initial_layout),
    )
    pass_manager.post_optimization = PassManager(
        [
            FoldRzzAngle(),
            Optimize1qGatesDecomposition(target=target),
            RemoveIdentityEquivalent(target=target),
        ]
    )
    return pass_manager


def read_noise_from_backend(backend: Any) -> dict[str, Any]:
    """Extract per-qubit readout errors and per-edge 2q gate errors from ``backend``.

    Values come back as plain dicts keyed by ``int`` qubit index or ``"u-v"`` edge
    string, so the result is JSON-serialisable alongside the counts. A missing value
    is *absent* rather than ``None``, so consumers never have to distinguish
    "unknown" from "known but null".

    This is the backend's *reported* calibration, which is what makes it cheap (no
    job) but also potentially hours stale -- :mod:`.characterisation` measures the
    same quantity live. Recorded either way as provenance for the run.
    """
    readout_error: dict[int, float] = {}
    p01: dict[int, float] = {}
    p10: dict[int, float] = {}
    two_qubit_errors: dict[str, float] = {}
    two_qubit_gate: str | None = None

    properties = None
    get_properties = getattr(backend, "properties", None)
    if callable(get_properties):
        try:
            properties = get_properties()
        except Exception:  # noqa: BLE001 - some fakes raise; fall back to empty
            properties = None

    if properties is not None:
        n_qubits = getattr(backend, "num_qubits", None) or len(
            getattr(properties, "qubits", []) or []
        )
        for qubit in range(n_qubits):
            # Each of the three is read independently: a device may report the
            # aggregate readout_error without the directional p01/p10 pair.
            try:
                readout_error[qubit] = float(properties.readout_error(qubit))
            except Exception:  # noqa: BLE001
                pass
            try:
                p01[qubit] = float(properties.qubit_property(qubit, "prob_meas1_prep0")[0])
            except Exception:  # noqa: BLE001
                pass
            try:
                p10[qubit] = float(properties.qubit_property(qubit, "prob_meas0_prep1")[0])
            except Exception:  # noqa: BLE001
                pass

    target = getattr(backend, "target", None)
    if target is not None:
        for name in _TWO_QUBIT_GATE_CANDIDATES:
            if name in target.operation_names:
                two_qubit_gate = name
                try:
                    for qargs, instruction_properties in target[name].items():
                        if instruction_properties is None or instruction_properties.error is None:
                            continue
                        if len(qargs) != 2:
                            continue
                        u, v = int(qargs[0]), int(qargs[1])
                        two_qubit_errors[f"{u}-{v}"] = float(instruction_properties.error)
                except Exception:  # noqa: BLE001
                    pass
                break

    return {
        "readout_error": readout_error,
        "p01": p01,
        "p10": p10,
        "two_qubit_gate": two_qubit_gate,
        "two_qubit_errors": two_qubit_errors,
    }


def virtual_to_physical(transpiled_circuit: Any) -> dict[int, int]:
    """Return a transpiled circuit's virtual-to-physical qubit mapping.

    Note this is *not* needed to interpret sampled bitstrings: ``SamplerV2`` returns
    counts in virtual/classical-bit order, so column ``i`` is virtual qubit ``i``
    whatever the physical placement. Its use is keying per-qubit hardware noise data
    back to the physical qubits the circuit actually ran on.
    """
    layout = getattr(transpiled_circuit, "layout", None)
    if layout is None:
        return {}
    virtual_layout = layout.final_virtual_layout()
    return {int(v._index): int(p) for v, p in virtual_layout.get_virtual_bits().items()}


def edges_along_layout(layout_map: dict[int, int], noise: dict[str, Any]) -> dict[str, float]:
    """Restrict ``noise['two_qubit_errors']`` to edges whose both ends are on the layout.

    Args:
        layout_map: virtual-to-physical mapping from :func:`virtual_to_physical`.
        noise: the dict returned by :func:`read_noise_from_backend`.
    """
    physical = set(layout_map.values())
    selected: dict[str, float] = {}
    for key, value in (noise.get("two_qubit_errors") or {}).items():
        u_str, _, v_str = key.partition("-")
        if not v_str:
            continue
        u, v = int(u_str), int(v_str)
        if u in physical and v in physical:
            selected[key] = value
    return selected


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
