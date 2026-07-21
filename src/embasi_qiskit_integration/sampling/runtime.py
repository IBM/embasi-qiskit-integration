# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""IBM Quantum Runtime bitstring sampler (optional ``hardware`` extra).

Plugs the SQD pipeline into real quantum hardware via ``QiskitRuntimeService``.
Assuming credentials are configured (``QiskitRuntimeService.save_account(...)``
once, or the ``QISKIT_IBM_TOKEN`` env var), sampling on hardware is just::

    from embasi_qiskit_integration.solvers import SQDSolver
    from embasi_qiskit_integration.sampling.runtime import RuntimeSampler

    solver = SQDSolver(RuntimeSampler(), shots=100_000, seed=24)   # least-busy backend
    result = solver.solve(ham)

or pick a backend explicitly with ``RuntimeSampler(backend="ibm_kingston")``.

The circuit is transpiled to the target backend's ISA before submission (real
backends only accept ISA circuits); a ``MockSampler``/``AerSampler`` skip that
step. Everything is lazily imported so ``qiskit-ibm-runtime`` is only needed when
hardware sampling is actually requested.
"""

from __future__ import annotations


class RuntimeSampler:
    """Sampler backed by ``qiskit-ibm-runtime`` (real hardware or cloud sim).

    Args:
        backend: backend name to target (e.g. ``"ibm_kingston"``). If omitted,
            the least-busy operational backend with enough qubits is chosen.
        service: an existing ``QiskitRuntimeService`` (else one is created from
            saved credentials / env). Injectable for testing.
        backend_obj: an already-resolved backend object (skips service lookup).
        min_num_qubits: floor for least-busy selection (default: the circuit's
            qubit count).
        optimization_level: preset transpiler level for the ISA translation
            (0-3; default 3, the most aggressive optimization).
        default_shots: shot budget when ``sample`` is called without ``shots``.
        options: ``SamplerOptions`` (or dict) forwarded to ``SamplerV2``.
    """

    def __init__(
        self,
        backend: str | None = None,
        *,
        service=None,
        backend_obj=None,
        min_num_qubits: int | None = None,
        optimization_level: int = 3,
        default_shots: int = 100_000,
        options=None,
    ):
        self.backend_name = backend
        self._service = service
        self._backend_obj = backend_obj
        self.min_num_qubits = min_num_qubits
        self.optimization_level = optimization_level
        self.default_shots = default_shots
        self.options = options

    def _require_runtime(self):
        try:
            import qiskit_ibm_runtime  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "RuntimeSampler requires qiskit-ibm-runtime; install the "
                "'hardware' extra: pip install -e '.[hardware]'"
            ) from exc

    def service(self):
        """The (lazily created) ``QiskitRuntimeService``.

        Created from configured credentials (saved account or the
        ``QISKIT_IBM_TOKEN`` env var) when not injected.
        """
        if self._service is None:
            self._require_runtime()
            from qiskit_ibm_runtime import QiskitRuntimeService

            self._service = QiskitRuntimeService()
        return self._service

    def resolve_backend(self, *, num_qubits: int | None = None):
        """Resolve the backend to run on (cached after first call)."""
        if self._backend_obj is not None:
            return self._backend_obj
        svc = self.service()
        if self.backend_name is not None:
            self._backend_obj = svc.backend(self.backend_name)
        else:
            floor = self.min_num_qubits or num_qubits
            self._backend_obj = svc.least_busy(min_num_qubits=floor)
        return self._backend_obj

    def sample(
        self, circuit, shots: int | None = None, *, seed: int | None = None
    ) -> dict[str, int]:
        self._require_runtime()
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2

        # seed is accepted for API symmetry; hardware sampling is not seedable.
        del seed
        shots = self.default_shots if shots is None else shots

        backend = self.resolve_backend(num_qubits=circuit.num_qubits)

        # Real backends only accept circuits in their ISA — transpile to the
        # backend target before submitting.
        pass_manager = generate_preset_pass_manager(
            optimization_level=self.optimization_level, backend=backend
        )
        isa_circuit = pass_manager.run(circuit)

        sampler = RuntimeSamplerV2(mode=backend, options=self.options)
        result = sampler.run([isa_circuit], shots=shots).result()
        data = result[0].data
        return {str(k): int(v) for k, v in _counts(data).items()}


def _counts(data_bin):
    """Extract counts from a SamplerV2 result DataBin's single register."""
    if hasattr(data_bin, "meas"):
        return data_bin.meas.get_counts()
    names = list(data_bin.keys())
    if len(names) != 1:
        raise ValueError(
            f"expected a single classical register, found {names!r}; "
            "measure into one register or a register named 'meas'"
        )
    return data_bin[names[0]].get_counts()
