# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""IBM Quantum Runtime bitstring sampler (optional ``hardware`` extra).

Plugs the SQD pipeline into real quantum hardware via ``QiskitRuntimeService``.
Assuming credentials are configured (``QiskitRuntimeService.save_account(...)``
once, or the ``QISKIT_IBM_TOKEN`` env var).
"""

from __future__ import annotations

from embasi_qiskit_integration.circuit_run.base import SamplerMixin


class RuntimeSampler(SamplerMixin):
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

    .. important::
       **Real hardware runs should enable measurement twirling**, which is not on by
       default::

           RuntimeSampler(options={"twirling": {"enable_measure": True}})

       Readout error is the noise channel this pipeline is most sensitive to: a
       flipped bit changes a sampled determinant's Hamming weight, so SQD's
       postselection discards that shot outright. The cost is wasted shot budget
       *and* a subspace biased toward whichever configurations happened to survive
       -- so it moves the recovered energy, not merely its variance.
       ``{"dynamical_decoupling": {"enable": True}}`` additionally suppresses
       idle-qubit decoherence on deep circuits.

       Note there is no sampler-side ``resilience_level``: that is an Estimator
       option, and passing it here raises a ``ValidationError``.
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

    def resolve_backend(self, *, num_qubits: int | None = None):
        """Resolve the backend to run on (cached on this instance after the first call).

        The cache lives here rather than in
        :func:`~embasi_qiskit_integration.circuit_run.backend.resolve_backend` so
        that repeated sampling with one sampler queries ``least_busy`` only once.
        """
        if self._backend_obj is not None:
            return self._backend_obj

        from embasi_qiskit_integration.circuit_run.backend import resolve_backend

        backend, service, _is_fake = resolve_backend(
            self.backend_name,
            service=self._service,
            min_num_qubits=self.min_num_qubits or num_qubits,
        )
        # Cache both: a resolved backend and any service constructed on the way.
        self._backend_obj = backend
        self._service = service
        return self._backend_obj

    def run(
        self, circuits: list, shots: int | None = None, *, seed: int | None = None
    ) -> list[dict[str, int]]:
        """Transpile every circuit to the backend's ISA and submit as one job.

        One pass manager is built for the whole batch, and one ``SamplerV2`` job
        carries every circuit -- so a large ensemble costs one submission rather
        than one per circuit.
        """
        from embasi_qiskit_integration.circuit_run.backend import prepare_isa, require_runtime
        from embasi_qiskit_integration.circuit_run.counts import (
            counts_per_binding_from_pub_result,
        )

        require_runtime("RuntimeSampler")
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2

        shots = self.default_shots if shots is None else shots
        if not circuits:
            return []

        backend = self.resolve_backend(num_qubits=max(qc.num_qubits for qc in circuits))

        # Real backends only accept circuits in their ISA. ``seed`` cannot make
        # hardware *sampling* reproducible, but it does pin the transpiler's
        # stochastic layout/routing passes -- which matter at optimization_level 3,
        # since a different physical layout means different noise.
        isa_circuits = prepare_isa(
            list(circuits),
            backend,
            optimization_level=self.optimization_level,
            seed_transpiler=seed,
        )

        sampler = RuntimeSamplerV2(mode=backend, options=self.options)
        result = sampler.run(isa_circuits, shots=shots).result()
        counts: list[dict[str, int]] = []
        for pub_result in result:
            counts.extend(counts_per_binding_from_pub_result(pub_result))
        return counts
