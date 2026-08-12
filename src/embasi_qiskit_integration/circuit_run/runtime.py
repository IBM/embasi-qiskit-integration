# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""IBM Quantum Runtime bitstring sampler (optional ``hardware`` extra).

Plugs the SQD pipeline into real quantum hardware via ``QiskitRuntimeService``.
Assuming credentials are configured (``QiskitRuntimeService.save_account(...)``
once, or the ``QISKIT_IBM_TOKEN`` env var).
"""

from __future__ import annotations

import logging

from embasi_qiskit_integration.circuit_run.base import SamplerMixin

logger = logging.getLogger(__name__)


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
        enable_readout_characterisation: measure the device's per-qubit readout
            error before submitting and pin the best 1-D chain, instead of letting
            the transpiler place the circuit against the backend's reported
            calibration (see :mod:`.characterisation` and :mod:`.layout`). Costs one
            extra short job. Off by default, and refused on a simulated backend,
            whose "measured" error is only its configured noise model.
        readout_error_threshold: readout-error ceiling for a usable qubit; relaxed
            automatically when pruning at it leaves no chain wide enough.
        n_rand_twirl: twirling randomizations in the characterisation job.
        n_shots_per_twirl: shots per randomization.
        hot_coupler_ps: append ``xslow`` postselection re-measurements to the
            characterisation program.

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
        enable_readout_characterisation: bool = False,
        readout_error_threshold: float = 0.03,
        n_rand_twirl: int = 300,
        n_shots_per_twirl: int = 25,
        hot_coupler_ps: bool = False,
    ):
        self.backend_name = backend
        self._service = service
        self._backend_obj = backend_obj
        self.min_num_qubits = min_num_qubits
        self.optimization_level = optimization_level
        self.default_shots = default_shots
        self.options = options
        self.enable_readout_characterisation = enable_readout_characterisation
        self.readout_error_threshold = readout_error_threshold
        self.n_rand_twirl = n_rand_twirl
        self.n_shots_per_twirl = n_shots_per_twirl
        self.hot_coupler_ps = hot_coupler_ps
        # Provenance of the last run: the chosen layout and its score, the avoided
        # qubits, the backend's reported noise, and the per-edge errors along the
        # layout actually used. Populated by :meth:`run`.
        self.hardware_characterisation: dict | None = None

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

    def _characterise(self, backend, circuits: list) -> dict | None:
        """Measure the device's readout error and pick a layout, or return ``None``.

        Returns ``None`` -- leaving :meth:`run` on the default-layout transpile --
        when characterisation is off, when ``samplomatic`` is unavailable, or when
        the backend is a local simulator. The last case is a refusal rather than an
        attempt: a ``Fake*`` device's "measured" readout error is whatever its noise
        model was configured with, so pruning on it would dress a fixed model up as
        a measurement and could reject qubits on a device that has none.

        A characterisation failure is logged and falls back rather than aborting the
        run: the default layout still produces valid (if noisier) counts, whereas
        raising would throw away a queued job's worth of work.
        """
        if not self.enable_readout_characterisation:
            return None

        from embasi_qiskit_integration.circuit_run.backend import _is_fake_backend
        from embasi_qiskit_integration.circuit_run.characterisation import (
            characterise_readout,
            samplomatic_available,
        )

        if _is_fake_backend(backend):
            logger.warning(
                "Skipping readout characterisation on simulated backend %r: its "
                "readout error is a configured noise model, not a measurement.",
                getattr(backend, "name", backend),
            )
            return None

        if not samplomatic_available():
            logger.warning(
                "Skipping readout characterisation: the optional 'samplomatic' "
                "package is not installed; using a default-layout transpile."
            )
            return None

        required_size = max(qc.num_qubits for qc in circuits)
        try:
            return characterise_readout(
                backend=backend,
                qubit_layout=list(range(backend.num_qubits)),
                required_size=required_size,
                n_rand_twirl=self.n_rand_twirl,
                n_shots_per_twirl=self.n_shots_per_twirl,
                hot_coupler_ps=self.hot_coupler_ps,
                readout_error_threshold=self.readout_error_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the run
            logger.warning(
                "Readout characterisation failed (%s: %s); falling back to a "
                "default-layout transpile.",
                type(exc).__name__,
                exc,
            )
            return None

    def run(
        self, circuits: list, shots: int | None = None, *, seed: int | None = None
    ) -> list[dict[str, int]]:
        """Transpile every circuit to the backend's ISA and submit as one job.

        One pass manager is built for the whole batch, and one ``SamplerV2`` job
        carries every circuit -- so a large ensemble costs one submission rather
        than one per circuit.
        """
        from embasi_qiskit_integration.circuit_run.backend import (
            edges_along_layout,
            prepare_isa,
            read_noise_from_backend,
            require_runtime,
            virtual_to_physical,
        )
        from embasi_qiskit_integration.circuit_run.counts import (
            counts_per_binding_from_pub_result,
        )

        require_runtime("RuntimeSampler")
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2

        shots = self.default_shots if shots is None else shots
        if not circuits:
            return []

        backend = self.resolve_backend(num_qubits=max(qc.num_qubits for qc in circuits))
        readout_characterisation = self._characterise(backend, list(circuits))

        if readout_characterisation is not None:
            # Place the circuits on the measured-best chain, over the pruned map.
            from qiskit.transpiler import CouplingMap

            from embasi_qiskit_integration.circuit_run.backend import (
                build_pinned_pass_manager,
            )

            pass_manager = build_pinned_pass_manager(
                backend,
                CouplingMap(readout_characterisation["coupling_map_pruned"]),
                readout_characterisation["best_layout"],
            )
            isa_circuits = list(pass_manager.run(list(circuits)))
        else:
            isa_circuits = prepare_isa(
                list(circuits),
                backend,
                optimization_level=self.optimization_level,
                seed_transpiler=seed,
            )

        # Record what the run actually used, for provenance alongside the counts.
        noise = read_noise_from_backend(backend)
        layout_map = virtual_to_physical(isa_circuits[0]) if isa_circuits else {}
        self.hardware_characterisation = {
            "backend": getattr(backend, "name", None),
            "readout_characterisation": readout_characterisation,
            "readout_info": noise,
            "layout_virtual_to_physical": {str(k): str(v) for k, v in layout_map.items()},
            "two_qubit_errors_on_layout": edges_along_layout(layout_map, noise),
        }

        sampler = RuntimeSamplerV2(mode=backend, options=self.options)
        result = sampler.run(isa_circuits, shots=shots).result()
        counts: list[dict[str, int]] = []
        for pub_result in result:
            counts.extend(counts_per_binding_from_pub_result(pub_result))
        return counts
