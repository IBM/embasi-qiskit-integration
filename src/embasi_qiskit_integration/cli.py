# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point: ``embasi-qiskit-integration solve <dir>``.

Reads a job (``<dir>/job.fcidump`` + sidecar), runs the requested solver, and
writes ``<dir>/result.npz`` atomically (or ``<dir>/result.ERROR`` on failure).
``import-fcidump <file> <dir>`` turns an external FCIDUMP (WEST/QDET, CP2K, ...) into
such a job directory.
This is the process-B side of the two-process handoff; it needs neither EmbASI
nor FHI-aims.

Built on ``pydantic-settings``: each command is a settings model whose fields
map to CLI flags (and environment variables under the ``EQI_`` prefix).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import ClassVar, Literal

from pydantic_settings import (
    BaseSettings,
    CliApp,
    CliPositionalArg,
    CliSubCommand,
    SettingsConfigDict,
)

from embasi_qiskit_integration import ipc
from embasi_qiskit_integration._sweep import FloatSweep, IntSweep


class SolveCommand(BaseSettings):
    """Solve a job directory and write ``result.npz``."""

    model_config = SettingsConfigDict(env_prefix="EQI_", cli_parse_args=False)

    directory: CliPositionalArg[str]
    solver: Literal["sqd", "fci"] = "sqd"
    sampler: Literal["aer", "mock", "runtime"] = "runtime"
    # Which Aer simulator, when --sampler aer; None takes the package default (MPS).
    aer_method: str | None = None
    # MPS only; the cap is what buys the memory saving.
    mps_max_bond_dimension: int | None = None
    mps_truncation_threshold: float | None = None
    counts: str | None = None
    backend: str | None = None  # runtime backend name; else least-busy
    optimization_level: int = 1  # runtime ISA-transpile level (0-3)
    shots: int = 1_000  # per circuit for SQD
    seed: int = 42
    optimize: bool | None = None
    time_limit: float = 10.0  # per-solve wall-clock limit for the relabel solve (0 = none)
    # Relabel backend: HiGHS MILP, or OR-Tools CP-SAT (shallower circuits above ~16 modes
    # when given threads; cpsat_workers multiplies with workers).
    relabel_solver: Literal["highs", "cpsat"] = "cpsat"
    cpsat_workers: int = 1
    cpsat_deterministic: bool = False  # reproducible CP-SAT (interleaved, weaker search)
    # Seed each evolution time's draws independently instead of sharing them.
    per_time_randomization: bool = False
    # relabel_cache: reuse one permutation per sampled model (False: solve every circuit).
    # highs_deterministic: pin each HiGHS solve to one thread.
    relabel_cache: bool = True
    highs_deterministic: bool = False
    # Processes used to build the circuit ensemble. 0 means one per CPU.
    workers: int = 1
    # SQD solver parameters
    sqd_method: Literal["exact", "qdrift"] = "qdrift"
    # Sweep axes: the ensemble spans time x num_groups, pooled into one SQD run.
    evolution_time: FloatSweep = [1.0, 2.0, 3.0]
    num_groups: IntSweep = [10]
    num_randomizations: int = 500  # for method="qdrift", random circuits per combination
    measure_twirling: bool = True
    # Idle-qubit decoherence suppression; off by default (it lengthens the schedule).
    dynamical_decoupling: bool = False
    sampler_options: str | None = None
    enable_readout_characterisation: bool = False
    readout_error_threshold: float = 0.03
    n_rand_twirl: int = 300  # twirling randomizations in the characterisation job
    n_shots_per_twirl: int = 25
    hot_coupler_ps: bool = False  # append xslow postselection re-measurements
    watch: bool = False
    timeout: float = 300.0

    # Exit code stashed by cli_cmd so main() can return it (CliApp.run returns
    # the model instance, not the command's return value).
    _exit_code: int = 0

    def cli_cmd(self) -> None:
        self._exit_code = self._run()

    def _run(self) -> int:
        job_dir = Path(self.directory)

        if self.watch:
            _wait_for_job(job_dir, timeout=self.timeout)

        try:
            ham = ipc.read_job(job_dir)
        except Exception as exc:  # noqa: BLE001 - report any read/parse failure
            ipc.write_error(exc, job_dir)
            print(f"error: failed to read job in {job_dir}: {exc}", file=sys.stderr)
            return 1

        try:
            solver = self._build_solver()
            result = ipc.rank0_solve(solver, ham)
        except Exception as exc:  # noqa: BLE001 - persist traceback for process A
            ipc.write_error(exc, job_dir)
            print(f"error: solve failed: {exc}", file=sys.stderr)
            return 1

        path = ipc.write_result(result, job_dir)
        print(f"energy = {result.energy:.10f} Ha")
        print(f"wrote {path}")
        return 0

    def _build_solver(self):
        from embasi_qiskit_integration.solvers import FCISolver, SQDSolver

        if self.solver == "fci":
            return FCISolver()
        return SQDSolver(
            self._build_sampler(),
            shots=self.shots,
            seed=self.seed,
            optimize=self.optimize,
            time_limit=self.time_limit,
            relabel_solver=self.relabel_solver,
            cpsat_workers=self.cpsat_workers,
            cpsat_deterministic=self.cpsat_deterministic,
            per_time_randomization=self.per_time_randomization,
            relabel_cache=self.relabel_cache,
            highs_deterministic=self.highs_deterministic,
            workers=self.workers,
            method=self.sqd_method,
            evolution_time=self.evolution_time,
            num_groups=self.num_groups,
            num_randomizations=self.num_randomizations,
        )

    def _sampler_options(self) -> dict | None:
        """Assemble the ``SamplerV2`` options from the flags, or ``None``.

        Only meaningful for ``--sampler runtime``; the local and replay samplers
        take no options, so nothing is built for them (``build_sampler`` rejects
        options it cannot use, and silently dropping the twirling flag on a
        simulator run would be misleading either way).

        ``--sampler_options`` is merged *over* the flags one level deep, so
        ``{"twirling": {"num_randomizations": 64}}`` refines the twirling block
        rather than replacing it wholesale.
        """
        if self.sampler != "runtime":
            return None

        options: dict = {
            "twirling": {"enable_measure": self.measure_twirling},
            "dynamical_decoupling": {"enable": self.dynamical_decoupling},
        }

        if self.sampler_options:
            try:
                overrides = json.loads(self.sampler_options)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"--sampler_options is not valid JSON: {exc}") from exc
            if not isinstance(overrides, dict):
                raise SystemExit(
                    f"--sampler_options must be a JSON object, got {type(overrides).__name__}"
                )
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(options.get(key), dict):
                    options[key] = {**options[key], **value}
                else:
                    options[key] = value

        return options

    def _build_sampler(self):
        """Construct the requested sampler via the shared dispatch.

        ``runtime`` uses configured IBM Quantum credentials and the least-busy
        backend unless ``--backend`` names one. Dispatch errors are re-raised as
        ``SystemExit`` so the CLI reports them as usage errors rather than a
        traceback.
        """
        from embasi_qiskit_integration.circuit_run import build_sampler

        try:
            return build_sampler(
                self.sampler,
                counts=self.counts,
                backend=self.backend,
                optimization_level=self.optimization_level,
                default_shots=self.shots,
                aer_method=self.aer_method,
                mps_max_bond_dimension=self.mps_max_bond_dimension,
                mps_truncation_threshold=self.mps_truncation_threshold,
                options=self._sampler_options(),
                enable_readout_characterisation=self.enable_readout_characterisation,
                readout_error_threshold=self.readout_error_threshold,
                n_rand_twirl=self.n_rand_twirl,
                n_shots_per_twirl=self.n_shots_per_twirl,
                hot_coupler_ps=self.hot_coupler_ps,
            )
        except ValueError as exc:
            raise SystemExit(f"--sampler {self.sampler!r}: {exc}") from exc


class ImportFcidumpCommand(BaseSettings):
    """Turn an externally built FCIDUMP into a job directory ``solve`` can run.

    The front end for active-space Hamiltonians this package did not downfold itself:
    WEST/QDET (Quantum ESPRESSO), CP2K's ``&ACTIVE_SPACE &FCIDUMP``, Molpro, ...
    Restricted and ``UHF=1`` files are both read; an unrestricted one keeps its
    ``(h1a, h1b)`` / ``(aa, ab, bb)`` blocks (in the job's ``.npz`` sidecar), which
    ``--solver fci`` consumes exactly.

    These Hamiltonians come from a *different* embedding than this package's projection
    embedding (QDET uses a screened interaction with a double-counting correction; CP2K's
    is range-separated DFT embedding), so their energies are not comparable with a
    ``--qm_code`` run on the same system.
    """

    model_config = SettingsConfigDict(env_prefix="EQI_IMPORT_", cli_parse_args=False)

    fcidump: CliPositionalArg[str]
    directory: CliPositionalArg[str]
    # Override the header's spin sector, for a writer that emits MS2 unsigned.
    nelec_alpha: int | None = None
    nelec_beta: int | None = None
    source: str | None = None  # free-text provenance stored in the job's meta

    _exit_code: int = 0

    def cli_cmd(self) -> None:
        self._exit_code = self._run()

    def _run(self) -> int:
        from embasi_qiskit_integration.hamiltonian import fcidump

        src = Path(self.fcidump)
        if not src.is_file():
            print(f"error: no such FCIDUMP: {src}", file=sys.stderr)
            return 1
        if (self.nelec_alpha is None) != (self.nelec_beta is None):
            print("error: give --nelec_alpha and --nelec_beta together", file=sys.stderr)
            return 1
        try:
            ham = fcidump.read(src)
        except Exception as exc:  # noqa: BLE001 - report any parse failure
            print(f"error: could not read {src}: {exc}", file=sys.stderr)
            return 1
        updates: dict = {
            "meta": {
                **ham.meta,
                "imported_from": str(src.resolve()),
                **({"source": self.source} if self.source else {}),
            }
        }
        if self.nelec_alpha is not None and self.nelec_beta is not None:
            nelec = (int(self.nelec_alpha), int(self.nelec_beta))
            if sum(nelec) != sum(ham.nelec):
                print(
                    f"error: --nelec_alpha + --nelec_beta = {sum(nelec)} but the FCIDUMP "
                    f"has NELEC={sum(ham.nelec)}",
                    file=sys.stderr,
                )
                return 1
            updates["nelec"] = nelec
        ham = ham.model_copy(update=updates)
        import warnings

        with warnings.catch_warnings():
            # A spin-resolved job is lossless through the sidecar; the warning is for
            # callers expecting a self-contained restricted FCIDUMP.
            warnings.simplefilter("ignore", UserWarning)
            path = ipc.write_job(ham, Path(self.directory))
        spin = "unrestricted" if ham.is_spin_dependent else "restricted"
        print(f"imported {spin} CAS({sum(ham.nelec)}e, {ham.norb}o), nelec={ham.nelec}")
        print(f"wrote {path}")
        return 0


class Cli(BaseSettings):
    """embasi-qiskit-integration command-line interface."""

    model_config = SettingsConfigDict(env_prefix="EQI_")

    solve: CliSubCommand[SolveCommand]
    import_fcidump: CliSubCommand[ImportFcidumpCommand]

    # Propagated up from the chosen subcommand.
    exit_code: ClassVar[int] = 0

    def cli_cmd(self) -> None:
        sub = CliApp.run_subcommand(self)
        Cli.exit_code = getattr(sub, "_exit_code", 0)


def _wait_for_job(job_dir: Path, *, timeout: float, poll: float = 0.2) -> None:
    target = job_dir / f"{ipc.JOB_STEM}.fcidump"
    deadline = time.monotonic() + timeout
    while not target.exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"timed out waiting for {target}")
        time.sleep(poll)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    Cli.exit_code = 0
    CliApp.run(Cli, cli_args=argv)
    return Cli.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
