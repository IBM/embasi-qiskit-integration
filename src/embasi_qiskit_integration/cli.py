# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Command-line entry point: ``embasi-qiskit-integration solve <dir>``.

Reads a job (``<dir>/job.fcidump`` + sidecar), runs the requested solver, and
writes ``<dir>/result.npz`` atomically (or ``<dir>/result.ERROR`` on failure).
This is the process-B side of the two-process handoff; it needs neither EmbASI
nor FHI-aims.

Built on ``pydantic-settings``: each command is a settings model whose fields
map to CLI flags (and environment variables under the ``EQI_`` prefix).
"""

from __future__ import annotations

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


class SolveCommand(BaseSettings):
    """Solve a job directory and write ``result.npz``."""

    model_config = SettingsConfigDict(env_prefix="EQI_", cli_parse_args=False)

    directory: CliPositionalArg[str]
    solver: Literal["sqd", "fci"] = "sqd"
    sampler: Literal["aer", "mock", "runtime"] = "aer"
    counts: str | None = None
    backend: str | None = None  # runtime backend name; else least-busy
    optimization_level: int = 3  # runtime ISA-transpile level
    shots: int = 100_000
    seed: int | None = None
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
        return SQDSolver(self._build_sampler(), shots=self.shots, seed=self.seed)

    def _build_sampler(self):
        if self.sampler == "mock":
            if not self.counts:
                raise SystemExit("--sampler mock requires --counts <path>")
            from embasi_qiskit_integration.sampling.base import MockSampler

            return MockSampler(self.counts)
        if self.sampler == "aer":
            from embasi_qiskit_integration.sampling.aer import AerSampler

            return AerSampler()
        if self.sampler == "runtime":
            from embasi_qiskit_integration.sampling.runtime import RuntimeSampler

            # Uses configured IBM Quantum credentials; least-busy backend unless
            # --backend names one.
            return RuntimeSampler(
                backend=self.backend,
                optimization_level=self.optimization_level,
                default_shots=self.shots,
            )
        raise SystemExit(f"unknown sampler {self.sampler!r}")


class Cli(BaseSettings):
    """embasi-qiskit-integration command-line interface."""

    model_config = SettingsConfigDict(env_prefix="EQI_")

    solve: CliSubCommand[SolveCommand]

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
