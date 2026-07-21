# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Standalone end-to-end SQD demo: integrals -> SqDRIFT -> Aer -> SQD vs FCI.

Executable documentation of the full quantum path. Configured via
``pydantic-settings`` (CLI flags + ``EQI_SQD_`` env vars)::

    python scripts/sqd_prototype.py
    python scripts/sqd_prototype.py --shots 200000 --seed 7 --tolerance 1e-3

Uses the frozen N2 CAS(8o, 10e) integrals under ``tests/data``. If
``qiskit-fermions`` is installed, it builds and samples the SqDRIFT ansatz
with Aer; otherwise it replays the committed ``mock_counts.json`` (same source),
so the demo runs in any environment. Prints the SQD-vs-FCI energy delta and
asserts it is within ``--tolerance`` Ha.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.sampling.base import MockSampler
from embasi_qiskit_integration.solvers import FCISolver
from embasi_qiskit_integration.sqd.driver import run_sqd

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


class SqdPrototype(BaseSettings):
    """Run the integrals -> SqDRIFT -> Aer -> SQD vs FCI demo."""

    model_config = SettingsConfigDict(env_prefix="EQI_SQD_", cli_parse_args=True)

    data_dir: Path = DATA_DIR
    stem: str = "n2_8o10e"
    shots: int = 100_000
    seed: int = 24
    tolerance: float = 2e-3
    evolution_time: float = 1.0
    samples_per_batch: int = 300
    num_batches: int = 5
    max_iterations: int = 5

    def cli_cmd(self) -> None:
        ham = fcidump.read(self.data_dir / f"{self.stem}.fcidump")

        fci = FCISolver().solve(ham)
        print(f"FCI energy:  {fci.energy:.10f} Ha")

        counts, source = self._sample_counts(ham)
        print(
            f"counts:      {len(counts)} distinct bitstrings "
            f"({sum(counts.values())} shots) via {source}"
        )

        sqd = run_sqd(
            ham,
            counts,
            samples_per_batch=self.samples_per_batch,
            num_batches=self.num_batches,
            max_iterations=self.max_iterations,
            seed=self.seed,
        )
        delta = sqd.energy - fci.energy
        print(f"SQD energy:  {sqd.energy:.10f} Ha")
        print(f"delta:       {delta:+.2e} Ha (tolerance {self.tolerance:.0e})")

        assert abs(delta) <= self.tolerance, (
            f"SQD-FCI delta {delta:.2e} exceeds {self.tolerance:.0e} Ha"
        )
        print("OK: SQD within tolerance of FCI.")

    def _sample_counts(self, ham) -> tuple[dict[str, int], str]:
        """Return (counts, source): Aer over SqDRIFT if available, else replay."""
        from embasi_qiskit_integration.circuits.sqdrift import sqdrift_available

        if sqdrift_available():
            from qiskit import transpile
            from qiskit_aer import AerSimulator

            from embasi_qiskit_integration.circuits.sqdrift import build_sqdrift_circuits

            circuits = build_sqdrift_circuits(ham, method="exact", time=self.evolution_time)
            sim = AerSimulator()
            tqc = transpile(circuits[0], sim, optimization_level=0)
            counts = sim.run(tqc, shots=self.shots, seed_simulator=42).result().get_counts()
            return {k: int(v) for k, v in counts.items()}, "aer+sqdrift"

        counts = MockSampler(self.data_dir / "mock_counts.json").sample(
            circuit=None, shots=self.shots
        )
        return counts, "frozen mock_counts.json"


def main() -> None:
    CliApp.run(SqdPrototype)


if __name__ == "__main__":
    main()
