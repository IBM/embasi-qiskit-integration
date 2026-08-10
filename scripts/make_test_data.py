# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate the frozen test data committed under ``tests/data``.

Builds N2 / 6-31G, extracts the CAS(8o, 10e) active-space integrals, folds them
into an :class:`EmbeddedHamiltonian`, writes ``<stem>.fcidump`` + ``.npz``,
computes the FCI reference energy, and stores it in the npz ``meta``. Also emits
``mock_counts.json`` (frozen SQD sampling counts) when the SqDRIFT backend is
available.

Configured via ``pydantic-settings`` — every knob is a CLI flag (and an
``EQI_MKDATA_`` environment variable). Run with defaults to regenerate the
committed data::

    python scripts/make_test_data.py
    python scripts/make_test_data.py --bond-length 1.20 --ncas 6 --nelecas 6 \
        --stem n2_6o6e
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.solvers import FCISolver

DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"


class MakeTestData(BaseSettings):
    """Build and write the frozen N2 active-space test data."""

    model_config = SettingsConfigDict(env_prefix="EQI_MKDATA_", cli_parse_args=True)

    # Geometry / basis / active space (pinned defaults reproduce committed data).
    atom: str = "N 0 0 0; N 0 0 {bond_length}"
    basis: str = "6-31g"
    bond_length: float = 1.09  # Angstrom
    ncas: int = 8
    nelecas: int = 10

    # Output location.
    data_dir: Path = DATA_DIR
    stem: str = "n2_8o10e"

    # Frozen SQD sampling settings (kept in sync with the committed counts).
    mock_counts_shots: int = 100_000
    mock_counts_aer_seed: int = 42
    mock_counts_evolution_time: float = 1.0
    write_mock_counts: bool = True

    def cli_cmd(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ham = self.build_active_hamiltonian()

        fcidump_path = self.data_dir / f"{self.stem}.fcidump"

        # FCI reference on the active-space integrals (the oracle).
        ref_energy = FCISolver().solve(ham).energy
        print(f"FCI reference energy: {ref_energy:.10f} Ha")

        # Fold the reference energy and a content hash into meta, then write.
        sha = _integrals_sha(ham)
        ham.meta["reference_fci_energy"] = float(ref_energy)
        ham.meta["sha"] = sha
        fcidump.write(ham, fcidump_path)

        print(f"Wrote {fcidump_path}")
        print(f"Wrote {fcidump_path.with_suffix('.npz')}")
        print(f"integrals sha256[:16] = {sha}")

        if self.write_mock_counts:
            self._maybe_write_mock_counts(ham)

    def build_active_hamiltonian(self) -> EmbeddedHamiltonian:
        from pyscf import ao2mo, mcscf, scf
        from pyscf import gto as pyscf_gto

        atom = self.atom.format(bond_length=self.bond_length)
        mol = pyscf_gto.M(atom=atom, basis=self.basis, verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        mc = mcscf.CASCI(mf, self.ncas, self.nelecas)
        h1, e_core = mc.get_h1eff()
        h2 = ao2mo.restore(1, mc.get_h2eff(), self.ncas)

        na = self.nelecas // 2
        nb = self.nelecas - na
        meta = {
            "system": "N2",
            "basis": self.basis,
            "bond_length_angstrom": self.bond_length,
            "active_space": {"ncas": self.ncas, "nelecas": self.nelecas},
            "rhf_energy": float(mf.e_tot),
        }
        return EmbeddedHamiltonian(
            h1=np.asarray(h1),
            h2=np.asarray(h2),
            e_core=float(e_core),
            nelec=(na, nb),
            meta=meta,
        )

    def _maybe_write_mock_counts(self, ham: EmbeddedHamiltonian) -> None:
        try:
            from embasi_qiskit_integration.circuit_generator.sqdrift import sqdrift_available
            from embasi_qiskit_integration.circuit_run import merge_counts
            from embasi_qiskit_integration.circuit_run.aer import AerSampler
            from embasi_qiskit_integration.solvers import SQDSolver
        except ImportError as exc:
            print(f"Skipping mock_counts.json (quantum extras missing: {exc})")
            return

        if not sqdrift_available():
            print("Skipping mock_counts.json (qiskit-fermions not installed)")
            return

        # Generate through the same build -> prep -> sample path the solver uses,
        # so the frozen fixture's provenance is the real pipeline.
        solver = SQDSolver(
            AerSampler(),
            shots=self.mock_counts_shots,
            evolution_time=self.mock_counts_evolution_time,
            seed=self.mock_counts_aer_seed,
        )
        counts = merge_counts(
            solver.sampler.run(
                solver.build_circuits(ham),
                self.mock_counts_shots,
                seed=self.mock_counts_aer_seed,
            )
        )

        path = self.data_dir / "mock_counts.json"
        with path.open("w") as fh:
            json.dump(counts, fh)
        print(f"Wrote {path} ({len(counts)} distinct bitstrings, {sum(counts.values())} shots)")


def _integrals_sha(ham: EmbeddedHamiltonian) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(ham.h1, dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(ham.h2, dtype="<f8").tobytes())
    h.update(json.dumps([ham.e_core, list(ham.nelec)]).encode())
    return h.hexdigest()[:16]


def main() -> None:
    CliApp.run(MakeTestData)


if __name__ == "__main__":
    main()
