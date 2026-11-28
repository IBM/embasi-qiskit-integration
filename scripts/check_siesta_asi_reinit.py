#!/usr/bin/env python
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproducer: two ASI init/run/finalize cycles in one process.

On ``qcomp-5.4-asi`` (checked 2026-11-24, Ubuntu 24.04 / GCC 13 / OpenMPI /
ScaLAPACK) the **second** ``ASI_init`` segfaults inside
``initatom -> elec_corr_setup -> radfft -> gpfa`` because
``fft_gpfa_trigs_reset`` deallocates ``trigs`` without resetting ``nold``.  A
second ``ASI_run`` can also crash in ``m_fft`` (same issue with
``frsttime``/``OldMesh``).  The patch in
``scripts/patches/siesta-asi-embasi-fixes.patch`` fixes both; with it applied the
script prints ``REINIT_OK``.  This uses only ``asi4py`` + ``libsiesta.so``, so it
is independent of embasi-qiskit-integration and EmbASI.

TODO(siesta-asi-embasi-fixes): remove this note once SIESTA's ASI reinitialization
is fixed upstream.

Usage::

    ASI_LIB_PATH=/path/to/libsiesta.so \\
    SIESTA_PP_PATH=/path/to/pseudos \\
    python scripts/check_siesta_asi_reinit.py --cycles 2
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from ase.build import molecule
from ase.calculators.siesta import Siesta
from asi4py.asecalc import ASI_ASE_calculator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", default="H2O", help="ASE molecule name (default: H2O)")
    parser.add_argument("--cycles", type=int, default=2, help="init/run/finalize cycles")
    parser.add_argument("--work-dir", type=Path, help="scratch directory (default: a temp dir)")
    args = parser.parse_args()

    lib = os.environ.get("ASI_LIB_PATH")
    pseudo = os.environ.get("SIESTA_PP_PATH")
    if not lib or not pseudo:
        parser.error("ASI_LIB_PATH and SIESTA_PP_PATH must be set")
    if not Path(lib).is_file():
        parser.error(f"ASI_LIB_PATH does not name a file: {lib}")

    root = args.work_dir or Path(tempfile.mkdtemp(prefix="asi-reinit."))
    root.mkdir(parents=True, exist_ok=True)
    atoms = molecule(args.molecule)

    def init_via_ase(asi) -> None:
        calc = Siesta(xc="PBE", pseudo_path=pseudo, symlink_pseudos=False)
        calc.write_input(asi.atoms, properties=["forces"])

    for cycle in range(args.cycles):
        print(f"cycle {cycle}: ASI_init", flush=True)
        calc = ASI_ASE_calculator(
            lib, init_via_ase, None, atoms, work_dir=str(root / f"cycle{cycle}")
        )
        print(f"cycle {cycle}: ASI_run", flush=True)
        calc.get_potential_energy()
        print(f"cycle {cycle}: ASI_finalize", flush=True)
        calc.close()
        print(f"cycle {cycle}: done", flush=True)

    print("REINIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
