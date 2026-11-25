# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Opt-in check against a real, ASI-enabled SIESTA shared library.

Enable with ``SIESTA_ASI_REAL=1`` and provide:

* ``ASI_LIB_PATH`` -- ``libsiesta.so`` built with ``SIESTA_WITH_ASI=ON`` and
  ``BUILD_SHARED_LIBS=ON`` (single-rank, unpolarized Gamma);
* ``SIESTA_PP_PATH`` -- the pseudopotential directory.

This deliberately runs **one** ASI init/run/finalize cycle and checks the real
S/H extraction; the reinitialization patch in ``scripts/patches/`` allows
repeated cycles, and the full WF path is covered by
``tests/test_siesta_qcomp_real.py``.  Both real modules are skipped on GitHub
Actions and when Docker/SIESTA is unavailable.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.backends import siesta as S

pytestmark = [
    pytest.mark.siesta_real,
    pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="requires Docker/SIESTA; skipped on GitHub Actions",
    ),
    pytest.mark.skipif(
        os.environ.get("SIESTA_ASI_REAL") != "1",
        reason="set SIESTA_ASI_REAL=1 and provide ASI_LIB_PATH/SIESTA_PP_PATH",
    ),
]

# H2O with DZP in SIESTA's standard pseudopotentials; the branch's own
# Tests/21.Embedding/test_asi_h2o.cpp asserts the same size.
H2O_DZP_NAO = 23


def _require_real() -> None:
    version = S._asi4py_version()
    if version is None or version < S.MIN_ASI4PY:
        pytest.skip("asi4py >= 1.4.10 is required for SIESTA flavour 3")
    if not os.environ.get("ASI_LIB_PATH") or not os.environ.get("SIESTA_PP_PATH"):
        pytest.skip("ASI_LIB_PATH and SIESTA_PP_PATH must be set")


def test_real_siesta_single_evaluation(tmp_path):
    _require_real()

    from ase.build import molecule

    from embasi_qiskit_integration.backends._embasi_compat import install_siesta_adapter
    from embasi_qiskit_integration.backends.callback import LayerEvaluator

    install_siesta_adapter()
    cfg = SimpleNamespace(
        xc_ll="PBE",
        xc_hl="PBE",
        basis="sto-3g",
        spin=0,
        unrestricted=False,
        scf_max_cycle=1,
        siesta_basis="DZP",
        siesta_mesh_cutoff=200.0,
        siesta_workdir=str(tmp_path / "work"),
    )
    atoms = molecule("H2O")
    calc = S.siesta_calculator(cfg, xc="PBE", directory=".", max_scf_cycles=1, require_asi=True)
    evaluator = LayerEvaluator(atoms, calc, n_spin=1, label=str(tmp_path / "one"))
    out = evaluator(np.zeros((H2O_DZP_NAO, H2O_DZP_NAO)))

    assert out.overlap.shape == (H2O_DZP_NAO, H2O_DZP_NAO)
    assert np.allclose(np.diag(out.overlap), 1.0, atol=1e-10)
    assert out.hamiltonian[0].shape == (H2O_DZP_NAO, H2O_DZP_NAO)
    assert np.isfinite(out.energy)
    assert np.asarray(evaluator.basis_atoms).shape == (H2O_DZP_NAO,)
