# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real WF-in-DFT check against the same-pseudopotential PySCF FCI.

Enable with ``SIESTA_QCOMP_REAL=1`` and provide:

* ``ASI_LIB_PATH`` -- ``libsiesta.so`` built with ``SIESTA_WITH_ASI=ON``,
  ``SIESTA_WITH_QCOMP=ON`` and ``SIESTA_WITH_HYBRIDXC=ON``;
* ``SIESTA_PP_PATH`` -- pseudopotentials.

Built with the ASI fixes in ``scripts/patches/`` and a libint 1.x built with
``MAX_AM_ERI=5``; see ``scripts/patches/README.md``.  PySCF is imported before
libsiesta so its h5py does not meet a conflicting HDF5 loaded by NetCDF.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

# Load h5py via PySCF before libsiesta, but only when this opt-in module can run.
pyscf_fci = pytest.importorskip("pyscf.fci") if os.environ.get("SIESTA_QCOMP_REAL") == "1" else None
from ase.build import molecule  # noqa: E402
from embasi.embedding import ProjectionEmbedding  # noqa: E402

from embasi_qiskit_integration.backends import siesta as S  # noqa: E402
from embasi_qiskit_integration.backends._embasi_compat import (  # noqa: E402
    install_siesta_adapter,
    prepare_projection_embedding,
)
from embasi_qiskit_integration.backends.base import (  # noqa: E402
    CAP_ARBITRARY_DM_VEFF,
    CAP_ERI,
    CAP_EXACT_EXCHANGE,
)
from embasi_qiskit_integration.projection_embedding_adapter import (  # noqa: E402
    ProjectionEmbeddingAdapter,
)
from embasi_qiskit_integration.solvers import FCISolver  # noqa: E402

pytestmark = [
    pytest.mark.siesta_real,
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="requires Docker/SIESTA; skipped on GitHub Actions",
    ),
    pytest.mark.skipif(
        os.environ.get("SIESTA_QCOMP_REAL") != "1",
        reason="set SIESTA_QCOMP_REAL=1 and provide a QCOMP ASI library",
    ),
]


def test_real_qcomp_wf_in_dft_matches_same_pseudopotential_pyscf_fci(tmp_path):
    version = S._asi4py_version()
    if version is None or version < S.MIN_ASI4PY:
        pytest.skip("asi4py >= 1.4.10 is required for SIESTA flavour 3")
    if not os.environ.get("ASI_LIB_PATH") or not os.environ.get("SIESTA_PP_PATH"):
        pytest.skip("ASI_LIB_PATH and SIESTA_PP_PATH must be set")

    install_siesta_adapter()
    cfg = SimpleNamespace(
        xc_ll="PBE",
        xc_hl="HF",
        basis="sto-3g",
        spin=0,
        unrestricted=False,
        scf_max_cycle=100,
        relax_hf=False,
        siesta_basis="DZP",
        siesta_mesh_cutoff=200.0,
        siesta_workdir=str(tmp_path / "work"),
    )
    atoms = molecule("H2O")
    backend = S.build_siesta_backend(atoms, 0, cfg)
    capabilities = backend.integrals.capabilities
    assert {CAP_ERI, CAP_EXACT_EXCHANGE, CAP_ARBITRARY_DM_VEFF} <= capabilities

    projection = ProjectionEmbedding(
        atoms,
        [1, 2, 2],
        backend.calc_ll,
        backend.calc_hl,
        localisation="SPADE",
        projection="level-shift",
        run_dir=str(tmp_path / "projection"),
        parallel=False,
        gc=False,
    )
    prepare_projection_embedding(projection)
    adapter = ProjectionEmbeddingAdapter(projection, backend.integrals, mu=1.0e6)
    # The valence-aware electron count gives 8 electrons for H2O; the automatic
    # SPADE gap then splits the NAO basis 4e/4e.  The workflow's --a_nmos fixes
    # three spatial A orbitals (6e on O), the chemically neutral split.
    adapter.run_low_level(a_nmos=3)

    assert backend.integrals._eri is not None
    assert backend.integrals._eri.n_eri > 0
    orbitals = adapter.build_orbitals(n_frozen_occ=0, n_virtual=4)
    ham = adapter.embedded_hamiltonian(orbitals)
    # Pseudopotential valence count: H2O has 8 electrons, and the manual SPADE
    # state gives neutral O 6e in 7 A orbitals (3 occupied + 4 virtual).
    assert ham.nelec == (3, 3)
    assert ham.norb == 7
    assert np.isfinite(ham.e_core)
    assert ham.h2.shape == (ham.norb,) * 4

    # --- invariants on the SIESTA Hamiltonian -------------------------------
    s = adapter._s_arr
    c = orbitals.c_active
    assert np.allclose(c.T @ s @ c, np.eye(ham.norb), atol=1e-8)
    # Both the localized A density and the correlated RDM integrate to 6e.
    assert np.einsum("ij,ji->", s, adapter._dm_a_arr) == pytest.approx(6.0, abs=1e-6)
    # The embedded one-body operator must be Hermitian and the ERIs must have
    # the chemists'-notation permutation symmetries.
    assert np.allclose(ham.h1, ham.h1.T, atol=1e-10)
    assert np.allclose(ham.h2, ham.h2.transpose(1, 0, 2, 3), atol=1e-10)
    assert np.allclose(ham.h2, ham.h2.transpose(0, 1, 3, 2), atol=1e-10)
    assert np.allclose(ham.h2, ham.h2.transpose(2, 3, 0, 1), atol=1e-10)

    # Exact same-pseudopotential reference: call PySCF's FCI kernel directly
    # on the SIESTA h1/h2/e_core (not through FCISolver), so the comparison
    # uses the same pseudopotential, basis, and Hamiltonian.
    from pyscf import fci

    e_direct, _ci = fci.direct_spin1.kernel(ham.h1, ham.h2, ham.norb, ham.nelec)
    result = FCISolver().solve(ham)
    assert float(e_direct) + ham.e_core == pytest.approx(result.energy, abs=1e-9)
    energy = adapter.projection_energy(result, orbitals)
    dm_hl = adapter.rdm1_ao(result.rdm1, orbitals)
    assert np.einsum("ij,ji->", s, dm_hl) == pytest.approx(6.0, abs=1e-6)
    assert abs(float(np.einsum("ij,ji->", dm_hl, adapter.p_b))) < 1e-6
    assert np.isfinite(energy.total) and np.isfinite(energy.e_high_A)
    assert np.isfinite(energy.e_low_A) and np.isfinite(energy.e_low_total)
    assert abs(energy.footing_shift) < 1e-6
    assert abs(energy.projector_leak) < 1e-6
    assert energy.total == pytest.approx(
        energy.e_low_total - energy.e_low_A + energy.e_high_A + energy.correction
    )
