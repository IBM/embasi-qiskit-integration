# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Live UNO-SPADE checks against real EmbASI (``localisation="UNO-SPADE"``).

Marked ``embasi``/``slow`` like the other live tests, and additionally skipped when the
installed EmbASI has no UNO-SPADE.  Run with::

    EMBASI_AVAILABLE=1 pytest -m embasi tests/test_uno_spade_live.py
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.embasi, pytest.mark.slow]
HA2EV = 27.211384500


@pytest.fixture
def oh_water(tmp_path, monkeypatch):
    pytest.importorskip("embasi.uno_spade_localisation")
    xyz = tmp_path / "oh_water.xyz"
    xyz.write_text(
        "5\n\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\nO 4.0 0.0 0.0\nH 4.0 0.0 0.96\nH 4.9 0.0 -0.3\n"
    )
    monkeypatch.chdir(tmp_path)
    return xyz


def _adapter(xyz, monkeypatch, xc_ll):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--xyz",
            str(xyz),
            "--active_atoms",
            "[0,1]",
            f"--xc_ll={xc_ll}",
            "--xc_hl=hf",
            "--spin",
            "1",
            "--unrestricted=True",
            "--spin_downfold=True",
            "--localisation",
            "UNO-SPADE",
            "--selector",
            "uno",
            "--uno_fill",
            "apc",
            "--apc_max_size",
            "[2,2]",
        ],
    )
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    wf = EmbeddingWorkflow()
    emb = wf._build_adapter(parallel=False)
    emb.run_low_level()
    return wf, emb


@pytest.mark.parametrize("xc_ll", ["hf", "pbe"])
def test_relaxed_determinant_reproduces_embasi_mean_field(oh_water, monkeypatch, xc_ll):
    """The relaxed embedded-UHF determinant, all of A active, through the per-spin downfold
    and energy assembly, must equal EmbASI's own X-in-HF total for the same partition --
    also for a KS low level, whose embedding potential is genuinely spin-dependent."""
    import pyscf
    from embasi.embedding import ProjectionEmbedding
    from pyscf.pbc.tools.pyscf_ase import PySCF, ase_atoms_to_pyscf

    from embasi_qiskit_integration.contract import SolverResult
    from embasi_qiskit_integration.embedding import build_atoms

    wf, emb = _adapter(oh_water, monkeypatch, xc_ll)
    emb.relax_active_hf()
    s = emb._s_arr
    n_env = emb._mo_spin(emb.p.mo_coeffs_B_LL, 0).shape[1]
    n_a = sum(emb._mo_spin(emb.p.mo_coeffs_A_LL, k).shape[1] for k in (0, 1))
    n_orb = s.shape[0] - n_env
    oa, ob = emb.build_orbitals_uno(max_size=(n_a, n_orb), fixed=True, fill="apc", use_relaxed=True)
    ham = emb.embedded_hamiltonian_spin((oa, ob))
    c = oa.c_active
    ga, gb = (c.T @ s @ d @ s @ c for d in emb._as_ao_pair(emb.p.A_HL.density_matrices_out))
    h2 = np.asarray(ham.h2).reshape((n_orb,) * 4)
    g = ga + gb
    e_det = (
        ham.e_core
        + np.sum(ham.h1a * ga)
        + np.sum(ham.h1b * gb)
        + 0.5 * np.einsum("pqrs,pq,rs->", h2, g, g)
        - 0.5 * (np.einsum("prqs,pq,rs->", h2, ga, ga) + np.einsum("prqs,pq,rs->", h2, gb, gb))
    )
    res = SolverResult(
        energy=e_det, rdm1=g, rdm2=np.zeros((n_orb,) * 4), rdm1a=ga, rdm1b=gb, diagnostics={}
    )
    total = emb.projection_energy(res, oa, ob).total

    atoms, charge = build_atoms(wf)
    mol = pyscf.M(atom=ase_atoms_to_pyscf(atoms), basis="sto-3g", spin=1, charge=charge)
    ref = ProjectionEmbedding(
        atoms,
        embed_mask=[1, 1, 2, 2, 2],
        calc_base_ll=PySCF(method=mol.UKS(xc=xc_ll)),
        calc_base_hl=PySCF(method=mol.UKS(xc="hf")),
        projection="level-shift",
        localisation="UNO-SPADE",
    )
    ref.run()
    assert total == pytest.approx(ref.DFT_AinB_total_energy / HA2EV, abs=1e-6)
