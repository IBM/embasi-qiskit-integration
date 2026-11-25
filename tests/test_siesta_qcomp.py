# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SIESTA-QCOMP NetCDF ERI provider and ERI-based exact exchange."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.backends import siesta as S
from embasi_qiskit_integration.backends._siesta_qcomp import SiestaEriProvider
from embasi_qiskit_integration.backends.base import (
    CAP_ARBITRARY_DM_VEFF,
    CAP_ERI,
    CAP_EXACT_EXCHANGE,
)
from embasi_qiskit_integration.backends.callback import CallbackIntegrals, Evaluation

h5py = pytest.importorskip("h5py")

# Two-orbital H2 values from a real QCOMP DROP run (symmetry-unique quartets).
H2_INDICES = np.array(
    [[1, 1, 1, 1], [2, 2, 1, 1], [2, 2, 2, 2], [2, 2, 2, 1], [2, 1, 1, 1], [2, 1, 2, 1]],
    dtype=np.int32,
)
H2_VALUES = np.array(
    [0.48666418, 0.37592095, 0.48666418, 0.33174232, 0.33174232, 0.2567824],
    dtype=np.float32,
)


def write_qcomp_file(
    path, indices=H2_INDICES, values=H2_VALUES, *, unit="Hartree", transpose=False
):
    with h5py.File(path, "w") as handle:
        data = indices.T if transpose else indices
        handle.create_dataset("eri_indx", data=data)
        handle.create_dataset("eri_vals", data=np.asarray(values, dtype=np.float32))
        if unit is not None:
            handle["eri_vals"].attrs["unit"] = unit
    return path


def test_provider_expands_permutations_and_contracts(tmp_path):
    provider = SiestaEriProvider(write_qcomp_file(tmp_path / "eris.nc"), n_ao=2)
    assert provider.n_eri == len(H2_VALUES)
    tensor = provider.dense()
    assert tensor.shape == (2, 2, 2, 2)
    # 8-fold permutational symmetry after expansion.
    for perm in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        assert np.allclose(tensor, tensor.transpose(perm))

    dm = np.array([[1.2, 0.3], [0.3, 0.7]])
    j_expected = np.einsum("pqrs,rs->pq", tensor, dm)
    k_expected = np.einsum("prqs,rs->pq", tensor, dm)
    j, k = provider.jk(dm)
    assert np.allclose(j, j_expected)
    assert np.allclose(k, k_expected)

    coeff = np.array([[0.9, 0.1], [0.2, 0.8]])
    mo_expected = np.einsum("pqrs,pi,qj,rk,sl->ijkl", tensor, coeff, coeff, coeff, coeff)
    assert np.allclose(provider.eri_mo(coeff), mo_expected)
    assert np.allclose(provider.eri_mo_mixed(coeff, coeff), mo_expected)


def test_provider_accepts_fortran_dimension_order(tmp_path):
    provider = SiestaEriProvider(write_qcomp_file(tmp_path / "eris.nc", transpose=True), n_ao=2)
    assert provider.dense().shape == (2, 2, 2, 2)


def test_provider_keeps_given_nao_when_file_omits_high_indices(tmp_path):
    provider = SiestaEriProvider(write_qcomp_file(tmp_path / "eris.nc"), n_ao=4)
    assert provider.n_ao == 4
    assert provider.dense().shape == (4, 4, 4, 4)
    j, k = provider.jk(np.eye(4))
    assert j.shape == k.shape == (4, 4)


def test_provider_refuses_missing_empty_and_wrong_units(tmp_path):
    with pytest.raises(RuntimeError, match="no SIESTA-QCOMP ERI file"):
        SiestaEriProvider(tmp_path / "missing.nc", n_ao=2).dense()

    with pytest.raises(ValueError, match="holds no ERIs"):
        SiestaEriProvider(
            write_qcomp_file(
                tmp_path / "empty.nc", indices=np.empty((0, 4), dtype=np.int32), values=[]
            ),
            n_ao=2,
        ).dense()

    with pytest.raises(ValueError, match="expected Hartree"):
        SiestaEriProvider(write_qcomp_file(tmp_path / "ryd.nc", unit="Rydberg"), n_ao=2).dense()


def _zero_evaluator(dm):
    return Evaluation(hamiltonian=(np.zeros((2, 2)),), overlap=np.eye(2), energy=0.0)


def test_callback_integrals_uses_eri_jk_and_declares_capabilities(tmp_path):
    provider = SiestaEriProvider(write_qcomp_file(tmp_path / "eris.nc"), n_ao=2)
    ints = CallbackIntegrals(
        _zero_evaluator,
        _zero_evaluator,
        None,  # no UHF evaluator
        nao=2,
        eri_provider=provider,
        arbitrary_dm_veff=True,
    )
    capabilities = ints.capabilities
    assert CAP_ERI in capabilities
    assert CAP_EXACT_EXCHANGE in capabilities
    assert CAP_ARBITRARY_DM_VEFF in capabilities

    dm = np.array([[1.1, 0.2], [0.2, 0.9]])
    j_expected, k_expected = provider.jk(dm)
    j, k = ints.get_jk(dm)
    assert np.allclose(j, j_expected)
    assert np.allclose(k, k_expected)
    assert np.allclose(ints.get_k(dm), k_expected)
    assert np.allclose(ints.veff_hf(dm), j_expected - 0.5 * k_expected)


def test_callback_integrals_high_level_reference_is_opt_in(tmp_path):
    provider = SiestaEriProvider(write_qcomp_file(tmp_path / "eris.nc"), n_ao=2)
    plain = CallbackIntegrals(_zero_evaluator, _zero_evaluator, None, nao=2, eri_provider=provider)
    assert plain.high_level_reference_energy(None) == 0.0

    shifted = CallbackIntegrals(
        _zero_evaluator,
        _zero_evaluator,
        None,
        nao=2,
        eri_provider=provider,
        high_level_reference=lambda layer: 2.5,
    )
    assert shifted.high_level_reference_energy(None) == pytest.approx(2.5)


def test_nuclear_repulsion_from_psml_and_psf(monkeypatch, tmp_path):
    from ase import Atoms, units

    atoms = Atoms("OH2", positions=[[0, 0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    r = 1.0 / units.Bohr
    expected = 2 * (6.0 * 1.0) / r + (1.0 * 1.0) / (2.0 / units.Bohr)

    psml = tmp_path / "psml"
    psml.mkdir()
    (psml / "O.psml").write_text('<pseudo-atom-spec atomic-label="O" z-pseudo="6"/>')
    (psml / "H.psml").write_text('<pseudo-atom-spec atomic-label="H" z-pseudo="1"/>')
    monkeypatch.setenv("SIESTA_PP_PATH", str(psml))
    assert S.nuclear_repulsion(atoms) == pytest.approx(expected)
    assert S._valence_charge("O", str(psml)) == 6.0

    psf = tmp_path / "psf"
    psf.mkdir()
    for element, z in (("O", 6.0), ("H", 1.0)):
        (psf / f"{element}.psf").write_text(
            f"{element}  ca\n header\n 1s 1.00 r= 1.25\n 4 0 1 0.1 0.2 {z:.1f}\n"
        )
    monkeypatch.setenv("SIESTA_PP_PATH", str(psf))
    assert S.nuclear_repulsion(atoms) == pytest.approx(expected)


def test_siesta_calculator_writes_qcomp_fdf(monkeypatch, tmp_path):
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir()
    for element in ("H", "O"):
        (pseudo / f"{element}.psml").write_text("")
    monkeypatch.setenv("SIESTA_PP_PATH", str(pseudo))
    cfg = SimpleNamespace(siesta_basis="DZP", siesta_mesh_cutoff=200.0, scf_max_cycle=1)
    calc = S.siesta_calculator(cfg, xc="PBE", directory=".", require_asi=True, qcomp_eris=True)
    fdf = calc.parameters["fdf_arguments"]
    assert fdf["QComp.Workflow"] == "DROP"
    assert fdf["QComp.Protocol"] == "FILE"
    assert fdf["QComp.DataAddress"].endswith(".nc")
    assert fdf["HFX.TruncateDM"] is False
    assert fdf["HFX.StoreERIsTolerance"] <= 1e-12
    # QComp-only runs use the short-range erfc(omega*r)/r operator; tiny omega
    # recovers the full 1/r Coulomb ERIs needed by the high-level FCI.
    assert fdf["HFX.Omega"] <= 1e-7
