# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Same-basis reference for SIESTA-QCOMP's exported ERIs.

QCOMP's ``eri_vals`` are 4-centre ERIs in the unit-normalised Gaussian fit
basis (``sphi``); ASI's H/S uses the same NAO2GTO fit with the confinement
factor applied, which differs from ``sphi`` by only ~1e-7.  Unless
``HFX.Omega`` is made tiny the stored ERIs are **short-range**
``erfc(omega*r)/r`` integrals, so the production backend sets
``HFX.Omega=1e-8`` for every QCOMP low-level calculation.  This test checks the
resulting full-Coulomb ERIs against a PySCF primitive-Gaussian reconstruction
of the same ``sphi`` basis.

Enable with ``SIESTA_QCOMP_REAL=1`` and ``ASI_LIB_PATH``/``SIESTA_PP_PATH``;
skipped on CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# PySCF (and its h5py) must load before libsiesta/NetCDF, as in the other real
# SIESTA tests; the reference integrals are computed with PySCF.
if os.environ.get("SIESTA_QCOMP_REAL") == "1":
    pytest.importorskip("pyscf.gto")

from embasi_qiskit_integration.backends import siesta as S
from embasi_qiskit_integration.backends._embasi_compat import install_siesta_adapter
from embasi_qiskit_integration.backends.callback import LayerEvaluator
from embasi_qiskit_integration.backends._siesta_qcomp import SiestaEriProvider

from _qcomp_basis import ao_table, eri_value, parse_orbgauss, self_eri

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


@dataclass(frozen=True)
class Case:
    name: str
    ase_key: str


CASES = (Case("H2O", "H2O"), Case("H2", "H2"), Case("CH4", "CH4"))


def _require_real() -> None:
    version = S._asi4py_version()
    if version is None or version < S.MIN_ASI4PY:
        pytest.skip("asi4py >= 1.4.10 is required for SIESTA flavour 3")
    if not os.environ.get("ASI_LIB_PATH") or not os.environ.get("SIESTA_PP_PATH"):
        pytest.skip("ASI_LIB_PATH and SIESTA_PP_PATH must be set")


def _stored_offdiagonal_quartets(
    atoms, table, ref: dict[tuple[int, int, int, int], float], limit: int = 12
) -> list[tuple[int, int, int, int]]:
    """Off-diagonal quartets that QCOMP stored, in a representable layout.

    Each stored quartet must match the exact same-basis value at those indices.
    Quartets are restricted to one atom per element so ``_primitive_mol`` can
    express PySCF's per-element basis list.  QCOMP screens individual quartets,
    so absent entries are skipped rather than treated as zeros.
    """
    from collections import defaultdict

    out: list[tuple[int, int, int, int]] = []
    for quartet in ref:
        if len(set(quartet)) < 4:
            continue
        used: dict[str, set[int]] = defaultdict(set)
        for index in quartet:
            ia = table[index][0]
            used[atoms[ia].symbol].add(ia)
        if any(len(ids) > 1 for ids in used.values()):
            continue
        out.append(tuple(int(i) for i in quartet))
        # Keep the loop bounded: each exact reference rebuilds a PySCF Mole.
        if len(out) >= limit:
            break
    return out


def _qcomp_eri_map(provider: SiestaEriProvider) -> dict[tuple[int, int, int, int], float]:
    """QCOMP's 0-based quartet map, read through the provider's HDF5 fallback.

    The test has already loaded libsiesta, whose NetCDF HDF5 can conflict with
    the wheel h5py.  ``SiestaEriProvider`` handles that with a clean subprocess
    read; reuse it instead of opening the file with h5py here.
    """
    provider._load()
    assert provider._indices is not None and provider._values is not None
    idx = np.asarray(provider._indices, dtype=np.int64)
    val = np.asarray(provider._values, dtype=float).reshape(-1)
    assert idx.shape[0] == val.size
    return {tuple(int(x) for x in row): float(v) for row, v in zip(idx, val, strict=True)}


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_qcomp_eris_match_same_basis_reference(case: Case, tmp_path: Path):
    _require_real()
    from ase.build import molecule

    # Build one real QCOMP evaluation: this writes qcomp_eris.nc and ORB.GAUSS.*.
    atoms = molecule(case.ase_key)
    cfg = SimpleNamespace(
        xc_ll="PBE",
        xc_hl="HF",
        basis="sto-3g",
        spin=0,
        unrestricted=False,
        scf_max_cycle=1,
        relax_hf=False,
        siesta_basis="DZP",
        siesta_mesh_cutoff=200.0,
        siesta_workdir=str(tmp_path / "work"),
    )
    install_siesta_adapter()
    calc = S.siesta_calculator(
        cfg, xc="PBE", directory=".", max_scf_cycles=1, require_asi=True, qcomp_eris=True
    )
    # The production setting; check it is present so the file is full Coulomb.
    fdf = calc.parameters["fdf_arguments"]
    assert float(fdf["HFX.Omega"]) <= 1e-7
    n_ao, _ = S.probe_basis_asi(
        atoms, calc, directory=tmp_path / "work" / "probe", lib_path=os.environ["ASI_LIB_PATH"]
    )
    evaluator = LayerEvaluator(atoms, calc, n_spin=1, label=str(tmp_path / "work" / "eval"))
    evaluator(np.zeros((n_ao, n_ao)))

    provider = SiestaEriProvider(tmp_path / "work" / "eval" / "qcomp_eris.nc", n_ao=n_ao)
    assert provider.n_eri > 0
    shells = parse_orbgauss(tmp_path / "work" / "eval")
    table = ao_table(atoms, shells)
    assert len(table) == n_ao
    ref = _qcomp_eri_map(provider)

    # Every diagonal self-ERI must match the same-basis full-Coulomb reference.
    for ao in range(n_ao):
        q = ref.get((ao, ao, ao, ao))
        if q is None:
            # A thresholded run can omit an exactly-zero quartet; not expected
            # for self terms, but do not make the reference depend on it.
            continue
        exact, kind = self_eri(atoms, table, ao)
        assert kind == "full"
        assert exact == pytest.approx(q, abs=2e-5), (case.name, ao, q, exact)

    # Stored off-diagonal quartets validate the (pq|rs) index ordering.  QCOMP
    # screens individual quartets, so compare the ones it wrote; missing entries
    # are not zeros.
    quartets = _stored_offdiagonal_quartets(atoms, table, ref)
    if not quartets:
        # H2 has two equivalent H atoms and no stored quartet confined to one
        # of them; the self-ERIs above are the available check there.
        pytest.skip(f"{case.name}: no representable stored off-diagonal quartets")
    for quartet in quartets:
        exact, kind = eri_value(atoms, table, quartet)
        assert kind == "full"
        assert exact == pytest.approx(float(ref[quartet]), abs=2e-5), (
            case.name,
            quartet,
            float(ref[quartet]),
            exact,
        )
