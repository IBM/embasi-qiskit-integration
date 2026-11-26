# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Environment/assembly parity between the SIESTA and PySCF embedding paths.

The feasibility note (``analysis/butyronitrile-pbe-hf-fci/component-diagnosis/
same_hamiltonian_feasibility.md``) splits the missing comparison into two tests:

* **Step 1 -- exact two-path assembly.**  Feed one common set of matrices (the
  ``S``, ``hcore``, ``veff_ll(D_A)``, ``v_emb``, ``P_B``, ``D_A``, ``D_B`` a SIESTA
  run exports) through both integral backends and assemble the embedded Hamiltonian
  with :class:`~embasi_qiskit_integration.projection_embedding_adapter.
  ProjectionEmbeddingAdapter`.  Where the inputs are identical, ``h1``, ``h2``,
  ``e_core`` and ``v_emb`` must agree to round-off: this rules out adapter
  asymmetry and the SIESTA-specific ERI expansion (sparse quartets, 8-fold
  permutation weights, ``erf``-screened fit basis) as sources of a cross-code gap.
* **Step 2 -- frozen-density Hartree parity.**  Reconstruct the ``sphi`` fit basis
  and its ERIs from ``ORB.GAUSS.*`` with PySCF, then compare ``J(D_B)`` computed
  from QCOMP's stored ERIs with the PySCF reconstruction.  This is the largest
  code-specific piece of ``v_emb`` after ``H``/``S``, and it can only be compared
  at the fit/screening level (~1e-5), not bit-for-bit.

The local tests use a synthetic **SIESTA-shaped** fixture: a small real Gaussian
system supplies the Hamiltonian, its exact ERIs are written into QCOMP's NetCDF
format, and the environment matrices are built from a real RHF density.  The
fixture is exact by construction, so any difference is an assembly bug.  The
``siesta_real`` test repeats Step 1 and Step 2 on an actual SIESTA H2O/DZP run;
it is opt-in (``SIESTA_QCOMP_REAL=1`` plus the ASI library and pseudopotentials)
and skipped on CI, like the other real-SIESTA tests.

What is *not* compared, even in the real test: SIESTA's NAO basis/pseudopotential
against PySCF's Gaussian basis/pseudopotential, and the PBE XC part of ``v_emb``.
Those need a common basis, pseudopotential and portable environment density/grid
(Step 3 in the note).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.backends._siesta_qcomp import SiestaEriProvider
from embasi_qiskit_integration.backends.callback import CallbackIntegrals, Evaluation
from embasi_qiskit_integration.backends.pyscf import PySCFIntegrals
from embasi_qiskit_integration.projection_embedding_adapter import (
    ProjectionEmbeddingAdapter,
)

from _qcomp_basis import ao_table, eri_value, hartree_matrix, parse_orbgauss, sphi_projection

h5py = pytest.importorskip("h5py")

ATOL = 1.0e-10
#: Step 2 is limited by the sphi fit/screening difference, not the assembly.
SPHI_ATOL = 1.0e-5


# --------------------------------------------------------------------------- #
# Synthetic SIESTA-shaped fixture
# --------------------------------------------------------------------------- #
def _water_fixture():
    """A real Gaussian H2O plus SIESTA-shaped environment matrices.

    The ERIs are PySCF's exact ones, so writing them to QCOMP's NetCDF format and
    reading them back is lossless: the SIESTA sparse-quartet path and the PySCF
    engine then have literally the same two-electron input.
    """
    from ase.build import molecule
    from pyscf import gto, scf

    atoms = molecule("H2O")
    mol = gto.M(
        atom=[[atom.symbol, tuple(atom.position)] for atom in atoms],
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    assert mf.converged

    s = np.asarray(mol.intor("int1e_ovlp"))
    hcore = np.asarray(mol.intor("int1e_kin") + mol.intor("int1e_nuc"))
    enuc = float(mol.energy_nuc())
    tensor = np.asarray(mol.intor("int2e")).reshape([mol.nao] * 4)

    # Split the five occupied orbitals into a 2-orbital "A" and a 3-orbital "B".
    # Canonical MOs are S-orthonormal, so span(A) is S-orthogonal to span(B) and
    # the level-shift projector is invisible inside A.
    c_occ = np.asarray(mf.mo_coeff)[:, :5]
    c_a = c_occ[:, :2]
    c_b = c_occ[:, 2:]
    dm_a = 2.0 * c_a @ c_a.T
    dm_b = 2.0 * c_b @ c_b.T
    veff_da = np.asarray(mf.get_veff(mol, dm_a))
    veff_tot = np.asarray(mf.get_veff(mol, dm_a + dm_b))
    v_emb = veff_tot - veff_da
    mu = 1.0e3
    p_b = mu * s @ dm_b @ s

    assert np.allclose(c_a.T @ s @ c_a, np.eye(2), atol=1e-12)
    assert np.allclose(c_b.T @ s @ c_b, np.eye(3), atol=1e-12)
    assert np.einsum("ij,ji->", s, dm_a) == pytest.approx(4.0, abs=1e-12)
    assert np.einsum("ij,ji->", s, dm_b) == pytest.approx(6.0, abs=1e-12)

    return SimpleNamespace(
        atoms=atoms,
        mol=mol,
        mf=mf,
        s=s,
        hcore=hcore,
        enuc=enuc,
        tensor=tensor,
        c_a=c_a,
        c_b=c_b,
        dm_a=dm_a,
        dm_b=dm_b,
        veff_da=veff_da,
        v_emb=v_emb,
        p_b=p_b,
        mu=mu,
    )


@pytest.fixture(scope="module")
def water():
    return _water_fixture()


def _write_qcomp_file(path: Path, tensor: np.ndarray) -> Path:
    """Write every symmetry-unique quartet into QCOMP's ``eri_indx``/``eri_vals``.

    The provider expands the 8-fold orbit; writing each orbit exactly once makes
    ``SiestaEriProvider.dense()`` reproduce ``tensor`` with no screening loss.
    Values are float64 so the round trip is exact to machine precision.
    """
    n = tensor.shape[0]
    indices: list[tuple[int, int, int, int]] = []
    values: list[float] = []
    for p in range(n):
        for q in range(p, n):
            for r in range(n):
                for s in range(r, n):
                    # One representative per (pq|rs) ~ (rs|pq) orbit.
                    if (p, q) > (r, s):
                        continue
                    indices.append((p + 1, q + 1, r + 1, s + 1))
                    values.append(float(tensor[p, q, r, s]))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("eri_indx", data=np.asarray(indices, dtype=np.int32))
        handle.create_dataset("eri_vals", data=np.asarray(values, dtype=float))
        handle["eri_vals"].attrs["unit"] = "Hartree"
    return path


class _DenseEri:
    """ERI provider over an in-memory ``(pq|rs)`` tensor (the "static" path)."""

    def __init__(self, tensor: np.ndarray):
        self._tensor = np.asarray(tensor, dtype=float)

    def jk(self, dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dm = np.asarray(dm, dtype=float)
        return (
            np.einsum("pqrs,rs->pq", self._tensor, dm, optimize=True),
            np.einsum("prqs,rs->pq", self._tensor, dm, optimize=True),
        )

    def eri_mo(self, coeff: np.ndarray) -> np.ndarray:
        c = np.asarray(coeff, dtype=float)
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", self._tensor, c, c, c, c, optimize=True)

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        a, b = np.asarray(mo_a, dtype=float), np.asarray(mo_b, dtype=float)
        return np.einsum("pqrs,pi,qj,rk,sl->ijkl", self._tensor, a, a, b, b, optimize=True)


def _static_evaluator(hcore, veff_da, s, dm_a):
    """A two-point evaluator: ``H[0] = hcore`` and ``H[D_A] = hcore + veff_ll(D_A)``.

    Both derivations the adapter uses (``hcore = H[0]`` and
    ``veff_ll(d) = H[d] - H[0]``) are exercised.  Refusing any other density keeps
    the static path from silently answering a question it does not know.
    """

    def evaluate(dm):
        d = dm[0] + dm[1] if isinstance(dm, tuple) else np.asarray(dm, dtype=float)
        if np.allclose(d, 0.0, atol=0.0, rtol=0.0):
            block = hcore
        elif np.allclose(d, dm_a, atol=0.0, rtol=0.0):
            block = hcore + veff_da
        else:
            raise AssertionError("the static evaluator only knows H[0] and H[D_A]")
        return Evaluation((block,), s, 0.0)

    return evaluate


class _StubLayer:
    """The ``A_LL`` pieces :meth:`_assemble_fock_a_only` reads."""

    def __init__(self, kinetic: np.ndarray, estat_plus_xc: np.ndarray):
        self.hamiltonian_kinetic = kinetic
        self.hamiltonian_estat_plus_xc = estat_plus_xc
        self.no_scf = True

    def run_noscf(self, dm_in=None):  # noqa: D401 - the adapter never needs it here
        """No-op: the densities are supplied through ``construct_embedding_potential``."""


class _StubProjection:
    """Minimal ``ProjectionEmbedding`` stand-in exposing captured SIESTA state.

    ``construct_embedding_potential`` returns exactly the arrays a real SIESTA run
    wrote out, so ``run_low_level`` assembles ``F_emb = h_kin + h_estat + v_emb +
    P_B`` through the production code path with no live EmbASI or ASI library.
    """

    projection = "level-shift"

    def __init__(
        self,
        *,
        kinetic: np.ndarray,
        estat_plus_xc: np.ndarray,
        s: np.ndarray,
        dm_a: np.ndarray,
        dm_b: np.ndarray,
        v_emb: np.ndarray,
        p_b: np.ndarray,
        c_a: np.ndarray,
        c_b: np.ndarray,
        mu: float,
        a_spin: int | None = 0,
    ):
        self.mu_val = mu
        self.A_LL = _StubLayer(kinetic, estat_plus_xc)
        self.mo_coeffs_A_LL = c_a
        self.mo_coeffs_B_LL = c_b
        self.A_spin = a_spin
        self._result = (dm_a, dm_b, s, v_emb, p_b)

    def construct_embedding_potential(self, dma_in=None, dmb_in=None, a_nspade_mos=None):
        return self._result


def _drive_adapter(ints, fixture, *, n_frozen_occ=1, n_virtual=2):
    """Run the production downfold over one integral backend and a captured state."""
    stub = _StubProjection(
        kinetic=fixture.hcore,
        estat_plus_xc=fixture.veff_da,
        s=fixture.s,
        dm_a=fixture.dm_a,
        dm_b=fixture.dm_b,
        v_emb=fixture.v_emb,
        p_b=fixture.p_b,
        c_a=fixture.c_a,
        c_b=fixture.c_b,
        mu=fixture.mu,
    )
    adapter = ProjectionEmbeddingAdapter(stub, ints, mu=fixture.mu)
    adapter.run_low_level(dma_in=fixture.dm_a, dmb_in=fixture.dm_b)
    orbitals = adapter.build_orbitals(n_frozen_occ=n_frozen_occ, n_virtual=n_virtual)
    ham = adapter.embedded_hamiltonian(orbitals)
    return SimpleNamespace(adapter=adapter, orbitals=orbitals, ham=ham)


# --------------------------------------------------------------------------- #
# Step 1: the SIESTA ERI provider is the same operator as PySCF's engine
# --------------------------------------------------------------------------- #
def test_qcomp_eri_provider_matches_pyscf_engine(tmp_path, water):
    """QCOMP's stored quartets reproduce PySCF's J, K and MO ERIs exactly.

    The fixture writes PySCF's own tensor, so this isolates the SIESTA-specific
    expansion (8-fold permutation weights, sparse ``bincount`` contractions,
    dense/sparse MO transform) from any basis-set effect.
    """
    from pyscf import scf

    provider = SiestaEriProvider(
        _write_qcomp_file(tmp_path / "qcomp_eris.nc", water.tensor), n_ao=water.mol.nao
    )
    direct = PySCFIntegrals(water.mf)
    engine = scf.hf.RHF(water.mol)

    rng = np.random.default_rng(7)
    x = rng.normal(size=(water.mol.nao, water.mol.nao))
    dm = x + x.T  # not PSD; J/K are linear in dm and the test is algebraic
    j_q, k_q = provider.jk(dm)
    j_p, k_p = engine.get_jk(water.mol, dm)
    np.testing.assert_allclose(j_q, j_p, atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(k_q, k_p, atol=ATOL, rtol=0.0)

    c_a = np.linalg.qr(rng.normal(size=(water.mol.nao, 3)))[0]
    c_b = np.linalg.qr(rng.normal(size=(water.mol.nao, 3)))[0]
    np.testing.assert_allclose(provider.eri_mo(c_a), direct.eri_mo(c_a), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(
        provider.eri_mo_mixed(c_a, c_b), direct.eri_mo_mixed(c_a, c_b), atol=ATOL, rtol=0.0
    )


def test_two_path_embedded_hamiltonian_is_exact(tmp_path, water):
    """The same exported matrices give the same ``h1``/``h2``/``e_core``/``v_emb``.

    Path 1 is the SIESTA assembly: ``CallbackIntegrals`` over a static evaluator
    exporting SIESTA's ``H[0]``/``H[D_A]``, with the QCOMP NetCDF ERIs.
    Path 2 is the PySCF assembly: ``PySCFIntegrals`` over a PySCF ``Mole`` for its
    native J/K and ``ao2mo``, with the *same* SIESTA-exported one-electron matrices
    injected (``S``, ``hcore``, ``veff_ll(D_A)``, ``E_nuc``).  Both drive the same
    ``ProjectionEmbeddingAdapter`` with the same ``v_emb``/``P_B``/``D_A``/``D_B``.

    Because the fixture's ERIs are PySCF's exact tensor written to QCOMP format,
    any difference beyond round-off is an assembly bug; a real SIESTA run would
    differ at the sphi/screening level, which is Step 2 and is not this test.
    """
    provider = SiestaEriProvider(
        _write_qcomp_file(tmp_path / "qcomp_eris.nc", water.tensor), n_ao=water.mol.nao
    )
    evaluate = _static_evaluator(water.hcore, water.veff_da, water.s, water.dm_a)
    ints_siesta = CallbackIntegrals(
        evaluate,
        evaluate,
        None,
        nao=water.mol.nao,
        eri_provider=provider,
        arbitrary_dm_veff=True,
        nuclear_energy=lambda: water.enuc,
    )

    ints_pyscf = PySCFIntegrals(water.mf)
    # Inject exactly the one-electron/environment matrices SIESTA exported.  The
    # two-electron half stays PySCF-native (RHF J/K and ao2mo).
    ints_pyscf.overlap = lambda: water.s  # type: ignore[method-assign]
    ints_pyscf.hcore = lambda: water.hcore  # type: ignore[method-assign]
    ints_pyscf.energy_nuc = lambda: water.enuc  # type: ignore[method-assign]

    def veff_da(dm):
        d = dm[0] + dm[1] if isinstance(dm, tuple) else np.asarray(dm, dtype=float)
        if not np.allclose(d, water.dm_a, atol=0.0, rtol=0.0):
            raise AssertionError("the injected veff_ll only knows D_A")
        return water.veff_da

    ints_pyscf.veff_ll = veff_da  # type: ignore[method-assign]

    siesta = _drive_adapter(ints_siesta, water)
    pyscf = _drive_adapter(ints_pyscf, water)

    # Both paths must have built the same active space...
    assert pyscf.ham.norb == siesta.ham.norb
    assert pyscf.ham.nelec == siesta.ham.nelec
    np.testing.assert_allclose(pyscf.orbitals.coeff, siesta.orbitals.coeff, atol=1e-12, rtol=0.0)

    # ...and the same operators and downfolded Hamiltonian.
    for name, left, right, atol in [
        ("p_b", siesta.adapter.p_b, pyscf.adapter.p_b, 0.0),
        ("v_emb", siesta.adapter.v_emb, pyscf.adapter.v_emb, 1e-12),
        ("h_emb", siesta.adapter.h_emb, pyscf.adapter.h_emb, ATOL),
        ("h1", siesta.ham.h1, pyscf.ham.h1, ATOL),
        ("h2", siesta.ham.h2, pyscf.ham.h2, ATOL),
    ]:
        np.testing.assert_allclose(right, left, atol=atol, rtol=0.0, err_msg=name)
    assert pyscf.ham.e_core == pytest.approx(siesta.ham.e_core, abs=ATOL)
    # The active space must be clean in both (P_B invisible inside A).
    assert float(siesta.ham.meta["p_b_leak"]) < 1e-6
    assert float(pyscf.ham.meta["p_b_leak"]) < 1e-6


# --------------------------------------------------------------------------- #
# Step 2 machinery: sphi reconstruction on an exactly representable basis
# --------------------------------------------------------------------------- #
def _write_orbgauss(
    directory: Path,
    label: str,
    *,
    n: int,
    ell: int,
    zeta: int,
    exponents: list[float],
    coefficients: list[float],
) -> None:
    """One ``ORB.GAUSS.*`` file in the format ``parse_orbgauss`` reads."""
    name = f"ORB.GAUSS.S{ell + 1}.{zeta}.{label}"
    lines = [
        "#       species label, n, l, zeta, ngaussians_orbital",
        f"#      {label}  {n}  {ell}  {zeta}  {len(exponents)}",
        "#     exponent_gaussian  coefficient_gaussian",
    ]
    lines += [
        f"   {exponent:.10E}    {coefficient:.10E}"
        for exponent, coefficient in zip(exponents, coefficients, strict=True)
    ]
    (directory / name).write_text("\n".join(lines) + "\n")


def test_sphi_hartree_matrix_matches_individual_eri_reference(tmp_path):
    """``hartree_matrix`` reproduces QCOMP's ``sphi`` basis at machine precision.

    A one-atom s+p basis is exactly representable by both constructions (no
    confinement, no screening), so ``J(D)`` must equal the element-wise ERI
    reference used by ``test_siesta_same_basis`` -- not merely the same code path.
    """
    from ase import Atoms

    root = tmp_path / "orbgauss"
    root.mkdir()
    _write_orbgauss(root, "C.1", n=2, ell=0, zeta=1, exponents=[1.3], coefficients=[1.0])
    _write_orbgauss(
        root,
        "C.1",
        n=2,
        ell=1,
        zeta=1,
        exponents=[0.8, 0.45],
        coefficients=[1.0, 0.6],
    )
    atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)])
    shells = parse_orbgauss(root)
    table = ao_table(atoms, shells)
    assert len(table) == 4  # one s + three p AOs

    # The constructed sphi functions are orthonormal in their own overlap.
    mol, weights = sphi_projection(atoms, shells)
    s_phi = weights.T @ mol.intor("int1e_ovlp") @ weights
    np.testing.assert_allclose(s_phi, np.eye(4), atol=1e-12, rtol=0.0)

    dm = np.zeros((4, 4))
    dm[1, 1] = 2.0
    dm[0, 2] = dm[2, 0] = 0.5
    j = hartree_matrix(atoms, shells, dm)
    for p in range(4):
        for q in range(4):
            ref_11, kind_11 = eri_value(atoms, table, (p, q, 1, 1))
            ref_02, kind_02 = eri_value(atoms, table, (p, q, 0, 2))
            assert kind_11 == kind_02 == "full"
            expected = 2.0 * ref_11 + ref_02  # D[1,1]=2, 2*D[0,2]=1
            assert j[p, q] == pytest.approx(expected, abs=1e-12), (p, q)


# --------------------------------------------------------------------------- #
# Opt-in real SIESTA: Step 1 + Step 2 on one H2O/DZP low-level run
# --------------------------------------------------------------------------- #
@pytest.mark.siesta_real
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("SIESTA_QCOMP_REAL") != "1" or os.environ.get("CI") == "true",
    reason="set SIESTA_QCOMP_REAL=1 and provide a QCOMP ASI library",
)
def test_real_siesta_assembly_and_frozen_density_parity(tmp_path: Path):
    """Repeat Steps 1-2 on the matrices SIESTA actually exports for H2O/DZP."""
    from ase.build import molecule

    from embasi_qiskit_integration.backends import siesta as S
    from embasi_qiskit_integration.backends._embasi_compat import (
        install_siesta_adapter,
        prepare_projection_embedding,
    )

    version = S._asi4py_version()
    if version is None or version < S.MIN_ASI4PY:
        pytest.skip("asi4py >= 1.4.10 is required for SIESTA flavour 3")
    if not os.environ.get("ASI_LIB_PATH") or not os.environ.get("SIESTA_PP_PATH"):
        pytest.skip("ASI_LIB_PATH and SIESTA_PP_PATH must be set")

    from embasi.embedding import ProjectionEmbedding

    install_siesta_adapter()
    atoms = molecule("H2O")
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
    backend = S.build_siesta_backend(atoms, 0, cfg)
    ints = backend.integrals
    provider = ints._eri
    assert provider is not None
    # Step 2 reconstructs full-Coulomb sphi ERIs; refuse a short-range file.
    assert float(backend.calc_ll.parameters["fdf_arguments"]["HFX.Omega"]) <= 1e-7

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
    mu = 1.0e6
    adapter = ProjectionEmbeddingAdapter(projection, ints, mu=mu)
    adapter.run_low_level(a_nmos=3)
    assert provider.n_eri > 0, "the low-level run wrote no QCOMP ERIs"
    orbitals = adapter.build_orbitals(n_frozen_occ=1, n_virtual=3)
    ham = adapter.embedded_hamiltonian(orbitals)

    # ---- capture exactly what the SIESTA path used ------------------------
    s = np.asarray(adapter._s_arr)
    dm_a = np.asarray(adapter._dm_a_for_veff)
    dm_b = np.asarray(adapter._dm_b_arr)
    hcore = np.asarray(ints.hcore())
    veff_da = np.asarray(ints.veff_ll(dm_a))
    enuc = float(ints.energy_nuc())
    fock = np.asarray(adapter._fock_arr)
    v_emb_embasi = np.asarray(adapter._v_emb_embasi)
    p_b = np.asarray(adapter.p_b)
    v_emb = np.asarray(adapter.v_emb)
    h_emb = np.asarray(adapter.h_emb)
    c_a = np.asarray(adapter.mo_a_ll)
    c_b = np.asarray(adapter.mo_b_ll)
    tensor = np.asarray(provider.dense())
    kinetic = adapter._as_ao_operator(projection.A_LL.hamiltonian_kinetic)
    estat = adapter._as_ao_operator(projection.A_LL.hamiltonian_estat_plus_xc)
    a_spin = getattr(projection, "A_spin", None)
    nao = int(s.shape[0])

    # The exported A_LL blocks must reproduce the F_emb the downfold used.
    np.testing.assert_allclose(fock, kinetic + estat + v_emb_embasi + p_b, atol=ATOL, rtol=0.0)

    # ---- Step 1: same arrays through the static/CallbackIntegrals path ----
    evaluate = _static_evaluator(hcore, veff_da, s, dm_a)
    ints_static = CallbackIntegrals(
        evaluate,
        evaluate,
        None,
        nao=nao,
        eri_provider=_DenseEri(tensor),
        arbitrary_dm_veff=True,
        nuclear_energy=lambda: enuc,
    )
    stub = _StubProjection(
        kinetic=kinetic,
        estat_plus_xc=estat,
        s=s,
        dm_a=dm_a,
        dm_b=dm_b,
        v_emb=v_emb_embasi,
        p_b=p_b,
        c_a=c_a,
        c_b=c_b,
        mu=mu,
        a_spin=a_spin,
    )
    adapter_static = ProjectionEmbeddingAdapter(stub, ints_static, mu=mu)
    adapter_static.run_low_level(dma_in=dm_a, dmb_in=dm_b)
    orbitals_static = adapter_static.build_orbitals(n_frozen_occ=1, n_virtual=3)
    ham_static = adapter_static.embedded_hamiltonian(orbitals_static)

    np.testing.assert_allclose(orbitals_static.coeff, orbitals.coeff, atol=ATOL, rtol=0.0)
    for name, left, right, atol in [
        ("fock", fock, np.asarray(adapter_static._fock_arr), ATOL),
        ("p_b", p_b, np.asarray(adapter_static.p_b), ATOL),
        ("v_emb", v_emb, np.asarray(adapter_static.v_emb), ATOL),
        ("h_emb", h_emb, np.asarray(adapter_static.h_emb), ATOL),
        ("h1", ham.h1, ham_static.h1, ATOL),
        ("h2", ham.h2, ham_static.h2, ATOL),
    ]:
        max_diff = float(np.abs(np.asarray(right) - np.asarray(left)).max())
        print(f"[parity] {name}: max|SIESTA - static| = {max_diff:.3e}")
        np.testing.assert_allclose(right, left, atol=atol, rtol=0.0, err_msg=name)
    assert ham_static.e_core == pytest.approx(ham.e_core, abs=ATOL)
    assert ham_static.nelec == ham.nelec

    # ---- Step 2: frozen-density Hartree J(D_B) in the sphi basis ----------
    workdir = Path(provider.path).parent
    shells = parse_orbgauss(workdir)
    table = ao_table(atoms, shells)
    assert len(table) == nao, "ORB.GAUSS shells do not match SIESTA's AO count"
    np.testing.assert_allclose(dm_b, dm_b.T, atol=1e-10, rtol=0.0)

    j_qcomp, _k_qcomp = provider.jk(dm_b)
    j_sphi = hartree_matrix(atoms, shells, dm_b)
    j_diff = float(np.abs(j_qcomp - j_sphi).max())
    e_hartree_qcomp = 0.5 * float(np.einsum("ij,ji->", dm_b, j_qcomp))
    e_hartree_sphi = 0.5 * float(np.einsum("ij,ji->", dm_b, j_sphi))
    print(
        f"[parity] J(D_B): max|QCOMP - sphi| = {j_diff:.3e}; "
        f"E_Hartree {e_hartree_qcomp:.10f} vs {e_hartree_sphi:.10f} Ha "
        f"(diff {e_hartree_qcomp - e_hartree_sphi:+.3e})"
    )
    assert j_diff < SPHI_ATOL
    assert e_hartree_qcomp == pytest.approx(e_hartree_sphi, abs=SPHI_ATOL)
