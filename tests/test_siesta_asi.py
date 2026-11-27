# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SIESTA ASI backend and the temporary EmbASI compatibility shim.

These tests do not need SIESTA: ``_fake_asi`` compiles a flavour-3 ASI library
that mimics SIESTA's observable callback order (one total H per SCF step).  The
real ``libsiesta.so`` is validated on the HPC build; this covers the Python seam,
the shim's extraction semantics and the backend's ASI-mode configuration.

TODO(embasi-siesta-shim): tests marked with this tag exercise the shim and must
be removed or adapted (to EmbASI's own Siesta adapter) together with
``backends/_embasi_compat.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.backends import siesta as S

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _fake_asi  # noqa: E402


def _skip_reason() -> str | None:
    version = S._asi4py_version()
    if version is None:
        return "asi4py is not installed"
    if version < S.MIN_ASI4PY:
        return f"asi4py {'.'.join(map(str, version))} has no SIESTA flavour 3"
    try:
        import embasi  # noqa: F401
    except Exception as exc:  # EmbASI imports mpi4py, which may lack a runtime
        return f"EmbASI unavailable: {exc}"
    return None


_SKIP = _skip_reason()
pytestmark = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")


@pytest.fixture(scope="session")
def fake_asi_lib(tmp_path_factory: pytest.TempPathFactory) -> str:
    try:
        library = _fake_asi.build_fake_asi_library(tmp_path_factory.mktemp("fake_asi"))
    except Exception as exc:  # compiler missing/broken
        pytest.skip(f"cannot compile the fake ASI library: {exc}")
    if library is None:
        pytest.skip("no C compiler for the fake ASI library")
    return str(library)


@pytest.fixture
def siesta_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_asi_lib: str):
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir()
    # Minimal PSML headers: the shim reads z-pseudo for the valence electron
    # count (10 all-electron vs 8 valence for H2O).  The fake library does not
    # parse the potentials themselves.
    for element, z_pseudo in (("H", 1), ("O", 6)):
        (pseudo / f"{element}.psml").write_text(
            f'<?xml version="1.0"?>\n<psml><pseudo-atom-spec atomic-label="{element}" '
            f'z-pseudo="{z_pseudo}"/></psml>\n'
        )
    monkeypatch.setenv("SIESTA_PP_PATH", str(pseudo))
    monkeypatch.setenv("ASI_LIB_PATH", fake_asi_lib)
    monkeypatch.delenv("ASE_SIESTA_COMMAND", raising=False)
    from embasi_qiskit_integration.backends._embasi_compat import install_siesta_adapter

    install_siesta_adapter()
    return SimpleNamespace(lib=fake_asi_lib, pseudo=pseudo, tmp=tmp_path)


def make_cfg(tmp_path: Path, **overrides):
    values = dict(
        xc_ll="PBE",
        xc_hl="HF",  # WF-in-DFT (the SIESTA backend is WF-only)
        basis="sto-3g",
        spin=0,
        unrestricted=False,
        scf_max_cycle=1,
        siesta_basis="DZP",
        siesta_mesh_cutoff=200.0,
        siesta_workdir=str(tmp_path / "work"),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def make_calculator(siesta_env, tmp_path: Path):
    return S.siesta_calculator(make_cfg(tmp_path), xc="PBE", directory=".", require_asi=True)


# TODO(embasi-siesta-shim): remove once EmbASI ships its own Siesta adapter.
def test_shim_registers_only_when_embasi_lacks_siesta(monkeypatch: pytest.MonkeyPatch):
    from embasi import qmcode_adapters

    monkeypatch.setattr(
        qmcode_adapters, "implemented_calculators", dict(qmcode_adapters.implemented_calculators)
    )
    qmcode_adapters.implemented_calculators.pop("Siesta", None)
    from embasi_qiskit_integration.backends._embasi_compat import install_siesta_adapter

    assert install_siesta_adapter() is True
    assert "Siesta" in qmcode_adapters.implemented_calculators
    assert install_siesta_adapter() is False  # EmbASI now has one; shim bypassed


# TODO(embasi-siesta-shim): remove/port once EmbASI ships its own Siesta adapter.
def test_adapter_maps_directives_to_siesta_fdf(siesta_env):
    from embasi_qiskit_integration.backends._embasi_compat import SiestaAdapter

    calc = make_calculator(siesta_env, siesta_env.tmp)
    assert type(calc).__name__ == "Siesta"  # EmbASI dispatches on this name
    assert hasattr(calc, "write_inputfiles")
    adapter = SiestaAdapter()

    adapter.set_noscf_calc_params(calc)
    fdf = calc.parameters["fdf_arguments"]
    assert fdf["MaxSCFIterations"] == 1
    assert fdf["SCFMustConverge"] is False
    assert fdf["DM.UseSaveDM"] is False

    adapter.set_max_scf_cycles(calc, 7)
    assert calc.parameters["fdf_arguments"]["MaxSCFIterations"] == 7
    adapter.set_qm_total_charge(None, calc, 1.0)
    assert calc.parameters["fdf_arguments"]["NetCharge"] == 1.0

    with pytest.raises(NotImplementedError, match="ghost"):
        adapter.set_ghost_atoms(None, calc, [True])


# TODO(embasi-siesta-shim): remove/port with this module.
def test_noscf_fdf_logs_energy_at_imported_density(siesta_env):
    """The non-SCF fdf must make ``total.log``'s energy terms use the input density.

    ``same_density_reference_energy`` pairs the logged ``Exc`` with the imported
    density ``D_in``.  SIESTA >= 4.1 defaults to Hamiltonian mixing, which logs
    the one-step decomposition at the post-step ``DM_out`` instead; DM mixing
    evaluates it at ``D_in``, and ``DM.MixSCF1 False`` keeps the final post-SCF
    energy at ``DM_out`` (the density EmbASI's non-SCF correction targets).
    """
    from embasi_qiskit_integration.backends._embasi_compat import SiestaAdapter

    calc = make_calculator(siesta_env, siesta_env.tmp)
    SiestaAdapter().set_noscf_calc_params(calc)
    fdf = calc.parameters["fdf_arguments"]
    assert fdf["SCF.Mix"] == "density-matrix"
    assert fdf["DM.MixSCF1"] is False
    # The one-step non-SCF contract itself is unchanged.
    assert fdf["MaxSCFIterations"] == 1
    assert fdf["SCFMustConverge"] is False


# TODO(embasi-siesta-shim): keep the fdf/DM/pseudo assertions; drop the
# write_inputfiles call with the shim once EmbASI writes input itself.
def test_calculator_writes_single_dm_use_save_false_and_pseudos(siesta_env, tmp_path):
    import re

    from ase.build import molecule

    work = tmp_path / "fdf"
    work.mkdir()
    calc = make_calculator(siesta_env, tmp_path)
    calc.directory = str(work)
    calc.write_inputfiles(molecule("H2O"), properties=["energy"])

    fdf = (work / "siesta.fdf").read_text()
    assert len(re.findall(r"DM\.UseSaveDM\s+\S+", fdf)) == 1
    assert re.search(r"DM\.UseSaveDM\s+False", fdf)
    # Symlink-free copy: some shared filesystems reject symlinks, and SIESTA
    # would then die with "Pseudopotential file not found".
    assert (work / "O.1.psml").is_file()
    assert (work / "H.2.psml").is_file()


# TODO(embasi-siesta-shim): remove/port once EmbASI's Siesta adapter defines the
# Hamiltonian-component extraction (last H = total, or upstream K/V/H exports).
def test_adapter_extracts_last_total_hamiltonian(siesta_env, monkeypatch, tmp_path):
    from asi4py.pyasi import ASIlib
    from ase.build import molecule
    from embasi_qiskit_integration.backends._embasi_compat import SiestaAdapter

    monkeypatch.setenv("FAKE_HAM_CALLS", "3")
    atoms = molecule("H2O")
    calc = make_calculator(siesta_env, tmp_path)
    work = tmp_path / "asi"
    work.mkdir()

    def initializer(asi):
        calc.write_inputfiles(asi.atoms, properties=["energy"])
        (work / "asi.log").write_text("")

    asi = ASIlib(siesta_env.lib, initializer, None, atoms, work_dir=str(work))
    asi.init()

    adapter = SiestaAdapter()
    fake_calc = SimpleNamespace(asi=asi)
    fake_embed = SimpleNamespace(
        blacs_ctxt_tag=None,
        blacs_descr_tag=None,
        flag_huz=False,
        density_matrix_in=None,
        fock_embedding_matrix=None,
        truncate=False,
    )
    adapter.register_import_export_hooks(fake_calc, fake_embed)
    asi.run()
    assert asi.ham_count == 3

    embed = SimpleNamespace(
        atoms=SimpleNamespace(calc=fake_calc),
        parallel=False,
        truncate=False,
        fock_embedding_matrix=None,
    )
    out = adapter.extract_matrices(embed)
    assert out["n_basis"] == 7
    assert np.asarray(out["basis_atoms"]).tolist() == [0, 0, 0, 1, 1, 2, 2]
    # H is exported once per SCF step with a distinct diagonal; the LAST one is
    # the total (-0.5 + 0.01*2 for the third export), not the first.
    assert float(out["ham_tot"][0, 0][0, 0]) == pytest.approx(-0.48)
    assert np.allclose(out["ham_kin"][0, 0], out["ham_tot"][0, 0])
    assert np.allclose(out["ham_estat_xc"][0, 0], 0.0)
    assert float(out["ovlp"][0, 0][0, 0]) == pytest.approx(1.0)
    assert float(out["dm"][0, 0][0, 0]) == pytest.approx(2.0)
    asi.close()


def test_atomic_reference_energy_parses_total_log(tmp_path):
    layer = tmp_path / "layer"
    layer.mkdir()
    (layer / "total.log").write_text("siesta: Ena     =       27.211384500\n")
    assert S.atomic_reference_energy(SimpleNamespace(outdir=str(layer))) == pytest.approx(1.0)
    assert S.atomic_reference_energy(SimpleNamespace(outdir=str(tmp_path / "missing"))) == 0.0


def test_same_density_reference_uses_the_full_bare_energy(tmp_path):
    layer_dir = tmp_path / "layer"
    layer_dir.mkdir()
    (layer_dir / "total.log").write_text("siesta: Exc     =       27.211384500\n")
    dm = np.array([[2.0, 0.0], [0.0, 1.0]])
    hcore = np.array([[-1.0, 0.0], [0.0, -2.0]])
    j = np.array([[3.0, 0.0], [0.0, 1.0]])
    ints = SimpleNamespace(
        hcore=lambda: hcore,
        get_jk=lambda density: (j, None),
        energy_nuc=lambda: 0.5,
    )
    layer = SimpleNamespace(
        outdir=str(layer_dir),
        # eV, like EmbASI's layers
        ev_corr_total_energy=-10.0 * 27.211384500,
        _siesta_input_density=dm,
    )
    bare = (
        float(np.einsum("ij,ji->", dm, hcore))
        + 0.5 * float(np.einsum("ij,ji->", dm, j))
        + 1.0
        + 0.5
    )
    assert S.same_density_reference_energy(layer, ints) == pytest.approx(bare + 10.0)

    # Without the shim's captured density, use the fallback reference.
    (layer_dir / "total.log").write_text("siesta: Ena     =       27.211384500\n")
    layer._siesta_input_density = None
    assert S.same_density_reference_energy(layer, ints) == pytest.approx(1.0)


def test_probe_basis_asi_reads_descriptors(siesta_env, tmp_path):
    from ase.build import molecule

    atoms = molecule("H2O")
    calc = make_calculator(siesta_env, tmp_path)
    n_basis, basis_atoms = S.probe_basis_asi(
        atoms, calc, directory=tmp_path / "probe", lib_path=siesta_env.lib
    )
    assert n_basis == 7
    assert basis_atoms.tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_build_backend_end_to_end(siesta_env, tmp_path):
    from ase.build import molecule

    atoms = molecule("H2O")
    backend = S.build_siesta_backend(atoms, 0, make_cfg(tmp_path))
    ints = backend.integrals
    assert backend.name == "siesta"
    assert ints.nao == 7
    assert ints.ao_atom_map().tolist() == [0, 0, 0, 1, 1, 2, 2]
    assert ints.overlap().shape == (7, 7)
    assert float(ints.hcore()[0, 0]) == pytest.approx(-0.5)
    assert np.allclose(ints.veff_ll(np.zeros((7, 7))), 0.0)


# TODO(embasi-siesta-shim): remove with the prime_layer call in callback.py.
def test_wf_route_declares_eri_and_exact_exchange(siesta_env, tmp_path):
    from ase.build import molecule

    from embasi_qiskit_integration.backends.base import (
        CAP_ARBITRARY_DM_VEFF,
        CAP_ERI,
        CAP_EXACT_EXCHANGE,
    )

    atoms = molecule("H2O")
    backend = S.build_siesta_backend(atoms, 0, make_cfg(tmp_path, xc_hl="HF"))
    capabilities = backend.integrals.capabilities
    assert {CAP_ERI, CAP_EXACT_EXCHANGE, CAP_ARBITRARY_DM_VEFF} <= capabilities
    assert backend.integrals._eri is not None


def test_build_backend_refuses_the_dft_route(siesta_env, tmp_path):
    from ase.build import molecule

    with pytest.raises(NotImplementedError, match="WF-in-DFT only"):
        S.build_siesta_backend(molecule("H2O"), 0, make_cfg(tmp_path, xc_hl="PBE"))


def test_wf_route_refuses_relax_hf(siesta_env, tmp_path):
    from ase.build import molecule

    with pytest.raises(NotImplementedError, match="relax_hf"):
        S.build_siesta_backend(molecule("H2O"), 0, make_cfg(tmp_path, xc_hl="HF", relax_hf=True))


def test_layer_evaluator_primes_nonscf_layer(siesta_env, tmp_path):
    from ase.build import molecule
    from embasi_qiskit_integration.backends.callback import LayerEvaluator

    atoms = molecule("H2O")
    evaluator = LayerEvaluator(
        atoms, make_calculator(siesta_env, tmp_path), n_spin=1, label=str(tmp_path / "layer")
    )
    assert hasattr(evaluator._layer, "_basis_info")
    # H2O: O 6 + 2 H x 1 = 8 pseudopotential valence electrons, not Z=10.
    assert evaluator._layer._input_fragment_nelectrons == pytest.approx(8.0)


# TODO(embasi-siesta-shim): remove with the prepare_projection_embedding call
# in embedding.py once EmbASI primes AB_LL itself.
def test_prepare_projection_embedding_primes_ab_ll(siesta_env, tmp_path):
    from ase.build import molecule
    from embasi.atoms_embedding_asi import AtomsEmbed
    from embasi_qiskit_integration.backends._embasi_compat import prepare_projection_embedding

    atoms = molecule("H2O")
    layer = AtomsEmbed(
        atoms, make_calculator(siesta_env, tmp_path), embed_mask=None, outdir=str(tmp_path / "ab")
    )
    assert not hasattr(layer, "_input_fragment_nelectrons")
    assert prepare_projection_embedding(SimpleNamespace(AB_LL=layer)) is True
    assert layer._input_fragment_nelectrons == pytest.approx(8.0)
    assert hasattr(layer, "_basis_info")


def test_build_backend_requires_asi(siesta_env, monkeypatch, tmp_path):
    from ase.build import molecule

    monkeypatch.delenv("ASI_LIB_PATH")
    with pytest.raises(RuntimeError, match="ASI_LIB_PATH"):
        S.build_siesta_backend(molecule("H2O"), 0, make_cfg(tmp_path))


def test_build_backend_refuses_spin(siesta_env, tmp_path):
    from ase.build import molecule

    with pytest.raises(NotImplementedError, match="unpolarized"):
        S.build_siesta_backend(molecule("H2O"), 0, make_cfg(tmp_path, spin=1))


def test_build_backend_refuses_multi_rank(monkeypatch, tmp_path):
    from ase.build import molecule

    monkeypatch.setattr(S, "_mpi_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="single MPI rank"):
        S.build_siesta_backend(molecule("H2O"), 0, make_cfg(tmp_path))


def test_require_asi_support_rejects_old_wrapper(monkeypatch, tmp_path):
    library = tmp_path / "libsiesta.so"
    library.write_text("")
    monkeypatch.setenv("ASI_LIB_PATH", str(library))
    monkeypatch.setattr(S, "_asi4py_version", lambda: (1, 4, 9))
    with pytest.raises(RuntimeError, match="asi4py 1.4.9"):
        S.require_asi_support()
