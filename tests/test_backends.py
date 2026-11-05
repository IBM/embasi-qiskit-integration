# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The QM-backend seam (``embasi_qiskit_integration.backends``).

Two kinds of check:

* the seam holds -- no module outside ``backends/`` reaches into a PySCF ``Mole`` / SCF
  object, so another QM code can be plugged in behind the protocol;
* the PySCF backend's new accessors reproduce, exactly, the direct PySCF calls they
  replaced.  These compare against PySCF's own ``Mole`` methods, not against another
  call into the backend.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embasi_qiskit_integration.backends import (
    MissingCapabilityError,
    as_ao_basis,
    atom_ao_indices,
    require_capability,
)
from embasi_qiskit_integration.backends.pyscf import PySCFBasis, PySCFIntegrals

PKG = Path(__file__).resolve().parents[1] / "src" / "embasi_qiskit_integration"

# PySCF used as a *math library on plain arrays* (FCI, selected CI, FCIDUMP I/O, APC
# truncation, DIIS, ao2mo.restore on a given tensor) is backend-neutral and allowed
# anywhere.  What is not allowed outside `backends/` is building or querying the QM
# system itself: Mole/SCF construction, AO integrals, density fitting.
_FORBIDDEN_MODULES = {"pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.df", "pyscf.pbc"}
# (`ao_labels` / `search_ao_label` are protocol methods on the backend now, so they are
# not in this list; the Mole-only spellings are.)
_FORBIDDEN_ATTRS = {"intor", "intor_symmetric", "aoslice_by_atom", "get_veff", "get_hcore"}
# Modules where PySCF is the subject, not the QM code: solver-side custom-integral RHF
# (`solvers._rhf_from_integrals` seeds LUCJ from bare h1/h2) and the geometry helper
# the scripts use to build a Mole for themselves.
_EXEMPT = {"solvers.py", "molecule/geometry.py"}
# Known, scheduled exceptions: (module, function), listed by name so any debt is
# visible here rather than silently tolerated.  Empty: calculator construction lives in
# `backends/pyscf.py`.
_PENDING: set[tuple[str, str]] = set()


def _package_modules():
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if rel.startswith("backends/") or rel in _EXEMPT:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and (rel, node.name) in _PENDING:
                node.body = []  # drop the pending function's body from the scan
        yield rel, tree


def test_pending_exemptions_still_exist():
    """Each pending exemption names a real function; delete the entry once it moves."""
    for rel, name in _PENDING:
        tree = ast.parse((PKG / rel).read_text())
        assert any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(tree)), (
            f"{rel}:{name} no longer exists; remove it from _PENDING"
        )


def test_no_pyscf_qm_code_access_outside_backends():
    offences = []
    for rel, tree in _package_modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                names = {f"{mod}.{a.name}" for a in node.names} | {mod}
                if names & _FORBIDDEN_MODULES:
                    offences.append(f"{rel}:{node.lineno} from {mod} import ...")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_MODULES or alias.name == "pyscf":
                        offences.append(f"{rel}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_ATTRS:
                    offences.append(f"{rel}:{node.lineno} .{node.attr}")
                # `<x>.ints.mol` / `<x>.ints.mf` / `.mf_ll` / `atoms.calc.mol`
                if node.attr in {"mol", "mf", "mf_ll"} and isinstance(node.value, ast.Attribute):
                    if node.value.attr in {"ints", "calc"}:
                        offences.append(f"{rel}:{node.lineno} .{node.value.attr}.{node.attr}")
    assert not offences, "PySCF QM-code access outside backends/:\n" + "\n".join(offences)


# --------------------------------------------------------------------------- #
# PySCFBasis / PySCFIntegrals parity with the direct Mole calls
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def mol():
    from pyscf import gto

    return gto.M(atom="Fe 0 0 0; O 0 0 1.6; H 0 0.8 2.1", basis="sto-3g", spin=3, verbose=0)


def test_ao_atom_map_matches_aoslice(mol):
    basis = PySCFBasis(mol)
    amap = basis.ao_atom_map()
    assert amap.shape == (mol.nao,)
    for atom, (_s0, _s1, a0, a1) in enumerate(mol.aoslice_by_atom()):
        assert (amap[a0:a1] == atom).all()
    for atoms in ([0], [1, 2], [2, 0], []):
        direct = basis.atom_ao_indices(atoms)
        via_map = atom_ao_indices(amap, atoms)
        np.testing.assert_array_equal(direct, via_map)


def test_ao_labels_and_search_match_mole(mol):
    basis = PySCFBasis(mol)
    assert basis.ao_labels() == [tuple(lab) for lab in mol.ao_labels(fmt=False)]
    labels = ["Fe 3d", "O 2p"]
    np.testing.assert_array_equal(basis.search_ao_label(labels), mol.search_ao_label(labels))


def test_describe_carries_the_fingerprint_fields(mol):
    d = PySCFBasis(mol).describe()
    assert (d["basis"], d["natm"], d["charge"], d["spin"]) == ("sto-3g", 3, 0, 3)
    np.testing.assert_array_equal(d["atom_coords"], mol.atom_coords())
    assert d["atom_symbols"] == ["Fe", "O", "H"]


def test_as_ao_basis_wraps_a_mole_and_passes_a_backend_through(mol):
    wrapped = as_ao_basis(mol)
    assert isinstance(wrapped, PySCFBasis) and wrapped.mol is mol
    backend = PySCFBasis(mol)
    assert as_ao_basis(backend) is backend
    with pytest.raises(TypeError, match="not an AO basis"):
        as_ao_basis(SimpleNamespace())


@pytest.fixture(scope="module")
def water_uks():
    from pyscf import gto

    m = gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="sto-3g", verbose=0)
    mf_ll = m.UKS(xc="PBE").run()
    mf_hl = m.UKS(xc="HF")
    return m, mf_hl, mf_ll


def test_energy_elec_ll_and_spin_flag_match_the_mean_field(water_uks):
    from pyscf import scf

    _m, mf_hl, mf_ll = water_uks
    ints = PySCFIntegrals(mf_hl, mf_ll)
    dm = np.asarray(mf_ll.make_rdm1()).sum(axis=0)
    assert ints.energy_elec_ll(dm) == mf_ll.energy_elec(dm=dm)[0]
    assert ints.ll_spin_resolved is True
    assert PySCFIntegrals(scf.RHF(_m)).ll_spin_resolved is False


def test_hl_status_and_describe(water_uks):
    _m, mf_hl, mf_ll = water_uks
    ints = PySCFIntegrals(mf_hl, mf_ll, density_fit=True)
    assert ints.hl_scf_status() == {
        "converged": mf_hl.converged,
        "conv_tol": mf_hl.conv_tol,
        "max_cycle": mf_hl.max_cycle,
    }
    d = ints.describe()
    assert d["xc_hl"] == "HF" and d["density_fit"] == "True"
    assert "density_fit" in ints.capabilities
    assert "density_fit" not in PySCFIntegrals(mf_hl, mf_ll).capabilities


def test_layer_status_and_footing(water_uks):
    m, _mf_hl, mf_ll = water_uks
    ints = PySCFIntegrals(mf_ll)
    layer = SimpleNamespace(atoms=SimpleNamespace(calc=SimpleNamespace(method=mf_ll, mol=m)))
    status = ints.layer_scf_status(layer)
    assert status is not None and status["converged"] is True
    assert status["e_tot"] == mf_ll.e_tot
    assert ints.layer_scf_status(SimpleNamespace()) is None
    hcore, enuc = ints.layer_footing(layer)
    np.testing.assert_array_equal(hcore, m.intor("int1e_kin") + m.intor("int1e_nuc"))
    assert enuc == m.energy_nuc()


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #
class _DFTOnlyBackend:
    """A DFT-in-DFT-only code: matrices but no ERIs / exact exchange (SIESTA-like)."""

    capabilities = frozenset({"arbitrary_dm_veff"})


def test_require_capability():
    require_capability(_DFTOnlyBackend(), "arbitrary_dm_veff", "x")
    require_capability(SimpleNamespace(), "eri", "x")  # undeclared stand-in: let through
    with pytest.raises(MissingCapabilityError, match="'eri' capability.*_DFTOnlyBackend"):
        require_capability(_DFTOnlyBackend(), "eri", "the downfold")


def test_avas_projector_requires_the_minao_capability():
    from embasi_qiskit_integration.selectors import avas_ao_projector

    class _NoMinao(_DFTOnlyBackend):
        def ao_atom_map(self):
            return np.zeros(2, dtype=int)

    with pytest.raises(MissingCapabilityError, match="minao_projector"):
        avas_ao_projector(_NoMinao(), ["O 2p"])


def test_downfold_refuses_a_backend_without_eris():
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    adapter = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    adapter.ints = _DFTOnlyBackend()
    with pytest.raises(MissingCapabilityError, match="'hcore' capability"):
        adapter._require_downfold_capabilities("the active-space downfold")


def test_pyscf_backend_declares_everything_the_downfold_needs(water_uks):
    _m, mf_hl, mf_ll = water_uks
    caps = PySCFIntegrals(mf_hl, mf_ll).capabilities
    assert {"eri", "exact_exchange", "hcore", "arbitrary_dm_veff", "energy_elec"} <= caps
    assert "minao_projector" in caps


# --------------------------------------------------------------------------- #
# SCF guard through a backend-supplied status
# --------------------------------------------------------------------------- #
def test_scf_guard_honours_a_backend_status():
    from embasi_qiskit_integration.projection_embedding_adapter import _require_scf_converged

    layer = SimpleNamespace()  # no PySCF method behind it
    with pytest.raises(RuntimeError, match="did not converge in 40 cycles"):
        _require_scf_converged(layer, "x", {"converged": False, "max_cycle": 40, "e_tot": None})
    _require_scf_converged(layer, "x", {"converged": True, "max_cycle": 40})
    _require_scf_converged(layer, "x", {"converged": False, "max_cycle": 0})  # non-SCF
    _require_scf_converged(layer, "x", None)  # backend exposes no flag


# --------------------------------------------------------------------------- #
# Registry / --qm_code (Stage 2)
# --------------------------------------------------------------------------- #
def test_only_pyscf_is_registered_and_is_the_default():
    from embasi_qiskit_integration.backends.registry import (
        DEFAULT_QM_CODE,
        available_backends,
    )

    assert DEFAULT_QM_CODE == "pyscf"
    assert available_backends() == ["pyscf"]


def test_make_backend_refuses_an_unknown_code():
    from embasi_qiskit_integration.backends.registry import make_backend

    with pytest.raises(NotImplementedError, match="'siesta' is not available.*pyscf"):
        make_backend("siesta", atoms=None, charge=0, cfg=SimpleNamespace())


def test_pyscf_only_settings_are_refused_for_another_backend():
    from embasi_qiskit_integration.backends.registry import check_backend_settings

    check_backend_settings("pyscf", SimpleNamespace(scf_newton=True, density_fit=True))
    check_backend_settings("cp2k", SimpleNamespace(scf_newton=False, init_guess=None))
    with pytest.raises(ValueError, match="--scf_newton is a pyscf option.*--qm_code cp2k"):
        check_backend_settings("cp2k", SimpleNamespace(scf_newton=True))
    with pytest.raises(ValueError, match="--df_auxbasis is a pyscf option"):
        check_backend_settings("cp2k", SimpleNamespace(df_auxbasis="cc-pvdz-ri"))


def test_register_backend_rejects_a_duplicate_name():
    from embasi_qiskit_integration.backends.registry import available_backends, register_backend

    available_backends()  # make sure pyscf is loaded
    with pytest.raises(ValueError, match="already registered"):
        register_backend("pyscf")(lambda atoms, charge, cfg: None)


def test_pyscf_builder_constructs_matching_calculators_and_integrals():
    """The builder hands EmbASI and the integrals the SAME mean fields (h_emb relies on it)."""
    from ase import Atoms

    from embasi_qiskit_integration.backends.registry import make_backend

    atoms = Atoms("OH2", positions=[(0, 0, 0), (0, 0.757, 0.587), (0, -0.757, 0.587)])
    cfg = SimpleNamespace(
        basis="sto-3g",
        spin=0,
        unrestricted=True,
        xc_ll="PBE",
        xc_hl="HF",
        scf_max_cycle=17,
        init_guess="atom",
        init_guess_breaksym=None,
        scf_newton=False,
        density_fit=False,
        df_auxbasis="weigend",
    )
    backend = make_backend("pyscf", atoms, 0, cfg)
    ints = backend.integrals
    assert backend.name == "pyscf"
    assert backend.calc_ll.method is ints.mf_ll
    assert backend.calc_hl.method is ints.mf
    assert ints.ll_spin_resolved  # unrestricted -> UKS
    assert ints.mf_ll.xc == "PBE" and ints.mf.xc == "HF"
    assert ints.mf_ll.max_cycle == ints.mf.max_cycle == 17
    assert ints.mf_ll.init_guess == "atom"
    assert ints.describe()["density_fit"] == "weigend"  # auxbasis wins over the bool
    assert ints.nao == 7


def test_qm_code_setting_accepts_only_implemented_backends(monkeypatch):
    import sys

    from pydantic import ValidationError

    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    monkeypatch.setattr(sys, "argv", ["prog"])
    assert EmbeddingWorkflow(_cli_parse_args=False).qm_code == "pyscf"
    with pytest.raises(ValidationError, match="qm_code"):
        EmbeddingWorkflow(_cli_parse_args=False, qm_code="siesta")
