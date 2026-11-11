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
_FORBIDDEN_MODULES = ("pyscf.gto", "pyscf.scf", "pyscf.dft", "pyscf.df", "pyscf.pbc")
# (`ao_labels` / `search_ao_label` are protocol methods on the backend now, so they are
# not in this list; the Mole-only spellings are.)
_FORBIDDEN_ATTRS = {"intor", "intor_symmetric", "aoslice_by_atom", "get_veff", "get_hcore"}
# PySCF QM objects, however they are reached: `adapter.ints.mol`, an aliased
# `ints = adapter.ints; ints.mf_ll`, or `layer.atoms.calc.mol`.  Nothing outside
# `backends/` has a legitimate attribute by these names.
# `_hf` is PySCFIntegrals' private restricted integral engine.
_FORBIDDEN_HANDLES = {"mol", "mf", "mf_ll", "mf_hl", "_hf"}
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


def _is_forbidden_module(name: str) -> bool:
    return name == "pyscf" or any(
        name == mod or name.startswith(mod + ".") for mod in _FORBIDDEN_MODULES
    )


def _offences(rel: str, tree: ast.AST) -> list[str]:
    """Every PySCF QM-code access in ``tree`` (the scan the guard test applies)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # `from pyscf import scf` reaches the forbidden module through the imported
            # name; `from pyscf import lib` is the math library and stays allowed.
            names = [node.module] if node.module != "pyscf" else []
            names += [f"{node.module}.{a.name}" for a in node.names]
            if any(_is_forbidden_module(n) and n != "pyscf" for n in names):
                out.append(f"{rel}:{node.lineno} from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_module(alias.name):
                    out.append(f"{rel}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.Attribute):
            if node.attr in _FORBIDDEN_ATTRS or node.attr in _FORBIDDEN_HANDLES:
                out.append(f"{rel}:{node.lineno} .{node.attr}")
            # The PySCF SCF object behind an EmbASI layer.
            elif node.attr == "method" and _names_attr(node.value, "calc"):
                out.append(f"{rel}:{node.lineno} .calc.method")
        elif isinstance(node, ast.Call) and _is_getattr(node):
            # The string spelling: `getattr(ints, "mol", None)`, `getattr(calc, "method")`.
            attr = node.args[1].value
            if attr in _FORBIDDEN_ATTRS or attr in _FORBIDDEN_HANDLES:
                out.append(f"{rel}:{node.lineno} getattr(..., {attr!r})")
            elif attr == "method" and _names_attr(node.args[0], "calc"):
                out.append(f"{rel}:{node.lineno} getattr(<calc>, 'method')")
    return out


def _is_getattr(node: ast.Call) -> bool:
    """``getattr(obj, "<constant>", ...)``."""
    return (
        isinstance(node.func, ast.Name)
        and node.func.id in ("getattr", "hasattr")
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    )


def _names_attr(node: ast.AST, attr: str) -> bool:
    """Whether ``node`` reaches ``attr``, as ``x.<attr>`` or ``getattr(x, "<attr>")``."""
    if isinstance(node, ast.Attribute):
        return node.attr == attr
    return isinstance(node, ast.Call) and _is_getattr(node) and node.args[1].value == attr


def test_no_pyscf_qm_code_access_outside_backends():
    offences = [o for rel, tree in _package_modules() for o in _offences(rel, tree)]
    assert not offences, "PySCF QM-code access outside backends/:\n" + "\n".join(offences)


def test_guard_catches_aliased_and_submodule_access():
    """The scan flags each pattern it exists to stop, including the indirect spellings.

    ``reference_protocol`` on ``main`` read ``ints = adapter.ints; ints.mf_ll.energy_elec``,
    which an ``.ints.mf_ll`` chain match never sees.
    """
    cases = {
        "ints = adapter.ints\nints.mf_ll.energy_elec(dm)": ".mf_ll",
        "basis.mol.nao": ".mol",
        "layer.atoms.calc.method.converged": ".calc.method",
        "layer.atoms.calc.mol": ".mol",
        "from pyscf.scf.uhf import UHF": "from pyscf.scf.uhf",
        "import pyscf.scf.hf": "import pyscf.scf.hf",
        "import pyscf": "import pyscf",
        "from pyscf import gto": "from pyscf import",
        "from pyscf.pbc.tools.pyscf_ase import PySCF": "from pyscf.pbc.tools",
        "x.intor('int1e_ovlp')": ".intor",
        # String / private spellings, invisible to an Attribute-only scan.
        'getattr(self.ints, "mol", None)': "getattr(..., 'mol')",
        "hasattr(ints, 'mf_ll')": "getattr(..., 'mf_ll')",
        "ints.mf_hl.xc": ".mf_hl",
        "ints._hf.get_k(d)": "._hf",
        'getattr(getattr(layer, "atoms"), "calc").method': ".calc.method",
        'getattr(layer.atoms.calc, "method", None)': "getattr(<calc>, 'method')",
        'getattr(getattr(layer.atoms, "calc"), "method")': "getattr(<calc>, 'method')",
    }
    for src, expected in cases.items():
        found = _offences("probe.py", ast.parse(src))
        assert any(expected in f for f in found), f"guard missed {src!r}: {found}"
    # PySCF as a math library on plain arrays stays allowed.
    for src in (
        "from pyscf import lib",
        "from pyscf import ao2mo",
        "from pyscf.fci import selected_ci",
        "from pyscf.mcscf.apc import Chooser",
        "from pyscf.tools import fcidump",
        "self.method = method",
        'getattr(sampler, "method", None)',
        'getattr(cfg, "qm_code", "pyscf")',
    ):
        assert not _offences("probe.py", ast.parse(src)), src


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
    # The footing is PySCF's full core Hamiltonian (ECP terms included); without an ECP
    # that is T + V_nuc, up to summation-order round-off.
    np.testing.assert_allclose(hcore, mf_ll.get_hcore(), atol=1e-12)
    np.testing.assert_allclose(hcore, m.intor("int1e_kin") + m.intor("int1e_nuc"), atol=1e-12)
    assert enuc == m.energy_nuc()


def test_layer_footing_hcore_includes_the_ecp():
    """Regression: the footing dropped ECP terms (50.5 Ha on HI/def2-SVP)."""
    from pyscf import gto, scf

    from embasi_qiskit_integration.backends.pyscf import pyscf_layer_footing

    mol = gto.M(atom="I 0 0 0; H 0 0 1.6", basis="def2-svp", ecp={"I": "def2-svp"}, verbose=0)
    assert mol._ecp  # the test is only meaningful with a real ECP
    layer = SimpleNamespace(atoms=SimpleNamespace(calc=SimpleNamespace(mol=mol)))
    hcore, enuc = pyscf_layer_footing(layer)
    np.testing.assert_allclose(hcore, scf.RHF(mol).get_hcore(), atol=1e-12)
    assert enuc == pytest.approx(mol.energy_nuc())


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #
class _DFTOnlyBackend:
    """A DFT-in-DFT-only code: matrices but no ERIs / exact exchange."""

    capabilities = frozenset({"arbitrary_dm_veff"})


def test_require_capability():
    require_capability(_DFTOnlyBackend(), "arbitrary_dm_veff", "x")
    require_capability(SimpleNamespace(), "eri", "x")  # undeclared stand-in: let through
    with pytest.raises(MissingCapabilityError, match="'eri' capability.*_DFTOnlyBackend"):
        require_capability(_DFTOnlyBackend(), "eri", "the downfold")


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
def test_qm_code_defaults_to_pyscf_and_accepts_only_registered_backends(monkeypatch):
    """``--qm_code`` and the registry list the same codes, so neither can drift."""
    import sys
    import typing

    from pydantic import ValidationError

    from embasi_qiskit_integration.backends.registry import DEFAULT_QM_CODE, available_backends
    from embasi_qiskit_integration.embedding import EmbeddingWorkflow

    assert DEFAULT_QM_CODE == "pyscf"
    assert DEFAULT_QM_CODE in available_backends()
    literal = typing.get_args(EmbeddingWorkflow.model_fields["qm_code"].annotation)
    assert sorted(literal) == available_backends()
    monkeypatch.setattr(sys, "argv", ["prog"])
    assert EmbeddingWorkflow(_cli_parse_args=False).qm_code == "pyscf"
    with pytest.raises(ValidationError, match="qm_code"):
        EmbeddingWorkflow(_cli_parse_args=False, qm_code="no-such-code")


def test_refused_qm_code_reports_its_reason_before_embasi_is_imported(monkeypatch):
    import builtins
    import sys

    from embasi_qiskit_integration.embedding import EmbeddingWorkflow, build_adapter

    real_import = builtins.__import__

    def no_embasi(name, *args, **kwargs):
        if name == "embasi" or name.startswith("embasi."):
            raise ImportError("simulated: no MPI library")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(builtins, "__import__", no_embasi)
    with pytest.raises(NotImplementedError, match="plane-wave code"):
        build_adapter(EmbeddingWorkflow(_cli_parse_args=False, qm_code="espresso"), parallel=False)


def test_make_backend_refuses_an_unknown_code():
    from embasi_qiskit_integration.backends.registry import make_backend

    with pytest.raises(NotImplementedError, match="'no-such-code' is not available.*pyscf"):
        make_backend("no-such-code", atoms=None, charge=0, cfg=SimpleNamespace())


def test_pyscf_only_settings_are_declared_and_refused_for_another_backend():
    from embasi_qiskit_integration.backends import registry
    from embasi_qiskit_integration.backends.registry import check_backend_settings

    registry.available_backends()
    assert set(registry._BACKEND_ONLY_SETTINGS["pyscf"]) == {
        "scf_newton",
        "init_guess",
        "init_guess_breaksym",
        "density_fit",
        "df_auxbasis",
    }
    check_backend_settings("pyscf", SimpleNamespace(scf_newton=True, density_fit=True))
    check_backend_settings("cp2k", SimpleNamespace(scf_newton=False, init_guess=None))
    with pytest.raises(ValueError, match="--scf_newton is a pyscf option.*--qm_code cp2k"):
        check_backend_settings("cp2k", SimpleNamespace(scf_newton=True))
    with pytest.raises(ValueError, match="--df_auxbasis is a pyscf option"):
        check_backend_settings("cp2k", SimpleNamespace(df_auxbasis="cc-pvdz-ri"))


def test_make_backend_refuses_integrals_without_capabilities(monkeypatch):
    """A backend that declares no capabilities would be treated as a stand-in (#1)."""
    from embasi_qiskit_integration.backends import registry

    registry.available_backends()
    built = {}

    def build(atoms, charge, cfg):
        return registry.QMBackend("fake", None, None, built["ints"])

    monkeypatch.setitem(registry._REGISTRY, "fake", build)
    built["ints"] = SimpleNamespace()  # no capabilities at all
    with pytest.raises(TypeError, match="without a `capabilities` frozenset"):
        registry.make_backend("fake", None, 0, SimpleNamespace())
    built["ints"] = SimpleNamespace(capabilities={"eri"})  # a set, not a frozenset
    with pytest.raises(TypeError, match="capabilities"):
        registry.make_backend("fake", None, 0, SimpleNamespace())
    built["ints"] = SimpleNamespace(capabilities=frozenset())
    assert registry.make_backend("fake", None, 0, SimpleNamespace()).integrals is built["ints"]


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


# --------------------------------------------------------------------------- #
# Protocol enforcement: PySCF fallbacks are for stand-ins only
# --------------------------------------------------------------------------- #
class _RealBackendWithoutSpinVeff:
    """A backend that declares capabilities (so it is not a stand-in) but omits methods."""

    capabilities = frozenset({"hcore", "arbitrary_dm_veff", "exact_exchange", "eri"})

    def veff_ll(self, dm):  # the spin-averaged potential
        return np.zeros((2, 2))


def test_backend_method_falls_back_only_for_stand_ins():
    from embasi_qiskit_integration.backends.base import backend_method

    def fallback(*_):
        return "pyscf"

    assert backend_method(SimpleNamespace(), "describe", fallback) is fallback
    assert backend_method(SimpleNamespace(), "describe") is None
    with pytest.raises(MissingCapabilityError, match="does not implement describe"):
        backend_method(_RealBackendWithoutSpinVeff(), "describe", fallback)


def _spin_adapter(ints):
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
    )

    ad = ProjectionEmbeddingAdapter.__new__(ProjectionEmbeddingAdapter)
    ad.ints = ints
    ad._dm_a_spin = (np.eye(2), np.zeros((2, 2)))
    return ad


def _real_adapter():
    ad = _spin_adapter(_RealBackendWithoutSpinVeff())
    ad.p = SimpleNamespace(A_LL=SimpleNamespace())
    return ad


@pytest.mark.parametrize(
    "call, match",
    [
        # Without veff_ll_spin a real backend must not get the channel mean (0.42 Ha off).
        (lambda ad: ad._veff_ll_channel(0), "veff_ll_spin"),
        (lambda ad: ad._layer_scf_status(SimpleNamespace()), "layer_scf_status"),
        (lambda ad: ad._a_fragment_footing(), "layer_footing"),
    ],
    ids=["veff_ll_spin", "layer_scf_status", "layer_footing"],
)
def test_real_backend_missing_a_method_fails(call, match):
    with pytest.raises(MissingCapabilityError, match=match):
        call(_real_adapter())


def test_stand_in_without_veff_ll_spin_still_uses_the_mean():
    class _Stub:
        def veff_ll(self, dm):
            return np.full((2, 2), 7.0)

    ad = _spin_adapter(_Stub())
    ad._dm_a = np.eye(2)
    np.testing.assert_array_equal(ad._veff_ll_channel(1), np.full((2, 2), 7.0))


# --------------------------------------------------------------------------- #
# Capability gates on AVAS, the downfold, h_emb, the footing and the K-based rankings
# --------------------------------------------------------------------------- #
def _avas_without_minao():
    from embasi_qiskit_integration.selectors import avas_ao_projector

    class _NoMinao(_DFTOnlyBackend):
        def ao_atom_map(self):
            return np.zeros(2, dtype=int)

    avas_ao_projector(_NoMinao(), ["O 2p"])


def _dft_only_adapter():
    ad = _spin_adapter(_DFTOnlyBackend())
    ad._fock = np.eye(2)
    return ad


@pytest.mark.parametrize(
    "call, match",
    [
        (_avas_without_minao, "minao_projector"),
        (
            lambda: _dft_only_adapter()._require_downfold_capabilities("the AS downfold"),
            "'hcore' capability",
        ),
        (lambda: _dft_only_adapter().h_emb, "h_emb needs the 'hcore'"),
        (lambda: _dft_only_adapter()._veff_ll_channel(0), "spin-0 embedded one-body.*'hcore'"),
        (lambda: _dft_only_adapter()._a_fragment_footing(), "footing rebase needs the 'hcore'"),
    ],
    ids=["avas", "downfold", "h_emb", "veff_channel", "footing"],
)
def test_a_backend_without_the_capability_is_refused(call, match):
    with pytest.raises(MissingCapabilityError, match=match):
        call()


def test_k_based_rankings_are_gated_on_exact_exchange():
    """Every ``ints.get_k`` call in the adapter sits behind an exact_exchange check."""
    src = (PKG / "projection_embedding_adapter.py").read_text()
    lines = src.splitlines()
    calls = [i for i, line in enumerate(lines) if "self.ints.get_k(" in line]
    assert calls, "expected get_k callers"
    for i in calls:
        window = "\n".join(lines[max(0, i - 3) : i])
        assert "require_capability(self.ints, CAP_EXACT_EXCHANGE" in window, lines[i]


# --------------------------------------------------------------------------- #
# SCF hint is backend-specific; settings are declared by the backend itself
# --------------------------------------------------------------------------- #
def test_scf_hint_names_pyscf_flags_only_for_pyscf():
    from embasi_qiskit_integration.projection_embedding_adapter import _require_scf_converged

    with pytest.raises(RuntimeError) as exc:
        _require_scf_converged(SimpleNamespace(), "x", {"converged": False, "max_cycle": 3})
    assert "--scf_newton" not in str(exc.value) and "--scf_max_cycle" in str(exc.value)
    layer = SimpleNamespace(
        atoms=SimpleNamespace(
            calc=SimpleNamespace(method=SimpleNamespace(converged=False, max_cycle=3))
        )
    )
    with pytest.raises(RuntimeError, match="--scf_newton"):
        _require_scf_converged(layer, "x")


def test_fingerprint_does_not_read_private_backend_state():
    src = (PKG / "projection_embedding_adapter.py").read_text()
    assert "_density_fit" not in src
