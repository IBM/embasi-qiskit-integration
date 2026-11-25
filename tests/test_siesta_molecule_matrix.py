# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Opt-in small-molecule validation matrix for the SIESTA-QCOMP WF-in-DFT path.

This extends ``tests/test_siesta_qcomp_real.py`` (H2O only) to a set of small,
closed-shell, unpolarised molecules: H2O, H2, CH4 and NH3.  It exercises the
same production pieces -- :func:`~embasi_qiskit_integration.backends.siesta.
build_siesta_backend`, EmbASI's ``ProjectionEmbedding`` (SPADE + level shift),
:class:`~embasi_qiskit_integration.projection_embedding_adapter.
ProjectionEmbeddingAdapter` and :class:`~embasi_qiskit_integration.solvers.
FCISolver` -- and checks, per molecule:

* the active orbitals are S-orthonormal;
* ``Tr[S D_A]`` and ``Tr[S D_total]`` reproduce the pseudopotential valence
  electron counts, and ``Tr[S D_A] + Tr[S D_B]`` the supersystem count;
* ``P_B`` is invisible in the active space (element-wise and via the energy
  assembly's ``projector_leak``);
* ``h1`` is Hermitian and ``h2`` has the chemists'-notation permutation
  symmetries;
* every energy term is finite and the Eq. 8 identity
  ``total = e_low_total - e_low_A + e_high_A + correction`` holds;
* the active-space size and electron count are exactly the requested ones.

The high-level reference is an exact same-pseudopotential, same-basis check:
PySCF's ``fci.direct_spin1.kernel`` is called directly on the SIESTA
``h1``/``h2``/``e_core`` that the adapter exported, and its energy must agree
with ``FCISolver`` to 1e-9 Ha.  No all-electron low-level PBE reference is
used.

Fragment choice
---------------
SIESTA's ASI interface is unpolarised, and SPADE assigns whole doubly occupied
orbitals to subsystem A, so the A electron count is ``2 * a_nmos``.  The
fragment masks below are therefore chosen to give a closed-shell A, neutral
except for H2 (where the closed-shell constraint puts the single bonding pair
on one fragment):

========  ============  =========  ======  ====================================
molecule  A fragment    mask       a_nmos  A electrons (valence)
========  ============  =========  ======  ====================================
H2O       O             1,2,2      3       6  (O)
H2        H (one atom)  1,2        1       2  (the single bonding pair)
CH4       C             1,2,2,2,2  2       4  (C)
NH3       NH            1,1,2,2    3       6  (N 5 + H 1)
========  ============  =========  ======  ====================================

H2 has only one occupied MO, so SPADE's automatic singular-value gap has
nothing to cut; ``a_nmos=1`` is required and assigns the single bonding pair to
A.  The two H atoms share that pair equally, so subsystem B carries no occupied
orbital and ``P_B`` is exactly zero -- the remaining checks still validate the
whole workflow on the H2 Hamiltonian.  NH3's neutral N fragment would carry five
electrons (impossible for a restricted doubly occupied partition), so A is the
even-electron NH fragment instead.

Degeneracy guard
----------------
EmbASI's SPADE path diagonalises the transformed Fock with ``np.linalg.eig``
and builds the localized densities from those eigenvectors.  For an exactly
degenerate occupied subspace the returned eigenvectors need not be orthogonal,
and CH4's Td ``t2`` set is triply degenerate: on the ideal geometry the
localized A/B MOs fail the adapter's S-orthonormality check (max deviation
~2e-5) before the downfold runs.  The CH4 case therefore moves its H atoms by a
deterministic 1e-3 Angstrom (fixed seed) before the SIESTA run.  That splits
``t2`` -- the MOs are then S-orthonormal to ~4e-11 and every invariant below
passes -- while leaving the chemistry unchanged at this scale.  NH3's C3v
``e`` pair did not need the guard (it passed the adapter's S-orthonormality
check on the ideal geometry), and neither did the non-degenerate H2/H2O cases.

Determinism caveat
------------------
The assertions below are implementation invariants and an exact
same-pseudopotential cross-solver check.  QCOMP's ERI screening makes stored
values vary slightly between otherwise identical runs, so ``E_FCI`` and
``E_total`` are not bit-for-bit reproducible.  The quartet count, every
invariant, and the direct-vs-``FCISolver`` comparison are stable; the last is
exact because both consume the same ``h1``/``h2``.

Opt-in and environment
----------------------
The test skips unless ``SIESTA_QCOMP_REAL=1`` and skips on CI, exactly like the
existing real test.  It needs:

* ``ASI_LIB_PATH`` -- ``libsiesta.so`` built with ASI, QCOMP and hybrid-XC;
* ``SIESTA_PP_PATH`` -- pseudopotentials for H, C, N and O.

Run inside the cached builder image (no rebuild, no network)::

    docker run --rm \\
      --mount type=bind,src="$PWD",dst=/repo,readonly \\
      --mount type=volume,src=embasi-siesta-asi-cache,dst=/cache \\
      -e SIESTA_QCOMP_REAL=1 \\
      -e ASI_LIB_PATH=/cache/build/Src/libsiesta.so \\
      -e SIESTA_PP_PATH=/cache/siesta/Tests/Pseudos \\
      -e PYTHONPATH=/repo/src -e LD_LIBRARY_PATH=/cache/build/Src \\
      -e TMPDIR=/cache/pytest-tmp \\
      -w /cache/work embasi-siesta-asi-builder:851e389b567d \\
      /cache/venv/bin/pytest -q -s -p no:cacheprovider \\
      /repo/tests/test_siesta_molecule_matrix.py

PySCF must be imported before the ASI library loads SIESTA's HDF5-linked
NetCDF, as in the other real SIESTA tests.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Load PySCF (and its h5py) before libsiesta, but only when this opt-in module
# can run.  See test_siesta_qcomp_real.py for why the order matters.
_OPT_IN = os.environ.get("SIESTA_QCOMP_REAL") == "1" and not bool(os.environ.get("CI"))
if _OPT_IN:
    pytest.importorskip("pyscf.fci")
    from embasi_qiskit_integration.backends._embasi_compat import (  # noqa: E402
        install_siesta_adapter,
        prepare_projection_embedding,
    )

from embasi_qiskit_integration.backends import siesta as S  # noqa: E402
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


@dataclass(frozen=True)
class MoleculeCase:
    """One matrix entry: a molecule plus its chemically neutral A fragment."""

    name: str
    ase_key: str
    embed_mask: tuple[int, ...]
    a_nmos: int
    fragment: str
    valence_electrons: int
    n_virtual: int = 4
    #: Deterministic H-atom displacement (Angstrom) that lifts an exact
    #: electronic degeneracy; see the module docstring.  Zero for most cases.
    symmetry_break_angstrom: float = 0.0
    #: Set when the unpolarised ASI/QCOMP stack cannot express this system.
    skip_reason: str | None = None


#: H2O first: it is the case already validated by test_siesta_qcomp_real.py,
#: so a matrix run reports the known-good reference before the new cases.
CASES: tuple[MoleculeCase, ...] = (
    MoleculeCase(
        name="H2O",
        ase_key="H2O",
        embed_mask=(1, 2, 2),  # O is A; the two H are B
        a_nmos=3,  # O: 6 valence e in 3 doubly occupied A orbitals
        fragment="O",
        valence_electrons=8,
    ),
    MoleculeCase(
        name="H2",
        ase_key="H2",
        embed_mask=(1, 2),  # one H is A, the other B
        a_nmos=1,  # one occupied MO; SPADE's automatic gap has no cut to make
        fragment="H",
        valence_electrons=2,
    ),
    MoleculeCase(
        name="CH4",
        ase_key="CH4",
        embed_mask=(1, 2, 2, 2, 2),  # C is A; the four H are B
        a_nmos=2,  # C: 4 valence e in 2 doubly occupied A orbitals
        fragment="C",
        valence_electrons=8,
        # EmbASI's SPADE eigensolver uses np.linalg.eig; at exact Td symmetry
        # the triply degenerate t2 occupied subspace comes back with
        # non-orthonormal eigenvectors (measured max 2e-5), so the localized
        # A/B MOs fail the S-orthonormality check before the downfold.  A
        # deterministic 1e-3 A H displacement lifts the degeneracy (MO
        # orthonormality then ~4e-11); the chemistry is unchanged at this
        # scale.  See the module docstring's "Degeneracy guard" section.
        symmetry_break_angstrom=1.0e-3,
    ),
    MoleculeCase(
        name="NH3",
        ase_key="NH3",
        embed_mask=(1, 1, 2, 2),  # N + one H are A; the other two H are B
        a_nmos=3,  # NH: 6 valence e (neutral N would be an odd 5)
        fragment="NH",
        valence_electrons=8,
    ),
)

_CASE_IDS = [case.name for case in CASES]

#: Fixed seed for the symmetry-breaking H displacements; keeps them reproducible.
_SYMMETRY_BREAK_SEED = 20240607


def _atoms(case: MoleculeCase):
    """ASE atoms for a case, with its optional deterministic symmetry break."""
    from ase.build import molecule

    atoms = molecule(case.ase_key)
    if case.symmetry_break_angstrom:
        rng = np.random.default_rng(_SYMMETRY_BREAK_SEED)
        for index, symbol in enumerate(atoms.get_chemical_symbols()):
            if symbol == "H":
                atoms.positions[index] += case.symmetry_break_angstrom * rng.standard_normal(3)
    return atoms


def _require_real_siesta(case: MoleculeCase) -> None:
    """Skip unless the ASI library, pseudopotentials and asi4py are usable."""
    version = S._asi4py_version()
    if version is None or version < S.MIN_ASI4PY:
        pytest.skip("asi4py >= 1.4.10 is required for SIESTA flavour 3")
    lib = os.environ.get("ASI_LIB_PATH")
    pseudo = os.environ.get("SIESTA_PP_PATH")
    if not lib or not pseudo:
        pytest.skip("ASI_LIB_PATH and SIESTA_PP_PATH must be set")
    if not Path(lib).is_file():
        pytest.skip(f"ASI_LIB_PATH does not name a file: {lib}")
    pseudo_dir = Path(pseudo)
    if not pseudo_dir.is_dir():
        pytest.skip(f"SIESTA_PP_PATH is not a directory: {pseudo_dir}")
    # The case's own elements, so a molecule-specific missing pseudo is named.
    symbols = set(_atoms(case).get_chemical_symbols())
    missing = sorted(
        symbol
        for symbol in symbols
        if not (pseudo_dir / f"{symbol}.psml").is_file()
        and not (pseudo_dir / f"{symbol}.psf").is_file()
    )
    if missing:
        pytest.skip(f"no pseudopotential for {', '.join(missing)} in {pseudo_dir}")


def _cfg(work: Path) -> SimpleNamespace:
    """Workflow settings shared with test_siesta_qcomp_real.py (PBE low level)."""
    return SimpleNamespace(
        xc_ll="PBE",
        xc_hl="HF",
        basis="sto-3g",  # a Gaussian basis has no effect on the SIESTA path
        spin=0,
        unrestricted=False,
        scf_max_cycle=100,
        relax_hf=False,
        siesta_basis="DZP",
        siesta_mesh_cutoff=200.0,
        siesta_workdir=str(work / "work"),
    )


def _solve_case(case: MoleculeCase, tmp_path: Path) -> dict:
    """Build, run and check one molecule; return the reported metrics."""
    from embasi.embedding import ProjectionEmbedding
    from pyscf import fci

    total_started = time.monotonic()
    install_siesta_adapter()

    cfg = _cfg(tmp_path)
    atoms = _atoms(case)
    backend = S.build_siesta_backend(atoms, 0, cfg)
    capabilities = backend.integrals.capabilities
    assert {CAP_ERI, CAP_EXACT_EXCHANGE, CAP_ARBITRARY_DM_VEFF} <= capabilities

    # The pseudopotential valence count is the reference the shim's SPADE
    # partition uses; check it against the matrix table before the SCF.
    pseudo_dir = os.environ["SIESTA_PP_PATH"]
    valence = sum(S._valence_charge(symbol, pseudo_dir) for symbol in atoms.get_chemical_symbols())
    assert valence == pytest.approx(case.valence_electrons, rel=0, abs=1e-12)

    projection = ProjectionEmbedding(
        atoms,
        list(case.embed_mask),
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

    t0 = time.monotonic()
    adapter.run_low_level(a_nmos=case.a_nmos)
    t_low = time.monotonic() - t0

    assert backend.integrals._eri is not None
    assert backend.integrals._eri.n_eri > 0

    t1 = time.monotonic()
    orbitals = adapter.build_orbitals(n_frozen_occ=0, n_virtual=case.n_virtual)
    ham = adapter.embedded_hamiltonian(orbitals)
    t_ham = time.monotonic() - t1

    t2 = time.monotonic()
    result = FCISolver().solve(ham)
    # Same-basis reference: PySCF's FCI kernel on the very h1/h2/e_core the
    # adapter exported (not a Gaussian re-calculation).
    e_direct, _ci = fci.direct_spin1.kernel(ham.h1, ham.h2, ham.norb, ham.nelec)
    energy = adapter.projection_energy(result, orbitals)
    t_fci = time.monotonic() - t2

    s = adapter._s_arr
    c = orbitals.c_active
    dm_a = adapter._dm_a_arr
    dm_hl = adapter.rdm1_ao(result.rdm1, orbitals)

    # Report before asserting, so a failing molecule still prints its sizes,
    # energies and timings into the pytest log.
    metrics = {
        "case": case,
        "nao": int(s.shape[0]),
        "n_occ": int(orbitals.n_occ),
        "norb": int(ham.norb),
        "nelec": tuple(int(n) for n in ham.nelec),
        "n_eri": int(backend.integrals._eri.n_eri),
        "e_core": float(ham.e_core),
        "e_fci": float(result.energy),
        "e_direct": float(e_direct) + float(ham.e_core),
        "e_total": float(energy.total),
        "correction": float(energy.correction),
        "p_b_leak": float(energy.projector_leak),
        "footing_shift": float(energy.footing_shift),
        "t_low": t_low,
        "t_ham": t_ham,
        "t_fci": t_fci,
        "t_total": time.monotonic() - total_started,
    }
    print(_report(metrics))

    # --- exact size / electron bookkeeping ---------------------------------
    assert orbitals.n_occ == case.a_nmos
    assert len(orbitals.inactive) == 0
    assert ham.norb == orbitals.n_active_orbitals == case.a_nmos + case.n_virtual
    assert ham.nelec == (case.a_nmos, case.a_nmos)
    assert ham.h1.shape == (ham.norb, ham.norb)
    assert ham.h2.shape == (ham.norb,) * 4

    # --- active orbitals and densities -------------------------------------
    assert np.allclose(c.T @ s @ c, np.eye(ham.norb), atol=1e-8)
    assert np.einsum("ij,ji->", s, dm_a) == pytest.approx(2.0 * case.a_nmos, rel=0, abs=1e-6)
    assert np.einsum("ij,ji->", s, dm_hl) == pytest.approx(float(sum(ham.nelec)), rel=0, abs=1e-6)
    # A + B is the supersystem density and integrates to the valence count.
    assert np.einsum("ij,ji->", s, dm_a + adapter._dm_b_arr) == pytest.approx(
        case.valence_electrons, rel=0, abs=1e-6
    )
    assert np.allclose(dm_a, dm_a.T, atol=1e-10)

    # --- projector / operator invariants -----------------------------------
    assert np.abs(c.T @ adapter.p_b @ c).max() < 1e-6
    assert np.allclose(ham.h1, ham.h1.T, atol=1e-10)
    assert np.allclose(ham.h2, ham.h2.transpose(1, 0, 2, 3), atol=1e-10)
    assert np.allclose(ham.h2, ham.h2.transpose(0, 1, 3, 2), atol=1e-10)
    assert np.allclose(ham.h2, ham.h2.transpose(2, 3, 0, 1), atol=1e-10)
    assert np.isfinite(ham.h1).all() and np.isfinite(ham.h2).all()
    assert np.isfinite(ham.e_core)
    assert float(ham.meta["p_b_leak"]) < 1e-6

    # --- energy terms and the Eq. 8 identity -------------------------------
    terms = np.array(
        [
            energy.e_low_total,
            energy.e_low_A,
            energy.e_high_A,
            energy.correction,
            energy.projector_leak,
            energy.footing_shift,
            energy.total,
            result.energy,
        ],
        dtype=float,
    )
    assert np.isfinite(terms).all()
    assert abs(energy.projector_leak) < 1e-6
    assert abs(energy.footing_shift) < 1e-6
    assert energy.total == pytest.approx(
        energy.e_low_total - energy.e_low_A + energy.e_high_A + energy.correction,
        rel=0,
        abs=1e-9,
    )

    # --- same-pseudopotential FCI check -------------------------------------
    # Both paths call fci.direct_spin1 on the same tensors, so this is a
    # bit-level assembly check, not a basis/functional comparison.
    assert float(e_direct) + ham.e_core == pytest.approx(result.energy, rel=0, abs=1e-9)
    return metrics


def _report(metrics: dict) -> str:
    case = metrics["case"]
    guard = (
        f" sym_break={case.symmetry_break_angstrom:.1e}A" if case.symmetry_break_angstrom else ""
    )
    return (
        f"[matrix] {case.name}: A={case.fragment}{guard} nao={metrics['nao']} "
        f"active={metrics['nelec']}e/{metrics['norb']}o (n_occ={metrics['n_occ']}) "
        f"n_eri={metrics['n_eri']} e_core={metrics['e_core']:.10f} "
        f"E_FCI={metrics['e_fci']:.10f} E_PySCF_direct={metrics['e_direct']:.10f} "
        f"E_total={metrics['e_total']:.10f} corr={metrics['correction']:+.3e} "
        f"leak={metrics['p_b_leak']:.2e} footing={metrics['footing_shift']:.2e} "
        f"t_low={metrics['t_low']:.1f}s t_ham={metrics['t_ham']:.1f}s "
        f"t_fci={metrics['t_fci']:.1f}s t_total={metrics['t_total']:.1f}s"
    )


@pytest.mark.parametrize("case", CASES, ids=_CASE_IDS)
def test_siesta_molecule_matrix(case: MoleculeCase, tmp_path):
    """SIESTA-QCOMP WF-in-DFT invariants + same-pseudopotential PySCF FCI."""
    if case.skip_reason is not None:
        pytest.skip(case.skip_reason)
    _require_real_siesta(case)

    _solve_case(case, tmp_path)
