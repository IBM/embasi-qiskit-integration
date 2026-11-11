# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""FCIDUMP + npz sidecar round-trip fidelity."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.hamiltonian import fcidump


def _random_ham(norb: int, nelec: tuple[int, int], rng: np.random.Generator, meta=None):
    a = rng.standard_normal((norb, norb))
    h1 = a + a.T
    b = rng.standard_normal((norb, norb, norb, norb))
    b = b + b.transpose(1, 0, 2, 3)
    b = b + b.transpose(0, 1, 3, 2)
    b = b + b.transpose(2, 3, 0, 1)
    return EmbeddedHamiltonian(h1=h1, h2=b, e_core=-3.14159, nelec=nelec, meta=meta or {})


def test_roundtrip_exact(tmp_path, rng):
    ham = _random_ham(4, (3, 2), rng, meta={"sha": "deadbeef", "note": "n2-like"})
    path = tmp_path / "job.fcidump"
    fcidump.write(ham, path)
    back = fcidump.read(path)

    assert np.max(np.abs(back.h1 - ham.h1)) < 1e-12
    assert np.max(np.abs(back.h2 - ham.h2)) < 1e-12
    assert abs(back.e_core - ham.e_core) < 1e-12
    assert back.nelec == ham.nelec
    assert back.meta == ham.meta


def test_sidecar_written(tmp_path, rng):
    ham = _random_ham(3, (2, 1), rng)
    path = tmp_path / "job.fcidump"
    fcidump.write(ham, path)
    assert (tmp_path / "job.npz").exists()


def test_read_without_sidecar_closed_shell(tmp_path, rng):
    """Without the sidecar, a closed shell is still recovered from the header.

    ``MS2=0`` pins the split, so this case needs no extra information.
    """
    ham = _random_ham(4, (2, 2), rng)
    path = tmp_path / "job.fcidump"
    fcidump.write(ham, path)
    (tmp_path / "job.npz").unlink()

    back = fcidump.read(path)
    assert back.nelec == (2, 2)
    assert abs(back.e_core - ham.e_core) < 1e-12
    assert back.meta == {}


def test_signed_ms2_roundtrips_open_shell_without_a_sidecar(tmp_path, rng):
    """A signed ``MS2`` pins the spin sector, so the header alone is enough.

    ``write`` emits ``MS2 = n_alpha - n_beta``, so ``(1, 2)`` writes ``MS2=-1`` and
    ``(2, 1)`` writes ``MS2=+1``: the two are distinguishable with no sidecar.
    """
    for nelec in ((2, 1), (1, 2), (3, 1)):
        ham = _random_ham(3, nelec, rng)
        path = tmp_path / f"job_{nelec[0]}{nelec[1]}.fcidump"
        fcidump.write(ham, path)
        path.with_suffix(".npz").unlink()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert fcidump.read(path).nelec == nelec


def test_unsigned_positive_ms2_warns_instead_of_refusing(tmp_path, rng):
    """A positive ``MS2`` from an *unsigned* writer is read alpha-rich, with a warning.

    We cannot tell ``(2, 1)`` from ``(1, 2)`` in an unsigned header, so the reader
    takes the near-universal alpha-rich convention and says so. Refusing outright
    would also reject our own signed files, which is worse.
    """
    ham = _random_ham(3, (1, 2), rng)
    path = tmp_path / "unsigned.fcidump"
    fcidump.write(ham, path, signed_ms2=False)  # as PySCF's from_mcscf would
    path.with_suffix(".npz").unlink()

    with pytest.warns(UserWarning, match="alpha-rich"):
        # MS2=+1 is indistinguishable from the (2, 1) sector without the sidecar.
        assert fcidump.read(path).nelec == (2, 1)


def test_inconsistent_header_is_rejected(tmp_path, rng):
    """``|MS2| > NELEC`` or a parity mismatch cannot be a spin split at all."""
    ham = _random_ham(3, (2, 1), rng)
    path = tmp_path / "bad.fcidump"
    fcidump.write(ham, path)
    path.with_suffix(".npz").unlink()
    text = path.read_text().replace("MS2=1", "MS2=9")
    path.write_text(text)

    with pytest.raises(ValueError, match="inconsistent header"):
        fcidump.read(path)


def test_open_shell_roundtrip_uses_the_sidecar(tmp_path, rng):
    """With the sidecar, both spin orderings round-trip exactly.

    The sidecar is what makes the split authoritative, so ``(1, 2)`` must come back
    as ``(1, 2)`` and not as its alpha-rich mirror.
    """
    for nelec in ((2, 1), (1, 2)):
        ham = _random_ham(3, nelec, rng)
        path = tmp_path / f"os_{nelec[0]}{nelec[1]}.fcidump"
        fcidump.write(ham, path)
        assert fcidump.read(path).nelec == nelec


def test_written_header_is_parseable_by_qiskit_fermions(tmp_path, rng):
    """The header must stay readable by the qiskit-fermions loader.

    ``circuit_generator.operator`` writes a temp FCIDUMP and immediately reloads it
    through ``FCIDump.from_file``, whose Rust MS2 parse *panics* (not raises) on a
    negative value. That path therefore passes ``signed_ms2=False``; this test pins
    that the unsigned header it writes stays parseable, including for the beta-rich
    ``(1, 2)`` sector that a signed write would crash on.
    """
    pytest.importorskip("qiskit_fermions")
    from qiskit_fermions.operators.library import FCIDump

    for nelec in ((2, 1), (1, 2), (2, 2)):
        ham = _random_ham(3, nelec, rng)
        path = tmp_path / f"qf_{nelec[0]}{nelec[1]}.fcidump"
        fcidump.write(ham, path, signed_ms2=False)
        loaded = FCIDump.from_file(str(path))
        assert loaded.norb == 3
        assert loaded.nelec == sum(nelec)


def test_closed_shell_roundtrip(tmp_path, rng):
    ham = _random_ham(5, (3, 3), rng, meta={"reference_energy": -108.1})
    path = tmp_path / "cs.fcidump"
    fcidump.write(ham, path)
    back = fcidump.read(path)
    assert back.nelec == (3, 3)
    assert np.max(np.abs(back.h2 - ham.h2)) < 1e-12
    assert back.meta["reference_energy"] == -108.1


# --------------------------------------------------------------------------- #
# Unrestricted (UHF=1) FCIDUMP and the lossless spin sidecar
# --------------------------------------------------------------------------- #
def _uhf_ham():
    """A genuinely unrestricted Hamiltonian: OH doublet UHF, both channels' own orbitals."""
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    ca, cb = mf.mo_coeff
    n = ca.shape[1]
    hcore = mf.get_hcore()
    h1a, h1b = ca.T @ hcore @ ca, cb.T @ hcore @ cb
    aa = ao2mo.restore(1, ao2mo.kernel(mol, ca), n)
    bb = ao2mo.restore(1, ao2mo.kernel(mol, cb), n)
    ab = ao2mo.general(mol, (ca, ca, cb, cb), compact=False).reshape(n, n, n, n)
    ham = EmbeddedHamiltonian(
        h1=0.5 * (h1a + h1b),
        h2=aa,
        e_core=float(mol.energy_nuc()),
        nelec=mol.nelec,
        h1a=h1a,
        h1b=h1b,
        h2_spin=(aa, ab, bb),
    )
    return ham, mf


def _fci(ham):
    from embasi_qiskit_integration.solvers import FCISolver

    return FCISolver().solve(ham).energy


def _assert_same_spin_blocks(a, b, atol=1e-12):
    np.testing.assert_allclose(a.h1a, b.h1a, atol=atol)
    np.testing.assert_allclose(a.h1b, b.h1b, atol=atol)
    for x, y in zip(a.h2_spin, b.h2_spin, strict=True):
        np.testing.assert_allclose(x, y, atol=atol)
    assert a.nelec == b.nelec
    assert a.e_core == pytest.approx(b.e_core, abs=atol)


def _write_separated(ham, path):
    """Molpro-style: indices restart at 1, a `0 0 0 0` line closes each block."""
    n = ham.norb
    aa, ab, bb = ham.h2_spin
    lines = [f"&FCI NORB={n},NELEC={sum(ham.nelec)},MS2={ham.nelec[0] - ham.nelec[1]},UHF=1,", "/"]

    def eri(t, same):
        for i in range(n):
            for j in range(i + 1):
                for k in range(n):
                    for m in range(k + 1):
                        if same and (i * (i + 1) // 2 + j) < (k * (k + 1) // 2 + m):
                            continue
                        if abs(t[i, j, k, m]) > 1e-14:
                            lines.append(f"{t[i, j, k, m]:.16E} {i + 1} {j + 1} {k + 1} {m + 1}")
        lines.append("0.0 0 0 0 0")

    eri(aa, True)
    eri(bb, True)
    eri(ab, False)
    for h in (ham.h1a, ham.h1b):
        for i in range(n):
            for j in range(i + 1):
                lines.append(f"{h[i, j]:.16E} {i + 1} {j + 1} 0 0")
        lines.append("0.0 0 0 0 0")
    lines.append(f"{ham.e_core:.16E} 0 0 0 0")
    path.write_text("\n".join(lines) + "\n")


def _write_beta_alpha(ham, path):
    """CP2K writes (aa|bb) with the beta pair second; a (bb|aa) line must land transposed."""
    fcidump.write_uhf(ham, path)
    n = ham.norb
    text = path.read_text().splitlines()
    swapped = []
    for line in text:
        parts = line.split()
        if len(parts) == 5 and not line.lstrip().startswith(("&", "/", "ORBSYM", "ISYM", "UHF")):
            i, j, k, m = (int(x) for x in parts[1:])
            if 0 < i <= n and k > n:  # an (aa|bb) line -> write it as (bb|aa)
                parts = [parts[0], str(k), str(m), str(i), str(j)]
        swapped.append(" ".join(parts) if len(parts) == 5 else line)
    path.write_text("\n".join(swapped) + "\n")


_WRITERS = {
    "spin-orbital": fcidump.write_uhf,
    "separated": _write_separated,
    "beta-alpha": _write_beta_alpha,
}


@pytest.mark.parametrize("layout", list(_WRITERS))
def test_uhf_fcidump_reads_back_exactly_in_every_layout(tmp_path, layout):
    """Full space of an OH doublet: an alpha/beta mix-up would move FCI by Hartrees."""
    ham, mf = _uhf_ham()
    path = tmp_path / "uhf.fcidump"
    _WRITERS[layout](ham, path)
    assert "UHF=1" in path.read_text().split("/")[0]
    back = fcidump.read(path)  # read() dispatches on UHF=1, no sidecar needed
    _assert_same_spin_blocks(ham, back)
    assert back.meta["fcidump_uhf_layout"] == (
        "separated" if layout == "separated" else "spin-orbital"
    )
    e = _fci(back)
    assert e < mf.e_tot  # correlation lowers the UHF energy
    assert e == pytest.approx(_fci(ham), abs=1e-10)


def test_spin_resolved_ham_roundtrips_through_write_via_the_sidecar(tmp_path):
    """write() keeps a restricted FCIDUMP (spin-averaged h1) but the sidecar keeps the
    spin blocks, so a spin-resolved job survives the solve CLI's job directory."""
    ham, _mf = _uhf_ham()
    path = tmp_path / "job.fcidump"
    with pytest.warns(UserWarning, match="sidecar keeps the spin blocks"):
        fcidump.write(ham, path)
    back = fcidump.read(path)
    _assert_same_spin_blocks(ham, back)
    assert _fci(back) == pytest.approx(_fci(ham), abs=1e-10)


def test_restricted_write_is_unchanged_by_the_spin_sidecar(tmp_path, rng):
    ham = _random_ham(3, (2, 1), rng)
    fcidump.write(ham, tmp_path / "job.fcidump")
    with np.load(tmp_path / "job.npz") as npz:
        assert set(npz.files) == {"e_core", "nelec", "meta_json"}


_RESTRICTED_BODY = " 0.5 1 1 1 1\n 0.5 2 2 2 2\n-1.0 1 1 0 0\n-0.5 2 2 0 0\n 0.0 0 0 0 0\n"
_UHF_BODY = " 0.5 1 1 1 1\n 0.5 3 3 3 3\n 0.4 1 1 3 3\n-1.0 1 1 0 0\n-0.9 3 3 0 0\n 0.0 0 0 0 0\n"


@pytest.mark.parametrize(
    "flags, body",
    [
        ("UHF=.FALSE.,TREL=.FALSE.,\n ORBSYM=1,1,\n ISYM=1,\n &END", _RESTRICTED_BODY),
        ("IUHF=1,\n /", _UHF_BODY),
        ("UHF=.TRUE.,\n /", _UHF_BODY),
    ],
    ids=["logical-false", "iuhf", "logical-true"],
)
def test_logical_and_iuhf_header_flags_select_the_reader(tmp_path, flags, body):
    """Regression: Fortran logicals and the IUHF spelling were not read."""
    path = tmp_path / "x.fcidump"
    path.write_text(f"&FCI NORB=2,NELEC=2,MS2=0,{flags}\n{body}")
    ham = fcidump.read(path)
    assert ham.norb == 2 and ham.is_spin_dependent == (body is _UHF_BODY)
    if ham.is_spin_dependent:
        assert ham.h1b[0, 0] == pytest.approx(-0.9)


def test_orbital_energy_lines_do_not_overwrite_matrix_elements(tmp_path):
    """`eps i 0 0 0` lines used to land in h1[i-1, -1] (the last column)."""
    path = tmp_path / "eps.fcidump"
    path.write_text(
        "&FCI NORB=2,NELEC=2,MS2=0,UHF=1,\n /\n"
        " 0.5 1 1 1 1\n 0.5 3 3 3 3\n 0.4 1 1 3 3\n"
        "-1.0 1 1 0 0\n-0.3 2 2 0 0\n-0.9 3 3 0 0\n-0.2 4 4 0 0\n"
        "-7.77 2 0 0 0\n-6.66 4 0 0 0\n 0.0 0 0 0 0\n"
    )
    ham = fcidump.read(path)
    np.testing.assert_allclose(np.diag(ham.h1a), [-1.0, -0.3])
    np.testing.assert_allclose(np.diag(ham.h1b), [-0.9, -0.2])


# --------------------------------------------------------------------------- #
# The FCIDUMP front end: import an external file, then solve it
# --------------------------------------------------------------------------- #
def _scaled(ham, factor):
    update = {"h1": factor * ham.h1, "h2": factor * ham.h2, "e_core": factor * ham.e_core}
    if ham.is_spin_dependent:
        update |= {
            "h1a": factor * ham.h1a,
            "h1b": factor * ham.h1b,
            "h2_spin": tuple(factor * b for b in ham.h2_spin),
        }
    return ham.model_copy(update=update)


@pytest.mark.parametrize(
    "layout, unit",
    [
        ("spin-orbital", "hartree"),
        ("separated", "hartree"),
        ("restricted", "rydberg"),
        ("spin-orbital", "rydberg"),
    ],
)
def test_import_fcidump_then_solve_reproduces_the_fci_energy(tmp_path, rng, layout, unit):
    """WEST writes its FCIDUMP in Rydberg; read as Hartree the energy doubles."""
    from embasi_qiskit_integration import ipc
    from embasi_qiskit_integration.cli import main

    ham = _random_ham(3, (2, 2), rng) if layout == "restricted" else _uhf_ham()[0]
    external = tmp_path / "external.fcidump"
    if layout == "restricted":
        fcidump.write(_scaled(ham, 2), external)
        fcidump._sidecar_path(external).unlink(missing_ok=True)  # a foreign file has none
    else:
        _WRITERS[layout](_scaled(ham, 2) if unit == "rydberg" else ham, external)
    job = tmp_path / "job"
    argv = ["import_fcidump", str(external), str(job), "--source", "test", "--energy_unit", unit]
    assert main(argv) == 0
    imported = ipc.read_job(job)
    np.testing.assert_allclose(imported.h1, ham.h1, atol=1e-12)
    np.testing.assert_allclose(imported.h2, ham.h2, atol=1e-12)
    assert imported.e_core == pytest.approx(ham.e_core, abs=1e-12)
    if ham.is_spin_dependent:
        _assert_same_spin_blocks(ham, imported)
    assert imported.meta["source"] == "test"
    assert main(["solve", str(job), "--solver", "fci"]) == 0
    assert ipc.read_result(job).energy == pytest.approx(_fci(ham), abs=1e-10)


def test_import_fcidump_rejects_an_inconsistent_spin_override(tmp_path, rng, capsys):
    from embasi_qiskit_integration.cli import main

    ham = _random_ham(3, (2, 2), rng)
    external = tmp_path / "x.fcidump"
    fcidump.write(ham, external)
    rc = main(
        [
            "import_fcidump",
            str(external),
            str(tmp_path / "j"),
            "--nelec_alpha",
            "3",
            "--nelec_beta",
            "2",
        ]
    )
    assert rc == 1
    assert "NELEC=4" in capsys.readouterr().err


def test_to_hartree_refuses_an_unknown_unit(rng):
    with pytest.raises(ValueError, match="unknown energy unit"):
        fcidump.to_hartree(_random_ham(2, (1, 1), rng), "ev")
