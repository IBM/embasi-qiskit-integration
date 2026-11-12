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
        assert set(npz.files) == {"e_core", "nelec", "meta_json", "fcidump_sha256"}


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


@pytest.mark.parametrize("uhf", [False, True], ids=["restricted", "uhf"])
def test_cp2k_multiplicity_ms2_and_the_nelec_override(tmp_path, rng, uhf):
    """CP2K 2023.1-2026.1 write MS2 = 2S+1 (a singlet says MS2=1, an odd NELEC-MS2): it is
    read as the multiplicity, with a warning; an explicit nelec bypasses the header."""
    path = tmp_path / "cp2k.fcidump"
    ham = _random_ham(3, (2, 2), rng)
    (fcidump.write_uhf if uhf else fcidump.write)(ham, path)
    path.with_suffix(".npz").unlink(missing_ok=True)
    path.write_text(path.read_text().replace("MS2=0", "MS2=1"))
    with pytest.warns(UserWarning, match="multiplicity"):
        assert fcidump.read(path).nelec == (2, 2)
    assert fcidump.read(path, nelec=(3, 1)).nelec == (3, 1)
    with pytest.raises(ValueError, match="does not add up"):
        fcidump.read(path, nelec=(3, 2))


@pytest.mark.parametrize("variant", ["d_exponents", "long_header", "no_commas", "orbital_energies"])
def test_restricted_reader_takes_foreign_headers_and_number_formats(tmp_path, rng, variant):
    """Molpro/NECI-style files PySCF's reader refuses: Fortran ``D`` exponents, a header
    wrapped over more than 10 lines, a comma-less namelist, ``eps i 0 0 0`` lines."""
    ham = _random_ham(3, (1, 1), rng)
    path = tmp_path / "f.fcidump"
    fcidump.write(ham, path)
    path.with_suffix(".npz").unlink()
    text = path.read_text()
    if variant == "d_exponents":
        head, _, body = text.partition("&END")
        rows = [ln.split() for ln in body.splitlines() if ln.strip()]
        text = (
            head
            + "&END\n"
            + "".join(f"{float(r[0]):.16E} {' '.join(r[1:])}\n".replace("E", "D") for r in rows)
        )
    elif variant == "long_header":
        text = text.replace("ISYM=1,", "ISYM=1,\n" + "  IUHF=0,\n" * 12, 1)
    elif variant == "no_commas":
        lines = text.splitlines()
        lines[0] = lines[0].replace(",", " ")
        text = "\n".join(lines) + "\n"
    else:
        core = text.rstrip().splitlines()[-1]
        text = text.replace(core, "  -9.9 1 0 0 0\n  -8.8 2 0 0 0\n" + core)
    path.write_text(text)
    back = fcidump.read(path)
    np.testing.assert_allclose(back.h1, ham.h1, atol=1e-14)
    np.testing.assert_allclose(back.h2, ham.h2, atol=1e-14)
    assert back.e_core == pytest.approx(ham.e_core, abs=1e-14) and back.nelec == (1, 1)


def _restricted_file(tmp_path, rng):
    ham = _random_ham(2, (1, 1), rng)
    path = tmp_path / "r.fcidump"
    fcidump.write(ham, path)
    path.with_suffix(".npz").unlink()
    return ham, path


def test_header_ecore_and_header_terminators(tmp_path, rng):
    """ECORE= in the header is the core energy when no 0 0 0 0 line follows; a terminator
    with trailing text still closes the header."""
    ham, path = _restricted_file(tmp_path, rng)
    lines = path.read_text().splitlines()
    body = [ln for ln in lines[4:] if not ln.split()[1:] == ["0", "0", "0", "0"]]
    path.write_text("\n".join([lines[0], " ECORE=1.5,", " &END,", *body]) + "\n")
    back = fcidump.read(path)
    assert back.e_core == 1.5
    np.testing.assert_allclose(back.h2, ham.h2, atol=1e-14)


@pytest.mark.parametrize(
    "bbaa, alpha_index",
    [(False, "odd"), (True, "odd"), (False, "even")],
    ids=["psi4", "psi4-bbaa", "neci"],
)
def test_interleaved_uhf_layout_of_neci_and_psi4(tmp_path, bbaa, alpha_index):
    """NECI/HANDE/Psi4 ``UHF=.TRUE.``: NORB spin orbitals, alpha/beta interleaved, only a
    core line with zero indices; Psi4 puts alpha on odd indices, NECI on even ones.  FCI
    on the file must equal direct_uhf on the integrals, and with distinct alpha/beta
    blocks the reader refuses to guess the parity."""
    from pyscf import ao2mo, fci, gto, scf

    from embasi_qiskit_integration.solvers import FCISolver

    mol = gto.M(atom="O 0 0 0; H 0 0 1.0", basis="sto-3g", spin=1, verbose=0)
    ca, cb = scf.UHF(mol).run().mo_coeff
    n, hc = ca.shape[1], scf.hf.get_hcore(mol)
    h1a, h1b = ca.T @ hc @ ca, cb.T @ hc @ cb
    aa = ao2mo.restore(1, ao2mo.full(mol, ca), n)
    bb = ao2mo.restore(1, ao2mo.full(mol, cb), n)
    ab = ao2mo.general(mol, (ca, ca, cb, cb), compact=False).reshape((n,) * 4)
    ref = fci.direct_uhf.kernel((h1a, h1b), (aa, ab, bb), n, mol.nelec)[0] + mol.energy_nuc()

    alpha, beta = (lambda p: 2 * p + 1), (lambda p: 2 * p + 2)
    if alpha_index == "even":
        alpha, beta = beta, alpha
    lines = [f"&FCI NORB={2 * n},NELEC={mol.nelectron},MS2=1,", " UHF=.TRUE.,", " /"]
    for eri, f1, f2 in ((aa, alpha, alpha), (bb, beta, beta), (ab, alpha, beta)):
        for i, j, k, m in np.ndindex(eri.shape):
            if i >= j and k >= m and abs(eri[i, j, k, m]) > 1e-12:
                idx = (
                    (f2(k), f2(m), f1(i), f1(j))
                    if bbaa and f1 is not f2
                    else (f1(i), f1(j), f2(k), f2(m))
                )
                lines.append(f"{eri[i, j, k, m]:28.20E}" + "".join(f"{x:4d}" for x in idx))
    for h, f in ((h1a, alpha), (h1b, beta)):
        lines += [
            f"{h[i, j]:28.20E}{f(i):4d}{f(j):4d}   0   0" for i, j in zip(*np.tril_indices(n))
        ]
    lines.append(f"{mol.energy_nuc():28.20E}   0   0   0   0")
    path = tmp_path / "psi4.fcidump"
    path.write_text("\n".join(lines) + "\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="alpha_index"):
            fcidump.read(path)
        ham = fcidump.read(path, alpha_index=alpha_index)
    assert ham.meta["fcidump_uhf_layout"] == "interleaved" and ham.norb == n
    assert ham.nelec == (5, 4)
    assert FCISolver().solve(ham).energy == pytest.approx(ref, abs=1e-10)


_H = " &FCI NORB=2,NELEC={n},MS2={ms2},\n &END\n"
_BODY = " 0.5 1 1 1 1\n 0.4 2 2 1 1\n-1.0 1 1 0 0\n 0.3 0 0 0 0\n"


@pytest.mark.parametrize(
    "text, match",
    [
        (_H.format(n=3, ms2=9) + _BODY, "inconsistent header"),
        (_H.format(n=3, ms2=0) + _BODY, "odd NELEC=3"),  # CP2K 7.1-2022.x: MS2=0 always
        (" &FCI NORB=2,NELEC=2,MS2=0,\n" + _BODY, "never closed"),
        (_H.format(n=2, ms2=0) + _BODY + " 0.1 2 1 1\n", "malformed integral line"),
        (_H.format(n=2, ms2=0) + " 0.5 1 1 1 1\n 0.4 2 2 1 1\n", "truncated"),
        (" &FCI NELEC=2,\n &END\n" + _BODY, "no NORB/NELEC"),
    ],
    ids=["ms2-range", "odd-nelec-ms2-0", "unclosed", "short-line", "truncated", "no-norb"],
)
def test_broken_files_are_refused(tmp_path, text, match):
    path = tmp_path / "bad.fcidump"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        fcidump.read(path)


@pytest.mark.parametrize(
    "nelec, legacy, match", [((1, 1), False, "hash mismatch"), ((2, 1), True, "3 electrons")]
)
def test_a_foreign_sidecar_is_refused(tmp_path, rng, nelec, legacy, match):
    """By the FCIDUMP hash it records, or (a sidecar from before the hash) its electron count."""
    from embasi_qiskit_integration import ipc

    ipc.write_job(_random_ham(2, (1, 1), rng), tmp_path / "a")
    ipc.write_job(_random_ham(2, nelec, rng), tmp_path / "b")
    sidecar = tmp_path / "b" / "job.npz"
    if legacy:
        with np.load(sidecar) as npz:
            old = {k: npz[k] for k in npz.files if k != "fcidump_sha256"}
        np.savez(sidecar, **old)
    sidecar.replace(tmp_path / "a" / "job.npz")
    with pytest.raises(ValueError, match=match):
        ipc.read_job(tmp_path / "a")


def test_a_spin_dependent_job_is_exact_without_its_sidecar(tmp_path, rng):
    """Jobs are written UHF=1, so losing job.npz cannot spin-average them."""
    from embasi_qiskit_integration import ipc

    ham = _random_ham(3, (2, 1), rng)
    ham = ham.model_copy(
        update={"h1a": ham.h1 + 0.1, "h1b": ham.h1 - 0.1, "h2_spin": (ham.h2,) * 3}
    )
    ipc.write_job(ham, tmp_path)
    (tmp_path / "job.npz").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back = ipc.read_job(tmp_path)
    np.testing.assert_allclose(back.h1a, ham.h1a, atol=1e-14)
    np.testing.assert_allclose(back.h1b, ham.h1b, atol=1e-14)
