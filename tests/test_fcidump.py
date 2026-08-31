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
