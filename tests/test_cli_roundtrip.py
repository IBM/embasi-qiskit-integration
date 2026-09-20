# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Two-process file handoff: CLI writes result.npz / result.ERROR."""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration import ipc
from embasi_qiskit_integration.cli import main
from embasi_qiskit_integration.hamiltonian import fcidump


@pytest.fixture
def n2_ham(data_dir):
    return fcidump.read(data_dir / "n2_8o10e.fcidump")


def test_cli_fci_roundtrip(tmp_path, n2_ham):
    ipc.write_job(n2_ham, tmp_path)
    rc = main(["solve", str(tmp_path), "--solver", "fci"])
    assert rc == 0

    result = ipc.read_result(tmp_path)
    assert abs(result.energy - n2_ham.meta["reference_fci_energy"]) < 1e-8
    assert result.rdm1.shape == (n2_ham.norb, n2_ham.norb)
    assert result.diagnostics["solver"] == "pyscf-fci"


def test_cli_sqd_mock_roundtrip(tmp_path, n2_ham, data_dir):
    ipc.write_job(n2_ham, tmp_path)
    counts = str(data_dir / "mock_counts.json")
    rc = main(
        [
            "solve",
            str(tmp_path),
            "--solver",
            "sqd",
            "--sampler",
            "mock",
            "--counts",
            counts,
            "--seed",
            "24",
        ]
    )
    assert rc == 0

    result = ipc.read_result(tmp_path)
    assert abs(result.energy - n2_ham.meta["reference_fci_energy"]) <= 2e-3
    assert result.rdm2 is not None
    assert result.diagnostics["sampler"] == "MockSampler"


def test_cli_error_on_corrupt_input(tmp_path):
    # No job.fcidump present -> a result.ERROR file with a traceback.
    (tmp_path / "job.fcidump").write_text("this is not a valid fcidump\n")
    rc = main(["solve", str(tmp_path), "--solver", "fci"])
    assert rc == 1
    assert (tmp_path / "result.ERROR").exists()
    assert not (tmp_path / "result.npz").exists()


def test_result_written_atomically(tmp_path, n2_ham):
    result = ipc.read_result  # sanity: function exists
    assert callable(result)
    from embasi_qiskit_integration.contract import SolverResult

    res = SolverResult(energy=-1.0, rdm1=np.eye(2), diagnostics={"x": 1})
    path = ipc.write_result(res, tmp_path)
    assert path.name == "result.npz"
    assert not (tmp_path / "result.tmp.npz").exists()  # temp cleaned up
    back = ipc.read_result(tmp_path)
    assert back.energy == -1.0
    assert back.diagnostics["x"] == 1


def test_documented_cli_flags_are_accepted(tmp_path):
    """Every flag the README shows must actually parse.

    pydantic-settings derives flag names verbatim from the field names, so they use
    *underscores* -- ``--optimization_level``, not ``--optimization-level``. The
    hyphenated spelling is rejected with exit code 2, and the README had shipped
    exactly that for ``--optimization-level``. This pins the real spellings so the
    docs and the parser cannot drift apart again.

    Exit code 1 is the pass condition: parsing succeeded and the run then failed on
    the empty job directory. Exit 2 means the parser rejected the flag.
    """
    from embasi_qiskit_integration.cli import main

    base = ["solve", str(tmp_path), "--sampler", "mock", "--counts", "tests/data/mock_counts.json"]
    documented = [
        ["--optimization_level", "3"],
        ["--measure_twirling", "false"],
        ["--dynamical_decoupling", "true"],
        ["--sampler_options", '{"twirling": {"num_randomizations": 64}}'],
        ["--optimize", "false"],
        ["--time_limit", "5.0"],
        ["--workers", "0"],
        ["--shots", "1000"],
        ["--seed", "24"],
        ["--enable_readout_characterisation", "false"],
        ["--readout_error_threshold", "0.03"],
    ]
    for extra in documented:
        assert main(base + extra) == 1, f"{extra[0]} was not accepted by the parser"


def test_hyphenated_flag_spelling_is_rejected(tmp_path):
    """Guards the assumption behind the test above: hyphens really do not work.

    If pydantic-settings ever starts accepting both spellings this fails, and the
    README could then use whichever reads better.
    """
    import pytest

    from embasi_qiskit_integration.cli import main

    with pytest.raises(SystemExit):
        main(["solve", str(tmp_path), "--optimization-level", "3"])


def test_spin_resolved_rdms_survive_the_job_directory(tmp_path):
    """``rdm1a``/``rdm1b`` must cross the process boundary, not be silently dropped.

    ``write_result`` used to serialise only ``energy``/``rdm1``/``rdm2``/``diagnostics``,
    so an unrestricted solve lost its spin resolution on the way out: nothing raised,
    ``is_spin_resolved`` came back ``False``, and the unrestricted outer-loop branch
    fell back to the spin-summed density.  That is invisible in the energy until the
    open-shell physics is wrong, so pin the round-trip.
    """
    from embasi_qiskit_integration.contract import SolverResult

    rdm1a = np.diag([1.0, 1.0, 0.0])
    rdm1b = np.diag([1.0, 0.0, 0.0])
    res = SolverResult(energy=-1.5, rdm1=rdm1a + rdm1b, rdm1a=rdm1a, rdm1b=rdm1b, diagnostics={})
    ipc.write_result(res, tmp_path)
    back = ipc.read_result(tmp_path)

    assert back.is_spin_resolved
    np.testing.assert_allclose(back.rdm1a, rdm1a)
    np.testing.assert_allclose(back.rdm1b, rdm1b)
    # The sector survives too -- (2, 1) is a doublet, which a spin-summed rdm1 alone
    # cannot distinguish from any other split of three electrons.
    assert back.check_spin_sector((2, 1)) == pytest.approx(0.0, abs=1e-9)


def test_restricted_result_stays_spin_summed(tmp_path):
    """A result with no spin pair must round-trip unchanged (no empty arrays written)."""
    from embasi_qiskit_integration.contract import SolverResult

    res = SolverResult(energy=-1.0, rdm1=np.eye(3), diagnostics={})
    ipc.write_result(res, tmp_path)
    back = ipc.read_result(tmp_path)

    assert not back.is_spin_resolved
    assert back.rdm1a is None and back.rdm1b is None


def test_half_a_spin_pair_on_disk_is_rejected(tmp_path):
    """A truncated file must fail loudly rather than degrade to spin-summed.

    The pair is written together, so half of it means a corrupted or hand-edited
    ``result.npz``.  ``SolverResult``'s own validator rejects half a pair; this pins
    that ``read_result`` routes through it instead of quietly dropping the orphan.
    """
    from embasi_qiskit_integration.contract import SolverResult

    rdm1a = np.diag([1.0, 1.0, 0.0])
    rdm1b = np.diag([1.0, 0.0, 0.0])
    res = SolverResult(energy=-1.5, rdm1=rdm1a + rdm1b, rdm1a=rdm1a, rdm1b=rdm1b, diagnostics={})
    ipc.write_result(res, tmp_path)

    path = tmp_path / "result.npz"
    with np.load(path) as npz:
        kept = {k: npz[k] for k in npz.files if k != "rdm1b"}
    np.savez(path, **kept)

    with pytest.raises(ValueError, match="together"):
        ipc.read_result(tmp_path)
