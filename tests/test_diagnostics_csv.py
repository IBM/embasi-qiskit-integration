# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CSV diagnostics logger."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np

from embasi_qiskit_integration.backends.base import MissingCapabilityError
from embasi_qiskit_integration.diagnostics_csv import (
    DIAGNOSTICS_CSV_COLUMNS,
    append_diagnostics_row,
    dump_active_space,
    json_cell,
    natural_occupations,
)


class TestJsonCell:
    def test_none_returns_empty_string(self):
        assert json_cell(None) == ""

    def test_list_round_trips(self):
        lst = [1.0, 2.5, -3.14]
        result = json_cell(lst)
        assert json.loads(result) == lst

    def test_numpy_array_converts_to_list(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = json_cell(arr)
        assert json.loads(result) == [1.0, 2.0, 3.0]

    def test_dict_round_trips(self):
        d = {"a": 1, "b": 2.5}
        result = json_cell(d)
        assert json.loads(result) == d


class TestNaturalOccupations:
    def test_hermitian_matrix_returns_eigenvalues(self):
        rdm1 = np.diag([2.0, 1.8, 0.2, 0.0])
        result = natural_occupations(rdm1)
        assert result is not None
        assert len(result) == 4
        assert result == sorted([2.0, 1.8, 0.2, 0.0], reverse=True)

    def test_non_hermitian_matrix_returns_none(self):
        rdm1 = np.array([[1.0, 0.5], [0.0, 1.0]])  # not symmetric
        result = natural_occupations(rdm1)
        assert result is None

    def test_non_square_matrix_returns_none(self):
        rdm1 = np.array([[1.0, 0.5, 0.1], [0.0, 1.0, 0.0]])
        result = natural_occupations(rdm1)
        assert result is None

    def test_1d_array_returns_none(self):
        rdm1 = np.array([1.0, 2.0, 3.0])
        result = natural_occupations(rdm1)
        assert result is None

    def test_none_input_returns_none(self):
        result = natural_occupations(None)
        assert result is None

    def test_slightly_non_hermitian_within_tolerance(self):
        # Create a nearly-Hermitian matrix (off-diagonal asymmetry < 1e-6)
        rdm1 = np.array([[1.0, 0.5 + 1e-7], [0.5 - 1e-7, 1.0]])
        result = natural_occupations(rdm1)
        assert result is not None
        assert len(result) == 2

    def test_slightly_non_hermitian_exceeds_tolerance(self):
        # Create a non-Hermitian matrix with asymmetry > 1e-6
        rdm1 = np.array([[1.0, 0.5 + 1e-5], [0.5 - 1e-5, 1.0]])
        result = natural_occupations(rdm1)
        assert result is None


class TestAppendDiagnosticsRow:
    def test_creates_file_if_missing(self, tmp_path):
        path = tmp_path / "diag.csv"
        row = {"run_id": "test123", "cycle": 0}
        append_diagnostics_row(path, row)

        assert path.exists()
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "test123"

    def test_writes_header_on_first_row(self, tmp_path):
        path = tmp_path / "diag.csv"
        row = {"cycle": 0}
        append_diagnostics_row(path, row)

        with path.open() as fh:
            lines = fh.readlines()
        # First line should be the header
        assert "run_id" in lines[0]
        assert "cycle" in lines[0]

    def test_does_not_duplicate_header(self, tmp_path):
        path = tmp_path / "diag.csv"
        row1 = {"cycle": 0}
        row2 = {"cycle": 1}
        append_diagnostics_row(path, row1)
        append_diagnostics_row(path, row2)

        with path.open(newline="") as fh:
            lines = fh.readlines()
        # Count lines that look like headers (contain "cycle" and "run_id")
        headers = [line for line in lines if "cycle" in line and "run_id" in line]
        assert len(headers) == 1  # Only one header, not two

    def test_restval_for_missing_keys(self, tmp_path):
        path = tmp_path / "diag.csv"
        row = {"run_id": "test", "cycle": 0}  # Missing most columns
        append_diagnostics_row(path, row)

        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # All columns should be present in the header
        assert all(col in rows[0] for col in DIAGNOSTICS_CSV_COLUMNS[:10])
        # Missing values should be empty strings
        assert rows[0].get("embedding_energy", "") == ""

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "diag.csv"
        assert not path.parent.exists()
        row = {"cycle": 0}
        append_diagnostics_row(path, row)
        assert path.exists()

    def test_appends_multiple_rows(self, tmp_path):
        path = tmp_path / "diag.csv"
        for i in range(3):
            row = {"cycle": i, "converged": False}
            append_diagnostics_row(path, row)

        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3
        assert [int(r["cycle"]) for r in rows] == [0, 1, 2]

    def test_preserves_float_precision(self, tmp_path):
        path = tmp_path / "diag.csv"
        energy = 1.23456789123456789
        row = {"E_solver": energy}
        append_diagnostics_row(path, row)

        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        # str(float) round-trips exactly
        recovered = float(rows[0]["E_solver"])
        assert recovered == energy


class TestDumpActiveSpace:
    def _fixture(self, tmp_path, labels):
        nao, nocc = 4, 2
        coeff = np.eye(nao)
        active = np.array([1, 3])
        overlap = np.eye(nao)

        class _Ints:
            def overlap(self):
                return overlap

            def ao_atom_map(self):
                return np.array([0, 0, 1, 1])

            def ao_labels(self):
                if labels is None:
                    raise MissingCapabilityError("no labels")
                return labels

            def describe(self):
                return {"atom_symbols": ["N", "C"], "atom_coords": np.zeros((2, 3))}

        emb = SimpleNamespace(ints=_Ints(), _avas_info={"projector": "working-basis fragment AOs"})
        orbitals = SimpleNamespace(
            coeff=coeff,
            active=active,
            inactive=np.array([0, 2]),
            n_occ=nocc,
            n_occ_b=None,
            energy=np.array([-1.0, -0.5, 0.2, 0.8]),
        )
        result = SimpleNamespace(rdm1=np.diag([1.9, 0.1]))
        cfg = SimpleNamespace(
            selector="ao-avas",
            avas_ao_labels=["N 2p"],
            avas_max_size=(2, 2),
            n_frozen_occ=0,
            qm_code="siesta",
            active_atoms=[0, 1],
        )
        return emb, orbitals, result, cfg

    def test_writes_coefficients_labels_and_rdm(self, tmp_path):
        labels = [
            (0, "N", "2s", "s"),
            (0, "N", "2p", "px"),
            (1, "C", "2s", "s"),
            (1, "C", "2p", "px"),
        ]
        emb, orbitals, result, cfg = self._fixture(tmp_path, labels)
        path = dump_active_space(
            tmp_path / "a" / "dump.npz",
            emb=emb,
            orbitals=orbitals,
            result=result,
            cfg=cfg,
            cycle=0,
            geometry_file="08.inp",
            geometry_parameter=0.8,
        )
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["active"], [1, 3])
            np.testing.assert_array_equal(data["coeff_active"], np.eye(4)[:, [1, 3]])
            np.testing.assert_array_equal(data["label_shell"], ["2s", "2p", "2s", "2p"])
            np.testing.assert_allclose(data["rdm1_active"], np.diag([1.9, 0.1]))
            meta = json.loads(data["meta"].item())
        assert meta["selector"] == "ao-avas"
        assert meta["active_atoms"] == [0, 1]
        assert meta["geometry_parameter"] == 0.8

    def test_backend_without_labels_still_writes_the_dump(self, tmp_path):
        emb, orbitals, result, cfg = self._fixture(tmp_path, labels=None)
        path = dump_active_space(
            tmp_path / "dump.npz", emb=emb, orbitals=orbitals, result=result, cfg=cfg
        )
        with np.load(path, allow_pickle=False) as data:
            assert "label_shell" not in data
            np.testing.assert_array_equal(data["ao_atom_map"], [0, 0, 1, 1])

    def test_writes_the_active_space_hamiltonian_when_supplied(self, tmp_path):
        emb, orbitals, result, cfg = self._fixture(tmp_path, labels=None)
        ham = SimpleNamespace(
            norb=2,
            nelec=(1, 1),
            e_core=0.25,
            h1=np.array([[1.0, 0.1], [0.1, 2.0]]),
            h2=np.arange(16.0).reshape(2, 2, 2, 2),
        )
        path = dump_active_space(
            tmp_path / "dump.npz",
            emb=emb,
            orbitals=orbitals,
            result=result,
            cfg=cfg,
            ham=ham,
        )
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_allclose(data["active_energy"], [-0.5, 0.8])
            np.testing.assert_array_equal(data["ham_nelec"], [1, 1])
            np.testing.assert_allclose(data["h1"], ham.h1)
            np.testing.assert_allclose(data["h2"], ham.h2)

    def _downfolded_fixture(self, tmp_path):
        emb, orbitals, result, cfg = self._fixture(tmp_path, labels=None)
        nao = 4
        hcore = 0.5 * np.eye(nao)
        h_emb = np.diag([-1.0, -0.5, 0.4, 0.9])
        f_emb = h_emb + np.diag([0.05, 0.05, 0.02, 0.02])
        dm_a_init = 2.0 * np.eye(nao)[:, :2] @ np.eye(nao)[:, :2].T

        def veff_hf(dm):
            return 0.1 * np.asarray(dm)

        def veff_ll(dm):
            return 0.2 * np.asarray(dm)

        class _DownfoldedInts:
            def __init__(self):
                self._ints = emb.ints

            def __getattr__(self, name):
                return getattr(self._ints, name)

            def hcore(self):
                return hcore

            def veff_hf(self, dm):
                return veff_hf(dm)

            def veff_ll(self, dm):
                return veff_ll(dm)

        canonical_coeff = np.eye(nao)
        canonical_energy = np.array([-1.0, -0.5, 0.2, 0.8])
        emb.ints = _DownfoldedInts()
        emb.h_emb = h_emb
        emb._fock_arr = f_emb
        emb.v_emb = h_emb - hcore
        emb.p_b = np.zeros((nao, nao))
        emb._dm_a_init = dm_a_init
        emb._avas_info = {
            **emb._avas_info,
            "canonical_coeff": canonical_coeff,
            "canonical_energy": canonical_energy,
            "ao_projector": np.eye(nao),
        }
        return (
            emb,
            orbitals,
            result,
            cfg,
            {
                "hcore": hcore,
                "h_emb": h_emb,
                "f_emb": f_emb,
                "dm_a_init": dm_a_init,
                "canonical_coeff": canonical_coeff,
                "canonical_energy": canonical_energy,
            },
        )

    def test_downfolded_onebody_block_is_opt_in(self, tmp_path):
        emb, orbitals, result, cfg, expected = self._downfolded_fixture(tmp_path)
        default = dump_active_space(
            tmp_path / "default.npz", emb=emb, orbitals=orbitals, result=result, cfg=cfg
        )
        with np.load(default, allow_pickle=False) as data:
            assert "h_emb" not in data
            assert "veff_hf_in" not in data
            assert "canonical_coeff" not in data
            assert "downfolded_onebody" not in data

        path = dump_active_space(
            tmp_path / "extended.npz",
            emb=emb,
            orbitals=orbitals,
            result=result,
            cfg=cfg,
            downfolded_onebody=True,
        )
        with np.load(path, allow_pickle=False) as data:
            assert bool(data["downfolded_onebody"]) is True
            np.testing.assert_allclose(data["h_emb"], expected["h_emb"])
            np.testing.assert_allclose(data["f_emb"], expected["f_emb"])
            np.testing.assert_allclose(data["hcore"], expected["hcore"])
            np.testing.assert_allclose(data["dm_a_init"], expected["dm_a_init"])
            np.testing.assert_allclose(data["canonical_coeff"], expected["canonical_coeff"])
            np.testing.assert_allclose(data["canonical_energy"], expected["canonical_energy"])
            # dm_in is the inactive density 2 c_in c_in^T; dm_act_occ the active
            # occupied one.  The fixture has inactive columns [0, 2] and active [1].
            c_in = np.eye(4)[:, [0, 2]]
            np.testing.assert_allclose(data["dm_in"], 2.0 * c_in @ c_in.T)
            np.testing.assert_allclose(data["dm_act_occ"], 2.0 * np.diag([0, 1.0, 0, 0]))
            np.testing.assert_allclose(data["veff_hf_in"], 0.1 * data["dm_in"])
            np.testing.assert_allclose(data["veff_hf_act_occ"], 0.1 * data["dm_act_occ"])
            np.testing.assert_allclose(data["veff_ll_a"], expected["f_emb"] - expected["h_emb"])
            # The standard dump's arrays are untouched by the extension.
            np.testing.assert_array_equal(data["active"], [1, 3])
            np.testing.assert_allclose(data["rdm1_active"], np.diag([1.9, 0.1]))

    def test_downfolded_onebody_tolerates_a_backend_without_veff(self, tmp_path):
        emb, orbitals, result, cfg, _ = self._downfolded_fixture(tmp_path)

        class _BareInts:
            def __init__(self):
                self._ints = emb.ints

            def overlap(self):
                return self._ints.overlap()

            def ao_atom_map(self):
                return self._ints.ao_atom_map()

            def ao_labels(self):
                return self._ints.ao_labels()

            def describe(self):
                return self._ints.describe()

        emb.ints = _BareInts()
        path = dump_active_space(
            tmp_path / "extended.npz",
            emb=emb,
            orbitals=orbitals,
            result=result,
            cfg=cfg,
            downfolded_onebody=True,
        )
        with np.load(path, allow_pickle=False) as data:
            assert "h_emb" in data
            assert "veff_hf_in" not in data
            assert "veff_ll_in" not in data
