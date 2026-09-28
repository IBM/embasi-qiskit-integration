# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CSV diagnostics logger."""

from __future__ import annotations

import csv
import json

import numpy as np

from embasi_qiskit_integration.diagnostics_csv import (
    DIAGNOSTICS_CSV_COLUMNS,
    append_diagnostics_row,
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
