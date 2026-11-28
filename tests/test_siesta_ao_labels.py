# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SIESTA AO labels from ``.ORB_INDX`` and label-aware ``ao-avas`` targeting.

SIESTA's ASI basis descriptors carry only the atom index, so ``ao_labels`` used to
be unavailable and ``--selector ao-avas --avas_ao_labels ...`` refused.  The
low-level evaluator writes ``.ORB_INDX``; :func:`orb_indx_ao_labels` exposes it to
:class:`CallbackIntegrals` lazily.  These tests pin two things: the lazy provider
labels the basis in the ASI order, and the default all-fragment-AO selection is
unchanged (the provider is never read without labels).
"""

from __future__ import annotations

import numpy as np
import pytest

from embasi_qiskit_integration.backends.callback import CallbackIntegrals, Evaluation
from embasi_qiskit_integration.backends.siesta import (
    ao_labels_from_orbitals,
    orb_indx_ao_labels,
    read_orb_indx,
)
from embasi_qiskit_integration.selectors import fragment_ao_projector

# io ia is spec iao n l m z p sym rc isc(3) iuo
_ORB_INDX_ROWS = [
    "     1     1  1    N.1   1  2  0  0  1  F            s   5.425  0  0  0     1",
    "     2     1  1    N.1   2  2  1 -1  1  F           py   6.350  0  0  0     2",
    "     3     1  1    N.1   3  2  1  0  1  F           pz   6.350  0  0  0     3",
    "     4     1  1    N.1   4  2  1  1  1  F           px   6.350  0  0  0     4",
    "     5     2  2    C.2   1  2  0  0  1  F            s   6.161  0  0  0     5",
    "     6     2  2    C.2   2  2  1 -1  1  F           py   7.183  0  0  0     6",
    "     7     2  2    C.2   3  2  1  0  1  F           pz   7.183  0  0  0     7",
    "     8     2  2    C.2   4  2  1  1  1  F           px   7.183  0  0  0     8",
]
_ATOM_MAP = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)


def _write_orb_indx(path, rows=_ORB_INDX_ROWS):
    path.write_text(
        "orbitals in unit cell\n"
        "    io    ia is   spec iao  n  l  m  z  p          sym      rc    isc     iuo\n"
        + "\n".join(rows)
        + "\n"
    )
    return path


def _identity_ints(*, nao=8, labels=None):
    """A SIESTA-shaped CallbackIntegrals with S = I and no real evaluations."""
    overlap = np.eye(nao)

    def evaluate(_dm):
        return Evaluation((np.zeros_like(overlap),), overlap, 0.0)

    return CallbackIntegrals(
        evaluate, evaluate, None, nao=nao, basis_atoms=_ATOM_MAP, ao_labels=labels
    )


# --------------------------------------------------------------------------- #
# .ORB_INDX label provider
# --------------------------------------------------------------------------- #
def test_orb_indx_labels_match_the_basis_order(tmp_path):
    path = _write_orb_indx(tmp_path / "siesta.ORB_INDX")
    orbitals = read_orb_indx(path)
    labels = ao_labels_from_orbitals(orbitals)
    assert labels == [
        (0, "N", "2s", "s"),
        (0, "N", "2p", "py"),
        (0, "N", "2p", "pz"),
        (0, "N", "2p", "px"),
        (1, "C", "2s", "s"),
        (1, "C", "2p", "py"),
        (1, "C", "2p", "pz"),
        (1, "C", "2p", "px"),
    ]

    provider = orb_indx_ao_labels(path, nao=8, basis_atoms=_ATOM_MAP)
    assert provider() == labels


def test_orb_indx_provider_rejects_a_mismatched_basis(tmp_path):
    path = _write_orb_indx(tmp_path / "siesta.ORB_INDX")
    with pytest.raises(RuntimeError, match="lists 8 orbitals but the ASI basis has 9"):
        orb_indx_ao_labels(path, nao=9)()
    wrong_map = _ATOM_MAP.copy()
    wrong_map[0] = 1
    with pytest.raises(RuntimeError, match="atom map differs"):
        orb_indx_ao_labels(path, nao=8, basis_atoms=wrong_map)()


def test_polarisation_functions_keep_their_star(tmp_path):
    # SIESTA marks DZP polarisation functions with '*': "3d*" is a polarisation
    # function, never the (absent) 3d valence shell.  A label match must respect it.
    rows = [
        "     1     1  1    N.1   1  2  0  0  1  F            s   5.425  0  0  0     1",
        "     2     1  1    N.1   2  3  2  0  1  T         Pdz2   6.042  0  0  0     2",
    ]
    path = _write_orb_indx(tmp_path / "siesta.ORB_INDX", rows=rows)
    labels = orb_indx_ao_labels(path, nao=2, basis_atoms=np.array([0, 0]))()
    assert labels == [(0, "N", "2s", "s"), (0, "N", "3d*", "Pdz2")]

    overlap = np.eye(2)

    def evaluate(_dm):
        return Evaluation((np.zeros_like(overlap),), overlap, 0.0)

    ints = CallbackIntegrals(
        evaluate,
        evaluate,
        None,
        nao=2,
        basis_atoms=np.array([0, 0]),
        ao_labels=labels,
    )
    assert ints.search_ao_label(["N 3d"]).size == 0
    np.testing.assert_array_equal(ints.search_ao_label(["N 3d*"]), [1])


# --------------------------------------------------------------------------- #
# lazy resolution in CallbackIntegrals
# --------------------------------------------------------------------------- #
def test_callback_resolves_a_label_provider_once(tmp_path):
    path = _write_orb_indx(tmp_path / "siesta.ORB_INDX")
    calls = []

    def provider():
        calls.append(1)
        return orb_indx_ao_labels(path, nao=8, basis_atoms=_ATOM_MAP)()

    ints = _identity_ints(labels=provider)
    np.testing.assert_array_equal(ints.search_ao_label(["N 2p"]), [1, 2, 3])
    np.testing.assert_array_equal(ints.search_ao_label(["C 2p"]), [5, 6, 7])
    assert calls == [1]  # cached after the first request
    assert len(ints.ao_labels()) == 8


def test_callback_rejects_a_provider_with_the_wrong_count():
    ints = _identity_ints(labels=lambda: [(0, "N", "2s", "s")])
    with pytest.raises(ValueError, match="1 labels for a 8-function basis"):
        ints.ao_labels()


# --------------------------------------------------------------------------- #
# selection: labels switch the target set; omitting them keeps every fragment AO
# --------------------------------------------------------------------------- #
def test_label_aware_projector_targets_only_the_labelled_shells(tmp_path):
    path = _write_orb_indx(tmp_path / "siesta.ORB_INDX")
    ints = _identity_ints(labels=orb_indx_ao_labels(path, nao=8, basis_atoms=_ATOM_MAP))
    all_fragment = fragment_ao_projector(ints, None, atoms=[0, 1])
    labelled = fragment_ao_projector(ints, ["N 2p", "C 2p"], atoms=[0, 1])
    # S = I: the all-AO target spans the whole fragment (M = I there), while the
    # labelled target keeps only the six 2p AOs and has no weight on either 2s.
    np.testing.assert_allclose(all_fragment, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(labelled, np.diag([0, 1, 1, 1, 0, 1, 1, 1]), atol=1e-12)


def test_default_ao_avas_never_reads_the_label_provider(tmp_path):
    path = _write_orb_indx(tmp_path / "siesta.ORB_INDX")
    calls = []

    def provider():
        calls.append(1)
        return orb_indx_ao_labels(path, nao=8, basis_atoms=_ATOM_MAP)()

    ints = _identity_ints(labels=provider)
    # The portable SIESTA path: no --avas_ao_labels -> every fragment AO, no labels.
    np.testing.assert_allclose(
        fragment_ao_projector(ints, None, atoms=[0, 1]), np.eye(8), atol=1e-12
    )
    assert calls == []
