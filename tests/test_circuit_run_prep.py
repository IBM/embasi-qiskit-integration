# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Initial-state preparation: HF/bitstring prep, precedence, and composition."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from embasi_qiskit_integration.circuit_run.prep import (
    bitstring_prep_circuit,
    compose_full_circuit,
    hf_prep_circuit,
    resolve_initial_state,
)


def _x_qubits(circuit: QuantumCircuit) -> set[int]:
    """The qubit indices carrying an X gate."""
    return {
        instruction.qubits[0]._index
        for instruction in circuit.data
        if instruction.operation.name == "x"
    }


# ----- hf_prep_circuit ---------------------------------------------------- #


def test_hf_prep_blocked_alpha_beta_ordering():
    """Alpha block fills from 0, beta block from n_orbitals."""
    qc = hf_prep_circuit(num_qubits=16, n_orbitals=8, n_alpha=5, n_beta=5)
    assert _x_qubits(qc) == {0, 1, 2, 3, 4, 8, 9, 10, 11, 12}


def test_hf_prep_open_shell():
    """Unequal alpha/beta counts occupy their own blocks independently."""
    qc = hf_prep_circuit(num_qubits=8, n_orbitals=4, n_alpha=3, n_beta=1)
    assert _x_qubits(qc) == {0, 1, 2, 4}


def test_hf_prep_vacuum_has_no_gates():
    qc = hf_prep_circuit(num_qubits=4, n_orbitals=2, n_alpha=0, n_beta=0)
    assert _x_qubits(qc) == set()
    assert len(qc.data) == 0


def test_hf_prep_permutation_relabels_occupied():
    """A permutation maps each occupied mode through to its new qubit."""
    permutation = [3, 2, 1, 0]
    qc = hf_prep_circuit(num_qubits=4, n_orbitals=2, n_alpha=1, n_beta=1, permutation=permutation)
    # occupied modes are {0, 2} -> permuted to {3, 1}
    assert _x_qubits(qc) == {3, 1}


def test_hf_prep_rejects_width_mismatch():
    with pytest.raises(ValueError, match="expected 2\\*n_orbitals"):
        hf_prep_circuit(num_qubits=10, n_orbitals=8, n_alpha=5, n_beta=5)


def test_hf_prep_rejects_too_many_electrons():
    with pytest.raises(ValueError, match="do not fit"):
        hf_prep_circuit(num_qubits=4, n_orbitals=2, n_alpha=3, n_beta=1)


def test_hf_prep_rejects_bad_permutation():
    with pytest.raises(ValueError, match="rearrangement"):
        hf_prep_circuit(num_qubits=4, n_orbitals=2, n_alpha=1, n_beta=1, permutation=[0, 1, 2, 2])


# ----- bitstring_prep_circuit --------------------------------------------- #


def test_bitstring_prep_is_msb_left():
    """The last character is qubit 0 -- matching get_counts() output."""
    qc = bitstring_prep_circuit("1000")
    assert _x_qubits(qc) == {3}
    assert _x_qubits(bitstring_prep_circuit("0001")) == {0}


def test_bitstring_prep_roundtrips_hf_layout():
    """An HF bitstring rendered MSB-left prepares the same qubits as hf_prep."""
    hf = hf_prep_circuit(num_qubits=8, n_orbitals=4, n_alpha=2, n_beta=2)
    # occupied {0,1,4,5} -> MSB-left string of width 8
    bits = "".join("1" if i in {0, 1, 4, 5} else "0" for i in range(7, -1, -1))
    assert _x_qubits(bitstring_prep_circuit(bits)) == _x_qubits(hf)


def test_bitstring_prep_matches_hf_for_open_shell():
    """An asymmetric case pins the alpha/beta block order.

    With ``n_alpha == n_beta`` a swapped spin block is invisible, so this uses
    5 alpha / 3 beta: the rightmost ``n_orbitals`` characters must be the alpha
    block. Reading the string left-to-right instead puts the electrons in the
    wrong spin sector, which SQD then postselects away entirely.
    """
    n_orbitals, n_alpha, n_beta = 8, 5, 3
    hf = hf_prep_circuit(2 * n_orbitals, n_orbitals, n_alpha, n_beta)
    assert _x_qubits(hf) == {0, 1, 2, 3, 4, 8, 9, 10}

    # MSB-left: beta block leftmost, alpha block rightmost.
    bits = "0000011100011111"
    assert _x_qubits(bitstring_prep_circuit(bits)) == _x_qubits(hf)
    # The naive left-to-right reading is a different (wrong) state.
    assert _x_qubits(bitstring_prep_circuit("1111100011100000")) != _x_qubits(hf)


def test_bitstring_prep_ignores_spaces():
    assert _x_qubits(bitstring_prep_circuit("00 01")) == {0}


@pytest.mark.parametrize("bad", ["", "0102", "abc", "  "])
def test_bitstring_prep_rejects_invalid(bad):
    with pytest.raises(ValueError, match="only 0/1"):
        bitstring_prep_circuit(bad)


# ----- resolve_initial_state precedence ----------------------------------- #


def test_resolve_prefers_included_flag_over_everything():
    """A core that already holds its reference state gets an identity prep.

    Discarding an explicitly-requested determinant must warn: silently sampling a
    different state yields a plausible-looking energy with no indication why.
    """
    core = QuantumCircuit(4)
    core.metadata = {"initial_state_included": True}
    with pytest.warns(UserWarning, match="already includes its initial state"):
        prep = resolve_initial_state(
            num_qubits=4,
            initial_state_bitstring="1111",
            n_alpha=1,
            n_beta=1,
            n_orbitals=2,
            core=core,
        )
    assert len(prep.data) == 0
    assert prep.name == "initial_state_included"


def test_resolve_prefers_bitstring_over_hf():
    prep = resolve_initial_state(
        num_qubits=4, initial_state_bitstring="0011", n_alpha=1, n_beta=1, n_orbitals=2
    )
    assert _x_qubits(prep) == {0, 1}  # bitstring, not HF's {0, 2}


def test_resolve_falls_back_to_hf():
    prep = resolve_initial_state(num_qubits=4, n_alpha=1, n_beta=1, n_orbitals=2)
    assert _x_qubits(prep) == {0, 2}


def test_resolve_included_flag_false_does_not_short_circuit():
    """An explicit False must fall through to the normal precedence chain."""
    core = QuantumCircuit(4)
    core.metadata = {"initial_state_included": False}
    prep = resolve_initial_state(num_qubits=4, n_alpha=1, n_beta=1, n_orbitals=2, core=core)
    assert _x_qubits(prep) == {0, 2}


def test_resolve_rejects_bitstring_width_mismatch():
    with pytest.raises(ValueError, match="!= num_qubits"):
        resolve_initial_state(num_qubits=8, initial_state_bitstring="0011")


def test_resolve_raises_without_any_source():
    with pytest.raises(ValueError, match="cannot resolve an initial state"):
        resolve_initial_state(num_qubits=4)


def test_resolve_raises_on_partial_hf_spec():
    """Partially-specified HF counts are an error, not a silent vacuum."""
    with pytest.raises(ValueError, match="cannot resolve an initial state"):
        resolve_initial_state(num_qubits=4, n_alpha=1, n_orbitals=2)


# ----- compose_full_circuit ----------------------------------------------- #


def test_compose_applies_prep_before_core():
    prep = bitstring_prep_circuit("01")
    core = QuantumCircuit(2)
    core.h(1)
    full = compose_full_circuit(prep, core, measure=False)
    names = [instruction.operation.name for instruction in full.data]
    assert names == ["x", "h"]


def test_compose_adds_measure_all_by_default():
    full = compose_full_circuit(QuantumCircuit(2), QuantumCircuit(2))
    assert any(instruction.operation.name == "measure" for instruction in full.data)
    assert full.num_clbits == 2


def test_compose_can_skip_measurement():
    full = compose_full_circuit(QuantumCircuit(2), QuantumCircuit(2), measure=False)
    assert full.num_clbits == 0


def test_compose_rejects_width_mismatch():
    with pytest.raises(ValueError, match="must match"):
        compose_full_circuit(QuantumCircuit(2), QuantumCircuit(3))


def test_permutation_is_read_from_core_metadata():
    """A relabeled core must get its reference determinant in the permuted order.

    The generator records the applied permutation on the circuit, so a caller that
    does not know the core was optimized still gets a matching prep -- forgetting it
    would prepare the determinant on the wrong modes.
    """
    permutation = [3, 0, 5, 1, 4, 2]
    core = QuantumCircuit(6)
    core.metadata = {"initial_state_included": False, "permutation": permutation}

    from_metadata = resolve_initial_state(
        num_qubits=6, n_alpha=1, n_beta=1, n_orbitals=3, core=core
    )
    explicit = hf_prep_circuit(6, 3, 1, 1, permutation=permutation)
    unpermuted = hf_prep_circuit(6, 3, 1, 1)

    assert _x_qubits(from_metadata) == _x_qubits(explicit)
    # And it genuinely differs from the unpermuted prep, so the test has teeth.
    assert _x_qubits(from_metadata) != _x_qubits(unpermuted)


def test_explicit_permutation_overrides_core_metadata():
    """An explicitly passed permutation wins over the circuit's recorded one."""
    core = QuantumCircuit(6)
    core.metadata = {"permutation": [3, 0, 5, 1, 4, 2]}

    identity = resolve_initial_state(
        num_qubits=6,
        n_alpha=1,
        n_beta=1,
        n_orbitals=3,
        core=core,
        permutation=[0, 1, 2, 3, 4, 5],
    )

    assert _x_qubits(identity) == _x_qubits(hf_prep_circuit(6, 3, 1, 1))


def test_core_without_permutation_metadata_is_unpermuted():
    """An unrelabeled core (permutation None) must get the plain HF prep."""
    core = QuantumCircuit(6)
    core.metadata = {"initial_state_included": False, "permutation": None}

    resolved = resolve_initial_state(num_qubits=6, n_alpha=1, n_beta=1, n_orbitals=3, core=core)

    assert _x_qubits(resolved) == _x_qubits(hf_prep_circuit(6, 3, 1, 1))
