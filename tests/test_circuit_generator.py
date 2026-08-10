# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Ansatz circuits: HF reference (ffsim) and SqDRIFT (qiskit-fermions)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from qiskit import transpile
from qiskit_aer import AerSimulator

from embasi_qiskit_integration.circuit_generator.hf import build_hf_circuit
from embasi_qiskit_integration.circuit_generator.sqdrift import (
    build_sqdrift_circuits,
    sqdrift_available,
)
from embasi_qiskit_integration.circuit_run.aer import AerSampler
from embasi_qiskit_integration.hamiltonian import fcidump

requires_fermions = pytest.mark.skipif(
    not sqdrift_available(), reason="qiskit-fermions not installed (fermions extra)"
)


@pytest.fixture
def n2_ham(data_dir):
    return fcidump.read(data_dir / "n2_8o10e.fcidump")


def test_hf_circuit_qubit_count(n2_ham):
    qc = build_hf_circuit(n2_ham)
    assert qc.num_qubits == 2 * n2_ham.norb


def test_hf_circuit_bitstring(n2_ham):
    """The HF-only circuit sampled with Aer yields one deterministic bitstring."""
    qc = build_hf_circuit(n2_ham)
    tqc = transpile(qc, AerSimulator(), optimization_level=0)
    counts = AerSampler().sample(tqc, shots=2000, seed=1)

    assert len(counts) == 1
    bitstring = next(iter(counts))
    # blocked alpha|beta ordering: total occupation == n_alpha + n_beta
    assert bitstring.count("1") == sum(n2_ham.nelec)
    assert len(bitstring) == 2 * n2_ham.norb


@requires_fermions
def test_sqdrift_exact_one_circuit_per_time(n2_ham):
    """``exact`` yields one full-evolution circuit per requested time."""
    # Default time is the 3-element sweep [1.0, 2.0, 3.0].
    assert len(build_sqdrift_circuits(n2_ham, method="exact")) == 3

    single = build_sqdrift_circuits(n2_ham, method="exact", time=1.0)
    assert len(single) == 1
    assert single[0].num_qubits == 2 * n2_ham.norb

    assert len(build_sqdrift_circuits(n2_ham, method="exact", time=[0.5, 1.0])) == 2


@requires_fermions
def test_sqdrift_qdrift_sweeps_time_by_num_terms(n2_ham):
    """qDRIFT fans out over time x num_terms x randomizations."""
    circuits = build_sqdrift_circuits(
        n2_ham, method="qdrift", time=1.0, num_terms=10, num_randomizations=3, seed=42
    )
    assert len(circuits) == 3
    for qc in circuits:
        assert qc.num_qubits == 2 * n2_ham.norb

    # 2 times x 2 term-counts x 2 randomizations = 8
    swept = build_sqdrift_circuits(
        n2_ham,
        method="qdrift",
        time=[0.5, 1.0],
        num_terms=[8, 10],
        num_randomizations=2,
        seed=42,
    )
    assert len(swept) == 8


@requires_fermions
def test_sqdrift_time_accepts_scalar_or_sequence(n2_ham):
    """A scalar time and a length-1 sequence must build the same circuit."""
    scalar = build_sqdrift_circuits(n2_ham, method="exact", time=1.0, seed=42)
    listed = build_sqdrift_circuits(n2_ham, method="exact", time=[1.0], seed=42)
    assert _op_signature(scalar[0]) == _op_signature(listed[0])


@requires_fermions
def test_sqdrift_rejects_empty_sweep_axes(n2_ham):
    with pytest.raises(ValueError, match="at least one evolution time"):
        build_sqdrift_circuits(n2_ham, method="exact", time=[])
    with pytest.raises(ValueError, match="at least one term count"):
        build_sqdrift_circuits(n2_ham, method="qdrift", time=1.0, num_terms=[])


@requires_fermions
def test_sqdrift_exact_spans_ci_space(n2_ham):
    """The exact-evolution ansatz samples many determinants at the right
    Hamming weights (this is what feeds SQD)."""
    circuits = build_sqdrift_circuits(n2_ham, method="exact", time=1.0)
    tqc = transpile(circuits[0], AerSimulator(), optimization_level=0)
    counts = AerSampler().sample(tqc, shots=20_000, seed=1)
    norb = n2_ham.norb
    na, nb = n2_ham.nelec
    # many distinct configurations, each with the correct alpha/beta occupation
    assert len(counts) > 20
    good = sum(1 for b in counts if b[norb:].count("1") == na and b[:norb].count("1") == nb)
    assert good == len(counts)
    assert all(len(b) == 2 * norb for b in counts)


@requires_fermions
def test_sqdrift_qdrift_randomizations_differ(n2_ham):
    """Per-randomization seeding gives distinct draws, not one repeated circuit."""
    circuits = build_sqdrift_circuits(
        n2_ham, method="qdrift", num_terms=10, num_randomizations=3, seed=42
    )
    ops = [_op_signature(qc) for qc in circuits]
    assert len(set(ops)) > 1, "all randomizations identical; per-draw seeding is not in effect"


@requires_fermions
def test_sqdrift_qdrift_reproducible_in_process(n2_ham):
    """The same seed rebuilds byte-identical qDRIFT circuits within one process."""
    kwargs = {"method": "qdrift", "num_terms": 10, "num_randomizations": 3, "seed": 42}
    first = build_sqdrift_circuits(n2_ham, **kwargs)
    second = build_sqdrift_circuits(n2_ham, **kwargs)
    assert [_op_signature(qc) for qc in first] == [_op_signature(qc) for qc in second]


@requires_fermions
def test_sqdrift_qdrift_reproducible_across_processes(n2_ham, data_dir):
    """The same seed rebuilds identical qDRIFT circuits in a *separate* process.

    This is what the canonical group ordering in ``circuit_generator.operator``
    exists for: ``group_terms_by_electronic_structure`` labels groups in a
    process-dependent order, so without canonicalization a seeded qDRIFT draw
    maps to a different physical group in a fresh interpreter and this assertion
    fails while the in-process test above still passes.
    """
    script = (
        "import json, sys\n"
        "from embasi_qiskit_integration.hamiltonian import fcidump\n"
        "from embasi_qiskit_integration.circuit_generator.sqdrift import build_sqdrift_circuits\n"
        "ham = fcidump.read(sys.argv[1])\n"
        "cs = build_sqdrift_circuits(ham, method='qdrift', num_terms=10,\n"
        "                            num_randomizations=3, seed=42)\n"
        "print(json.dumps([[(i.operation.name, tuple(c._index for c in i.qubits))\n"
        "                   for i in qc.data] for qc in cs]))\n"
    )
    fcidump_path = str(data_dir / "n2_8o10e.fcidump")
    runs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script, fcidump_path],
            capture_output=True,
            text=True,
            check=True,
        )
        # fcidump.read() prints a "Parsing <path>" banner, so take only the
        # last non-empty line -- the JSON payload the script emits.
        payload = [line for line in proc.stdout.splitlines() if line.strip()][-1]
        runs.append(json.loads(payload))

    assert runs[0] == runs[1], "seeded qDRIFT circuits differ between processes"

    # And the subprocess result must match *this* process, not merely itself.
    # JSON has no tuples, so normalise both sides to nested lists before comparing.
    def _as_json_shape(circuits) -> list:
        return json.loads(json.dumps([_op_signature(qc) for qc in circuits]))

    in_process = _as_json_shape(
        build_sqdrift_circuits(n2_ham, method="qdrift", num_terms=10, num_randomizations=3, seed=42)
    )
    assert runs[0] == in_process


@requires_fermions
def test_canonicalize_handles_non_contiguous_group_labels():
    """Group ranking must survive a label that no term carries.

    ``num_groups()`` is the largest label + 1, so a gap in the labels makes it
    disagree with the count of labels actually present. Reading the weights from
    upstream's ``group_weights()`` (which reports 0.0 for an unused label) keeps
    this working; recomputing them with ``np.add.at`` + ``np.unique`` raised
    ``ValueError`` on the length mismatch.
    """
    from qiskit_fermions.operators import FermionOperator

    from embasi_qiskit_integration.circuit_generator.operator import _canonicalize_group_order

    # Labels 0 and 5 used; 1-4 carry no term.
    operator = FermionOperator.from_terms_with_groups(
        [
            ([(True, 0), (False, 0)], 1.0, 0),
            ([(True, 1), (False, 1)], 2.0, 5),
            ([(True, 2), (False, 2)], 3.0, 0),
        ]
    )
    relabeled = _canonicalize_group_order(operator)
    # Every term is preserved, and the two occupied groups stay distinct.
    assert len(list(relabeled.iter_terms())) == 3
    assert len(set(relabeled.groups)) == 2


@requires_fermions
def test_filter_trivial_is_separate_from_filter_diagonal_terms(n2_ham):
    """The two filters are distinct mechanisms, not two names for one thing.

    ``filter_diagonal_terms`` prunes the *operator* before grouping (needed for
    reproducible seeded draws); ``filter_trivial`` is a pass option rejecting
    individual *draws* that cannot change the occupation. With the operator
    already pruned there is nothing left for the pass to reject, so it is a no-op
    -- but with pruning off it visibly changes the circuit, which is what proves
    the option reaches the pass.
    """
    kwargs = {"method": "qdrift", "num_terms": 10, "num_randomizations": 1, "seed": 42}

    pruned_on = _op_signature(
        build_sqdrift_circuits(n2_ham, **kwargs, filter_diagonal_terms=True, filter_trivial=True)[0]
    )
    pruned_off = _op_signature(
        build_sqdrift_circuits(n2_ham, **kwargs, filter_diagonal_terms=True, filter_trivial=False)[
            0
        ]
    )
    assert pruned_on == pruned_off

    unpruned_on = _op_signature(
        build_sqdrift_circuits(n2_ham, **kwargs, filter_diagonal_terms=False, filter_trivial=True)[
            0
        ]
    )
    unpruned_off = _op_signature(
        build_sqdrift_circuits(n2_ham, **kwargs, filter_diagonal_terms=False, filter_trivial=False)[
            0
        ]
    )
    assert unpruned_on != unpruned_off, "filter_trivial is not reaching QDriftTrotterization"


@requires_fermions
def test_filter_trivial_defaults_to_where_occupation_is_visible(n2_ham):
    """The pass can only filter draws when it can see the occupation.

    ``filter_trivial`` compares a sampled term against the reference occupation,
    which it reads from the ``InitializeModes`` gate. On a bare circuit there is
    no such gate, so forcing it on is inert *and* makes qiskit warn -- hence the
    default tracks ``include_initial_state`` and neither path warns.
    """
    import warnings

    for include in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            build_sqdrift_circuits(
                n2_ham,
                method="qdrift",
                num_terms=10,
                num_randomizations=1,
                seed=42,
                include_initial_state=include,
            )


@requires_fermions
def test_sqdrift_records_initial_state_flag(n2_ham):
    """The metadata flag tells the run stage whether a prep is still needed."""
    baked = build_sqdrift_circuits(n2_ham, method="exact")
    bare = build_sqdrift_circuits(n2_ham, method="exact", include_initial_state=False)
    assert baked[0].metadata["initial_state_included"] is True
    assert bare[0].metadata["initial_state_included"] is False


@requires_fermions
def test_bare_core_plus_hf_prep_equals_initialize_modes(n2_ham):
    """Both initial-state routes sample the identical distribution.

    ``InitializeModes`` is synthesised by the Jordan-Wigner preset into X gates on
    the occupied modes -- exactly what ``hf_prep_circuit`` applies. This pins that
    equivalence, which is what makes it safe for the generator to emit bare cores
    and let the run stage choose the reference state.
    """
    from embasi_qiskit_integration.circuit_run.prep import (
        compose_full_circuit,
        resolve_initial_state,
    )

    baked = build_sqdrift_circuits(n2_ham, method="exact", time=1.0, seed=42)[0]
    core = build_sqdrift_circuits(
        n2_ham, method="exact", time=1.0, seed=42, measure=False, include_initial_state=False
    )[0]
    na, nb = n2_ham.nelec
    prep = resolve_initial_state(
        num_qubits=core.num_qubits, n_alpha=na, n_beta=nb, n_orbitals=n2_ham.norb
    )
    composed = compose_full_circuit(prep, core)

    sampler = AerSampler()
    baked_counts = sampler.sample(
        transpile(baked, AerSimulator(), optimization_level=0), shots=20_000, seed=11
    )
    composed_counts = sampler.sample(
        transpile(composed, AerSimulator(), optimization_level=0), shots=20_000, seed=11
    )
    assert baked_counts == composed_counts


def _op_signature(circuit) -> tuple:
    """A hashable (gate name, qubit indices) signature of a circuit's body."""
    return tuple(
        (instruction.operation.name, tuple(q._index for q in instruction.qubits))
        for instruction in circuit.data
    )
