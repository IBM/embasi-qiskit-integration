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
    relabel_available,
    sqdrift_available,
)
from embasi_qiskit_integration.circuit_run.aer import AerSampler
from embasi_qiskit_integration.circuit_run.prep import (
    compose_full_circuit,
    resolve_initial_state,
)
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
    assert len(build_sqdrift_circuits(n2_ham, method="exact", optimize=False)) == 3

    single = build_sqdrift_circuits(n2_ham, method="exact", time=1.0, optimize=False)
    assert len(single) == 1
    assert single[0].num_qubits == 2 * n2_ham.norb

    assert len(build_sqdrift_circuits(n2_ham, method="exact", time=[0.5, 1.0], optimize=False)) == 2


@requires_fermions
def test_sqdrift_qdrift_sweeps_time_by_num_groups(n2_ham):
    """qDRIFT fans out over time x num_groups x randomizations."""
    circuits = build_sqdrift_circuits(
        n2_ham,
        method="qdrift",
        time=1.0,
        num_groups=10,
        num_randomizations=3,
        seed=42,
        optimize=False,
    )
    assert len(circuits) == 3
    for qc in circuits:
        assert qc.num_qubits == 2 * n2_ham.norb

    # 2 times x 2 term-counts x 2 randomizations = 8
    swept = build_sqdrift_circuits(
        n2_ham,
        method="qdrift",
        time=[0.5, 1.0],
        num_groups=[8, 10],
        num_randomizations=2,
        seed=42,
        optimize=False,
    )
    assert len(swept) == 8


@requires_fermions
def test_sqdrift_qdrift_matches_reference_construction(n2_ham):
    """The qDRIFT path reproduces the canonical SqDRIFT recipe gate-for-gate.

    The reference builds, per randomization: a fresh ``FermionicCircuit`` holding
    only ``Evolution`` (vacuum -- no ``InitializeModes``), run through a fresh
    ``generate_preset_jw_pass_manager(seed_transpiler=rng_seed + i)`` whose
    optimization stage is ``QDriftTrotterization(num_groups, rng=rng_seed + i)``,
    with ``measure_all(inplace=False)`` appended afterwards.

    Transcribed inline here and compared including rotation parameters, so a
    divergence in construction order, seeding, or the operator would fail this.

    Pinned with ``optimize=False`` because that is the recipe transcribed above:
    mode relabeling deliberately *changes* the circuit (it is what shortens it), so
    the gate-for-gate identity only holds without it.
    ``test_optimize_produces_permutations_and_shortens_circuits`` covers the
    relabeled branch.
    """
    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import Evolution
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.passes import QDriftTrotterization
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    from embasi_qiskit_integration.circuit_generator.operator import build_canonical_operator

    time, num_groups, n_rand, seed = 1.0, 10, 3, 42

    normal, num_modes = build_canonical_operator(n2_ham, atol=1e-16, filter_diagonal_terms=True)

    def reference_one(draw_seed: int):
        pm = generate_preset_jw_pass_manager(seed_transpiler=draw_seed)
        pm.optimization = FermionicPassManager([QDriftTrotterization(num_groups, rng=draw_seed)])
        circ = FermionicCircuit(num_modes)
        circ.append(Evolution(num_modes, normal, time), circ.modes)
        return pm.run(circ)

    reference = [reference_one(seed + i).measure_all(inplace=False) for i in range(n_rand)]
    ours = build_sqdrift_circuits(
        n2_ham,
        method="qdrift",
        time=time,
        num_groups=num_groups,
        num_randomizations=n_rand,
        seed=seed,
        include_initial_state=False,
        optimize=False,
    )

    def detailed(circuit) -> list:
        """Gate name, qubit indices, and numeric parameters for every instruction."""
        return [
            (
                instruction.operation.name,
                tuple(q._index for q in instruction.qubits),
                tuple(
                    float(p) for p in instruction.operation.params if isinstance(p, (int, float))
                ),
            )
            for instruction in circuit.data
        ]

    assert len(ours) == len(reference) == n_rand
    for mine, theirs in zip(ours, reference, strict=True):
        assert detailed(mine) == detailed(theirs)


@requires_fermions
def test_sqdrift_seeds_restart_per_sweep_combination(n2_ham):
    """Randomization seeds restart at ``seed`` for every (time, num_groups) combo.

    Each combination is an independent pass over ``range(num_randomizations)``, so
    combinations sharing a randomization index share a qDRIFT draw -- which isolates
    the effect of time / num_groups from sampling noise. A flattened per-circuit
    index would instead make every draw unique.
    """
    circuits = build_sqdrift_circuits(
        n2_ham,
        method="qdrift",
        time=[1.0, 2.0],
        num_groups=10,
        num_randomizations=2,
        seed=42,
        include_initial_state=False,
        optimize=False,
    )
    assert len(circuits) == 4
    # index 0/1 = time 1.0, index 2/3 = time 2.0
    assert _op_signature(circuits[0]) == _op_signature(circuits[2])  # both randomization 0
    assert _op_signature(circuits[1]) == _op_signature(circuits[3])  # both randomization 1
    assert _op_signature(circuits[0]) != _op_signature(circuits[1])  # different draws


@requires_fermions
def test_sqdrift_time_accepts_scalar_or_sequence(n2_ham):
    """A scalar time and a length-1 sequence must build the same circuit."""
    scalar = build_sqdrift_circuits(n2_ham, method="exact", time=1.0, seed=42, optimize=False)
    listed = build_sqdrift_circuits(n2_ham, method="exact", time=[1.0], seed=42, optimize=False)
    assert _op_signature(scalar[0]) == _op_signature(listed[0])


@requires_fermions
def test_sqdrift_rejects_empty_sweep_axes(n2_ham):
    with pytest.raises(ValueError, match="at least one evolution time"):
        build_sqdrift_circuits(n2_ham, method="exact", time=[])
    with pytest.raises(ValueError, match="at least one group count"):
        build_sqdrift_circuits(n2_ham, method="qdrift", time=1.0, num_groups=[])


@requires_fermions
def test_sqdrift_exact_spans_ci_space(n2_ham):
    """The exact-evolution ansatz samples many determinants at the right
    Hamming weights (this is what feeds SQD)."""
    circuits = build_sqdrift_circuits(n2_ham, method="exact", time=1.0, optimize=False)
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
        n2_ham,
        method="qdrift",
        num_groups=10,
        num_randomizations=3,
        seed=42,
        optimize=False,
    )
    ops = [_op_signature(qc) for qc in circuits]
    assert len(set(ops)) > 1, "all randomizations identical; per-draw seeding is not in effect"


@requires_fermions
def test_sqdrift_qdrift_reproducible_in_process(n2_ham):
    """The same seed rebuilds byte-identical qDRIFT circuits within one process."""
    kwargs = {
        "method": "qdrift",
        "num_groups": 10,
        "num_randomizations": 3,
        "seed": 42,
        "optimize": False,
    }
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
        "cs = build_sqdrift_circuits(ham, method='qdrift', num_groups=10,\n"
        "                            num_randomizations=3, seed=42, optimize=False)\n"
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
        build_sqdrift_circuits(
            n2_ham, method="qdrift", num_groups=10, num_randomizations=3, seed=42, optimize=False
        )
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
    # optimize=False: this test compares gate signatures, and mode relabeling would
    # change them for reasons unrelated to the two filters under test.
    kwargs = {
        "method": "qdrift",
        "num_groups": 10,
        "num_randomizations": 1,
        "seed": 42,
        "optimize": False,
    }

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
                num_groups=10,
                num_randomizations=1,
                seed=42,
                include_initial_state=include,
            )


@requires_fermions
def test_sqdrift_records_initial_state_flag(n2_ham):
    """The metadata flag tells the run stage whether a prep is still needed."""
    baked = build_sqdrift_circuits(n2_ham, method="exact", optimize=False)
    bare = build_sqdrift_circuits(
        n2_ham, method="exact", include_initial_state=False, optimize=False
    )
    assert baked[0].metadata["initial_state_included"] is True
    assert bare[0].metadata["initial_state_included"] is False


@requires_fermions
def test_bare_core_plus_hf_prep_equals_initialize_modes(n2_ham):
    """Both initial-state routes sample the identical distribution.

    ``InitializeModes`` is synthesised by the Jordan-Wigner preset into X gates on
    the occupied modes -- exactly what ``hf_prep_circuit`` applies. This pins that
    equivalence, which is what makes it safe for the generator to emit bare cores
    and let the run stage choose the reference state.

    ``optimize=False`` because the two builds would otherwise be relabeled
    *independently* -- ``filter_trivial`` tracks ``include_initial_state``, so the
    baked and bare variants sample different draws and get different permutations,
    which has nothing to do with the equivalence under test. The relabeled
    equivalent of this property is
    ``test_relabeled_circuits_are_only_usable_after_undoing_the_permutation``.
    """
    from embasi_qiskit_integration.circuit_run.prep import (
        compose_full_circuit,
        resolve_initial_state,
    )

    baked = build_sqdrift_circuits(n2_ham, method="exact", time=1.0, seed=42, optimize=False)[0]
    core = build_sqdrift_circuits(
        n2_ham,
        method="exact",
        time=1.0,
        seed=42,
        measure=False,
        include_initial_state=False,
        optimize=False,
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


# ----- mode relabeling (optimize=True) and parallel generation --------------- #

requires_relabel = pytest.mark.skipif(
    not relabel_available(), reason="relabel solver stack not installed (relabel extra)"
)


@requires_fermions
@requires_relabel
def test_optimize_produces_permutations_and_shortens_circuits(n2_ham):
    """``optimize=True`` relabels the modes, and that is the point: shorter circuits.

    ``RelabelModes`` reorders the fermionic modes to minimize the span of the
    sampled excitations, which shortens the synthesised circuit. Both halves are
    asserted -- a permutation is reported per circuit, and the depth actually drops
    against the same seeded draws built without relabeling.
    """
    kwargs = {
        "method": "qdrift",
        "time": 1.0,
        "num_groups": 10,
        "num_randomizations": 3,
        "seed": 42,
        "include_initial_state": False,
    }
    optimized = build_sqdrift_circuits(n2_ham, **kwargs, optimize=True)
    plain = build_sqdrift_circuits(n2_ham, **kwargs, optimize=False)

    assert optimized.num_permutations_found == 3
    assert optimized.any_permuted
    assert plain.num_permutations_found == 0
    assert plain.permutations == [None, None, None]

    num_modes = 2 * n2_ham.norb
    for permutation in optimized.permutations:
        assert sorted(permutation) == list(range(num_modes))

    # The whole reason to relabel: the ensemble gets cheaper to run.
    assert sum(qc.depth() for qc in optimized) < sum(qc.depth() for qc in plain)


@requires_fermions
@requires_relabel
def test_optimize_permutation_is_reproducible(n2_ham):
    """The same seeded draw must yield the same permutation, every time.

    This does not hold for the raw solver output: the excitation-span MILP is
    degenerate and HiGHS is not a pure function of the model (solving one model
    repeatedly in a single process returned one optimum twice, then a different
    one). The permutation has to be undone on the sampled counts, so an unstable
    one would make a seeded run irreproducible -- hence
    ``canonicalize_permutation``, which this pins.
    """
    kwargs = {
        "method": "qdrift",
        "time": 1.0,
        "num_groups": 10,
        "num_randomizations": 2,
        "seed": 42,
        "include_initial_state": False,
    }
    runs = [build_sqdrift_circuits(n2_ham, **kwargs).permutations for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


@requires_fermions
@requires_relabel
def test_canonicalize_permutation_is_independent_of_the_solver_pick():
    """Canonicalization must not depend on which optimum the solver happened to return.

    Two solver picks observed for one real draw refined to different objective
    values, so a search seeded from the pick is start-dependent. Canonicalizing from
    a fixed candidate set removes that: wildly different inputs must all produce the
    same permutation for the same excitation set.
    """
    from embasi_qiskit_integration.circuit_generator.sqdrift import (
        _excitation_spans,
        canonicalize_permutation,
    )

    excitations = [(0, 3), (1, 4), (2, 5), (0, 1, 4, 5), (2, 3, 4, 5)]
    starts = [
        [0, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 0],
        [3, 0, 5, 1, 4, 2],
        [1, 3, 5, 0, 2, 4],
    ]
    canonical = [canonicalize_permutation(start, excitations) for start in starts]
    assert len({tuple(p) for p in canonical}) == 1

    # And it is a real improvement, not just a stable arbitrary answer.
    result = canonical[0]
    assert sorted(result) == list(range(6))
    assert _excitation_spans(result, excitations) <= _excitation_spans(starts[0], excitations)


@requires_fermions
@requires_relabel
def test_permutation_is_mirrored_into_circuit_metadata(n2_ham):
    """The run stage reads the permutation off the circuit, so it must be there."""
    result = build_sqdrift_circuits(
        n2_ham,
        method="qdrift",
        time=1.0,
        num_groups=10,
        num_randomizations=2,
        seed=42,
        include_initial_state=False,
    )
    for circuit, permutation in zip(result.circuits, result.permutations, strict=True):
        assert circuit.metadata["permutation"] == permutation
    # The pyomo solver-results object must not survive: it is unserializable and
    # would have to cross a process boundary on the sharded path.
    assert "permutation.opt_result" not in result.circuits[0].metadata


@requires_fermions
def test_optimize_false_records_no_permutation(n2_ham):
    """Without relabeling the metadata key is present but null, not missing."""
    result = build_sqdrift_circuits(
        n2_ham, method="exact", time=1.0, seed=42, optimize=False, include_initial_state=False
    )
    assert result.circuits[0].metadata["permutation"] is None


@requires_fermions
def test_result_behaves_like_the_circuit_list(n2_ham):
    """Existing list-style callers keep working after the result-object change."""
    result = build_sqdrift_circuits(
        n2_ham, method="exact", time=[1.0, 2.0], seed=42, optimize=False
    )
    assert len(result) == 2
    assert len(list(result)) == 2
    assert result[0] is result.circuits[0]
    assert result[:1] == result.circuits[:1]


@requires_fermions
def test_workers_produce_identical_circuits_to_sequential(n2_ham):
    """Sharding across processes must be an optimization, not a change of output.

    Each randomization is an independently seeded draw, so a contiguous slice built
    in a worker must equal the same slice built in-process -- including its
    permutation, since that is applied to the sampled counts. Uses ``optimize=False``
    to keep the test quick; the relabel path is covered by the reproducibility test
    above.
    """
    kwargs = {
        "method": "qdrift",
        "time": 1.0,
        "num_groups": 10,
        "num_randomizations": 4,
        "seed": 42,
        "include_initial_state": False,
        "optimize": False,
    }
    sequential = build_sqdrift_circuits(n2_ham, **kwargs, workers=1)
    sharded = build_sqdrift_circuits(n2_ham, **kwargs, workers=2)

    assert [_op_signature(qc) for qc in sequential] == [_op_signature(qc) for qc in sharded]
    assert sequential.permutations == sharded.permutations
    assert sequential.randomization_indices == sharded.randomization_indices


@requires_fermions
def test_workers_preserve_sweep_order(n2_ham):
    """Chunks complete out of order; the result must still be the sequential order."""
    kwargs = {
        "method": "qdrift",
        "time": [1.0, 2.0],
        "num_groups": 10,
        "num_randomizations": 2,
        "seed": 42,
        "include_initial_state": False,
        "optimize": False,
    }
    sequential = build_sqdrift_circuits(n2_ham, **kwargs, workers=1)
    sharded = build_sqdrift_circuits(n2_ham, **kwargs, workers=4)
    assert sequential.randomization_indices == sharded.randomization_indices == [0, 1, 0, 1]
    assert [_op_signature(qc) for qc in sequential] == [_op_signature(qc) for qc in sharded]


@requires_fermions
def test_operator_cache_serves_repeated_builds(n2_ham):
    """The operator is built once per (ham, atol, filter) and reused.

    This is what makes the per-randomization fan-out affordable: a worker pays the
    expensive construction once, not once per draw.
    """
    from embasi_qiskit_integration.circuit_generator import operator as operator_module

    operator_module._OPERATOR_CACHE.clear()
    first, num_modes = operator_module.build_canonical_operator(n2_ham)
    assert len(operator_module._OPERATOR_CACHE) == 1

    second, _ = operator_module.build_canonical_operator(n2_ham)
    assert second is first  # served from cache, not rebuilt

    # A different option is a different key, not a cache hit.
    operator_module.build_canonical_operator(n2_ham, filter_diagonal_terms=False)
    assert len(operator_module._OPERATOR_CACHE) == 2

    bypassed, _ = operator_module.build_canonical_operator(n2_ham, use_cache=False)
    assert bypassed is not first


@requires_fermions
def test_optimize_requires_the_relabel_extra(n2_ham, monkeypatch):
    """Asking for relabeling without the solver stack must raise, not silently skip.

    ``RelabelModes`` only warns when it has no solver, so a missing extra would
    otherwise look like "optimization ran and found nothing" -- and the circuits
    would be unpermuted while the caller believed they were optimized.
    """
    from embasi_qiskit_integration.circuit_generator import sqdrift as sqdrift_module

    monkeypatch.setattr(sqdrift_module, "relabel_available", lambda: False)
    with pytest.raises(ImportError, match="pyomo"):
        build_sqdrift_circuits(n2_ham, method="exact", time=1.0, optimize=True)


@requires_fermions
@requires_relabel
def test_relabeled_circuits_are_only_usable_after_undoing_the_permutation(n2_ham):
    """A relabeled circuit's raw counts are wrong; un-permuted they match the reference.

    This is the correctness property that makes ``optimize=True`` usable at all.
    ``RelabelModes`` reorders the modes, so the circuit samples bitstrings in the
    *permuted* order: fed to SQD as-is, the occupations land on the wrong orbitals.

    Both directions are asserted against an unpermuted reference distribution:

    - raw relabeled counts are essentially disjoint from it (total variation ~1);
    - after ``unpermute_counts`` they agree to sampling noise, and the electron
      counts per spin block are restored.
    """
    from embasi_qiskit_integration.circuit_run.permutation import unpermute_counts

    kwargs = {
        "method": "qdrift",
        "time": 1.0,
        "num_groups": 10,
        "num_randomizations": 1,
        "seed": 42,
        "measure": False,
        "include_initial_state": False,
    }
    relabeled = build_sqdrift_circuits(n2_ham, **kwargs, optimize=True)
    reference = build_sqdrift_circuits(n2_ham, **kwargs, optimize=False)
    permutation = relabeled.permutations[0]
    assert permutation is not None, "expected a permutation for this draw"

    na, nb = n2_ham.nelec
    norb = n2_ham.norb
    sampler = AerSampler()

    def sample(core) -> dict:
        # resolve_initial_state reads the permutation off the core's metadata, so
        # the reference determinant is prepared in that circuit's own mode order.
        prep = resolve_initial_state(
            num_qubits=core.num_qubits,
            n_alpha=na,
            n_beta=nb,
            n_orbitals=norb,
            core=core,
        )
        full = compose_full_circuit(prep, core)
        return sampler.sample(
            transpile(full, AerSimulator(), optimization_level=0), shots=40_000, seed=7
        )

    reference_counts = sample(reference[0])
    raw_counts = sample(relabeled[0])
    restored_counts = unpermute_counts(raw_counts, permutation)

    def total_variation(left: dict, right: dict) -> float:
        left_total = sum(left.values()) or 1
        right_total = sum(right.values()) or 1
        return 0.5 * sum(
            abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
            for key in set(left) | set(right)
        )

    # Raw relabeled counts are in the wrong basis: nearly disjoint from the truth.
    assert total_variation(reference_counts, raw_counts) > 0.9
    # Undoing the permutation recovers the reference distribution.
    assert total_variation(reference_counts, restored_counts) < 0.1

    # The restored bitstrings sit in the correct alpha/beta occupation sector.
    def in_sector(counts: dict) -> float:
        total = sum(counts.values()) or 1
        good = sum(
            count
            for bitstring, count in counts.items()
            if bitstring[norb:].count("1") == na and bitstring[:norb].count("1") == nb
        )
        return good / total

    assert in_sector(restored_counts) == pytest.approx(1.0)

    # Note the sector check alone would NOT catch the bug: relabeling permutes
    # modes across the two spin blocks but this draw keeps most weight in the
    # (na, nb) sector anyway, so raw counts look ~95% "valid" while being the wrong
    # determinants entirely. Only the distribution comparison above exposes it --
    # which is why the un-permutation has to be structural rather than validated.
    assert in_sector(raw_counts) > 0.5


def test_resolve_workers_treats_zero_as_one_per_cpu():
    """``workers=0`` means "use every CPU", and must not collapse to sequential.

    Regression: normalising with ``int(workers or 1)`` mapped ``0`` to ``1`` before
    the "<= 0 means auto" branch could see it, so ``workers=0`` silently ran
    single-process while the docs promised one worker per CPU.
    """
    import os

    from embasi_qiskit_integration.circuit_generator.sqdrift import _resolve_workers

    cpus = os.cpu_count() or 1
    assert _resolve_workers(0) == cpus
    assert _resolve_workers(-1) == cpus
    assert _resolve_workers(1) == 1
    assert _resolve_workers(4) == 4
    # An unset value stays sequential rather than saturating the machine.
    assert _resolve_workers(None) == 1


def test_plan_tasks_covers_every_draw_exactly_once():
    """Sharding must partition the sweep: no dropped and no duplicated draws."""
    from collections import Counter

    from embasi_qiskit_integration.circuit_generator.sqdrift import _plan_tasks

    tasks = _plan_tasks([1.0, 2.0], [10, 15], 5, 2, 42)
    seen: Counter = Counter()
    for combos, rand_range in tasks:
        for combo in combos:
            for index in rand_range:
                seen[(combo, index)] += 1

    assert set(seen.values()) == {1}
    assert len(seen) == 2 * 2 * 5  # times x num_groups x randomizations

    # An unseeded build is not addressable by index, so it stays a single task.
    assert len(_plan_tasks([1.0, 2.0], [10], 4, 8, None)) == 1
    # More workers than draws must not create empty chunks.
    assert [list(r) for _, r in _plan_tasks([1.0], [10], 1, 8, 42)] == [[0]]
    with pytest.raises(ValueError, match="num_randomizations must be positive"):
        _plan_tasks([1.0], [10], 0, 1, 42)


# ----- logging: the relabel outcome must be observable ----------------------- #


@requires_fermions
@requires_relabel
def test_logs_how_many_circuits_got_a_permutation(n2_ham, caplog):
    """A successful relabel run reports its hit rate at INFO."""
    import logging

    with caplog.at_level(
        logging.INFO, logger="embasi_qiskit_integration.circuit_generator.sqdrift"
    ):
        build_sqdrift_circuits(
            n2_ham,
            method="qdrift",
            time=1.0,
            num_groups=10,
            num_randomizations=2,
            seed=42,
            include_initial_state=False,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("2/2 circuits" in message for message in messages), messages
    # The sweep plan is reported before the work starts.
    assert any("qdrift sweep" in message for message in messages), messages


@requires_fermions
@requires_relabel
def test_warns_when_relabeling_finds_nothing(n2_ham, caplog):
    """Zero permutations must WARN, naming the knob that most likely caused it.

    ``RelabelModes`` degrades silently -- an infeasible or timed-out solve returns
    the circuit unchanged -- so ``optimize=True`` yielding no permutations is
    otherwise indistinguishable from relabeling having worked. An unusably small
    ``time_limit`` forces exactly that state.
    """
    import logging

    with caplog.at_level(
        logging.WARNING, logger="embasi_qiskit_integration.circuit_generator.sqdrift"
    ):
        result = build_sqdrift_circuits(
            n2_ham,
            method="qdrift",
            time=1.0,
            num_groups=10,
            num_randomizations=2,
            seed=42,
            include_initial_state=False,
            time_limit=1e-9,
        )

    assert result.num_permutations_found == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("0/2 circuits" in message for message in warnings), warnings
    assert any("time_limit" in message for message in warnings), warnings


@requires_fermions
def test_no_relabel_logging_when_optimization_is_off(n2_ham, caplog):
    """``optimize=False`` must not report a relabel outcome it never attempted."""
    import logging

    with caplog.at_level(
        logging.INFO, logger="embasi_qiskit_integration.circuit_generator.sqdrift"
    ):
        build_sqdrift_circuits(n2_ham, method="exact", time=1.0, optimize=False)

    messages = [record.getMessage() for record in caplog.records]
    assert not any("optimize:" in message for message in messages), messages
    # The build itself is still announced.
    assert any("exact" in message for message in messages), messages


def test_solver_report_is_not_logged_at_info():
    """HiGHS's own solver report must not drown our records at INFO.

    pyomo pipes the solver banner / presolve / B&B table into a logger at
    ``config.log_level``, which defaults to INFO -- one report per solve, so a
    4500-circuit sweep would emit thousands of lines as soon as a host enables INFO
    logging. The adapter demotes it to DEBUG.
    """
    import logging

    pytest.importorskip("pyomo")
    from embasi_qiskit_integration.circuit_generator.relabel import (
        NativeHighsSolverAdapter,
        relabel_available,
    )

    if not relabel_available():
        pytest.skip("relabel solver stack not installed (relabel extra)")

    from pyomo.environ import Binary, ConcreteModel, Objective, Var, minimize

    model = ConcreteModel()
    model.x = Var(domain=Binary)
    model.obj = Objective(expr=model.x, sense=minimize)

    adapter = NativeHighsSolverAdapter()
    logger_name = "pyomo.contrib.appsi.solvers.highs"
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    solver_logger = logging.getLogger(logger_name)
    solver_logger.addHandler(handler)
    previous_level = solver_logger.level
    solver_logger.setLevel(logging.INFO)
    try:
        adapter.solve(model)
    finally:
        solver_logger.removeHandler(handler)
        solver_logger.setLevel(previous_level)

    at_info = [r.getMessage() for r in records if r.levelno >= logging.INFO]
    assert not at_info, f"HiGHS report leaked at INFO: {at_info[:3]}"
