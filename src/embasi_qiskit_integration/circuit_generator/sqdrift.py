# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SqDRIFT ansatz circuits via ``qiskit-fermions`` (primary ansatz)."""

from __future__ import annotations

from typing import Any

from embasi_qiskit_integration.circuit_generator.operator import (
    build_canonical_operator,
    fermions_available,
)
from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def sqdrift_available() -> bool:
    """True if ``qiskit-fermions`` is importable in this environment."""
    return fermions_available()


def _hf_occupation(norb: int, nelec: tuple[int, int]) -> list[bool]:
    """HF occupation over the ``2*norb`` modes (alpha block then beta block)."""
    na, nb = nelec
    occ = [False] * (2 * norb)
    for i in range(na):
        occ[i] = True
    for i in range(nb):
        occ[norb + i] = True
    return occ


def build_sqdrift_circuits(
    ham: EmbeddedHamiltonian,
    *,
    method: str = "exact",
    num_terms: int = 200,
    num_randomizations: int = 10,
    time: float = 1.0,
    filter_diagonal_terms: bool = True,
    filter_trivial: bool | None = None,
    atol: float = 1e-16,
    seed: int | None = None,
    measure: bool = True,
    include_initial_state: bool = True,
) -> list:
    """Build SqDRIFT sampling circuits for ``ham``.

    Each circuit is a :class:`qiskit.QuantumCircuit` on ``2 * ham.norb`` qubits
    carrying the Hamiltonian time-evolution, preceded by the HF reference unless
    ``include_initial_state=False``.

    Args:
        method: ``"exact"`` (single full-evolution circuit) or ``"qdrift"``
            (ensemble of qDRIFT randomizations).
        num_terms: qDRIFT term-groups per circuit (``method="qdrift"`` only).
        num_randomizations: number of circuits to return (``method="qdrift"``;
            ``"exact"`` always returns one circuit).
        time: evolution time fed to the ``Evolution`` gate.
        filter_diagonal_terms: drop occupation-diagonal terms that do not affect
            sampled bitstrings. Applied during operator construction, before
            grouping (see :mod:`.operator`).
        filter_trivial: forwarded to ``QDriftTrotterization`` (``method="qdrift"``
            only). Rejects a *sampled* term that cannot change the occupation
            (acting only within the occupied or only within the unoccupied set),
            so it does not waste one of the ``num_terms`` slots. This is a
            distinct mechanism from ``filter_diagonal_terms``: that one prunes the
            operator up front, this one filters draws during sampling.

            The pass can only filter when it can see the occupation, i.e. when the
            reference state is inside the circuit. ``None`` (default) therefore
            tracks ``include_initial_state``; forcing ``True`` on a bare circuit
            has no effect and makes qiskit emit a ``UserWarning``.
        atol: tolerance for simplifying the normal-ordered operator.
        seed: base RNG seed. Randomization ``i`` uses ``seed + i`` for both the
            qDRIFT sampler and the transpiler, so each draw is independently
            reproducible. ``None`` leaves both unseeded (non-reproducible).
        measure: append ``measure_all()`` to each circuit.
        include_initial_state: prepare the HF reference inside the circuit
            (default). ``False`` emits the bare evolution, leaving the reference
            state to the run stage; either way the choice is recorded in
            ``metadata["initial_state_included"]``.

    Returns:
        The generated circuits, in randomization order.
    """
    if method not in ("exact", "qdrift"):
        raise ValueError(f"unknown method {method!r}; use 'exact' or 'qdrift'")

    # The qDRIFT pass filters a draw by comparing it against the occupation, which
    # it can only read from an InitializeModes gate. On a bare circuit the option
    # is inert and qiskit warns, so default it to wherever the reference state is.
    if filter_trivial is None:
        filter_trivial = include_initial_state

    from qiskit_fermions.circuit import FermionicCircuit
    from qiskit_fermions.circuit.library import Evolution, InitializeModes
    from qiskit_fermions.transpiler import FermionicPassManager
    from qiskit_fermions.transpiler.passes import QDriftTrotterization
    from qiskit_fermions.transpiler.presets import generate_preset_jw_pass_manager

    # Diagonal filtering + canonical group ordering happen during operator
    # construction, not in the qDRIFT pass, so what is grouped is exactly what
    # is sampled -- see the :mod:`.operator` docstring.
    normal, num_modes = build_canonical_operator(
        ham, atol=atol, filter_diagonal_terms=filter_diagonal_terms
    )
    occ = _hf_occupation(ham.norb, ham.nelec)

    def _fresh_circuit() -> Any:
        """Build a fresh evolution circuit for one randomization.

        A new circuit (and its own metadata dict) per pass-manager run: sharing
        one circuit across randomizations lets qiskit passes leak metadata
        between results, since a pass that returns its input DAG unchanged still
        carries the previous randomization's entries (aliased by object id).
        """
        circ = FermionicCircuit(num_modes)
        if include_initial_state:
            circ.append(InitializeModes(occ), circ.modes)
        circ.append(Evolution(num_modes, normal, time), circ.modes)
        return circ

    def _pass_manager(draw_seed: int | None) -> Any:
        """A preset JW pass manager, seeded when ``draw_seed`` is given."""
        if draw_seed is None:
            return generate_preset_jw_pass_manager()
        return generate_preset_jw_pass_manager(seed_transpiler=draw_seed)

    if method == "exact":
        circuits = [_pass_manager(seed).run(_fresh_circuit())]
    else:

        def _run_one(draw_seed: int | None) -> Any:
            """Build one randomization with its own seeded pass manager.

            ``filter_trivial`` filters the pass's own *draws*; the occupation-
            diagonal terms of the operator were already pruned during its
            construction, which is a separate mechanism (see :mod:`.operator` --
            doing that pruning here instead would break the canonical group order
            the seeded draw depends on).
            """
            pm = _pass_manager(draw_seed)
            pm.optimization = FermionicPassManager(
                [QDriftTrotterization(num_terms, filter_trivial=filter_trivial, rng=draw_seed)]
            )
            return pm.run(_fresh_circuit())

        circuits = [_run_one(None if seed is None else seed + i) for i in range(num_randomizations)]

    if measure:
        circuits = [qc.measure_all(inplace=False) for qc in circuits]

    # Record whether the reference state is already inside the circuit, so the
    # run stage can tell whether a prep still has to be prepended.
    for qc in circuits:
        qc.metadata = {**(qc.metadata or {}), "initial_state_included": include_initial_state}
    return circuits
