# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SqDRIFT ansatz circuits via ``qiskit-fermions`` (primary ansatz).

Construction mirrors the reference SqDRIFT step: one canonicalized grouped
operator (see :mod:`.operator`), then per randomization a *fresh* circuit and a
*fresh* ``seed``-seeded JW pass manager carrying ``QDriftTrotterization``, swept
over ``time x num_groups``, with ``measure_all`` appended last.
``build_sqdrift_circuits(method="qdrift", include_initial_state=False)``
reproduces the reference's circuits exactly.

Two capabilities go beyond it, both opt-in and both defaulted so the reference
behaviour is what you get on the path that matters:

- ``include_initial_state`` prepends the Hartree-Fock reference via
  ``InitializeModes``. The reference always evolves the *vacuum*; passing
  ``False`` (what :class:`~embasi_qiskit_integration.solvers.SQDSolver` does)
  gives the identical bare circuit, leaving the determinant to the run stage.
- ``method="exact"`` synthesises the full evolution with no qDRIFT sampling. The
  reference has no such mode -- ``method="qdrift"`` is its behaviour. "exact" is
  the deterministic reference path used for the frozen counts fixture and the
  SQD-vs-FCI check.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from embasi_qiskit_integration.circuit_generator.operator import (
    build_canonical_operator,
    fermions_available,
)
from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def sqdrift_available() -> bool:
    """True if ``qiskit-fermions`` is importable in this environment."""
    return fermions_available()


def _as_float_list(value: float | Sequence[float]) -> list[float]:
    """Coerce a scalar or sequence of evolution times into a list of floats.

    A bare ``1.0`` is accepted as ``[1.0]`` so existing scalar callers keep
    working; the swept form is a sequence.
    """
    if isinstance(value, (int, float)):
        return [float(value)]
    times = [float(t) for t in value]
    if not times:
        raise ValueError("time must contain at least one evolution time")
    return times


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
    num_groups: int | Sequence[int] = 15,
    num_randomizations: int = 500,
    time: float | Sequence[float] = (1.0, 2.0, 3.0),
    filter_diagonal_terms: bool = True,
    filter_trivial: bool | None = None,
    atol: float = 1e-16,
    seed: int | None = 42,
    measure: bool = True,
    include_initial_state: bool = True,
) -> list:
    """Build SqDRIFT sampling circuits for ``ham``.

    Each circuit is a :class:`qiskit.QuantumCircuit` on ``2 * ham.norb`` qubits
    carrying the Hamiltonian time-evolution, preceded by the HF reference unless
    ``include_initial_state=False``.

    Defaults mirror the reference SqDRIFT settings (``time`` ``[1.0, 2.0, 3.0]``,
    ``num_groups`` ``[10, 15, 20]``, ``num_randomizations`` 500, ``seed`` 42), so a
    bare ``method="qdrift"`` call produces that full sweep: **4500 circuits**
    (3 times x 3 term-counts x 500 randomizations). Narrow the axes for a smaller
    budget -- ``SQDSolver`` does exactly that, pinning one ``(time, num_groups)``
    pair and its own ``num_randomizations``.

    Args:
        method: ``"exact"`` (one full-evolution circuit per ``time``) or
            ``"qdrift"`` (ensemble over time x num_groups x randomizations).
        num_groups: how many term-groups qDRIFT samples per circuit
            (``method="qdrift"`` only) -- the length of the randomized product,
            so it trades circuit depth against accuracy. Scalar or sequence; a
            sequence is a sweep axis. Named for consistency with the reference
            settings; it is passed positionally to ``QDriftTrotterization``,
            whose own parameter is called ``num_terms``. Note this is *not*
            ``FermionOperator.num_groups()`` (how many groups the Hamiltonian
            has, typically ~1000) -- it is how many of them get drawn.
        num_randomizations: randomizations per ``(time, num_groups)`` combination
            (``method="qdrift"``; ``"exact"`` yields one circuit per ``time``).
        time: evolution time(s) t for ``exp(-i t H)``, fed to the ``Evolution``
            gate. Scalar or sequence; a sequence is a sweep axis, combined with
            ``num_groups`` as a cartesian product.
        filter_diagonal_terms: drop occupation-diagonal terms that do not affect
            sampled bitstrings. Applied during operator construction, before
            grouping (see :mod:`.operator`).
        filter_trivial: forwarded to ``QDriftTrotterization`` (``method="qdrift"``
            only). Rejects a *sampled* term that cannot change the occupation
            (acting only within the occupied or only within the unoccupied set),
            so it does not waste one of the ``num_groups`` slots. This is a
            distinct mechanism from ``filter_diagonal_terms``: that one prunes the
            operator up front, this one filters draws during sampling.

            The pass can only filter when it can see the occupation, i.e. when the
            reference state is inside the circuit. ``None`` (default) therefore
            tracks ``include_initial_state``; forcing ``True`` on a bare circuit
            has no effect and makes qiskit emit a ``UserWarning``.
        atol: tolerance for simplifying the normal-ordered operator.
        seed: base RNG seed (default 42, as in the reference settings).
            Randomization ``i`` of each ``(time, num_groups)`` combination uses
            ``seed + i`` for both the qDRIFT sampler and the transpiler, so each
            draw is independently reproducible. The index restarts per combination,
            so combinations sharing a randomization index share a draw -- this
            matches the reference implementation and isolates the effect of
            ``time``/``num_groups`` from sampling noise. ``None`` leaves both
            unseeded (non-reproducible).
        measure: append ``measure_all()`` to each circuit.
        include_initial_state: prepare the HF reference inside the circuit
            (default). ``False`` emits the bare evolution, leaving the reference
            state to the run stage; either way the choice is recorded in
            ``metadata["initial_state_included"]``.

    Returns:
        The generated circuits. Order is the flattened sweep: for each ``time``,
        for each ``num_groups``, each randomization in turn.
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

    # The operator depends only on (ham, atol, filter_diagonal_terms) -- `time`
    # enters the Evolution gate and `num_groups` only the qDRIFT pass -- so one
    # build above serves every combination of the sweep.
    times = _as_float_list(time)
    group_counts = (
        [int(num_groups)] if isinstance(num_groups, int) else [int(n) for n in num_groups]
    )
    if not group_counts:
        raise ValueError("num_groups must contain at least one group count")

    def _fresh_circuit(evolution_time: float) -> Any:
        """Build a fresh evolution circuit at ``evolution_time``.

        A new circuit (and its own metadata dict) per pass-manager run: sharing
        one circuit across randomizations lets qiskit passes leak metadata
        between results, since a pass that returns its input DAG unchanged still
        carries the previous randomization's entries (aliased by object id).
        """
        circ = FermionicCircuit(num_modes)
        if include_initial_state:
            circ.append(InitializeModes(occ), circ.modes)
        circ.append(Evolution(num_modes, normal, evolution_time), circ.modes)
        return circ

    def _pass_manager(draw_seed: int | None) -> Any:
        """A preset JW pass manager, seeded when ``draw_seed`` is given."""
        if draw_seed is None:
            return generate_preset_jw_pass_manager()
        return generate_preset_jw_pass_manager(seed_transpiler=draw_seed)

    def _draw_seed(randomization: int) -> int | None:
        """Seed for randomization ``i``: ``seed + i`` (None stays unseeded).

        The index is the randomization number *within* a ``(time, num_groups)``
        combination, and restarts at 0 for each combination -- matching the
        reference implementation, where every combination is a separate ``build()``
        call over ``range(num_randomizations)``. So combinations that share a
        randomization index also share a qDRIFT draw, which isolates the effect of
        ``time``/``num_groups`` from sampling noise.
        """
        return None if seed is None else seed + randomization

    circuits = []
    if method == "exact":
        # One full-evolution circuit per time; num_groups/num_randomizations are
        # qDRIFT-only knobs and do not apply. No sampling happens here, so the
        # seed only feeds the transpiler and need not vary across times.
        for evolution_time in times:
            circuits.append(_pass_manager(_draw_seed(0)).run(_fresh_circuit(evolution_time)))
    else:

        def _run_one(evolution_time: float, n_groups: int, draw_seed: int | None) -> Any:
            """Build one randomization with its own seeded pass manager.

            ``filter_trivial`` filters the pass's own *draws*; the occupation-
            diagonal terms of the operator were already pruned during its
            construction, which is a separate mechanism (see :mod:`.operator` --
            doing that pruning here instead would break the canonical group order
            the seeded draw depends on).
            """
            pm = _pass_manager(draw_seed)
            pm.optimization = FermionicPassManager(
                [QDriftTrotterization(n_groups, filter_trivial=filter_trivial, rng=draw_seed)]
            )
            return pm.run(_fresh_circuit(evolution_time))

        for evolution_time in times:
            for n_groups in group_counts:
                # Seeds restart at `seed` for every combination, as in the
                # reference implementation (one build() call per combination).
                for randomization in range(num_randomizations):
                    circuits.append(_run_one(evolution_time, n_groups, _draw_seed(randomization)))

    if measure:
        circuits = [qc.measure_all(inplace=False) for qc in circuits]

    # Record whether the reference state is already inside the circuit, so the
    # run stage can tell whether a prep still has to be prepended.
    for qc in circuits:
        qc.metadata = {**(qc.metadata or {}), "initial_state_included": include_initial_state}
    return circuits
