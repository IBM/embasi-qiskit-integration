# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Initial-state preparation, prepended to a bare ansatz circuit at run time.

The ansatz circuit ("core") encodes the *evolution*; which determinant it evolves
is decided here and composed in front of it. Keeping the two separable means one
generated core can be sampled from different reference states without rebuilding
it.

:func:`resolve_initial_state` picks the prep circuit by precedence:

0. the core already contains its own initial state (``metadata
   ["initial_state_included"]``) -- an identity prep, so nothing is applied twice;
1. an explicit ``initial_state_bitstring``;
2. Hartree-Fock from ``(n_alpha, n_beta, n_orbitals)``.

Ordering convention throughout: Jordan-Wigner with a **blocked alpha|beta** mode
layout -- the alpha block occupies qubits ``[0, n_alpha)`` and the beta block
``[n_orbitals, n_orbitals + n_beta)``. This matches the ffsim / qiskit-fermions
layout the circuit generator emits.
"""

from __future__ import annotations

import warnings

from qiskit import QuantumCircuit


def hf_prep_circuit(
    num_qubits: int,
    n_orbitals: int,
    n_alpha: int,
    n_beta: int,
    permutation: list[int] | None = None,
) -> QuantumCircuit:
    """Build the Hartree-Fock determinant as X gates on the occupied JW qubits.

    Args:
        num_qubits: circuit width; must equal ``2 * n_orbitals``.
        n_orbitals: spatial orbitals.
        n_alpha: alpha electrons (occupying ``[0, n_alpha)``).
        n_beta: beta electrons (occupying ``[n_orbitals, n_orbitals + n_beta)``).
        permutation: optional mode relabeling; occupied indices are mapped
            through it, so a core circuit synthesised under a permuted mode
            order still receives the matching reference state.

    Raises:
        ValueError: if ``num_qubits != 2 * n_orbitals``, if the electron counts
            do not fit the orbitals, or if ``permutation`` is not a permutation
            of ``range(num_qubits)``.
    """
    if 2 * n_orbitals != num_qubits:
        raise ValueError(f"expected 2*n_orbitals ({2 * n_orbitals}) qubits, got {num_qubits}")
    if not 0 <= n_alpha <= n_orbitals or not 0 <= n_beta <= n_orbitals:
        raise ValueError(
            f"electron counts (n_alpha={n_alpha}, n_beta={n_beta}) do not fit "
            f"n_orbitals={n_orbitals}"
        )

    # Occupied modes in the blocked ordering: alpha block, then beta block.
    occupied = list(range(n_alpha)) + [n_orbitals + i for i in range(n_beta)]

    if permutation is not None:
        if sorted(permutation) != list(range(num_qubits)):
            raise ValueError("permutation must be a rearrangement of range(num_qubits)")
        occupied = [permutation[mode] for mode in occupied]

    circuit = QuantumCircuit(num_qubits, name="hf_prep")
    if occupied:
        circuit.x(occupied)
    return circuit


def bitstring_prep_circuit(bitstring: str) -> QuantumCircuit:
    """Build a computational-basis state-prep circuit from an MSB-left bitstring.

    The convention matches Qiskit's ``get_counts()`` output: the *last* character
    corresponds to qubit ``0``. An X gate is applied on every qubit whose
    character is ``'1'``. Spaces are ignored, so a register-separated string from
    a counts dict can be passed through unchanged.

    **Spin blocks are mirrored relative to qubit indices.** Because the string is
    MSB-left while the alpha block occupies the *low* qubits, the rightmost
    ``n_orbitals`` characters are the **alpha** block and the leftmost
    ``n_orbitals`` are the **beta** block -- the opposite of how the string reads
    left-to-right. For ``n_orbitals=2, n_alpha=1, n_beta=0`` the correct string is
    ``"0001"`` (X on qubit 0); ``"1000"`` would occupy a *beta* orbital instead.
    A symmetric case (``n_alpha == n_beta``) cannot reveal a swapped block, so
    check the convention against :func:`hf_prep_circuit` when in doubt.

    Raises:
        ValueError: if the string is empty or contains anything but ``0``/``1``.
    """
    clean = bitstring.replace(" ", "")
    if not clean or any(character not in "01" for character in clean):
        raise ValueError(f"bitstring must be non-empty and contain only 0/1: {bitstring!r}")

    num_qubits = len(clean)
    circuit = QuantumCircuit(num_qubits, name="bitstring_prep")
    # clean[-(i + 1)] is qubit i: the string is MSB-left, qubit 0 rightmost.
    occupied = [i for i in range(num_qubits) if clean[-(i + 1)] == "1"]
    if occupied:
        circuit.x(occupied)
    return circuit


def resolve_initial_state(
    *,
    num_qubits: int,
    initial_state_bitstring: str | None = None,
    n_alpha: int | None = None,
    n_beta: int | None = None,
    n_orbitals: int | None = None,
    permutation: list[int] | None = None,
    core: QuantumCircuit | None = None,
) -> QuantumCircuit:
    """Resolve the initial-state prep circuit for a core ansatz circuit.

    Precedence (highest first):

    0. ``core`` carries ``metadata["initial_state_included"]`` -- the reference
       state is already baked into the core, so an identity prep is returned and
       nothing is applied twice.
    1. ``initial_state_bitstring`` -- MSB-left; its length must equal
       ``num_qubits``. Mind the mirroring: the *rightmost* ``n_orbitals``
       characters are the alpha block (see :func:`bitstring_prep_circuit`).
    2. Hartree-Fock from ``(n_alpha, n_beta, n_orbitals)``.

    Args:
        num_qubits: width of the core circuit the prep will be composed with.
        initial_state_bitstring: explicit determinant, MSB-left.
        n_alpha: alpha electrons (HF fallback).
        n_beta: beta electrons (HF fallback).
        n_orbitals: spatial orbitals (HF fallback); ``2 * n_orbitals`` must equal
            ``num_qubits``.
        permutation: optional mode relabeling, forwarded to
            :func:`hf_prep_circuit`.
        core: the core circuit, consulted only for its metadata flag.

    Raises:
        ValueError: if the bitstring length disagrees with ``num_qubits``, or if
            no source yields a prep circuit.
    """
    if core is not None and bool((core.metadata or {}).get("initial_state_included")):
        # Prepending anything here would double-apply the reference state.
        if initial_state_bitstring is not None:
            # Warn rather than silently sample a different determinant than asked:
            # the energy would look plausible with no indication it was ignored.
            warnings.warn(
                "core circuit already includes its initial state; ignoring the "
                f"requested initial_state_bitstring={initial_state_bitstring!r}. "
                "Rebuild the core with include_initial_state=False to choose the "
                "reference determinant at run time.",
                stacklevel=2,
            )
        return QuantumCircuit(num_qubits, name="initial_state_included")

    if initial_state_bitstring is not None:
        clean = initial_state_bitstring.replace(" ", "")
        if len(clean) != num_qubits:
            raise ValueError(
                f"initial_state_bitstring length {len(clean)} != num_qubits {num_qubits}"
            )
        return bitstring_prep_circuit(initial_state_bitstring)

    if isinstance(n_orbitals, int) and isinstance(n_alpha, int) and isinstance(n_beta, int):
        return hf_prep_circuit(num_qubits, n_orbitals, n_alpha, n_beta, permutation=permutation)

    raise ValueError(
        "cannot resolve an initial state: pass initial_state_bitstring, or all of "
        f"n_orbitals/n_alpha/n_beta as integers (got n_orbitals={n_orbitals}, "
        f"n_alpha={n_alpha}, n_beta={n_beta}), or a core circuit whose metadata "
        "sets initial_state_included"
    )


def compose_full_circuit(
    prep: QuantumCircuit, core: QuantumCircuit, add_measure_all: bool = True
) -> QuantumCircuit:
    """Compose ``prep`` then ``core`` on a fresh register, optionally measuring.

    Args:
        prep: the initial-state circuit (from :func:`resolve_initial_state`).
        core: the ansatz/evolution circuit; its width must match ``prep``'s.
        add_measure_all: append ``measure_all()`` to the composed circuit.

    Raises:
        ValueError: if ``prep`` and ``core`` have different widths.
    """
    if prep.num_qubits != core.num_qubits:
        raise ValueError(
            f"prep has {prep.num_qubits} qubits, core has {core.num_qubits}; must match"
        )
    full = QuantumCircuit(core.num_qubits, name=core.name or "full")
    full.compose(prep, inplace=True)
    full.compose(core, inplace=True)
    if add_measure_all:
        full.measure_all()
    return full
