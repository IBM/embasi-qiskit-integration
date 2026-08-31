# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Spin-block layout conventions for Jordan-Wigner bitstrings.

This package agrees with ``qiskit-addon-sqd``: in a bit array the *right* half (higher
indices) is the alpha sector. The addon states it outright -- "the alpha part
concatenated on the right-hand side" -- and splits on ``samples[:, norb:]`` for alpha,
``samples[:, :norb]`` for beta. Counts produced here feed the addon directly, with no
conversion.

The naming below exists for an *external* producer that emits the mirrored order:

``ALPHA_BETA`` (native): qubit ``i`` is alpha spatial orbital ``i`` for ``i < norb``;
qubit ``norb + i`` is beta orbital ``i``. Because Qiskit's counts keys are MSB-left, the
*rightmost* ``norb`` characters of a key are the alpha block -- the mirroring documented
on :func:`~embasi_qiskit_integration.circuit_run.prep.bitstring_prep_circuit`. The
reference determinant occupies the lowest index of each block
(:func:`~embasi_qiskit_integration.circuit_run.prep.hf_prep_circuit`).

``BETA_ALPHA``: the two halves exchanged. Some circuit builders emit this order; declare
it and convert on the way in with :func:`swap_spin_halves`.

The two agree exactly when ``n_alpha == n_beta``, so a genuine mismatch is invisible on
every closed-shell system and first appears open shell -- as a plausible energy with the
right total Hamming weight and a passing particle-number check.
:func:`verify_counts_sector` is the guard for counts arriving from outside;
:func:`swap_spin_halves` converts a genuinely ``BETA_ALPHA`` producer's bitstrings.
"""

from __future__ import annotations

import numpy as np

ALPHA_BETA = "alpha_beta"
BETA_ALPHA = "beta_alpha"

#: The convention this package produces and consumes everywhere.
NATIVE_SPIN_LAYOUT = ALPHA_BETA


def swap_spin_halves(bitstring: str, num_orbitals: int) -> str:
    """Exchange the two spin halves of a bitstring (``ALPHA_BETA`` <-> ``BETA_ALPHA``).

    Raises:
        ValueError: if the length is not ``2 * num_orbitals``.
    """
    if len(bitstring) != 2 * num_orbitals:
        raise ValueError(
            f"bitstring length {len(bitstring)} != 2 * num_orbitals ({2 * num_orbitals})"
        )
    return bitstring[num_orbitals:] + bitstring[:num_orbitals]


def create_hf_reference(*, num_orbitals: int, num_elec_a: int, num_elec_b: int) -> np.ndarray:
    """The Hartree-Fock reference bit **array**, in the ``BETA_ALPHA`` array layout.

    Beta block in the first ``norb`` entries, alpha in the last ``norb``, with the
    occupied orbitals at the *highest* index of each block. As an array this is the same
    right-half-is-alpha convention the addon uses -- ``arr[norb:]`` is the alpha
    occupation -- so it is the array form of an externally-supplied reference
    determinant, useful for asserting a layout assumption rather than re-deriving it by
    hand.

    This package's own reference *circuit* is
    :func:`~embasi_qiskit_integration.circuit_run.prep.hf_prep_circuit`, which places
    the occupied orbitals at the lowest index of each block instead.

    Returns:
        ``int8`` array of length ``2 * num_orbitals``, Hamming weight
        ``num_elec_a + num_elec_b``.
    """
    if num_elec_a < 0 or num_elec_b < 0:
        raise ValueError(
            f"negative electron count: num_elec_a={num_elec_a}, num_elec_b={num_elec_b}"
        )
    if num_elec_a > num_orbitals or num_elec_b > num_orbitals:
        raise ValueError(
            f"electron count exceeds orbital count: num_elec_a={num_elec_a}, "
            f"num_elec_b={num_elec_b}, num_orbitals={num_orbitals}"
        )
    n_qubits = 2 * num_orbitals
    arr = np.zeros(n_qubits, dtype=np.int8)
    arr[num_orbitals - num_elec_b : num_orbitals] = 1
    arr[n_qubits - num_elec_a : n_qubits] = 1
    return arr


def counts_sector(counts: dict[str, int], norb: int) -> tuple[int, int] | None:
    """Return the ``(n_alpha, n_beta)`` all bitstrings share, or ``None`` if mixed.

    Reads ``counts`` in this package's :data:`NATIVE_SPIN_LAYOUT`: keys are MSB-left
    (Qiskit's ``get_counts`` convention), so the *rightmost* ``norb`` characters are
    the alpha block -- the same mirroring documented on
    :func:`~embasi_qiskit_integration.circuit_run.prep.bitstring_prep_circuit`.

    Returning ``None`` rather than raising on a mixed pool keeps this usable as a guard
    on post-noise counts, where mixed sectors are expected rather than exceptional.
    """
    sectors = set()
    for key in counts:
        clean = key.replace(" ", "")
        if len(clean) != 2 * norb:
            raise ValueError(
                f"bitstring {key!r} has {len(clean)} bits, expected 2*norb ({2 * norb})"
            )
        alpha = clean[norb:].count("1")  # rightmost norb chars = alpha block
        beta = clean[:norb].count("1")
        sectors.add((alpha, beta))
        if len(sectors) > 1:
            return None
    return sectors.pop() if sectors else None


def verify_counts_sector(
    counts: dict[str, int], norb: int, nelec: tuple[int, int], *, source: str = "counts"
) -> None:
    """Assert externally-produced ``counts`` use this package's spin layout.

    Run this on counts (or an explicit determinant) crossing into this package from
    another repo. It catches a swapped alpha/beta block -- which is undetectable by a
    Hamming-weight or ``trace(rdm1)`` check, because both are spin-blind -- by
    comparing the *per-block* occupation against ``nelec``.

    Only decisive when the dominant configurations sit in a definite sector, so it is
    a no-op for genuinely mixed-sector pools (post-noise counts), and silent for a
    closed shell, where the two layouts are identical.

    Raises:
        ValueError: if every bitstring shares a sector and that sector is ``nelec``
            reversed -- the signature of a ``BETA_ALPHA`` producer.
    """
    na, nb = nelec
    if na == nb:
        return  # the two conventions coincide; nothing to detect
    observed = counts_sector(counts, norb)
    if observed is None or observed == (na, nb):
        return
    if observed == (nb, na):
        raise ValueError(
            f"{source} appear to use the {BETA_ALPHA!r} spin layout: every bitstring "
            f"has (n_alpha, n_beta) = {observed} but this Hamiltonian is {nelec}, the "
            "exact reversal.  Check the sampler prepared the determinant this "
            f"Hamiltonian describes; if the producer really is {BETA_ALPHA!r}, convert "
            "with swap_spin_halves().  See this module's docstring for the conventions."
        )
    raise ValueError(
        f"{source} sit in sector {observed}, but this Hamiltonian is nelec={nelec}. "
        "The sampler prepared a different determinant than the Hamiltonian describes."
    )
