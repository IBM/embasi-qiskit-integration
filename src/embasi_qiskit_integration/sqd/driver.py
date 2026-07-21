# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""SQD driver: measurement counts -> energy + RDMs.

Wraps ``qiskit_addon_sqd``'s high-level
:func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`, which runs
the configuration-recovery + projection + diagonalization loop in the fermionic
CI space and returns an ``SCIResult`` carrying spin-summed ``rdm1``/``rdm2``.

The total energy is ``result.energy + ham.e_core``; per-iteration energies are
collected via the addon's ``callback`` hook and stored in ``diagnostics``.
"""

from __future__ import annotations

import numpy as np

from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult


def run_sqd(
    ham: EmbeddedHamiltonian,
    counts: dict[str, int],
    *,
    samples_per_batch: int = 300,
    num_batches: int = 5,
    max_iterations: int = 5,
    symmetrize_spin: bool = True,
    seed: int | None = None,
) -> SolverResult:
    """Run SQD on measurement ``counts`` for the active-space ``ham``.

    Args:
        counts: ``{bitstring: count}`` over ``2 * ham.norb`` bits (alpha|beta
            blocked, matching the ffsim/qiskit-fermions Jordan-Wigner layout).
        samples_per_batch: configurations sampled per batch per iteration.
        num_batches: batches per configuration-recovery iteration.
        max_iterations: configuration-recovery iterations.
        symmetrize_spin: enforce spin symmetry in the recovered subspace.
        seed: RNG seed (int) for the SQD sampling — deterministic when set.

    Returns:
        :class:`SolverResult` with total energy, spin-summed ``rdm1``/``rdm2``,
        and diagnostics (per-iteration energies, subspace stats, settings).
    """
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    norb = ham.norb
    num_bits = 2 * norb
    bit_array = BitArray.from_counts(counts, num_bits=num_bits)

    iteration_energies: list[float] = []

    def _callback(results) -> None:
        # results: list[SCIResult] for the batches of this iteration.
        iteration_energies.append(float(min(r.energy for r in results)))

    rng = np.random.default_rng(seed) if seed is not None else None
    result = diagonalize_fermionic_hamiltonian(
        ham.h1,
        ham.h2,
        bit_array,
        samples_per_batch=samples_per_batch,
        norb=norb,
        nelec=ham.nelec,
        num_batches=num_batches,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        callback=_callback,
        seed=rng,
    )

    energy = float(result.energy) + ham.e_core
    rdm1 = np.asarray(result.rdm1)
    rdm2 = np.asarray(result.rdm2) if result.rdm2 is not None else None

    # Per-iteration total energies (electronic + core) for convergence checks.
    iteration_totals = [e + ham.e_core for e in iteration_energies]

    diagnostics = {
        "solver": "qiskit-addon-sqd",
        "iteration_energies": iteration_totals,
        "n_iterations": len(iteration_totals),
        "n_distinct_bitstrings": len(counts),
        "n_shots": int(sum(counts.values())),
        "samples_per_batch": samples_per_batch,
        "num_batches": num_batches,
        "max_iterations": max_iterations,
        "symmetrize_spin": symmetrize_spin,
        "seed": seed,
    }
    return SolverResult(energy=energy, rdm1=rdm1, rdm2=rdm2, diagnostics=diagnostics)
