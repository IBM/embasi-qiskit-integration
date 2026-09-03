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

from embasi_qiskit_integration.circuit_run.spin_layout import (
    NATIVE_SPIN_LAYOUT,
    verify_counts_sector,
)
from embasi_qiskit_integration.contract import EmbeddedHamiltonian, SolverResult


def run_sqd(
    ham: EmbeddedHamiltonian,
    counts: dict[str, int],
    *,
    samples_per_batch: int = 300,
    num_batches: int = 5,
    max_iterations: int = 5,
    symmetrize_spin: bool | None = None,
    spin_sq: float | None = None,
    initial_occupancies: tuple[np.ndarray, np.ndarray] | None = None,
    include_configurations=None,
    seed: int | None = None,
) -> SolverResult:
    """Run SQD on measurement ``counts`` for the active-space ``ham``.

    Args:
        counts: ``{bitstring: count}`` over ``2 * ham.norb`` bits (alpha|beta
            blocked, matching the ffsim/qiskit-fermions Jordan-Wigner layout).
        samples_per_batch: configurations sampled per batch per iteration.
        num_batches: batches per configuration-recovery iteration.
        max_iterations: configuration-recovery iterations.
        symmetrize_spin: merge the alpha and beta CI-string pools so the subspace is
            invariant under exchanging the two spins.  ``None`` (default) derives it
            from the sector: ``True`` for a closed shell, ``False`` when
            ``n_alpha != n_beta``.  Merging pools of different Hamming weight is
            meaningless, and the addon rejects it outright, so an explicit ``True``
            on an open-shell ``ham`` raises here rather than deeper in the stack.
        spin_sq: target total spin ``S(S+1)`` to project onto. ``None`` (default)
            imposes no projection, matching the addon's own default -- the solve then
            returns the variationally lowest state in the ``nelec`` sector *whatever its
            multiplicity*, since ``nelec`` fixes only ``Sz``, never ``S``. Set it when a
            wrong-``S`` state could lie below the one you want (an ``Sz = 0`` open-shell
            sector holds both singlet and triplet). Threaded via the addon's
            ``sci_solver`` hook, since ``diagonalize_fermionic_hamiltonian`` takes no
            ``spin_sq`` of its own; the penalty is applied by ``fci.addons.fix_spin_``.
            Note it *penalises* the wrong multiplicity rather than searching for a
            higher-``S`` state, so it cannot raise the answer above the true ground
            state of the sector.
        initial_occupancies: per-spin ``(occ_alpha, occ_beta)`` prior seeding
            configuration recovery. Without it the addon postselects the first
            iteration on the exact ``(n_alpha, n_beta)`` sector instead, and raises if
            nothing survives -- likelier open shell, where the asymmetric target sector
            is a smaller slice of a noisy distribution with no spin degeneracy to help.
        include_configurations: configurations to force into every subspace (e.g. the
            Hartree-Fock determinant), as the addon's ``include_configurations``.
        seed: RNG seed (int) for the SQD sampling — deterministic when set.

    Returns:
        :class:`SolverResult` with total energy, spin-summed ``rdm1``/``rdm2``,
        and diagnostics (per-iteration energies, subspace stats, settings).
    """
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    na, nb = ham.nelec
    if symmetrize_spin is None:
        symmetrize_spin = na == nb
    elif symmetrize_spin and na != nb:
        raise ValueError(
            f"symmetrize_spin=True is invalid for nelec={ham.nelec}: merging the alpha "
            f"and beta CI-string pools presumes they hold the same number of electrons "
            f"({na} != {nb}).  Pass symmetrize_spin=False, or leave it as None to have "
            "it derived from the sector."
        )

    norb = ham.norb
    num_bits = 2 * norb
    if not counts:
        raise ValueError(
            "counts is empty; SQD has nothing to diagonalize. Check that the "
            "sampler actually ran (an empty circuit list yields no counts) and "
            "that the circuits carry measurements."
        )
    verify_counts_sector(counts, norb, ham.nelec, source="SQD counts")

    bit_array = BitArray.from_counts(counts, num_bits=num_bits)

    iteration_energies: list[float] = []

    def _callback(results) -> None:
        # results: list[SCIResult] for the batches of this iteration.
        iteration_energies.append(float(min(r.energy for r in results)))

    rng = np.random.default_rng(seed) if seed is not None else None

    sci_solver = None
    if spin_sq is not None:
        import functools

        from qiskit_addon_sqd.fermion import solve_sci_batch

        sci_solver = functools.partial(solve_sci_batch, spin_sq=spin_sq)

    extra: dict = {}
    if initial_occupancies is not None:
        extra["initial_occupancies"] = initial_occupancies
    if include_configurations is not None:
        extra["include_configurations"] = include_configurations

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
        sci_solver=sci_solver,
        callback=_callback,
        seed=rng,
        **extra,
    )

    spin_square: float | None = None
    try:
        spin_square = float(result.sci_state.spin_square())
    except Exception:  # pragma: no cover - diagnostic only, never fatal
        spin_square = None

    energy = float(result.energy) + ham.e_core
    rdm1 = np.asarray(result.rdm1)
    rdm2 = np.asarray(result.rdm2) if result.rdm2 is not None else None

    rdm1a: np.ndarray | None = None
    rdm1b: np.ndarray | None = None
    try:
        pair = np.asarray(result.sci_state.rdm(rank=1, spin_summed=False))
    except Exception:  # pragma: no cover - older addon without the spin-resolved rdm
        pair = None
    if pair is not None and pair.shape == (2, norb, norb):
        if np.allclose(pair[0] + pair[1], rdm1, atol=1e-8):
            rdm1a, rdm1b = pair[0], pair[1]

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
        "spin_layout": NATIVE_SPIN_LAYOUT,
        "spin_sq_target": spin_sq,
        "spin_square": spin_square,
        # min possible S(S+1) for this sector: S >= |na - nb| / 2
        "spin_square_min": (abs(na - nb) / 2.0) * (abs(na - nb) / 2.0 + 1.0),
        "spin_resolved_rdm1": rdm1a is not None,
        "seed": seed,
    }
    return SolverResult(
        energy=energy,
        rdm1=rdm1,
        rdm2=rdm2,
        rdm1a=rdm1a,
        rdm1b=rdm1b,
        diagnostics=diagnostics,
    )
