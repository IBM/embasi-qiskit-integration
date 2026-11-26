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


def _solve_sci_batch_from_reference(
    ci_strings, one_body_tensor, two_body_tensor, norb, nelec, *, spin_sq=None
):
    """The addon's ``solve_sci_batch``, but each Davidson starts from the aufbau determinant.

    Left to itself, pyscf's ``selected_ci.kernel_fixed_space`` starts from the single
    lowest-diagonal determinant (plus 1e-5 noise).  Near dissociation that determinant
    can sit in a different symmetry sector than the ground state, and the iteration never
    leaves it, so an excited root comes back even when the subspace is the full space.
    Starting from the aufbau determinant -- the one the sampling circuits prepare --
    reaches the ground state, including high-spin ground states.  A subspace without that
    determinant keeps pyscf's own guess.
    """
    from pyscf.fci import selected_ci
    from qiskit_addon_sqd.fermion import solve_sci

    ref_a, ref_b = (1 << nelec[0]) - 1, (1 << nelec[1]) - 1
    results = []
    for strs_a, strs_b in ci_strings:
        kwargs = {}
        ia = np.flatnonzero(np.asarray(strs_a) == ref_a)
        ib = np.flatnonzero(np.asarray(strs_b) == ref_b)
        if ia.size and ib.size:
            ci0 = np.zeros((len(strs_a), len(strs_b)))
            ci0[ia[0], ib[0]] = 1.0
            # Must be an SCIvector: kernel_fixed_space silently ignores a plain array.
            kwargs["ci0"] = selected_ci._as_SCIvector(ci0, (strs_a, strs_b))
        results.append(
            solve_sci(
                (strs_a, strs_b),
                one_body_tensor,
                two_body_tensor,
                norb=norb,
                nelec=nelec,
                spin_sq=spin_sq,
                **kwargs,
            )
        )
    return results


#: ``sci-solvers`` solvers whose constructor accepts ``compute_rdm2=True``.
#: Torch-SCI needs it for the spin-summed RDMs SQD consumes; SBD gates its RDMs
#: the same way; the general-algebra family builds the 2-RDM only on request.
#: PySCF's constructor uses ``with_rdm2`` (and defaults it on), while
#: truncated-Davidson defaults the callable's ``compute_rdm2`` to True.
_RDM2_KWARG = {
    "torch_sci": "compute_rdm2",
    "sbd": "compute_rdm2",
    "arpack": "compute_rdm2",
    "davidson": "compute_rdm2",
    "lobpcg": "compute_rdm2",
    "primme": "compute_rdm2",
    "slepc": "compute_rdm2",
    "evsl": "compute_rdm2",
    "pyscf_davidson": "compute_rdm2",
}


#: qiskit-addon-sqd's built-in selected-CI path.  It is not a solver named after
#: the Qiskit ecosystem: the addon supplies :func:`solve_sci`, which runs PySCF's
#: selected-CI diagonalization.  The explicit name keeps it distinct from the
#: optional ``sci-solvers`` backends (and from sci-solvers' own ``pyscf`` entry).
QISKIT_ADDON_SOLVER = "qiskit_addon_sqd"


def _available_solvers(available: tuple[str, ...]) -> tuple[str, ...]:
    """The built-in baseline first, then torch-sci (the public default), then rest."""
    ordered = [QISKIT_ADDON_SOLVER]
    for name in ("torch_sci", *available):
        if name not in ordered:
            ordered.append(name)
    return tuple(ordered)


def resolve_sci_solver(name: str, spin_sq: float | None, torch_sci_kwargs: dict | None):
    """Return the ``sci_solver`` callable for qiskit-addon-sqd.

    ``"qiskit_addon_sqd"`` wraps the addon's PySCF selected-CI batch solver with
    the aufbau starting vector and needs no extra.  Every other name is
    dispatched through IBM/sci-solvers'
    :func:`~sci_solvers.energy_solvers.factory.create_solver` (``torch_sci`` is
    the default in :class:`SQDSolver`); importing the package is deferred so the
    optional ``sci-solvers`` extra is only required when selected.
    ``torch_sci_kwargs`` is forwarded only to ``torch_sci`` -- the other solvers
    have different constructor options.

    ``symmetrize_spin=False`` is deliberate: qiskit-addon-sqd already passes
    separate alpha/beta CI-string batches, and the open-shell SQD path must not
    merge them.  A solver that needs closed-shell spin symmetry can still be
    configured through its own options.
    """
    if name == QISKIT_ADDON_SOLVER:
        import functools

        return functools.partial(_solve_sci_batch_from_reference, spin_sq=spin_sq)
    try:
        from sci_solvers.energy_solvers.factory import AVAILABLE_SOLVERS, create_solver
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            f"sci_solver={name!r} needs the optional `sci-solvers` extra: "
            "uv sync --extra sci-solvers"
        ) from exc
    names = _available_solvers(AVAILABLE_SOLVERS)
    if name not in AVAILABLE_SOLVERS:
        raise ValueError(f"unknown sci_solver {name!r}; available: {', '.join(names)}")

    if name == "torch_sci":
        kwargs = dict(torch_sci_kwargs or {})
        import torch

        # The engine defaults to CUDA.  Auto-detect instead: CUDA, then Apple
        # MPS, then CPU.  An explicit device/devices in torch_sci_kwargs wins.
        # MPS has no float64, so select float32 there unless the caller chose a
        # dtype; on CUDA/CPU the engine's float64 default is unchanged.
        device = kwargs.get("device")
        if device is None and not kwargs.get("devices"):
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and (
                torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
            kwargs["device"] = device
        if str(device).startswith("mps"):
            kwargs.setdefault("dtype", torch.float32)
            # torch-sci rejects a tighter residual on MPS (there is no float64);
            # 1e-5 is its own documented floor and avoids a full iteration budget.
            kwargs.setdefault("davidson_tolerance", 1e-5)
        else:
            # The engine's 1e-4 Davidson residual leaves ~1e-6 errors in the
            # spin-resolved RDMs, too loose for the embedding feedback.
            kwargs.setdefault("davidson_tolerance", 1e-8)
        # The factory's qiskit-compatible default differs from the engine's;
        # make the choice explicit (and silence its warning).
        kwargs.setdefault("apply_ab_projection", False)
        # macOS: if qiskit-aer was imported before the package-level OpenMP
        # preload, its bundled libomp is already loaded and torch must stay
        # single-threaded to avoid the duplicate-runtime segfault.  Normally
        # preload_torch_openmp() ran first and this is a no-op.
        from embasi_qiskit_integration._openmp import preload_torch_openmp

        if not preload_torch_openmp():
            torch.set_num_threads(1)
    else:
        kwargs = {}
    kwargs.setdefault("symmetrize_spin", False)
    rdm2_kwarg = _RDM2_KWARG.get(name)
    if rdm2_kwarg is not None:
        kwargs.setdefault(rdm2_kwarg, True)

    # The unified interface exposes the qiskit-addon-sqd callable and returns
    # SCIResult objects (torch's raw __call__ returns EigenStates instead).
    try:
        solver = create_solver(name, **kwargs)
    except ValueError as exc:
        raise ValueError(f"unknown sci_solver {name!r}; available: {', '.join(names)}") from exc
    return solver.sci_solver_callable


def _normalise_rdm1(rdm1: np.ndarray, nelec: float) -> np.ndarray:
    """Symmetrise ``rdm1`` and remove float32 particle-number drift.

    Rewriting the trace is only a numerical clean-up: a deviation larger than
    ``1e-4`` means the solve itself is wrong, so leave it for the workflow's
    particle-number check to report instead of hiding it.
    """
    rdm1 = 0.5 * (np.asarray(rdm1, dtype=float) + np.asarray(rdm1, dtype=float).T)
    trace = float(np.trace(rdm1))
    if trace > 0.0 and abs(trace - nelec) < 1e-4:
        rdm1 = rdm1 * (nelec / trace)
    return rdm1


def _normalise_rdm1_pair(pair: np.ndarray, nelec: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrise and trace-normalise each spin block of an RDM pair."""
    out = []
    for rdm, target in zip(pair, nelec, strict=True):
        rdm = 0.5 * (np.asarray(rdm, dtype=float) + np.asarray(rdm, dtype=float).T)
        trace = float(np.trace(rdm))
        if trace > 0.0 and abs(trace - target) < 1e-4:
            rdm = rdm * (float(target) / trace)
        out.append(rdm)
    return out[0], out[1]


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
    sci_solver: str = QISKIT_ADDON_SOLVER,
    torch_sci_kwargs: dict | None = None,
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
        sci_solver: ``"qiskit_addon_sqd"`` (default here) for the
            aufbau-starting PySCF selected-CI path supplied by qiskit-addon-sqd,
            ``"torch_sci"`` for IBM/sci-solvers, or any other name from
            ``sci_solvers.energy_solvers.factory.AVAILABLE_SOLVERS``.  The
            public :class:`SQDSolver` default is ``"torch_sci"``; this low-level
            helper keeps the dependency-free addon path as its own default.
            A spin-dependent ``ham`` is supported only by the ``"torch_sci"``
            spin-orbital path; other solvers raise a clear error instead of
            silently using the spin-averaged Hamiltonian.
        torch_sci_kwargs: Constructor options forwarded to ``torch_sci`` only.

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

    # A spin-dependent Hamiltonian needs the exact (h1a, h1b, h2_spin) tensors.
    # torch_sci is the only solver with that path; everything else must fail
    # loudly rather than silently diagonalize the spin-averaged Hamiltonian.
    spin_resolved_subspace = bool(ham.is_spin_dependent)
    if spin_resolved_subspace:
        if sci_solver != "torch_sci":
            raise NotImplementedError(
                f"sci_solver={sci_solver!r} does not support a spin-dependent "
                "Hamiltonian (h1a/h1b/h2_spin). Use the default 'torch_sci' "
                "solver, or FCISolver for a classical unrestricted solve."
            )
        from embasi_qiskit_integration.sqd.spin_orbital import (
            make_torch_spin_orbital_solver,
        )

        solver_callable = make_torch_spin_orbital_solver(
            ham, spin_sq=spin_sq, torch_sci_kwargs=torch_sci_kwargs
        )
        spin_solver_backend = "torch_sci_spin"
    else:
        solver_callable = resolve_sci_solver(sci_solver, spin_sq, torch_sci_kwargs)
        spin_solver_backend = None

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
        sci_solver=solver_callable,
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
    rdm1 = np.asarray(result.rdm1, dtype=float)
    rdm2 = np.asarray(result.rdm2, dtype=float) if result.rdm2 is not None else None

    rdm1a: np.ndarray | None = None
    rdm1b: np.ndarray | None = None
    try:
        # qiskit-addon-sqd's make_rdm1s operates in float64; upcast a float32
        # MPS amplitude array before building the RDMs.
        state = result.sci_state
        amplitudes = np.asarray(state.amplitudes)
        if amplitudes.dtype != np.float64:
            from qiskit_addon_sqd.fermion import SCIState

            state = SCIState(
                amplitudes=amplitudes.astype(np.float64),
                ci_strs_a=state.ci_strs_a,
                ci_strs_b=state.ci_strs_b,
                norb=state.norb,
                nelec=state.nelec,
            )
        pair = np.asarray(state.rdm(rank=1, spin_summed=False), dtype=float)
    except Exception:  # pragma: no cover - state without a spin-resolved 1-RDM
        pair = None
    if pair is not None and pair.shape == (2, norb, norb):
        # The pair and the spin-summed RDM are built from the same SCI state;
        # a float32 solver (MPS has no float64) leaves them agreeing only to
        # float32 precision.  Make the spin-resolved pair authoritative:
        # qiskit's RDM builder runs in float64, so it is the better-conditioned
        # source once normalised.
        rdm1a, rdm1b = _normalise_rdm1_pair(pair, ham.nelec)
        rdm1 = rdm1a + rdm1b
    if rdm1a is None:
        rdm1 = _normalise_rdm1(rdm1, sum(ham.nelec))

    # Extract SQD subspace dimension: the actual number of configurations diagonalized.
    # Per qiskit-addon-sqd's docstring, "the full dimension of the SCI subspace is the
    # product of the dimensions of the individual spin sectors", enforced as an
    # invariant in SCIState: amplitudes.shape == (len(ci_strs_a), len(ci_strs_b)).
    # So the total is amplitudes.size.
    final_sqd_subspace_dimension: int | None = None
    try:
        final_sqd_subspace_dimension = int(result.sci_state.amplitudes.size)
    except Exception:  # pragma: no cover - diagnostic only, never fatal
        pass
    # Note: there is no "number of recovered determinants" field exposed by
    # qiskit-addon-sqd.configuration_recovery or SCIResult/SCIState (configuration
    # recovery operates on an internal bitstring matrix each iteration that is never
    # retained in the returned result). Leave this field unmapped in diagnostics.

    # Per-iteration total energies (electronic + core) for convergence checks.
    iteration_totals = [e + ham.e_core for e in iteration_energies]

    diagnostics = {
        "solver": "qiskit-addon-sqd",
        "sci_solver": (
            "torch-sci-spinless"
            if spin_solver_backend == "torch_sci_spin"
            else "torch-sci-solver"
            if sci_solver == "torch_sci"
            else "qiskit-addon-sqd:solve-sci"
            if sci_solver == QISKIT_ADDON_SOLVER
            else f"sci-solvers:{sci_solver}"
        ),
        "spin_solver_backend": spin_solver_backend,
        "spin_resolved_subspace": spin_resolved_subspace,
        "spin_averaged_hamiltonian": bool(ham.is_spin_dependent and not spin_resolved_subspace),
        "h1_spin_diff": (
            float(np.abs(np.asarray(ham.h1a) - np.asarray(ham.h1b)).max())
            if ham.is_spin_dependent
            else 0.0
        ),
        "h2_spin_present": bool(ham.h2_spin is not None),
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
        "final_sqd_subspace_dimension": final_sqd_subspace_dimension,
    }
    return SolverResult(
        energy=energy,
        rdm1=rdm1,
        rdm2=rdm2,
        rdm1a=rdm1a,
        rdm1b=rdm1b,
        diagnostics=diagnostics,
    )
