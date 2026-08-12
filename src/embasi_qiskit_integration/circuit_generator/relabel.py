# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Mode relabeling for SqDRIFT circuits."""

from __future__ import annotations

import importlib.util
from typing import Any


def relabel_available() -> bool:
    """True if the optional relabel solver stack (``pyomo`` + HiGHS) is importable.

    ``RelabelModes`` needs ``pyomo`` to build the MILP and a solver to solve it;
    :class:`NativeHighsSolverAdapter` uses pyomo's bundled appsi HiGHS interface,
    which requires ``highspy``. With either missing the pass cannot find a
    permutation (it warns and returns the circuit unchanged), so callers should
    check this before requesting ``optimize=True``.

    Probes with ``find_spec`` rather than importing, so the classical and replay
    paths never pay for pulling pyomo in. A broken or partially-installed package
    counts as unavailable: ``find_spec`` raises for those, and this gates a default,
    so it must answer rather than propagate.
    """
    for module in ("pyomo", "highspy"):
        try:
            if importlib.util.find_spec(module) is None:
                return False
        except (ImportError, ValueError):  # pragma: no cover - broken installs only
            return False
    return True


class NativeHighsSolverAdapter:
    """Solver adapter for ``RelabelModes`` backed by native appsi HiGHS.

    ``RelabelModes.find_permutation`` calls ``solver.solve(model)`` and then reads
    the solution back via ``pyomo.environ.value(model.y[i])``. The legacy
    ``pyomo.environ.SolverFactory("appsi_highs")`` wrapper defaults to
    ``load_solutions=True``, which raises ``RuntimeError`` whenever a solve returns
    no feasible solution -- and it does so even for the feasible randomizations in
    a batch, leaving every circuit unpermuted.

    This adapter uses the native ``pyomo.contrib.appsi.solvers.highs.Highs``
    interface with ``load_solution=False`` so an infeasible/unbounded solve does
    not raise inside pyomo. We then:

    - load the solution into the model on success, so the pass can read ``model.y``;
    - re-raise ``RuntimeError`` on an infeasible solve, which ``RelabelModes``
      already catches to fall back to "no permutation" gracefully.

    The net effect: feasible randomizations yield real mode permutations, while
    genuinely-infeasible ones (e.g. a degenerate tiny active space whose sampled
    terms all reduce to identity) still degrade cleanly instead of poisoning the
    whole batch.

    Solves are pinned to a single thread (see :meth:`solve`). That removes one
    source of run-to-run variation but **not all of them** -- see
    :mod:`~embasi_qiskit_integration.circuit_generator.sqdrift`'s
    ``canonicalize_permutation``, which breaks the remaining ties downstream.
    """

    def __init__(self, time_limit: float | None = None, deterministic: bool = True) -> None:
        """Configure each solve.

        Args:
            time_limit: optional wall-clock limit (seconds) per solve. A solve that
                hits it yields no permutation, and the circuit is kept unpermuted.
            deterministic: pin HiGHS to a single thread and disable its concurrent
                MIP mode (default).

                The excitation-span MILP is highly degenerate -- many mode orderings
                achieve the same optimal span -- so *which* optimum comes back is not
                determined by the model alone. Pinning the threads removes the
                load-dependent part of that. It does not make HiGHS a pure function
                of its input: solving the identical model repeatedly in one process
                was observed to return one optimum twice and then a different one, so
                the solver carries state across instances. The permutation is
                therefore canonicalized after the solve rather than relied upon to
                come back stable; this flag only narrows the variation.

                Set False to let HiGHS parallelize an individual solve.
        """
        self._time_limit = time_limit
        self._deterministic = deterministic

    def solve(self, model: Any) -> Any:
        """Solve ``model`` with native HiGHS; load vars on success, raise if infeasible."""
        import logging

        from pyomo.contrib.appsi.solvers.highs import Highs

        opt = Highs()
        opt.config.load_solution = False
        opt.config.log_level = logging.DEBUG
        if self._time_limit:
            opt.highs_options["time_limit"] = float(self._time_limit)
        if self._deterministic:
            opt.highs_options["threads"] = 1
            opt.highs_options["parallel"] = "off"
        result = opt.solve(model)
        if result.best_feasible_objective is None:
            raise RuntimeError(f"no feasible solution (termination={result.termination_condition})")
        result.solution_loader.load_vars()
        return result
