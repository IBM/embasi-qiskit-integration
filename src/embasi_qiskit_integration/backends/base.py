# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The code-agnostic seam between the embedding workflow and a QM code.

Everything the workflow needs from the QM code beyond what EmbASI's
``ProjectionEmbedding`` hands back goes through the two protocols here:

* :class:`AOBasis` -- metadata about the AO basis (which atom each function sits on,
  its shell label), used to pick fragment AOs and AVAS targets.
* :class:`AOIntegrals` -- AO-basis operators and integrals (overlap, ``hcore``, mean-field
  potentials at a given density, ERIs), used by the downfold.  It is also an
  :class:`AOBasis`, so one backend object serves both.

A backend advertises what it can actually supply in ``capabilities``; a path that needs
something a backend lacks fails up front through :func:`require_capability` rather than
producing a wrong number.  PySCF (:mod:`.pyscf`) supplies everything.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

# A density argument to the `veff_*` accessors: either the spin-summed AO matrix or a
# genuine `(alpha, beta)` pair.  The pair matters for a KS low level, whose xc is
# nonlinear in the spin densities -- see `ProjectionEmbeddingAdapter._dm_a_for_veff`.
DensityArg = np.ndarray | tuple[np.ndarray, np.ndarray]

# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #
#: AO-basis 4-index ERIs for arbitrary orbitals (``eri_mo``/``eri_mo_mixed``).
CAP_ERI = "eri"
#: Hartree-Fock J/K at an arbitrary density (``veff_hf``, ``veff_uhf``, ``get_k``).
CAP_EXACT_EXCHANGE = "exact_exchange"
#: A bare one-electron ``hcore = T + V_ext`` separate from the Hartree+xc potential.
CAP_HCORE = "hcore"
#: Mean-field potentials (``veff_ll``/``veff_hl``) at an arbitrary density, cheaply.
CAP_ARBITRARY_DM_VEFF = "arbitrary_dm_veff"
#: The low-level electronic energy at an arbitrary density (``energy_elec_ll``).
CAP_ENERGY_ELEC = "energy_elec"
#: A minimal-reference-basis (MINAO) cross-overlap projector, for AVAS.
CAP_MINAO_PROJECTOR = "minao_projector"
#: Density-fitted ERIs.
CAP_DENSITY_FIT = "density_fit"


class MissingCapabilityError(NotImplementedError):
    """The selected QM backend cannot supply what this path needs."""


def require_capability(backend: Any, capability: str, what: str) -> None:
    """Raise :class:`MissingCapabilityError` if ``backend`` lacks ``capability``.

    A backend that declares no ``capabilities`` at all (a test stand-in) is let through:
    the check exists to stop a *real* backend that knows it cannot supply a quantity, not
    to force every stub to enumerate what it fakes.
    """
    caps = getattr(backend, "capabilities", None)
    if caps is None or capability in caps:
        return
    raise MissingCapabilityError(
        f"{what} needs the '{capability}' capability, which the "
        f"{type(backend).__name__} backend does not provide "
        f"(it provides: {', '.join(sorted(caps)) or 'none'})"
    )


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
@runtime_checkable
class AOBasis(Protocol):
    """Metadata about the AO basis the embedding's matrices are expressed in."""

    @property
    def nao(self) -> int: ...

    def ao_atom_map(self) -> np.ndarray:
        """``(nao,)`` int array: the atom index each AO is centred on.

        Atom indices are positions in the (region-1-first) atom list the workflow hands
        the QM code, so they line up with ``active_atoms`` after the reorder.  AOs of one
        atom are assumed contiguous.
        """

    def ao_labels(self) -> list[tuple[int, str, str, str]]:
        """One ``(atom_index, element, shell, m_label)`` per AO, e.g. ``(0, "O", "2p", "x")``."""

    def search_ao_label(self, labels: Sequence[str]) -> np.ndarray:
        """AO indices matching any of ``labels`` (``"Fe 3d"``, ``"C 2p"``, ...)."""

    def describe(self) -> dict[str, Any]:
        """Identity of the system and method, for state fingerprints.

        Keys read by the fingerprint: ``basis``, ``natm``, ``charge``, ``spin``,
        ``atom_coords`` (an ``(natm, 3)`` array, hashed rather than stored),
        ``atom_symbols``, and optionally ``xc_hl`` / ``density_fit``.
        """


class AOIntegrals(AOBasis, Protocol):
    """AO-basis quantities the adapter needs beyond ProjectionEmbedding."""

    def overlap(self) -> np.ndarray: ...
    def hcore(self) -> np.ndarray: ...

    def veff_hl(self, dm: DensityArg) -> np.ndarray:
        """Effective potential of the *high-level* calculator, at a 2-occupancy dm.

        Must match ``calc_base_hl``.  Retained for callers that want the
        high-level mean field explicitly; the :attr:`h_emb` inverse uses
        :meth:`veff_ll` instead -- see that method.

        ``dm`` may be an ``(alpha, beta)`` pair; see :meth:`veff_ll` for why that
        matters and what the return must be in that case.
        """

    def veff_ll(self, dm: DensityArg) -> np.ndarray:
        """Effective potential of the *low-level* calculator, at a 2-occupancy dm.

        Must match ``calc_base_ll``: it exists only to undo the mean-field
        contribution EmbASI folded into ``F_emb``.  ``F_emb`` is assembled from
        the **A_LL** one-electron blocks (see
        :meth:`ProjectionEmbeddingAdapter._assemble_fock_a_only`), so the mean
        field baked into it is the *low-level* one, and that is what
        :attr:`ProjectionEmbeddingAdapter.h_emb` must subtract back off.

        ``dm`` is either a spin-summed ``(nao, nao)`` matrix or an
        ``(alpha, beta)`` pair -- the adapter passes the pair whenever it has one
        (:attr:`ProjectionEmbeddingAdapter._dm_a_for_veff`), because an
        unrestricted KS ``get_veff`` handed a summed matrix silently substitutes
        ``d/2`` for both channels and loses the spin polarisation.  The **return**
        is always a single ``(nao, nao)`` operator: a spin-resolved backend must
        average its two channels, not sum them (a potential is per-electron).
        """

    def veff_ll_spin(self, dm: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Each spin channel's low-level effective potential at an ``(alpha, beta)`` pair."""

    def energy_elec_ll(self, dm: np.ndarray) -> float:
        """The low-level calculator's electronic energy at a 2-occupancy ``dm`` (no E_nuc)."""

    def veff_hf(self, dm: np.ndarray) -> np.ndarray:
        """Hartree-Fock effective potential J[dm] - K[dm]/2, at a 2-occupancy dm.

        Used for the inactive-core downfold, independently of the high level.
        """

    def get_k(self, dm: np.ndarray) -> np.ndarray:
        """Bare HF exchange operator K[dm] (no J, no 0.5 factor), at a 2-occupancy dm.

        Used only by the APC active-space ranking (:meth:`ProjectionEmbeddingAdapter.
        build_orbitals_apc_concentric`), which needs K's diagonal separately from the
        combined ``veff_hf`` (King & Gagliardi, *J. Chem. Theory Comput.* **2021**, 17,
        7. eq. 18: ``(ia|ia) ~ 0.5 K_aa``).
        """

    def veff_uhf(self, dm_a: np.ndarray, dm_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Unrestricted HF mean field of a spin-resolved 1-occupancy density pair.

        Returns ``(v_alpha, v_beta, e_two_body)`` where
        ``v_sigma = J[dm_a + dm_b] - K[dm_sigma]`` and
        ``e_two_body = 0.5 tr[d J[d]] - 0.5 (tr[dm_a K[dm_a]] + tr[dm_b K[dm_b]])``
        with ``d = dm_a + dm_b``.

        This is the frozen-core fold the *per-spin* downfold needs, and it is **not**
        obtainable from :meth:`veff_hf`.  ``veff_hf`` is the restricted ``J - K/2``:
        correct only when the two channels share an orbital set, because Coulomb is a
        functional of the total density but **exchange couples like spins only**.
        Using the spin-averaged form for two genuinely different cores understates the
        channel splitting -- measured (ERI-only ground truth, 6-31g, valence-like cores
        overlapping by 0.34) ``max|v_avg - v_alpha| = 1.09 Ha`` and **+0.78 Ha
        (+492 kcal/mol)** on the CAS total.  It vanishes only as the two cores coincide,
        which is why a deep-1s frozen core (overlap 0.9999998) hides it at 0.02
        kcal/mol.

        The energy is returned alongside the potentials rather than left to the caller
        because the exchange self-interaction does not factor out of
        ``0.5 tr[d v]``: that expression is only valid for the restricted ``J - K/2``.
        """

    def eri_mo(self, mo_coeff: np.ndarray) -> np.ndarray:
        """Chemist-notation (pq|rs), shape (nmo,)*4, for the given MO block."""

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        """Mixed-spin ``(p_a q_a | r_b s_b)``, shape ``(nmo,)*4``."""

    def energy_nuc(self) -> float: ...

    def layer_scf_status(self, layer: Any) -> dict[str, Any] | None:
        """``{"converged", "max_cycle", "e_tot", "conv_tol"}`` of an EmbASI layer's SCF.

        ``None`` when the layer's calculator does not expose a convergence flag; the
        guard then lets it through (EmbASI itself gives no flag).
        """

    def hl_scf_status(self) -> dict[str, Any]:
        """Convergence settings/state of the high-level mean field, for diagnostics."""

    def layer_footing(self, layer: Any) -> tuple[np.ndarray, float]:
        """``(h_core, E_nuc)`` of the frame an EmbASI layer integrates on."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def as_ao_basis(obj: Any) -> AOBasis:
    """Return ``obj`` as an :class:`AOBasis`.

    A backend passes straight through.  A bare PySCF ``Mole`` (anything with
    ``aoslice_by_atom``) is wrapped in :class:`~.pyscf.PySCFBasis`, so existing callers
    that still hand a ``mol`` keep working.
    """
    if hasattr(obj, "ao_atom_map"):
        return obj
    if hasattr(obj, "aoslice_by_atom"):
        from embasi_qiskit_integration.backends.pyscf import PySCFBasis

        return PySCFBasis(obj)
    raise TypeError(
        f"{type(obj).__name__} is not an AO basis: it needs ao_atom_map() (or be a PySCF Mole)"
    )


def atom_ao_indices(ao_atom_map: np.ndarray, atoms: Iterable[int]) -> np.ndarray:
    """AO indices of ``atoms``, concatenated in the order the atoms are given."""
    ao_atom_map = np.asarray(ao_atom_map)
    ranges = [np.flatnonzero(ao_atom_map == a) for a in atoms]
    return np.hstack(ranges).astype(int) if ranges else np.empty(0, dtype=int)
