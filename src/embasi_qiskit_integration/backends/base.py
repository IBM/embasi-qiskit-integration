# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The code-agnostic seam between the embedding workflow and a QM code.

:class:`AOBasis` describes the AO basis (atom and shell of each function);
:class:`AOIntegrals` adds the operators and integrals the downfold needs.  A backend
lists what it can supply in ``capabilities``, and a path needing something missing fails
up front through :func:`require_capability`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

# A spin-summed AO density or an (alpha, beta) pair (see `_dm_a_for_veff`).
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

    A backend without ``capabilities`` (a test stand-in) is let through.
    """
    caps = getattr(backend, "capabilities", None)
    if caps is None or capability in caps:
        return
    raise MissingCapabilityError(
        f"{what} needs the '{capability}' capability, which the "
        f"{type(backend).__name__} backend does not provide "
        f"(it provides: {', '.join(sorted(caps)) or 'none'})"
    )


def backend_method(backend: Any, name: str, fallback: Any = None) -> Any:
    """``backend.<name>``, or ``fallback`` for a test stand-in that does not implement it.

    Only objects without ``capabilities`` get the fallback; a real backend missing a
    protocol method raises instead of silently getting PySCF's behaviour.
    """
    method = getattr(backend, name, None)
    if callable(method):
        return method
    if getattr(backend, "capabilities", None) is None:
        return fallback
    raise MissingCapabilityError(
        f"the {type(backend).__name__} backend declares capabilities but does not "
        f"implement {name}() of the AOIntegrals protocol"
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
        """``(nao,)`` atom index of each AO, in the region-1-first atom order."""

    def ao_labels(self) -> list[tuple[int, str, str, str]]:
        """One ``(atom_index, element, shell, m_label)`` per AO, e.g. ``(0, "O", "2p", "x")``."""

    def search_ao_label(self, labels: Sequence[str]) -> np.ndarray:
        """AO indices matching any of ``labels`` (``"Fe 3d"``, ``"C 2p"``, ...)."""

    def describe(self) -> dict[str, Any]:
        """Fingerprint fields: ``basis``, ``natm``, ``charge``, ``spin``, ``atom_coords``,
        ``atom_symbols``, optionally ``xc_hl`` / ``density_fit``."""


class AOIntegrals(AOBasis, Protocol):
    """AO-basis quantities the adapter needs beyond ProjectionEmbedding."""

    def overlap(self) -> np.ndarray: ...
    def hcore(self) -> np.ndarray: ...

    def veff_hl(self, dm: DensityArg) -> np.ndarray:
        """Effective potential of ``calc_base_hl`` at a 2-occupancy dm (see :meth:`veff_ll`)."""

    def veff_ll(self, dm: DensityArg) -> np.ndarray:
        """Effective potential of ``calc_base_ll`` at a 2-occupancy dm.

        It undoes the low-level mean field EmbASI folded into ``F_emb`` (see
        :attr:`ProjectionEmbeddingAdapter.h_emb`).  ``dm`` may be an ``(alpha, beta)``
        pair; the return is one ``(nao, nao)`` operator, the mean of the spin channels.
        """

    def veff_ll_spin(self, dm: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Each spin channel's low-level effective potential at an ``(alpha, beta)`` pair."""

    def energy_elec_ll(self, dm: np.ndarray) -> float:
        """The low-level calculator's electronic energy at a 2-occupancy ``dm`` (no E_nuc)."""

    def veff_hf(self, dm: np.ndarray) -> np.ndarray:
        """HF potential ``J[dm] - K[dm]/2`` at a 2-occupancy dm (inactive-core downfold)."""

    def get_k(self, dm: np.ndarray) -> np.ndarray:
        """Bare exchange ``K[dm]`` at a 2-occupancy dm (APC ranking, King & Gagliardi 2021)."""

    def veff_uhf(self, dm_a: np.ndarray, dm_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Unrestricted HF mean field ``(v_alpha, v_beta, e_two_body)`` of a 1-occupancy pair.

        ``v_s = J[dm_a + dm_b] - K[dm_s]``.  The per-spin frozen-core fold needs this:
        :meth:`veff_hf` is exact only when both channels share an orbital set, since
        exchange couples like spins only.
        """

    def eri_mo(self, mo_coeff: np.ndarray) -> np.ndarray:
        """Chemist-notation (pq|rs), shape (nmo,)*4, for the given MO block."""

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        """Mixed-spin ``(p_a q_a | r_b s_b)``, shape ``(nmo,)*4``."""

    def energy_nuc(self) -> float: ...

    def layer_scf_status(self, layer: Any) -> dict[str, Any] | None:
        """``{"converged", "max_cycle", "e_tot", "conv_tol"}`` of a layer's SCF, or ``None``."""

    def hl_scf_status(self) -> dict[str, Any]:
        """Convergence settings/state of the high-level mean field, for diagnostics."""

    def layer_footing(self, layer: Any) -> tuple[np.ndarray, float]:
        """``(h_core, E_nuc)`` of the frame an EmbASI layer integrates on."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def as_ao_basis(obj: Any) -> AOBasis:
    """``obj`` as an :class:`AOBasis`; a PySCF ``Mole`` is wrapped in :class:`~.pyscf.PySCFBasis`."""
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
