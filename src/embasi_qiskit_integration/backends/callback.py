# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic callback backend: every integral from a Hamiltonian evaluation at a density.

EmbASI drives any QM code it supports through the same two operations: run a layer, and
read back the matrices it exported (``overlap``, ``hamiltonian_total``, ``total_energy``)
-- for FHI-aims through ASI callbacks, for PySCF as attributes.  A non-SCF run
(``AtomsEmbed.run_noscf(dm)``) evaluates the Hamiltonian at an *injected* density, and
everything :class:`~.base.AOIntegrals` needs follows from that one operation:

* ``S``                       -- exported by every run;
* ``hcore = H[0]``             -- the Hamiltonian at zero density;
* ``veff(dm) = H[dm] - H[0]``  -- for the low- and high-level calculators alike;
* ``E_nuc = E[0]``, ``E_elec(dm) = E[dm] - E[0]``;
* ``J[d]``, ``K[d]`` from one *unrestricted HF* evaluation at the pair ``(d, 0)``:
  ``v_beta = J[d]`` and ``v_alpha = J[d] - K[d]``, so ``K = v_beta - v_alpha``.

Nothing here knows which QM code sits underneath.  The parity test drives it through
EmbASI's PySCF adapter and compares it with :class:`~.pyscf.PySCFIntegrals`; because the
numbers come back through EmbASI's run/extract path rather than direct PySCF calls, that
is a genuine cross-check.

Four-index ERIs are **not** derivable cheaply this way (they would cost one evaluation
per orbital pair), so ``eri_mo`` / ``eri_mo_mixed`` come from an optional code-specific
``eri_provider``; without one the backend does not declare ``eri`` and the WF-in-DFT
downfold refuses up front.  Each evaluation is a full QM-code invocation, so the
backend also does not declare ``arbitrary_dm_veff`` unless told the code is cheap enough
for the reference protocol's SCF-in-SCF loop.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from embasi_qiskit_integration.backends.base import (
    CAP_ARBITRARY_DM_VEFF,
    CAP_ENERGY_ELEC,
    CAP_ERI,
    CAP_EXACT_EXCHANGE,
    CAP_HCORE,
    DensityArg,
    MissingCapabilityError,
    atom_ao_indices,
)

# EmbASI reports energies in eV with this inline factor (see projection_embedding_adapter).
_EMBASI_HA2EV = 27.211384500


@dataclass(frozen=True)
class Evaluation:
    """One non-SCF evaluation: the Hamiltonian per spin channel, the overlap, the energy."""

    hamiltonian: tuple[np.ndarray, ...]  # one (nao, nao) block per spin channel
    overlap: np.ndarray
    energy: float  # Hartree


class LayerEvaluator:
    """Evaluates one calculator at a given density through a standalone EmbASI layer.

    ``calc`` is an ASE calculator EmbASI has a ``QMCodeAdapter`` for.  The layer is the
    same ``AtomsEmbed`` EmbASI uses inside ``ProjectionEmbedding``, built on the whole
    system with no embedding mask, so its basis is the supersystem one.  Every call is a
    deep copy of the calculator (EmbASI's own ``run_noscf``), so evaluations never leak
    state into each other or into the calculators handed to ``ProjectionEmbedding``.
    """

    def __init__(self, atoms: Any, calc: Any, *, n_spin: int, label: str, charge: int = 0):
        from embasi.atoms_embedding_asi import AtomsEmbed

        self.n_spin = int(n_spin)
        self.label = label
        self._layer = AtomsEmbed(atoms, calc, embed_mask=None, outdir=label)
        # EmbASI writes the layer's total charge onto the calculator before every run
        # (`set_qm_total_charge`), defaulting to 0.  Left unset, a charged system is
        # re-run as neutral -- and for an odd charge PySCF refuses the Mole outright
        # (electron count and spin no longer have the same parity).
        self._layer.input_total_charge = float(charge)

    def _wrap(self, dm: DensityArg) -> Any:
        from embasi.ks_array import SpinKpointArray

        if self.n_spin == 2:
            if isinstance(dm, tuple):
                a, b = (np.ascontiguousarray(np.asarray(x, dtype=float)) for x in dm)
            else:
                a = b = np.ascontiguousarray(0.5 * np.asarray(dm, dtype=float))
            return SpinKpointArray({(0, 0): a, (1, 0): b}, n_spin=2, n_kpoints=1)
        total = np.asarray(dm[0]) + np.asarray(dm[1]) if isinstance(dm, tuple) else dm
        block = np.ascontiguousarray(np.asarray(total, dtype=float))
        return SpinKpointArray({(0, 0): block}, n_spin=1, n_kpoints=1)

    def __call__(self, dm: DensityArg) -> Evaluation:
        layer = self._layer
        # EmbASI caches the eigenvalue sum on the layer (`if not hasattr(self, "ev_sum")`)
        # for its non-SCF energy correction; this layer is reused across densities, so a
        # cached sum would belong to the *previous* evaluation.  (Inert for PySCF, which
        # needs no correction; live for FHI-aims-like codes.)
        if hasattr(layer, "ev_sum"):
            del layer.ev_sum
        layer.run_noscf(dm_in=self._wrap(dm))
        ham = layer.hamiltonian_total
        blocks = tuple(
            _real_block(ham[s, 0], f"{self.label} H[spin {s}]") for s in range(self.n_spin)
        )
        overlap = _real_block(layer.overlap[0, 0], f"{self.label} S")
        # A non-SCF run's energy must be E[dm], not the eigenvalue-sum energy of the
        # first diagonalisation: EmbASI corrects it into `ev_corr_total_energy` for codes
        # that need it (FHI-aims) and sets it equal to `total_energy` otherwise.
        energy = getattr(layer, "ev_corr_total_energy", None)
        if energy is None:
            energy = layer.total_energy
        return Evaluation(blocks, overlap, float(energy) / _EMBASI_HA2EV)

    @property
    def basis_atoms(self) -> np.ndarray:
        return np.asarray(self._layer.basis_atoms, dtype=int)


def _real_block(block: Any, what: str) -> np.ndarray:
    arr = np.asarray(block)
    if np.iscomplexobj(arr):
        if np.abs(arr.imag).max(initial=0.0) > 1e-9:
            raise ValueError(f"{what} has a nonzero imaginary part; multi-k is not supported")
        arr = arr.real
    return np.ascontiguousarray(arr, dtype=float)


# An evaluator is anything mapping a density to an Evaluation; LayerEvaluator is the
# EmbASI one, tests inject their own.
Evaluator = Callable[[DensityArg], Evaluation]
# (mo_coeff,) -> (pq|rs) and (mo_a, mo_b) -> (p_a q_a|r_b s_b)
EriProvider = Any


class CallbackIntegrals:
    """:class:`~.base.AOIntegrals` from Hamiltonian evaluations; see the module docstring.

    Args:
        eval_ll: evaluator for the low-level calculator (``calc_base_ll``).
        eval_hl: evaluator for the high-level calculator (``calc_base_hl``).
        eval_hf: evaluator for an *unrestricted Hartree-Fock* calculator on the same
            system and basis, used for ``veff_hf`` / ``get_k`` / ``veff_uhf``.  ``None``:
            the code cannot do exact exchange, and ``exact_exchange`` is not declared.
        nao: number of AO basis functions (sizes the zero-density evaluation).
        basis_atoms: AO-to-atom map, ``(nao,)``; ``None`` if the code reports none.
        ao_labels: ``(atom, element, shell, m)`` per AO, if the code reports them.
        describe: fingerprint fields (``basis``, ``natm``, ``charge``, ``spin``,
            ``atom_coords``, ``atom_symbols``, and optionally ``xc_hl``).
        ll_spin_resolved: whether the low-level calculator has two spin channels.
        eri_provider: object with ``eri_mo(c)`` and ``eri_mo_mixed(ca, cb)``, or
            ``None`` (no ``eri`` capability).
        cheap_evaluations: declare ``arbitrary_dm_veff`` (evaluations are cheap enough to
            sit inside an SCF loop -- true for an in-process code, not for a file-driven
            one).
        scf_status: ``layer -> dict | None`` for EmbASI layers (``None``: no flag).
        hl_status: ``() -> dict`` for the high-level mean field.
        layer_footing: ``layer -> (hcore, E_nuc)`` for an EmbASI layer; ``None`` uses
            this backend's own supersystem ``hcore``/``E_nuc`` (right since EmbASI's
            ``ghosts=0``: every layer integrates on the supersystem frame).
    """

    def __init__(
        self,
        eval_ll: Evaluator,
        eval_hl: Evaluator,
        eval_hf: Evaluator | None,
        *,
        nao: int,
        basis_atoms: np.ndarray | None = None,
        ao_labels: list[tuple[int, str, str, str]] | None = None,
        describe: dict[str, Any] | None = None,
        ll_spin_resolved: bool = False,
        eri_provider: EriProvider | None = None,
        cheap_evaluations: bool = False,
        scf_status: Callable[[Any], dict[str, Any] | None] | None = None,
        hl_status: Callable[[], dict[str, Any]] | None = None,
        layer_footing: Callable[[Any], tuple[np.ndarray, float]] | None = None,
    ):
        self._eval_ll = eval_ll
        self._eval_hl = eval_hl
        self._eval_hf = eval_hf
        self._nao = int(nao)
        self._basis_atoms = None if basis_atoms is None else np.asarray(basis_atoms, int)
        if self._basis_atoms is not None and self._basis_atoms.shape != (self._nao,):
            raise ValueError(
                f"basis_atoms has shape {self._basis_atoms.shape}, expected ({self._nao},)"
            )
        self._ao_labels = ao_labels
        self._describe = dict(describe or {})
        self._ll_spin_resolved = bool(ll_spin_resolved)
        self._eri = eri_provider
        self._cheap = bool(cheap_evaluations)
        self._scf_status = scf_status
        self._hl_status = hl_status
        self._layer_footing = layer_footing
        self._zero: dict[str, Evaluation] = {}

    # ---------------- capabilities ---------------- #
    @property
    def capabilities(self) -> frozenset[str]:
        caps = {CAP_HCORE, CAP_ENERGY_ELEC}
        if self._eval_hf is not None:
            caps.add(CAP_EXACT_EXCHANGE)
        if self._eri is not None:
            caps.add(CAP_ERI)
        if self._cheap:
            caps.add(CAP_ARBITRARY_DM_VEFF)
        return frozenset(caps)

    @property
    def ll_spin_resolved(self) -> bool:
        return self._ll_spin_resolved

    # ---------------- zero-density reference ---------------- #
    def _at_zero(self, which: str) -> Evaluation:
        """``H[0]`` / ``E[0]`` for one evaluator, cached (they do not depend on dm)."""
        if which not in self._zero:
            ev = {"ll": self._eval_ll, "hl": self._eval_hl, "hf": self._eval_hf}[which]
            if ev is None:
                raise MissingCapabilityError("this backend has no HF evaluator")
            zero = np.zeros((self._nao, self._nao))
            out = ev((zero, zero)) if which == "hf" else ev(zero)
            if out.overlap.shape != (self._nao, self._nao):
                raise ValueError(
                    f"the {which} evaluator returned a {out.overlap.shape} overlap for a "
                    f"{self._nao}-function basis"
                )
            self._zero[which] = out
        return self._zero[which]

    # ---------------- AOBasis ---------------- #
    @property
    def nao(self) -> int:
        return self._nao

    def ao_atom_map(self) -> np.ndarray:
        if self._basis_atoms is None:
            raise ValueError("this backend was built without an AO-to-atom map")
        return self._basis_atoms

    def atom_ao_indices(self, atoms) -> np.ndarray:
        return atom_ao_indices(self.ao_atom_map(), list(atoms))

    def ao_labels(self) -> list[tuple[int, str, str, str]]:
        if self._ao_labels is None:
            raise MissingCapabilityError("this backend's QM code reports no AO labels")
        return list(self._ao_labels)

    def search_ao_label(self, labels: Sequence[str]) -> np.ndarray:
        """AO indices whose ``"<element> <shell>"`` matches any of ``labels``."""
        wanted = {" ".join(label.split()) for label in labels}
        return np.array(
            [
                i
                for i, (_a, el, shell, _m) in enumerate(self.ao_labels())
                if f"{el} {shell}" in wanted
            ],
            dtype=int,
        )

    def describe(self) -> dict[str, Any]:
        return dict(self._describe)

    # ---------------- one-electron / energies ---------------- #
    def overlap(self) -> np.ndarray:
        return self._at_zero("ll").overlap

    def hcore(self) -> np.ndarray:
        h0 = self._at_zero("ll").hamiltonian
        return h0[0]

    def energy_nuc(self) -> float:
        return self._at_zero("ll").energy

    def energy_elec_ll(self, dm: np.ndarray) -> float:
        return self._eval_ll(np.asarray(dm)).energy - self._at_zero("ll").energy

    # ---------------- mean-field potentials ---------------- #
    def _veff_channels(self, which: str, dm: DensityArg) -> tuple[np.ndarray, ...]:
        ev = {"ll": self._eval_ll, "hl": self._eval_hl}[which]
        h = ev(dm).hamiltonian
        h0 = self._at_zero(which).hamiltonian
        return tuple(hs - h0s for hs, h0s in zip(h, h0, strict=True))

    @staticmethod
    def _mean(channels: tuple[np.ndarray, ...]) -> np.ndarray:
        # A potential is per-electron: average the channels, never sum them.
        return channels[0] if len(channels) == 1 else 0.5 * (channels[0] + channels[1])

    def veff_ll(self, dm: DensityArg) -> np.ndarray:
        return self._mean(self._veff_channels("ll", dm))

    def veff_hl(self, dm: DensityArg) -> np.ndarray:
        return self._mean(self._veff_channels("hl", dm))

    def veff_ll_spin(self, dm: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        channels = self._veff_channels("ll", dm)
        return (channels[0], channels[0]) if len(channels) == 1 else (channels[0], channels[1])

    # ---------------- exact exchange ---------------- #
    def _hf_pair(self, dm_a: np.ndarray, dm_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(v_alpha, v_beta)`` of the unrestricted-HF evaluator at ``(dm_a, dm_b)``."""
        if self._eval_hf is None:
            raise MissingCapabilityError("this backend's QM code has no exact exchange")
        h = self._eval_hf((np.asarray(dm_a), np.asarray(dm_b))).hamiltonian
        h0 = self._at_zero("hf").hamiltonian
        if len(h) != 2:
            raise ValueError("the HF evaluator must be unrestricted (two spin channels)")
        return h[0] - h0[0], h[1] - h0[1]

    def get_jk(self, dm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(J[dm], K[dm])`` from one UHF evaluation at ``(dm, 0)``."""
        v_a, v_b = self._hf_pair(dm, np.zeros_like(dm))
        return v_b, v_b - v_a

    def get_k(self, dm: np.ndarray) -> np.ndarray:
        return self.get_jk(np.asarray(dm))[1]

    def veff_hf(self, dm: np.ndarray) -> np.ndarray:
        # Restricted J - K/2 at a 2-occupancy dm == the UHF potential at (dm/2, dm/2).
        d = 0.5 * np.asarray(dm)
        v_a, _v_b = self._hf_pair(d, d)
        return v_a

    def veff_uhf(self, dm_a, dm_b) -> tuple[np.ndarray, np.ndarray, float]:
        d_a, d_b = np.asarray(dm_a), np.asarray(dm_b)
        v_a, v_b = self._hf_pair(d_a, d_b)
        # e_two_body = 0.5 tr[d_a v_a] + 0.5 tr[d_b v_b]  (each v_sigma = J - K_sigma)
        e_two_body = float(0.5 * (np.einsum("ij,ji->", d_a, v_a) + np.einsum("ij,ji->", d_b, v_b)))
        return v_a, v_b, e_two_body

    # ---------------- ERIs ---------------- #
    def eri_mo(self, mo_coeff: np.ndarray) -> np.ndarray:
        if self._eri is None:
            raise MissingCapabilityError("this backend's QM code provides no ERIs")
        return self._eri.eri_mo(mo_coeff)

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        if self._eri is None:
            raise MissingCapabilityError("this backend's QM code provides no ERIs")
        return self._eri.eri_mo_mixed(mo_a, mo_b)

    # ---------------- SCF status / footing ---------------- #
    def layer_scf_status(self, layer: Any) -> dict[str, Any] | None:
        return None if self._scf_status is None else self._scf_status(layer)

    def hl_scf_status(self) -> dict[str, Any]:
        if self._hl_status is None:
            return {"converged": None, "conv_tol": None, "max_cycle": None}
        return self._hl_status()

    def layer_footing(self, layer: Any) -> tuple[np.ndarray, float]:
        if self._layer_footing is not None:
            return self._layer_footing(layer)
        return self.hcore(), self.energy_nuc()
