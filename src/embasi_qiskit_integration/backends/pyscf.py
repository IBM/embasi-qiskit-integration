# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""PySCF backend: the default QM code behind the embedding workflow.

The only module allowed to build or reach into PySCF ``Mole`` / SCF objects for the
QM system itself: :func:`build_pyscf_backend` (registered as ``--qm_code pyscf``)
constructs the calculators, and everything else talks to :class:`PySCFIntegrals`
through the :class:`~embasi_qiskit_integration.backends.base.AOIntegrals` protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from embasi_qiskit_integration.backends.base import (
    CAP_ARBITRARY_DM_VEFF,
    CAP_DENSITY_FIT,
    CAP_ENERGY_ELEC,
    CAP_ERI,
    CAP_EXACT_EXCHANGE,
    CAP_HCORE,
    CAP_MINAO_PROJECTOR,
    DensityArg,
)
from embasi_qiskit_integration.backends.callback import CallbackIntegrals, LayerEvaluator
from embasi_qiskit_integration.backends.registry import QMBackend, register_backend


def is_spin_resolved(mf: Any) -> bool:
    """Whether a PySCF mean field is UHF/ROHF-shaped (two spin channels in ``get_veff``)."""
    from pyscf import scf as _scf

    return isinstance(mf, (_scf.uhf.UHF, _scf.rohf.ROHF))


def _layer_method(layer: Any) -> Any:
    """The PySCF SCF object behind an EmbASI layer, or ``None`` for another backend."""
    return getattr(getattr(getattr(layer, "atoms", None), "calc", None), "method", None)


def pyscf_layer_scf_status(layer: Any) -> dict[str, Any] | None:
    """SCF status of the PySCF method behind an EmbASI layer; ``None`` if it has none."""
    mf = _layer_method(layer)
    if mf is None or not hasattr(mf, "converged"):
        return None
    return {
        "converged": bool(mf.converged),
        "max_cycle": getattr(mf, "max_cycle", 1),
        "e_tot": getattr(mf, "e_tot", None),
        "conv_tol": getattr(mf, "conv_tol", None),
        "hint": _PYSCF_SCF_HINT,
    }


# Remedies for a stalled SCF that only exist on the PySCF backend (`--scf_newton` is
# refused for any other `--qm_code`).
_PYSCF_SCF_HINT = (
    "Try --scf_newton True (second-order SCF, for stalls at small HOMO-LUMO gaps) "
    "or a larger --scf_max_cycle."
)


def pyscf_layer_footing(layer: Any) -> tuple[np.ndarray, float]:
    """``(T + V_nuc, E_nuc)`` of the ``Mole`` an EmbASI PySCF layer integrates on."""
    mol = layer.atoms.calc.mol
    hcore = np.asarray(mol.intor("int1e_kin") + mol.intor("int1e_nuc"))
    return hcore, float(mol.energy_nuc())


class PySCFBasis:
    """:class:`~embasi_qiskit_integration.backends.base.AOBasis` over a PySCF ``Mole``."""

    def __init__(self, mol):
        self.mol = mol

    @property
    def nao(self) -> int:
        return int(self.mol.nao)

    def ao_atom_map(self) -> np.ndarray:
        aoslice = self.mol.aoslice_by_atom()
        out = np.empty(int(aoslice[-1, 3]) if len(aoslice) else 0, dtype=int)
        for atom, (_s0, _s1, a0, a1) in enumerate(aoslice):
            out[a0:a1] = atom
        return out

    def atom_ao_indices(self, atoms: Iterable[int]) -> np.ndarray:
        """AO indices of ``atoms`` straight from ``aoslice_by_atom`` (no map round trip)."""
        aoslice = self.mol.aoslice_by_atom()
        ranges = [np.arange(aoslice[a, 2], aoslice[a, 3]) for a in atoms]
        return np.hstack(ranges).astype(int) if ranges else np.empty(0, dtype=int)

    def ao_labels(self) -> list[tuple[int, str, str, str]]:
        return [tuple(lab) for lab in self.mol.ao_labels(fmt=False)]

    def search_ao_label(self, labels: Sequence[str]) -> np.ndarray:
        return np.asarray(self.mol.search_ao_label(list(labels)), dtype=int)

    def describe(self) -> dict[str, Any]:
        mol = self.mol
        return {
            "basis": str(mol.basis),
            "natm": int(mol.natm),
            "charge": int(mol.charge),
            "spin": int(mol.spin),
            "atom_coords": np.asarray(mol.atom_coords(), dtype=float),
            "atom_symbols": [mol.atom_symbol(i) for i in range(mol.natm)],
        }

    def valence_ao_projector(
        self, ao_labels: Sequence[str], atoms: Iterable[int] | None = None, minao: str = "minao"
    ) -> np.ndarray:
        """AVAS projector ``M = S_21^T S_22^{-1} S_21`` onto MINAO target AOs.

        See :func:`embasi_qiskit_integration.selectors.avas_ao_projector`.
        """
        import scipy.linalg
        from pyscf import gto

        mol = self.mol
        pmol = mol.copy()
        pmol.atom = mol._atom
        pmol.unit = "B"
        pmol.symmetry = False
        pmol.basis = minao
        pmol.build(False, False)
        ref = PySCFBasis(pmol)

        targets = ref.search_ao_label(list(ao_labels))
        if atoms is not None:
            atoms = list(atoms)
            targets = np.intersect1d(targets, ref.atom_ao_indices(atoms))
        if targets.size == 0:
            ref_ao = ref.atom_ao_indices(atoms) if atoms is not None else range(pmol.nao)
            labels = pmol.ao_labels(fmt=False)
            shells = sorted({" ".join(labels[i][1:3]) for i in ref_ao})
            raise ValueError(
                f"AVAS labels {list(ao_labels)} match no {minao} AO"
                f"{' of the active fragment' if atoms is not None else ''}; available shells: "
                f"{', '.join(shells)}"
            )

        s22 = pmol.intor_symmetric("int1e_ovlp")[np.ix_(targets, targets)]
        s21 = gto.intor_cross("int1e_ovlp", pmol, mol)[targets]
        return s21.T @ scipy.linalg.solve(s22, s21, assume_a="pos")


class PySCFIntegrals(PySCFBasis):
    """Backend for the ``examples/PySCF`` path of the qm-code-adapter branch.

    ``density_fit`` selects the two-electron transform in :meth:`eri_mo`:

    - ``False`` (default): the exact dense ``ao2mo`` N^5 transform.  Bit-for-bit
      what the workflow produced before this option existed.
    - ``True`` / an auxbasis string: a density-fitted ``(pq|rs)`` via
      ``pyscf.df.DF``.  This is an *approximation* (the DF/RI error, typically
      1e-3-1e-4 Ha on valence integrals), traded for O(N^3) storage so the active
      space can outgrow the few dozen orbitals the dense (nmo)^4 tensor allows.
      A string is passed straight through as the fitting auxbasis (e.g.
      ``"cc-pvtz-ri"``); ``True`` lets PySCF pick the default aux for ``mol.basis``.

    ``mf_ll`` is the same object handed to ``calc_base_ll``, and backs
    :meth:`veff_ll` -- the mean field :attr:`ProjectionEmbeddingAdapter.h_emb`
    subtracts back off ``F_emb``.  It defaults to ``mf_hl`` so a caller that only
    has one mean field (the stub adapters in the tests, and the ``xc_hl == xc_ll``
    case where the two coincide anyway) still constructs; pass the real low-level
    object whenever the two levels differ, or ``h_emb`` strips the wrong veff.
    """

    def __init__(self, mf_hl, mf_ll=None, *, density_fit: bool | str = False):
        super().__init__(mf_hl.mol)
        self.mf = mf_hl  # the same object passed to calc_base_hl
        self.mf_ll = mf_hl if mf_ll is None else mf_ll  # the one passed to calc_base_ll
        # Integral engine only; never kernel()'d.  `scf.hf.RHF` explicitly, NOT `mol.RHF()`:
        # on a `spin != 0` Mole the latter dispatches to ROHF, whose `get_veff`/`get_k`
        # return a `(2, nao, nao)` pair that the downfold and APC ranking cannot contract
        # against `(nao, nao)` blocks.  Restricted regardless of `mol.spin`.
        from pyscf import scf as _pyscf_scf

        self._hf = _pyscf_scf.hf.RHF(mf_hl.mol)
        self._density_fit = density_fit
        # A pyscf.df.DF, built lazily on the first eri_mo call. PySCF is untyped,
        # so this is Any rather than a precise DF type.
        self._df: Any = None

    def overlap(self) -> np.ndarray:
        return np.asarray(self.mol.intor("int1e_ovlp"))

    def hcore(self) -> np.ndarray:
        return np.asarray(self.mf.get_hcore())

    @staticmethod
    def _spin_average(v: np.ndarray) -> np.ndarray:
        """Collapse a ``(2, nao, nao)`` veff pair to the restricted equivalent.

        A UKS/UHF/ROHF ``get_veff`` returns one matrix per spin channel.  The
        spin-restricted downfold this adapter performs wants a single ``(nao, nao)``
        operator, and the correct reduction is the **mean**, not the sum: ``veff`` is a
        *potential* (a per-electron operator), unlike a density matrix.  Verified on a
        closed shell, where ``0.5 * (v_alpha + v_beta)`` reproduces the restricted
        ``get_veff`` exactly while the sum is twice too large -- routing this through
        ``_as_ao_total`` (which sums, correctly, for densities) would silently double
        the mean field that ``h_emb`` subtracts back off.
        """
        v = np.asarray(v)
        if v.ndim == 3 and v.shape[0] == 2:
            return 0.5 * (v[0] + v[1])
        return v

    @staticmethod
    def _as_veff_arg(mf, dm: DensityArg) -> np.ndarray:
        """Shape a density argument for ``mf.get_veff``, collapsing a pair if ``mf`` is restricted.

        An unrestricted ``get_veff`` wants ``(2, nao, nao)`` and reads the two channels
        as genuine alpha/beta; a *restricted* one wants ``(nao, nao)`` and would read a
        3-D array as a batch of densities.  So the pair is only passed through when the
        mean field can interpret it, and summed otherwise -- which is exactly right,
        since a restricted functional depends on the total density alone.
        """
        d = np.asarray(dm[0]) + np.asarray(dm[1]) if isinstance(dm, tuple) else np.asarray(dm)
        if not isinstance(dm, tuple):
            return d
        if is_spin_resolved(mf):
            return np.stack([np.asarray(dm[0]), np.asarray(dm[1])])
        return d

    def veff_hl(self, dm: DensityArg) -> np.ndarray:
        return self._spin_average(self.mf.get_veff(self.mol, self._as_veff_arg(self.mf, dm)))

    def veff_ll(self, dm: DensityArg) -> np.ndarray:
        return self._spin_average(self.mf_ll.get_veff(self.mol, self._as_veff_arg(self.mf_ll, dm)))

    def veff_ll_spin(self, dm: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Each spin channel's low-level effective potential at an ``(alpha, beta)`` pair.

        The per-channel counterpart of :meth:`veff_ll` (which returns the channel mean):
        ``v_sigma = J[d_a + d_b] - K[d_sigma]`` for HF, the spin-resolved KS potential
        otherwise.  A restricted low level has one potential for both channels.
        """
        v = np.asarray(self.mf_ll.get_veff(self.mol, self._as_veff_arg(self.mf_ll, dm)))
        if v.ndim == 3 and v.shape[0] == 2:
            return v[0], v[1]
        return v, v

    def veff_hf(self, dm) -> np.ndarray:
        # `_hf` is the restricted engine (see __init__), so this is normally already
        # (nao, nao); the reduction guards a spin-resolved `dm` argument, which puts
        # even RHF.get_veff on its two-channel branch.
        return self._spin_average(self._hf.get_veff(self.mol, np.asarray(dm)))

    def get_k(self, dm) -> np.ndarray:
        return self._spin_average(self._hf.get_k(self.mol, np.asarray(dm)))

    def veff_uhf(self, dm_a, dm_b) -> tuple[np.ndarray, np.ndarray, float]:
        """``(v_alpha, v_beta, e_two_body)`` for a spin-resolved core; see the protocol.

        ``self._hf`` is the restricted *integral engine* (never kernel()'d), and
        ``get_j``/``get_k`` on it are plain contractions of the AO ERIs with whatever
        density they are handed -- verified against a direct ``int2e`` contraction.  No
        spin averaging happens anywhere here; that is what :meth:`veff_hf` does, and what
        this method exists to avoid.

        Each channel's exchange is fetched in its own call rather than as one stacked
        ``(2, nao, nao)`` density: the stacked form gives identical numbers (checked) but
        makes PySCF log ``Incompatible dm dimension. Treat dm as RHF density matrix.`` on
        every evaluation, which is noise on a per-cycle code path.

        Like :meth:`veff_hf` and :meth:`get_k`, this uses the exact AO ERIs even when
        the backend was built with ``density_fit``; the DF approximation is applied only
        in :meth:`eri_mo`/:meth:`eri_mo_mixed`, matching the pre-existing behaviour of
        the restricted fold rather than introducing a second convention.
        """
        d_a = np.asarray(dm_a)
        d_b = np.asarray(dm_b)
        d_tot = d_a + d_b
        j_tot = np.asarray(self._hf.get_j(self.mol, d_tot))
        k_a = np.asarray(self._hf.get_k(self.mol, d_a))
        k_b = np.asarray(self._hf.get_k(self.mol, d_b))
        e_two_body = float(
            0.5 * np.einsum("ij,ji->", d_tot, j_tot)
            - 0.5 * (np.einsum("ij,ji->", d_a, k_a) + np.einsum("ij,ji->", d_b, k_b))
        )
        return j_tot - k_a, j_tot - k_b, e_two_body

    def eri_mo(self, mo_coeff) -> np.ndarray:
        """Chemist-notation ``(pq|rs)`` over the given MO block, shape (nmo,)*4.

        Dense by default; density-fitted when the backend was constructed with
        ``density_fit`` (see the class docstring).  Both return a full unpacked
        tensor -- the DF path only changes how it is *built*, not its shape, so
        the downfold and the solver are oblivious to the choice.
        """
        from pyscf import ao2mo

        nmo = mo_coeff.shape[1]
        if not self._density_fit:
            return ao2mo.restore(1, ao2mo.kernel(self.mol, mo_coeff), nmo)

        # Density-fitted transform: (pq|rs) ~= sum_P (pq|P)(P|rs), O(N^3) storage.
        if self._df is None:
            from pyscf import df

            self._df = df.DF(self.mol)
            if isinstance(self._density_fit, str):
                self._df.auxbasis = self._density_fit
            self._df.build()
        return ao2mo.restore(1, self._df.ao2mo(mo_coeff), nmo)

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        """Mixed-spin ``(p_a q_a | r_b s_b)``, shape ``(nmo,)*4``.

        The ``(aa|bb)`` block of an unrestricted downfold, where the first index pair
        comes from the alpha orbitals and the second from the beta ones.  Needs the
        four-orbital-set ``ao2mo.general``, not ``ao2mo.kernel``: the two pairs are
        different orbital sets, so the result has only 4-fold symmetry
        (``(pq|rs) == (qp|rs) == (pq|sr)``) and **not** the 8-fold property -- which is
        why :class:`~embasi_qiskit_integration.contract.EmbeddedHamiltonian` checks it
        separately.
        """
        from pyscf import ao2mo

        nmo = mo_a.shape[1]
        if mo_b.shape[1] != nmo:
            raise ValueError(
                f"alpha and beta blocks must share a dimension; got {nmo} and {mo_b.shape[1]}"
            )
        if not self._density_fit:
            out = ao2mo.general(self.mol, (mo_a, mo_a, mo_b, mo_b), compact=False)
            return np.asarray(out).reshape(nmo, nmo, nmo, nmo)
        if self._df is None:
            from pyscf import df

            self._df = df.DF(self.mol)
            if isinstance(self._density_fit, str):
                self._df.auxbasis = self._density_fit
            self._df.build()
        out = self._df.ao2mo((mo_a, mo_a, mo_b, mo_b), compact=False)
        return np.asarray(out).reshape(nmo, nmo, nmo, nmo)

    def energy_nuc(self) -> float:
        return float(self.mol.energy_nuc())

    # ---------------- capabilities / energies / SCF status ---------------- #
    @property
    def capabilities(self) -> frozenset[str]:
        caps = {
            CAP_ERI,
            CAP_EXACT_EXCHANGE,
            CAP_HCORE,
            CAP_ARBITRARY_DM_VEFF,
            CAP_ENERGY_ELEC,
            CAP_MINAO_PROJECTOR,
        }
        if self._density_fit:
            caps.add(CAP_DENSITY_FIT)
        return frozenset(caps)

    @property
    def ll_spin_resolved(self) -> bool:
        """Whether the low-level mean field carries two spin channels (UHF/UKS/ROHF)."""
        return is_spin_resolved(self.mf_ll)

    def energy_elec_ll(self, dm: np.ndarray) -> float:
        """``mf_ll.energy_elec(dm)[0]``: the low-level electronic energy at ``dm``."""
        return float(self.mf_ll.energy_elec(dm=dm)[0])

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["xc_hl"] = str(getattr(self.mf, "xc", None))
        out["density_fit"] = str(self._density_fit)
        return out

    def layer_scf_status(self, layer: Any) -> dict[str, Any] | None:
        return pyscf_layer_scf_status(layer)

    def hl_scf_status(self) -> dict[str, Any]:
        return {
            "converged": getattr(self.mf, "converged", None),
            "conv_tol": getattr(self.mf, "conv_tol", None),
            "max_cycle": getattr(self.mf, "max_cycle", None),
        }

    def layer_footing(self, layer: Any) -> tuple[np.ndarray, float]:
        return pyscf_layer_footing(layer)


# --------------------------------------------------------------------------- #
# Calculator construction (``--qm_code pyscf``)
# --------------------------------------------------------------------------- #
def _pyscf_mean_fields(atoms: Any, charge: int, cfg: Any) -> tuple[Any, Any, Any]:
    """``(mol, mf_ll, mf_hl)`` for ``atoms``, configured from the workflow settings.

    ``atoms`` must already be in the region-1-first order ``ProjectionEmbedding`` needs,
    so the ``Mole``'s atom indices line up with ``active_atoms``.  Shared by the direct
    (``pyscf``) and the callback (``pyscf-callback``) backends so the two run on
    identical calculators.
    """
    import pyscf
    from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf

    mol = pyscf.M(
        atom=ase_atoms_to_pyscf(atoms),
        basis=cfg.basis,
        charge=charge,
        spin=cfg.spin,
    )
    # A non-zero spin needs an unrestricted low level; PySCF's UKS carries the
    # spin axis EmbASI already threads through SPADE (n_spins > 1).
    if cfg.spin != 0 or cfg.unrestricted:
        mf_ll = mol.UKS(xc=cfg.xc_ll)
        mf_hl = mol.UKS(xc=cfg.xc_hl)
    else:
        mf_ll = mol.KS(xc=cfg.xc_ll)
        mf_hl = mol.KS(xc=cfg.xc_hl)
    # EmbASI runs each SCF on these objects (`mf.kernel(dm0=mf.get_init_guess())` when it
    # has no density of its own), so their cycle cap and guess settings are honoured.
    for mf in (mf_ll, mf_hl):
        if cfg.scf_max_cycle is not None:
            mf.max_cycle = cfg.scf_max_cycle
        if getattr(cfg, "init_guess", None) is not None:
            mf.init_guess = cfg.init_guess
        breaksym = getattr(cfg, "init_guess_breaksym", None)
        if breaksym is not None and hasattr(mf, "init_guess_breaksym"):
            mf.init_guess_breaksym = breaksym
    # Second-order (Newton) SCF for the low level.  At stretched bonds the small HOMO-LUMO
    # gap leaves DIIS circling a converged energy whose orbital gradient never meets
    # conv_tol_grad, and EmbASI then falls back to the *input* density.  Low level only:
    # DIIS converges the high-level (embedded HF) SCF in cases where Newton does not.
    # Read with a default so drivers whose settings predate the flag are unaffected.
    if getattr(cfg, "scf_newton", False):
        mf_ll = mf_ll.newton()
    return mol, mf_ll, mf_hl


@register_backend(
    "pyscf",
    backend_only_settings={
        "scf_newton": False,
        "init_guess": None,
        "init_guess_breaksym": None,
        "density_fit": False,
        "df_auxbasis": None,
    },
)
def build_pyscf_backend(atoms: Any, charge: int, cfg: Any) -> QMBackend:
    """PySCF ``Mole`` + KS/UKS calculators for EmbASI, and the integrals over them."""
    from pyscf.pbc.tools.pyscf_ase import PySCF

    _mol, mf_ll, mf_hl = _pyscf_mean_fields(atoms, charge, cfg)
    # PySCFIntegrals wraps the *same* mf_hl/mf_ll objects handed to
    # calc_base_hl/calc_base_ll, so veff_ll undoes exactly what EmbASI folded into
    # F_emb (which is built from the A_LL blocks -- see the h_emb property) and
    # veff_hl remains available as the high-level mean field.
    density_fit: bool | str = getattr(cfg, "df_auxbasis", None) or getattr(
        cfg, "density_fit", False
    )
    return QMBackend(
        name="pyscf",
        calc_ll=PySCF(method=mf_ll),
        calc_hl=PySCF(method=mf_hl),
        integrals=PySCFIntegrals(mf_hl, mf_ll, density_fit=density_fit),
    )


# --------------------------------------------------------------------------- #
# PySCF through EmbASI's callback/matrix path (``--qm_code pyscf-callback``)
# --------------------------------------------------------------------------- #
class _PySCFEri:
    """The ERI hook a callback backend needs, served by PySCF's ``ao2mo``."""

    def __init__(self, ints: PySCFIntegrals):
        self._ints = ints

    def eri_mo(self, mo_coeff: np.ndarray) -> np.ndarray:
        return self._ints.eri_mo(mo_coeff)

    def eri_mo_mixed(self, mo_a: np.ndarray, mo_b: np.ndarray) -> np.ndarray:
        return self._ints.eri_mo_mixed(mo_a, mo_b)


class PySCFCallbackIntegrals(CallbackIntegrals):
    """:class:`CallbackIntegrals` over PySCF, plus PySCF's AVAS projector.

    Every one- and two-body mean-field quantity comes back through EmbASI's
    run/extract path (``AtomsEmbed.run_noscf``); only the 4-index ERIs and the MINAO
    cross overlap are PySCF-specific hooks, since no Hamiltonian evaluation yields them.
    """

    def __init__(self, *args: Any, basis: PySCFBasis, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._pyscf_basis = basis

    @property
    def capabilities(self) -> frozenset[str]:
        return super().capabilities | {CAP_MINAO_PROJECTOR}

    def valence_ao_projector(self, ao_labels, atoms=None, minao: str = "minao") -> np.ndarray:
        return self._pyscf_basis.valence_ao_projector(ao_labels, atoms=atoms, minao=minao)


@register_backend("pyscf-callback", settings_of="pyscf")
def build_pyscf_callback_backend(atoms: Any, charge: int, cfg: Any) -> QMBackend:
    """The PySCF calculators of ``pyscf``, with integrals from EmbASI layer evaluations.

    Meant as the reference implementation of the generic callback route: it must agree
    with ``pyscf`` to round-off, and any other code EmbASI can drive gets the same
    treatment by supplying its own calculators (and, optionally, an ERI hook).
    """
    from pyscf.pbc.tools.pyscf_ase import PySCF

    mol, mf_ll, mf_hl = _pyscf_mean_fields(atoms, charge, cfg)
    calc_ll, calc_hl = PySCF(method=mf_ll), PySCF(method=mf_hl)
    # The exchange evaluator: an unrestricted HF on the same Mole, so one evaluation at
    # (d, 0) separates J from K.  Never handed to ProjectionEmbedding.
    calc_hf = PySCF(method=mol.UHF())
    ll_spin = is_spin_resolved(mf_ll)
    density_fit: bool | str = getattr(cfg, "df_auxbasis", None) or getattr(
        cfg, "density_fit", False
    )
    direct = PySCFIntegrals(mf_hl, mf_ll, density_fit=density_fit)
    basis = PySCFBasis(mol)
    ints = PySCFCallbackIntegrals(
        LayerEvaluator(atoms, calc_ll, n_spin=2 if ll_spin else 1, label="cb_ll"),
        LayerEvaluator(atoms, calc_hl, n_spin=2 if is_spin_resolved(mf_hl) else 1, label="cb_hl"),
        LayerEvaluator(atoms, calc_hf, n_spin=2, label="cb_hf"),
        nao=int(mol.nao),
        basis_atoms=basis.ao_atom_map(),
        ao_labels=basis.ao_labels(),
        describe={
            **basis.describe(),
            "xc_hl": str(getattr(mf_hl, "xc", None)),
            "density_fit": str(density_fit),
        },
        ll_spin_resolved=ll_spin,
        eri_provider=_PySCFEri(direct),
        # In-process: an evaluation is one PySCF Fock build, cheap enough for an SCF loop.
        cheap_evaluations=True,
        scf_status=pyscf_layer_scf_status,
        hl_status=direct.hl_scf_status,
        basis=basis,
    )
    return QMBackend(name="pyscf-callback", calc_ll=calc_ll, calc_hl=calc_hl, integrals=ints)
