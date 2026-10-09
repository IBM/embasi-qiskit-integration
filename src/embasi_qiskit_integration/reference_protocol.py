# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""The WF-in-DFT protocol of the reference projection-embedding implementation.

Rossmannek et al., J. Phys. Chem. Lett. 2023 (arXiv:2302.03052), code at
github.com/mrossinek/qiskit-nature-projection-embedding.  It differs from this
package's default WF-in-DFT energy (``ProjectionEmbeddingAdapter.projection_energy``)
in two places, and the two give measurably different curves for a KS low level:

* **Variational SCF-in-SCF.**  Subsystem A's HF density minimizes

      E[γ_A] = E_HF[γ_A] + E_LL[γ_A + γ_B] - E_LL[γ_A] + E_nuc

  with the embedding potential ``F_LL[γ_A + γ_B] - F_LL[γ_A]`` rebuilt from the
  current ``γ_A`` every iteration.  The default path instead keeps ``v_emb`` frozen at
  the low-level density and adds the first-order ``tr[(γ_HL - γ_LL) v_emb]``, which
  grows large at stretched bonds.
* **Correlation on top.**  The total is ``E[γ_A] + (E_solver - E_ref)``: the active-space
  correlation in the canonical orbitals of the converged embedded Fock matrix, the
  active virtuals being the lowest canonical concentric-localization virtuals.

For an HF low level both reduce to the default path (the SPADE density of A is
already self-consistent).  :func:`reference_wf_in_dft_hamiltonian` returns the
active-space Hamiltonian with ``e_core`` chosen so that ``E_solver + e_core`` is the
protocol's total, so FCI and SQD both run through it unchanged.  Single-shot only:
the protocol has no density feedback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from embasi_qiskit_integration.contract import EmbeddedHamiltonian
from embasi_qiskit_integration.selectors import concentric_localization_selector


@dataclass(frozen=True)
class ReferenceProtocolInfo:
    """What the protocol converged to, for the run log."""

    e_variational: float  # E[γ_A] at convergence (Ha)
    e_reference: float  # active-space HF-determinant energy, E_ref + e_core = E[γ_A]
    n_iterations: int
    n_virtual_kept: int  # concentric-localization virtuals before the n_virtual cap
    active_orbital_energies: tuple[float, ...] = ()  # diag of the embedded Fock (Ha)


def variational_hf_in_dft(ints, dm_a, dm_b, p_b, n_occ_a, *, max_iter=200, conv_tol=1e-10):
    """Minimize ``E[γ_A]`` (module docstring) over A's closed-shell HF density.

    ``dm_a``/``dm_b`` are spin-summed AO densities (``dm_b`` stays fixed), ``p_b`` the
    level-shift projector that keeps A orthogonal to B.  Returns ``(dm_a, fock, energy,
    n_iterations)``; ``fock`` is the converged embedded Fock matrix including ``p_b``.
    """
    from pyscf import lib

    h = np.asarray(ints.hcore())
    s = np.asarray(ints.overlap())
    e_nuc = ints.energy_nuc()

    def fock_ll(dm):
        return h + ints.veff_ll(dm)

    def energy(dm):
        e_hf = np.einsum("ij,ji->", h, dm) + 0.5 * np.einsum("ij,ji->", ints.veff_hf(dm), dm)
        e_ll_tot = ints.mf_ll.energy_elec(dm=dm + dm_b)[0]
        e_ll_a = ints.mf_ll.energy_elec(dm=dm)[0]
        return float(e_hf + e_ll_tot - e_ll_a + e_nuc)

    def fock(dm):
        return h + ints.veff_hf(dm) + (fock_ll(dm + dm_b) - fock_ll(dm)) + p_b

    diis = lib.diis.DIIS()
    dm, e_old = np.asarray(dm_a), energy(np.asarray(dm_a))
    for iteration in range(1, max_iter + 1):
        f = fock(dm)
        f = diis.update(f, xerr=f @ dm @ s - s @ dm @ f)
        _eps, c = sla.eigh(f, s)
        dm_new = 2.0 * c[:, :n_occ_a] @ c[:, :n_occ_a].T
        e_new = energy(dm_new)
        converged = abs(e_new - e_old) < conv_tol and np.abs(dm_new - dm).max() < 1e-6
        dm, e_old = dm_new, e_new
        if converged:
            return dm, fock(dm), e_new, iteration
    raise RuntimeError(
        f"variational HF-in-DFT SCF for subsystem A did not converge in {max_iter} iterations"
    )


def reference_wf_in_dft_hamiltonian(
    adapter, fragment_ao, *, n_frozen_occ: int, n_virtual: int, n_shells: int = 0
) -> tuple[EmbeddedHamiltonian, ReferenceProtocolInfo]:
    """The protocol's active-space Hamiltonian, after ``adapter.run_low_level()``.

    ``fragment_ao`` are subsystem A's AO indices (the concentric-localization basis).
    The active space is A's occupied orbitals ``n_frozen_occ:`` plus the ``n_virtual``
    lowest canonical virtuals of the kept shells (``n_shells`` after shell 0).
    """
    ints = adapter.ints
    s = np.asarray(ints.overlap())
    dm_a0, dm_b = np.asarray(adapter._dm_a), np.asarray(adapter._dm_b)
    p_b = np.asarray(adapter.p_b)
    n_occ_a = int(round(np.einsum("ij,ji->", dm_a0, s) / 2.0))

    dm_a, f, e_var, n_iter = variational_hf_in_dft(ints, dm_a0, dm_b, p_b, n_occ_a)

    # Canonical orbitals of the embedded Fock matrix.  B's occupied orbitals sit at
    # ~mu (the projector), so keep only A's occupied block and the genuine virtuals.
    eps, c = sla.eigh(f, s)
    n_occ_b = int(round(np.einsum("ij,ji->", dm_b, s) / 2.0))
    n_virt = c.shape[1] - n_occ_a - n_occ_b
    coeff = np.hstack([c[:, :n_occ_a], c[:, n_occ_a : n_occ_a + n_virt]])
    if n_virt and eps[n_occ_a + n_virt - 1] > 0.5 * float(np.abs(p_b).max()):
        raise RuntimeError("projector-shifted orbitals leaked into the virtual block")

    localise = concentric_localization_selector(
        s,
        np.asarray(fragment_ao),
        f,
        n_shells=n_shells,
        max_virtual=n_virtual,
        canonical=True,
        # The reference code keeps len(fragment_ao) shell-0 columns even past the SVD rank
        # (its active virtuals then depend on LAPACK's null-space basis); matched here.
        shell0_size=len(fragment_ao),
    )
    coeff_loc, sigma2 = localise(coeff, n_occ_a)
    n_kept = int(np.sum(sigma2 > 0.5))
    if n_kept < n_virtual:
        raise ValueError(f"only {n_kept} concentric virtuals kept; n_virtual={n_virtual}")
    c_act = np.hstack(
        [coeff_loc[:, n_frozen_occ:n_occ_a], coeff_loc[:, n_occ_a : n_occ_a + n_virtual]]
    )
    n_act_occ = n_occ_a - n_frozen_occ

    # One-body operator seen by the active electrons: everything in F except their own
    # mean field, i.e. h + G_HF[core] + v_emb + P_B.
    c_aocc = coeff_loc[:, n_frozen_occ:n_occ_a]
    h1 = c_act.T @ (f - ints.veff_hf(2.0 * c_aocc @ c_aocc.T)) @ c_act
    h2 = ints.eri_mo(c_act)
    occ = range(n_act_occ)
    e_ref = float(
        2.0 * sum(h1[i, i] for i in occ)
        + sum(2.0 * h2[i, i, j, j] - h2[i, j, j, i] for i in occ for j in occ)
    )
    ham = EmbeddedHamiltonian(
        h1=h1,
        h2=h2,
        e_core=e_var - e_ref,
        nelec=(n_act_occ, n_act_occ),
        meta={"protocol": "reference-wf-in-dft"},
    )
    energies = tuple(float(e) for e in np.diag(c_act.T @ f @ c_act))
    return ham, ReferenceProtocolInfo(e_var, e_ref, n_iter, n_kept, energies)
