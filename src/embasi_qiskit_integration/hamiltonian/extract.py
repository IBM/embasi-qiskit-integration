# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""EmbASI-facing extraction of an active-space :class:`EmbeddedHamiltonian`.

This is the *only* module (with :mod:`workflow` and :mod:`ipc`) that touches
EmbASI. All EmbASI attribute accesses are marked ``# TODO(embasi-api)`` and
resolved against the EmbASI source
(https://github.com/tamm-cci/EmbASI, ``embasi/atoms_embedding_asi.py`` +
``embasi/embedding.py``). Nothing here imports EmbASI at module load; the glue
is exercised in CI via :class:`FakeEmbASI` in the tests, and against real
EmbASI only when ``EMBASI_AVAILABLE=1``.

Projection-based embedding (Manby et al., JCTC 2012) builds the embedded Fock
operator::

    F^{A-in-B} = h_core + g^{high}[gamma_A] + v_emb[gamma_A, gamma_B] + P_B

where ``P_B = mu * S @ D_B @ S`` is the level-shift projector (EmbASI:
``ProjectionEmbedding.calculate_levelshift_projector`` -> ``self.P_b``,
``self.mu_val``). We fold the embedding potential + projector into the one-body
integrals of the active space and rebuild the two-body integrals in PySCF
(Strategy B), because ASI/FHI-aims does not export the active-space ERIs
directly.
"""

from __future__ import annotations

import numpy as np

from embasi_qiskit_integration.contract import EmbeddedHamiltonian


def embedded_hamiltonian_from_embasi(emb, active_orbitals) -> EmbeddedHamiltonian:
    """Build an :class:`EmbeddedHamiltonian` from an EmbASI embedding object.

    Args:
        emb: an EmbASI ``ProjectionEmbedding`` (or ``AtomsEmbed``) instance that
            has completed the low-level SCF so its matrices are populated.
        active_orbitals: sequence of localized-orbital indices defining the
            active space (subsystem A) to hand to the high-level solver.

    Returns:
        The active-space Hamiltonian with the embedding potential + level-shift
        projector folded into ``h1``.
    """
    C = _localized_coeffs(emb, active_orbitals)
    hcore = _core_hamiltonian(emb)
    v_emb = _embedding_potential(emb)

    # Fold embedding potential + projector into the one-body operator, then
    # project onto the active localized-orbital basis.
    f_embedded = hcore + v_emb
    h1 = C.T @ f_embedded @ C

    h2 = _two_body_active(emb, C)
    e_core = _core_energy(emb)
    nelec = _active_nelec(emb, active_orbitals)

    meta = {
        "source": "embasi",
        "embedding": "projection-based (level-shift)",
        "n_active_orbitals": int(len(active_orbitals)),
    }
    return EmbeddedHamiltonian(h1=h1, h2=h2, e_core=e_core, nelec=nelec, meta=meta)


def _localized_coeffs(emb, active_orbitals) -> np.ndarray:
    """Localized MO coefficients for the active subsystem A (AO x n_active)."""
    # TODO(embasi-api): EmbASI computes SPADE-localized orbitals in
    # ProjectionEmbedding.spade_localisation(atomsembed, hamiltonian, overlap),
    # solving the Roothan-Hall problem at the wrapper level. The resulting
    # coefficient matrix is not a documented public attribute; verify how it is
    # stored (e.g. via roothan_hall_eigensolver output) and slice active_orbitals.
    C_full = np.asarray(getattr(emb, "mo_coeff"))  # placeholder attribute name
    return C_full[:, list(active_orbitals)]


def _core_hamiltonian(emb) -> np.ndarray:
    """One-electron core Hamiltonian (kinetic + nuclear attraction) in AO basis."""
    # TODO(embasi-api): EmbASI exposes hamiltonian_kinetic (self._ham_kin) and
    # hamiltonian_total (self._ham_tot); the bare core h = kinetic + e-n. Confirm
    # whether an e-n / hcore matrix is exported directly by ASI or must be
    # reconstructed as hamiltonian_total - electron-electron (estat_plus_xc).
    return np.asarray(emb.hamiltonian_kinetic)


def _embedding_potential(emb) -> np.ndarray:
    """Embedding potential + level-shift projector P_B in AO basis."""
    # TODO(embasi-api): the level-shift projector is ProjectionEmbedding.P_b
    # (= mu_val * S @ D_B @ S, set in calculate_levelshift_projector). The full
    # embedding Fock is fock_embedding_matrix (self._dm); confirm whether it
    # already includes P_b or whether we add P_b explicitly.
    return np.asarray(emb.fock_embedding_matrix)


def _two_body_active(emb, C: np.ndarray) -> np.ndarray:
    """Active-space two-electron integrals (chemists' (pq|rs)).

    Strategy B: rebuild the ERIs in PySCF from the embedding geometry/basis and
    transform into the localized active basis ``C``. Strategy A (direct ASI ERI
    export) would replace this if ASI exposes the integrals.
    """
    # TODO(embasi-api): obtain the ASE Atoms + basis for subsystem A from emb
    # (emb.atoms / layer basis info) to construct a matching PySCF Mole, then
    # ao2mo into the localized active basis C. Placeholder builds a zero tensor
    # of the correct shape so the contract/shape logic is exercised in CI.
    n = C.shape[1]
    return np.zeros((n, n, n, n))


def _core_energy(emb) -> float:
    """Scalar offset: nuclear repulsion + environment/frozen-core energy."""
    # TODO(embasi-api): EmbASI accumulates energies in output_data_dict
    # ["TOTALENERGY"]; the PbE core offset is E_low(total) - E_low(A) plus the
    # nuclear term. Verify the exact bookkeeping before wiring the real value.
    return float(getattr(emb, "e_core", 0.0))


def _active_nelec(emb, active_orbitals) -> tuple[int, int]:
    """(n_alpha, n_beta) electrons in the active space."""
    # TODO(embasi-api): derive from the localized active-space occupation /
    # subsystem population (ProjectionEmbedding.calc_subsys_pop). Placeholder
    # assumes a closed-shell active space filling the active orbitals.
    n_active = len(active_orbitals)
    na = getattr(emb, "n_active_alpha", n_active // 2)
    nb = getattr(emb, "n_active_beta", n_active - n_active // 2)
    return (int(na), int(nb))
