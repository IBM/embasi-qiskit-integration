# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Driver run under ``mpirun`` by ``test_mpi.py`` to exercise the workflow's MPI
orchestration WITHOUT EmbASI's broken ``parallel=True`` scalapack SPADE path.

EmbASI's ``roothan_hall_eigensolver_scalapack.hamiltonian_eigensolv_parallel``
does ``overlap[0,0].gl_m`` and assumes a scalapack-distributed matrix object,
but under ``parallel=True`` it receives a plain numpy ndarray -> AttributeError,
which blocks the real ``construct_embedding_potential`` on every rank.

To test OUR MPI code -- rank-guarded logging (once), rank-0 solve + broadcast,
both ranks reaching the same point -- we replace ``ProjectionEmbedding`` with a
small, self-consistent mock built from a real PySCF RHF, so every adapter
validation (S-orthonormality, occupied-span, integral populations, P_B leak)
still passes on honest numbers.  Nothing here is imported by the package or the
workflow script; it exists only for the MPI test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np


class _MockProjectionEmbedding:
    """Stands in for embasi.embedding.ProjectionEmbedding on the CEF path.

    Partitions a real closed-shell RHF into an "A" (first ``n_occ_a`` occupied
    MOs) and "B" (remaining occupied) subsystem, and hands back the same 5-tuple
    ``construct_embedding_potential`` returns -- ``(γ^A, γ^B, S, v_emb, P_B)`` as
    (1, 1, nao, nao) arrays, to match the SpinKpointArray layout the adapter
    squeezes.  The adapter reassembles ``F_emb = h_kin^A + h_estat_xc^A + v_emb +
    P_B`` from the ``A_LL`` one-electron blocks exposed here plus the returned
    ``v_emb``/``P_B``, so this mock mirrors the real read path.
    """

    projection = "level-shift"

    def __init__(self, mol, mu: float, n_occ_a: int):
        from pyscf import scf

        mf = scf.RHF(mol).run()
        c = mf.mo_coeff  # full orthonormal MO set (nao, nao)
        occ = mf.mo_occ > 0
        c_occ = c[:, occ]
        n_occ = c_occ.shape[1]
        nao = c.shape[1]
        assert 0 < n_occ_a < n_occ

        self._mol = mol
        self._mu = mu
        self.mu_val = mu  # adapter cross-checks this against its own mu
        self._s = mol.intor("int1e_ovlp")
        self._hcore = mf.get_hcore()
        self._mf = mf

        # A_LL one-electron blocks the adapter reads to reassemble F_emb, plus the
        # A_LL.atoms.calc.mol reach-through used by _a_fragment_footing.  h_core is
        # split kinetic + "estat_plus_xc" (everything else) as EmbASI names them.
        h_kin = np.asarray(mol.intor("int1e_kin"))
        h_core = np.asarray(mf.get_hcore())
        self._h_kin_a = h_kin
        self._h_estat_xc_a = h_core - h_kin
        self.A_LL = SimpleNamespace(
            atoms=SimpleNamespace(calc=SimpleNamespace(mol=mol)),
            hamiltonian_kinetic=h_kin[np.newaxis, np.newaxis, :, :],
            hamiltonian_estat_plus_xc=(h_core - h_kin)[np.newaxis, np.newaxis, :, :],
        )
        # Surrogate low-level energies (eV) under EmbASI's own names, so the
        # projection-energy assembly runs; constants -> a fixed offset only.
        self.subsys_A_lowlvl_totalen = -1.0
        self.subsys_AB_lowlvl_scftotalen = -2.5

        # Partition the OCCUPIED space into A (first n_occ_a) and B (rest); all
        # virtuals belong to A so build_orbitals recovers occupied + virtual A.
        b_cols = np.arange(n_occ_a, n_occ)
        c_a = c_occ[:, :n_occ_a]
        c_b = c_occ[:, n_occ_a:]
        self._dm_a = 2.0 * (c_a @ c_a.T)
        self._dm_b = 2.0 * (c_b @ c_b.T)

        # Build F_emb by construction so that eigh(F_emb, S) returns EXACTLY the
        # canonical MOs, with subsystem-B occupied orbitals pushed to ~mu by the
        # level shift and everything else at its canonical orbital energy.  Then
        # the below-floor block is precisely the A space and the span check is
        # satisfied to machine precision.  F_emb = S C diag(eps) C^T S.
        eps = mf.mo_energy.copy()
        eps[b_cols] += mu  # level shift lifts B out of the A window
        sc = self._s @ c
        self._fock = sc @ np.diag(eps) @ sc.T

        # Occupied A MOs, (1, nao, nocc_a) -- what mo_coeffs_A_LL looks like.
        self.mo_coeffs_A_LL = c_a[np.newaxis, :, :]
        self.mo_coeffs_B_LL = c_b[np.newaxis, :, :]
        _ = nao

    def construct_embedding_potential(
        self, dmab_in=None, dma_in=None, dmb_in=None, a_nspade_mos=None
    ):
        # Signature mirrors embasi.embedding.ProjectionEmbedding, so a kwarg the
        # adapter starts passing shows up here as a wrong *answer*, not a
        # TypeError from the mock.  The A/B split is fixed at construction
        # (``n_occ_a``), so an explicit SPADE-MO count would have to repartition
        # this mock to be honoured -- the MPI test never sets one, and silently
        # ignoring it would make the returned densities disagree with the request.
        assert a_nspade_mos is None, (
            "mock partitions A/B at construction via n_occ_a; honouring "
            f"a_nspade_mos={a_nspade_mos!r} would need a repartition"
        )
        # The dma_in/dmb_in/dmab_in feedback path is not exercised by the MPI test
        # (a single pass, no feedback under mock).
        # P_B is the level-shift projector mu * S γ^B S; v_emb is defined so the
        # adapter's reassembly reproduces self._fock bit-for-bit.
        p_b = self._mu * (self._s @ self._dm_b @ self._s)
        v_emb = self._fock - self._h_kin_a - self._h_estat_xc_a - p_b
        return (
            self._dm_a[np.newaxis, np.newaxis, :, :],
            self._dm_b[np.newaxis, np.newaxis, :, :],
            self._s[np.newaxis, np.newaxis, :, :],
            v_emb[np.newaxis, np.newaxis, :, :],
            p_b[np.newaxis, np.newaxis, :, :],
        )


def main() -> None:
    from mpi4py import MPI

    from embasi_qiskit_integration.ipc import rank0_solve
    from embasi_qiskit_integration.projection_embedding_adapter import (
        ProjectionEmbeddingAdapter,
        PySCFIntegrals,
    )
    from embasi_qiskit_integration.solvers import FCISolver

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    def log(msg: str) -> None:
        if rank == 0:
            print(msg, flush=True)

    import pyscf

    # Tiny, deterministic system; every rank builds it identically (as EmbASI's
    # collective SCF would).
    mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22", basis="sto-3g")
    mf_hl = mol.RHF()

    if size > 1:
        log("running under MPI with 2 ranks")

    mock = _MockProjectionEmbedding(mol, mu=1.0e6, n_occ_a=1)
    adapter = ProjectionEmbeddingAdapter(mock, PySCFIntegrals(mf_hl), mu=1.0e6)

    adapter.run_low_level()  # collective on every rank, mirrors the real flow
    orbitals = adapter.build_orbitals(n_frozen_occ=0, n_virtual=None)
    ham = adapter.embedded_hamiltonian(orbitals)

    # Only rank 0 solves; result broadcast to all ranks.
    result = rank0_solve(FCISolver(), ham)

    # Every rank must hold the same broadcast result.
    assert result is not None
    print(f"RANK {rank} REACHED_SOLVE E_solver {result.energy:.10f}", flush=True)
    log("Step done: solve broadcast to all ranks")


if __name__ == "__main__":
    main()
