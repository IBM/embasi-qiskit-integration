# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
"""Driver run under ``mpirun`` by ``test_mpi.py`` to exercise the workflow's MPI
orchestration WITHOUT EmbASI's broken ``parallel=True`` scalapack SPADE path.

EmbASI's ``roothan_hall_eigensolver_scalapack.hamiltonian_eigensolv_parallel``
does ``overlap[0,0].gl_m`` and assumes a scalapack-distributed matrix object,
but under ``parallel=True`` it receives a plain numpy ndarray -> AttributeError,
which blocks the real ``construct_embedded_fock`` on every rank.

To test OUR MPI code -- rank-guarded logging (once), rank-0 solve + broadcast,
both ranks reaching the same point -- we replace ``ProjectionEmbedding`` with a
small, self-consistent mock built from a real PySCF RHF, so every adapter
validation (S-orthonormality, occupied-span, integral populations, P_B leak)
still passes on honest numbers.  Nothing here is imported by the package or the
workflow script; it exists only for the MPI test.
"""

from __future__ import annotations

import numpy as np


class _MockProjectionEmbedding:
    """Stands in for embasi.embedding.ProjectionEmbedding on the CEF path.

    Partitions a real closed-shell RHF into an "A" (first ``n_occ_a`` occupied
    MOs) and "B" (remaining occupied) subsystem, and hands back the same
    3-tuple ``construct_embedded_fock`` returns -- as (1, nao, nao) arrays, to
    match the SpinKpointArray layout the adapter squeezes.
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
        self._s = mol.intor("int1e_ovlp")
        self._hcore = mf.get_hcore()
        self._mf = mf

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

    def construct_embedded_fock(self, dmab_in=None):
        # dmab_in path is not exercised by the MPI test (no feedback under mock).
        return (
            self._dm_a[np.newaxis, :, :],
            self._dm_b[np.newaxis, :, :],
            self._fock[np.newaxis, :, :],
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
