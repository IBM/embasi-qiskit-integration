# Embedding workflow: what it does, and how it maps onto lahs-workflows

Working notes. **Not committed** — scratch reference for the PR discussion.

All file/line references are against this branch (`refactor/circuit-generator-run`).

---

## Part 1 — What `scripts/embedding_workflow.py` actually does

The script is a 24-line shim. It exists only so the workflow is *importable*:
tests and an interaction-energy driver use `EmbeddingWorkflow` directly rather
than loading a script by path.

### Delegation chain

```
scripts/embedding_workflow.py   →  main()
  └─ EmbeddingWorkflow (pydantic-settings BaseSettings)   ← 24 CLI flags
       └─ .run()  →  ._build_adapter()  ._build_selector()  ._build_solver()
                     └─ ._run_outer_loop()      ← the actual science
```

`--help` works without EmbASI installed (options are generated from the settings
model). An actual run needs the `embed` extra; without it, it stops at
`ModuleNotFoundError: No module named 'ase'` (`embedding.py:322`).

### Setup: `_build_adapter()` (`embedding.py:319`)

Builds a live EmbASI calculation, mirroring the EmbASI developers' own PySCF
example:

1. Pulls a molecule from the **s26 benchmark set** via ASE (`s26_index=22` →
   methanol dimer; `n_atoms=6` truncates to the monomer).
2. Builds an `embed_mask`: `1` = high-level (`active_atoms`, default the OH
   fragment), `2` = low-level environment.
3. **Reorders the atoms** so region-1 comes first — `ProjectionEmbedding`
   requires a sorted mask — and rebuilds the PySCF `Mole` from the reordered
   atoms so the two stay in sync.
4. Creates two KS calculators (`xc_ll`, `xc_hl`, both PBE by default) and a
   `ProjectionEmbedding` with `projection="level-shift"`.
5. Wraps **the same `mf_hl` object** in `PySCFIntegrals`. That identity is
   load-bearing: `veff_hl` must undo exactly what EmbASI folded into `F_emb`.

### The loop: `_run_outer_loop()` (`embedding.py:185`)

A **self-consistent outer loop**, not a single pass. Four steps per cycle:

| step | call | what happens |
|---|---|---|
| 2 | `emb.build_orbitals()` → `emb.embedded_hamiltonian()` | Build subsystem-A orbitals, apply the active-space cut (`n_virtual` / `selector`), downfold to an `EmbeddedHamiltonian` (`h1`, `h2`, `e_core`) |
| 3 | `self._solve(ham, solver)` | Hand it to **SQD or FCI**. Under MPI only rank 0 solves; result is broadcast |
| 4 | `emb.projection_energy()` | Assemble `E = E_low(total) − E_low(A) + E_high(A) + corr` |
| 5 | `emb.run_low_level(dm_ab_in=fed)` | Feed the correlated 1-RDM back, rebuild the embedded Fock |

Then convergence check, repeat. `--max_cycles 1` reduces it to one shot.

### Three non-obvious details

**Density mixing is load-bearing.** The undamped map `γ → F(γ)` *diverges* for
this problem — `embedding.py:204` records it as verified: `max|Δγ^A|` grows
back and E swings **±0.03 Ha**. Step 5 therefore mixes linearly,
`fed = α·new + (1−α)·prev`, with `mix_alpha=0.5` converging `max|Δγ^A|`
monotonically to ~1e-7. `--mix_alpha 1.0` restores the raw feedback *and the
divergence*.

**Convergence tolerates solver noise.** SQD is stochastic, so the fixed point is
noisy in a way the deterministic PbE literature does not address. Default
`converge_on="energy"` stops on `|ΔE| < e_tol` alone; `"energy_and_density"` also
demands `max|Δγ^A| < rho_tol`. `--reseed_sqd` chooses honest resampling each
cycle vs. freezing cycle 0's subspace.

**An energy "footing shift" is applied and surfaced.** `E_high(A)` is rebased
onto `E_low(A)`'s ghosted-subsystem nuclear frame before the subtraction.
Without it the two A-terms sit **~4 Ha apart** and the paper's Eq. 8 cancellation
fails (`projection_embedding_adapter.py:703`). The workflow logs the shift rather
than applying it silently. Came in with PR #2 (`fc83673 "fix: fixing energy
footing"`).

---

## Part 2 — `_solve` does *neither* the quantum run nor SqDRIFT

`_solve` (`embedding.py:406`) is **plumbing only**. Its whole job is deciding
*where* the solve happens:

- `handoff="in-process"` → `rank0_solve(solver, ham)`
- `handoff="two-process"` → rank 0 writes a job dir, solves it, writes the result
  back, broadcasts it — standing in for a separate
  `embasi-qiskit-integration solve <dir> --watch` process.

Both paths broadcast the `SolverResult` so every MPI rank returns the same object.

**It isn't necessarily quantum at all.** `_solve` takes whatever
`_build_solver()` returned; with `--solver fci` the same call runs exact
diagonalization via PySCF — no circuits, no sampling, no SqDRIFT. That
polymorphism is why the embedding loop doesn't know which solver it drives.

### Real chain

```
_solve(ham, solver)                    ← MPI + handoff plumbing ONLY
  └─ rank0_solve(solver, ham)          ← rank 0 solves, result broadcast
       └─ solver.solve(ham)            ← polymorphic: SQDSolver *or* FCISolver
            │
            └─ SQDSolver.solve():
                 build_circuits(ham)   ← SqDRIFT generation + initial-state prep
                 sampler.run(...)      ← the quantum run
                 merge_counts(...)     ← pool the ensemble
                 run_sqd(...)          ← classical diagonalization
```

So, precisely: **SqDRIFT generation** is `build_circuits`, and **the quantum run**
is `sampler.run` — two distinct stages inside `solve`, not inside `_solve`.

Inside `build_circuits` (the seam this PR moved):

```
build_sqdrift_circuits(ham, include_initial_state=False)   ← circuit_generator
     ↓  N bare evolution circuits
resolve_initial_state() + compose_full_circuit()           ← circuit_run.prep
     ↓  N full circuits
sampler.run(circuits, shots) → N counts dicts             ← the quantum run
     ↓
merge_counts() → one pooled distribution
     ↓
run_sqd() → energy + RDMs
```

This runs **once per cycle**. A 15-cycle run with `method="qdrift",
num_randomizations=4` builds and samples **60 circuits** total. That is also why
`--reseed_sqd` exists: it decides whether cycle *n* redraws its subspace or
reuses cycle 0's.

---

## Part 3 — Mapping onto lahs-workflows

The framing "`_solve` wraps what we wanted from lahs, the other steps are EmbASI
computation or adjustments" is **mostly right, with one significant correction.**

### `SQDSolver.solve` wraps THREE lahs steps, not two

| inside `SQDSolver.solve` | lahs step | ported? |
|---|---|---|
| `build_circuits()` | `circuit_generator` | ✅ yes |
| `sampler.run()` | `quantum_circuit_run` | ✅ yes |
| **`run_sqd()`** | **`postprocess_bitstrings`** | ❌ **no — `qiskit-addon-sqd` directly** |

That third row is the open question. lahs has a whole `postprocess_bitstrings`
step — "full SQD pipeline with iteration", **1428 lines**, backed by a
**24-module** `post_processing/` package (configuration recovery, subsampling,
expansion, bitstring merging). We do that stage with a thin `run_sqd()` wrapper
around a single `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`
call (`sqd/driver.py:67`).

**Implication for the eventual swap:** `circuit_generator` and `circuit_run` are
drop-in. SQD post-processing is a *third* decision not yet made, and lahs'
version is considerably richer than ours.

### The boundary is not only "EmbASI vs lahs"

Two further lahs steps overlap with things the embedding loop does by hand:

- **`one_rdm_extractor`** — lahs' plug for closing a refinement loop by feeding
  an RDM back. Our step 5 (`emb.rdm1_ao(...)` + `run_low_level(dm_ab_in=fed)`) is
  the EmbASI-specific equivalent.
- **`metric_extractor`** — lifts the headline energy out of artifacts into the
  record. Our step 4 (`emb.projection_energy`) plays that role.

Neither is portable: lahs' versions operate on files and a workflow record; ours
operate on live EmbASI objects. So they *are* "EmbASI computation" as framed —
but the concepts exist on both sides, worth knowing so nobody tries to port them.

Also **`fcidump_generator`**: lahs generates its own active-space Hamiltonian
(AVAS/NOON selection). Our step 2 (`build_orbitals` + `embedded_hamiltonian`) is
EmbASI's projection-embedding replacement for it — same pipeline slot, different
physics. That is why `selectors.py` / `concentric_selector` exists rather than
reusing lahs' selection modes.

### Corrected picture

```
Step 1  EmbASI low-level embedding        ← EmbASI only
Step 2  build_orbitals → embedded_ham     ← EmbASI's answer to lahs fcidump_generator
Step 3  _solve → SQDSolver.solve
          ├─ build_circuits    ← lahs circuit_generator        [ported]
          ├─ sampler.run       ← lahs quantum_circuit_run      [ported]
          └─ run_sqd           ← lahs postprocess_bitstrings   [NOT ported]
Step 4  projection_energy                 ← EmbASI PbE assembly (cf. metric_extractor)
Step 5  rdm1 feedback                     ← EmbASI feedback   (cf. one_rdm_extractor)
```

### One caveat on "adjustments"

Steps 4 and 5 are **not** cosmetic glue. The footing shift in step 4 is worth
~4 Ha, and the density mixing in step 5 is what stops the loop diverging. Both
are load-bearing physics.

---

## Verification status of these notes

- Code paths, line numbers, and the lahs step inventory: **read from source** on
  this branch and in `/Users/dic/Software/lahs-workflows`.
- The ~4 Ha footing and ±0.03 Ha divergence figures: **quoted from the source
  comments** that record them (`projection_embedding_adapter.py:703`,
  `embedding.py:204`) — not independently re-measured here.
- The embedding workflow itself was **not executed**: EmbASI is not installed in
  this environment (`--help` works; a real run needs `.[embed]`). Interface
  described from code, not from a live run.
- The 11 `embasi`-marked tests are identical on this branch and `origin/main`
  (verified by diffing collected node IDs), and all EmbASI test files are
  unchanged by this PR.
