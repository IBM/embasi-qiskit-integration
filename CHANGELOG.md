# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial open-source release of `embasi-qiskit-integration`, coupling EmbASI
  projection-based embedding with the Qiskit SQD stack.
- SqDRIFT mode relabeling (`build_sqdrift_circuits(optimize=True)`, on by default
  when the new `relabel` extra is installed):
  `qiskit-fermions`' `RelabelModes` reorders the fermionic modes to minimize the
  excitation span, shortening the circuits (N2 CAS(8o,10e): per-draw depths
  390/202/413 → 369/153/273). The solver seam is `circuit_generator.relabel.
  NativeHighsSolverAdapter` (native appsi HiGHS with `load_solution=False`, so an
  infeasible solve degrades to "no permutation" instead of poisoning the batch).
- `circuit_run.permutation`: `unpermute_counts`, `unpermute_counts_list` and
  `unpermute_bitstrings`, which map sampled bitstrings from a relabeled mode order
  back to the original. **Required** for relabeled circuits: raw counts are in the
  permuted basis and have ~1.0 total variation distance from the truth. `SQDSolver`
  applies it per circuit *before* pooling, and `resolve_initial_state` reads
  `metadata["permutation"]` so the reference determinant is prepared in the
  matching order — relabeling is therefore transparent to `SQDSolver` callers, and
  the SQD energy is unchanged while the circuits get shorter.
- `canonical_permutation` (default **False**) selects how the mode permutation is
  obtained. The default applies the MILP solver's own ordering in one pass chain.
  Setting it True derives the permutation from a fixed candidate set instead
  (`canonicalize_permutation`), costing a second pass-manager run per randomization
  but making a seeded build reproducible — which the default is not, because the
  excitation-span model is degenerate and HiGHS is not a pure function of it
  (solving one model repeatedly in a single process returned one optimum five times
  and then a different one; depths 382 vs 368). Since the permutation is undone on
  the counts, that instability reaches the pooled distribution. Also exposed as
  `SQDSolver(canonical_permutation=...)`.
- Parallel SqDRIFT generation: `build_sqdrift_circuits(workers=N)` (and
  `SQDSolver(workers=N)` / `--workers`) shards each combination's randomizations
  into contiguous seed-chunks across processes, byte-identical to the sequential
  build (N2 CAS(8o,10e), 8 draws with relabeling: 81.5s → 12.5s at `workers=8`).
  A process-local operator cache in `circuit_generator.operator` keeps the operator
  construction to once per worker.
- Noise-aware layout selection. New modules `circuit_run.layout` (chain search,
  coupling-map pruning, joint-readout-fidelity ranking, adaptive threshold
  relaxation), `circuit_run.metrics` (per-qubit readout error, xslow flip fidelity,
  TREX fidelities) and `circuit_run.characterisation` (the `samplomatic` twirl
  program, twirl correction, and the compose helpers `characterise_readout` /
  `select_readout_layout`); `circuit_run.backend` gains `build_pinned_pass_manager`,
  `read_noise_from_backend`, `virtual_to_physical` and `edges_along_layout`.

  `RuntimeSampler(enable_readout_characterisation=True)` (also `--sampler runtime
  --enable_readout_characterisation true`) measures the device before submitting,
  prunes qubits above `readout_error_threshold` (0.03), pins the best 1-D chain, and
  transpiles over the pruned map with `target=` rather than `backend=` so gate
  durations survive. Defaults `n_rand_twirl=300`, `n_shots_per_twirl=25`,
  `hot_coupler_ps=False`, and **off by default** — it costs
  an extra job. Refused on simulated backends, whose "measured" readout error is
  only their configured noise model; a characterisation failure logs and falls back
  to the default-layout transpile rather than losing the run. Every run now records
  the layout used, the backend's reported noise and the per-edge errors along that
  layout on `sampler.hardware_characterisation`.

  Note there is deliberately no sampler-side `resilience_level` (not a
  `SamplerOptions` field in the pinned runtime version) and no session sizing: the
  sampler submits a single job with `mode=backend` and never opens a session.
- Sampler error-suppression options are now reachable from the shared dispatch and
  the CLI: `build_sampler(..., options=...)`, plus `--measure_twirling` (default
  **true**, unlike bare `SamplerV2`, since a default `solve` targets hardware and
  readout error is what SQD is most sensitive to), `--dynamical_decoupling`
  (default false) and `--sampler_options '<json>'` for anything else, merged over
  the flags one level deep. `RuntimeSampler` already accepted `options` and its
  docstring called for measurement twirling on hardware, but nothing forwarded it —
  so the documented mitigation was previously unreachable through any supported
  entry point. Passing `options` to the Aer or mock sampler is now an error rather
  than a silently ignored setting.
- Circuit generation reports progress through the standard `logging` module
  (`embasi_qiskit_integration.circuit_generator.sqdrift`): the sweep plan up front,
  and the relabel hit rate per chunk plus a merged total in the parent. A run where
  `RelabelModes` found **no** permutations now logs a warning naming `time_limit` —
  previously `optimize=True` silently producing unpermuted circuits was
  indistinguishable from relabeling having worked. HiGHS's own per-solve report is
  demoted to DEBUG so it cannot bury those records.
- `build_sqdrift_circuits` now returns a `SqdriftBuildResult` (`circuits`,
  `permutations`, `randomization_indices`, `num_permutations_found`), mirroring the
  reference `SQDRIFTBuildResult`. It iterates, indexes and sizes like the circuit
  list it replaces, so existing list-style callers are unaffected.
- `circuit_run.prep`: initial-state resolution (`resolve_initial_state`,
  `hf_prep_circuit`, `bitstring_prep_circuit`, `compose_full_circuit`). The
  reference determinant is now chosen at run time and prepended to a bare ansatz
  circuit, so one generated circuit can be sampled from different reference
  states. `SQDSolver(initial_state_bitstring=...)` exposes this.
- `circuit_run.backend`: `resolve_backend` (explicit name, `least_busy`, injected
  object, or a local `Fake*` device needing no credentials) and `prepare_isa`,
  which builds one pass manager per batch instead of one per circuit.
- `circuit_run.counts`: one shared `SamplerV2`-result reader plus `merge_counts`.
- `circuit_run.build_sampler` / `SamplerKind`: the `--sampler` dispatch, declared
  once instead of separately in the CLI and the demo scripts.
- Sampler ensembles: `sampler.run(circuits, shots)` samples a batch and returns
  one counts dict per circuit. `SQDSolver(method="qdrift", num_randomizations=N)`
  pools N randomizations (total budget `N * shots`), which is markedly more
  accurate than one circuit at the same budget.
- `build_sqdrift_circuits(include_initial_state=False)` emits the bare evolution;
  circuits record `metadata["initial_state_included"]`.
- `SQDSolver.build_circuits(ham)` is public, for inspecting the exact circuits
  `solve` would sample.

### Changed
- **Behaviour:** `build_sqdrift_circuits` now relabels the modes by default
  (`optimize=True` when the `relabel` extra is present), matching the reference
  step, so the generated circuits differ from before — shorter, and sampling in a
  permuted mode order. `SQDSolver` un-permutes transparently, but code that samples
  `build_sqdrift_circuits` output directly must apply `unpermute_counts_list` (or
  pass `optimize=False` for the previous circuits). The gate-for-gate equivalence
  with the bare qDRIFT recipe now holds on the `optimize=False` path.
- **Breaking:** the `sampling` package is now `circuit_run`, and `circuits` +
  `mapping` are now `circuit_generator`, so circuit generation and circuit
  execution are separable halves that can each be replaced independently. Update
  imports from `embasi_qiskit_integration.sampling.{base,aer,runtime}` to
  `embasi_qiskit_integration.circuit_run.{base,aer,runtime}`, and from
  `embasi_qiskit_integration.circuits.*` to
  `embasi_qiskit_integration.circuit_generator.*`.
- `SQDSolver` no longer discards circuits: it previously built the full SqDRIFT
  ensemble and sampled only `circuits[0]`.
- `scripts/sqd_prototype.py` and `scripts/make_test_data.py` now drive the solver
  pipeline instead of their own copy of build → transpile → `AerSimulator.run`.
  The prototype consequently honours its `--seed` (it previously hard-coded
  `seed_simulator=42`); pass `--seed 42` to reproduce earlier output.

### Removed
- Three definitions that were never called anywhere: `circuit_run.layout.plot_layout`
  (a gate-map debugging aid that would have raised `ImportError`, since matplotlib is
  not a dependency), `circuit_run.counts.counts_from_pub_result` (a one-line wrapper
  whose own docstring steered callers to the per-binding version) and `ipc.is_rank0`
  (`rank0_solve` does its own rank check). `circuit_generator.lucj` is deliberately
  kept as a reference placeholder for the deferred LUCJ ansatz.

### Fixed
- **Breaking (minor):** `compose_full_circuit`'s `add_measure_all` parameter is now
  called `measure`, matching `build_sqdrift_circuits(measure=...)` which controls the
  same choice at generation time. The package previously used two names for one
  concept in adjacent call sites. Its docstring now also states the invariant the
  default relies on: a core passed here is expected to be unmeasured, so composing an
  already-measured core without `measure=False` appends a second `measure_all()`.
- Documentation corrections found by an internal consistency audit: the
  `build_sqdrift_circuits` docstring claimed a default sweep of 4500 circuits and a
  `num_groups` default of `[10, 15, 20]`, both stale since the default became the
  scalar `15` (a bare `method="qdrift"` call builds 1500); a docstring referenced a
  `_ExcitationCollector` class that does not exist; and the deliberate split between
  the CLI's `--optimization_level 1` and the Python API's `3` is now explained at
  both sites rather than looking like a mistake.
- `build_sampler` and the CLI now forward `n_rand_twirl`, `n_shots_per_twirl` and
  `hot_coupler_ps`, which existed on `RuntimeSampler` but were unreachable through
  either entry point.
- `n_frozen_occ` is no longer silently ignored when a selector is set. It was
  validated and then dropped, because the selector branch replaced the range that
  applied it — so `--n_frozen_occ 2 --selector concentric` froze *nothing*. Selectors
  choose virtuals and leave the occupied block alone by contract, so the freeze is
  now applied after them. The failure was quiet: `e_core` and `nelec` stayed
  mutually consistent, so energies looked right while the active space (and the
  qubit count the flag exists to bound) was larger than requested. A freeze that
  would empty the active space now raises. The accompanying log line said
  `--n_virtual is advisory when a selector is set`, which was backwards —
  `n_virtual` is passed through as the selector's `max_virtual` ceiling — and now
  reports both the virtual count and the freeze.
- `SolverResult.check_particle_number` enforces `trace(rdm1)` against the active
  electron count, and the embedding workflow now calls it (tolerance
  `--rdm_trace_tol`, default 1e-6) instead of only printing the two numbers side by
  side. The RDM becomes the next cycle's density, so a wrong particle-number sector
  previously propagated through the outer loop behind plausible energies. Measured
  deviations are ~1e-14 for FCI and SQD alike, so the check has no false-positive
  risk.
- FCIDUMP no longer reconstructs a wrong spin split from a sidecar-less open-shell
  file. `MS2` is written unsigned (pyscf's `write_head` recomputes `abs(na - nb)`
  from a `(na, nb)` pair, and the qiskit-fermions Rust reader *panics* on a leading
  `-`, so a signed header would crash the circuit path), which makes `NELEC=3,
  MS2=1` fit both `(2, 1)` and `(1, 2)`. The reader previously assumed the
  alpha-rich one, silently flipping the spin of every beta-rich Hamiltonian; it now
  raises and points at the sidecar, which remains authoritative. Latent behind the
  closed-shell-only adapter, and the old tests could not see it because their
  fixtures were all alpha-rich.
- Corrected the CLI flag spellings throughout the docs. pydantic-settings derives
  flags verbatim from field names, so every multi-word flag takes **underscores**;
  the documented hyphenated forms were rejected with a usage error. This affected
  `--optimization-level` in the README, `--n-virtual` / `--n-frozen-occ` /
  `--max-cycles` / `--mix-alpha` in the `embedding` docstring and
  `scripts/embedding_workflow.py`, and `--bond-length` in
  `scripts/make_test_data.py`. A regression test now runs every flag the README
  shows and asserts the hyphenated spelling is still rejected, so the two cannot
  drift apart again.
- Documented the `initial_state_bitstring` spin-block order. The string is
  MSB-left, so the **rightmost** `n_orbitals` characters are the alpha block —
  mirrored from how it reads. This was undocumented and easy to get backwards on
  an open-shell system (the naive reading silently populates the wrong spin
  sector, after which SQD postselects every shot away). Symmetric examples cannot
  reveal the mistake, so the docs and a regression test now use `n_alpha != n_beta`.
- `resolve_initial_state` now warns instead of silently discarding an explicit
  `initial_state_bitstring` when the core circuit already carries its own
  reference state.
- `_canonicalize_group_order` computes its per-group tie-break weight with an
  explicit `np.add.at` / `np.unique` reduction, falling back to `group_weights()`
  only when the group labels are non-contiguous — the one case where that reduction
  raises `ValueError`, since `num_groups()` is the largest label plus one. The
  fallback therefore never changes an ordering the reduction could have produced.
- `RuntimeSampler` now passes `seed` through to the ISA transpile, making
  hardware layout selection reproducible (the parameter was previously dead).
- `AerSampler` pins the simulation method (default `"statevector"`) instead of
  relying on Aer's `"automatic"` selection, so a given `seed` stays reproducible.
- `run_sqd` raises a clear error on empty counts rather than failing obscurely
  inside `BitArray.from_counts`.
- Seeded qDRIFT circuit generation is now reproducible. Diagonal terms are
  filtered *before* grouping and group labels are relabelled into a
  content-derived canonical order, so a given seed yields identical circuits
  across processes (previously it diverged even between calls in one process).
- Removed the duplicated counts-extraction helpers in the Aer and Runtime
  samplers, which disagreed on whether they returned a `BitArray` or its counts.
- Dropped the unused `mapping.to_qubit_op`: the Jordan-Wigner mapping is the
  transpiler pass the generator runs, not a separate stage.

### Known issues
- `tests/data/mock_counts.json` predates the qDRIFT determinism fix and no longer
  matches what the current pipeline generates (235 distinct bitstrings vs 254,
  with 163 in common). It is harmless — `MockSampler` replays it verbatim and
  ignores circuits — but regenerating it with
  `scripts/make_test_data.py` will shift every `MockSampler`-driven test's
  numbers, so it was left untouched.

[Unreleased]: https://github.com/IBM/embasi-qiskit-integration/commits/main
