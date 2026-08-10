# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Initial open-source release of `embasi-qiskit-integration`, coupling EmbASI
  projection-based embedding with the Qiskit SQD stack.
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

### Fixed
- Documented the `initial_state_bitstring` spin-block order. The string is
  MSB-left, so the **rightmost** `n_orbitals` characters are the alpha block —
  mirrored from how it reads. This was undocumented and easy to get backwards on
  an open-shell system (the naive reading silently populates the wrong spin
  sector, after which SQD postselects every shot away). Symmetric examples cannot
  reveal the mistake, so the docs and a regression test now use `n_alpha != n_beta`.
- `resolve_initial_state` now warns instead of silently discarding an explicit
  `initial_state_bitstring` when the core circuit already carries its own
  reference state.
- `_canonicalize_group_order` reads group weights from upstream's
  `group_weights()` rather than recomputing them, fixing a `ValueError` when group
  labels are non-contiguous and keeping the ranking aligned with the qDRIFT pass
  by construction. Numerically identical on existing paths.
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
