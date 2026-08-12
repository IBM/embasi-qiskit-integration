# embasi-qiskit-integration

Integration library coupling **[EmbASI](https://github.com/tamm-cci/EmbASI)**
(projection-based QM/QM embedding via ASI) with the **Qiskit** stack
(`qiskit-fermions`, `qiskit-addon-sqd`, `ffsim`) to run **Sample-based Quantum
Diagonalization (SQD)** as the high-level solver inside an embedding workflow.

The entire quantum side (integrals → circuit → sampling → SQD → energy + RDMs)
runs and is tested **without** EmbASI, FHI-aims, network access, or quantum
hardware. EmbASI is only required to close the embedding loop (Phase 8).

## Prerequisites

- **Python ≥ 3.10** (developed and tested on 3.12).
- **[uv](https://docs.astral.sh/uv/)** for environment management (recommended).
- **A Rust toolchain** — required to build `qiskit-fermions`, which ships a
  native extension. Install it with:

  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  source "$HOME/.cargo/env"
  rustc --version   # confirm it is on PATH
  ```

  On macOS you can alternatively `brew install rust`. Rust is only needed for
  the optional `fermions` extra; the core quantum path (`quantum` extra) does
  not require it.

## Installation

```bash
# with uv (recommended)
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,quantum]"

# optional extras
uv pip install -e ".[fermions]"   # builds qiskit-fermions from git; needs Rust (see Prerequisites)
uv pip install -e ".[hardware]"   # IBM Quantum Runtime
```

`qiskit-fermions` is not published on PyPI, so the `fermions` extra installs it
directly from git (`git+https://github.com/Qiskit/qiskit-fermions.git`) and
compiles its Rust extension at install time.

## Quickstart

Generate the frozen N2 CAS(8o, 10e) test integrals (committed under
`tests/data`, regenerate with):

```bash
uv run python scripts/make_test_data.py   # writes n2_8o10e.{fcidump,npz} + mock_counts.json
```

### 1. Classical reference (FCISolver)

```python
from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.solvers import FCISolver

ham = fcidump.read("tests/data/n2_8o10e.fcidump")
res = FCISolver().solve(ham)
print(res.energy)            # -108.9585095430 Ha; res.rdm1 / res.rdm2 populated
```

### 2. SQD with Aer (noiseless)

```python
from embasi_qiskit_integration.circuit_run.aer import AerSampler
from embasi_qiskit_integration.solvers import SQDSolver

# SQDSolver builds the SqDRIFT ansatz, samples it, and runs the SQD loop.
# (AerSampler + SqDRIFT needs the `quantum` + `fermions` extras.)
res = SQDSolver(AerSampler(), shots=100_000, seed=24).solve(ham)
print(res.energy)            # within 2e-3 Ha of FCI; res.diagnostics carries provenance
```

For CI / offline runs, replay frozen counts with `MockSampler` instead of
`AerSampler` — same interface, fully deterministic.

### 3. SQD on real quantum hardware

The sampler is the only thing that changes.

**Set up credentials once** (install the `hardware` extra first, `uv pip install
-e ".[hardware]"`):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<IBM_TOKEN>")
```

or export `QISKIT_IBM_TOKEN` in your environment. Then swap in `RuntimeSampler`:

```python
from embasi_qiskit_integration.circuit_run.runtime import RuntimeSampler
from embasi_qiskit_integration.solvers import SQDSolver

sampler = RuntimeSampler()                          # least-busy real backend
# sampler = RuntimeSampler(backend="ibm_kingston")  # or pick one explicitly
# sampler = RuntimeSampler(optimization_level=2)    # ISA-transpile level (default 3)
res = SQDSolver(sampler, shots=100_000).solve(ham)
```

`RuntimeSampler` resolves a backend from `QiskitRuntimeService` (an explicit
`backend=`, else `least_busy`), transpiles the circuits to that backend's ISA with
`optimization_level` (default **3**), and submits them as a single `SamplerV2`
job — no other code changes. Note that a real device queues jobs, so a full SQD
run can take a while.

To rehearse the hardware path offline, name a simulated device — it resolves
locally and needs no credentials:

```python
sampler = RuntimeSampler(backend="FakeManilaV2")
```

**On real hardware, enable measurement twirling.** `SamplerV2` leaves it off, so
construct the sampler with it on (the CLI enables it for you — see §4):

```python
sampler = RuntimeSampler(options={"twirling": {"enable_measure": True}})
```

Readout error is the channel this pipeline is most sensitive to: a flipped bit
changes a sampled determinant's Hamming weight, so SQD's postselection discards
that shot — wasting budget *and* biasing the subspace toward whichever
configurations happened to survive, which moves the energy rather than just its
variance. `build_sampler` takes the same `options`, so the shared dispatch can
reach it too.

**Noise-aware layout selection** (`enable_readout_characterisation=True`) measures
the device before submitting instead of trusting its reported calibration:

```python
sampler = RuntimeSampler(enable_readout_characterisation=True)
```

It submits a short `samplomatic` twirled measure-only job (300 randomizations × 25
shots), derives the per-qubit readout error from the twirl-corrected shots, deletes
every qubit above `readout_error_threshold` (0.03) from the coupling map, searches
the survivors for 1-D chains of the circuit's width, ranks them by joint readout
fidelity `prod(1 - error)`, and pins the winner as an explicit `initial_layout`.
When pruning fragments the lattice the threshold relaxes by ×1.5 (warning when it
does) until a chain exists.

This matters for the same reason twirling does: a flipped readout bit changes a
determinant's Hamming weight, SQD postselects the shot away, and the surviving
sample is biased — so *which physical qubits you land on* moves the energy. Qiskit's
default layout passes optimize against calibration data that can be hours stale.

Costs one extra short job, and is **off by default**. It is refused on simulated
backends, whose "measured" readout error is just their configured noise model. The
chosen layout, its score, the rejected qubits and the backend's reported noise are
recorded on `sampler.hardware_characterisation` after each run. Note the chain
search assumes a **1-D** layout, which suits the SqDRIFT/LUCJ circuits here.

Quick smoke test that credentials + submission work (a 2-qubit Bell circuit, no
SQD):

```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure_all()
print(RuntimeSampler().sample(qc, shots=1024))   # -> counts dict from hardware
```

### 4. Two-process CLI handoff

The CLI defaults to the **SQD pipeline on hardware** (`--solver sqd --sampler
runtime`), so a bare `solve` needs configured IBM Quantum credentials (see §3).
Swap the sampler for `aer` (local simulation) or `mock` (frozen replay) to run
offline:

```bash
# Process A writes <jobdir>/job.fcidump, then:
uv run embasi-qiskit-integration solve <jobdir>                       # sqd on hardware (default)
uv run embasi-qiskit-integration solve <jobdir> \
    --backend ibm_kingston --optimization_level 3                     # pick a backend explicitly
uv run embasi-qiskit-integration solve <jobdir> --backend FakeManilaV2 # simulated device, no credentials
uv run embasi-qiskit-integration solve <jobdir> --sampler aer         # local noiseless simulation
uv run embasi-qiskit-integration solve <jobdir> --sampler mock \
    --counts tests/data/mock_counts.json --seed 24                    # offline, deterministic
uv run embasi-qiskit-integration solve <jobdir> --solver fci          # classical reference
# -> writes <jobdir>/result.npz (or result.ERROR on failure)
```

Defaults are `--shots 10000` per circuit, `--optimization_level 1`, `--seed 42`.

**Error suppression (runtime sampler only).** `--measure_twirling` defaults to
**true** here — unlike bare `SamplerV2`, because a bare `solve` targets hardware
and readout error is what SQD is most sensitive to (see §3). Idle-qubit
`--dynamical_decoupling` is off by default.
`--sampler_options` takes any other `SamplerV2` option as JSON, merged over the two
flags one level deep:

```bash
uv run embasi-qiskit-integration solve <jobdir> --measure_twirling false        # opt out
uv run embasi-qiskit-integration solve <jobdir> --dynamical_decoupling true     # deep circuits
uv run embasi-qiskit-integration solve <jobdir> \
    --sampler_options '{"twirling": {"num_randomizations": 64}}'                # refine twirling
uv run embasi-qiskit-integration solve <jobdir> \
    --enable_readout_characterisation true --readout_error_threshold 0.03       # measure, then pin a layout
```

**Circuit-generation knobs.** `--optimize` / `--time_limit` control mode relabeling
and `--workers` shards circuit construction across processes (see §2 above);
`--workers 0` uses one per CPU. Generation progress and the relabel hit rate are
reported through the standard `logging` module under the
`embasi_qiskit_integration` logger — a zero-permutation run warns, so
"optimization silently did nothing" is visible:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Demos

[`scripts/sqd_prototype.py`](scripts/sqd_prototype.py) runs the whole
quantum path (integrals → SqDRIFT → Aer → SQD → energy vs FCI) and asserts the
delta is ≤ 2e-3 Ha:

```bash
uv run python scripts/sqd_prototype.py
```

[`scripts/embedding_workflow.py`](scripts/embedding_workflow.py) runs the full
embedding flow against **real EmbASI** — EmbASI low-level embedding → extract an
active-space Hamiltonian → solve with SQD/FCI → assemble the
projection-based-embedding energy → feed the 1-RDM back. It drives a live
`embasi.embedding.ProjectionEmbedding` (methanol monomer, OH active fragment,
HF-in-PBE, sto-3g), so it needs EmbASI installed (see below).

By default it takes the **WF-in-DFT** path (`--xc_hl HF`: an HF mean field plus an
active-space quantum solve, with the `concentric` selector), so the default run
genuinely exercises the SQD pipeline. A Kohn-Sham `--xc_hl` (PBE0/PBE) instead
routes DFT-in-DFT, where the solver and selector are inert.

```bash
uv run python scripts/embedding_workflow.py                        # WF-in-DFT: sqd + aer (default)
uv run python scripts/embedding_workflow.py --sampler runtime      # sqd on hardware
uv run python scripts/embedding_workflow.py --handoff two-process  # file handoff
uv run python scripts/embedding_workflow.py --solver fci           # classical FCI reference
uv run python scripts/embedding_workflow.py --xc_hl PBE0           # DFT-in-DFT (solver inert)
```

It is MPI-safe (the solve runs on rank 0 and the result is broadcast), so it can
be driven the way real EmbASI runs — under `mpirun` (needs the `embed` extra for
`mpi4py`):

```bash
mpirun -n 2 uv run python scripts/embedding_workflow.py --solver sqd
```

## Ansatz

The primary sampling ansatz is **SqDRIFT** (`circuit_generator/sqdrift.py`, via
`qiskit-fermions`); the LUCJ ansatz is currently on hold.

`method="qdrift"` produces an *ensemble* of randomized circuits instead of one
exact-evolution circuit. Each is sampled at `shots` and the counts are pooled, so
the total budget is `num_randomizations * shots` — at a fixed budget this is
substantially more accurate than a single circuit:

```python
res = SQDSolver(AerSampler(), shots=5_000, method="qdrift", num_randomizations=4).solve(ham)
```

Seeded generation is reproducible across processes: randomization `i` is drawn
with seed `seed + i`, and the Hamiltonian's term groups are relabelled into a
canonical order so a given seed always maps to the same physical group.

### Mode relabeling (`optimize`)

`optimize=True` (the default when the `relabel` extra is installed) runs
`qiskit-fermions`' `RelabelModes` pass, which reorders the fermionic modes to
minimize the span of the sampled excitations. That shortens the synthesised
circuits substantially — on N2 CAS(8o,10e) the per-draw depth dropped from
390/202/413 to 369/153/273 across three randomizations.

The catch, and the reason this is not just a free win: **a relabeled circuit
samples bitstrings in the permuted mode order.** Used as-is the occupations land
on the wrong orbitals — measured against an unpermuted reference the raw counts
have a total variation distance of ~1.0, i.e. an entirely disjoint distribution.
`SQDSolver` handles both ends of this for you:

- the reference determinant is prepared in each circuit's own permuted order
  (`resolve_initial_state` reads `metadata["permutation"]`), and
- each circuit's counts are mapped back to the original order **before** the
  ensemble is pooled (`circuit_run.permutation.unpermute_counts_list`) — pooling
  first would mix mutually inconsistent mode orders.

Building circuits yourself means owning that second step:

```python
from embasi_qiskit_integration.circuit_run import unpermute_counts_list

result = build_sqdrift_circuits(ham, method="qdrift", num_randomizations=8)
counts = sampler.run(result.circuits, shots)
counts = unpermute_counts_list(counts, result.permutations)   # required!
```

By default the permutation is whatever the MILP solver returned, applied in a
single pass chain. **That is not reproducible**: the excitation-span model is
degenerate and HiGHS is not a pure function of it — solving one model repeatedly in
a single process returned one optimum five times and then a different one (depths
382 vs 368). Since the permutation is undone on the counts, that instability
reaches the pooled distribution. Pass `canonical_permutation=True` to derive the
permutation from a fixed candidate set instead, at the cost of a second
pass-manager run per randomization:

```python
build_sqdrift_circuits(ham, method="qdrift", canonical_permutation=True)
```

Requesting `optimize=True` without the `relabel` extra raises instead of silently
producing unpermuted circuits.

### Parallel generation (`workers`)

`workers=N` shards a combination's randomizations into contiguous seed-chunks,
one per worker process. Because each randomization is an independently seeded
draw, the output is byte-identical to the sequential build — only faster (N2
CAS(8o,10e), 8 draws with relabeling: 81.5s → 12.5s at `workers=8`). A
process-local operator cache keeps the expensive operator construction to once
per worker. `workers=0` means one per CPU.

Workers are spawned, so a *script* using `workers > 1` must guard its entry point
with `if __name__ == "__main__":` (the standard `multiprocessing` requirement);
without it the workers fail with `BrokenProcessPool`. Notebooks and the CLI are
unaffected.

## Package layout

The quantum path is split into two packages, so either half can be swapped out
independently:

| package | role |
| --- | --- |
| `circuit_generator/` | integrals → fermionic operator → mapped circuit (`operator.py`, `sqdrift.py`, `hf.py`) |
| `circuit_run/` | circuits → measured counts (`prep.py`, `backend.py`, `counts.py`, and the samplers) |

The seam between them is deliberate: the generator emits *bare* evolution
circuits, and `circuit_run.prep.resolve_initial_state` picks the reference
determinant and prepends it at run time. Precedence is an explicit
`initial_state_bitstring`, else Hartree-Fock from the active-space electron
counts; a circuit that already carries its own reference state (flagged in
`metadata["initial_state_included"]`) is left alone.

```python
# sample a chosen determinant instead of the HF reference — no circuit rebuild
# N2 CAS(8o,10e) is 5 alpha + 5 beta; this is its HF determinant written out:
res = SQDSolver(AerSampler(), initial_state_bitstring="0001111100011111").solve(ham)
```

**Bit order:** the string is MSB-left (matching `get_counts()`), so its *last*
character is qubit 0. Since the alpha block sits on the low qubits, the **rightmost
`n_orbitals` characters are alpha** and the leftmost are beta — mirrored from how
the string reads. A `n_alpha == n_beta` case looks the same either way, so the
asymmetric example is the one to reason from: for `n_orbitals=8` with 5 alpha and
3 beta, the determinant is `"0000011100011111"` (alpha on qubits 0–4, beta on
8–10). Cross-check against `hf_prep_circuit` if unsure — a swapped block silently
populates the wrong spin sector, and SQD will then postselect every shot away.

## Development

Enable the [pre-commit](https://pre-commit.com/) hooks (ruff lint + format,
plus basic file hygiene) once after cloning — they mirror the CI checks:

```bash
uv run pre-commit install          # installs the git hook
uv run pre-commit run --all-files  # optional: run against the whole tree
```

## Testing

The default suite runs the entire quantum + classical path with no EmbASI,
FHI-aims, network, or hardware — the EmbASI-marked tests are auto-skipped:

```bash
uv run pytest -q                      # EmbASI tests auto-skipped
uv run ruff check src && uv run mypy src
```

### Running the EmbASI-marked tests

The `@pytest.mark.embasi` tests only run when `EMBASI_AVAILABLE=1`, and they
build a real `embasi.embedding.ProjectionEmbedding`. The package's own logic
(orbital bookkeeping, downfolding, the outer loop, MPI orchestration) is covered
in the default suite via small PySCF-backed test doubles, so you only need this
to exercise the real embedding backend.

1. **Install EmbASI** via the `embed` extra (EmbASI is on PyPI; this pulls it
   plus `ase`, `asi4py`, and `mpi4py`):

   ```bash
   uv pip install -e ".[embed]"
   ```

2. **Provide a QM driver.** EmbASI communicates with a QM package through the
   ASI API; the supported driver is **FHI-aims**, built as a shared library and
   exposed via `asi4py` (typically the `ASI_LIB_PATH` environment variable
   pointing at `libaims.so`). See the
   [EmbASI docs](https://tamm-cci.github.io/EmbASI/index.html) and the
   [ASI templates](https://gitlab.com/pvst/asi) for building it. This step
   requires an FHI-aims license and is not needed for anything else in this
   project.

3. **Run the marked tests:**

   ```bash
   EMBASI_AVAILABLE=1 uv run pytest -m embasi -q
   ```

   Without a completed FHI-aims embedding run the real-EmbASI smoke test skips
   itself with an explanatory message; with the driver in place it exercises the
   extraction/feedback path against live EmbASI matrices.

## References & citation

This library couples the **EmbASI** projection-based embedding framework to the
Qiskit SQD stack. The embedding formalism — the DFT-in-DFT and WF-in-DFT energy
expressions, the level-shift projector, and the reference results this package's
docstrings cite as "the paper" (Eq. 2, 6, 8, 19; §2.2; Fig. 3B) — is described
in:

> G. Bramley, P. Stishenko, O. van Vuren, V. Blum, and A. J. Logsdail,
> *A General Pythonic Framework for DFT-in-DFT and WF-in-DFT Embedding*,
> ChemRxiv (2025), preprint. DOI: [10.26434/chemrxiv-2025-c23jf](https://doi.org/10.26434/chemrxiv-2025-c23jf).

If you use this integration, please cite the EmbASI paper above. The projection /
level-shift embedding scheme it implements originates with:

> F. R. Manby, M. Stella, J. D. Goodpaster, and T. F. Miller III,
> *A Simple, Exact Density-Functional-Theory Embedding Scheme*,
> J. Chem. Theory Comput. **8**, 2564–2568 (2012).
> DOI: [10.1021/ct300544e](https://doi.org/10.1021/ct300544e).

EmbASI itself is developed at <https://github.com/tamm-cci/EmbASI>.

## License

This project is licensed under the [Apache License 2.0](LICENSE). Every source file carries an SPDX license header:

```
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
```
