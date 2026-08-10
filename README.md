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

**On real hardware, enable measurement twirling** — it is off by default:

```python
sampler = RuntimeSampler(options={"twirling": {"enable_measure": True}})
```

Quick smoke test that credentials + submission work (a 2-qubit Bell circuit, no
SQD):

```python
from qiskit import QuantumCircuit
qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure_all()
print(RuntimeSampler().sample(qc, shots=1024))   # -> counts dict from hardware
```

### 4. Two-process CLI handoff

The CLI defaults to the **SQD pipeline** (`--solver sqd --sampler aer`); swap the
sampler for `mock` (offline) or `runtime` (hardware):

```bash
# Process A writes <jobdir>/job.fcidump, then:
uv run embasi-qiskit-integration solve <jobdir>                       # sqd + aer (default)
uv run embasi-qiskit-integration solve <jobdir> --sampler runtime     # sqd on hardware
uv run embasi-qiskit-integration solve <jobdir> --sampler runtime \
    --backend ibm_kingston --optimization-level 3
uv run embasi-qiskit-integration solve <jobdir> --sampler mock \
    --counts tests/data/mock_counts.json --seed 24                    # offline, deterministic
uv run embasi-qiskit-integration solve <jobdir> --solver fci          # classical reference
# -> writes <jobdir>/result.npz (or result.ERROR on failure)
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
PBE-in-PBE, sto-3g), so it needs EmbASI installed (see below).

It defaults to the SQD pipeline:

```bash
uv run python scripts/embedding_workflow.py                        # sqd + aer (default)
uv run python scripts/embedding_workflow.py --sampler runtime      # sqd on hardware
uv run python scripts/embedding_workflow.py --handoff two-process  # file handoff
uv run python scripts/embedding_workflow.py --solver fci           # classical reference
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

## License

This project is licensed under the [Apache License 2.0](LICENSE). Every source file carries an SPDX license header:

```
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
```
