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
from embasi_qiskit_integration.sampling.aer import AerSampler
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
from embasi_qiskit_integration.sampling.runtime import RuntimeSampler
from embasi_qiskit_integration.solvers import SQDSolver

sampler = RuntimeSampler()                          # least-busy real backend
# sampler = RuntimeSampler(backend="ibm_kingston")  # or pick one explicitly
# sampler = RuntimeSampler(optimization_level=2)    # ISA-transpile level (default 3)
res = SQDSolver(sampler, shots=100_000).solve(ham)
```

`RuntimeSampler` resolves a backend from `QiskitRuntimeService` (an explicit
`backend=`, else `least_busy`), transpiles the circuit to that backend's ISA with
`optimization_level` (default **3**), and submits via `SamplerV2` — no other code
changes. Note that a real device queues jobs, so a full SQD run can take a while.

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

[`scripts/embedding_workflow.py`](scripts/embedding_workflow.py) sketches the
full embedding flow — EmbASI low-level embedding → extract an active-space
Hamiltonian → solve with FCI/SQD → assemble the projection-based-embedding
energy → feed the 1-RDM back — with **EmbASI mocked** (backed by a real PySCF
active space) so it runs with no QM driver. Inline `# REAL:` comments mark where
a live EmbASI object plugs in:

The sketch defaults to the SQD pipeline:

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

See [`docs/api-pins.md`](docs/api-pins.md) for the resolved third-party API
surface (qiskit-addon-sqd, ffsim, qiskit-fermions, EmbASI).

## Ansatz

The primary sampling ansatz is **SqDRIFT** (`circuits/sqdrift.py`, via
`qiskit-fermions`); the LUCJ ansatz is currently on hold. See `docs/api-pins.md`
for the SqDRIFT synthesis details and caveats.

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

The `@pytest.mark.embasi` tests only run when `EMBASI_AVAILABLE=1`, and a full
run of them needs **EmbASI + a QM driver (FHI-aims)**. The EmbASI glue in this
project is otherwise fully covered in the default suite via a `FakeEmbASI` stub,
so you only need this to exercise the real embedding backend.

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

This project is licensed under the [Apache License 2.0](LICENSE).

Every source file carries an SPDX license header:

```
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
```
