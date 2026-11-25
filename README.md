# embasi-qiskit-integration

Integration library coupling **[EmbASI](https://github.com/tamm-cci/EmbASI)**
projection-based embedding with the **Qiskit** stack (`qiskit-fermions`,
`qiskit-addon-sqd`, `ffsim`) to run **Sample-based Quantum Diagonalization (SQD)**
inside an embedding workflow.

The quantum path (integrals → circuit → sampling → SQD → energy and RDMs) runs and
is tested without EmbASI, FHI-aims, network access or quantum hardware. EmbASI is
only needed to close the embedding loop.

## Requirements

- **Python 3.11+**. Use 3.12+ for the `sci-solvers` extra and for `--extra all`;
  the commands below use 3.12.
- **[uv](https://docs.astral.sh/uv/)** for environment management (recommended).
- **Rust** only if `qiskit-fermions` has no prebuilt wheel for your platform.
  PyPI ships wheels for macOS, manylinux and Windows, so most users need none;
  the core quantum path never needs Rust.

## Installation

The core package (numpy, scipy, PySCF, pydantic, pydantic-settings) covers the
classical path only (`FCISolver`, integrals, FCIDUMP I/O). Everything else is an
extra:

| Extra | Installs | Needed for |
|---|---|---|
| `quantum` | `qiskit`, `qiskit-aer`, `qiskit-addon-sqd`, `ffsim` | `SQDSolver`, Aer sampling |
| `fermions` | `qiskit-fermions>=0.3.1,<0.4` | SqDRIFT circuits (any non-mock SQD run) |
| `relabel` | `pyomo`, `highspy`, `ortools` | [mode relabeling](#mode-relabeling-optimize); required by `SQDSolver`'s default `optimize=True` |
| `sci-solvers` | `sci-solvers` + `torch-sci-solver[sqd]` from [IBM/sci-solvers](https://github.com/IBM/sci-solvers); pulls `torch`; Python ≥3.12 | non-`qiskit` `--sci_solver` choices (default `torch_sci`) |
| `hardware` | `qiskit-ibm-runtime`, `samplomatic` | `RuntimeSampler`, readout characterisation |
| `embed` | EmbASI (git, `qm-code-adapter`) and its ASI stack (`ase`, `asi4py`, `mpi4py`, `scalapack4py`) | `scripts/embedding_workflow.py`, EmbASI-marked tests |
| `siesta` | `asi4py` git pin (ASI flavour 3) + `h5py` | `--qm_code siesta` (SIESTA itself is not a Python dependency) |
| `mpi` | `mpich` (Linux) | an MPI runtime when the host has none |
| `chem` | `rdkit` | SMILES-derived charges in `.xyz` metadata |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `mypy`, `pre-commit`, `rdkit` | tests and lint |
| `all` | `quantum` + `fermions` + `relabel` + `hardware` + `embed` + `siesta` + `sci-solvers` | every runtime path (not `mpi`, `chem`, `dev`) |

`sci-solvers` is the only extra that installs torch (transitively). Install from
the lockfile with uv:

```bash
uv sync --python 3.12 --extra all --extra dev    # everything
uv sync --python 3.12 --extra all --extra mpi    # Linux host without a system MPI
uv sync --python 3.12 --extra quantum --extra fermions --extra relabel \
    --extra sci-solvers                          # quantum path only
source .venv/bin/activate                         # optional, instead of `uv run`
```

`uv sync` removes packages not in the selected extras, so name every extra you
need each time. Without uv: `pip install -e ".[all,dev]"`.

**MPI.** EmbASI imports `mpi4py`, which needs an MPI library at import time
(otherwise `RuntimeError: cannot load MPI library`). Use a system MPI where one
exists (for example Homebrew `open-mpi` on macOS); on a Linux host without one,
add the `mpi` extra (bundled MPICH). Do not add it on a cluster whose own
`mpirun` you use: the bundled library takes precedence and every rank runs as
rank 0 of 1.

## Usage

### Classical reference (FCI)

The committed test data is N2/6-31G CAS(8o, 10e); regenerate it with:

```bash
uv run python scripts/make_test_data.py   # writes n2_8o10e.{fcidump,npz} + mock_counts.json
```

```python
from embasi_qiskit_integration.hamiltonian import fcidump
from embasi_qiskit_integration.solvers import FCISolver

ham = fcidump.read("tests/data/n2_8o10e.fcidump")
res = FCISolver().solve(ham)
print(res.energy)  # -108.9585095430 Ha; res.rdm1 / res.rdm2 are populated
```

The SQD snippets below reuse this `ham`.

### SQD with Aer

```python
from embasi_qiskit_integration.circuit_run.aer import AerSampler
from embasi_qiskit_integration.solvers import SQDSolver

res = SQDSolver(AerSampler(), shots=100_000, seed=24).solve(ham)
print(res.energy)  # within 2e-3 Ha of FCI
```

`SQDSolver` defaults to `sci_solver="torch_sci"` (the `sci-solvers` extra) and
`optimize=True` (the `relabel` extra). Pass `sci_solver="qiskit"` for
qiskit-addon-sqd's PySCF path. For an offline, deterministic run,
`MockSampler("tests/data/mock_counts.json")` has the same sampler interface.

#### Torch-SCI subspace diagonalizer

The default SQD subspace diagonalizer is IBM/sci-solvers' PyTorch-native
`torch_sci`. Add the extra to the same sync as the rest of the quantum path; it
requires Python ≥3.12:

```bash
uv sync --python 3.12 --extra quantum --extra fermions --extra relabel --extra sci-solvers
```

```python
res = SQDSolver(
    AerSampler(),
    shots=100_000,
    seed=24,
    sci_solver="torch_sci",              # the default
    torch_sci_kwargs={"mode": "tpb"},
).solve(ham)
print(res.diagnostics["sci_solver"])     # "torch-sci-solver"
```

```bash
uv run embasi-qiskit-integration solve <jobdir> --sci_solver torch_sci
```

The device auto-detects CUDA, then Apple MPS, then CPU; override it with
`--torch_sci_device cuda|mps|cpu`. MPS runs in float32 because it has no float64.
`torch_sci_mode` is `tpb` (default) or `gdb`, and `--torch_sci_apply_ab_projection`
exposes the engine's αβ projection. Without the extra, the torch path raises an
install hint and its test skips.

#### Aer simulation methods

`AerSampler` defaults to `matrix_product_state`, which is memory-frugal and
reaches wider registers than `statevector`; use `statevector` for exact, small
spaces. Cap the MPS bond dimension with `mps_max_bond_dimension`:

```python
AerSampler(method="statevector")
AerSampler(method="matrix_product_state", mps_max_bond_dimension=64)
```

From the CLI: `--aer_method statevector`, `--mps_max_bond_dimension 64`,
`--mps_truncation_threshold`. The method is validated against the installed Aer
and recorded in `res.diagnostics["sampler_method"]`; seeds do not reproduce
across methods.

### SQD on real quantum hardware

The sampler is the only thing that changes. Configure credentials once (needs the
`hardware` extra):

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<IBM_TOKEN>")
```

or export `QISKIT_IBM_TOKEN`. Then:

```python
from embasi_qiskit_integration.circuit_run.runtime import RuntimeSampler
from embasi_qiskit_integration.solvers import SQDSolver

sampler = RuntimeSampler()                          # least-busy real backend
# sampler = RuntimeSampler(backend="ibm_kingston")  # or pick one explicitly
# sampler = RuntimeSampler(optimization_level=2)    # ISA transpile level (default 3)
res = SQDSolver(sampler, shots=100_000).solve(ham)
```

`RuntimeSampler` resolves a backend (an explicit `backend=`, else `least_busy`),
transpiles every circuit to that backend's ISA with `optimization_level`, and
submits the batch as one `SamplerV2` job.

To rehearse the hardware path offline, name a simulated device. Any `Fake*`
backend class exported by `qiskit_ibm_runtime.fake_provider` works by name, for
example `FakeMarrakesh` or `FakeNighthawk`; it resolves locally and needs no
credentials:

```python
sampler = RuntimeSampler(backend="FakeMarrakesh")
```

**Measurement twirling.** `SamplerV2` leaves it off, so enable it on hardware:

```python
sampler = RuntimeSampler(options={"twirling": {"enable_measure": True}})
```

Readout error is the channel SQD is most sensitive to: a flipped bit changes a
determinant's Hamming weight, so SQD postselects the shot away, wasting budget and
biasing the subspace.

**Readout characterisation.** `RuntimeSampler(enable_readout_characterisation=True)`
submits a short samplomatic twirled measure-only job (300 randomizations × 25
shots), prunes qubits above `readout_error_threshold` (default 0.03; relaxed by
×1.5 if no chain remains), ranks the surviving 1-D chains by joint readout
fidelity `prod(1 - error)`, and pins the best as `initial_layout`. It costs one
extra job, is off by default, and is skipped with a warning on simulated
backends. The chosen layout and backend noise are recorded on
`sampler.hardware_characterisation`. There is no sampler-side
`resilience_level` (an Estimator option); passing one raises.

### CLI

The CLI's `solve` is process B of a two-process handoff: process A writes
`<jobdir>/job.fcidump` (plus the `.npz` sidecar), then:

```bash
uv run embasi-qiskit-integration solve <jobdir>                       # sqd + runtime (default)
uv run embasi-qiskit-integration solve <jobdir> --backend ibm_kingston
uv run embasi-qiskit-integration solve <jobdir> --backend FakeMarrakesh  # simulated, no credentials
uv run embasi-qiskit-integration solve <jobdir> --sampler aer          # local noiseless
uv run embasi-qiskit-integration solve <jobdir> --sampler mock \
    --counts tests/data/mock_counts.json --seed 24                     # frozen replay
uv run embasi-qiskit-integration solve <jobdir> --solver fci           # classical reference
```

A bare `solve` uses the runtime sampler and needs configured credentials; use
`--backend FakeMarrakesh`, `--sampler aer`, or `--sampler mock` to run offline.
Output is `<jobdir>/result.npz` (or `result.ERROR` on failure). CLI defaults are
`--shots 1000` per circuit, `--optimization_level 1`, `--seed 42`; `--watch`
waits for `job.fcidump` (default timeout 300 s). SQD defaults to
`sci_solver="torch_sci"` (the `sci-solvers` extra); pass `--sci_solver qiskit`
for qiskit-addon-sqd's PySCF path.

**`--sqd_method`** selects how the SqDRIFT ansatz is built (also accepted by
`scripts/embedding_workflow.py`):

- `exact`: one full-evolution circuit per `--evolution_time`; `--num_randomizations`
  and `--num_groups` are ignored.
- `qdrift` (default): `--num_randomizations` (default 500) circuits per
  `(evolution_time, num_groups)`, each sampled at `--shots` and pooled. The real
  budget is `len(evolution_time) * len(num_groups) * num_randomizations * shots`.
  With the defaults (three evolution times, one `num_groups=10` point, 500
  randomizations) a bare run builds 1500 circuits and 500 relabel solves; narrow
  the axes before pointing it at hardware.

```bash
uv run embasi-qiskit-integration solve <jobdir> --sqd_method exact
uv run embasi-qiskit-integration solve <jobdir> --evolution_time 1.0 \
    --num_groups 15 --num_randomizations 100
```

**Error suppression (runtime sampler only).** `--measure_twirling` defaults to
true here, unlike bare `SamplerV2`; `--dynamical_decoupling` is off by default.
`--sampler_options` takes any other `SamplerV2` option as JSON and merges it over
those flags one level deep:

```bash
uv run embasi-qiskit-integration solve <jobdir> --measure_twirling false
uv run embasi-qiskit-integration solve <jobdir> --dynamical_decoupling true
uv run embasi-qiskit-integration solve <jobdir> \
    --sampler_options '{"twirling": {"num_randomizations": 64}}'
uv run embasi-qiskit-integration solve <jobdir> \
    --enable_readout_characterisation true --readout_error_threshold 0.03
```

**Circuit-generation knobs.** `--optimize`, `--time_limit`, `--relabel_solver`,
`--cpsat_workers`, `--cpsat_deterministic`, `--per_time_randomization` and
`--workers` control relabeling and parallel generation (`--workers 0` means one
per CPU). Progress and the relabel hit rate go through the
`embasi_qiskit_integration` logger (`logging.basicConfig(level=logging.INFO)`); a
zero-permutation run warns.

### Demos and embedding workflow

[`scripts/sqd_prototype.py`](scripts/sqd_prototype.py) runs the whole quantum
path (integrals → SqDRIFT → Aer → SQD → energy vs FCI) and asserts the delta is
at most 2e-3 Ha. Without `qiskit-fermions` it replays `mock_counts.json`:

```bash
uv run python scripts/sqd_prototype.py
```

[`scripts/embedding_workflow.py`](scripts/embedding_workflow.py) runs the full
embedding flow against real EmbASI: low-level embedding → active-space
Hamiltonian → SQD/FCI → projection-based-embedding energy → 1-RDM feedback. It
drives a live `embasi.embedding.ProjectionEmbedding` (methanol monomer, OH active
fragment, HF-in-PBE, sto-3g), so it needs EmbASI (the `embed` extra) and the
quantum-path extras, including `relabel` for the default `optimize=True` and
`sci-solvers` for the default `torch_sci` diagonalizer.

```bash
uv run python scripts/embedding_workflow.py                        # WF-in-DFT: sqd + aer (default)
uv run python scripts/embedding_workflow.py --sampler runtime      # sqd on hardware
uv run python scripts/embedding_workflow.py --handoff two-process  # file handoff
uv run python scripts/embedding_workflow.py --solver fci           # classical FCI reference
uv run python scripts/embedding_workflow.py --xc_hl PBE0           # DFT-in-DFT (solver inert)
```

By default it takes the WF-in-DFT path (`--xc_ll PBE`, `--xc_hl HF`) with the
`concentric-cl` selector. A Kohn-Sham `--xc_hl` (PBE0/PBE) routes DFT-in-DFT,
where the solver and selector are inert.

#### Per-cycle diagnostics to CSV (`--diagnostics_csv`)

`--diagnostics_csv <path>` appends one row per outer-loop cycle (off by default):

```bash
uv run python scripts/embedding_workflow.py --xyz mol.xyz --active_atoms '[0,1]' \
    --max_cycles 30 --shots 1000 --diagnostics_csv output/diagnostics.csv
```

The file is appended to, not overwritten: the header is written only when the
file is missing or empty, and `run_id` (a per-run UUID) separates runs. Rows are
flushed immediately, so a killed run keeps its completed cycles. A bad path or
missing datum is caught and reported as `[diagnostics] skipped cycle N: ...`;
only rank 0 writes, and parent directories are created. Columns cover run and
geometry identity, cycle state, active space, energy terms, convergence/density
and SQD subspace size; the authoritative list is `DIAGNOSTICS_CSV_COLUMNS` in
[`diagnostics_csv.py`](src/embasi_qiskit_integration/diagnostics_csv.py).
`--output_path` instead appends one summary line per run.

#### Starting from an `.xyz`

Both scripts have built-in geometries (methanol monomer for
`embedding_workflow.py`, N2 for `make_test_data.py`); pass `--xyz` to use your
own:

```bash
uv run python scripts/embedding_workflow.py --xyz mol.xyz --active_atoms '[1,5]'
uv run python scripts/make_test_data.py --xyz tests/data/methanol.xyz \
    --basis sto-3g --ncas 4 --nelecas 4 --stem methanol_4o4e
```

The comment line carries `;`-separated metadata:

```text
6
smiles=CO; charge=0
C     0.000000    0.000000    0.000000
...
```

`charge=` and `smiles=` are optional. If both are given they are cross-checked
and a contradiction is refused rather than silently picking one. SMILES-derived
charge needs the `chem` extra (rdkit); without it, use `charge=`.
`--active_atoms` indexes the `.xyz` atom order.

The workflow is MPI-safe (the solve runs on rank 0 and the result is broadcast),
so it can be driven under `mpirun` like a real EmbASI run (needs `mpi4py` from
the `embed` extra and a matching MPI library; see [Installation](#installation)):

```bash
mpirun -n 2 uv run python scripts/embedding_workflow.py --solver sqd
```

## Ansatz

The primary sampling ansatz is **SqDRIFT** (`circuit_generator/sqdrift.py`, via
`qiskit-fermions`); LUCJ is on hold (`build_lucj_circuit` raises).

`method="qdrift"` samples an *ensemble* of randomized circuits instead of one
exact-evolution circuit. Each is sampled at `shots` and the counts are pooled, so
the total budget is `num_randomizations * shots` per `(time, num_groups)` point:

```python
res = SQDSolver(AerSampler(), shots=5_000, method="qdrift", num_randomizations=4).solve(ham)
```

Seeded generation is reproducible across processes: randomization `i` uses seed
`seed + i`, and the Hamiltonian's term groups are relabelled into a canonical
order so a seed always maps to the same physical group.

### Mode relabeling (optimize)

`optimize=True` (the default of `SQDSolver` and `scripts/embedding_workflow.py`;
the `solve` CLI enables it when the `relabel` extra is installed) runs
`qiskit-fermions`' `RelabelModes` pass, reordering fermionic modes to shorten
each circuit. A relabeled circuit samples in the permuted mode order, so the
reference determinant must be prepared in that order and the counts mapped back
before pooling. `SQDSolver` does both; if you build circuits yourself, you must
unpermute:

```python
from embasi_qiskit_integration.circuit_generator.sqdrift import build_sqdrift_circuits
from embasi_qiskit_integration.circuit_run import unpermute_counts_list
from embasi_qiskit_integration.circuit_run.aer import AerSampler

result = build_sqdrift_circuits(
    ham, method="qdrift", num_randomizations=8, include_initial_state=True
)
counts = AerSampler().run(result.circuits, 10_000)
counts = unpermute_counts_list(counts, result.permutations)  # required
```

`relabel_solver` is `cpsat` (default; OR-Tools) or `highs`, each capped by
`time_limit` (10 s); CP-SAT's parallel search (`cpsat_workers`, default 1) wins
above roughly 16 modes. Relabeling buys little under noiseless Aer, so pass
`--optimize False` there and keep it on for hardware. The permutation is cached
per draw, so a time sweep solves each draw once; `per_time_randomization=True`
gives each time its own draws, `relabel_cache=False` disables the cache, and
`cpsat_deterministic=True` makes CP-SAT reproducible but weaker. Tied optima can
otherwise vary between runs. On macOS, import `highspy` before `ortools` if you
import both directly; the CP-SAT backend already does.

### Parallel generation (workers)

`workers=N` shards each combination's randomizations into contiguous seed-chunks
across processes; the output is byte-identical to the sequential build.
`workers=0` means one per CPU, which on a batch node is the whole node, so give
an explicit count there. Keep `workers` at or below the cores you have: relabel
solves are wall-clock limited, so an oversubscribed one can stop at a different
permutation and change the sampled counts. Scripts using `workers > 1` need the
usual `if __name__ == "__main__":` guard; notebooks and the CLI do not.

## Package layout

| package | role |
| --- | --- |
| `circuit_generator/` | integrals → fermionic operator → mapped circuit (`operator.py`, `sqdrift.py`, `hf.py`) |
| `circuit_run/` | circuits → measured counts (`prep.py`, `backend.py`, `counts.py`, samplers) |

The generator emits bare evolution circuits; `circuit_run.prep.resolve_initial_state`
prepends the reference at run time. Precedence is an explicit
`initial_state_bitstring`, else Hartree-Fock from the active-space electron
counts; a circuit that already includes its state
(`metadata["initial_state_included"]`) is left alone.

### Initial state and bit order

```python
# N2 CAS(8o,10e) HF determinant (5 alpha + 5 beta); optimize=False keeps the mode order.
res = SQDSolver(
    AerSampler(), initial_state_bitstring="0001111100011111", optimize=False
).solve(ham)
```

The string is MSB-left (matching `get_counts()`), so its *last* character is
qubit 0. The alpha block sits on the low qubits, so the **rightmost `n_orbitals`
characters are alpha** and the leftmost are beta. For `n_orbitals=8` with 5 alpha
and 3 beta, the determinant is `"0000011100011111"` (alpha on qubits 0-4, beta on
8-10). Check against `hf_prep_circuit` if unsure: a swapped block silently
populates the wrong spin sector, and SQD then postselects every shot away.
`initial_state_bitstring` cannot be combined with `optimize=True`.

## Development

`uv run pre-commit install` enables the [pre-commit](https://pre-commit.com/)
hooks (ruff lint + format, plus file hygiene) that mirror the CI checks.

## Testing

`uv run pytest -q` runs the quantum and classical paths with no EmbASI, FHI-aims,
network or hardware (EmbASI-marked tests auto-skip);
`uv run ruff check src && uv run mypy src` lints.

### EmbASI-marked tests

The `@pytest.mark.embasi` tests only run when `EMBASI_AVAILABLE=1`, and they build
a real `embasi.embedding.ProjectionEmbedding`. Install EmbASI via the `embed`
extra (it pulls `ase`, `asi4py`, `mpi4py` and `scalapack4py`; `mpi4py` needs an
MPI library, see [Installation](#installation)):

```bash
uv sync --python 3.12 --extra all --extra dev
EMBASI_AVAILABLE=1 uv run pytest -m embasi -q
```

Provide an FHI-aims shared library through the ASI API (typically `ASI_LIB_PATH`
pointing at `libaims.so`). FHI-aims requires a licence and is not needed for
anything else here; without a completed embedding run the real-EmbASI smoke test
skips itself with an explanatory message.

## SIESTA backend (WF-in-DFT)

`--qm_code siesta` is **WF-in-DFT only**: PBE (`--xc_ll`) is the low-level
environment, the high level is a wavefunction calculation (`--xc_hl HF`; the
correlated solve is SQD or FCI), and the two-electron integrals come from
SIESTA-QCOMP. EmbASI drives SIESTA in-process through ASI flavour 3.

The ASI interface requires a **neutral, unpolarized Gamma-point calculation on a
single MPI rank**. The backend refuses charged systems, a non-HF `--xc_hl`,
`--spin`/`--unrestricted`, `--relax_hf` (SIESTA has no pure-HF functional), a
multi-rank run, and a non-default Gaussian `--basis` (use `--siesta_basis`).

**SIESTA is not a Python dependency.** The `siesta` extra installs only the ASI
wrapper (`asi4py` 1.4.10 from ASI main; PyPI's 1.4.9 maps flavours 0-2 only) and
`h5py`. Build SIESTA with
`-DSIESTA_WITH_ASI=ON -DSIESTA_WITH_QCOMP=ON -DSIESTA_WITH_HYBRIDXC=ON -DBUILD_SHARED_LIBS=ON`
and libint 1.x with `MAX_AM_ERI=5`; see
[`scripts/patches/README.md`](scripts/patches/README.md) for the NetCDF and
reinitialization details, or use the Docker harness below. Point `ASI_LIB_PATH`
at the resulting `libsiesta.so` (or `.dylib`) and `SIESTA_PP_PATH` at the
`<Element>.psml` pseudopotentials. No `siesta` executable is needed: the library
is loaded in-process. SIESTA itself is GPL-3.0 and is not vendored here.

### Run the workflow

From the repository root:

```bash
# Runtime extras (`sci-solvers` provides torch-sci; `dev` runs the tests below).
uv sync --python 3.12 --extra quantum --extra fermions --extra sci-solvers \
    --extra embed --extra siesta --extra dev

export ASI_LIB_PATH=/path/to/libsiesta.so
export SIESTA_PP_PATH=/path/to/pseudopotentials

# H2O with A = O (atom 0): the validated 6e / 7-orbital active space.
cat > water.xyz <<'EOF'
3
water
O  0.000000  0.000000  0.117300
H  0.000000  0.757200 -0.469200
H  0.000000 -0.757200 -0.469200
EOF

uv run python scripts/embedding_workflow.py \
    --qm_code siesta --xc_ll PBE --xc_hl HF --spin 0 \
    --xyz water.xyz --active_atoms '[0]' \
    --a_nmos 3 --selector none --n_virtual 4 \
    --siesta_basis DZP --siesta_mesh_cutoff 200 \
    --solver sqd --sampler aer --optimize False --max_cycles 1
```

The default subspace diagonalizer is `torch_sci` (the sync above installs
`sci-solvers`). `--max_cycles 1` is one embedding pass; drop it for the
self-consistent outer loop (default 15). `--optimize False` skips relabeling,
which buys nothing under noiseless Aer; on hardware leave it on and add the
`relabel` extra. Use `--solver fci` for a classical cross-check, or
`--sampler runtime` for hardware (add the `hardware` extra). If PySCF's h5py
clashes with the HDF5 that libsiesta/NetCDF loads, import PySCF before SIESTA or
run the solve through the two-process handoff (`--handoff two-process`; see
[CLI](#cli)). The default qDRIFT sweep is three evolution times, one
`num_groups=10` point, and 500 randomizations per cycle (1500 circuits); narrow
`--evolution_time` / `--num_groups` / `--num_randomizations` for a quicker run.

### Docker harness

Without a local QCOMP build, the harness clones the private `qcomp-5.4-asi`
branch over SSH, applies the maintained FFT-reinit patch, builds libint/SIESTA in
a cached image and volume, and runs the real WF test (25-45 min cold, about 5 min
warm). The image builds and uses its own MPI-enabled HDF5/NetCDF stack.

```bash
SIESTA_DOCKER_INTEGRATION=1 uv run pytest -q tests/test_siesta_docker_integration.py
```

It skips without Docker, on CI, or when the default SSH clone has no `~/.ssh`;
set `SIESTA_DOCKER_REMOTE` to override the remote. See
[`docker/siesta-asi/README.md`](docker/siesta-asi/README.md) for the cache, logs
and reset instructions.

To run a test directly against the cached image, use the tag the harness prints
(the SHA-12 of `docker/siesta-asi/Dockerfile` plus
`scripts/build_parallel_netcdf.sh`):

```bash
docker run --rm \
  --mount type=bind,src="$PWD",dst=/repo,readonly \
  --mount type=volume,src=embasi-siesta-asi-cache,dst=/cache \
  -e SIESTA_QCOMP_REAL=1 \
  -e ASI_LIB_PATH=/cache/build/Src/libsiesta.so \
  -e SIESTA_PP_PATH=/cache/siesta/Tests/Pseudos \
  -e PYTHONPATH=/repo/src \
  -e LD_LIBRARY_PATH=/cache/build/Src:/opt/parallel-netcdf/lib:/opt/parallel-netcdf/hdf5/lib \
  -e TMPDIR=/cache/pytest-tmp \
  -w /cache/work "embasi-siesta-asi-builder:<tag>" \
  /cache/venv/bin/pytest -q /repo/tests/test_siesta_qcomp_real.py
```

### Native build

For a native build without Docker, use
[`scripts/build_siesta_native.sh`](scripts/build_siesta_native.sh) (Linux and
macOS; see [`scripts/BUILD_SIESTA_NATIVE.md`](scripts/BUILD_SIESTA_NATIVE.md)).
It produces the same ASI/QCOMP `libsiesta` for local development:

```bash
scripts/build_siesta_native.sh --parallel-netcdf
```

The script builds a cached MPI HDF5/NetCDF-C/NetCDF-Fortran stack; alternatively,
point `NC_CONFIG`/`NF_CONFIG`/`NETCDF_ROOT` at an existing parallel installation.

### Opt-in real tests

These need a QCOMP build (`ASI_LIB_PATH`, `SIESTA_PP_PATH`), local or in the
Docker image:

```bash
SIESTA_QCOMP_REAL=1 uv run pytest -q \
    tests/test_siesta_qcomp_real.py tests/test_siesta_same_basis.py \
    tests/test_siesta_molecule_matrix.py
SIESTA_ASI_REAL=1 uv run pytest -q tests/test_siesta_asi_real.py
```

All of them skip on CI.

### Validation status

Proven against a real H2O/PBE/DZP QCOMP build (A=O, `--a_nmos 3`, 6 electrons in
7 A orbitals; `tests/test_siesta_qcomp_real.py`) and the H2O/H2/CH4/NH3 matrix
(`tests/test_siesta_molecule_matrix.py`):

- electron count and active-space size;
- active orbitals S-orthonormal (`C_active^T S C_active = I`);
- density trace (`Tr[S D_A] = N_A`, plus the total and A+B sums);
- zero projector leak in the active space;
- Hermitian `h1`, and every chemists'-notation `h2` permutation symmetry;
- the energy-assembly identity
  `total = e_low_total - e_low_A + e_high_A + correction`;
- same-pseudopotential FCI: `pyscf.fci.direct_spin1.kernel` runs directly on the
  SIESTA `h1`/`h2`/`e_core` (pseudopotential included) and agrees with
  `FCISolver` to 1e-9 Ha;
- QCOMP ERIs against an exact PySCF primitive-Gaussian reconstruction of the
  same `sphi` basis, including stored off-diagonal quartets (H2O and CH4; H2 has
  no stored quartet confined to one of its equivalent H atoms;
  `tests/test_siesta_same_basis.py`).

Only exact same-pseudopotential/same-basis comparisons are made. PySCF cannot
read SIESTA's PSML pseudopotential format, so no all-electron Gaussian low-level
PBE reference is used. QCOMP screens ERIs, so stored values and absolute FCI
totals are not bit-for-bit reproducible; the quartet count and the invariants are
stable, and the same-basis test checks stored off-diagonal entries rather than
treating absent ones as zeros.

### Removing the temporary shim

`--qm_code siesta` works today only because this repository ships a temporary
compatibility shim. It exists until EmbASI/SIESTA provide the required adapter
and interface upstream. The shim is self-contained in this repository, is
exercised only by the SIESTA path, and can be deleted once upstream support
lands. The removable pieces are:

- `src/embasi_qiskit_integration/backends/_embasi_compat.py` and its call sites:
  `install_siesta_adapter()` in `backends/siesta.py`,
  `prepare_projection_embedding()` in `embedding.py`, `prime_layer()` in
  `backends/callback.py`, and the `write_inputfiles` alias in `backends/siesta.py`;
- every marker tagged `TODO(embasi-siesta-shim)` (grep the tag; the module
  docstring has the full checklist), including the shim-marked cases in
  `tests/test_siesta_asi.py`;
- `same_density_reference_energy` and the shim's input-density capture in
  `backends/siesta.py`, once upstream EmbASI/ASI exposes `E_bare(D_A)` and the
  low-level layer energy through one API;
- the `asi4py` git pin in the `siesta` extra and the matching `[tool.uv.sources]`
  override (`TODO(asi4py-1.4.10)`), once `asi4py >= 1.4.10` is on PyPI;
- the FFT-reinitialization patch in `scripts/patches/`
  (`TODO(siesta-asi-embasi-fixes)`), once that fix is merged upstream.

## References

The embedding formalism (the DFT-in-DFT and WF-in-DFT energy expressions, the
level-shift projector, and the reference results cited in the package
docstrings) is described in:

> G. Bramley, P. Stishenko, O. van Vuren, V. Blum, and A. J. Logsdail,
> *A General Pythonic Framework for DFT-in-DFT and WF-in-DFT Embedding*,
> ChemRxiv (2025), preprint. DOI: [10.26434/chemrxiv-2025-c23jf](https://doi.org/10.26434/chemrxiv-2025-c23jf).

If you use this integration, please cite the EmbASI paper above. The projection /
level-shift embedding scheme originates with:

> F. R. Manby, M. Stella, J. D. Goodpaster, and T. F. Miller III,
> *A Simple, Exact Density-Functional-Theory Embedding Scheme*,
> J. Chem. Theory Comput. **8**, 2564-2568 (2012).
> DOI: [10.1021/ct300544e](https://doi.org/10.1021/ct300544e).

EmbASI itself is developed at <https://github.com/tamm-cci/EmbASI>.

## License

This project is licensed under the [Apache License 2.0](LICENSE). Every source
file carries an SPDX license header:

```text
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
```
