# Native SIESTA-QCOMP/ASI build (Linux and macOS)

`scripts/build_siesta_native.sh` is the non-Docker alternative to
`docker/siesta-asi/Dockerfile` and the build part of `tests/_siesta_docker.py`.
It builds the `qcomp-5.4-asi` branch with the same options:

```
SIESTA_WITH_ASI=ON  SIESTA_WITH_QCOMP=ON  SIESTA_WITH_HYBRIDXC=ON
BUILD_SHARED_LIBS=ON  MPI  ScaLAPACK  NetCDF  FFTW
bundled External/libint with MAX_AM_ERI=5
```

It produces `libsiesta.so` (Linux) or `libsiesta.dylib` (macOS) for EmbASI's
in-process ASI driver.  SIESTA is **not** installed as a Python package and the
script never runs pip.  The maintained FFT-reinit patch is applied
idempotently.  Parallel NetCDF is required.

## Prerequisites

macOS (Homebrew):

```sh
brew install cmake ninja pkg-config gcc open-mpi scalapack openblas fftw
```

Linux (Debian/Ubuntu; the Docker image uses the same packages):

```sh
sudo apt install build-essential cmake ninja-build gfortran g++ pkg-config git \
  autoconf automake libtool m4 zlib1g-dev \
  openmpi-bin libopenmpi-dev libscalapack-openmpi-dev libopenblas-dev \
  libfftw3-dev
```

The parallel-stack build additionally needs `curl`, `tar` and `make` (present
by default on macOS and in the Docker image) and network access on its first
run.  A Python environment with `asi4py >= 1.4.10`, EmbASI, PySCF, ASE,
mpi4py, h5py and pytest is still needed for the test; the project `.venv` is
used when present.  The script only prints the environment and test command.

The default CMake target also builds SIESTA's utilities (and `siesta`), matching
the Docker harness and making `cmake --install` complete; this is the bulk of a
cold build.  A cold macOS build with dependencies already installed took about
16 minutes on a 16-core M4 Max.  Reruns with unchanged inputs are cache hits
(about 10 seconds).

## Quick start

Managed clone (private GitLab project over SSH; your agent/keys are used).
Use `--parallel-netcdf` to build a cached MPI stack, or omit it when
`NC_CONFIG`/`NF_CONFIG` already point at a parallel NetCDF build:

```sh
# build MPI HDF5 + NetCDF-C + NetCDF-Fortran and SIESTA
scripts/build_siesta_native.sh --parallel-netcdf
```

The script uses `ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new` for
git by default (fail fast instead of prompting).  Export `GIT_SSH_COMMAND` to
override it.

Existing checkout (used as-is; HEAD is never reset), self-built parallel stack:

```sh
scripts/build_siesta_native.sh \
  --siesta-src "$HOME/src/siesta-qcomp" \
  --build-dir "$HOME/build/siesta-asi" \
  --install-prefix "$HOME/opt/siesta-asi" \
  --parallel-netcdf
```

Useful options: `--clone-dir`, `--remote`, `--branch`, `--update`,
`--no-fetch`, `--jobs`, `--clean`, `--force`, `--stage-root`; run `--help` for
the full list.  Environment overrides include `CC`, `CXX`, `FC`, `MPICC`,
`MPICXX`, `MPIFC`, `MPI_HOME`, `CMAKE`, `CMAKE_GENERATOR`, `NETCDF_ROOT`,
`NC_CONFIG`, `NF_CONFIG`, `SCALAPACK_LIBRARY`, `BLAS_LIBRARY`,
`SIESTA_REMOTE` and `SIESTA_BRANCH`.

The workflow is idempotent: source/submodule updates, patch application, a
libint stamp and a SIESTA configure/build stamp are checked on every run.
`--clean` removes the build trees (not the source or install prefix);
`--force` re-runs configure/build without deleting the trees.

## NetCDF: parallel is required

SIESTA-QCOMP writes ERIs with `NF90_MPIIO`; a NetCDF library without MPI-IO
fails at runtime.  The script therefore requires a parallel NetCDF-C +
NetCDF-Fortran installation: either build the cached stack with
`--parallel-netcdf` (recommended), or point `NC_CONFIG`/`NF_CONFIG`/
`NETCDF_ROOT` at an existing parallel build.  It checks
`nc-config --has-parallel4` and `nf-config`, rejects a non-parallel
installation with a clear error, and always sets
`SIESTA_WITH_NCDF_PARALLEL=ON`.

### Self-built parallel stack (`--parallel-netcdf`)

`scripts/build_parallel_netcdf.sh` is a reusable helper shared by the native
script and the Docker image.  It builds, with the active MPI wrappers:

- HDF5 1.14.6 (CMake/two-level namespace on macOS; autotools on Linux; shared,
  HL) under `<stage-root>/parallel-netcdf/install/hdf5`;
- NetCDF-C 4.9.3 (`--enable-parallel-tests`, shared) and NetCDF-Fortran 4.6.2
  (parallel, shared) under `<stage-root>/parallel-netcdf/install`;
- downloads, extracted sources, build trees, logs and the probe under
  `<stage-root>/parallel-netcdf/work`.

The helper pins source SHA-256 checksums, writes per-component and summary
stamps, and runs a real Fortran/MPI probe that calls
`nf90_create(..., IOR(NF90_NETCDF4, NF90_MPIIO), comm=MPI_COMM_WORLD%mpi_val)`,
`nf90_var_par_access` and a write.  SIESTA is built only after the probe
reports `PARALLEL_NETCDF_PROBE_OK`.  `--clean` passes `--clean` to the helper
(rebuilds the stack, keeps downloads/sources); `--force` only forces the SIESTA
reconfigure/build.  A cache-hit rerun takes seconds; the cold helper build took
about three minutes on the 16-core M4 Max.

On macOS, OpenMPI's `mpicc`/`mpifort` wrappers append `-Wl,-flat_namespace`.
The helper therefore builds HDF5 with CMake (not autotools/libtool) and uses
generated no-flat MPI wrappers for the HDF5/NetCDF steps, keeping the stack's
two-level namespace.  Without this, a Python process that imports PySCF/h5py
first (h5py bundles HDF5 2.x) lets h5py's HDF5 interpose on the stack's HDF5
1.14 symbols and NetCDF crashes in `H5P_set`/`H5CX_pop` during `ASI_init`.
`libsiesta` itself references NetCDF, not HDF5 directly, so its own wrapper
flag does not affect the stack.

The normal Docker path uses the same helper at image-build time (prefix
`/opt/parallel-netcdf`) and therefore also uses `SIESTA_WITH_NCDF_PARALLEL=ON`.

## Platform branch logic

| Item | Linux | macOS |
| --- | --- | --- |
| Shared library | `Src/libsiesta.so` | `Src/libsiesta.dylib` |
| Runtime path | `LD_LIBRARY_PATH` | `DYLD_LIBRARY_PATH` |
| BLAS/LAPACK | OpenBLAS (LAPACK in the same library) | Homebrew OpenBLAS; Apple Accelerate is avoided because SIESTA rejects its BLAS return convention |
| MPI | OpenMPI/MPICH wrappers via `mpicc`/`mpifort` | Homebrew OpenMPI wrappers; mpi4py must link the same `libmpi` |
| ScaLAPACK | distro `scalapack-openmpi` | Homebrew `scalapack` (linked against Homebrew OpenMPI) |
| CMake 4 | not needed on Ubuntu 24.04 | `CMAKE_POLICY_VERSION_MINIMUM=3.5` is exported for libint 1.x and its nested generator |

`ASI_LIB_PATH` keeps its name on both platforms and may point at a `.dylib`.
`libsiesta` links Homebrew/NetCDF/OpenBLAS/ScaLAPACK libraries by absolute
install names, so `DYLD_LIBRARY_PATH` is mostly needed for SIESTA's own shared
component libraries in the build tree.  Avoid launching the test through a
SIP-protected system binary; use the project virtualenv's Python.

The bundled libint is built on both platforms.  Do not substitute a distro
libint: Ubuntu's `libint-dev` uses `MAX_AM_ERI=7` and segfaults in `nao2gto`.

## Validation

After a build, `otool -L Src/libsiesta.dylib` (macOS) or
`ldd Src/libsiesta.so` (Linux) should show MPI, ScaLAPACK, NetCDF, FFTW and the
Fortran runtime.  Then:

```sh
source build/siesta-native/env.sh
.venv/bin/python -m pytest -q tests/test_siesta_qcomp_real.py  # env.sh exports SIESTA_QCOMP_REAL=1
```

The test is the same one the Docker harness runs
(`SIESTA_QCOMP_REAL=1`, `ASI_LIB_PATH`, `SIESTA_PP_PATH`, one MPI rank,
`OMP_NUM_THREADS=1`).  The ASI interface supports only one rank, unpolarized
Gamma-point sampling and the WF-in-DFT/PBE path.

## Validation status

- **Docker route**: the full harness passed with validated image
  `embasi-siesta-asi-builder:04d0c9141f58`. `1 passed, 26 warnings` in
  297.11 s inside the container (container exit 0; outer pytest 476.77 s),
  with `netcdf=parallel`, `NetCDF_PARALLEL=TRUE`, and only the maintained FFT
  patch applied. Removing the system NetCDF packages changes the Dockerfile
  hash; the current tag is `embasi-siesta-asi-builder:0b38b80bbf7f`. The normal
  parallel stack/path is unchanged, but that final tag was not re-run here.
- **Native macOS parallel route (final)**: the helper built the CMake-HDF5
  1.14.6 stack with no-flat MPI wrappers and explicit HDF5 rpaths; the
  installed HDF5, NetCDF-C and NetCDF-Fortran dylibs all report `TWOLEVEL`,
  `nc-config --has-parallel4` = `yes`, and the `NF90_MPIIO` probe printed
  `PARALLEL_NETCDF_PROBE_OK`. The native script built and installed SIESTA
  against it (exit 0; CMake `NetCDF_PARALLEL=TRUE`; `libsiesta.dylib` links
  the stack's `libnetcdf`/`libnetcdff`). The PySCF/h5py-preload `ASI_init`
  repro passes (`REINIT_OK`), and the full test passed:
  `1 passed, 22 warnings in 472.63s (0:07:52)`, exit 0, with a valid
  parallel-NetCDF ERI file (20,873 quartets, 23 AOs, `n_spin=1`). This is the
  fix for the earlier flat-namespace HDF5 1.14 vs h5py HDF5 2.1.1 crash.
