# Temporary SIESTA ASI reinitialization fix

`siesta-asi-embasi-fixes.patch` contains the **only SIESTA source change needed for
WF-in-DFT**: the two FFT reset fixes that make repeated `ASI_init`/`ASI_run`
cycles safe.  Apply it from the SIESTA source root:

```sh
git -C /path/to/siesta apply /path/to/embasi-qiskit-integration/scripts/patches/siesta-asi-embasi-fixes.patch
```

## Contents

1. **`Src/m_fft_gpfa.F`** — `fft_gpfa_trigs_reset` deallocates `trigs` but leaves
   `nold`; a second `ASI_init` with the same FFT length skips `setgpfa_alloc` and
   `gpfa` dereferences the deallocated table. Set `nold = -1` with the reset.
2. **`Src/fft.F`** — the same defect in `m_fft`'s `fft_trigs_reset`
   (`frsttime`/`OldMesh` were not reset), crashing the next `ASI_run` in
   `dhscf_init`/`poison`. Set `frsttime = .TRUE.` and `OldMesh = 0`.

No modify-H / DFT-in-DFT patch is carried: the WF path uses `run_low_level`
(AB_LL SCF, A_LL noscf, `LayerEvaluator`) and never runs an embedded `A_HL` SCF.

## Validation

With the patch applied (Ubuntu 24.04, GCC 13, OpenMPI, ScaLAPACK):

- `scripts/check_siesta_asi_reinit.py --cycles 3` prints `REINIT_OK`.
- `SIESTA_ASI_REAL=1 pytest tests/test_siesta_asi_real.py` passes.
- `SIESTA_QCOMP_REAL=1 ASI_LIB_PATH=... SIESTA_PP_PATH=... pytest tests/test_siesta_qcomp_real.py`
  passes (about 5 minutes: the full WF path runs the embedding and QCOMP ERI
  generation several times).
- A real H2O/PBE/DZP WF-in-DFT run (A=O, neutral 6e in 7 A orbitals via
  `--a_nmos 3`, PBE low level, HF high level) completes through
  `projection_energy` with EmbASI's low-level A energy and the same-density
  reference; QCOMP's own `hfx_matrix` equals `2K` from the ERI provider exactly
  through its `d2s_indx` layout.
- `tests/test_siesta_qcomp_real.py` and `tests/test_siesta_molecule_matrix.py`
  call PySCF's FCI kernel on the SIESTA `h1`/`h2`/`e_core` and match
  `FCISolver` to 1e-9 Ha.  This is an exact same-pseudopotential comparison;
  no all-electron Gaussian low-level reference is used.

## QCOMP / WF-in-DFT build notes

The WF route additionally needs a QCOMP-enabled build (`SIESTA_WITH_QCOMP=ON`,
`SIESTA_WITH_HYBRIDXC=ON`, NetCDF).  Two build pitfalls caused real crashes:

- **libint must be 1.x with `MAX_AM_ERI=5`** (the bundled `External/libint`
  default).  Ubuntu's `libint-dev` 1.2.1 is built with `MAX_AM_ERI=7`; linking it
  makes SIESTA's `build_eri` table sizes disagree and H2O segfaults in
  `nao2gto`.  Build the submodule:
  `cmake -S External/libint -B libint_build -DMAX_AM_ERI=5 -DBUILD_FPIC=ON -DBUILD_SHARED_LIBS=OFF ...`
  and point `Libint_DIR`/`CMAKE_PREFIX_PATH` at its install.
- **QCOMP writes NetCDF with `NF90_MPIIO`**, so it needs a parallel NetCDF
  build.  A NetCDF library without MPI-IO aborts with
  `NetCDF: Parallel operation on file opened for non-parallel access`.  Build
  the MPI-enabled HDF5/NetCDF-C/NetCDF-Fortran stack with
  `scripts/build_parallel_netcdf.sh` (used by both the native build and the
  Docker image), or provide an existing parallel installation through
  `NC_CONFIG`/`NF_CONFIG`/`NETCDF_ROOT`.

h5py's bundled HDF5 can clash with an in-process flat-namespace NetCDF/HDF5
stack; the parallel stack is built with two-level namespaces on macOS, and the
ERI provider also falls back to a clean subprocess read if h5py cannot read the
file.

TODO(siesta-asi-embasi-fixes): delete this directory once the reinit fix and
build notes are upstreamed.
