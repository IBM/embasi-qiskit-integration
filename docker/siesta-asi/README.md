# Full SIESTA-QCOMP Docker integration test

This directory holds the builder image and instructions for the opt-in,
end-to-end WF-in-DFT test:

- SIESTA is cloned from `git@gitlab.com:qcomp4dft/dft/siesta.git`, branch
  `qcomp-5.4-asi`, over SSH. The harness mounts `$HOME` read-only at
  `/host-home`, so the container uses the host's SSH keys and `~/.ssh/config`;
  no credentials are copied into the image.
- The maintained FFT-reinitialization fix
  (`scripts/patches/siesta-asi-embasi-fixes.patch`) is applied.
- The image carries a shared, MPI-enabled HDF5 1.14.6 + NetCDF-C 4.9.3 +
  NetCDF-Fortran 4.6.2 stack under `/opt/parallel-netcdf`, built by the
  reusable `scripts/build_parallel_netcdf.sh`. SIESTA is configured with
  `SIESTA_WITH_NCDF_PARALLEL=ON`, so QCOMP uses its native `NF90_MPIIO` path.
  The ASI interface itself is still run with one MPI rank.
- libint comes from SIESTA's `External/libint` submodule and is built with
  `MAX_AM_ERI=5` (Ubuntu's `libint-dev` uses 7 and segfaults H2O).
- SIESTA is built with ASI, QCOMP, hybrid-XC, MPI, ScaLAPACK, parallel NetCDF,
  `BUILD_SHARED_LIBS=ON` and no optional heavy submodules.

## Run

From the repository root:

```sh
SIESTA_DOCKER_INTEGRATION=1 pytest -q tests/test_siesta_docker_integration.py
```

Cold-run progress (image/parallel NetCDF build, clone, libint, SIESTA, venv,
test) is written to the run log described below; `tail -f` that file to follow
a long build. The first run takes roughly 25-45 minutes; later runs with the
same image, branch commit, patch hashes and build configuration finish in
about the test's own runtime (~5 minutes).

The harness stages the Dockerfile and `scripts/build_parallel_netcdf.sh` in a
small temporary build context because the Dockerfile needs that helper for its
`/opt/parallel-netcdf` layer. A manual `docker build` must copy the helper next
to the Dockerfile (`COPY build_parallel_netcdf.sh ...` is relative to the build
context).

The test skips by default, when `CI` is set, when the `docker` CLI is absent,
when `docker info` fails, or when the default SSH clone has no `~/.ssh`
directory.  A missing or passphrase-protected key fails the clone with the full
SSH error in the run log; use an SSH agent or set `SIESTA_DOCKER_REMOTE` to a
credentialed HTTPS URL or a mirror visible inside the container.

The small-molecule validation matrix runs the same way.  It covers H2O, H2,
CH4 and NH3, asserting the projection-embedding invariants and comparing the
adapter's embedded Hamiltonian with a same-pseudopotential PySCF FCI solve
for each:

```sh
docker run --rm \
  --mount type=bind,src="$PWD",dst=/repo,readonly \
  --mount type=volume,src=embasi-siesta-asi-cache,dst=/cache \
  -e SIESTA_QCOMP_REAL=1 \
  -e ASI_LIB_PATH=/cache/build/Src/libsiesta.so \
  -e SIESTA_PP_PATH=/cache/siesta/Tests/Pseudos \
  -e PYTHONPATH=/repo/src \
  -e LD_LIBRARY_PATH=/cache/build/Src:/opt/parallel-netcdf/lib:/opt/parallel-netcdf/hdf5/lib \
  -e TMPDIR=/cache/pytest-tmp \
  -w /cache/work embasi-siesta-asi-builder:<tag> \
  /cache/venv/bin/pytest -q -s /repo/tests/test_siesta_molecule_matrix.py
```

Like the other real tests it skips on CI and unless `SIESTA_QCOMP_REAL=1`; the
module docstring lists each molecule's A/B mask and `a_nmos` choice.

The SIESTA URL defaults to the private project over SSH
(`git@gitlab.com:qcomp4dft/dft/siesta.git`).  The host's `$HOME` is mounted
read-only, and `known_hosts` is copied into the container's writable `/tmp`
(accept-new otherwise) so the clone is non-interactive.  If SSH is not
available, set `SIESTA_DOCKER_REMOTE` to a credentialed HTTPS URL or to a
path/mirror visible inside the container (for example a bare mirror copied into
the cache volume).  Submodules are fetched from their upstream URLs.
`SIESTA_DOCKER_BRANCH` overrides the branch (default `qcomp-5.4-asi`);
`SIESTA_DOCKER_SSH_COMMAND` overrides the SSH command (default BatchMode with
`/tmp/known_hosts`).

## Caching

- Builder image: tagged by
  `sha256(Dockerfile + scripts/build_parallel_netcdf.sh)[:12]`.
  `docker image inspect` decides whether a build is needed. The image build
  runs the helper with `--prefix /opt/parallel-netcdf --cleanup`, so the source
  downloads/build trees are not kept in the layer; the installed stack and its
  stamps are.
- Named volume: `embasi-siesta-asi-cache`. It holds:
  - `/cache/siesta` — the clone (only reset when the build stamp changes);
  - `/cache/libint-build`, `/cache/libint-install` — libint 1.x,
    `MAX_AM_ERI=5`;
  - `/cache/build` — the SIESTA build tree and `Src/libsiesta.so`, built with
    `SIESTA_WITH_NCDF_PARALLEL=ON` against `/opt/parallel-netcdf`;
  - `/cache/venv` — Python venv with numpy, scipy, ase, mpi4py,
    scalapack4py, pyscf, h5py, pydantic, pydantic-settings, pytest,
    asi4py (from the clone's `asi-tests/asi/pyasi`) and EmbASI
    (`qm-code-adapter`, `--no-deps`);
  - `/cache/stamps` — branch, commit, patch hashes, libint commit and build
    configuration; the venv has its own stamp.
  - `/cache/work` and `/cache/pytest-tmp` — container work/temp directories.

A cache hit still performs `git fetch` to resolve `origin/qcomp-5.4-asi`; if
the remote commit, maintained patch, parallel-stack summary stamp, libint
commit and build configuration match the stamp and `Src/libsiesta.so` exists,
no reset, submodule update, build or pip install runs.

To force a full rebuild, remove the volume and the builder image (the harness
prints the current tag on a cache hit; `docker image ls` also lists it):

```sh
docker volume rm embasi-siesta-asi-cache
docker image ls embasi-siesta-asi-builder
docker rmi embasi-siesta-asi-builder:<tag>
```

Changing the Dockerfile or the helper changes the image tag; the volume's
build stamp also includes the parallel-stack summary, so a new image triggers
a SIESTA rebuild. Remove the volume when the base image/toolchain changes so
cached libint objects are not reused across compiler/library changes.

## Validation status

The normal parallel Docker path was validated end-to-end with image
`embasi-siesta-asi-builder:04d0c9141f58`: `1 passed, 26 warnings` in 297.11 s
inside the container (container exit 0; outer pytest 476.77 s), with
`netcdf=parallel`, `NetCDF_PARALLEL=TRUE`, and only the maintained FFT patch
applied. Removing the system NetCDF packages afterwards changes the Dockerfile
hash; the current tag is `embasi-siesta-asi-builder:0b38b80bbf7f`. The normal
parallel stack/path is unchanged, so a rebuild is expected to reproduce this
result, but the final tag was not re-run in this change.

## Logs and environment

The complete Docker/build/pytest output goes to a temporary
`embasi-siesta-docker-*.log`; set `SIESTA_DOCKER_LOG=/path/to/run.log` to
choose the path (the test prints it on failure). Override the run timeout
(default 4 hours) with `SIESTA_DOCKER_TIMEOUT` seconds.

The integration repository is mounted read-only at `/repo`; nothing outside
the named cache volume is written inside the container.
