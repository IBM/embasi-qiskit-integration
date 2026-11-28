# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Docker-based full integration harness for the real SIESTA-QCOMP WF test.

The pytest wrapper is :mod:`test_siesta_docker_integration`; this module holds
the mechanics so the wrapper stays a thin skip/assert layer:

* builder image tagged by the SHA-12 of ``docker/siesta-asi/Dockerfile`` plus
  ``scripts/build_parallel_netcdf.sh`` and built only when that tag is absent.
  The image contains a shared MPI HDF5 + NetCDF-C + NetCDF-Fortran stack under
  ``/opt/parallel-netcdf``, built by the reusable helper;
* named volume ``embasi-siesta-asi-cache`` holding the SIESTA clone, libint,
  build tree, Python venv, stamps and pytest work directory;
* a container-side bootstrap that clones ``qcomp-5.4-asi`` from GitLab (or
  fetches/resets an existing clone), updates only the required submodules,
  applies the maintained FFT patch, builds libint with ``MAX_AM_ERI=5`` and
  SIESTA with ASI/QCOMP against the image's parallel NetCDF
  (``SIESTA_WITH_NCDF_PARALLEL=ON``), creates the venv, and finally runs
  ``tests/test_siesta_qcomp_real.py``.

All expensive state lives in the named volume.  A second invocation with the
same branch commit, patch files and build configuration is a cache hit: the
source is fetched only to resolve the remote commit, then no clone/reset,
submodule update, build or pip install runs.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "docker" / "siesta-asi" / "Dockerfile"
PARALLEL_NETCDF_HELPER = REPO_ROOT / "scripts" / "build_parallel_netcdf.sh"
MAINTAINED_PATCH = REPO_ROOT / "scripts" / "patches" / "siesta-asi-embasi-fixes.patch"

IMAGE_REPO = "embasi-siesta-asi-builder"
VOLUME = "embasi-siesta-asi-cache"
#: Default: the private project over SSH.  The harness mounts ``$HOME``
#: read-only at ``/host-home`` so the container's git/ssh can use the host's
#: keys and ssh config.  ``SIESTA_DOCKER_REMOTE`` overrides this with another
#: SSH URL, a credentialed HTTPS URL, or a local mirror.
DEFAULT_SIESTA_REMOTE = "git@gitlab.com:qcomp4dft/dft/siesta.git"
SIESTA_REMOTE = os.environ.get("SIESTA_DOCKER_REMOTE", DEFAULT_SIESTA_REMOTE)
SIESTA_BRANCH = os.environ.get("SIESTA_DOCKER_BRANCH", "qcomp-5.4-asi")

#: The container test itself takes roughly five minutes; a cold run also builds
#: the image's parallel NetCDF stack, clones SIESTA, builds libint + SIESTA and
#: installs the venv (~25-45 minutes including the image build).
DEFAULT_TIMEOUT_S = 4 * 60 * 60

_CONTAINER_TEST = "/repo/tests/test_siesta_qcomp_real.py"
_CONTAINER_SCRIPT = r"""#!/bin/bash
# Bootstrap + run, executed inside the builder image with:
#   /repo          integration repository, read-only
#   /host-home     host $HOME, read-only (SSH keys/config; see GIT_SSH_COMMAND)
#   /cache         named volume with all persistent state
set -euo pipefail
trap 'printf "ERROR: bootstrap failed at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

step() { printf '\n[%s] == %s ==\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
sha256() { sha256sum "$1" | awk '{print $1}'; }

SIESTA_REMOTE="${SIESTA_REMOTE:-git@gitlab.com:qcomp4dft/dft/siesta.git}"
SIESTA_BRANCH="${SIESTA_BRANCH:-qcomp-5.4-asi}"

# Git/SSH: use the host's ~/.ssh (mounted read-only) but keep known_hosts in
# writable /tmp.  BatchMode makes a missing key/passphrase fail instead of
# hanging on a prompt, which keeps the harness non-interactive.
export HOME="${HOME:-/host-home}"
export GIT_SSH_COMMAND="${GIT_SSH_COMMAND:-ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/known_hosts}"
mkdir -p /tmp
if [ -f /host-home/.ssh/known_hosts ]; then
    cp /host-home/.ssh/known_hosts /tmp/known_hosts
fi

# $HOME is read-only.  Keep pip and other caches on the writable cache volume.
mkdir -p /cache/pip-cache /cache/xdg-cache 2>/dev/null || true
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/cache/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cache/xdg-cache}"

CACHE=/cache
SRC="$CACHE/siesta"
BUILD="$CACHE/build"
LIBINT_BUILD="$CACHE/libint-build"
LIBINT_PREFIX="$CACHE/libint-install"
VENV="$CACHE/venv"
STAMPS="$CACHE/stamps"
WORK="$CACHE/work"
PYTEST_TMP="$CACHE/pytest-tmp"
BUILD_STAMP="$STAMPS/build.stamp"
LIBINT_STAMP="$STAMPS/libint.stamp"
VENV_STAMP="$STAMPS/venv.stamp"

MAIN_PATCH=/repo/scripts/patches/siesta-asi-embasi-fixes.patch
CONTAINER_TEST=/repo/tests/test_siesta_qcomp_real.py

# The image carries a shared MPI HDF5/NetCDF stack built by
# scripts/build_parallel_netcdf.sh under /opt/parallel-netcdf.
PARALLEL_NETCDF=/opt/parallel-netcdf
NC_CONFIG="$PARALLEL_NETCDF/bin/nc-config"
NF_CONFIG="$PARALLEL_NETCDF/bin/nf-config"

mkdir -p "$STAMPS" "$WORK" "$PYTEST_TMP"

[ -f "$MAIN_PATCH" ] || die "missing maintained patch: $MAIN_PATCH"
[ -x "$NC_CONFIG" ] || die "image is missing parallel NetCDF: $NC_CONFIG"
[ -x "$NF_CONFIG" ] || die "image is missing parallel NetCDF-Fortran: $NF_CONFIG"
[ -f "$PARALLEL_NETCDF/.embasi-parallel-netcdf-stamp" ] \
    || die "image is missing the parallel NetCDF summary stamp"
"$NC_CONFIG" --has-parallel4 2>/dev/null | grep -qi '^yes' \
    || die "image NetCDF-C does not report parallel I/O"

# --------------------------------------------------------------------------
# 1. Source: clone once, otherwise fetch; only reset when a rebuild is due.
# --------------------------------------------------------------------------
if [ ! -d "$SRC/.git" ]; then
    step "Cloning $SIESTA_REMOTE ($SIESTA_BRANCH)"
    rm -rf "$SRC"
    git clone --branch "$SIESTA_BRANCH" --single-branch "$SIESTA_REMOTE" "$SRC"
else
    # Keep an existing clone's origin in sync with the configured remote
    # (e.g. when switching between a credentialled URL and a local mirror).
    git -C "$SRC" remote set-url origin "$SIESTA_REMOTE"
    step "Fetching origin/$SIESTA_BRANCH"
    git -C "$SRC" fetch --prune origin \
        "+refs/heads/$SIESTA_BRANCH:refs/remotes/origin/$SIESTA_BRANCH"
fi

COMMIT=$(git -C "$SRC" rev-parse "refs/remotes/origin/$SIESTA_BRANCH")
LIBINT_COMMIT=$(git -C "$SRC" rev-parse "$COMMIT:External/libint")
ASI4PY_COMMIT=$(git -C "$SRC" rev-parse "$COMMIT:asi-tests/asi")
MAIN_SHA=$(sha256 "$MAIN_PATCH")
NCDF_PARALLEL=ON
PNC_STACK_SHA=$(sha256 "$PARALLEL_NETCDF/.embasi-parallel-netcdf-stamp")
NETCDF_CONFIG="netcdf=parallel;netcdf_parallel=ON;parallel_netcdf_prefix=$PARALLEL_NETCDF;parallel_stack_stamp_sha256=$PNC_STACK_SHA"

BUILD_CONFIG="branch=$SIESTA_BRANCH;commit=$COMMIT;libint=$LIBINT_COMMIT;build_type=Release;shared=ON;mpi=ON;$NETCDF_CONFIG;fftw=ON;hybridxc=ON;qcomp=ON;asi=ON;libint_max_am_eri=5;tests=OFF"
CONFIG_SHA=$(printf '%s' "$BUILD_CONFIG" | sha256sum | awk '{print $1}')

EXPECTED_STAMP="branch=$SIESTA_BRANCH
commit=$COMMIT
maintained_patch_sha256=$MAIN_SHA
libint_commit=$LIBINT_COMMIT
build_config_sha256=$CONFIG_SHA
build_config=$BUILD_CONFIG"

printf 'Resolved %s @ %s\n' "$SIESTA_BRANCH" "$COMMIT"

if [ -f "$BUILD_STAMP" ] && [ -f "$BUILD/Src/libsiesta.so" ] \
   && [ "$(cat "$BUILD_STAMP")" = "$EXPECTED_STAMP" ]; then
    step "Cache HIT: source, patches, libint and libsiesta.so all match"
else
    step "Cache MISS: resetting source and rebuilding"
    rm -f "$BUILD_STAMP"
    git -C "$SRC" reset --hard "$COMMIT"
    git -C "$SRC" submodule sync --recursive
    git -C "$SRC" submodule update --init --recursive -- \
        External/libfdf External/xmlf90 External/libpsml \
        External/libgridxc External/libint asi-tests/asi

    apply_patch() {
        local patch="$1"
        if git -C "$SRC" apply --reverse --check "$patch" >/dev/null 2>&1; then
            printf 'Already applied: %s\n' "$patch"
        elif git -C "$SRC" apply --check "$patch"; then
            git -C "$SRC" apply "$patch"
            printf 'Applied: %s\n' "$patch"
        else
            die "patch does not apply cleanly: $patch"
        fi
    }
    apply_patch "$MAIN_PATCH"

    if [ -f "$LIBINT_PREFIX/lib/libint.a" ] \
       && [ -f "$LIBINT_PREFIX/include/libint/libint.h" ] \
       && [ -f "$LIBINT_STAMP" ] \
       && [ "$(cat "$LIBINT_STAMP")" = "$LIBINT_COMMIT" ]; then
        step "libint cache HIT ($LIBINT_COMMIT, MAX_AM_ERI=5)"
    else
        step "Building libint $LIBINT_COMMIT with MAX_AM_ERI=5"
        rm -f "$LIBINT_STAMP"
        rm -rf "$LIBINT_BUILD" "$LIBINT_PREFIX"
        cmake -S "$SRC/External/libint" -B "$LIBINT_BUILD" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$LIBINT_PREFIX" \
            -DMAX_AM_ERI=5 \
            -DBUILD_FPIC=ON \
            -DBUILD_SHARED_LIBS=OFF
        cmake --build "$LIBINT_BUILD" --target install --parallel "$(nproc)"
        [ -f "$LIBINT_PREFIX/lib/libint.a" ] || die "libint install missing libint.a"
        [ -f "$LIBINT_PREFIX/include/libint/libint.h" ] || die "libint install missing libint.h"
        printf '%s\n' "$LIBINT_COMMIT" > "$LIBINT_STAMP"
    fi

    step "Configuring SIESTA (ASI + QCOMP + parallel NetCDF)"
    MULTIARCH=$(gcc -print-multiarch)
    export PKG_CONFIG_PATH="$LIBINT_PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export LD_LIBRARY_PATH="$PARALLEL_NETCDF/lib:$PARALLEL_NETCDF/hdf5/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PKG_CONFIG_PATH="$PARALLEL_NETCDF/lib/pkgconfig:$PKG_CONFIG_PATH"
    NETCDF_CMAKE_ARGS=(
        -DNetCDF_C_CONFIG_EXECUTABLE="$NC_CONFIG"
        -DNetCDF_Fortran_CONFIG_EXECUTABLE="$NF_CONFIG"
        -DNetCDF_C_INCLUDE_FILE="$PARALLEL_NETCDF/include/netcdf.h"
        -DNetCDF_Fortran_INCLUDE_FILE="$PARALLEL_NETCDF/include/netcdf.mod"
        -DNetCDF_C_LIBRARY="$PARALLEL_NETCDF/lib/libnetcdf.so"
        -DNetCDF_Fortran_LIBRARY="$PARALLEL_NETCDF/lib/libnetcdff.so"
        -DSIESTA_WITH_NCDF_PARALLEL=ON
    )
    rm -rf "$BUILD"
    cmake -S "$SRC" -B "$BUILD" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$CACHE/siesta-install" \
        -DCMAKE_C_COMPILER=mpicc \
        -DCMAKE_CXX_COMPILER=mpicxx \
        -DCMAKE_Fortran_COMPILER=mpifort \
        -DCMAKE_PREFIX_PATH="$LIBINT_PREFIX" \
        -DLibint_DIR="$LIBINT_PREFIX/share/cmake/Libint" \
        -DBLAS_LIBRARY="/usr/lib/$MULTIARCH/libopenblas.so" \
        -DLAPACK_LIBRARY="NONE" \
        -DSCALAPACK_LIBRARY="/usr/lib/$MULTIARCH/libscalapack-openmpi.so" \
        "${NETCDF_CMAKE_ARGS[@]}" \
        -DBUILD_SHARED_LIBS=ON \
        -DSIESTA_WITH_MPI=ON \
        -DSIESTA_WITH_MPI_INTERFACES=f08 \
        -DSIESTA_WITH_NETCDF=ON \
        -DSIESTA_WITH_FFTW=ON \
        -DSIESTA_WITH_HYBRIDXC=ON \
        -DSIESTA_WITH_QCOMP=ON \
        -DSIESTA_WITH_ASI=ON \
        -DASI_INCLUDE_DIR="$SRC/asi-tests/asi/include" \
        -DSIESTA_WITH_ELSI=OFF \
        -DSIESTA_WITH_ELPA=OFF \
        -DSIESTA_WITH_FLOOK=OFF \
        -DSIESTA_WITH_DFTD3=OFF \
        -DSIESTA_WITH_CHESS=OFF \
        -DSIESTA_WITH_WANNIER90=OFF \
        -DSIESTA_WITH_JSON=OFF \
        -DSIESTA_WITH_ZEROMQ=OFF \
        -DSIESTA_WITH_LIBXC=OFF \
        -DSIESTA_WITH_OPENMP=OFF \
        -DSIESTA_TESTS=OFF

    step "Building SIESTA and libsiesta.so"
    cmake --build "$BUILD" --parallel "$(nproc)"
    [ -f "$BUILD/Src/libsiesta.so" ] || die "build finished without Src/libsiesta.so"
    printf '%s\n' "$EXPECTED_STAMP" > "$BUILD_STAMP.tmp"
    mv "$BUILD_STAMP.tmp" "$BUILD_STAMP"
    step "SIESTA build complete"
fi

# --------------------------------------------------------------------------
# 2. Python venv (stamped; rebuilt only when the stamp or imports break)
# --------------------------------------------------------------------------
if [ ! -f "$SRC/asi-tests/asi/pyasi/pyproject.toml" ]; then
    step "Initializing asi-tests/asi for the Python wrapper"
    git -C "$SRC" submodule update --init --recursive -- asi-tests/asi
fi

PY_VERSION=$(python3 -V 2>&1)
VENV_STAMP_TEXT="python=$PY_VERSION
packages=numpy scipy ase mpi4py scalapack4py pyscf h5py pydantic pydantic-settings pytest
asi4py_commit=$ASI4PY_COMMIT
embasi_ref=qm-code-adapter"

venv_valid=0
if [ -x "$VENV/bin/pytest" ] && [ -f "$VENV_STAMP" ] \
   && [ "$(cat "$VENV_STAMP")" = "$VENV_STAMP_TEXT" ]; then
    if "$VENV/bin/python" -c \
        'import asi4py, ase, embasi, h5py, mpi4py, numpy, pydantic, pydantic_settings, pyscf, scipy, scalapack4py' \
        >/dev/null 2>&1; then
        venv_valid=1
    fi
fi

if [ "$venv_valid" = 1 ]; then
    step "venv cache HIT ($VENV)"
else
    step "Creating venv and installing test dependencies"
    rm -f "$VENV_STAMP"
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
    MPICC=mpicc "$VENV/bin/python" -m pip install \
        numpy scipy ase mpi4py scalapack4py pyscf h5py \
        pydantic pydantic-settings pytest
    "$VENV/bin/python" -m pip install --no-deps "$SRC/asi-tests/asi/pyasi"
    "$VENV/bin/python" -m pip install --no-deps \
        "embasi @ git+https://github.com/tamm-cci/EmbASI.git@qm-code-adapter"
    printf '%s\n' "$VENV_STAMP_TEXT" > "$VENV_STAMP.tmp"
    mv "$VENV_STAMP.tmp" "$VENV_STAMP"
fi

# --------------------------------------------------------------------------
# 3. Real WF-in-DFT test
# --------------------------------------------------------------------------
step "Running $CONTAINER_TEST"
rm -rf "$PYTEST_TMP"
mkdir -p "$PYTEST_TMP"
cd "$WORK"

export SIESTA_QCOMP_REAL=1
export ASI_LIB_PATH="$BUILD/Src/libsiesta.so"
export SIESTA_PP_PATH="$SRC/Tests/Pseudos"
export PYTHONPATH="/repo/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BUILD/Src:$PARALLEL_NETCDF/lib:$PARALLEL_NETCDF/hdf5/lib"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PYTEST_TMP"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

printf 'ASI_LIB_PATH=%s\nSIESTA_PP_PATH=%s\nnetcdf=parallel\nLD_LIBRARY_PATH=%s\nworkdir=%s\nTMPDIR=%s\n' \
    "$ASI_LIB_PATH" "$SIESTA_PP_PATH" "$LD_LIBRARY_PATH" "$WORK" "$TMPDIR"

"$VENV/bin/pytest" -q -p no:cacheprovider "$CONTAINER_TEST"
step "Integration test passed"
"""


@dataclass(frozen=True)
class IntegrationResult:
    """Outcome of one container run."""

    returncode: int
    log_path: Path
    image: str
    elapsed_s: float

    def tail(self, lines: int = 200) -> str:
        """Return the last ``lines`` lines of the run log."""
        return tail_file(self.log_path, lines)


def tail_file(path: str | Path, lines: int = 200) -> str:
    """Return the last ``lines`` lines of a log file (never loads it all)."""
    try:
        with Path(path).open("r", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines))
    except OSError as exc:  # pragma: no cover - only on an unreadable log
        return f"<cannot read {path}: {exc}>"


def docker_status() -> tuple[bool, str]:
    """Return ``(usable, detail)`` for the Docker CLI and daemon.

    ``docker info`` is the daemon probe: it catches a missing CLI (no
    executable), a stopped daemon, permission errors and a broken context.
    """
    executable = shutil.which("docker")
    if executable is None:
        return False, "docker CLI not found on PATH"
    try:
        proc = subprocess.run(
            [executable, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker info failed: {exc}"
    if proc.returncode != 0:
        output = (proc.stderr or proc.stdout).strip().splitlines()
        detail = output[-1] if output else f"exit status {proc.returncode}"
        return False, f"docker info failed: {detail}"
    return True, proc.stdout.strip()


def skip_reason() -> str | None:
    """Why the Docker integration test should skip, or ``None`` to run it."""
    if os.environ.get("SIESTA_DOCKER_INTEGRATION") != "1":
        return (
            "opt-in full Docker integration test; set "
            "SIESTA_DOCKER_INTEGRATION=1 to run it (20-40 min on a cold cache)"
        )
    if os.environ.get("CI"):
        return "Docker integration test is disabled on CI (CI is set)"
    if SIESTA_REMOTE.startswith(("git@", "ssh://")) and not (Path.home() / ".ssh").is_dir():
        return (
            "the default SIESTA clone uses SSH but ~/.ssh is absent; "
            "set SIESTA_DOCKER_REMOTE to a credentialed HTTPS URL or a mirror"
        )
    usable, detail = docker_status()
    if not usable:
        return detail
    return None


def dockerfile_sha12() -> str:
    """SHA-256 of the Dockerfile plus the parallel-NetCDF helper, first 12 hex."""
    digest = hashlib.sha256()
    for path in (DOCKERFILE, PARALLEL_NETCDF_HELPER):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def image_tag() -> str:
    """Builder image reference tagged by the Dockerfile/helper content hash."""
    return f"{IMAGE_REPO}:{dockerfile_sha12()}"


def _redact_cmd(cmd: list[str]) -> list[str]:
    """Hide userinfo from a credentialed SIESTA remote before logging."""
    redacted: list[str] = []
    for part in cmd:
        if part.startswith("SIESTA_REMOTE="):
            value = part.split("=", 1)[1]
            if "://" in value:
                scheme, rest = value.split("://", 1)
                authority, sep, tail = rest.partition("/")
                if "@" in authority:
                    authority = "<redacted>@" + authority.rsplit("@", 1)[1]
                    value = f"{scheme}://{authority}{sep}{tail}"
            part = f"SIESTA_REMOTE={value}"
        redacted.append(part)
    return redacted


def _run_logged(cmd: list[str], log, *, input_text: str | None = None, timeout=None):
    log.write("$ " + " ".join(_redact_cmd(cmd)) + "\n")
    log.flush()
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc


def ensure_image(log) -> str:
    """Return the builder image tag, building it only when absent."""
    tag = image_tag()
    inspect = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        log.write(f"Builder image cache HIT: {tag}\n")
        return tag

    log.write(f"Builder image cache MISS: building {tag}\n")
    log.flush()
    # Minimal build context: the Dockerfile plus the reusable parallel-NetCDF
    # helper it COPYs.  Keeping the context small avoids sending the repository
    # (including .git/.venv) to the daemon.
    with tempfile.TemporaryDirectory(prefix="embasi-siesta-docker-context-") as context:
        context_path = Path(context)
        shutil.copy2(DOCKERFILE, context_path / "Dockerfile")
        shutil.copy2(PARALLEL_NETCDF_HELPER, context_path / "build_parallel_netcdf.sh")
        proc = _run_logged(
            [
                "docker",
                "build",
                "--progress=plain",
                "-f",
                str(context_path / "Dockerfile"),
                "-t",
                tag,
                str(context_path),
            ],
            log,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker build failed with exit status {proc.returncode}; "
            f"see {getattr(log, 'name', '<log>')}\n"
            f"{tail_file(getattr(log, 'name', ''))}"
        )
    return tag


def ensure_volume(log) -> None:
    """Create the named cache volume if it does not exist."""
    inspect = subprocess.run(
        ["docker", "volume", "inspect", VOLUME],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        log.write(f"Cache volume present: {VOLUME}\n")
        return
    log.write(f"Creating cache volume: {VOLUME}\n")
    log.flush()
    proc = _run_logged(["docker", "volume", "create", VOLUME], log)
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker volume create {VOLUME} failed with exit status {proc.returncode}"
        )


def _run_command(image: str) -> list[str]:
    """``docker run`` invocation for the bootstrap container."""
    # The private GitLab project is cloned over SSH.  Mount $HOME read-only so
    # the container's git and ssh see the host's keys/config without copying
    # them; the script keeps known_hosts on the writable /tmp.
    home = Path.home()
    # ssh expands ``~/.ssh`` from the passwd entry (/root), not $HOME, so point
    # -i at the mounted host keys explicitly.  IdentitiesOnly keeps it from
    # trying the container's empty /root/.ssh.
    key_names = (
        "id_rsa",
        "id_ecdsa",
        "id_ed25519",
        "id_ecdsa_sk",
        "id_ed25519_sk",
        "id_dsa",
    )
    key_args = "".join(
        f" -i /host-home/.ssh/{name}" for name in key_names if (home / ".ssh" / name).is_file()
    )
    ssh_command = os.environ.get(
        "SIESTA_DOCKER_SSH_COMMAND",
        "ssh -o BatchMode=yes -o IdentitiesOnly=yes "
        "-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/known_hosts" + key_args,
    )
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--name",
        f"embasi-siesta-asi-run-{os.getpid()}",
        "--mount",
        f"type=bind,src={REPO_ROOT},dst=/repo,readonly",
        "--mount",
        f"type=bind,src={home},dst=/host-home,readonly",
        "--mount",
        f"type=volume,src={VOLUME},dst=/cache",
        "--workdir",
        "/cache/work",
        "-e",
        "HOME=/host-home",
        "-e",
        f"GIT_SSH_COMMAND={ssh_command}",
        "-e",
        "GIT_TERMINAL_PROMPT=0",
        "-e",
        f"SIESTA_REMOTE={SIESTA_REMOTE}",
        "-e",
        f"SIESTA_BRANCH={SIESTA_BRANCH}",
        "-e",
        "SIESTA_QCOMP_REAL=1",
        "-e",
        "ASI_LIB_PATH=/cache/build/Src/libsiesta.so",
        "-e",
        "SIESTA_PP_PATH=/cache/siesta/Tests/Pseudos",
        "-e",
        "PYTHONPATH=/repo/src",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-e",
        "PIP_CACHE_DIR=/cache/pip-cache",
        "-e",
        "XDG_CACHE_HOME=/cache/xdg-cache",
        "-e",
        "TMPDIR=/cache/pytest-tmp",
        image,
        "bash",
        "-s",
    ]


def run_integration(
    *, timeout: float | None = None, log_path: str | Path | None = None
) -> IntegrationResult:
    """Build/cache everything and run the container test.

    Writes the complete Docker/build/pytest output to ``log_path`` (by default
    a fresh temporary file, or ``$SIESTA_DOCKER_LOG`` when set) and returns the
    exit status plus the log path.  Raises :class:`RuntimeError` only when the
    harness itself cannot start (Docker build/volume failure, timeout).
    """
    usable, detail = docker_status()
    if not usable:
        raise RuntimeError(detail)

    if timeout is None:
        timeout = float(os.environ.get("SIESTA_DOCKER_TIMEOUT", DEFAULT_TIMEOUT_S))

    if log_path is None:
        env_log = os.environ.get("SIESTA_DOCKER_LOG")
        if env_log:
            log_path = Path(env_log).expanduser()
        else:
            fd, name = tempfile.mkstemp(prefix="embasi-siesta-docker-", suffix=".log")
            os.close(fd)
            log_path = Path(name)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with log_path.open("w") as log:
        image = ensure_image(log)
        ensure_volume(log)
        cmd = _run_command(image)
        try:
            proc = _run_logged(cmd, log, input_text=_CONTAINER_SCRIPT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"container run exceeded {timeout:.0f}s; see {log_path}\n{tail_file(log_path)}"
            ) from exc
        elapsed = time.monotonic() - started
        log.write(f"\nContainer exited with status {proc.returncode} after {elapsed:.1f}s\n")
    return IntegrationResult(
        returncode=proc.returncode,
        log_path=log_path,
        image=image,
        elapsed_s=elapsed,
    )
