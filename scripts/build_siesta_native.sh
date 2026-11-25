#!/usr/bin/env bash
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#
# Build the SIESTA QCOMP/ASI branch natively on Linux or macOS.
#
# This is the non-Docker counterpart of docker/siesta-asi/Dockerfile plus the
# build part of tests/_siesta_docker.py.  It produces an ASI-enabled
# libsiesta.so / libsiesta.dylib for EmbASI's in-process ASI driver; it never
# installs SIESTA as a Python package and never runs pip.
#
# See scripts/BUILD_SIESTA_NATIVE.md for prerequisites and platform notes.

set -euo pipefail

trap 'printf "ERROR: %s failed at line %s: %s\n" "${0##*/}" "$LINENO" "$BASH_COMMAND" >&2' ERR

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

MAINTAINED_PATCH="$REPO_ROOT/scripts/patches/siesta-asi-embasi-fixes.patch"
PARALLEL_NETCDF_HELPER="$SCRIPT_DIR/build_parallel_netcdf.sh"

DEFAULT_REMOTE="git@gitlab.com:qcomp4dft/dft/siesta.git"
DEFAULT_BRANCH="qcomp-5.4-asi"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: scripts/build_siesta_native.sh [options]

Build SIESTA with ASI + QCOMP + hybrid-XC on Linux or macOS (no Docker).
Parallel NetCDF is required; use --parallel-netcdf to build a cached stack or
provide an existing parallel NC_CONFIG/NF_CONFIG.

Source selection (choose at most one of --siesta-src / --clone-dir):
  --siesta-src DIR      Use an existing SIESTA checkout as-is.  The script
                        applies the maintained FFT patch; it never resets the
                        HEAD.
  --clone-dir DIR       Managed clone location.  Cloned on first use.  With
                        --update the clone is fetched and hard-reset to the
                        remote branch (default: <stage-root>/siesta-src).
  --remote URL          Git remote for a managed clone (default:
                        $SIESTA_REMOTE, else git@gitlab.com:qcomp4dft/dft/siesta.git).
  --branch NAME         Branch for a managed clone (default: $SIESTA_BRANCH,
                        else qcomp-5.4-asi).
  --update              Fetch and hard-reset a managed clone to origin/BRANCH.
  --no-fetch            Never touch the network for source/submodules.  Requires
                        an existing checkout with initialized submodules.

Build layout:
  --stage-root DIR      Root for default source/build/install/libint paths.
                        Default: $SIESTA_NATIVE_ROOT, else <repo>/build/siesta-native.
  --build-dir DIR       SIESTA CMake build tree (default: <stage-root>/build).
  --install-prefix DIR  CMake install prefix (default: <stage-root>/install).
  --jobs N              Parallel build jobs (default: $JOBS, else CPU count).
  --clean               Remove the SIESTA build tree, libint build tree and
                        libint install prefix before configuring.  The source
                        checkout and --install-prefix are left alone.
  --force               Re-run configure and rebuild even if the stamps match.
  --parallel-netcdf     Build a cached MPI-enabled HDF5 + NetCDF-C +
                        NetCDF-Fortran stack under <stage-root>/parallel-netcdf
                        and build SIESTA against it with
                        SIESTA_WITH_NCDF_PARALLEL=ON.  Needs network on the
                        first run; later runs use the cache.  Required unless
                        NC_CONFIG/NF_CONFIG already name a parallel NetCDF-C /
                        NetCDF-Fortran installation.
  --build-parallel-netcdf
                        Alias for --parallel-netcdf.

Environment overrides:
  CC, CXX, FC           Compiler commands (otherwise MPI wrappers are used).
  MPICC, MPICXX, MPIFC  Explicit MPI wrapper commands (take priority).
  MPIF90                Accepted as an alias for MPIFC.
  MPI_HOME, MPIRUN      MPI installation / launcher used for discovery/diagnostics.
  MPI_HOME              Prepend $MPI_HOME/bin to PATH and CMAKE_PREFIX_PATH.
  CMAKE, CMAKE_GENERATOR
                        CMake executable / generator (default: Ninja if found).
  SIESTA_NATIVE_ROOT    Default --stage-root.
  SIESTA_REMOTE, SIESTA_BRANCH
                        Defaults for --remote / --branch.
  SIESTA_MPI_INTERFACES f08 (default), legacy or none.
  NETCDF_ROOT, NC_CONFIG, NF_CONFIG
                        NetCDF C / NetCDF-Fortran installation hints.
  NETCDF_FORTRAN_ROOT   NetCDF-Fortran prefix (when different from C).
  SIESTA_PARALLEL_NETCDF_ROOT
                        Cache root for --parallel-netcdf (default:
                        <stage-root>/parallel-netcdf).
  HDF5_VERSION, NETCDF_C_VERSION, NETCDF_FORTRAN_VERSION
                        Self-built stack versions (defaults: 1.14.6, 4.9.3,
                        4.6.2).
  HDF5_SOURCE_URL, NETCDF_C_SOURCE_URL, NETCDF_FORTRAN_SOURCE_URL
  HDF5_SHA256, NETCDF_C_SHA256, NETCDF_FORTRAN_SHA256
                        Override source URLs / checksums.  When changing a
                        version also set the matching checksum.
  SCALAPACK_LIBRARY, BLAS_LIBRARY, LAPACK_LIBRARY
                        Explicit library paths (LAPACK defaults to NONE when
                        OpenBLAS is used, as in the Docker build).
  CMAKE_PREFIX_PATH     Extra CMake prefixes.

The script prints the environment needed by
  tests/test_siesta_qcomp_real.py
and writes it to <stage-root>/env.sh.  It does not install Python packages.
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
SIESTA_SRC_OPT=""
CLONE_DIR=""
REMOTE=${SIESTA_REMOTE:-$DEFAULT_REMOTE}
BRANCH=${SIESTA_BRANCH:-$DEFAULT_BRANCH}
STAGE_ROOT=${SIESTA_NATIVE_ROOT:-"$REPO_ROOT/build/siesta-native"}
BUILD_DIR=""
INSTALL_PREFIX=""
JOBS=${JOBS:-}
GENERATOR=${CMAKE_GENERATOR:-}
CLEAN=0
FORCE=0
UPDATE=0
NO_FETCH=0
BUILD_PARALLEL_NETCDF=0

while [ $# -gt 0 ]; do
    case "$1" in
        --siesta-src)      [ $# -ge 2 ] || die "--siesta-src needs a directory"; SIESTA_SRC_OPT=$2; shift 2 ;;
        --siesta-src=*)    SIESTA_SRC_OPT=${1#*=}; shift ;;
        --clone-dir)       [ $# -ge 2 ] || die "--clone-dir needs a directory"; CLONE_DIR=$2; shift 2 ;;
        --clone-dir=*)     CLONE_DIR=${1#*=}; shift ;;
        --remote)          [ $# -ge 2 ] || die "--remote needs a URL"; REMOTE=$2; shift 2 ;;
        --remote=*)        REMOTE=${1#*=}; shift ;;
        --branch)          [ $# -ge 2 ] || die "--branch needs a name"; BRANCH=$2; shift 2 ;;
        --branch=*)        BRANCH=${1#*=}; shift ;;
        --stage-root)      [ $# -ge 2 ] || die "--stage-root needs a directory"; STAGE_ROOT=$2; shift 2 ;;
        --stage-root=*)    STAGE_ROOT=${1#*=}; shift ;;
        --build-dir)       [ $# -ge 2 ] || die "--build-dir needs a directory"; BUILD_DIR=$2; shift 2 ;;
        --build-dir=*)     BUILD_DIR=${1#*=}; shift ;;
        --install-prefix)  [ $# -ge 2 ] || die "--install-prefix needs a directory"; INSTALL_PREFIX=$2; shift 2 ;;
        --install-prefix=*) INSTALL_PREFIX=${1#*=}; shift ;;
        --jobs)            [ $# -ge 2 ] || die "--jobs needs a number"; JOBS=$2; shift 2 ;;
        --jobs=*)          JOBS=${1#*=}; shift ;;
        --clean)           CLEAN=1; shift ;;
        --force)           FORCE=1; shift ;;
        --update)          UPDATE=1; shift ;;
        --no-fetch)        NO_FETCH=1; shift ;;
        --parallel-netcdf|--build-parallel-netcdf) BUILD_PARALLEL_NETCDF=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 die "unknown argument: $1 (try --help)" ;;
    esac
done

if [ -n "$SIESTA_SRC_OPT" ] && [ -n "$CLONE_DIR" ]; then
    die "--siesta-src and --clone-dir are mutually exclusive"
fi

abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$PWD" "$1" ;;
    esac
}

STAGE_ROOT=$(abspath "$STAGE_ROOT")
if [ -n "$SIESTA_SRC_OPT" ]; then SIESTA_SRC_OPT=$(abspath "$SIESTA_SRC_OPT"); fi
if [ -n "$CLONE_DIR" ]; then CLONE_DIR=$(abspath "$CLONE_DIR"); fi
if [ -n "$BUILD_DIR" ]; then BUILD_DIR=$(abspath "$BUILD_DIR"); else BUILD_DIR="$STAGE_ROOT/build"; fi
if [ -n "$INSTALL_PREFIX" ]; then INSTALL_PREFIX=$(abspath "$INSTALL_PREFIX"); else INSTALL_PREFIX="$STAGE_ROOT/install"; fi
LIBINT_BUILD=${SIESTA_LIBINT_BUILD:-"$STAGE_ROOT/libint-build"}
LIBINT_INSTALL=${SIESTA_LIBINT_INSTALL:-"$STAGE_ROOT/libint-install"}
PNC_ROOT=${SIESTA_PARALLEL_NETCDF_ROOT:-"$STAGE_ROOT/parallel-netcdf"}
PNC_ROOT=$(abspath "$PNC_ROOT")
PNC_PREFIX="$PNC_ROOT/install"
PNC_WORK="$PNC_ROOT/work"
PARALLEL_NETCDF_STAMP_SHA=""

case "$JOBS" in
    '') ;;
    *[!0-9]*|0) die "--jobs must be a positive integer" ;;
esac

mkdir -p "$STAGE_ROOT"

# ---------------------------------------------------------------------------
# Platform and tool discovery
# ---------------------------------------------------------------------------
OS=$(uname -s)
ARCH=$(uname -m)
case "$OS" in
    Darwin) PLATFORM=macos; SHLIB_EXT=dylib; LIB_PATH_VAR=DYLD_LIBRARY_PATH ;;
    Linux)  PLATFORM=linux; SHLIB_EXT=so;    LIB_PATH_VAR=LD_LIBRARY_PATH ;;
    *)      die "unsupported platform: $OS (this script supports Linux and macOS)" ;;
esac

HOMEBREW_PREFIX=${HOMEBREW_PREFIX:-}
if [ -z "$HOMEBREW_PREFIX" ] && command -v brew >/dev/null 2>&1; then
    HOMEBREW_PREFIX=$(brew --prefix 2>/dev/null || true)
fi

if [ -z "$JOBS" ]; then
    if [ "$PLATFORM" = macos ]; then
        JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        JOBS=$(nproc 2>/dev/null || echo 4)
    fi
fi

CMAKE_BIN=${CMAKE:-cmake}
command -v "$CMAKE_BIN" >/dev/null 2>&1 || die "cmake not found (set CMAKE or install CMake >= 3.20)"
CMAKE_VERSION=$("$CMAKE_BIN" --version | awk 'NR == 1 { print $3 }')
CMAKE_MAJOR=${CMAKE_VERSION%%.*}
POLICY_ARG=""
if [ "${CMAKE_MAJOR:-0}" -ge 4 ] 2>/dev/null; then
    # CMake 4 removed compatibility with cmake_minimum_required(<3.5).  The
    # bundled libint 1.2.1 (and its nested generator project) still uses 3.0.
    # The environment variable also reaches ExternalProject configure steps.
    export CMAKE_POLICY_VERSION_MINIMUM=3.5
    POLICY_ARG="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
fi

if [ -z "$GENERATOR" ]; then
    if command -v ninja >/dev/null 2>&1; then
        GENERATOR=Ninja
    elif command -v make >/dev/null 2>&1; then
        GENERATOR="Unix Makefiles"
    else
        die "neither ninja nor make found"
    fi
fi

command -v git >/dev/null 2>&1 || die "git not found"
command -v pkg-config >/dev/null 2>&1 || die "pkg-config not found (required by SIESTA's libint/FFTW detection; macOS: brew install pkg-config; Debian/Ubuntu: pkg-config)"
# Fail fast instead of hanging on an interactive credential prompt.  SSH agents
# and configured HTTPS credential helpers still work.
export GIT_TERMINAL_PROMPT=${GIT_TERMINAL_PROMPT:-0}
export GIT_SSH_COMMAND=${GIT_SSH_COMMAND:-"ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"}

if command -v shasum >/dev/null 2>&1; then
    HASH_CMD="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
    HASH_CMD="sha256sum"
else
    die "need shasum or sha256sum"
fi

hash_file() { $HASH_CMD "$1" | awk '{print $1}'; }
hash_stdin() { $HASH_CMD | awk '{print $1}'; }

# MPI environment / wrappers.
if [ -n "${MPI_HOME:-}" ]; then
    [ -d "$MPI_HOME" ] || die "MPI_HOME does not exist: $MPI_HOME"
    PATH="$MPI_HOME/bin:$PATH"
    export PATH
fi

resolve_exec() {
    case "$1" in
        */*) [ -x "$1" ] && printf '%s\n' "$1" ;;
        *)   command -v "$1" 2>/dev/null || true ;;
    esac
}

pick_exec() {
    # pick_exec VAR-name candidate...
    local var=$1; shift
    local candidate resolved
    for candidate in "$@"; do
        [ -n "$candidate" ] || continue
        resolved=$(resolve_exec "$candidate")
        if [ -n "$resolved" ]; then
            printf -v "$var" '%s' "$resolved"
            return 0
        fi
    done
    printf -v "$var" '%s' ''
    return 1
}

pick_exec C_COMPILER "${MPICC:-}" "${CC:-}" mpicc
pick_exec CXX_COMPILER "${MPICXX:-}" "${CXX:-}" mpicxx
pick_exec F_COMPILER "${MPIFC:-}" "${MPIF90:-}" "${FC:-}" mpifort mpif90

[ -n "$C_COMPILER" ] || die "no C compiler: install MPI (mpicc) or set CC/MPICC"
[ -n "$CXX_COMPILER" ] || die "no C++ compiler: install MPI (mpicxx) or set CXX/MPICXX (hybrid-XC needs it)"
[ -n "$F_COMPILER" ] || die "no Fortran compiler: install gfortran + MPI (mpifort) or set FC/MPIFC"

C_VERSION=$("$C_COMPILER" --version 2>/dev/null | awk 'NR == 1 { print; exit }' || true)
CXX_VERSION=$("$CXX_COMPILER" --version 2>/dev/null | awk 'NR == 1 { print; exit }' || true)
F_VERSION=$("$F_COMPILER" --version 2>/dev/null | awk 'NR == 1 { print; exit }' || true)

MPI_IFACE=${SIESTA_MPI_INTERFACES:-f08}
case "$MPI_IFACE" in
    f08|legacy|none) ;;
    *) die "SIESTA_MPI_INTERFACES must be f08, legacy or none (got '$MPI_IFACE')" ;;
esac

MPIRUN=${MPIRUN:-}
if [ -z "$MPIRUN" ]; then
    MPIRUN=$(resolve_exec mpirun || true)
fi
MPI_DESCRIPTION=""
if [ -n "$MPIRUN" ]; then
    MPI_DESCRIPTION=$("$MPIRUN" --version 2>&1 | awk 'NR <= 3 { print; exit }' || true)
fi
MPI_FAMILY=unknown
case "$MPI_DESCRIPTION" in
    *"Open MPI"*|*"OpenRTE"*) MPI_FAMILY=openmpi ;;
    *MPICH*|*"Microsoft MPI"*) MPI_FAMILY=mpich ;;
    *"Intel(R) MPI"*|*"Intel MPI"*) MPI_FAMILY=intelmpi ;;
esac

# Dependency search helpers.
DIRS=""
add_dir() {
    [ -n "${1:-}" ] || return 0
    [ -d "$1" ] || return 0
    case "
$DIRS
" in
        *"
$1
"*) return 0 ;;
    esac
    DIRS="${DIRS}${DIRS:+
}$1"
}

ldflag_dirs() {
    local flag_dir
    for flag_dir in $1; do
        case "$flag_dir" in
            -L*) printf '%s\n' "${flag_dir#-L}" ;;
        esac
    done
}

find_lib_in_dirs() {
    # find_lib_in_dirs NAME [DIRS...]
    local name=$1; shift
    local dir ext candidate
    for dir in "$@"; do
        [ -n "$dir" ] || continue
        [ -d "$dir" ] || continue
        for ext in dylib so; do
            candidate="$dir/lib$name.$ext"
            if [ -f "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
        for candidate in "$dir/lib$name".*."$SHLIB_EXT"; do
            [ -f "$candidate" ] || continue
            printf '%s\n' "$candidate"
            return 0
        done
    done
    return 1
}

multiarch=$("$C_COMPILER" -print-multiarch 2>/dev/null || true)

# --- NetCDF C / Fortran ----------------------------------------------------
build_parallel_netcdf() {
    [ -f "$PARALLEL_NETCDF_HELPER" ] || die "missing parallel NetCDF helper: $PARALLEL_NETCDF_HELPER"
    local pnc_cc pnc_fc
    pick_exec pnc_cc "${MPICC:-}" mpicc || die "parallel NetCDF needs mpicc (set MPICC or install MPI)"
    pick_exec pnc_fc "${MPIFC:-}" "${MPIF90:-}" mpifort mpif90 || die "parallel NetCDF needs mpifort/mpif90"
    log "Using reusable parallel NetCDF helper: $PARALLEL_NETCDF_HELPER"
    local helper_args=()
    helper_args=(--prefix "$PNC_PREFIX" --work-dir "$PNC_WORK" --jobs "$JOBS" --mpicc "$pnc_cc" --mpifort "$pnc_fc")
    if [ "$CLEAN" -eq 1 ]; then
        helper_args+=(--clean)
    fi
    bash "$PARALLEL_NETCDF_HELPER" "${helper_args[@]}" || die "parallel NetCDF stack build failed"
    [ -x "$PNC_PREFIX/bin/nc-config" ] || die "parallel NetCDF install missing nc-config"
    [ -x "$PNC_PREFIX/bin/nf-config" ] || die "parallel NetCDF install missing nf-config"
    [ -f "$PNC_PREFIX/.embasi-parallel-netcdf-stamp" ] || die "parallel NetCDF install missing summary stamp"
}

if [ "$BUILD_PARALLEL_NETCDF" -eq 1 ]; then
    build_parallel_netcdf
    NC_CONFIG="$PNC_PREFIX/bin/nc-config"
    NF_CONFIG="$PNC_PREFIX/bin/nf-config"
    NC_PREFIX="$PNC_PREFIX"
    NF_PREFIX="$PNC_PREFIX"
    NC_INCLUDE="$PNC_PREFIX/include"
    NF_INCLUDE="$PNC_PREFIX/include"
    NETCDF_C_LIB=$(find_lib_in_dirs netcdf "$PNC_PREFIX/lib" || true)
    NETCDF_F_LIB=$(find_lib_in_dirs netcdff "$PNC_PREFIX/lib" || true)
    [ -n "$NETCDF_C_LIB" ] || die "parallel NetCDF install is missing libnetcdf"
    [ -n "$NETCDF_F_LIB" ] || die "parallel NetCDF install is missing libnetcdff"
    [ -f "$NC_INCLUDE/netcdf.h" ] || die "parallel NetCDF install is missing netcdf.h"
    [ -f "$NF_INCLUDE/netcdf.mod" ] || die "parallel NetCDF install is missing netcdf.mod"
    if ! "$NC_CONFIG" --has-parallel4 2>/dev/null | grep -qi '^yes'; then
        die "self-built NetCDF-C does not report parallel I/O"
    fi
    NETCDF_PARALLEL=1
    NETCDF_MODE=parallel
    NCDF_PARALLEL=ON
    PARALLEL_NETCDF_STAMP_SHA=$(hash_file "$PNC_PREFIX/.embasi-parallel-netcdf-stamp")
    log "Using self-built parallel NetCDF: $PNC_PREFIX (MPI-IO enabled)"
else
NC_CONFIG=${NC_CONFIG:-}
if [ -z "$NC_CONFIG" ] && [ -n "${NETCDF_ROOT:-}" ] && [ -x "$NETCDF_ROOT/bin/nc-config" ]; then
    NC_CONFIG="$NETCDF_ROOT/bin/nc-config"
fi
if [ -z "$NC_CONFIG" ]; then
    NC_CONFIG=$(resolve_exec nc-config || true)
fi
[ -n "$NC_CONFIG" ] || die "nc-config not found: install NetCDF development files (macOS: brew install netcdf; Debian/Ubuntu: libnetcdf-dev) or set NC_CONFIG/NETCDF_ROOT"

NF_CONFIG=${NF_CONFIG:-}
if [ -z "$NF_CONFIG" ] && [ -n "${NETCDF_FORTRAN_ROOT:-}" ] && [ -x "$NETCDF_FORTRAN_ROOT/bin/nf-config" ]; then
    NF_CONFIG="$NETCDF_FORTRAN_ROOT/bin/nf-config"
fi
if [ -z "$NF_CONFIG" ] && [ -n "${NETCDF_ROOT:-}" ] && [ -x "$NETCDF_ROOT/bin/nf-config" ]; then
    NF_CONFIG="$NETCDF_ROOT/bin/nf-config"
fi
if [ -z "$NF_CONFIG" ]; then
    NF_CONFIG=$(resolve_exec nf-config || true)
fi
[ -n "$NF_CONFIG" ] || die "nf-config not found: install NetCDF-Fortran (macOS: brew install netcdf-fortran; Debian/Ubuntu: libnetcdff-dev) or set NF_CONFIG/NETCDF_FORTRAN_ROOT"

NC_PREFIX=$("$NC_CONFIG" --prefix 2>/dev/null || true)
NF_PREFIX=$("$NF_CONFIG" --prefix 2>/dev/null || true)
NC_INCLUDE=$("$NC_CONFIG" --includedir 2>/dev/null || true)
NF_INCLUDE=$("$NF_CONFIG" --includedir 2>/dev/null || true)
[ -n "$NC_INCLUDE" ] || NC_INCLUDE="$NC_PREFIX/include"
[ -n "$NF_INCLUDE" ] || NF_INCLUDE="$NF_PREFIX/include"

DIRS=""
for d in $(ldflag_dirs "$("$NC_CONFIG" --libs 2>/dev/null || true)"); do add_dir "$d"; done
add_dir "$NC_PREFIX/lib"
add_dir "$NC_PREFIX/lib64"
NETCDF_C_LIB=""
for candidate in $(printf '%s\n' "$DIRS"); do
    NETCDF_C_LIB=$(find_lib_in_dirs netcdf "$candidate" || true)
    [ -n "$NETCDF_C_LIB" ] && break
done
[ -n "$NETCDF_C_LIB" ] || die "libnetcdf not found under $NC_PREFIX (need a shared NetCDF C library)"

DIRS=""
for d in $(ldflag_dirs "$("$NF_CONFIG" --flibs 2>/dev/null || true)"); do add_dir "$d"; done
add_dir "$NF_PREFIX/lib"
add_dir "$NF_PREFIX/lib64"
NETCDF_F_LIB=""
for candidate in $(printf '%s\n' "$DIRS"); do
    NETCDF_F_LIB=$(find_lib_in_dirs netcdff "$candidate" || true)
    [ -n "$NETCDF_F_LIB" ] && break
done
[ -n "$NETCDF_F_LIB" ] || die "libnetcdff not found under $NF_PREFIX (need a shared NetCDF-Fortran library)"

[ -f "$NC_INCLUDE/netcdf.h" ] || die "netcdf.h not found in $NC_INCLUDE"
[ -f "$NF_INCLUDE/netcdf.mod" ] || die "netcdf.mod not found in $NF_INCLUDE"

netcdf_flag_yes() {
    # netcdf_flag_yes CONFIG-FLAG...
    local config=$1; shift
    local flag out
    for flag in "$@"; do
        if ! "$config" "$flag" >/dev/null 2>&1; then
            continue
        fi
        out=$("$config" "$flag" 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
        case "$out" in
            yes|true|1) return 0 ;;
            no|false|0) return 1 ;;
        esac
    done
    return 2
}

NETCDF_PARALLEL=0
if netcdf_flag_yes "$NC_CONFIG" --has-parallel4 --has-parallel; then
    # nc-config says yes.  Trust nf-config only if it explicitly says no.
    if netcdf_flag_yes "$NF_CONFIG" --has-parallel4 --has-parallel 2>/dev/null; then
        NETCDF_PARALLEL=1
    elif [ $? -eq 1 ]; then
        NETCDF_PARALLEL=0
    else
        NETCDF_PARALLEL=1
    fi
fi

if [ "$NETCDF_PARALLEL" -ne 1 ]; then
    die "No parallel NetCDF configured: '$NC_CONFIG' reports no MPI-IO and '$NF_CONFIG' is not confirmed parallel. SIESTA-QCOMP requires NF90_MPIIO. Use --parallel-netcdf to build a cached MPI HDF5/NetCDF stack, or set NC_CONFIG/NF_CONFIG (or NETCDF_ROOT) to an existing parallel NetCDF-C/NetCDF-Fortran installation. A NetCDF library without MPI-IO is not supported."
fi
NETCDF_MODE=parallel
NCDF_PARALLEL=ON
fi

# --- ScaLAPACK -------------------------------------------------------------
SCALAPACK_LIB=${SCALAPACK_LIBRARY:-}
if [ -z "$SCALAPACK_LIB" ]; then
    DIRS=""
    for pkg in scalapack scalapack-openmpi; do
        if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists "$pkg" 2>/dev/null; then
            add_dir "$(pkg-config --variable=libdir "$pkg" 2>/dev/null || true)"
            for d in $(ldflag_dirs "$(pkg-config --libs-only-L "$pkg" 2>/dev/null || true)"); do add_dir "$d"; done
        fi
    done
    [ -n "$HOMEBREW_PREFIX" ] && add_dir "$HOMEBREW_PREFIX/opt/scalapack/lib"
    add_dir "/opt/homebrew/opt/scalapack/lib"
    add_dir "/usr/local/opt/scalapack/lib"
    [ -n "$multiarch" ] && add_dir "/usr/lib/$multiarch"
    add_dir /usr/lib64
    add_dir /usr/lib
    add_dir /usr/local/lib
    [ -n "${MPI_HOME:-}" ] && add_dir "$MPI_HOME/lib"
    for candidate in $(printf '%s\n' "$DIRS"); do
        for name in scalapack scalapack-openmpi scalapack-mpi; do
            SCALAPACK_LIB=$(find_lib_in_dirs "$name" "$candidate" || true)
            [ -n "$SCALAPACK_LIB" ] && break 2
        done
    done
fi
[ -n "$SCALAPACK_LIB" ] || die "ScaLAPACK not found: install it (macOS: brew install scalapack; Debian/Ubuntu: libscalapack-openmpi-dev) or set SCALAPACK_LIBRARY"
[ -f "$SCALAPACK_LIB" ] || die "SCALAPACK_LIBRARY is not a file: $SCALAPACK_LIB"
SCALAPACK_LIB_DIR=$(cd "$(dirname "$SCALAPACK_LIB")" && pwd)

# --- BLAS / LAPACK ---------------------------------------------------------
BLAS_LIB=${BLAS_LIBRARY:-}
LAPACK_LIB=${LAPACK_LIBRARY:-}
if [ -z "$BLAS_LIB" ]; then
    DIRS=""
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists openblas 2>/dev/null; then
        add_dir "$(pkg-config --variable=libdir openblas 2>/dev/null || true)"
        for d in $(ldflag_dirs "$(pkg-config --libs-only-L openblas 2>/dev/null || true)"); do add_dir "$d"; done
    fi
    [ -n "$HOMEBREW_PREFIX" ] && add_dir "$HOMEBREW_PREFIX/opt/openblas/lib"
    add_dir "/opt/homebrew/opt/openblas/lib"
    add_dir "/usr/local/opt/openblas/lib"
    [ -n "$multiarch" ] && add_dir "/usr/lib/$multiarch"
    add_dir /usr/lib64
    add_dir /usr/lib
    add_dir /usr/local/lib
    for candidate in $(printf '%s\n' "$DIRS"); do
        BLAS_LIB=$(find_lib_in_dirs openblas "$candidate" || true)
        [ -n "$BLAS_LIB" ] && break
    done
fi
if [ -z "$BLAS_LIB" ]; then
    warn "OpenBLAS not found; falling back to a system BLAS/LAPACK pair"
    DIRS=""
    [ -n "$multiarch" ] && add_dir "/usr/lib/$multiarch"
    add_dir /usr/lib64
    add_dir /usr/lib
    add_dir /usr/local/lib
    for candidate in $(printf '%s\n' "$DIRS"); do
        BLAS_LIB=$(find_lib_in_dirs blas "$candidate" || true)
        [ -n "$BLAS_LIB" ] && break
    done
    [ -n "$LAPACK_LIB" ] || {
        for candidate in $(printf '%s\n' "$DIRS"); do
            LAPACK_LIB=$(find_lib_in_dirs lapack "$candidate" || true)
            [ -n "$LAPACK_LIB" ] && break
        done
    }
    [ -n "$LAPACK_LIB" ] || die "no LAPACK found next to BLAS; set BLAS_LIBRARY and LAPACK_LIBRARY explicitly"
fi
[ -f "$BLAS_LIB" ] || die "BLAS_LIBRARY is not a file: $BLAS_LIB"
if [ -z "$LAPACK_LIB" ]; then
    case "$(basename "$BLAS_LIB")" in
        libopenblas*) LAPACK_LIB=NONE ;;
        *) die "LAPACK_LIBRARY is unset; set it explicitly or use OpenBLAS (which provides LAPACK)" ;;
    esac
fi

# --- FFTW ------------------------------------------------------------------
FFTW_LIB=""
DIRS=""
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists fftw3 2>/dev/null; then
    add_dir "$(pkg-config --variable=libdir fftw3 2>/dev/null || true)"
fi
[ -n "$HOMEBREW_PREFIX" ] && add_dir "$HOMEBREW_PREFIX/opt/fftw/lib"
add_dir "/opt/homebrew/opt/fftw/lib"
add_dir "/usr/local/opt/fftw/lib"
[ -n "$multiarch" ] && add_dir "/usr/lib/$multiarch"
add_dir /usr/lib64
add_dir /usr/lib
add_dir /usr/local/lib
for candidate in $(printf '%s\n' "$DIRS"); do
    FFTW_LIB=$(find_lib_in_dirs fftw3 "$candidate" || true)
    [ -n "$FFTW_LIB" ] && break
done
[ -n "$FFTW_LIB" ] || die "libfftw3 not found: install FFTW (macOS: brew install fftw; Debian/Ubuntu: libfftw3-dev)"

# --- PKG_CONFIG_PATH and CMAKE_PREFIX_PATH ---------------------------------
PKG_DIRS=""
add_pkg_dir() {
    [ -n "${1:-}" ] || return 0
    [ -d "$1/lib/pkgconfig" ] || return 0
    case ":$PKG_DIRS:" in
        *":$1/lib/pkgconfig:"*) return 0 ;;
    esac
    PKG_DIRS="${PKG_DIRS}${PKG_DIRS:+:}$1/lib/pkgconfig"
}
add_pkg_dir "$NC_PREFIX"
add_pkg_dir "$NF_PREFIX"
add_pkg_dir "$(dirname "$SCALAPACK_LIB_DIR")"
add_pkg_dir "/opt/homebrew/opt/scalapack"
add_pkg_dir "/opt/homebrew/opt/fftw"
add_pkg_dir "/opt/homebrew/opt/openblas"
if [ -n "$PKG_DIRS" ]; then
    PKG_CONFIG_PATH="${PKG_DIRS}${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
    export PKG_CONFIG_PATH
fi

CMAKE_PREFIXES="$LIBINT_INSTALL"
[ -n "$HOMEBREW_PREFIX" ] && CMAKE_PREFIXES="$CMAKE_PREFIXES;$HOMEBREW_PREFIX"
[ -n "${MPI_HOME:-}" ] && CMAKE_PREFIXES="$CMAKE_PREFIXES;$MPI_HOME"
[ -n "$NC_PREFIX" ] && CMAKE_PREFIXES="$CMAKE_PREFIXES;$NC_PREFIX"
[ -n "$NF_PREFIX" ] && [ "$NF_PREFIX" != "$NC_PREFIX" ] && CMAKE_PREFIXES="$CMAKE_PREFIXES;$NF_PREFIX"
if [ "$BUILD_PARALLEL_NETCDF" -eq 1 ]; then
    CMAKE_PREFIXES="$CMAKE_PREFIXES;$PNC_PREFIX/hdf5"
fi
[ -n "${CMAKE_PREFIX_PATH:-}" ] && CMAKE_PREFIXES="$CMAKE_PREFIXES;$CMAKE_PREFIX_PATH"

# ---------------------------------------------------------------------------
# Source checkout / submodules
# ---------------------------------------------------------------------------
if [ -n "$SIESTA_SRC_OPT" ]; then
    SRC=$SIESTA_SRC_OPT
    MANAGED=0
    [ -d "$SRC" ] || die "--siesta-src does not exist: $SRC"
else
    SRC=${CLONE_DIR:-"$STAGE_ROOT/siesta-src"}
    MANAGED=1
fi

if [ ! -e "$SRC/.git" ]; then
    [ "$MANAGED" -eq 1 ] || die "--siesta-src is not a git checkout: $SRC"
    [ "$NO_FETCH" -eq 0 ] || die "--no-fetch was given but no existing clone is available at $SRC"
    log "Cloning $REMOTE ($BRANCH) into $SRC"
    mkdir -p "$(dirname "$SRC")"
    git clone --branch "$BRANCH" --single-branch "$REMOTE" "$SRC" || die "git clone failed (the project is private; make sure your SSH key/agent or credentialed URL works)"
elif [ "$MANAGED" -eq 1 ] && [ "$UPDATE" -eq 1 ]; then
    [ "$NO_FETCH" -eq 0 ] || die "--update and --no-fetch are incompatible"
    log "Fetching origin/$BRANCH and resetting $SRC"
    git -C "$SRC" remote set-url origin "$REMOTE"
    git -C "$SRC" fetch --prune origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH" || die "git fetch failed"
    git -C "$SRC" reset --hard "refs/remotes/origin/$BRANCH"
fi

if [ "$MANAGED" -eq 0 ]; then
    log "Using existing checkout $SRC (HEAD is never reset)"
    if [ "$BRANCH" != "$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || true)" ]; then
        warn "checkout is not on branch '$BRANCH'; building the current HEAD anyway"
    fi
    if [ "$UPDATE" -eq 1 ]; then
        warn "--update only applies to a managed clone; --siesta-src is used at its current HEAD"
    fi
fi

# Refuse obviously dangerous or foreign directories before any rm -rf.
ensure_removable_build_tree() {
    local dir=$1
    case "$dir" in
        /|"$HOME"|"$REPO_ROOT"|"$SRC"|"$SRC"/*)
            die "refusing unsafe build directory: $dir" ;;
    esac
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ] \
       && [ ! -f "$dir/CMakeCache.txt" ] \
       && [ ! -f "$dir/.embasi-configure-stamp" ] \
       && [ ! -f "$dir/.embasi-build-stamp" ]; then
        die "refusing to delete non-empty directory that is not a CMake build tree: $dir (choose an empty --build-dir or remove it yourself)"
    fi
}

ensure_removable_libint_prefix() {
    local dir=$1
    case "$dir" in
        /|"$HOME"|"$REPO_ROOT"|"$SRC"|"$SRC"/*)
            die "refusing unsafe libint install prefix: $dir" ;;
    esac
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ] \
       && [ ! -f "$dir/.embasi-libint-stamp" ] \
       && [ ! -f "$dir/lib/libint.a" ] && [ ! -f "$dir/lib64/libint.a" ]; then
        die "refusing to delete non-empty directory that is not a libint install: $dir"
    fi
}

REQUIRED_SUBMODULES="External/libfdf External/xmlf90 External/libpsml External/libgridxc External/libint asi-tests/asi"
if [ "$NO_FETCH" -eq 0 ]; then
    log "Updating required submodules"
    git -C "$SRC" submodule sync --recursive >/dev/null
    git -C "$SRC" submodule update --init --recursive -- $REQUIRED_SUBMODULES \
        || die "submodule update failed (needs network access to the public submodule remotes; use --no-fetch only with a fully initialized checkout)"
else
    log "Skipping network access (--no-fetch)"
fi

for required in \
    "$SRC/External/libint/CMakeLists.txt" \
    "$SRC/External/libfdf/CMakeLists.txt" \
    "$SRC/External/libgridxc/CMakeLists.txt" \
    "$SRC/External/libpsml/CMakeLists.txt" \
    "$SRC/External/xmlf90/CMakeLists.txt" \
    "$SRC/asi-tests/asi/include/asi.h"; do
    [ -e "$required" ] || die "required source/submodule file is missing: $required (run without --no-fetch, or initialize submodules in the checkout)"
done

SRC_COMMIT=$(git -C "$SRC" rev-parse HEAD)
LIBINT_COMMIT=$(git -C "$SRC" rev-parse "HEAD:External/libint" 2>/dev/null || true)
[ -n "$LIBINT_COMMIT" ] || die "cannot resolve External/libint submodule commit"

# Keep patch handling idempotent and non-destructive.
apply_patch() {
    local patch=$1 label=$2
    [ -f "$patch" ] || die "missing patch: $patch"
    if git -C "$SRC" apply --reverse --check "$patch" >/dev/null 2>&1; then
        log "Patch already applied: $label"
    elif git -C "$SRC" apply --check "$patch" >/dev/null 2>&1; then
        git -C "$SRC" apply "$patch"
        log "Applied patch: $label"
    else
        die "patch does not apply cleanly: $patch (source has conflicting local changes or is not the expected branch)"
    fi
}

apply_patch "$MAINTAINED_PATCH" "maintained ASI FFT reinit fix"

SOURCE_DIFF_HASH=$(git -C "$SRC" diff HEAD -- Src | hash_stdin)
MAIN_PATCH_HASH=$(hash_file "$MAINTAINED_PATCH")

# ---------------------------------------------------------------------------
# Clean / build directories
# ---------------------------------------------------------------------------
if [ "$CLEAN" -eq 1 ]; then
    log "Cleaning build trees (source and --install-prefix are preserved)"
    ensure_removable_build_tree "$BUILD_DIR"
    ensure_removable_build_tree "$LIBINT_BUILD"
    ensure_removable_libint_prefix "$LIBINT_INSTALL"
    rm -rf "$BUILD_DIR" "$LIBINT_BUILD" "$LIBINT_INSTALL"
fi
mkdir -p "$BUILD_DIR" "$LIBINT_BUILD" "$LIBINT_INSTALL"

# ---------------------------------------------------------------------------
# libint (bundled External/libint, MAX_AM_ERI=5)
# ---------------------------------------------------------------------------
LIBINT_STAMP="$LIBINT_INSTALL/.embasi-libint-stamp"
LIBINT_STAMP_TEXT="commit=$LIBINT_COMMIT
max_am_eri=5
build_fpic=ON
build_shared_libs=OFF
platform=$PLATFORM/$ARCH
cmake=$CMAKE_VERSION
cxx=$CXX_VERSION"

find_libint_archive() {
    local d
    for d in "$LIBINT_INSTALL/lib" "$LIBINT_INSTALL/lib64"; do
        if [ -f "$d/libint.a" ]; then
            printf '%s\n' "$d/libint.a"
            return 0
        fi
    done
    return 1
}

if [ "$FORCE" -eq 0 ] && [ -f "$LIBINT_STAMP" ] \
   && [ "$(cat "$LIBINT_STAMP")" = "$LIBINT_STAMP_TEXT" ] \
   && [ -n "$(find_libint_archive || true)" ] \
   && [ -f "$LIBINT_INSTALL/include/libint/libint.h" ] \
   && [ -f "$LIBINT_INSTALL/share/cmake/Libint/LibintConfig.cmake" ]; then
    log "libint cache HIT ($LIBINT_COMMIT, MAX_AM_ERI=5)"
else
    log "Building bundled libint $LIBINT_COMMIT with MAX_AM_ERI=5"
    rm -f "$LIBINT_STAMP"
    # Keep the build tree unless --clean was given; an interrupted build
    # resumes and an option change is caught by CMake.
    LIBINT_ARGS=(
        -S "$SRC/External/libint"
        -B "$LIBINT_BUILD"
        -G "$GENERATOR"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="$LIBINT_INSTALL"
        -DMAX_AM_ERI=5
        -DBUILD_FPIC=ON
        -DBUILD_SHARED_LIBS=OFF
    )
    if [ -n "$POLICY_ARG" ]; then
        LIBINT_ARGS+=("$POLICY_ARG")
    fi
    "$CMAKE_BIN" "${LIBINT_ARGS[@]}"
    # CMAKE_POLICY_VERSION_MINIMUM is exported for the nested generator
    # ExternalProject when CMake >= 4; set it explicitly as well for callers
    # that use a non-exporting environment.
    CMAKE_POLICY_VERSION_MINIMUM=3.5 "$CMAKE_BIN" --build "$LIBINT_BUILD" --target install --parallel "$JOBS"
    [ -n "$(find_libint_archive || true)" ] || die "libint install is missing libint.a under $LIBINT_INSTALL"
    [ -f "$LIBINT_INSTALL/include/libint/libint.h" ] || die "libint install is missing include/libint/libint.h"
    [ -f "$LIBINT_INSTALL/share/cmake/Libint/LibintConfig.cmake" ] || die "libint install is missing its CMake package config"
    printf '%s\n' "$LIBINT_STAMP_TEXT" > "$LIBINT_STAMP"
    log "libint build complete"
fi

# Put the freshly built libint first so a distro libint (which may use
# MAX_AM_ERI=7) can never win the pkg-config lookup.  SIESTA's libint search
# prefers pkg-config before its CMake package; the .pc file points at our
# static libint/libderiv and has no C++ runtime dependency.
PKG_CONFIG_PATH="$LIBINT_INSTALL/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PKG_CONFIG_PATH
LIBINT_PC_VERSION=$(pkg-config --modversion libint 2>/dev/null || true)
if [ -n "$LIBINT_PC_VERSION" ]; then
    log "libint pkg-config reports $LIBINT_PC_VERSION from $LIBINT_INSTALL"
else
    die "the built libint is not visible through pkg-config (expected $LIBINT_INSTALL/lib/pkgconfig/libint.pc)"
fi

# ---------------------------------------------------------------------------
# SIESTA configure/build
# ---------------------------------------------------------------------------
BUILD_CONFIG="platform=$PLATFORM/$ARCH
branch=$BRANCH
source_commit=$SRC_COMMIT
source_diff_sha256=$SOURCE_DIFF_HASH
maintained_patch_sha256=$MAIN_PATCH_HASH
libint_commit=$LIBINT_COMMIT
libint_install=$LIBINT_INSTALL
install_prefix=$INSTALL_PREFIX
netcdf=parallel
ncdf_parallel=$NCDF_PARALLEL
parallel_netcdf=$([ "$BUILD_PARALLEL_NETCDF" -eq 1 ] && echo "$PNC_PREFIX" || echo none)
parallel_netcdf_stamp_sha256=${PARALLEL_NETCDF_STAMP_SHA:-none}
netcdf_c=$NETCDF_C_LIB
netcdf_fortran=$NETCDF_F_LIB
scalapack=$SCALAPACK_LIB
blas=$BLAS_LIB
lapack=$LAPACK_LIB
fftw=$FFTW_LIB
build_type=Release
shared=ON
mpi=ON
mpi_interfaces=$MPI_IFACE
fftw=ON
hybridxc=ON
qcomp=ON
asi=ON
libint_max_am_eri=5
tests=OFF
openmp=OFF
cmake=$CMAKE_VERSION
cc=$C_VERSION
cxx=$CXX_VERSION
fc=$F_VERSION"

CONFIG_STAMP="$BUILD_DIR/.embasi-configure-stamp"
BUILD_STAMP="$BUILD_DIR/.embasi-build-stamp"

find_libsiesta() {
    local d name
    for d in "$BUILD_DIR/Src" "$BUILD_DIR/lib"; do
        for name in "libsiesta.$SHLIB_EXT" libsiesta.so libsiesta.dylib; do
            if [ -f "$d/$name" ]; then
                printf '%s\n' "$d/$name"
                return 0
            fi
        done
    done
    return 1
}

if [ "$FORCE" -eq 1 ]; then
    rm -f "$CONFIG_STAMP" "$BUILD_STAMP"
fi

CONFIGURE_ARGS=(
    -S "$SRC"
    -B "$BUILD_DIR"
    -G "$GENERATOR"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    -DCMAKE_C_COMPILER="$C_COMPILER"
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER"
    -DCMAKE_Fortran_COMPILER="$F_COMPILER"
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIXES"
    -DLibint_DIR="$LIBINT_INSTALL/share/cmake/Libint"
    -DBLAS_LIBRARY="$BLAS_LIB"
    -DLAPACK_LIBRARY="$LAPACK_LIB"
    -DSCALAPACK_LIBRARY="$SCALAPACK_LIB"
    -DNetCDF_C_CONFIG_EXECUTABLE="$NC_CONFIG"
    -DNetCDF_Fortran_CONFIG_EXECUTABLE="$NF_CONFIG"
    -DNetCDF_C_INCLUDE_FILE="$NC_INCLUDE/netcdf.h"
    -DNetCDF_Fortran_INCLUDE_FILE="$NF_INCLUDE/netcdf.mod"
    -DNetCDF_C_LIBRARY="$NETCDF_C_LIB"
    -DNetCDF_Fortran_LIBRARY="$NETCDF_F_LIB"
    -DBUILD_SHARED_LIBS=ON
    -DSIESTA_WITH_MPI=ON
    -DSIESTA_WITH_MPI_INTERFACES="$MPI_IFACE"
    -DSIESTA_WITH_NETCDF=ON
    -DSIESTA_WITH_NCDF_PARALLEL="$NCDF_PARALLEL"
    -DSIESTA_WITH_FFTW=ON
    -DSIESTA_WITH_HYBRIDXC=ON
    -DSIESTA_WITH_QCOMP=ON
    -DSIESTA_WITH_ASI=ON
    -DASI_INCLUDE_DIR="$SRC/asi-tests/asi/include"
    -DSIESTA_WITH_ELSI=OFF
    -DSIESTA_WITH_ELPA=OFF
    -DSIESTA_WITH_FLOOK=OFF
    -DSIESTA_WITH_DFTD3=OFF
    -DSIESTA_WITH_CHESS=OFF
    -DSIESTA_WITH_WANNIER90=OFF
    -DSIESTA_WITH_JSON=OFF
    -DSIESTA_WITH_ZEROMQ=OFF
    -DSIESTA_WITH_LIBXC=OFF
    -DSIESTA_WITH_OPENMP=OFF
    -DSIESTA_TESTS=OFF
)

if [ "$FORCE" -eq 0 ] && [ -f "$BUILD_STAMP" ] \
   && [ "$(cat "$BUILD_STAMP")" = "$BUILD_CONFIG" ] \
   && [ -n "$(find_libsiesta || true)" ]; then
    log "SIESTA build cache HIT ($SRC_COMMIT)"
else
    rm -f "$BUILD_STAMP"
    if [ -f "$CONFIG_STAMP" ] && [ "$(cat "$CONFIG_STAMP")" = "$BUILD_CONFIG" ]; then
        log "Reusing configured build tree (incremental build)"
    else
        log "Configuring SIESTA (ASI + QCOMP + ${NETCDF_MODE} NetCDF)"
        ensure_removable_build_tree "$BUILD_DIR"
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"
        "$CMAKE_BIN" "${CONFIGURE_ARGS[@]}"
        printf '%s\n' "$BUILD_CONFIG" > "$CONFIG_STAMP"
    fi
    log "Building SIESTA and libsiesta.$SHLIB_EXT"
    "$CMAKE_BIN" --build "$BUILD_DIR" --parallel "$JOBS"
    LIBSIESTA=$(find_libsiesta || true)
    [ -n "$LIBSIESTA" ] || die "build finished without Src/libsiesta.$SHLIB_EXT"
    printf '%s\n' "$BUILD_CONFIG" > "$BUILD_STAMP.tmp"
    mv "$BUILD_STAMP.tmp" "$BUILD_STAMP"
    log "SIESTA build complete"
fi

LIBSIESTA=$(find_libsiesta || true)
[ -n "$LIBSIESTA" ] || die "no libsiesta.$SHLIB_EXT in $BUILD_DIR (stale stamp or failed build)"
LIBSIESTA_DIR=$(cd "$(dirname "$LIBSIESTA")" && pwd)

log "Installing SIESTA into $INSTALL_PREFIX"
INSTALL_LOG="$STAGE_ROOT/install.log"
if ! "$CMAKE_BIN" --install "$BUILD_DIR" >"$INSTALL_LOG" 2>&1; then
    tail -40 "$INSTALL_LOG" >&2 || true
    die "cmake --install failed for prefix $INSTALL_PREFIX (log: $INSTALL_LOG)"
fi

# ---------------------------------------------------------------------------
# Runtime environment and test command
# ---------------------------------------------------------------------------
RUNTIME_DIRS="$LIBSIESTA_DIR"
add_runtime_dir() {
    [ -n "${1:-}" ] || return 0
    [ -d "$1" ] || return 0
    case ":$RUNTIME_DIRS:" in
        *":$1:"*) return 0 ;;
    esac
    RUNTIME_DIRS="${RUNTIME_DIRS}:$1"
}
add_runtime_dir "$(dirname "$NETCDF_C_LIB")"
add_runtime_dir "$(dirname "$NETCDF_F_LIB")"
add_runtime_dir "$(dirname "$SCALAPACK_LIB")"
add_runtime_dir "$(dirname "$BLAS_LIB")"
add_runtime_dir "$(dirname "$FFTW_LIB")"
if [ "$BUILD_PARALLEL_NETCDF" -eq 1 ]; then
    add_runtime_dir "$PNC_PREFIX/hdf5/lib"
fi
[ -n "${MPI_HOME:-}" ] && add_runtime_dir "$MPI_HOME/lib"
# Do not add the whole Homebrew lib directory: libsiesta links its external
# dependencies by absolute install names (and internal ones through @rpath),
# so a broad DYLD_LIBRARY_PATH would only risk interposing unrelated libraries.

PP_PATH="$SRC/Tests/Pseudos"
[ -d "$PP_PATH" ] || die "pseudopotential directory missing: $PP_PATH"

PYTEST=""
if [ -x "$REPO_ROOT/.venv/bin/python" ] \
   && "$REPO_ROOT/.venv/bin/python" -I -c 'import pytest' >/dev/null 2>&1; then
    PYTEST="$REPO_ROOT/.venv/bin/python -m pytest"
else
    PYTEST="python3 -m pytest"
fi

ENV_FILE="$STAGE_ROOT/env.sh"
{
    printf '# Generated by scripts/build_siesta_native.sh -- source this file.\n'
    printf "export ASI_LIB_PATH='%s'\n" "$LIBSIESTA"
    printf "export SIESTA_PP_PATH='%s'\n" "$PP_PATH"
    printf "export PYTHONPATH='%s%s'\n" "$REPO_ROOT/src" "${PYTHONPATH:+:$PYTHONPATH}"
    printf "export %s='%s%s'\n" "$LIB_PATH_VAR" "$RUNTIME_DIRS" "${!LIB_PATH_VAR:+:${!LIB_PATH_VAR}}"
    printf 'export OMP_NUM_THREADS=1\n'
    printf 'export OPENBLAS_NUM_THREADS=1\n'
    printf 'export SIESTA_QCOMP_REAL=1\n'
} > "$ENV_FILE.tmp"
mv "$ENV_FILE.tmp" "$ENV_FILE"

if [ -x "$REPO_ROOT/.venv/bin/python" ] \
   && "$REPO_ROOT/.venv/bin/python" -I -c 'pass' >/dev/null 2>&1; then
    ASI4PY_VERSION=$("$REPO_ROOT/.venv/bin/python" -I -c \
        'import importlib.metadata as m; print(m.version("asi4py"))' 2>/dev/null || true)
    if [ -z "$ASI4PY_VERSION" ]; then
        warn "asi4py is not installed in $REPO_ROOT/.venv; the real test will skip"
    else
        log "asi4py $ASI4PY_VERSION in $REPO_ROOT/.venv"
    fi
fi

cat <<EOF

============================================================================
SIESTA native build complete
  platform      : $PLATFORM/$ARCH
  source        : $SRC
  source commit : $SRC_COMMIT
  library       : $LIBSIESTA
  install       : $INSTALL_PREFIX
  NetCDF mode   : $NETCDF_MODE (SIESTA_WITH_NCDF_PARALLEL=$NCDF_PARALLEL)
  libint        : $LIBINT_INSTALL (MAX_AM_ERI=5)
  jobs          : $JOBS
  env file      : $ENV_FILE

Environment for the real WF-in-DFT test:
  ASI_LIB_PATH=$LIBSIESTA
  SIESTA_PP_PATH=$PP_PATH
  $LIB_PATH_VAR=$RUNTIME_DIRS
  PYTHONPATH=$REPO_ROOT/src
  SIESTA_QCOMP_REAL=1

Run the test (about 5-12 minutes; QCOMP's nao2gto Gaussian fitting dominates):
  source "$ENV_FILE"
  $PYTEST -q tests/test_siesta_qcomp_real.py

With the build tree env:
  cd "$REPO_ROOT" && SIESTA_QCOMP_REAL=1 \\
    ASI_LIB_PATH="$LIBSIESTA" \\
    SIESTA_PP_PATH="$PP_PATH" \\
    PYTHONPATH="$REPO_ROOT/src" \\
    $LIB_PATH_VAR="$RUNTIME_DIRS" \\
    $PYTEST -q tests/test_siesta_qcomp_real.py
EOF

