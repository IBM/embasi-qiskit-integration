#!/usr/bin/env bash
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#
# Build a reusable MPI-enabled HDF5 + NetCDF-C + NetCDF-Fortran stack.
#
# Used by scripts/build_siesta_native.sh (--parallel-netcdf) and by the
# docker/siesta-asi builder image.  The stack is installed under <prefix>:
#
#   <prefix>/hdf5/                 HDF5 (parallel, shared, HL)
#   <prefix>/bin/nc-config         NetCDF-C + NetCDF-Fortran (shared)
#   <prefix>/bin/nf-config
#   <prefix>/include/netcdf.h, netcdf.mod
#   <prefix>/lib/libnetcdf.*, libnetcdff.*
#   <prefix>/.embasi-*-stamp       component and summary stamps
#
# Work state (downloads, extracted sources, build trees, logs, probe) lives in
# --work-dir.  --clean removes the install and build trees but keeps the
# downloads/sources; --cleanup removes the whole work directory (for image
# layers).  The build is idempotent: unchanged versions, compilers, options and
# source checksums are cache hits.

set -euo pipefail

trap 'printf "ERROR: %s failed at line %s: %s\n" "${0##*/}" "$LINENO" "$BASH_COMMAND" >&2' ERR

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DEFAULT_HDF5_VERSION=1.14.6
DEFAULT_NETCDF_C_VERSION=4.9.3
DEFAULT_NETCDF_FORTRAN_VERSION=4.6.2
DEFAULT_HDF5_SHA256=e4defbac30f50d64e1556374aa49e574417c9e72c6b1de7a4ff88c4b1bea6e9b
DEFAULT_NETCDF_C_SHA256=a474149844e6144566673facf097fea253dc843c37bc0a7d3de047dc8adda5dd
DEFAULT_NETCDF_FORTRAN_SHA256=df26b99d9003c93a8bc287b58172bf1c279676f8c10d6dd0daf8bc7204877096

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: scripts/build_parallel_netcdf.sh [options]

Build a shared, MPI-enabled HDF5, NetCDF-C and NetCDF-Fortran stack and verify
that NF90_MPIIO works with a small Fortran/MPI program.

  --prefix DIR          Install prefix (default: $PNC_PREFIX or
                        ./parallel-netcdf-install).  Must be dedicated to this
                        stack.  HDF5 is installed in DIR/hdf5; NetCDF-C and
                        NetCDF-Fortran share DIR.
  --work-dir DIR        Downloads, sources, build trees, logs and probe cache
                        (default: $PNC_WORK_DIR or <prefix>.work).
  --jobs N              Parallel make jobs (default: $JOBS or CPU count).
  --mpicc CMD           MPI C compiler (default: $MPICC or mpicc).
  --mpifort CMD         MPI Fortran compiler (default: $MPIFC, $MPIF90,
                        mpifort, or mpif90).
  --clean               Remove the install and build trees, keep downloads and
                        extracted sources, then rebuild.
  --force               Alias for --clean.
  --cleanup             After a successful build, remove --work-dir (useful for
                        Docker image layers; the install and stamps remain).
  --no-probe            Skip the runtime NF90_MPIIO probe (not recommended).
  --hdf5-version V      HDF5 version (default 1.14.6).
  --netcdf-c-version V  NetCDF-C version (default 4.9.3).
  --netcdf-fortran-version V
                        NetCDF-Fortran version (default 4.6.2).
  --hdf5-url URL, --netcdf-c-url URL, --netcdf-fortran-url URL
  --hdf5-sha256 H, --netcdf-c-sha256 H, --netcdf-fortran-sha256 H
                        Override source URLs/checksums.  When changing a
                        version also set the matching checksum.
  -h, --help            Show this help.

Environment equivalents: PNC_PREFIX, PNC_WORK_DIR, JOBS, MPICC, MPIFC, MPIF90,
HDF5_VERSION, NETCDF_C_VERSION, NETCDF_FORTRAN_VERSION, HDF5_SOURCE_URL,
NETCDF_C_SOURCE_URL, NETCDF_FORTRAN_SOURCE_URL, HDF5_SHA256,
NETCDF_C_SHA256, NETCDF_FORTRAN_SHA256.

On success the script prints the installed prefixes, nc-config/nf-config paths,
include directories and shared-library paths.
EOF
}

PREFIX=${PNC_PREFIX:-"$PWD/parallel-netcdf-install"}
WORK_DIR=${PNC_WORK_DIR:-"$PREFIX.work"}
JOBS=${JOBS:-}
MPICC_OPT=${MPICC:-}
MPIFC_OPT=${MPIFC:-${MPIF90:-}}
HDF5_VERSION_OPT=${HDF5_VERSION:-}
NETCDF_C_VERSION_OPT=${NETCDF_C_VERSION:-}
NETCDF_FORTRAN_VERSION_OPT=${NETCDF_FORTRAN_VERSION:-}
HDF5_URL_OPT=${HDF5_SOURCE_URL:-}
NETCDF_C_URL_OPT=${NETCDF_C_SOURCE_URL:-}
NETCDF_FORTRAN_URL_OPT=${NETCDF_FORTRAN_SOURCE_URL:-}
HDF5_SHA_OPT=${HDF5_SHA256:-}
NETCDF_C_SHA_OPT=${NETCDF_C_SHA256:-}
NETCDF_FORTRAN_SHA_OPT=${NETCDF_FORTRAN_SHA256:-}
CLEAN=0
CLEANUP=0
DO_PROBE=1

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix)        [ $# -ge 2 ] || die "--prefix needs a directory"; PREFIX=$2; shift 2 ;;
        --prefix=*)      PREFIX=${1#*=}; shift ;;
        --work-dir)      [ $# -ge 2 ] || die "--work-dir needs a directory"; WORK_DIR=$2; shift 2 ;;
        --work-dir=*)    WORK_DIR=${1#*=}; shift ;;
        --jobs)          [ $# -ge 2 ] || die "--jobs needs a number"; JOBS=$2; shift 2 ;;
        --jobs=*)        JOBS=${1#*=}; shift ;;
        --mpicc)         [ $# -ge 2 ] || die "--mpicc needs a command"; MPICC_OPT=$2; shift 2 ;;
        --mpicc=*)       MPICC_OPT=${1#*=}; shift ;;
        --mpifort)       [ $# -ge 2 ] || die "--mpifort needs a command"; MPIFC_OPT=$2; shift 2 ;;
        --mpifort=*)     MPIFC_OPT=${1#*=}; shift ;;
        --clean|--force) CLEAN=1; shift ;;
        --cleanup)       CLEANUP=1; shift ;;
        --no-probe)      DO_PROBE=0; shift ;;
        --hdf5-version)  [ $# -ge 2 ] || die "--hdf5-version needs a value"; HDF5_VERSION_OPT=$2; shift 2 ;;
        --hdf5-version=*) HDF5_VERSION_OPT=${1#*=}; shift ;;
        --netcdf-c-version) [ $# -ge 2 ] || die "--netcdf-c-version needs a value"; NETCDF_C_VERSION_OPT=$2; shift 2 ;;
        --netcdf-c-version=*) NETCDF_C_VERSION_OPT=${1#*=}; shift ;;
        --netcdf-fortran-version) [ $# -ge 2 ] || die "--netcdf-fortran-version needs a value"; NETCDF_FORTRAN_VERSION_OPT=$2; shift 2 ;;
        --netcdf-fortran-version=*) NETCDF_FORTRAN_VERSION_OPT=${1#*=}; shift ;;
        --hdf5-url)      [ $# -ge 2 ] || die "--hdf5-url needs a value"; HDF5_URL_OPT=$2; shift 2 ;;
        --hdf5-url=*)    HDF5_URL_OPT=${1#*=}; shift ;;
        --netcdf-c-url)  [ $# -ge 2 ] || die "--netcdf-c-url needs a value"; NETCDF_C_URL_OPT=$2; shift 2 ;;
        --netcdf-c-url=*) NETCDF_C_URL_OPT=${1#*=}; shift ;;
        --netcdf-fortran-url) [ $# -ge 2 ] || die "--netcdf-fortran-url needs a value"; NETCDF_FORTRAN_URL_OPT=$2; shift 2 ;;
        --netcdf-fortran-url=*) NETCDF_FORTRAN_URL_OPT=${1#*=}; shift ;;
        --hdf5-sha256)   [ $# -ge 2 ] || die "--hdf5-sha256 needs a value"; HDF5_SHA_OPT=$2; shift 2 ;;
        --hdf5-sha256=*) HDF5_SHA_OPT=${1#*=}; shift ;;
        --netcdf-c-sha256) [ $# -ge 2 ] || die "--netcdf-c-sha256 needs a value"; NETCDF_C_SHA_OPT=$2; shift 2 ;;
        --netcdf-c-sha256=*) NETCDF_C_SHA_OPT=${1#*=}; shift ;;
        --netcdf-fortran-sha256) [ $# -ge 2 ] || die "--netcdf-fortran-sha256 needs a value"; NETCDF_FORTRAN_SHA_OPT=$2; shift 2 ;;
        --netcdf-fortran-sha256=*) NETCDF_FORTRAN_SHA_OPT=${1#*=}; shift ;;
        -h|--help)       usage; exit 0 ;;
        *)               die "unknown argument: $1 (try --help)" ;;
    esac
done

abspath() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *)  printf '%s/%s\n' "$PWD" "$1" ;;
    esac
}

PREFIX=$(abspath "$PREFIX")
WORK_DIR=$(abspath "$WORK_DIR")
HDF5_PREFIX="$PREFIX/hdf5"
NC_PREFIX="$PREFIX"
DOWNLOADS="$WORK_DIR/downloads"
SRC_DIR="$WORK_DIR/src"
BUILD_DIR="$WORK_DIR/build"
LOGS="$WORK_DIR/logs"
PROBE_DIR="$WORK_DIR/probe"

case "$PREFIX" in /|"$HOME"|""|/usr|/usr/local|/opt|/var|/tmp) die "refusing unsafe --prefix: '$PREFIX'" ;; esac
case "$WORK_DIR" in /|"$HOME"|""|/usr|/usr/local|/opt|/var|/tmp) die "refusing unsafe --work-dir: '$WORK_DIR'" ;; esac
[ "$PREFIX" != "$WORK_DIR" ] || die "--prefix and --work-dir must differ"
case "$PREFIX" in "$WORK_DIR"/*) die "--work-dir must not contain --prefix: $WORK_DIR / $PREFIX" ;; esac
case "$WORK_DIR" in "$PREFIX"/*) die "--work-dir must not be inside --prefix: $WORK_DIR / $PREFIX" ;; esac

for tool in curl tar make cmake; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool not found"
done
if command -v shasum >/dev/null 2>&1; then
    HASH_CMD="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
    HASH_CMD="sha256sum"
else
    die "need shasum or sha256sum"
fi
hash_file() { $HASH_CMD "$1" | awk '{print $1}'; }

resolve_exec() {
    case "$1" in
        */*) [ -x "$1" ] && printf '%s\n' "$1" ;;
        *)   command -v "$1" 2>/dev/null || true ;;
    esac
}

pick_exec() {
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

pick_exec MPICC_BIN "$MPICC_OPT" mpicc || die "mpicc not found (set --mpicc or MPICC)"
pick_exec MPIFC_BIN "$MPIFC_OPT" mpifort mpif90 || die "mpifort/mpif90 not found (set --mpifort or MPIFC)"

# macOS OpenMPI wrappers unconditionally append -Wl,-flat_namespace.  A
# flat-namespace libhdf5/libnetcdf lets an unrelated HDF5 already loaded in the
# Python process (h5py bundles HDF5 2.x) interpose on the stack's HDF5 1.14
# symbols and crashes NetCDF.  Build the stack with wrappers that use the same
# compiler/MPI flags but omit -Wl,-flat_namespace, so the stack's dylibs keep
# the default two-level namespace.
prepare_no_flat_wrappers() {
    [ "$(uname -s)" = Darwin ] || return 0
    local show_cc show_fc tool_dir
    show_cc=$("$MPICC_BIN" -show 2>/dev/null || true)
    show_fc=$("$MPIFC_BIN" -show 2>/dev/null || true)
    case "$show_cc$show_fc" in
        *-Wl,-flat_namespace*) ;;
        *) return 0 ;;
    esac
    tool_dir="$WORK_DIR/toolchain"
    mkdir -p "$tool_dir"
    show_cc=$(printf '%s' "$show_cc" | sed 's/-Wl,-flat_namespace//g')
    show_fc=$(printf '%s' "$show_fc" | sed 's/-Wl,-flat_namespace//g')
    [ -n "$show_cc" ] || die "cannot derive a no-flat C compiler command from $MPICC_BIN"
    [ -n "$show_fc" ] || die "cannot derive a no-flat Fortran compiler command from $MPIFC_BIN"
    printf '#!/bin/sh\nexec %s "$@"\n' "$show_cc" > "$tool_dir/mpicc-noflat"
    printf '#!/bin/sh\nexec %s "$@"\n' "$show_fc" > "$tool_dir/mpifort-noflat"
    chmod +x "$tool_dir/mpicc-noflat" "$tool_dir/mpifort-noflat"
    MPICC_BIN="$tool_dir/mpicc-noflat"
    MPIFC_BIN="$tool_dir/mpifort-noflat"
    log "macOS MPI wrappers use -Wl,-flat_namespace; building the NetCDF stack with no-flat wrappers under $tool_dir"
}

prepare_no_flat_wrappers
MPICC_VERSION=$("$MPICC_BIN" --version 2>/dev/null | awk 'NR == 1 { print; exit }' || true)
MPIFC_VERSION=$("$MPIFC_BIN" --version 2>/dev/null | awk 'NR == 1 { print; exit }' || true)

HDF5_VERSION=${HDF5_VERSION_OPT:-$DEFAULT_HDF5_VERSION}
NETCDF_C_VERSION=${NETCDF_C_VERSION_OPT:-$DEFAULT_NETCDF_C_VERSION}
NETCDF_FORTRAN_VERSION=${NETCDF_FORTRAN_VERSION_OPT:-$DEFAULT_NETCDF_FORTRAN_VERSION}
HDF5_SHA256=${HDF5_SHA_OPT:-$DEFAULT_HDF5_SHA256}
NETCDF_C_SHA256=${NETCDF_C_SHA_OPT:-$DEFAULT_NETCDF_C_SHA256}
NETCDF_FORTRAN_SHA256=${NETCDF_FORTRAN_SHA_OPT:-$DEFAULT_NETCDF_FORTRAN_SHA256}
HDF5_SOURCE_URL=${HDF5_URL_OPT:-"https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz"}
NETCDF_C_SOURCE_URL=${NETCDF_C_URL_OPT:-"https://downloads.unidata.ucar.edu/netcdf-c/${NETCDF_C_VERSION}/netcdf-c-${NETCDF_C_VERSION}.tar.gz"}
NETCDF_FORTRAN_SOURCE_URL=${NETCDF_FORTRAN_URL_OPT:-"https://downloads.unidata.ucar.edu/netcdf-fortran/${NETCDF_FORTRAN_VERSION}/netcdf-fortran-${NETCDF_FORTRAN_VERSION}.tar.gz"}

if [ "$HDF5_VERSION" != "$DEFAULT_HDF5_VERSION" ] && [ "$HDF5_SHA256" = "$DEFAULT_HDF5_SHA256" ]; then
    die "HDF5_VERSION=$HDF5_VERSION but HDF5_SHA256 is the default 1.14.6 checksum; set HDF5_SHA256"
fi
if [ "$NETCDF_C_VERSION" != "$DEFAULT_NETCDF_C_VERSION" ] && [ "$NETCDF_C_SHA256" = "$DEFAULT_NETCDF_C_SHA256" ]; then
    die "NETCDF_C_VERSION=$NETCDF_C_VERSION but NETCDF_C_SHA256 is the default 4.9.3 checksum; set NETCDF_C_SHA256"
fi
if [ "$NETCDF_FORTRAN_VERSION" != "$DEFAULT_NETCDF_FORTRAN_VERSION" ] && [ "$NETCDF_FORTRAN_SHA256" = "$DEFAULT_NETCDF_FORTRAN_SHA256" ]; then
    die "NETCDF_FORTRAN_VERSION=$NETCDF_FORTRAN_VERSION but NETCDF_FORTRAN_SHA256 is the default 4.6.2 checksum; set NETCDF_FORTRAN_SHA256"
fi

if [ -z "$JOBS" ]; then
    if [ "$(uname -s)" = Darwin ]; then
        JOBS=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    else
        JOBS=$(nproc 2>/dev/null || echo 4)
    fi
fi
case "$JOBS" in *[!0-9]*|0) die "--jobs must be a positive integer" ;; esac

OS_NAME=$(uname -s)
if [ "$OS_NAME" = Darwin ]; then
    SHLIB_EXT=dylib
    LIB_PATH_VAR=DYLD_LIBRARY_PATH
else
    SHLIB_EXT=so
    LIB_PATH_VAR=LD_LIBRARY_PATH
fi
PLATFORM="$OS_NAME/$(uname -m)"

HDF5_TAR="$DOWNLOADS/hdf5-$HDF5_VERSION.tar.gz"
NETCDF_C_TAR="$DOWNLOADS/netcdf-c-$NETCDF_C_VERSION.tar.gz"
NETCDF_FORTRAN_TAR="$DOWNLOADS/netcdf-fortran-$NETCDF_FORTRAN_VERSION.tar.gz"
HDF5_SRC="$SRC_DIR/hdf5-$HDF5_VERSION"
NETCDF_C_SRC="$SRC_DIR/netcdf-c-$NETCDF_C_VERSION"
NETCDF_FORTRAN_SRC="$SRC_DIR/netcdf-fortran-$NETCDF_FORTRAN_VERSION"
HDF5_BUILD="$BUILD_DIR/hdf5"
NETCDF_C_BUILD="$BUILD_DIR/netcdf-c"
NETCDF_FORTRAN_BUILD="$BUILD_DIR/netcdf-fortran"

HDF5_STAMP="$HDF5_PREFIX/.embasi-hdf5-stamp"
NETCDF_C_STAMP="$NC_PREFIX/.embasi-netcdf-c-stamp"
NETCDF_FORTRAN_STAMP="$NC_PREFIX/.embasi-netcdf-fortran-stamp"
SUMMARY_STAMP="$NC_PREFIX/.embasi-parallel-netcdf-stamp"
PREFIX_MARKER="$NC_PREFIX/.embasi-parallel-netcdf-prefix"

# On macOS, autotools/libtool emits a flat-namespace HDF5 dylib, which lets
# h5py's bundled HDF5 2.x interpose on the stack's HDF5 1.14 symbols and
# crashes NetCDF; HDF5's CMake build keeps the default two-level namespace.
# Linux keeps the already-validated autotools build (flat namespace is not a
# concern with ELF symbol versioning/loading).
if [ "$OS_NAME" = Darwin ]; then
    HDF5_OPTIONS="cmake:HDF5_ENABLE_PARALLEL=ON;BUILD_SHARED_LIBS=ON;HDF5_BUILD_HL_LIB=ON;HDF5_BUILD_TOOLS=OFF;HDF5_BUILD_EXAMPLES=OFF;HDF5_BUILD_CPP_LIB=OFF;HDF5_BUILD_FORTRAN=OFF;BUILD_TESTING=OFF"
else
    HDF5_OPTIONS="--enable-parallel --enable-shared --disable-static --disable-fortran --disable-cxx --disable-java --disable-tools --disable-tests --enable-hl"
fi
NETCDF_C_OPTIONS="--enable-parallel-tests --disable-dap --disable-byterange --disable-testsets --enable-shared --disable-static"
NETCDF_FORTRAN_OPTIONS="--enable-parallel-tests --enable-shared --disable-static"

# Stable identities, not component stamp hashes: the stamp files do not exist
# yet when these are first computed, and a cache-hit must reproduce them.
HDF5_IDENTITY="version=$HDF5_VERSION;sha256=$HDF5_SHA256;platform=$PLATFORM;mpicc=$MPICC_BIN;mpicc_version=$MPICC_VERSION;options=$HDF5_OPTIONS"
NETCDF_C_LDFLAGS="-L$HDF5_PREFIX/lib -Wl,-rpath,$HDF5_PREFIX/lib"
NETCDF_FORTRAN_LDFLAGS="-L$NC_PREFIX/lib -Wl,-rpath,$NC_PREFIX/lib -Wl,-rpath,$HDF5_PREFIX/lib"
NETCDF_C_IDENTITY="version=$NETCDF_C_VERSION;sha256=$NETCDF_C_SHA256;platform=$PLATFORM;mpicc=$MPICC_BIN;mpicc_version=$MPICC_VERSION;hdf5={$HDF5_IDENTITY};ldflags={$NETCDF_C_LDFLAGS};options=$NETCDF_C_OPTIONS"
NETCDF_FORTRAN_IDENTITY="version=$NETCDF_FORTRAN_VERSION;sha256=$NETCDF_FORTRAN_SHA256;platform=$PLATFORM;mpicc=$MPICC_BIN;mpifort=$MPIFC_BIN;mpicc_version=$MPICC_VERSION;mpifort_version=$MPIFC_VERSION;netcdf_c={$NETCDF_C_IDENTITY};ldflags={$NETCDF_FORTRAN_LDFLAGS};options=$NETCDF_FORTRAN_OPTIONS"
HDF5_STAMP_TEXT=$HDF5_IDENTITY
NETCDF_C_STAMP_TEXT=$NETCDF_C_IDENTITY
NETCDF_FORTRAN_STAMP_TEXT=$NETCDF_FORTRAN_IDENTITY

mkdir -p "$WORK_DIR" "$DOWNLOADS" "$SRC_DIR" "$BUILD_DIR" "$LOGS" "$PROBE_DIR"
mkdir -p "$NC_PREFIX"
if [ -e "$NC_PREFIX" ] && [ -n "$(ls -A "$NC_PREFIX" 2>/dev/null)" ] && [ ! -f "$PREFIX_MARKER" ]; then
    die "prefix is not empty and was not created by this helper: $NC_PREFIX (choose a dedicated --prefix)"
fi
touch "$PREFIX_MARKER"

component_valid() {
    case "$1" in
        hdf5)
            [ -f "$HDF5_STAMP" ] && [ "$(cat "$HDF5_STAMP")" = "$HDF5_STAMP_TEXT" ] \
                && [ -f "$HDF5_PREFIX/lib/libhdf5.$SHLIB_EXT" ] \
                && [ -f "$HDF5_PREFIX/lib/libhdf5_hl.$SHLIB_EXT" ] \
                && [ -f "$HDF5_PREFIX/include/hdf5.h" ] ;;
        netcdf-c)
            [ -f "$NETCDF_C_STAMP" ] && [ "$(cat "$NETCDF_C_STAMP")" = "$NETCDF_C_STAMP_TEXT" ] \
                && [ -f "$NC_PREFIX/lib/libnetcdf.$SHLIB_EXT" ] \
                && [ -x "$NC_PREFIX/bin/nc-config" ] \
                && [ -f "$NC_PREFIX/include/netcdf.h" ] ;;
        netcdf-fortran)
            [ -f "$NETCDF_FORTRAN_STAMP" ] && [ "$(cat "$NETCDF_FORTRAN_STAMP")" = "$NETCDF_FORTRAN_STAMP_TEXT" ] \
                && [ -f "$NC_PREFIX/lib/libnetcdff.$SHLIB_EXT" ] \
                && [ -x "$NC_PREFIX/bin/nf-config" ] \
                && [ -f "$NC_PREFIX/include/netcdf.mod" ] ;;
        *) return 1 ;;
    esac
}

if [ "$CLEAN" -eq 1 ]; then
    log "Cleaning parallel NetCDF install and build trees (downloads/sources kept)"
    rm -rf "$PREFIX" "$BUILD_DIR"
    mkdir -p "$NC_PREFIX"
    touch "$PREFIX_MARKER"
fi

cache_ok=1
for component in hdf5 netcdf-c netcdf-fortran; do
    if ! component_valid "$component"; then
        cache_ok=0
    fi
done

if [ "$cache_ok" -eq 1 ]; then
    log "parallel NetCDF cache HIT ($HDF5_VERSION / $NETCDF_C_VERSION / $NETCDF_FORTRAN_VERSION)"
else
    log "parallel NetCDF cache MISS: building HDF5 $HDF5_VERSION + NetCDF-C $NETCDF_C_VERSION + NetCDF-Fortran $NETCDF_FORTRAN_VERSION"
    # Rebuild the stack from a clean prefix; keep downloads/extracted sources.
    rm -rf "$PREFIX" "$BUILD_DIR"
    mkdir -p "$NC_PREFIX" "$HDF5_PREFIX" "$BUILD_DIR"
    touch "$PREFIX_MARKER"

    download() {
        # download NAME URL SHA DEST
        local name=$1 url=$2 sha=$3 dest=$4 got
        if [ -f "$dest" ] && [ "$(hash_file "$dest")" = "$sha" ]; then
            log "$name source cache HIT ($(basename "$dest"))"
            return 0
        fi
        log "Downloading $name: $url"
        rm -f "$dest.part"
        curl -L --fail --retry 3 --max-time 900 -sS -o "$dest.part" "$url" \
            || die "download failed: $url"
        got=$(hash_file "$dest.part")
        [ "$got" = "$sha" ] || die "$name checksum mismatch: got $got expected $sha"
        mv "$dest.part" "$dest"
    }

    extract() {
        # extract TARBALL DEST
        local tarball=$1 dest=$2
        if [ -x "$dest/configure" ] && [ "$tarball" -ot "$dest/configure" ]; then
            log "source cache HIT: $(basename "$dest")"
            return 0
        fi
        rm -rf "$dest"
        log "Extracting $(basename "$tarball")"
        tar -xzf "$tarball" -C "$SRC_DIR" || die "cannot extract $tarball"
        [ -x "$dest/configure" ] || die "extracted source has no configure: $dest"
    }

    run_log() {
        # run_log LOG DESCRIPTION COMMAND...
        local logfile=$1 description=$2; shift 2
        log "$description (log: $logfile)"
        if ! "$@" >"$logfile" 2>&1; then
            tail -80 "$logfile" >&2 || true
            die "$description failed"
        fi
    }

    download HDF5 "$HDF5_SOURCE_URL" "$HDF5_SHA256" "$HDF5_TAR"
    download NetCDF-C "$NETCDF_C_SOURCE_URL" "$NETCDF_C_SHA256" "$NETCDF_C_TAR"
    download NetCDF-Fortran "$NETCDF_FORTRAN_SOURCE_URL" "$NETCDF_FORTRAN_SHA256" "$NETCDF_FORTRAN_TAR"
    extract "$HDF5_TAR" "$HDF5_SRC"
    extract "$NETCDF_C_TAR" "$NETCDF_C_SRC"
    extract "$NETCDF_FORTRAN_TAR" "$NETCDF_FORTRAN_SRC"

    mkdir -p "$HDF5_BUILD"
    if [ "$OS_NAME" = Darwin ]; then
        if command -v ninja >/dev/null 2>&1; then
            HDF5_GENERATOR=Ninja
        else
            HDF5_GENERATOR="Unix Makefiles"
        fi
        (
            cmake -S "$HDF5_SRC" -B "$HDF5_BUILD" -G "$HDF5_GENERATOR" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX="$HDF5_PREFIX" \
                -DCMAKE_C_COMPILER="$MPICC_BIN" \
                -DMPI_C_COMPILER="$MPICC_BIN" \
                -DHDF5_ENABLE_PARALLEL=ON \
                -DBUILD_SHARED_LIBS=ON \
                -DHDF5_BUILD_HL_LIB=ON \
                -DHDF5_BUILD_TOOLS=OFF \
                -DHDF5_BUILD_EXAMPLES=OFF \
                -DHDF5_BUILD_CPP_LIB=OFF \
                -DHDF5_BUILD_FORTRAN=OFF \
                -DBUILD_TESTING=OFF
        ) >"$LOGS/hdf5-configure.log" 2>&1 || {
            tail -80 "$LOGS/hdf5-configure.log" >&2 || true
            die "HDF5 CMake configure failed"
        }
        run_log "$LOGS/hdf5-build.log" "Building/installing HDF5" \
            cmake --build "$HDF5_BUILD" --target install --parallel "$JOBS"
    else
        (
            cd "$HDF5_BUILD"
            CC="$MPICC_BIN" CFLAGS="-O2 -fPIC" "$HDF5_SRC/configure" \
                --prefix="$HDF5_PREFIX" $HDF5_OPTIONS
        ) >"$LOGS/hdf5-configure.log" 2>&1 || {
            tail -80 "$LOGS/hdf5-configure.log" >&2 || true
            die "HDF5 configure failed"
        }
        run_log "$LOGS/hdf5-build.log" "Building HDF5" make -C "$HDF5_BUILD" -j "$JOBS"
        run_log "$LOGS/hdf5-install.log" "Installing HDF5" make -C "$HDF5_BUILD" install
    fi
    [ -f "$HDF5_PREFIX/lib/libhdf5.$SHLIB_EXT" ] || die "HDF5 install missing libhdf5.$SHLIB_EXT"
    [ -f "$HDF5_PREFIX/lib/libhdf5_hl.$SHLIB_EXT" ] || die "HDF5 install missing libhdf5_hl.$SHLIB_EXT"
    [ -f "$HDF5_PREFIX/include/hdf5.h" ] || die "HDF5 install missing include/hdf5.h"
    printf '%s\n' "$HDF5_STAMP_TEXT" > "$HDF5_STAMP"
    log "HDF5 build complete"

    export DYLD_LIBRARY_PATH="$HDF5_PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$HDF5_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    mkdir -p "$NETCDF_C_BUILD"
    (
        cd "$NETCDF_C_BUILD"
        CC="$MPICC_BIN" CFLAGS="-O2 -fPIC" \
            CPPFLAGS="-I$HDF5_PREFIX/include" \
            LDFLAGS="$NETCDF_C_LDFLAGS" \
            "$NETCDF_C_SRC/configure" --prefix="$NC_PREFIX" $NETCDF_C_OPTIONS
    ) >"$LOGS/netcdf-c-configure.log" 2>&1 || {
        tail -80 "$LOGS/netcdf-c-configure.log" >&2 || true
        die "NetCDF-C configure failed"
    }
    run_log "$LOGS/netcdf-c-build.log" "Building NetCDF-C" make -C "$NETCDF_C_BUILD" -j "$JOBS"
    run_log "$LOGS/netcdf-c-install.log" "Installing NetCDF-C" make -C "$NETCDF_C_BUILD" install
    [ -f "$NC_PREFIX/lib/libnetcdf.$SHLIB_EXT" ] || die "NetCDF-C install missing libnetcdf.$SHLIB_EXT"
    [ -x "$NC_PREFIX/bin/nc-config" ] || die "NetCDF-C install missing bin/nc-config"
    [ -f "$NC_PREFIX/include/netcdf.h" ] || die "NetCDF-C install missing include/netcdf.h"
    if ! "$NC_PREFIX/bin/nc-config" --has-parallel4 2>/dev/null | grep -qi '^yes'; then
        die "NetCDF-C was built without parallel I/O (nc-config --has-parallel4 is not yes)"
    fi
    printf '%s\n' "$NETCDF_C_STAMP_TEXT" > "$NETCDF_C_STAMP"
    log "NetCDF-C build complete (parallel I/O enabled)"

    export DYLD_LIBRARY_PATH="$NC_PREFIX/lib:$HDF5_PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$NC_PREFIX/lib:$HDF5_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    mkdir -p "$NETCDF_FORTRAN_BUILD"
    (
        cd "$NETCDF_FORTRAN_BUILD"
        CC="$MPICC_BIN" FC="$MPIFC_BIN" FCFLAGS="-O2 -fPIC" \
            CPPFLAGS="-I$NC_PREFIX/include" \
            LDFLAGS="$NETCDF_FORTRAN_LDFLAGS" \
            PATH="$NC_PREFIX/bin:$PATH" \
            "$NETCDF_FORTRAN_SRC/configure" --prefix="$NC_PREFIX" $NETCDF_FORTRAN_OPTIONS
    ) >"$LOGS/netcdf-fortran-configure.log" 2>&1 || {
        tail -80 "$LOGS/netcdf-fortran-configure.log" >&2 || true
        die "NetCDF-Fortran configure failed"
    }
    run_log "$LOGS/netcdf-fortran-build.log" "Building NetCDF-Fortran" make -C "$NETCDF_FORTRAN_BUILD" -j "$JOBS"
    run_log "$LOGS/netcdf-fortran-install.log" "Installing NetCDF-Fortran" make -C "$NETCDF_FORTRAN_BUILD" install
    [ -f "$NC_PREFIX/lib/libnetcdff.$SHLIB_EXT" ] || die "NetCDF-Fortran install missing libnetcdff.$SHLIB_EXT"
    [ -x "$NC_PREFIX/bin/nf-config" ] || die "NetCDF-Fortran install missing bin/nf-config"
    [ -f "$NC_PREFIX/include/netcdf.mod" ] || die "NetCDF-Fortran install missing include/netcdf.mod"
    printf '%s\n' "$NETCDF_FORTRAN_STAMP_TEXT" > "$NETCDF_FORTRAN_STAMP"
    log "NetCDF-Fortran build complete"
fi

if [ "$DO_PROBE" -eq 1 ]; then
    export DYLD_LIBRARY_PATH="$NC_PREFIX/lib:$HDF5_PREFIX/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH="$NC_PREFIX/lib:$HDF5_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cat > "$PROBE_DIR/probe.f90" <<'EOF'
program probe
  use netcdf
  use mpi_f08
  implicit none
  integer :: ncid, dimid, varid, iret
  call MPI_Init()
  iret = nf90_create('pnc_probe.nc', ior(NF90_NETCDF4, NF90_MPIIO), ncid, &
                     comm=MPI_COMM_WORLD%mpi_val, info=MPI_INFO_NULL%mpi_val)
  if (iret /= NF90_NOERR) error stop trim(nf90_strerror(iret))
  iret = nf90_def_dim(ncid, 'n', 1, dimid)
  if (iret /= NF90_NOERR) error stop trim(nf90_strerror(iret))
  iret = nf90_def_var(ncid, 'x', NF90_INT, (/dimid/), varid)
  if (iret /= NF90_NOERR) error stop trim(nf90_strerror(iret))
  iret = nf90_enddef(ncid)
  if (iret /= NF90_NOERR) error stop trim(nf90_strerror(iret))
  iret = nf90_var_par_access(ncid, varid, NF90_COLLECTIVE)
  if (iret /= NF90_NOERR) error stop trim(nf90_strerror(iret))
  iret = nf90_put_var(ncid, varid, 42)
  if (iret /= NF90_NOERR) error stop trim(nf90_strerror(iret))
  iret = nf90_close(ncid)
  if (iret /= NF90_NOERR) error stop trim(nf90_strerror(iret))
  call MPI_Finalize()
  print '(A)', 'PARALLEL_NETCDF_PROBE_OK'
end program probe
EOF
    rm -f "$PROBE_DIR/probe" "$PROBE_DIR/pnc_probe.nc"
    if ! "$MPIFC_BIN" "$PROBE_DIR/probe.f90" \
            -I"$NC_PREFIX/include" \
            -L"$NC_PREFIX/lib" -L"$HDF5_PREFIX/lib" \
            -lnetcdff -lnetcdf \
            -Wl,-rpath,"$NC_PREFIX/lib" -Wl,-rpath,"$HDF5_PREFIX/lib" \
            -o "$PROBE_DIR/probe" >"$LOGS/probe-compile.log" 2>&1; then
        tail -60 "$LOGS/probe-compile.log" >&2 || true
        die "parallel NetCDF probe failed to compile"
    fi
    if ! ( cd "$PROBE_DIR" && ./probe ) >"$LOGS/probe-run.log" 2>&1; then
        tail -60 "$LOGS/probe-run.log" >&2 || true
        die "parallel NetCDF MPI-IO probe failed to run (see $LOGS/probe-run.log)"
    fi
    grep -q 'PARALLEL_NETCDF_PROBE_OK' "$LOGS/probe-run.log" \
        || die "parallel NetCDF probe did not report success"
    [ -s "$PROBE_DIR/pnc_probe.nc" ] || die "parallel NetCDF probe did not write a file"
    log "parallel NetCDF MPI-IO probe OK"
fi

SUMMARY_STAMP_TEXT="hdf5_version=$HDF5_VERSION
netcdf_c_version=$NETCDF_C_VERSION
netcdf_fortran_version=$NETCDF_FORTRAN_VERSION
hdf5_identity=$HDF5_IDENTITY
netcdf_c_identity=$NETCDF_C_IDENTITY
netcdf_fortran_identity=$NETCDF_FORTRAN_IDENTITY
nc_has_parallel4=$("$NC_PREFIX/bin/nc-config" --has-parallel4 2>/dev/null || echo unknown)
nf_has_nc4=$("$NC_PREFIX/bin/nf-config" --has-nc4 2>/dev/null || echo unknown)
probe=$( [ "$DO_PROBE" -eq 1 ] && echo ok || echo skipped )
platform=$PLATFORM"
printf '%s\n' "$SUMMARY_STAMP_TEXT" > "$SUMMARY_STAMP"

if [ "$CLEANUP" -eq 1 ]; then
    case "$WORK_DIR" in
        /|"$HOME"|"$PREFIX") die "refusing to clean unsafe work dir: $WORK_DIR" ;;
    esac
    log "Removing work directory $WORK_DIR (install/stamps kept)"
    rm -rf "$WORK_DIR"
fi

cat <<EOF

Parallel NetCDF stack ready
  prefix              : $NC_PREFIX
  HDF5 prefix         : $HDF5_PREFIX
  nc-config           : $NC_PREFIX/bin/nc-config
  nf-config           : $NC_PREFIX/bin/nf-config
  C include           : $NC_PREFIX/include/netcdf.h
  Fortran include     : $NC_PREFIX/include/netcdf.mod
  C library           : $NC_PREFIX/lib/libnetcdf.$SHLIB_EXT
  Fortran library     : $NC_PREFIX/lib/libnetcdff.$SHLIB_EXT
  HDF5 library        : $HDF5_PREFIX/lib/libhdf5.$SHLIB_EXT
  parallel I/O        : $("$NC_PREFIX/bin/nc-config" --has-parallel4 2>/dev/null || echo unknown)
  runtime $LIB_PATH_VAR : $NC_PREFIX/lib:$HDF5_PREFIX/lib
  summary stamp       : $SUMMARY_STAMP
EOF
