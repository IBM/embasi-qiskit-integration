# Builder image for the full SIESTA-QCOMP / EmbASI integration test.
#
# This image carries the toolchain, the system libraries and a shared,
# MPI-enabled HDF5 + NetCDF-C + NetCDF-Fortran stack installed under
# /opt/parallel-netcdf.  The stack is built by the reusable
# scripts/build_parallel_netcdf.sh helper (copied into the build context by the
# harness), so SIESTA uses SIESTA_WITH_NCDF_PARALLEL=ON and QCOMP's native
# NF90_MPIIO path.  No system NetCDF package is installed in the image.
#
# The SIESTA source, libint (built from SIESTA's External/libint submodule with
# MAX_AM_ERI=5), the build tree and the Python venv live in the named cache
# volume, so rebuilding this image does not throw away the expensive build.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        gfortran \
        g++ \
        gcc \
        make \
        pkg-config \
        git \
        openssh-client \
        ca-certificates \
        curl \
        autoconf \
        automake \
        libtool \
        libtool-bin \
        bison \
        flex \
        m4 \
        perl \
        gzip \
        zlib1g-dev \
        openmpi-bin \
        libopenmpi-dev \
        libscalapack-openmpi-dev \
        libopenblas-dev \
        liblapack-dev \
        libfftw3-dev \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# The harness stages this file next to the Dockerfile before `docker build`.
COPY build_parallel_netcdf.sh /usr/local/bin/build_parallel_netcdf.sh
RUN chmod +x /usr/local/bin/build_parallel_netcdf.sh \
    && /usr/local/bin/build_parallel_netcdf.sh \
         --prefix /opt/parallel-netcdf \
         --work-dir /tmp/parallel-netcdf-work \
         --jobs "$(nproc)" \
         --cleanup

ENV PATH=/opt/parallel-netcdf/bin:$PATH

# The harness mounts the named cache volume here and runs everything as root.
WORKDIR /cache
