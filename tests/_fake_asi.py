# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0

"""Build a fake ASI flavour-3 shared library for SIESTA backend tests.

The real ``libsiesta.so`` needs a Linux SIESTA build with MPI, ScaLAPACK,
libint and NetCDF.  This stand-in implements the ASI C subset ``asi4py`` and
EmbASI actually call, with SIESTA's observable behaviour (one *total* H per SCF
step, type-101 basis descriptors), so the Python seam can be tested without
SIESTA.  It validates protocol wiring, not physics.

Compile with the host C compiler (``cc``/``clang``/``gcc``).  Tests skip when no
compiler is available; the Linux build is exercised in CI/containers.
"""

from __future__ import annotations

import inspect
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ScaLAPACK/BLACS symbols scalapack4py resolves when it wraps libsiesta.so.  The
# list is introspected from the installed package so a scalapack4py upgrade does
# not silently break the fake library.
_BLACS = (
    "blacs_get",
    "blacs_gridinit",
    "blacs_gridexit",
    "blacs_gridmap",
    "blacs_gridinfo",
    "numroc",
    "descinit",
)


def _scalapack_symbols() -> list[str]:
    from scalapack4py import func_argtypes

    source = inspect.getsource(func_argtypes)
    keys = re.findall(r'^\s*"([a-z0-9]+)"\s*:\s*\(', source, re.M)
    return sorted(set(keys) | set(_BLACS))


_C_TEMPLATE = r"""
/* Fake SIESTA ASI library for Python-side tests. ASI flavour 3. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

typedef struct { int matrix_type; int storage_type; } ASI_matrix_descr_t;
typedef void (*ASI_dmhs_callback_t)(void*, int, int, int*, const void*, ASI_matrix_descr_t*);
typedef int (*ASI_set_dmhs_callback_t)(void*, int, int, int*, void*, ASI_matrix_descr_t*);

#define N_ATOMS 3
#define N_BASIS 7
#define N_SPIN 1
#define N_KPTS 1
#define NBASIS_MAX 64

static int g_active = 0;
static int g_converged = 1;
static int g_ham_calls = 1;
static int g_last_ham_calls = 0;
static int g_dm_import_changed = 0;
static double g_energy = -10.0;
static int g_basis_atom[NBASIS_MAX];
typedef struct { int descr_size; int basis_func_descr_type; int atom_index; } local_descr_t;
static local_descr_t g_descrs[NBASIS_MAX];
static char g_last_input[512];

static ASI_dmhs_callback_t cb_dm, cb_ovlp, cb_ham;
static ASI_set_dmhs_callback_t cb_dm_init, cb_set_ham, cb_mod_ham;
static void *aux_dm, *aux_ovlp, *aux_ham, *aux_dm_init, *aux_set_ham, *aux_mod_ham;

static void fill_diag(double *m, double v) {
    int i;
    memset(m, 0, N_BASIS * N_BASIS * sizeof(double));
    for (i = 0; i < N_BASIS; i++) m[i + i * N_BASIS] = v;
}

int ASI_flavour(void) { return 3; }

void ASI_init(const char *inputpath, const char *outputfilename, int comm) {
    int i;
    const char *env;
    (void)outputfilename;
    (void)comm;
    g_active = 1; g_converged = 0; g_last_ham_calls = 0; g_dm_import_changed = 0;
    strncpy(g_last_input, inputpath ? inputpath : "", sizeof(g_last_input) - 1);
    for (i = 0; i < N_BASIS; i++) {
        g_basis_atom[i] = (i * N_ATOMS) / N_BASIS;
        g_descrs[i].descr_size = sizeof(local_descr_t);
        g_descrs[i].basis_func_descr_type = 101;
        g_descrs[i].atom_index = g_basis_atom[i];
    }
    env = getenv("FAKE_REQUIRE_INPUT");
    if (env && atoi(env) && access(inputpath, F_OK) != 0) {
        fprintf(stderr, "fake ASI: input '%s' not found in cwd\n", inputpath);
        exit(3);
    }
    env = getenv("FAKE_HAM_CALLS");
    g_ham_calls = env ? atoi(env) : 1;
    if (g_ham_calls < 1) g_ham_calls = 1;
    env = getenv("FAKE_ENERGY");
    if (env) g_energy = atof(env);
}

void ASI_finalize(void) { g_active = 0; }

void ASI_run(void) {
    double S[NBASIS_MAX * NBASIS_MAX], H[NBASIS_MAX * NBASIS_MAX], DM[NBASIS_MAX * NBASIS_MAX];
    ASI_matrix_descr_t md = {1, 0};  /* HERMSYM, DENSE_FULL */
    int step, i;
    if (!g_active) { fprintf(stderr, "fake ASI_run before init\n"); exit(4); }
    fill_diag(S, 1.0);
    fill_diag(DM, 0.0);
    for (i = 0; i < 5; i++) DM[i + i * N_BASIS] = 2.0;  /* 10 electrons, closed shell */
    /* SIESTA order: overlap, dm_init, dm, then one total H per SCF step. */
    if (cb_ovlp) cb_ovlp(aux_ovlp, 1, 1, NULL, S, &md);
    if (cb_dm_init) g_dm_import_changed = cb_dm_init(aux_dm_init, 1, 1, NULL, DM, &md);
    if (cb_dm) cb_dm(aux_dm, 1, 1, NULL, DM, &md);
    fill_diag(H, 0.0);
    for (step = 0; step < g_ham_calls; step++) {
        /* Distinct per export, so "last H" is observable in the tests. */
        for (i = 0; i < N_BASIS; i++) H[i + i * N_BASIS] = -0.5 + 0.1 * i + 0.01 * step;
        if (cb_ham) cb_ham(aux_ham, 1, 1, NULL, H, &md);
    }
    g_last_ham_calls = g_ham_calls;
    g_converged = 1;
}

int ASI_n_atoms(void) { return N_ATOMS; }
double ASI_energy(void) { return g_energy; }
const double *ASI_forces(void) { return NULL; }
const double *ASI_stress(void) { return NULL; }
const double *ASI_atomic_charges(int scheme) { (void)scheme; return NULL; }
void ASI_calc_esp(int n, const double *c, double *p, double *g) {
    (void)n; (void)c; (void)p; (void)g;
}
int ASI_get_basis_size(void) { return N_BASIS; }
int ASI_get_nspin(void) { return N_SPIN; }
int ASI_get_nkpts(void) { return N_KPTS; }
int ASI_get_n_local_ks(void) { return N_SPIN * N_KPTS; }
void ASI_get_local_ks(int *ks) {
    int s, k, i = 0;
    for (s = 1; s <= N_SPIN; s++) for (k = 1; k <= N_KPTS; k++) ks[i++] = s * 10 + k;
}
int ASI_is_hamiltonian_real(void) { return 1; }
int ASI_is_scf_converged(void) { return g_converged; }
void *ASI_get_basis_func_descr(int i) {
    if (i < 0 || i >= N_BASIS) return NULL;
    return (void *)&g_descrs[i];
}
const double *ASI_get_k_points(void) { static double k[3] = {0, 0, 0}; return k; }

void ASI_set_atom_coords(const double *c, int n) { (void)c; (void)n; }
void ASI_set_geometry(const double *c, int n, const double *l) { (void)c; (void)n; (void)l; }
void ASI_set_external_potential(void *cb, void *aux) { (void)cb; (void)aux; }
void ASI_register_external_potential(void *cb, void *aux) { (void)cb; (void)aux; }
void ASI_register_dm_callback(ASI_dmhs_callback_t cb, void *aux) { cb_dm = cb; aux_dm = aux; }
void ASI_register_overlap_callback(ASI_dmhs_callback_t cb, void *aux) { cb_ovlp = cb; aux_ovlp = aux; }
void ASI_register_hamiltonian_callback(ASI_dmhs_callback_t cb, void *aux) { cb_ham = cb; aux_ham = aux; }
void ASI_register_set_hamiltonian_callback(ASI_set_dmhs_callback_t cb, void *aux) { cb_set_ham = cb; aux_set_ham = aux; }
void ASI_register_modify_hamiltonian_callback(ASI_set_dmhs_callback_t cb, void *aux) { cb_mod_ham = cb; aux_mod_ham = aux; }
void ASI_register_dm_init_callback(ASI_set_dmhs_callback_t cb, void *aux) { cb_dm_init = cb; aux_dm_init = aux; }

/* Introspection used by the tests. */
const char *fake_last_input(void) { return g_last_input; }
int fake_last_ham_calls(void) { return g_last_ham_calls; }
int fake_dm_import_changed(void) { return g_dm_import_changed; }

/* ScaLAPACK/BLACS stubs: scalapack4py wraps the same library and resolves
   every routine at construction even though dense ASI matrices never use them. */
$stubs
"""


def build_fake_asi_library(directory: str | Path) -> Path | None:
    """Compile the fake ASI library into ``directory``; None when no compiler."""
    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "fake_libsiesta.c"
    stubs = "\n".join(f"void {name}_(void) {{}}" for name in _scalapack_symbols())
    source.write_text(_C_TEMPLATE.replace("$stubs", stubs))
    library = directory / (
        "libsiesta_fake.dylib" if sys.platform == "darwin" else "libsiesta_fake.so"
    )
    if sys.platform == "darwin":
        command = [compiler, "-dynamiclib", "-fPIC", "-O1", "-o", str(library), str(source)]
    else:
        command = [compiler, "-shared", "-fPIC", "-O1", "-o", str(library), str(source)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return library
