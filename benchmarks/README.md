# Structural benchmark: butyronitrile C-CN dissociation

This directory bundles the 12-atom butyronitrile geometries from

> Max Rossmannek, Fabijan Pavošević, Angel Rubio, Ivano Tavernelli,
> *Quantum Embedding Method for the Simulation of Strongly Correlated Systems
> on Quantum Computers*, J. Phys. Chem. Lett. **14**, 3491-3497 (2023).
> [doi:10.1021/acs.jpclett.3c00330](https://doi.org/10.1021/acs.jpclett.3c00330),
> [arXiv:2302.03052](https://arxiv.org/abs/2302.03052).

The scan stretches the C-CN bond from 0.8 to 2.5 A.  Each file is plain XYZ
(text, `12` atoms) despite the `.inp` extension.  Atom indices `[0, 1]` are the
N and C of the breaking C-CN bond; the reference protocol uses CAS(4e, 4o),
APC-concentric selection with five frozen occupied orbitals, a PBE low level,
and an HF high level.

The paper's original benchmark used VQE-in-DFT; this repository uses the same
geometries to validate the **classical FCI reference inside the embedding** and
to compare the independent PySCF and SIESTA backends.  FCI is deliberate: it
removes sampling noise from the cross-code comparison.  The paper's evaluation path is already implemented in
`embasi_qiskit_integration.reference_protocol` and exposed as
`--wf_in_dft_energy reference`: a variational SCF-in-SCF for subsystem A, then
the active-space correlation in the canonical concentric virtuals.  That is the
protocol to compare against the paper's `DFT_CAS4`/`DFT_CAS6` reference values;
the first-order APC scans below are implementation diagnostics.  PySCF
`--wf-in-dft-energy reference` reproduces the paper's DFT CAS(4,4) offsets to
<0.024 Ha (most points <1 mHa) and CAS(6,6) to <0.057 Ha.  CAS(4,4) is a
small active space; the separate CAS convergence script exists because the
high-level correction is still moving appreciably at (6,6), (8,8), and (10,10).

## Running the scan

The two backends use their native basis (PySCF: Gaussian/all-electron; SIESTA:
NAO/pseudopotential), so this is a **protocol-level** comparison, not an exact
same-basis one.  The examples use SIESTA `SZ` to keep the QCOMP runs affordable;
the backend default is `DZP`, and its 100-AO ERI transform is supported.
Exact same-pseudopotential checks live in `tests/test_siesta_qcomp_real.py`
and `tests/test_siesta_same_basis.py`.

```bash
# PySCF: PBE low level, HF high level, exact FCI active-space solver.
# CAS(4,4) is kept for protocol reproduction, but the headline comparison uses
# CAS(10,10) because the smaller space is not converged.  `selector none` with
# 2 frozen occupied / 5 virtuals pins a 10e/10o active space at every geometry;
# the APC selector re-selects orbitals along the stretch and creates jumps.
uv run python scripts/benchmarks/run_structural_scan.py \
    --backend pyscf --out-dir analysis/pyscf-pbe-hf-fci \
    --cas-size 10 10 --selector none --n-frozen-occ 2 --n-virtual 5 \
    --solver fci --max-cycles 1 --basis sto-3g \
    --scf-max-cycle 200 --scf-newton

# SIESTA: same protocol in the NAO basis.  A single MPI rank is required by
# SIESTA's ASI interface; export the library and pseudopotential paths first.
export ASI_LIB_PATH=/path/to/libsiesta.dylib
export SIESTA_PP_PATH=/path/to/pseudopotentials
uv run python scripts/benchmarks/run_structural_scan.py \
    --backend siesta --out-dir analysis/siesta-pbe-hf-fci \
    --cas-size 10 10 --selector none --n-frozen-occ 2 --n-virtual 5 \
    --solver fci --max-cycles 1 --siesta-basis SZ --jobs 4

# Paper reference protocol (Rossmannek et al.): variational SCF-in-SCF,
# CAS(4,4) = 5 frozen occupied + 2 concentric virtuals, single-shot.
uv run python scripts/benchmarks/run_structural_scan.py \
    --backend pyscf --out-dir analysis/pyscf-ref-cas4 \
    --selector none --n-frozen-occ 5 --n-virtual 2 --n-shells 1 \
    --wf-in-dft-energy reference --solver fci --max-cycles 1 --basis sto-3g

# Merge the summaries, write CSV/JSON/markdown, and draw the figures.
uv run python scripts/benchmarks/compare_structural_scan.py \
    --pyscf-dir analysis/pyscf-pbe-hf-fci \
    --siesta-dir analysis/siesta-pbe-hf-fci \
    --out-dir analysis/compare-pbe-hf
```

To check active-space convergence, run the same scan with
`--cas-size 6 6`, `--cas-size 8 8`, and `--cas-size 10 10` into separate
directories, then combine them:

```bash
uv run python scripts/benchmarks/plot_cas_convergence.py \
    --pyscf 4=/tmp/scan-pyscf-cas4 --pyscf 6=/tmp/scan-pyscf-cas6 \
    --pyscf 8=/tmp/scan-pyscf-cas8 --pyscf 10=/tmp/scan-pyscf-cas10 \
    --siesta 4=/tmp/scan-siesta-cas4 --siesta 6=/tmp/scan-siesta-cas6 \
    --siesta 8=/tmp/scan-siesta-cas8 --siesta 10=/tmp/scan-siesta-cas10 \
    --distances 1.3 2.5 --out-dir analysis/cas-convergence
```

## Geometry-independent, AO-defined active space (`--selector ao-avas`)

`--selector none --n-frozen-occ ... --n-virtual ...` and `--selector
apc-concentric` choose columns by energy/index (and APC rotates the virtual block
before choosing).  Two runs can therefore report the **same column indices** and
still describe **different active spans**: at R = 0.8 Å the fixed-index and
APC-concentric CAS(10,10) selections both say `active = [2..11]` / `frozen =
[0,1]`, yet their high-level energies differ by ~1.8 Ha.  The occupied orbitals can
also change character along the stretch even at a fixed count.

`--selector ao-avas` makes the selection AO-defined instead: the occupied and
virtual blocks of subsystem A are rotated onto the eigenvectors of a **fixed
fragment-AO projector** and the top `n_active_occ` / `n_active_virt` are kept
(`n_active_occ = nelec/2`, `n_active_virt = norb - n_active_occ`).  The target AOs
are matched in the *working* basis (no MINAO reference), and the counts come from
`--avas-max-size NELEC NORB`; the runner routes `--cas-size NELEC NORB` there for
this selector.  `--n-frozen-occ N` removes the lowest N occupied orbitals from the
ranked candidate pool (the true core); with `--n-frozen-occ 0` the core is whatever
the AO ranking leaves behind, so no energy-ordered occupied orbital is forced into
the active space.  `--n-virtual` is **not** used by `ao-avas`: the virtual count is
`norb - n_active_occ`, so CAS(4,4) is `--cas-size 4 4` and CAS(6,6) is
`--cas-size 6 6`.

PySCF, CAS(4,4) -- two occupied + two virtual orbitals ranked by weight on the
fragment's 2p AOs:

```bash
uv run python scripts/benchmarks/run_structural_scan.py \
    --backend pyscf --out-dir analysis/pyscf-ao-cas4 \
    --selector ao-avas --cas-size 4 4 --n-frozen-occ 0 \
    --avas-ao-labels "N 2p" "C 2p" \
    --solver fci --max-cycles 1 --basis sto-3g \
    --scf-max-cycle 200 --scf-newton
```

PySCF, CAS(6,6) -- three occupied + three virtual; include the 2s labels so the
target space spans the full fragment valence shell (with 2p only, the third virtual
has a much weaker target weight at short R):

```bash
uv run python scripts/benchmarks/run_structural_scan.py \
    --backend pyscf --out-dir analysis/pyscf-ao-cas6 \
    --selector ao-avas --cas-size 6 6 --n-frozen-occ 0 \
    --avas-ao-labels "N 2s" "N 2p" "C 2s" "C 2p" \
    --solver fci --max-cycles 1 --basis sto-3g \
    --scf-max-cycle 200 --scf-newton
```

SIESTA uses the same protocol.  **Omit `--avas-ao-labels`**: SIESTA's ASI basis
descriptors carry only the atom index, so the selector targets every AO of the
active fragment (on a pseudopotential/NAO basis those are valence functions).  The
pseudopotential has no 1s core, so `--n-frozen-occ` counts valence occupied
orbitals; keep it 0 for these CAS sizes.

```bash
export ASI_LIB_PATH=/path/to/libsiesta.dylib
export SIESTA_PP_PATH=/path/to/pseudopotentials
uv run python scripts/benchmarks/run_structural_scan.py \
    --backend siesta --out-dir analysis/siesta-ao-cas4 \
    --selector ao-avas --cas-size 4 4 --n-frozen-occ 0 \
    --solver fci --max-cycles 1 --siesta-basis SZ --jobs 4
# CAS(6,6): replace --cas-size 4 4 with --cas-size 6 6.
```

The fixed-space comparison in the current SIESTA report uses PySCF `def2-SVP`
and SIESTA `DZP` (`PAO.EnergyShift 0.1 eV`, `MeshCutoff 200 Ry`).  Both are single-shot
(`--max-cycles 1`) and keep the same AO-defined CAS(4,4):

```bash
uv run python scripts/benchmarks/run_structural_scan.py \
    --backend pyscf --out-dir analysis/pyscf-ao-cas4-def2svp \
    --geometries 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --selector ao-avas --cas-size 4 4 --n-frozen-occ 0 \
    --avas-ao-labels "N 2p" "C 2p" \
    --solver fci --max-cycles 1 --basis def2-svp \
    --scf-max-cycle 200 --scf-newton --jobs 4 --continue-on-error

uv run python scripts/benchmarks/run_structural_scan.py \
    --backend siesta --out-dir analysis/siesta-ao-cas4-dzp \
    --geometries 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 \
    --selector ao-avas --cas-size 4 4 --n-frozen-occ 0 \
    --siesta-basis DZP --siesta-energy-shift 0.1 --siesta-mesh-cutoff 200 \
    --solver fci --max-cycles 1 --jobs 4 --continue-on-error
```

The SIESTA A-layer is a one-step non-SCF evaluation.  The shim sets
`SCF.Mix density-matrix` and `DM.MixSCF1 False` so `total.log` reports the energy
terms at the imported density; without them SIESTA >= 4.1 logs `Exc` at `DM_out`,
which corrupts the same-density reference.

All 18 points complete; SIESTA selected active indices `[5, 6, 7, 8]` at every
geometry.  Low-level PBE and the full embedding relative-energy curve now track
closely; a residual 1.7-2.3 A hump remains.  The report's dashed basis probe is the
same PySCF calculation with `--basis sto-3g`; it keeps the AO-avas CAS(4,4)
target and isolates the Gaussian basis change.

What this fixes and what it does not:

* The **target AO set, occupied/virtual counts and selection rule** are fixed and do
  not depend on canonical energies.  The selected span is invariant to arbitrary
  rotations of the input occupied/virtual blocks (the APC-vs-`none` ambiguity) and,
  of course, to rotations inside the active span (the FCI is a CAS).
* For a fixed CAS(n,n), the selected span is **not literally identical at every
  geometry**: the fragment-AO character redistributes over the subsystem-A
  occupied/virtual blocks along the stretch, and a fixed-size block-decomposed CAS
  must follow it.  When the target AO space is fully retained by the budget the span
  *is* exactly fixed (H2/STO-3G in `tests/test_ao_avas_selector.py`).  The residual
  geometry dependence is the physical redistribution, not an arbitrary orbital
  rotation.
* A threshold-only cut (`--avas-threshold` without `--avas-max-size`) lets the size
  change between geometries; the fixed budget is the recommended benchmark mode.
* `--wf-in-dft-energy reference` is unaffected: the reference protocol builds its own
  canonical concentric active space from `--n-frozen-occ`/`--n-virtual` and ignores
  `--selector`.  The runner refuses `ao-avas` together with the reference protocol
  rather than silently ignoring the selector.


`scripts/benchmarks/diagnose_embedding_components.py` splits the embedding total
into `E_low_total`, `E_high(A)-E_low(A)` and the correction term, reports
mean-centred cross-code differences and well depths, and can compare several CAS
sizes (`--pyscf-cas 4=... --siesta-cas 10=...`).  Use it before interpreting a
raw correction offset as physics.

The comparison writes `comparison.csv`, `comparison.json`,
`comparison_summary.md`, and PNG/PDF figures (`totals`, `relative_energy`,
`embedding_correction`, `differences`, `cycles`).  The runner creates one
subdirectory per geometry with `run.log`, `diagnostics.csv`, and `energy.csv`;
`summary.csv` is safe to resume from and
`--overwrite` reruns a point.  `--geometries 08 12 25` limits a scan.  SIESTA
jobs write their `EmbASI_calc/` and `siesta_calc/` scratch under the per-geometry
work directory (and those names are git-ignored).  `--max-cycles 1` is a useful
pilot: it fixes the active space from the first density and avoids the
long-bond active-space switching of the iterative APC loop, at the cost of not
testing the density feedback.  The comparison script labels that protocol `S`.
For a cleaner cross-code shape, keep the active space AO-defined across the scan
with `--selector ao-avas` (see above); the APC-concentric default re-selects
orbitals when the bond stretches, which introduces a jump that each basis places
at a slightly different distance.  The older fixed-index control,
``--selector mulliken --n-frozen-occ 5 --n-virtual 2``, fixes the counts but still
lets the selected span rotate with the Fock.

Absolute totals from the two codes are not comparable.  `compare_structural_scan.py`
therefore plots and scores relative energies and the embedding correction,
reporting raw values plus mean-centred differences for the shape comparison.
Its `totals.png` keeps one panel per backend because a shared y-axis would be
dominated by the ~174 Ha all-electron/pseudopotential offset.
