# DMRG observables from block2 output

Two steps: contract the raw block2 2-RDMs into a small bundle, then compute the
observables (energies, permanent dipoles, Mulliken fragment charges / CT).

Requires `numpy` for step 1, and `numpy` + `pyscf` for step 2.

```bash
RUN=<path to dmrg folder containing dmrg_rundir and dmrg_scratch>

# Step 1: contract the 2-RDMs into a small bundle
process_block2_dmrg_output_for_observables.py --run-root-folder "$RUN" --out-prefix "$RUN/prepared"

# Step 2: observables
compute_observables.py \
    --rdm-npz  "$RUN/prepared_rdm1.npz" \
    --energies "$RUN/prepared_states.npz" \
    --n-elec 50 --n-orbs 50 \
    --geom  "$RUN/system1.xyz" \
    --basis def2-tzvp \
    --mo    "$RUN/win_mp2_natorbs_def2-TZVP_win_mp2.npz" \
    --out-prefix "$RUN/obs"
```

`$RUN` is the folder holding `dmrg_scratch/` and `dmrg_rundir/` as siblings.
Step 1 discovers the RDMs, FCIDUMP, energies and the geometry/MO/basis inputs
from it, and prints the step 2 command with every path filled in — so the values
above are what it emits, not defaults you need to look up. Add `--check` to
step 1 to report what it found without writing anything.

Notes:

- `--basis` is the one value not read from the DMRG output; step 1 guesses it
  from the folder name, so verify it against your run.
- `--n-roots` is not needed: it is inferred from the bundle.
- `--n-elec` / `--n-orbs` are optional too — both come from the FCIDUMP header.

See each script's `--help` for the full flag list.

