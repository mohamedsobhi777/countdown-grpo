# Frozen generalization inputs, version 1

Frozen on 2026-10-10 before model evaluations on these suites. Each input file
contains 256 unique puzzles; all have exact solver witnesses. `manifest.json`
contains input-file SHA-256 hashes. `witnesses.json` is a separate oracle artifact;
its entries align with input row indices and must never be passed to the model.

- `id_three.json` and `id_four.json`: the first 256 puzzles of each arity from the
  seeded, shuffled original dataset holdout (split seed 42, 2% hash partition).
  They are disjoint from all training examples in that partition, including
  future three-number-only runs. They may overlap the original pilot holdout,
  so these are arity-stratified evaluations, not wholly new independent data.
- `large_inputs.json`: inputs 101–200 and targets 10–100.
- `large_targets.json`: inputs 1–99 and targets 101–200.

Each shifted suite balances 128 three-number and 128 four-number puzzles.
Input numbers are drawn independently and uniformly with replacement. A target
is selected uniformly from the sorted reachable integer targets in the specified
range; input sets without eligible targets are rejected. This is a deliberately
constructed solvable distribution, not an IID sample from the original dataset.
Random seed 20261010 is namespaced by suite name. All four suites have distinct
canonical puzzle keys and are checked against the pilot training subset.

Regeneration uses `scripts/freeze_suites.py`, which refuses to overwrite a frozen
directory. Make a new version to change the protocol. The exact subset-DP oracle
enumerates arithmetic expression trees with binary +, -, *, / and all inputs
used once; fractional and negative intermediate values are allowed.
