# Approved laptop recovery of the controlled reward study

The user chose a complete laptop rerun after Colab L4 access was rejected.
This implements the six-run alternative described in the conversation. The
already-complete budget experiments, three-seed transfer study, and live demo
were run locally and remain valid; their outputs are retained.

## Fixed design

- New group: `controlled-local-v2`. Six fresh runs: binary/shaped reward for
  seeds 42, 43, and 44, in that order within each seed.
- Same pinned model/dataset, split seed 42, 20,000 mixed-arity training puzzles,
  300 optimizer steps, 32 completions per step, 256-token cap, and LoRA settings
  as the original controlled design. Each run gets its own 256-puzzle evaluation.
- One NVIDIA RTX 4090 Laptop GPU, existing Python 3.12 environment, reference
  kernels. No dependency synchronization or kernel installation during the queue.
- One clean detached checkout for all six runs. Keep documentation updates in
  the main checkout so they cannot change later runs' recorded training source.
- W&B source/configuration/metrics/completions/checkpoints for every run; save
  resumable checkpoints every 50 steps. Capture finished W&B runtime provenance
  and package requirements after each job and validate its raw scores.
- Aggregate only a complete six-run local grid. Require identical source,
  Python/packages, hardware, ordered train/holdout lists, and all 300 step records.
  Report per-seed results, mean/sample SD, Wilson intervals, and paired outcomes.
- Preserve Colab completed and interrupted runs as separate historical evidence.
  Never combine them with the local grid. No selection by observed reward or score.

## Execution and verification

`PYTHONPATH=src .venv/bin/python -u scripts/run_local_controlled.py` starts a new
queue and refuses an existing output group. Run it in persistent tmux. It stops
on training/evaluation/audit errors; API retrieval retries never restart training.
Check `outputs/controlled-local-v2/queue.json`, each job's status and metrics,
the actual process, and W&B. Inspect an interrupted job before explicitly resuming.

After all jobs finish, the queue runs the aggregate audit with `--profile local`
and writes `results/study/controlled-local`. Generate and review the seed plot,
publish results and figures, update Overleaf/W&B, and complete the final audit.
Tests cover accepted local provenance, rejection of Colab/local mixing, kernel
mismatch, incomplete training, and corrupted scores. A fresh full run verifies
actual training rather than relying on synthetic fixtures.
