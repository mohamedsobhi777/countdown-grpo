# Experiment: countdown-20260914-01

Completed 300 optimizer steps of direct GRPO on Qwen3.5-0.8B-Base with BF16 LoRA,
binary correctness reward, and no SFT. Seed: 42. Hardware: RTX 4090 Laptop GPU,
16 GB VRAM. Training runtime: 5,135.9127 seconds (85.6 minutes), excluding setup,
evaluation, and final artifact synchronization.

| Evaluation | Correct | Valid expressions | Decoding |
| --- | ---: | ---: | --- |
| Base checkpoint | 0 / 256 (0.0%) | 11 / 256 (4.3%) | Greedy, max 256 new tokens |
| Final LoRA adapter | 85 / 256 (33.2%) | 90 / 256 (35.2%) | Greedy, max 256 new tokens |

Both evaluations use the same 256 unique canonical puzzles. None overlap with
the selected 20,000 training puzzles. The latter is the available training subset,
not the number of distinct puzzles consumed in 300 steps: each step samples four
puzzles with eight candidate completions each (9,600 candidate completions total).

## Files

- [summary.json](summary.json): machine-readable aggregate measurements and run URLs.
- [baseline.json](baseline.json): original baseline report, configuration, and all completions.
- [after.json](after.json): original final report, configuration, and all completions.
- [metrics.jsonl](metrics.jsonl): 300 per-step training records plus the final runtime record.
- [learning-curve.png](learning-curve.png) ([SVG](learning-curve.svg)): per-step training reward, its trailing
  20-step mean, and the two held-out accuracy measurements.

Training reward averages stochastic samples on training puzzles. Held-out accuracy
uses one greedy answer per puzzle. These quantities are different; the chart does
not imply validation was measured at each training step. No intermediate held-out
evaluation or checkpoint selection was performed.

## Provenance and artifacts

- Source commit: [`0a432a272591f7683b711a9c867c01e7d939fca1`](https://github.com/mohamedsobhi777/countdown-grpo/tree/0a432a272591f7683b711a9c867c01e7d939fca1), clean working tree.
- Base model revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Dataset revision: `408f70d177020686d34a56bba5952feb45aaaee4`.
- [Baseline W&B run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/qu1ovu2z).
- [Training W&B run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/uv3hjjih).
- [Final evaluation W&B run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/1qrixzrf).
- [Final adapter artifact, version 0](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/uv3hjjih-adapter/v0).
- [Training artifact browser](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/uv3hjjih/artifacts): exact puzzle splits, source, model/checkpoint artifacts, and results.

The LoRA artifact requires the pinned base model; it is not a standalone checkpoint.
Paths under `outputs/` inside the JSON reports identify files on the original
training machine. The reports are retained unchanged; those paths are not required
to inspect them here.

## Validation and limits

The report exporter re-evaluated every saved completion with the repository's
strict verifier, checked all aggregate scores, confirmed the same unique puzzles
in both evaluations, and verified no intersection with the training puzzle set.
The first positive training reward appeared at step 31. Peak PyTorch-allocated
memory was 2.72 GiB; this is not total device memory usage.

One seed and 256 puzzles limit the strength of the conclusion. The score requires
both a correct expression and the required answer block. Roughly 59.9% of sampled
completions in the last 30 training steps reached the 256-token cap. General
reasoning, cross-seed stability, and performance with longer outputs were not tested.

To regenerate the checked archive and chart from the original local outputs:

```bash
PYTHONPATH=src uv run --no-project --with matplotlib python scripts/report_results.py \
  outputs/countdown-20260914-01 results/2026-09-14
```

The exporter is specific to this 300-step experiment and embeds its W&B run URLs.
