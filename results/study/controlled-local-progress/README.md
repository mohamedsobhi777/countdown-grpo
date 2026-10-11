# Laptop controlled reward study: interim evidence

Binary seed 42 is complete in `controlled-local-v2`. **Five runs remain**;
this single checkpoint is not the final reward comparison or a seed average.
Colab results are archived separately and are never pooled with this grid.

The checkpoint solves **80/256 (31.25%)** held-out puzzles under greedy decoding
with a 256-token cap. Its 95% Wilson interval is **25.9–37.2%**. Strict and
diagnostic scores agree. There are 162 truncated outputs; truncation overlaps
the outcome categories and is not established as their cause.

Diagnostic outcomes: 80 correct, 136 without a declared expression, 31 number
mismatches, five wrong values, and four invalid expressions. Training completed
all 300 steps in 5368.2092 seconds (89.5 minutes), with the first positive reward
at step 31 and mean reward 0.265625 over the final 30 steps.

The audit verified the prescribed configuration, all 300 step records, 20,000
unique training puzzles, train/holdout separation, ordered evaluation puzzles,
all raw scores, clean frozen source, laptop hardware, reference kernels, and
archived package requirements. Matching across all six runs remains pending.

- [Audited summary and source hashes](binary-42/summary.json)
- [Raw evaluation and completions](binary-42/evaluation.json)
- [Training metrics](binary-42/training/metrics.jsonl)
- [Training run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/eudebo7g)
- [Evaluation run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/17gdgf7i)
- [Final adapter](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/eudebo7g-adapter/v0)
- [Resumable step-300 checkpoint](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/model-binary-42-train/v13)

The checkpoint artifact version is tied to the training run above; collection
names are shared with historical runs, so use this explicit version.
