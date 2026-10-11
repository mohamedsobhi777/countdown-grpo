# Laptop controlled reward study: interim evidence

Binary seeds 42 and 43 and shaped seed 42 are complete in `controlled-local-v2`.
**Three runs remain**; the final multi-seed reward comparison is pending.
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

## Matched seed-42 comparison

| Metric | Binary reward | Shaped reward |
| --- | ---: | ---: |
| Strict and diagnostic correct | 80/256 (31.25%) | 1/256 (0.39%) |
| 95% Wilson interval | 25.9–37.2% | 0.07–2.18% |
| Valid expressions | 85/256 | 256/256 |
| Truncated outputs | 162/256 | 0/256 |
| Training time, 300 steps | 89.5 minutes | 53.4 minutes |

The shaped checkpoint produces valid expressions using all inputs on every
puzzle, but 255 evaluate to the wrong target. It gains one solution and loses
80 relative to binary reward; none are jointly correct. The difference is
**−30.86 percentage points** for this seed. The exact paired McNemar p-value
is 6.78e−23 (exploratory, unadjusted); it does not quantify training-seed
uncertainty. Four runs remain before the planned multi-seed conclusion.

Both runs passed the full audit and have identical ordered training/holdout
puzzles, clean frozen source, Python/packages, optional kernels, and hardware.
The shaped first positive reward at step 22 can reflect its 0.05 validity
bonus; raw training rewards are not directly comparable between objectives.

- [Paired counts and matching provenance](seed-42-comparison.json)
- [W&B matched-pair evidence archive](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/2lymwle8)
- [Shaped audited summary and source hashes](shaped-42/summary.json)
- [Shaped raw evaluation](shaped-42/evaluation.json)
- [Shaped training](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/ofj5pru1)
- [Shaped evaluation](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/8694c6tk)
- [Shaped adapter](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/ofj5pru1-adapter/v0)
- [Shaped resumable checkpoint](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/model-shaped-42-train/v13)

## Binary seed 43 completed

Binary seed 43 solves **13/256 (5.08%; Wilson 95% 2.99–8.49%)** after
300 steps in 85.5 minutes. The conservative diagnostic metric recovers three
additional answers (**16/256, 6.25%**); all three are truncated outputs whose
last standalone arithmetic line is correct. Their raw text and extracted
expressions are preserved in [diagnostic recoveries](binary-43/diagnostic-recoveries.json).
There are 219 truncated outputs. Diagnostic categories are 16 correct,
38 wrong values, 66 number mismatches, 99 invalid expressions, and 37 without
a declared expression. Truncation overlaps these categories.

The full run audit passed, and ordered training/holdout puzzles, source,
environment, and hardware match binary seed 42. The difference from seed 42
shows substantial variation between these two training seeds; the final
three-seed statistics and reward comparison remain pending. Shaped seed 43
is now training.

- [Audited summary](binary-43/summary.json)
- [Raw evaluation](binary-43/evaluation.json)
- [Training](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/d9it45u2)
- [Evaluation](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/dmxcq1wh)
- [Adapter](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/d9it45u2-adapter/v0)
- [Resumable checkpoint](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/model-binary-43-train/v13)
