# Controlled reward study: interim evidence

The binary and shaped seed-42 runs are complete. **Four runs remain pending**;
this is one matched seed, not a seed average or the final reward comparison.

| Metric | Binary seed 42 | Shaped seed 42 |
| --- | ---: | ---: |
| Strict exact solutions | 68/256 (26.6%) | 26/256 (10.2%) |
| 95% Wilson interval | 21.5–32.3% | 7.0–14.5% |
| Diagnostic exact solutions | 68/256 | 26/256 |
| Truncated outputs | 174/256 | 0/256 |
| Training | 300 steps, 192.9 minutes | 300 steps, 117.5 minutes |

The shaped checkpoint produces valid expressions using all supplied numbers on
all 256 puzzles, but 230 evaluate to the wrong target. Relative to binary reward,
it gains 11 solutions and loses 53 (15 jointly correct): **−16.4 percentage points**.
This seed shows improved expression validity alongside lower correctness. It
does not establish that reward shaping generally hurts: the other seeds remain
pending. The first positive shaped reward can reflect the 0.05 validity bonus;
raw training reward is not directly comparable across reward definitions.

Both runs have identical ordered training/holdout puzzles, clean source commit,
Python, complete normalized requirements, optional kernels, and hardware metadata.
[Paired counts and matching provenance](seed-42-comparison.json).

The downloaded archive passed SHA-256 verification. All 300 steps, configuration,
ordered evaluation puzzles, training/holdout separation, unique puzzle keys,
per-answer scores, aggregate accuracy, single-L4 hardware, pinned FLA packages,
and clean source provenance passed the controlled-run audit. Consistency across
all six runs remains to be checked after completion.

Binary diagnostic outcomes are 68 correct, 141 invalid syntax, 28 number mismatches,
10 without a declared expression, eight wrong values, and one ambiguous or
incomplete answer. Truncation is an overlapping flag, not another exclusive
outcome. These categories do not establish why the model failed.

- [Training and learning curves](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/fneh9kxb)
- [Evaluation and completions](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/9xwjji2l)
- [Final adapter](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/fneh9kxb-adapter/v0)
- [Audited summary, intervals, provenance, and source hashes](binary-42/summary.json)
- [Raw evaluation](binary-42/evaluation.json)
- [Training metrics](binary-42/training/metrics.jsonl)
- [W&B audited evidence artifact run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/zwu499xh)

The original laptop/fallback pilot scored 85/256. Its runtime environment differs
from this L4/FLA study; this comparison cannot isolate an effect of hardware or
kernels. The six-run study uses one fixed environment for its reward comparison.

## Shaped seed-42 evidence

- [Training](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/skfvvhnz)
- [Evaluation](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/83tmr7i5)
- [Final adapter](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/skfvvhnz-adapter/v0)
- [Audited summary and source hashes](shaped-42/summary.json)
- [Raw evaluation](shaped-42/evaluation.json)

For example, the first frozen puzzle supplies `[2, 4, 55, 25]` with target 81.
The shaped output claims `2 + 4 - 55 + 25 = 81` and declares that expression
in its answer block. The exact verifier computes −24 and rejects it as wrong.
This example illustrates the recorded failure category; it is not a separate
estimate of its prevalence.

[Audited seed-42 reward-pair evidence artifact](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/h8l3bwko).

## Binary seed 43 and runtime interruption

[Binary seed 43](binary-43/summary.json) passed the complete audit at
**64/256 (25.0%; Wilson 95% 20.1–30.6%)**, with strict and diagnostic agreement
and 180 truncated outputs. [Evaluation](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/pz78iffl).
Three controlled runs are complete. The next shaped run was interrupted;
[recovery evidence](../runtime-interruption/README.md) records its saved checkpoint
and the rejected replacement L4 request. The six-run comparison remains incomplete.
