# Controlled reward study: interim evidence

Only the binary-reward seed-42 run is complete. The remaining five runs are
pending; this is **not** a reward comparison or a seed average.

| Metric | Binary reward, seed 42 |
| --- | ---: |
| Strict exact solutions | 68/256 (26.6%) |
| 95% Wilson interval | 21.5–32.3% |
| Diagnostic exact solutions | 68/256 |
| Truncated outputs | 174/256 |
| Training | 300 steps, 192.9 minutes |
| First positive training reward | Step 28 |

The downloaded archive passed SHA-256 verification. All 300 steps, configuration,
ordered evaluation puzzles, training/holdout separation, unique puzzle keys,
per-answer scores, aggregate accuracy, single-L4 hardware, pinned FLA packages,
and clean source provenance passed the controlled-run audit. Consistency across
all six runs remains to be checked after completion.

Diagnostic outcomes are 68 correct, 141 invalid syntax, 28 number mismatches,
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
