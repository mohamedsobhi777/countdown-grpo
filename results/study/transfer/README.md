# Three-number-only training and transfer

Each model trained on 20,000 three-number puzzles for 300 steps with a fixed split. Per-seed comparisons against the single mixed-arity pilot are descriptive, not a balanced multi-seed training-distribution ablation. Three seeds share the same test puzzles; do not pool observations as independent. Pretraining exposure and format transfer remain possible. Shifted suites are constructed, not difficulty-matched.

| Evaluation | Seed 42 | Seed 43 | Seed 44 | Mean ± seed SD |
| --- | ---: | ---: | ---: | ---: |
| original_holdout | 79/256 | 93/256 | 77/256 | 32.4% ± 3.4 pp |
| id_three | 146/256 | 161/256 | 133/256 | 57.3% ± 5.5 pp |
| id_four | 26/256 | 38/256 | 23/256 | 11.3% ± 3.1 pp |
| large_inputs | 63/256 | 79/256 | 69/256 | 27.5% ± 3.2 pp |
| large_targets | 107/256 | 112/256 | 108/256 | 42.6% ± 1.0 pp |

Per-checkpoint Wilson intervals, diagnostic scores, failure counts, links, and paired comparisons are in [summary.json](summary.json).

![Seed comparison](transfer-seed-comparison.png)

[W&B analysis](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/0svwuwkr).
Finished training provenance and package requirements accompany each run.
Source audits: [seed 42](../local-source-audit.json),
[seed 43](../local-source-audit-seed43.json),
[seed 44](../local-source-audit-seed44.json). Seed 42's historical dirty-tree
flag remains documented. The figure separates checkpoint uncertainty from
across-seed variation; its metadata records the audited summary hash.
