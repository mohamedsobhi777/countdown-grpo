# Output-budget experiment

Wilson intervals describe puzzle-sampling uncertainty for a fixed checkpoint; they do not measure training-seed variation. Exact paired p-values are exploratory and unadjusted for multiple comparisons.

The 256-token files predate optional split_seed/train_num_count fields. Missing values are normalized to None (original seed, both arities); all other settings and canonical puzzle sets must match.

| Model / cap | Solved | Accuracy (95% Wilson) | Truncated | Mean tokens | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| [base-256](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/9bzxufft) | 0/256 | 0.0% (0.0–1.5) | 8 | 39.7 | 43.5 |
| [adapter-256](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/jn45a3oi) | 85/256 | 33.2% (27.7–39.2) | 145 | 175.8 | 184.6 |
| [base-512](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/m989nxpc) | 0/256 | 0.0% (0.0–1.5) | 8 | 47.7 | 73.2 |
| [adapter-512](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/vnvj9eez) | 86/256 | 33.6% (28.1–39.6) | 140 | 316.9 | 358.6 |
| [base-1024](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/5guwtvht) | 0/256 | 0.0% (0.0–1.5) | 8 | 63.7 | 117.4 |
| [adapter-1024](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/haj4kbrh) | 86/256 | 33.6% (28.1–39.6) | 139 | 595.8 | 710.0 |

All cells use the same puzzles and greedy decoding. Timings cover batched generation and scoring, excluding model loading; they are not single-request latency.

Raw outputs, canonical-puzzle paired gains/losses, unadjusted exact McNemar tests, and source hashes are included in [summary.json](summary.json).

Wilson method: [NIST statistical reference](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
