# Pilot generalization

Base versus original mixed-arity pilot only; three-number-only transfer training is a separate pending study.

Each suite contains 256 puzzles. ID suites are arity-stratified held-out dataset samples; shifted suites are constructed, solver-verified distributions. Do not pool suites or claim general reasoning. Wilson intervals concern puzzle sampling for one checkpoint, not training-seed variation. Paired tests are exploratory and unadjusted.

| Suite | Base solved | GRPO solved | GRPO accuracy (95% Wilson) | GRPO truncated |
| --- | ---: | ---: | ---: | ---: |
| id_three | 0/256 | [146/256](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/4y4ecf0x) | 57.0% (50.9–62.9) | 91/256 |
| id_four | 0/256 | [41/256](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/fgoiy67k) | 16.0% (12.0–21.0) | 184/256 |
| large_inputs | 0/256 | [61/256](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/0oiyienw) | 23.8% (19.0–29.4) | 165/256 |
| large_targets | 0/256 | [123/256](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/ou2kp5yu) | 48.0% (42.0–54.2) | 122/256 |

Input generation rules and witnesses: [frozen suites](../../../data/suites/v1/README.md). Every output was re-scored, matched to its frozen suite, and checked against the pilot training puzzles before aggregation. See [summary.json](summary.json) for failure counts, hashes, and paired comparisons.
