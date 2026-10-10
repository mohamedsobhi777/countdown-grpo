# Interim transfer-study evidence

Seed 42 completed 300 steps of three-number-only training in 98.7 minutes and all five evaluations. **Seeds 43 and 44 are still pending.** This is one checkpoint, not the completed multi-seed study.

| Evaluation | Solved | Accuracy (95% Wilson) | Truncated |
| --- | ---: | ---: | ---: |
| [Original holdout](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/bpz8xxao) | 79/256 | 30.9% (25.5–36.8) | 167/256 |
| [Three-number holdout](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/64lmacnd) | 146/256 | 57.0% (50.9–62.9) | 110/256 |
| [Four-number holdout](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/bi91r7xw) | 26/256 | 10.2% (7.0–14.5) | 211/256 |
| [Larger inputs (constructed)](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/wgqlwhq9) | 63/256 | 24.6% (19.7–30.2) | 188/256 |
| [Larger targets (constructed)](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/0j118djm) | 107/256 | 41.8% (35.9–47.9) | 145/256 |

Strict and conservative diagnostic scores agree in all five evaluations. Each file passed configuration, ordered-puzzle, suite-hash (where applicable), training-overlap, and per-answer score checks. Training used 20,000 unique three-number puzzles and completed all 300 steps.

All five evaluations for seed 42 are complete and audited. Seeds 43 and 44 remain pending. Comparisons against the single mixed-arity pilot are descriptive, not a balanced multi-seed training-distribution ablation. Constructed shifts are not difficulty-matched. No pooled observations or seed-averaged conclusion is reported.

[Full summary, uncertainty, failure categories, hashes, and descriptive paired comparisons](seed-42-summary.json). Raw completions are in the five `seed-42-*.json` evaluation files. Exported training configuration, metrics, requirements, and W&B provenance are in `seed-42-training/`. The initial dirty-checkout caveat is documented in [the source audit](../local-source-audit.json).

[Training](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/liev6g5l) · [Final adapter](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/liev6g5l-adapter/v0).
