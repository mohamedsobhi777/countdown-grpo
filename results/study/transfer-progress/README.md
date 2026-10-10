# Interim transfer-study evidence

Seed 42 completed 300 steps of three-number-only training in 98.7 minutes and all five evaluations. **Seeds 43 and 44 are still pending.** This is one checkpoint, not the completed multi-seed study.

Seed 43 has now completed training in **91.1 minutes** and its original-holdout
evaluation: **93/256 solved (36.3%; 95% Wilson 30.7–42.4%)**, with strict and
diagnostic scores agreeing and 157/256 truncated outputs. The full step sequence,
configuration, identical training/holdout order versus seed 42, absence of overlap,
and every answer's scores passed audit. Its four frozen-suite evaluations and
seed 44 remain pending. [Evaluation run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/3jlajmcr),
[raw completions](seed-43-evaluation.json), [audited summary](seed-43-holdout.json),
[source audit](../local-source-audit-seed43.json).

The table below covers **seed 42 only**.

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

## Output-budget limitation

All 110 three-number failures and 211 of 230 four-number failures hit the
256-token cap. Every four-number `no_declared_expression` outcome (198) is also
truncated. These are overlapping observations, not evidence that truncation
caused the errors or that longer outputs would solve them. Transfer scores are
conditional on this budget. The completed longer-output experiment used the
original mixed-arity pilot and cannot establish the budget effect for this
separately trained adapter. The summary includes the full solve/truncation
cross-tabulation, including one correct but truncated larger-input answer.
