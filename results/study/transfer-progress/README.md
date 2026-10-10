# Interim transfer-study evidence

Seeds 42 and 43 have completed 300 steps of three-number-only training and all five evaluations. **Seed 44 is training.** These are interim checkpoint results, not the completed multi-seed study.

Seed 43 trained in **91.1 minutes**. Its full step sequence, configuration,
identical training/holdout order versus seed 42, absence of training overlap,
frozen-suite hashes, and every answer's scores passed audit. Strict and diagnostic
scores agree on all five evaluations.

| Seed 43 evaluation | Solved | Accuracy (95% Wilson) | Truncated |
| --- | ---: | ---: | ---: |
| [original holdout](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/3jlajmcr) | 93/256 | 36.3% (30.7–42.4) | 157/256 |
| [id three](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/hwra03jw) | 161/256 | 62.9% (56.8–68.6) | 95/256 |
| [id four](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/iczg0p5h) | 38/256 | 14.8% (11.0–19.7) | 197/256 |
| [large inputs](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/ldpv43a3) | 79/256 | 30.9% (25.5–36.8) | 165/256 |
| [large targets](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/hakxhfxf) | 112/256 | 43.8% (37.8–49.9) | 140/256 |

The larger-input and larger-target suites are constructed and not difficulty-matched.
[Seed 43 summary and paired comparisons](seed-43-summary.json) ·
[source audit](../local-source-audit-seed43.json) ·
[training run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/3ifuj2rg) ·
[final adapter](https://wandb.ai/mohamedsobhi777/countdown-grpo/artifacts/model/3ifuj2rg-adapter/v0).
Raw completions are in the five `seed-43-*.json` evaluation files. All 95
three-number failures and 197 of 218 four-number failures are truncated;
this overlap does not establish causation or the benefit of a larger token budget.

The table below covers **seed 42 only**.

| Evaluation | Solved | Accuracy (95% Wilson) | Truncated |
| --- | ---: | ---: | ---: |
| [Original holdout](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/bpz8xxao) | 79/256 | 30.9% (25.5–36.8) | 167/256 |
| [Three-number holdout](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/64lmacnd) | 146/256 | 57.0% (50.9–62.9) | 110/256 |
| [Four-number holdout](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/bi91r7xw) | 26/256 | 10.2% (7.0–14.5) | 211/256 |
| [Larger inputs (constructed)](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/wgqlwhq9) | 63/256 | 24.6% (19.7–30.2) | 188/256 |
| [Larger targets (constructed)](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/0j118djm) | 107/256 | 41.8% (35.9–47.9) | 145/256 |

Strict and conservative diagnostic scores agree in all five evaluations. Each file passed configuration, ordered-puzzle, suite-hash (where applicable), training-overlap, and per-answer score checks. Training used 20,000 unique three-number puzzles and completed all 300 steps.

All five evaluations for seeds 42 and 43 are complete and audited. Seed 44 remains pending. Comparisons against the single mixed-arity pilot are descriptive, not a balanced multi-seed training-distribution ablation. Constructed shifts are not difficulty-matched. No pooled observations or seed-averaged conclusion is reported.

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

[Seed 43 audit run and evidence artifact](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/pt489axi).
