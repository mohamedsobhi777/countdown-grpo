# Study ledger

This ledger separates completed evidence from queued experiments. Full scope and
completion gates: [research-plan.md](research-plan.md).

## Completed, 2026-10-10

- Archived-output audit: strict and declared-expression scores both give 0/256
  for the base and 85/256 for the adapter. [W&B](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/jyyk55pq), [raw audit](../results/study/audit/summary.json).
- Exact oracle proves all 256 pilot holdout puzzles solvable. Deterministic
  inspection of 20 outputs recorded. [W&B oracle/review](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/7g22jiic), [review](../results/study/audit/review.md).
- Four generalization suites (256 examples each), frozen before model evaluation;
  input hashes and separate witnesses: [data/suites/v1](../data/suites/v1/README.md).
- Overleaf report created and compiled: [editor](https://www.overleaf.com/project/6aca3d97baaac8402a980db8), [LaTeX](../report/main.tex), [PDF](../report/main.pdf).
- 36 CPU tests pass, covering solver witnesses, safety, duplicate handling,
  fixed-split independence from training seed, benchmark determinism, Wilson
  interval reference values, and canonical-puzzle pairing validation.
- Budget analysis implemented in `scripts/analyze_budget_study.py`: requires all
  six cells, re-verifies scores/configs, archives raw outputs, and reports Wilson
  intervals plus paired gains/losses and exploratory unadjusted exact tests.
- Controlled-study analysis implemented in `scripts/analyze_controlled_study.py`:
  requires all six completed runs; checks exact settings, all 300 optimizer steps,
  20,000 training puzzles, identical ordered splits, no duplicate/overlap, and
  recomputed scores. Reports each seed and sample standard deviations separately
  from puzzle-sampling uncertainty. Synthetic evidence tests exercise rejection of
  incomplete steps/runs and altered scores; total CPU coverage is 41 tests.
- Local comparison demo implemented: all 256 archived pairs, outcome filters,
  conservative diagnostic feedback, custom exact-answer checker, optional live
  base/adapter generation. HTTP tests cover archival parity, validation, and
  disabled inference. Total CPU test count is now 40. Chromium smoke checks pass:
  all 256 examples, 85 successes/171 failures, exact-answer checking, disabled
  inference in archive mode, no JavaScript errors, and no horizontal overflow at
  390px width. Reproduce with `scripts/check_demo_browser.py`; screenshots are
  saved under `outputs/demo-{desktop,mobile}.png`. Live CUDA validation is pending.
  [Demo source, verification status, and screenshots in W&B](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/fbfflcpn).

## Running / queued

- Local tmux `countdown-budget-study`: six model/budget combinations at
  `outputs/study-budgets-20261010/`. Both 256-token evaluations reproduce the
  pilot scores; base at 512 also scores 0/256. Adapter at 512 solves 86/256:
  one gain and no losses versus 256 tokens, with 140/256 truncated outputs.
  [512-token adapter run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/vnvj9eez).
  The 1024-token cells remain pending.
- Colab session `countdown-study`, L4: six runs in `controlled-l4-v1`, binary/shaped
  rewards interleaved at seeds 42, 43, 44. Binary seed 42 is rerun so the six main
  comparisons share hardware. [First run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/4anzz6c2).
  Observe with `colab exec -s countdown-study -f scripts/colab_status.py`.
  First five steps averaged 50.5 seconds each; monitor this slower L4 runtime.
- Local tmux `countdown-generalization`: waits for budget evaluation, then runs
  base/pilot on all frozen suites, followed by three-number-only training at
  seeds 42, 43, 44 with original-holdout and four-suite evaluations. Outputs:
  `outputs/generalization-pilot-v1/` and `outputs/three-only-local-v1/`.
- Live-demo smoke test is queued after the two local GPU queues finish, in tmux
  `countdown-demo-live-check`. It performs actual base/adapter generation twice
  and verifies repeatability, then logs the output to W&B. This remains pending
  until `results/study/demo-live.json` exists with a successful recorded result.

Do not restart based only on old status files: inspect live tmux or remote PID.
Each queue stops its study on failure. The independent local three-number study
can still start if pilot generalization fails, so inspect both logs. Stop Colab
when outputs are secured; idle compute should not be retained.

## Remaining

Aggregate complete budget results with uncertainty and paired comparisons;
finish seed/reward and transfer runs; archive remote results; log analysis
artifacts; finish browser/live validation of the interactive demo; complete and compile the
Overleaf report; publish all evidence and audit every requirement. The study is
not yet complete.
