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
- Pilot generalization aggregation implemented in `scripts/analyze_generalization.py`:
  requires all eight cells, verifies frozen suite hashes/order and pilot training
  separation, and recomputes scores. It explicitly excludes the still-pending
  three-number-only training study. Tests reject changed hashes/order/settings;
  total CPU suite now has 44 passing tests.
- Transfer-study aggregation implemented in `scripts/analyze_transfer_study.py`:
  requires all three complete training runs and all 15 evaluations, enforces
  three-number-only training, frozen suites, identical split/order, all 300 steps,
  and per-answer score recomputation. A synthetic end-to-end test verifies the
  successful archive and rejects four-number training contamination before any
  result is published. The complete CPU suite now has **46 passing tests**.
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

- **Budget grid completed:** base 0/256 at all caps; adapter 85, 86, 86 at
  256/512/1024. The same one puzzle is gained by both longer caps; evaluation
  time rises from 184.6 to 710.0 seconds. All six files passed score and pairing
  checks. Two newly optional config fields required explicit historical-default
  normalization (documented in the report and tested). CPU suite: 42 passing.
  [Results](../results/study/budgets/README.md),
  [W&B analysis](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/ul7kozpl).
- Colab session `countdown-study`, L4: six runs in **`controlled-l4-fla-v1`**,
  binary/shaped rewards interleaved at seeds 42, 43, 44, all with the pinned FLA
  0.5.2 backend. The fresh queue is verified live (PID 14915); its first step took
  about 39 seconds. [First FLA run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/fneh9kxb).
  The prior fallback run (`4anzz6c2`) was stopped after 41 steps
  and archived as an excluded runtime trial.
  Observe with `colab exec -s countdown-study -f scripts/colab_status.py`.
  [Kernel investigation](../results/study/kernel-probe/README.md): numerical
  microchecks passed after correcting a probe-only positional-argument bug.
  The five-step full-training validation completed; warm steps averaged 37.63s
  versus about 50.8s for fallback. The original run's 266.62-second suspension
  and concurrent microprobe are explicitly recorded in its archived evidence.
- Colab A100 allocated as `countdown-fast-study`, group `controlled-a100-v1`;
  [first run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/i7dgueld).
  Intentionally stopped after three steps at 49.33 seconds/step versus 50.79 on
  L4 (about 3% faster, insufficient to justify switching this grid). Complete
  trial logs/manifests were downloaded and [archived in W&B](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/rat7r34m).
  **A100 VM termination confirmed.** The L4 FLA grid is the chosen comparison;
  the A100 trial is tagged and excluded from completed training results.
- **Pilot generalization completed and audited:** base/pilot on four frozen
  suites; pilot solves 146/256 three-number, 41/256 four-number, 61/256 larger-input,
  and 123/256 larger-target puzzles. [Results](../results/study/generalization-pilot/README.md),
  [W&B analysis](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/p33k249f).
- Local tmux `countdown-generalization`: now runs three-number-only training at
  seeds 42, 43, 44 with original-holdout and four-suite evaluations. Outputs:
  `outputs/generalization-pilot-v1/` and `outputs/three-only-local-v1/`.
- Live-demo smoke test is queued after the two local GPU queues finish, in tmux
  `countdown-demo-live-check`. It performs actual base/adapter generation twice through HTTP, then generates
  again through the browser button and checks rendered outputs, verifier details,
  repeatability, mobile layout, and JavaScript errors. Outputs and screenshots
  are logged to W&B. The extended browser path passed a synthetic transport check;
  that check did not run a model and is not evidence of CUDA success. This remains pending
  until `results/study/demo-live.json` exists with a successful recorded result.
- Transfer aggregation is queued in tmux `countdown-transfer-analysis` after
  `countdown-generalization` exits. Incomplete/failed studies are rejected; it
  never silently publishes a partial seed grid. Log: `outputs/launch/transfer-analysis.log`.

Do not restart based only on old status files: inspect live tmux or remote PID.
Each queue stops its study on failure. The independent local three-number study
can still start if pilot generalization fails, so inspect both logs. Stop Colab
when outputs are secured; idle compute should not be retained.

## Generalization figure

The completed pilot suite results now include a reproducible PNG/SVG/PDF figure
of strict accuracy (95% Wilson intervals) and diagnostic outcomes. The plot
records its source hash, treats truncation as an overlapping flag, and labels
shifted suites as constructed rather than difficulty-matched. It is included
in the repository README and Overleaf report. Controlled-run aggregation also
exports per-seed diagnostic category counts for the final failure analysis.

## Automatic collection and provenance checks

`scripts/collect_colab_study.py` collects each completed Colab job with a SHA-256
verified archive, retaining final adapters, manifests, metrics, and raw evaluations.
It exports W&B source, hardware, Python, package, and pinned-kernel metadata.
The controlled analysis rejects inconsistent environments and recomputes scores.
All 48 CPU tests pass, including rejection of corrupt and unfinished archives.
A live observation confirmed the existing queue; collection never restarts it.

The `countdown-colab-collector` tmux session polls every five minutes. With
`--release-when-complete`, it releases the dedicated Colab VM only after all six
jobs are complete, downloaded, and individually validated, and the remote queue
has stopped. It then runs the aggregate analysis with W&B logging locally.
Observation failures retain the VM and retry; training failures require inspection.
Log: `outputs/launch/colab-collector.log`.

## First transfer run: source provenance

The first local run (`liev6g5l`) reports `git_dirty=true` at commit `a2a8e9e`.
Its W&B source artifact `source-countdown-grpo-None:v15` was downloaded and
compared byte-for-byte with that commit: all 108 archived tracked files match.
Untracked files are excluded from that artifact, so the dirty flag is retained.
All package modules match comparison commit `34faa55` except `tracking.py`, whose
change adds Python/optional-kernel metadata only. No training or evaluation
module changed. The first run predates those metadata fields; its startup log
explicitly identifies the PyTorch fallback kernels. The later seeds still need
comparison after completion. [Hashes and audit scope](../results/study/local-source-audit.json).

## 2026-10-11: first transfer training and original-holdout evaluation

Seed 42 completed all 300 steps in 5924.0 seconds (98.7 minutes). The final
adapter and resumable step-300 checkpoint are committed in W&B. Its original
holdout evaluation solves 79/256 (30.9%; Wilson 95% interval 25.5–36.8%), with
strict and diagnostic scores agreeing. 167/256 outputs are truncated. The
configuration, complete step sequence, 20,000 unique three-number training
puzzles, held-out order, lack of overlap, and every answer's scores were checked.
[Raw interim evidence and links](../results/study/transfer-progress/README.md).
This is one seed, not the completed transfer study. Four suite evaluations and
seeds 43/44 remain pending; the independent Colab six-run grid continues.

## Remaining

Finish seed/reward and transfer runs; archive remote results; log analysis
artifacts; finish browser/live validation of the interactive demo; complete and compile the
Overleaf report; publish all evidence and audit every requirement. The study is
not yet complete.
