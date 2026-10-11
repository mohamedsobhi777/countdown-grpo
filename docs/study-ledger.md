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
All four frozen-suite evaluations subsequently finished and passed audits:
146/256 three-number, 26/256 four-number, 63/256 larger-input, and 107/256
larger-target puzzles. Strict and diagnostic scores agree throughout. The
20,000-puzzle training list, complete step sequence, configuration, frozen hashes,
ordered evaluation lists, lack of overlap, and all per-answer scores were checked
again before publishing the full single-seed table. Raw outputs, training metrics,
configuration, and exported W&B hardware/requirements records are archived in the
interim directory. Comparisons with the single mixed-arity pilot are descriptive.

Seed 42 is marked complete. Seed 43 is confirmed live (PID 685077), W&B
[3ifuj2rg](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/3ifuj2rg).
Seeds 43/44 and the independent Colab six-run grid remain pending. This is one
seed, not the completed transfer study.

## Transfer failure interpretation

A post-hoc cross-tabulation of the first seed's audited outputs shows that all
110 three-number failures and 211 of 230 four-number failures are truncated.
All 198 four-number no-declared-expression outcomes are truncated too. The
interim summary records these overlaps separately from the exclusive diagnostic
categories. The report now explicitly conditions transfer findings on the
256-token cap; the pilot's output-budget finding is not extrapolated to this
new adapter. These counts do not establish that truncation caused failure.

## 2026-10-11: first controlled run collected and audited

Binary reward, seed 42 (`fneh9kxb`) completed all 300 steps in 11,573.2 seconds
(192.9 minutes), followed by evaluation `9xwjji2l`. The collector downloaded its
SHA-256-verified archive and exported finished-run W&B provenance. The full
single-run scientific checks passed: settings, steps, puzzle order, no overlap,
unique keys, recomputed per-answer scores, and clean source/pinned FLA/single-L4
environment. It solves 68/256 (26.6%; Wilson 95% 21.5–32.3%), with 174 truncated
outputs and strict/diagnostic scores agreeing. The final adapter and resumable
step-300 checkpoint are committed in W&B. [Raw interim evidence and links](../results/study/controlled-progress/README.md).
The queue has started shaped-reward seed 42. Five runs remain; cross-run gates
and the reward comparison are pending. The historical laptop/fallback pilot is
not an environment-matched comparison for this run.

## Final figures and remaining work

Seed 43's completed local training provenance was exported from W&B and checked.
All 149 archived source files match its clean recorded commit `7c32957`.
Hardware metadata, base package versions, and the full normalized requirements
hash match seed 42; optional FLA/causal-convolution packages are absent. Every
package module matches seed 42 except `tracking.py`, whose diff only adds Python
and optional-kernel metadata. Seed 42's historical dirty flag and missing fields
remain documented rather than overwritten. [Seed-43 source hashes and scope](../results/study/local-source-audit-seed43.json),
[exported runtime provenance](../results/study/transfer-progress/seed-43-training/provenance.json).
Seed 44 review and the completed seed-43 evaluation set remain pending.
Seed 43's original-holdout evaluation (`3jlajmcr`) subsequently passed the full
single-evaluation audit: 93/256 solved (36.3%; Wilson 95% 30.7–42.4%), strict and
diagnostic scores agreeing, 157 truncated outputs. Training took 5463.0 seconds
(91.1 minutes). Its training and holdout puzzle lists match seed 42 exactly.
Raw completions, metrics, configuration, and the audited summary are published
under `results/study/transfer-progress`; the four frozen-suite evaluations remain
pending. This is not a completed seed grid or seed average.

The final seed-comparison plotting command is ready for the completed audited
aggregates. Run with system Python (Matplotlib installed there), after each
analysis has produced its full `summary.json`:

```bash
python3 scripts/plot_seed_studies.py controlled
python3 scripts/plot_seed_studies.py transfer
```

Each exports PNG/SVG/PDF and source-hash metadata. The plots separate individual
checkpoint Wilson intervals from sample standard deviation across seeds, show
all seeds, and reject incomplete grids or inconsistent count/aggregate values.
Synthetic rendering and rejection checks passed, including 0/256 and 256/256
endpoints; synthetic figures remain under ignored `outputs/` and are not study
results. Actual final figures and their report/W&B publication remain pending.

Finish seed/reward and transfer runs; archive remote results; log analysis
artifacts; finish browser/live validation of the interactive demo; complete and compile the
Overleaf report; publish all evidence and audit every requirement. The study is
not yet complete.

## Second transfer seed complete (2026-10-11)

Seed 43 completed all five evaluations: original holdout 93/256, three-number
161/256, four-number 38/256, larger inputs 79/256, and larger targets 112/256.
All scores, ordered frozen puzzles, hashes, configuration, all 300 training steps,
and training overlap were audited again before publication. Strict and diagnostic
scores agree in every suite. Raw completions and cross-tabulations are in
`results/study/transfer-progress/seed-43-summary.json` and adjacent evaluation files.
All 95 three-number failures and 197 of 218 four-number failures are truncated;
this is descriptive overlap, not a causal budget result. Seed 44 started
as queued, W&B run `mj074g8h`; its source/runtime audit and evaluations remain
pending. No seed-averaged finding is claimed yet.

### Third transfer seed startup provenance

The running seed-44 job (`mj074g8h`) started from clean commit
`41724757e60a4db869cb39d1bf78f910f96c85d5`. Its committed W&B source snapshot
`source-countdown-grpo-None:v22` contains 180 files, all byte-identical to that
commit. Every package Python module matches seed 43 exactly, as do the hardware,
Python version, core/optional package metadata, and normalized full requirements.
The source audit is `results/study/local-source-audit-seed44.json`; captured
startup provenance and requirements are under `transfer-progress/seed-44-training`.
This verifies startup provenance only: the run is still training and final
training/evaluation checks remain pending. Refresh its exported state after completion.

## First matched reward pair complete (2026-10-11)

Shaped seed 42 finished 300 steps in 7050.5079 seconds (117.5 minutes), then
scored 26/256 (10.2%; Wilson 95% 7.0–14.5%). All 256 expressions are valid;
230 have wrong values, and none are truncated. Strict and diagnostic counts
agree. The collector verified and secured its archive; both runs passed full
step/configuration/data/score audits, with identical ordered puzzle lists and
source/Python/full requirements/kernel/hardware provenance. Compared with
binary seed 42 (68/256), shaped gains 11 and loses 53, a −16.4-point difference.
This is a single-seed interim observation; the remaining four controlled runs
are pending. Raw rewards differ in definition and are not directly comparable.
Training `skfvvhnz`, evaluation `83tmr7i5`, final adapter `skfvvhnz-adapter:v0`,
and resumable step-300 checkpoint `model-shaped-42-train:v6` are committed in
W&B. Evidence is under `results/study/controlled-progress`. Binary seed 43
started automatically in the same controlled queue.

## Third transfer seed training complete (2026-10-11)

Seed 44 (`mj074g8h`) completed all 300 steps in 6017.4405 seconds (100.3 minutes),
with peak allocated GPU memory 2.711 GiB. W&B reports the training run finished
in `three-only-local-v1`; its configuration matches the local manifest, and its
final hardware and normalized requirements match the audited startup record.
The final adapter `mj074g8h-adapter:v0` and step-300 checkpoint
`model-three-only-44-train:v6` are committed. The checkpoint manifest includes
optimizer, scheduler, RNG, trainer state, adapter weights, and adapter config.
The exported provenance has been refreshed to finished state. Evaluation is
running; its five suites and the three-seed aggregate are not yet complete.

### Third transfer seed original holdout audited

The finished holdout evaluation (`p7nctb9i`) scores 77/256 (30.1%; Wilson 95%
24.8–36.0%), with matching strict/diagnostic counts and 172 truncated outputs.
All 256 answers were re-scored. Training settings, all 300 steps, the exact
ordered training/holdout lists versus seed 43, and absence of training overlap
passed verification. Raw answers, training metrics/configuration, and the
interim summary are published under `results/study/transfer-progress`. Four
frozen suites remain pending; the complete transfer aggregate is not ready.

## Three-seed transfer study complete (2026-10-11)

All three training runs and all 15 evaluations finished. The aggregate validator
checked every training configuration and step sequence, identical ordered train
and holdout lists, frozen suite hashes, absence of training overlap, adapter
lineage, and every archived score. Strict and diagnostic scores agree in all
cells. Mean accuracy ± sample SD is 32.4% ± 3.4 pp on the original holdout,
57.3% ± 5.5 pp on three-number puzzles, 11.3% ± 3.1 pp on four-number puzzles,
27.5% ± 3.2 pp on larger inputs, and 42.6% ± 1.0 pp on larger targets.
The same puzzles recur across seeds; no independent pooling is used. Shifted
suites are constructed, not difficulty-matched. Comparisons to the single
mixed-arity pilot remain descriptive.

Evidence is in `results/study/transfer`; W&B analysis is `0svwuwkr`. The figure
was generated from the complete audited summary and visually reviewed. Finished
training provenance and requirements accompany each seed; the source audits
retain the seed-42 historical dirty-tree caveat. The live demo GPU/browser
verification started automatically as `t3alv4y9` after the local queue finished.

### Live CUDA and browser demo verified

Run `t3alv4y9` finished successfully after the local queue released the GPU.
Real base/adapter HTTP requests produced nonempty, repeatable greedy outputs
after adapter toggling. The actual browser generation button returned the same
outputs, rendered completions and verifier details, passed mobile layout checks,
and produced no JavaScript errors. W&B retains screenshot and output artifacts;
`results/study/demo-live.json` records the passed checks. On the one integration
puzzle, the base expression has exact value 1274/19 and the adapter answer is 66.
This is not an additional benchmark. Archived-mode checks remain documented
separately. The demo and all transfer evaluations are now complete; four
controlled reward runs and the final consolidated audit remain outstanding.

## Colab runtime loss and recovery evidence

The original runtime disappeared from the server. W&B confirms binary seed 43
finished 300 steps in 11531.0974 seconds and evaluated at 64/256 (25.0%; Wilson
95% 20.1–30.6%), with 180 truncated outputs. The collector secured its archive
before runtime loss; full configuration, step, data, score, and provenance gates
pass. It is published in `results/study/controlled-progress/binary-43`.

Shaped seed 43 (`vd89ey0e`) is crashed with last logged step 197. Its committed
step-150 checkpoint (`model-shaped-43-train:v3`) was downloaded and all resume
files verified. Its partial evidence is preserved separately and excluded from
completed comparisons. The termination cause is not established. Colab lists no
active sessions; a replacement L4 request was rejected for quota/entitlement.
The collector was stopped after confirming the session is gone. The user was
asked to restore L4 access or choose a complete, consistently configured local
rerun. Recovery is pending; the study remains incomplete.

## Laptop recovery approved

The user selected a fresh six-run laptop controlled comparison. Group
`controlled-local-v2` uses the existing Python environment and fallback kernels
on the RTX 4090 Laptop GPU, with all original scientific settings unchanged.
A frozen detached checkout isolates training from ongoing documentation updates.
The runner captures finished provenance and audits every completed job before
aggregating the complete grid. Colab evidence remains separate.

### Local queue launched

After 48 CPU tests passed, tmux session `countdown-local-controlled` launched
the six-run queue. Frozen source is `f0dbd20170a8c1ed02e06577702d53dba17d261b`,
under `outputs/local-controlled-source`. Queue PID 4109686; first training run
`eudebo7g` is binary seed 42. Logs are in
`outputs/launch/controlled-local-v2.log`; job outputs and queue state are under
`outputs/local-controlled-source/outputs/controlled-local-v2`. The isolated
checkout uses the existing main `.venv` with its own source on PYTHONPATH.
No packages were installed or upgraded. Aggregate results will first appear in
the isolated checkout's `results/study/controlled-local` and will be audited,
plotted, and published to the main checkout after completion.

### First laptop controlled run completed

Binary seed 42 (`eudebo7g`) completed all 300 steps in 5368.2092 seconds
(89.5 minutes). Evaluation `17gdgf7i` scored 80/256 (31.25%; Wilson 95%
25.9–37.2%), with strict/diagnostic agreement and 162 truncated outputs.
The complete local-profile audit passed, including raw-score recomputation,
configuration, data separation/order, all training steps, and runtime provenance.
Evidence is in `results/study/controlled-local-progress/binary-42`.
Final adapter `eudebo7g-adapter:v0`, resume checkpoint
`model-binary-42-train:v13`, results, and completion logs are committed to W&B.
The queue advanced automatically to shaped seed 42. Five runs remain; no
multi-seed or reward-comparison conclusion is available from this first run.

### First matched laptop seed completed

Shaped seed 42 (`ofj5pru1`) completed all 300 steps in 3206.8835 seconds
(53.4 minutes). Evaluation `8694c6tk` scored 1/256 (0.39%; Wilson 95%
0.07–2.18%), with strict/diagnostic agreement, all 256 expressions valid,
255 wrong values, and no truncated outputs. The full local-profile audit passed.
Ordered puzzle lists, source, environment, and hardware match binary seed 42.
Shaped reward gains one solution and loses 80, a −30.86 percentage-point
difference for this seed; four runs remain before multi-seed conclusions.
The adapter `ofj5pru1-adapter:v0`, resume checkpoint
`model-shaped-42-train:v13`, results, completion logs, and evaluation artifact
are committed to W&B. Published evidence and paired counts are under
`results/study/controlled-local-progress`. Binary seed 43 started automatically.

### Third laptop controlled run completed

Binary seed 43 (`d9it45u2`) completed 300 steps in 5127.7501 seconds
(85.5 minutes). Evaluation `dmxcq1wh` scored 13/256 strict correct
(5.08%; Wilson 95% 2.99–8.49%) and 16/256 diagnostic correct (6.25%).
The three diagnostic recoveries are preserved separately with raw outputs;
all are truncated and use the last standalone arithmetic line. Total
truncation is 219/256. The full local audit passed, including matching ordered
puzzles, source, environment, and hardware with binary seed 42.
Adapter `d9it45u2-adapter:v0`, resumable checkpoint
`model-binary-43-train:v13`, results, and all 300 completion logs are committed
to W&B. Shaped seed 43 (`s6q8yswn`) started automatically; three runs remain.
