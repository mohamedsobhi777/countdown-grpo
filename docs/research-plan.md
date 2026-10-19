# Countdown RL study: approved execution plan

The user approved executing the five-part strengthening plan on 2026-10-10,
including W&B logging/artifacts, frequent GitHub pushes, an Overleaf report,
and Colab if additional compute is needed. No additional design approval is
needed for these experiments. The brainstorming skill informed the decomposition;
the referenced writing-plans skill is not installed, so this file records the plan.

## Research question

When does direct GRPO improve arithmetic task performance in a tiny language
model, and how much of the apparent improvement is compliance with answer format?
The existing 300-step run is a pilot, not the entire study.

## Deliverables and completion evidence

1. **Evaluation audit.** Preserve the strict pilot verifier and introduce a
   separately named, conservative format-independent metric. Specify extraction
   rules before interpreting results; never search answers for a correct expression.
   Re-score archived outputs, categorize failures, inspect a deterministic sample,
   and publish examples and auditable per-answer results. Tests must cover misleading
   intermediate equations, duplicate inputs, ambiguous answers, and unsafe syntax.
2. **Output-budget experiment.** Evaluate the base model and pilot adapter at
   256, 512, and 1024 new tokens on identical frozen puzzles. Preserve raw outputs,
   actual generated token counts, EOS/truncation status, runtimes, and both scores.
   The existing 256-token evaluations remain available as historical results.
3. **Controlled training study.** Binary correctness versus a 0.05 bonus for a
   valid incorrect expression, three training seeds (42, 43, 44), fixed split seed
   42, identical puzzle lists and budgets (300 steps, 32 completions/step, 256-token
   cap, LoRA rank 16). Reuse the pilot only if its exact settings/data match.
   Report per-seed results and dispersion, not just the best seed.
4. **Generalization.** Train on three-number puzzles only and evaluate disjoint
   three- and four-number sets. Evaluate larger input numbers and out-of-range
   targets on separately labeled, deterministic, solver-verified suites. Preserve
   generation rules and witnesses, keep answers out of prompts, and avoid claiming
   these constructed distributions are the original dataset. Freeze suites before
   running the trained models. Distinguish arithmetic transfer from format transfer.
5. **Failure analysis and demo.** Summarize format errors, invalid syntax, number
   reuse/omission, wrong values, division by zero, and token truncation (the latter
   is an independent flag). Provide a local interactive base/adapter comparison
   with exact verifier feedback and an archived-example mode usable without GPU.
6. **Report and provenance.** Keep a compiling Overleaf LaTeX report, repository
   copy, downloadable PDF, W&B run/artifact links, and an experiment ledger updated
   at each milestone. Include negative results, limitations, and outstanding work.
   No unmeasured claims or general-reasoning claims from Countdown accuracy alone.

## Sequence and implementation choices

- Audit archived outputs first; then run output-budget evaluations locally.
- Implement fixed-split seed controls and deterministic generalization suites.
- Run training comparisons in a persistent queue with per-job manifests, failure
  states, and checkpoints. Use local GPU initially; inspect Colab availability if
  queue runtime justifies it. Do not duplicate an already-live job after a timeout.
- Add failure analysis, confidence intervals for held-out accuracy, and paired
  comparisons on the same puzzles. Seed variation is reported separately from
  binomial uncertainty on a fixed checkpoint.
- Build the demo and finish the report only after evidence is available. Final
  completion requires working commands, verified artifacts, compiled report,
  tested demo, and a clean pushed repository covering every item above.

## Status

- [x] Inspect repository, pilot outputs, GPU, GitHub and Overleaf access.
- [x] Record user-approved design and implementation sequence.
- [x] Audit scores and failure categories; log evidence to W&B.
- [x] Create and compile initial Overleaf report.
- [x] Run matched output-budget evaluations.
- [x] Run fixed-split, multi-seed reward comparisons (laptop grid `controlled-local-v2`; see [results](../results/study/controlled-local/README.md)).
- [x] Run three-to-four-number transfer and distribution-shift evaluations.
- [x] Build and verify the interactive comparison demo.
- [x] Publish complete report and perform requirement-by-requirement audit ([completion audit](completion-audit.md), 2026-10-20).

## Approved compute recovery

After runtime loss and rejected L4 access, the user approved rerunning the full
six-run controlled comparison on the laptop. See
[the local recovery design](local-controlled-recovery.md). The completed local
budget, transfer, and demo results remain retained. The new controlled grid uses
a distinct group and is never pooled with Colab evidence.
