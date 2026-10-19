# Completion audit

Audited on 2026-10-20 against the six deliverables in
[research-plan.md](research-plan.md). Each row names the evidence that was
inspected, not just the claim it supports. Results come from the current
`main` branch.

## Deliverables

| # | Requirement | Evidence inspected | Verdict |
| --- | --- | --- | --- |
| 1 | Strict verifier preserved; separately named format-independent diagnostic with rules fixed before interpretation | `src/countdown_grpo/reward.py` (strict) and `diagnostics.py` (`declared-expression-v1`); rules in report §4 | Met |
| 1 | Archived outputs re-scored, failures categorized, deterministic sample inspected, per-answer results published | [audit/summary.json](../results/study/audit/summary.json): 0/256 base and 85/256 adapter under both metrics; [review.md](../results/study/audit/review.md): 20 records, first two per category; [solvability.json](../results/study/audit/solvability.json): 256/256 solvable | Met |
| 1 | Tests for misleading intermediate equations, duplicate inputs, ambiguous answers, unsafe syntax | `tests/test_diagnostics.py`: `test_ignores_claimed_rhs`, `test_does_not_search_for_correct_intermediate`, `test_duplicate_omission`, `test_rejects_multiple_tags`, `test_unsafe_syntax`; `tests/test_reward.py`: `test_invalid_and_exploit_attempts` | Met |
| 2 | Base and pilot adapter at 256/512/1024 tokens on identical frozen puzzles, with raw outputs, token counts, EOS/truncation, runtimes and both scores | Six reports in [budgets/](../results/study/budgets/); each answer records `completion`, `generated_tokens`, `truncated`, `correct`, `diagnostic`; each report records `generation_seconds` | Met |
| 3 | Binary vs 0.05 validity bonus, seeds 42/43/44, split seed 42, identical puzzles and budgets; per-seed results and dispersion | [controlled-local/summary.json](../results/study/controlled-local/summary.json): six runs, 300 steps each, same ordered puzzles, clean source `f0dbd20`; per-seed table, sample SD and paired differences in [README](../results/study/controlled-local/README.md). Pilot not reused; all six runs are fresh | Met (laptop rerun approved by the user after Colab loss) |
| 4 | Three-number-only training, disjoint 3- and 4-number evaluation, larger-input and out-of-range-target suites; frozen, solver-verified, answers kept out of prompts | [transfer/summary.json](../results/study/transfer/summary.json): 3 seeds x 5 suites; [data/suites/v1/manifest.json](../data/suites/v1/manifest.json) hashes, frozen 2026-10-10; witnesses in a separate file; `evaluate.py` rejects input rows with keys other than `nums`/`target` | Met |
| 4 | Distinguish arithmetic transfer from format transfer | Strict and diagnostic scores agree in all 15 transfer cells (report §9), so the 3-to-4-number drop is not recovered by relaxing the answer format | Met, within the parser's stated coverage limits |
| 5 | Failure summary: format errors, invalid syntax, reuse/omission, wrong values, division by zero, truncation as an independent flag | Diagnostic categories in every evaluation JSON; `division_by_zero` handled in `diagnostics.py` and tested (`test_division_zero_and_fraction`); truncation is a separate per-answer field (`test_truncation_is_independent`). No evaluated answer in the controlled grid hit division by zero | Met |
| 5 | Local interactive base/adapter demo with exact verifier feedback and a GPU-free archived mode | Re-run 2026-10-20: `scripts/check_demo_browser.py` passed (256 examples, filters, exact verification, archive mode, mobile layout, no JS errors). Live CUDA mode: [demo-live.json](../results/study/demo-live.json), state `passed` | Met |
| 6 | Compiling Overleaf report, repository copy, downloadable PDF, W&B links, ledger | Overleaf project `6aca3d97baaac8402a980db8` pulled 2026-10-20 and identical to `report/main.tex`; compiled `report/main.pdf` committed; report archived in W&B ([report-links.json](../results/study/report-links.json)); [study-ledger.md](study-ledger.md) | Met |
| 6 | Negative results, limitations, outstanding work; no unmeasured or general-reasoning claims | Negative results: shaping lowers accuracy, binary seed 43 fails, longer outputs gain one solution, 4-number transfer is weak. Limitations section in report; "no general reasoning" stated in §1 and the abstract | Met |

## Global gates

| Gate | Evidence | Verdict |
| --- | --- | --- |
| Working commands and tests | `PYTHONPATH=src python3 -m unittest discover -s tests`: 48 tests pass (2026-10-20) | Met |
| Verified artifacts | All 93 distinct W&B run/artifact links in README, report, docs and results resolve through the W&B API. 75 linked runs are `finished`; the 3 `crashed` runs are the documented interruptions (`4anzz6c2` L4 fallback, `i7dgueld` A100 trial, `vd89ey0e` Colab shaped-43) | Met |
| Clean pushed repository | Working tree clean; local `main` equals `origin/main` | Met |
| Never pool Colab and laptop results | Separate W&B groups (`controlled-l4-fla-v1`, `controlled-local-v2`) and directories; the aggregator's `local` profile accepts only laptop runs | Met |

## Gaps found and fixed in this audit

- `research-plan.md` still showed the multi-seed comparison and this audit as
  unchecked.
- The report kept running-log wording ("is now running", "remain
  unstarted", "will be reported separately") and an L4-only provenance
  paragraph. These were rewritten in the past tense, and the laptop audit path
  was added.
- Interim result pages (`controlled-progress`, `controlled-local-progress`,
  `runtime-interruption`) now open with a pointer to the final results, and
  `generalization-pilot` no longer calls the transfer study pending.

## Beyond the approved plan

The original improvement proposal (2026-10-10) also suggested showing how
failure categories change *during* training. The approved plan dropped that
item, and nothing in the study measured it. The study's main mechanistic claim,
that shaped policies settle on the validity bonus, rested only on final
checkpoints. The checkpoint-trajectory follow-up, pre-registered in
[study-ledger.md](study-ledger.md), measured it on 2026-10-20: see
[checkpoint-trajectory](../results/study/checkpoint-trajectory/README.md). Both
of its integrity gates passed: exact step-300 reproduction and exact training-reward
recomputation.
