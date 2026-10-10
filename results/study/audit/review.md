# Review of diagnostic extraction

Reviewed on 2026-10-10. Selection was deterministic: the first two records in
each observed diagnostic outcome category, separately for baseline and adapter.
This gives 20 records. This is an agent inspection of raw output, not a blinded
human annotation study; no inter-rater agreement is claimed.

| Report | Indices | Observations |
| --- | --- | --- |
| Baseline | 0, 2 | Claimed equation right-hand sides are incorrect. The parser evaluates the left side instead of accepting the target printed on the right. |
| Baseline | 1, 4 | Index 1 declares `420`, omitting all puzzle operands. Index 4 reuses `9`. Both fail number-multiset validation. |
| Baseline | 47, 66 | The final standalone fragments have incomplete syntax. Earlier repeated invalid attempts are not substituted for the declared candidate. |
| Baseline | 73, 196 | Equations embedded in commentary or approximate notation do not match the conservative standalone-line grammar. This illustrates diagnostic coverage limits. |
| Adapter | 0, 1 | Index 0's selected scratch calculation does not use the puzzle multiset; index 1 omits `57` despite reaching the target with the other inputs. |
| Adapter | 2, 4 | Tagged final expressions are correct and use all input operands exactly once. |
| Adapter | 8, 13 | Prose and semicolon-separated attempts do not provide a candidate under the declared extraction rule. No target-directed search is performed. |
| Adapter | 10, 32 | Last standalone arithmetic fragments are syntactically incomplete. |
| Adapter | 96, 140 | Tagged expressions use the inputs correctly but evaluate to `90` instead of `91`, and `6` instead of `61`. |
| Adapter | 118, 122 | Answer tags are opened without being closed. The parser declines fallback to earlier scratchwork. |

The inspection found no disagreement with the specified extraction rule in this
sample. It does reveal a limitation: no-final-answer outputs may contain useful
intermediate work that this deliberately conservative diagnostic does not credit.
Historical truncation cannot be verified without token metadata. Incomplete text
is described as incomplete rather than counted as proven token-cap truncation.

The exact subset-DP solver independently found witnesses for all 256 holdout
puzzles under the all-inputs-once rules. Thus unsolvable evaluation examples do
not explain the observed errors. Witnesses are in `solvability.json` and must
never be included in model prompts.
