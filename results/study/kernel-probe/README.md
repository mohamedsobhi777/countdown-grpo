# Optional linear-attention kernel investigation

The L4 fallback implementation initially averaged about 50.8 seconds per training
step. A three-step A100 trial averaged 49.3 seconds and was stopped and archived;
faster hardware alone did not materially help this setup.

FLA 0.5.2, fla-core 0.5.2, and einops 0.8.1 were installed in a **separate**
directory on the L4 VM. The running study's environment was not modified.
Representative Qwen3.5 shapes were compared with the unwrapped Transformers
reference implementations, using BF16 inputs and identical random tensors.

The corrected microprobe measured 3.27× chunk-forward/backward and 2.11× recurrent
forward speedups. Chunk output relative L2 difference was 0.44%; individual input
gradient relative L2 differences were 0.27–0.53%. The single recurrent output
matched exactly in this probe. Both returned states were checked for finiteness;
this is not a proof of equivalent long-horizon cache behavior. Thresholds were
2% relative output error and 5% relative gradient error. Timing excludes initial
compilation and averages three calls; concurrent training can affect these
microtimings. Full data and package versions: [kernel-probe-result.json](kernel-probe-result.json).

An initial probe incorrectly passed `beta` positionally to the recurrent FLA
function, where the fifth positional argument is `gk`. It produced nonfinite
values. Inspection of both signatures and Transformers' actual call identified
the probe bug; passing `g` and `beta` by keyword fixed it. This failure was **not**
a model-training failure or evidence that FLA itself is broken. Both probe logs
and the full fallback archive are preserved in the
[W&B investigation artifact](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/jk9p648k).

A bounded [five-step end-to-end training trial](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/oi2t5mfe)
completed successfully in 203.3 seconds of trainer time. The first step took
48.2 seconds; steps 2–5 averaged **37.63 seconds**, about 26% less than the prior
50.8-second fallback steps. The main process resumed automatically after a
recorded 266.62-second pause. This pause is retained in the archived fallback
timing evidence and must not be confused with training compute.

On this evidence, all six controlled reward runs were restarted consistently in
`controlled-l4-fla-v1` using the pinned `fast` extra. The fallback attempt was
intentionally interrupted after 41 steps, archived, and excluded from completed
controlled results. The five-step trial is also excluded. Its all-zero sparse
rewards do not establish learning quality; the full multi-seed study remains
necessary. The original pilot and local transfer runs retain the fallback backend.

Implementation reference: [FLA installation documentation](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md).
