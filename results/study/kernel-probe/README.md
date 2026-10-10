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
are retained locally and will be included in the W&B investigation artifact.

A bounded five-step end-to-end training trial is in progress. The main training
process is temporarily suspended during that trial and automatically resumed on
normal/handled exit; the exact pause duration is recorded. These trial outputs
are excluded from the six-run reward comparison. Kernel speedups alone are not
grounds to claim faster end-to-end training or to change the study backend.

Implementation reference: [FLA installation documentation](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md).
