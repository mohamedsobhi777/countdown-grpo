> Historical record. The interrupted Colab grid was not resumed; the user approved a full six-run laptop rerun, which is complete: see [controlled-local](../controlled-local/README.md).

# Colab runtime interruption

The original `countdown-study` session disappeared from the server. W&B records
shaped seed 43 (`vd89ey0e`) as crashed, with last logged step 197. No specific
termination cause is established. Binary seed 43 finished before the interruption
and its archive was collected and audited: 64/256 strict and diagnostic solves.

The interrupted shaped run is excluded from completed comparisons. Its step-150
checkpoint `model-shaped-43-train:v3` is committed in W&B and downloaded locally,
with adapter weights/config, optimizer, scheduler, RNG, and trainer state verified.
A replacement L4 allocation was rejected for quota or entitlement. No alternate
hardware has been silently mixed into the controlled comparison. The collector
was stopped after confirming no server sessions remain. Compute recovery is
pending; three complete runs, one interrupted run, and two unstarted runs do not
constitute a complete six-run comparison.

[Interrupted run](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/vd89ey0e) ·
[Completed binary seed 43](https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/pz78iffl).
