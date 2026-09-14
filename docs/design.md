# Design and implementation plan

Approved scope: implement the previously proposed local Qwen Countdown experiment,
without running training, and publish source to a private personal GitHub repository.

Use the official Qwen3.5-0.8B base checkpoint, load its text backbone, and apply
BF16 rank-16 LoRA. TRL generates eight candidates per prompt and optimizes GRPO
directly against verified equation correctness, without SFT. Limit completions
to 256 tokens and train for 300 optimizer steps by default. Full-parameter training
and vLLM were considered; LoRA with native generation minimizes setup and memory
requirements for the initial experiment.

Components: validated TOML configuration; bounded exact-arithmetic reward;
canonical puzzle deduplication and stable holdout; explicit model loader;
training with checkpoints and local timing metrics; separate base/adapter
evaluation with completion reports; CLI and setup documentation.

Implementation sequence:
1. Build and test reward parsing and duplicate-safe dataset partitioning.
2. Implement model loading, LoRA/GRPO wiring, persistence, and evaluation.
3. Validate configuration, command help, dependency resolution, and source syntax.
4. Review tracked files and publish a private repository on the authenticated account.

No model download, generation, benchmark, or training is part of this delivery.
CPU tests cannot establish GPU fit or learning success. A later ten-step trial
must validate checkpoint loading and memory and measure runtime before a full run.
