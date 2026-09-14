# Countdown GRPO

Direct reinforcement learning on [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
using [`Jiayi-Pan/Countdown-Tasks-3to4`](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4).
No supervised fine-tuning stage. Designed for one NVIDIA GPU with 16 GB VRAM,
using BF16 LoRA and Hugging Face TRL's GRPO trainer.

**Training has not been run.** CPU unit tests cover reward verification, data splits,
and configuration. GPU memory fit, training throughput, checkpoint loading, and
learning improvement must be validated on the first run. Earlier runtime estimates
were planning estimates, not measurements; budget several hours and benchmark first.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Install a CUDA-enabled
PyTorch build compatible with your NVIDIA driver when configuring the environment.
The training extra pins PyTorch 2.8.0 from PyPI (CUDA 12.8 on Linux x86-64);
check CUDA availability before
downloading the model:

```bash
uv sync --locked --extra train
uv run --extra train python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.is_bf16_supported())'
```

The checkpoint is approximately 1.77 GB. Dependencies and CUDA libraries require
additional disk space. Model and dataset downloads occur only when invoking `train`
or `evaluate`; the usual Hugging Face cache is used.

Inspect settings and run CPU tests without the training dependencies or downloads:

```bash
uv run countdown-grpo config
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Run later

Measure the pretrained model on a fixed held-out set:

```bash
uv run --extra train countdown-grpo evaluate --result outputs/baseline.json
```

Start a separate ten-step trial to check memory and approximate throughput:

```bash
uv run --extra train countdown-grpo train --steps 10 --output-dir outputs/trial
```

Then start the default 300-step experiment from the base checkpoint:

```bash
uv run --extra train countdown-grpo train
```

Each optimizer step consumes 32 completions: four puzzles with eight sampled
answers each. These are optimizer steps, not epochs over the whole dataset.
Defaults select up to 20,000 unique training puzzles, cap completions at 256 tokens,
and use LoRA rank 16 across language-model linear layers. The vision tower is
excluded when loading the checkpoint. GRPO is explicitly selected with `loss_type="grpo"`;
the KL coefficient is zero, so no reference model is held in memory.

Resume an interrupted default run from an actual saved checkpoint (replace 50
with the checkpoint number present on disk):

```bash
uv run --extra train countdown-grpo train --resume outputs/countdown/checkpoint-50
```

Resume requires the original config and output directory. Two recent checkpoints
are retained; the final adapter is saved separately to `outputs/countdown/adapter`.
Starting a new run in a nonempty output directory is refused.

Evaluate the adapter on the same puzzles with the same greedy decoding:

```bash
uv run --extra train countdown-grpo evaluate \
  --adapter outputs/countdown/adapter --result outputs/after.json
```

Compare `solve_rate` and `valid_rate` in the two reports and inspect their saved
completions. This measures greedy exact-solve accuracy, not pass@8. Keep the same
config, seed, dataset revision, and completion limit for a meaningful comparison.
The report refuses to overwrite an existing file. Training does not automatically
run evaluation; the commands above make its cost explicit.

## Reward and split

The prompt permits scratch work and requires exactly one final
`<answer>expression</answer>` block. For example, with `[44, 19, 35]` and target 98:

```text
<answer>(44 + 19) + 35</answer>
```

The verifier accepts only positive integer literals, binary `+ - * /`, and
parentheses; each input number must appear exactly once, respecting duplicates.
Fractional and negative intermediate results are permitted. It uses bounded AST
interpretation and exact rational arithmetic, never `eval`. Target copying,
extra numbers, multiple answer blocks, division by zero, powers, and code are rejected.

Default reward is 1 for a correct valid expression and 0 otherwise. Optional
`format_reward` in `[0, 0.1]` rewards valid but incorrect expressions. This changes
the objective and is disabled by default. With binary rewards, groups with no
reward variation provide no relative correctness signal; inspect reward variation
if training stalls. Higher solve rate is evidence of task improvement, not proof
that general reasoning has emerged.

Puzzles are deduplicated by `(target, sorted numbers)`. A seeded SHA-256 partition
assigns each canonical puzzle to train or evaluation before sampling. Duplicates
and number permutations cannot leak across the split. Evaluation defaults to 256
puzzles from the 2% holdout partition.

## Outputs and tuning

- `config.json`: effective experiment settings and pinned Hub revisions.
- `holdout.json`: selected evaluation puzzles.
- `metrics.jsonl`: TRL reward/length metrics, elapsed time, average seconds per
  optimizer step, approximate remaining time, and peak allocated GPU memory.
- `checkpoint-*`: resumable training state; `adapter/`: final LoRA adapter.
- Evaluation reports: config, solve rate, validity rate, and every completion.

Edit `configs/local.toml` or pass `--config path.toml`. If memory runs out, first
reduce `max_completion_length` to 128 or `gradient_accumulation_steps` to 16 or 8;
the effective completion batch must remain divisible by `num_generations`.
Gradient accumulation also determines the rollout batch size in this configuration.
Shorter completions can truncate useful answers and change the experiment.

Qwen3.5 uses hybrid linear attention. Optional `causal-conv1d` and Flash Linear
Attention kernels can improve performance; without them Transformers uses its
PyTorch fallback. They require their own CUDA/build compatibility checks and are
not required by this minimal setup. vLLM is deliberately left out of the initial
implementation. Consult the [Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
and [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer).

All metrics are local. No experiment tracking service or Hub upload is enabled.
Weights, outputs, and environment files are excluded from Git.
