import json
from pathlib import Path
import time

from .data import load_rows, render_prompt
from .model import load_model
from .reward import make_reward


def train(config, resume=None, run=None):
    from datasets import Dataset
    from peft import LoraConfig
    import torch
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    output = Path(config.output_dir)
    manifest = output / "config.json"
    if resume:
        checkpoint = Path(resume)
        if not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("--resume must point to a Trainer checkpoint directory")
        if not manifest.exists() or json.loads(manifest.read_text()) != config.to_dict():
            raise ValueError("Resume requires the same configuration and output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty; use --resume or a fresh --output-dir")
    set_seed(config.seed)
    model, tokenizer = load_model(config)
    train_rows, eval_rows = load_rows(config)
    dataset = Dataset.from_list([{**row, "prompt": render_prompt(tokenizer, row)} for row in train_rows])
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (output / "holdout.json").write_text(json.dumps(eval_rows, indent=2) + "\n")
    (output / "train-puzzles.json").write_text(json.dumps(train_rows) + "\n")
    if run:
        from .tracking import log_files
        (output / "wandb-run.json").write_text(json.dumps({"id": run.id, "url": run.url,
                                                         "entity": run.entity, "project": run.project}))
        log_files(run, "puzzles", "dataset", files=[output / "train-puzzles.json", output / "holdout.json"],
                  metadata={"dataset": config.dataset, "revision": config.dataset_revision})

    class Metrics(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.monotonic()
            self.start_step = state.global_step

        def on_log(self, args, state, control, logs=None, **kwargs):
            elapsed = time.monotonic() - self.start
            steps = state.global_step - self.start_step
            record = dict(logs or {}, step=state.global_step, elapsed_seconds=elapsed)
            if steps:
                record["seconds_per_step"] = elapsed / steps
                record["estimated_remaining_seconds"] = elapsed / steps * (state.max_steps - state.global_step)
            record["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if run:
                run.log({f"runtime/{key}": value for key, value in record.items()
                         if key in ("seconds_per_step", "estimated_remaining_seconds", "peak_vram_gib")}
                        | {"train/global_step": state.global_step})

    args = GRPOConfig(
        output_dir=str(output), max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate, bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.0, loss_type="grpo", use_vllm=False,
        temperature=1.0, top_p=1.0,
        logging_steps=1, save_steps=config.save_steps, save_total_limit=2,
        report_to="wandb" if run else "none", push_to_hub=False,
        run_name=run.name if run else None,
        log_completions=bool(run), num_completions_to_print=2,
        seed=config.seed, data_seed=config.seed,
    )
    peft = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha,
        target_modules="all-linear", lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=args,
        train_dataset=dataset, reward_funcs=make_reward(config.format_reward),
        peft_config=peft, callbacks=[Metrics()],
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    trainer.save_state()
    if run:
        log_files(run, "adapter", "model", directory=output / "adapter",
                  metadata={"base_model": config.model, "revision": config.model_revision})
        log_files(run, "results", "results", files=[manifest, output / "metrics.jsonl", output / "trainer_state.json"])
        log_files(run, "completions", "results", directory=output / "completions")
