import json
from pathlib import Path
import time
import hashlib

from .data import load_rows, render_prompt
from .model import load_model
from .reward import score_completion
from .diagnostics import analyze_completion


def evaluate(config, adapter=None, destination="outputs/evaluation.json", run=None, puzzles=None):
    import torch
    from transformers import set_seed

    path = Path(destination)
    if path.exists():
        raise ValueError("Evaluation output already exists; choose a new --result path")
    if run:
        run.config.update({"adapter": adapter, "result_path": destination})
        lineage = Path(adapter).parent / "wandb-run.json" if adapter else None
        if lineage and lineage.exists():
            source = json.loads(lineage.read_text())
            run.use_artifact(f"{source['entity']}/{source['project']}/{source['id']}-adapter:latest")
    set_seed(config.seed)
    model, tokenizer = load_model(config)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if puzzles:
        rows = json.loads(Path(puzzles).read_text())
        if not isinstance(rows, list) or not rows or any(set(r) != {'nums','target'} for r in rows):
            raise ValueError('Evaluation inputs must contain only nums and target; omit witnesses')
        if run:
            from .tracking import log_files
            run.config.update({'suite':puzzles,'suite_sha256':hashlib.sha256(Path(puzzles).read_bytes()).hexdigest()})
            log_files(run,'evaluation-inputs','dataset',files=[puzzles])
    else:
        _, rows = load_rows(config)
    results = []
    started = time.monotonic()
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        inputs = tokenizer([render_prompt(tokenizer, row) for row in batch], padding=True,
                           return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=config.max_completion_length,
                                    do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id)
        completions = tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for index, (row, completion) in enumerate(zip(batch, completions, strict=True)):
            score = score_completion(completion, row["nums"], row["target"])
            generated = tokens[index, inputs.input_ids.shape[1]:].tolist()
            eos_position = next((i for i, token in enumerate(generated) if token in eos_ids), None)
            generated_count = eos_position + 1 if eos_position is not None else len(generated)
            truncated = eos_position is None and generated_count >= config.max_completion_length
            results.append({**row, "completion": completion, "valid": score.valid, "correct": score.correct,
                            "generated_tokens": generated_count, "truncated": truncated,
                            "diagnostic": analyze_completion(completion, row['nums'], row['target'], truncated)})
        print(f"Evaluation {len(results)}/{len(rows)}; solved {sum(r['correct'] for r in results)}", flush=True)
        if run:
            run.log({"eval/completed": len(results), "eval/running_solve_rate": sum(r["correct"] for r in results) / len(results)})
    report = {
        "config": config.to_dict(), "adapter": adapter, "decoding": "greedy",
        "samples": len(results),
        "solve_rate": sum(r["correct"] for r in results) / len(results),
        "valid_rate": sum(r["valid"] for r in results) / len(results),
        "diagnostic_solve_rate": sum(r['diagnostic']['diagnostic_correct'] for r in results) / len(results),
        "truncation_rate": sum(r['truncated'] for r in results) / len(results),
        "generation_seconds": time.monotonic() - started,
        "diagnostic_policy": "declared-expression-v1",
        "wandb_url": run.url if run else None,
        "suite": puzzles or "original-holdout",
        "suite_sha256": hashlib.sha256(Path(puzzles).read_bytes()).hexdigest() if puzzles else None,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    if run:
        import wandb
        from .tracking import log_files
        run.summary.update({f"eval/{key}": report[key] for key in
                            ("samples", "solve_rate", "valid_rate", "diagnostic_solve_rate", "truncation_rate", "generation_seconds")})
        table = wandb.Table(columns=["numbers", "target", "completion", "valid", "correct"],
                            data=[[r["nums"], r["target"], r["completion"], r["valid"], r["correct"]] for r in results])
        run.log({"eval/completions": table})
        log_files(run, "evaluation", "evaluation", files=[path])
    print(json.dumps({key: value for key, value in report.items() if key not in ("results", "config")}, indent=2))
