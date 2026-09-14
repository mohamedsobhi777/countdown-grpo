import json
from pathlib import Path

from .data import load_rows, render_prompt
from .model import load_model
from .reward import score_completion


def evaluate(config, adapter=None, destination="outputs/evaluation.json", run=None):
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
    _, rows = load_rows(config)
    results = []
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        inputs = tokenizer([render_prompt(tokenizer, row) for row in batch], padding=True,
                           return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=config.max_completion_length,
                                    do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id)
        completions = tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for row, completion in zip(batch, completions, strict=True):
            score = score_completion(completion, row["nums"], row["target"])
            results.append({**row, "completion": completion, "valid": score.valid, "correct": score.correct})
        print(f"Evaluation {len(results)}/{len(rows)}; solved {sum(r['correct'] for r in results)}", flush=True)
        if run:
            run.log({"eval/completed": len(results), "eval/running_solve_rate": sum(r["correct"] for r in results) / len(results)})
    report = {
        "config": config.to_dict(), "adapter": adapter, "decoding": "greedy",
        "samples": len(results),
        "solve_rate": sum(r["correct"] for r in results) / len(results),
        "valid_rate": sum(r["valid"] for r in results) / len(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    if run:
        import wandb
        from .tracking import log_files
        run.summary.update({f"eval/{key}": report[key] for key in ("samples", "solve_rate", "valid_rate")})
        table = wandb.Table(columns=["numbers", "target", "completion", "valid", "correct"],
                            data=[[r["nums"], r["target"], r["completion"], r["valid"], r["correct"]] for r in results])
        run.log({"eval/completions": table})
        log_files(run, "evaluation", "evaluation", files=[path])
    print(json.dumps({key: value for key, value in report.items() if key not in ("results", "config")}, indent=2))
