import json
from pathlib import Path

from .data import load_rows, render_prompt
from .model import load_model
from .reward import score_completion


def evaluate(config, adapter=None, destination="outputs/evaluation.json"):
    import torch
    from transformers import set_seed

    path = Path(destination)
    if path.exists():
        raise ValueError("Evaluation output already exists; choose a new --result path")
    set_seed(config.seed)
    model, tokenizer = load_model(config)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    _, rows = load_rows(config)
    results = []
    for row in rows:
        inputs = tokenizer(render_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=config.max_completion_length,
                                    do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id)
        completion = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        score = score_completion(completion, row["nums"], row["target"])
        results.append({**row, "completion": completion, "valid": score.valid, "correct": score.correct})
    report = {
        "config": config.to_dict(), "adapter": adapter, "decoding": "greedy",
        "samples": len(results),
        "solve_rate": sum(r["correct"] for r in results) / len(results),
        "valid_rate": sum(r["valid"] for r in results) / len(results),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("results", "config")}, indent=2))
