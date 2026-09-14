import argparse
from dataclasses import replace
import json
from contextlib import nullcontext

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Qwen Countdown GRPO: train, evaluate, or inspect configuration")
    parser.add_argument("command", choices=["config", "train", "evaluate"])
    parser.add_argument("--config", default="configs/local.toml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", help="Trainer checkpoint path (train only)")
    parser.add_argument("--adapter", help="Saved LoRA adapter path (evaluate only)")
    parser.add_argument("--result", default="outputs/evaluation.json")
    parser.add_argument("--wandb", action="store_true", help="Upload code, metrics, results and model artifacts to W&B")
    parser.add_argument("--run-name")
    parser.add_argument("--group", help="Group baseline, training and final evaluation in W&B")
    args = parser.parse_args()
    config = load_config(args.config)
    changes = {}
    if args.steps is not None:
        changes["max_steps"] = args.steps
    if args.output_dir:
        changes["output_dir"] = args.output_dir
    config = replace(config, **changes)
    if args.resume and args.command != "train":
        parser.error("--resume is only supported for train")
    if args.adapter and args.command != "evaluate":
        parser.error("--adapter is only supported for evaluate")
    if args.command == "config":
        print(json.dumps(config.to_dict(), indent=2))
    else:
        from .tracking import session
        context = session(config, args.command, args.run_name, args.group) if args.wandb else nullcontext(None)
        with context as run:
            if args.command == "train":
                from .train import train
                train(config, args.resume, run=run)
            else:
                from .evaluate import evaluate
                evaluate(config, args.adapter, args.result, run=run)


if __name__ == "__main__":
    main()
