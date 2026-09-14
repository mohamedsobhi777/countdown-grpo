"""Run baseline, training, and final evaluation sequentially in isolated processes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default=datetime.now(timezone.utc).strftime("countdown-%Y%m%d-%H%M%S"))
    parser.add_argument("--config", default="configs/local.toml")
    args = parser.parse_args()
    root = Path("outputs") / args.group
    root.mkdir(parents=True, exist_ok=False)
    common = ["--config", args.config, "--wandb", "--group", args.group]
    stages = [
        ("baseline", ["evaluate", "--result", str(root / "baseline.json")]),
        ("train", ["train", "--output-dir", str(root / "training")]),
        ("after", ["evaluate", "--adapter", str(root / "training" / "adapter"), "--result", str(root / "after.json")]),
    ]
    for stage, command in stages:
        (root / "status.json").write_text(json.dumps({"stage": stage, "state": "running", "group": args.group}))
        result = subprocess.run([sys.executable, "-u", "-m", "countdown_grpo.cli", *command,
                                 *common, "--run-name", f"{args.group}-{stage}"])
        if result.returncode:
            (root / "status.json").write_text(json.dumps({"stage": stage, "state": "failed", "exit_code": result.returncode}))
            raise SystemExit(result.returncode)
    (root / "status.json").write_text(json.dumps({"state": "complete", "group": args.group}))


if __name__ == "__main__":
    main()
