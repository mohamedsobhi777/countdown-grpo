"""Export a checked, self-contained experiment report and learning curve."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from countdown_grpo.data import puzzle_key
from countdown_grpo.reward import score_completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    baseline = json.loads((args.source / "baseline.json").read_text())
    after = json.loads((args.source / "after.json").read_text())
    baseline_keys = [puzzle_key(row) for row in baseline["results"]]
    after_keys = [puzzle_key(row) for row in after["results"]]
    assert baseline_keys == after_keys, "Evaluation puzzles differ"
    assert len(set(after_keys)) == len(after_keys), "Duplicate evaluation puzzles"
    training = json.loads((args.source / "training/train-puzzles.json").read_text())
    assert not set(after_keys).intersection(map(puzzle_key, training)), "Train/eval overlap"
    for report in (baseline, after):
        for row in report["results"]:
            score = score_completion(row["completion"], row["nums"], row["target"])
            assert (score.valid, score.correct) == (row["valid"], row["correct"])
        count = len(report["results"])
        assert count == report["samples"]
        assert sum(r["correct"] for r in report["results"]) / count == report["solve_rate"]
        assert sum(r["valid"] for r in report["results"]) / count == report["valid_rate"]
    metrics = [json.loads(line) for line in (args.source / "training/metrics.jsonl").read_text().splitlines()]
    steps = [row for row in metrics if "reward" in row]
    assert [row["step"] for row in steps] == list(range(1, 301))
    summary = {
        "group": args.source.name,
        "seed": after["config"]["seed"],
        "evaluation_samples": after["samples"],
        "baseline_correct": sum(row["correct"] for row in baseline["results"]),
        "after_correct": sum(row["correct"] for row in after["results"]),
        "baseline_valid": sum(row["valid"] for row in baseline["results"]),
        "after_valid": sum(row["valid"] for row in after["results"]),
        "training_steps": steps[-1]["step"],
        "train_runtime_seconds": metrics[-1]["train_runtime"],
        "peak_allocated_vram_gib": max(row["peak_vram_gib"] for row in steps),
        "first_positive_reward_step": next(row["step"] for row in steps if row["reward"] > 0),
        "last_30_steps_mean_reward": sum(row["reward"] for row in steps[-30:]) / 30,
        "last_30_steps_mean_truncation_rate": sum(row["completions/clipped_ratio"] for row in steps[-30:]) / 30,
        "wandb_baseline": "https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/qu1ovu2z",
        "wandb_training": "https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/uv3hjjih",
        "wandb_evaluation": "https://wandb.ai/mohamedsobhi777/countdown-grpo/runs/1qrixzrf",
    }
    args.destination.mkdir(parents=True, exist_ok=True)
    for filename in ("baseline.json", "after.json"):
        shutil.copyfile(args.source / filename, args.destination / filename)
    shutil.copyfile(args.source / "training/metrics.jsonl", args.destination / "metrics.jsonl")
    (args.destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    x = [row["step"] for row in steps]
    reward = [row["reward"] for row in steps]
    rolling = [sum(reward[max(0, i - 19):i + 1]) / min(i + 1, 20) for i in range(len(reward))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].plot(x, reward, color="#93c5fd", alpha=0.65, linewidth=0.8, label="Per-step reward")
    axes[0].plot(x, rolling, color="#1d4ed8", linewidth=2, label="20-step trailing mean")
    axes[0].set(xlabel="Optimizer step", ylabel="Mean binary reward", title="Training: sampled completions", ylim=(0, 1))
    axes[0].legend(frameon=False, fontsize=8)
    values = [baseline["solve_rate"] * 100, after["solve_rate"] * 100]
    axes[1].bar(["Base model", "After GRPO"], values, color=["#94a3b8", "#1d4ed8"], width=0.5)
    for i, value in enumerate(values):
        axes[1].text(i, value + 2, f"{value:.1f}%", ha="center")
    axes[1].set(ylabel="Exact-solve accuracy (%)", title="Held-out: 256 puzzles, greedy decoding", ylim=(0, 100))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.savefig(args.destination / "learning-curve.svg", metadata={"Date": None})
    fig.savefig(args.destination / "learning-curve.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
