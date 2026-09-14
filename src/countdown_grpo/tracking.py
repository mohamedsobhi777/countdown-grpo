"""Explicit W&B sessions with source, dependency, and result artifacts."""
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess


def log_files(run, name, kind, files=(), directory=None, metadata=None):
    import wandb
    artifact = wandb.Artifact(f"{run.id}-{name}", type=kind, metadata=metadata)
    for path in files:
        path = Path(path)
        if path.is_file():
            artifact.add_file(str(path), name=path.name)
    if directory is not None:
        artifact.add_dir(str(directory))
    run.log_artifact(artifact)


@contextmanager
def session(config, job_type, name=None, group=None):
    import wandb
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    root = Path.cwd()
    directory = root / "outputs" / "wandb"
    directory.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        entity="mohamedsobhi777", project="countdown-grpo", job_type=job_type,
        name=name, group=group, config=config.to_dict(), dir=str(directory),
        mode="online", save_code=True,
    ) as run:
        tracked = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
        allowed = {str((root / path).resolve()) for path in tracked if path}
        run.log_code(root=str(root), include_fn=lambda path: str(Path(path).resolve()) in allowed)
        run.config.update({
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "wandb")},
        })
        Path(run.dir, "experiment-config.json").write_text(json.dumps(config.to_dict(), indent=2))
        run.save(str(Path(run.dir, "experiment-config.json")), base_path=run.dir)
        print(f"W&B run: {run.url}", flush=True)
        yield run
