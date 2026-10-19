"""Evaluate intermediate laptop controlled-grid checkpoints on the original holdout.

Each step-N checkpoint is fetched from W&B by its `checkpoint_global_step_N`
alias, required to be logged by the matching `controlled-local-v2` training run,
and evaluated with the unchanged greedy protocol. Step 300 re-evaluates the final
weights as a reproduction check against the published evaluation.
"""
import argparse
import json
from pathlib import Path
import subprocess

GROUP = 'checkpoint-trajectory-v1'
SOURCE = Path('results/study/controlled-local')
JOBS = [f'{reward}-{seed}' for seed in (42, 43, 44) for reward in ('binary', 'shaped')]
STEPS = (50, 100, 150, 200, 250, 300)
ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors', 'trainer_state.json')


def fetch_checkpoint(api, job, step, destination):
    reference = json.loads((SOURCE / job / 'training/wandb-run.json').read_text())
    name = f"{reference['entity']}/{reference['project']}/model-{job}-train:checkpoint_global_step_{step}"
    artifact = api.artifact(name)
    if artifact.logged_by().id != reference['id']:
        raise ValueError(f'{name} was not logged by laptop run {reference["id"]}')
    destination.mkdir(parents=True, exist_ok=True)
    for file in ADAPTER_FILES:
        artifact.get_entry(file).download(root=str(destination))
    state = json.loads((destination / 'trainer_state.json').read_text())
    if state['global_step'] != step:
        raise ValueError(f'{name} holds global step {state["global_step"]}, not {step}')
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', nargs='+', default=JOBS)
    parser.add_argument('--steps', nargs='+', type=int, default=STEPS)
    parser.add_argument('--root', type=Path, default=Path('outputs') / GROUP)
    parser.add_argument('--base', action='store_true', help='Also evaluate the untrained base model (step 0)')
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Run from a clean checkout so W&B provenance is exact')
    import wandb
    from countdown_grpo.config import load_config
    from countdown_grpo.evaluate import evaluate
    from countdown_grpo.tracking import session
    api = wandb.Api(timeout=60)
    base = args.root / 'base' / 'evaluation.json'
    if args.base and not base.exists():
        # Greedy decoding without an adapter does not depend on seed or reward.
        config = load_config(f'configs/study/{JOBS[0]}.toml')
        with session(config, 'evaluate', 'base-step0-eval', GROUP) as run:
            evaluate(config, None, str(base), run=run)
    for job in args.jobs:
        config = load_config(f'configs/study/{job}.toml')
        for step in args.steps:
            work = args.root / job / f'step-{step}'
            result = work / 'evaluation.json'
            if result.exists():
                print(f'{job} step {step}: already evaluated', flush=True)
                continue
            artifact = fetch_checkpoint(api, job, step, work / 'adapter')
            with session(config, 'evaluate', f'{job}-step{step}-eval', GROUP) as run:
                run.use_artifact(artifact)
                version = f"{artifact.qualified_name.split(':')[0]}:{artifact.version}"
                run.config.update({'checkpoint_artifact': version, 'checkpoint_step': step,
                                   'training_job': job})
                evaluate(config, str(work / 'adapter'), str(result), run=run)


if __name__ == '__main__':
    main()
