"""Run and audit six fresh laptop jobs; never merge them with the Colab grid."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from analyze_controlled_study import PROFILES, checked_run
from countdown_grpo.tracking import optional_version

GROUP = PROFILES['local']['group']
JOBS = [f'{reward}-{seed}' for seed in (42, 43, 44) for reward in ('binary', 'shaped')]


def export_provenance(work):
    import wandb
    reference = json.loads((work/'training/wandb-run.json').read_text())
    run = wandb.Api(timeout=30).run(f"{reference['entity']}/{reference['project']}/{reference['id']}")
    if run.state != 'finished' or run.group != GROUP:
        raise ValueError('Training run has not finished in the local group')
    config = json.loads((work/'training/config.json').read_text())
    if any(run.config.get(k) != v for k, v in config.items()):
        raise ValueError('W&B training configuration differs from local manifest')
    with tempfile.TemporaryDirectory() as temporary:
        metadata = json.loads(Path(run.file('wandb-metadata.json').download(root=temporary, replace=True).name).read_text())
        requirements = Path(run.file('requirements.txt').download(root=temporary, replace=True).name).read_text()
    record = {'training_url': run.url, 'state': run.state, 'group': run.group,
              'captured_at': datetime.now(timezone.utc).isoformat(),
              'config': {k: run.config.get(k) for k in
                         ('git_commit', 'git_dirty', 'packages', 'optional_kernel_packages', 'python_version')},
              'hardware': {k: metadata.get(k) for k in ('gpu', 'gpu_count', 'cpu_count', 'cudaVersion', 'python', 'os')}}
    record['config']['requirements_sha256'] = hashlib.sha256('\n'.join(sorted(requirements.splitlines())).encode()).hexdigest()
    (work/'training/requirements.txt').write_text(requirements)
    (work/'training/provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    files = ('evaluation.json', 'training/config.json', 'training/metrics.jsonl',
             'training/train-puzzles.json', 'training/holdout.json')
    (work/'collection.json').write_text(json.dumps({
        'method': 'local output; no remote archive',
        'source_sha256': {n: hashlib.sha256((work/n).read_bytes()).hexdigest() for n in files}}, indent=2)+'\n')


def main():
    import torch
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Use a clean, frozen checkout')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != PROFILES['local']['gpu']:
        raise RuntimeError('Expected the single laptop GPU')
    if {k: optional_version(k) for k in PROFILES['local']['kernels']} != PROFILES['local']['kernels']:
        raise RuntimeError('The local study requires the unchanged fallback kernel environment')
    root = Path('outputs')/GROUP
    root.mkdir(exist_ok=False)
    state = {'group': GROUP, 'pid': os.getpid(), 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'state': 'running'}
    def save(**changes):
        state.update(changes)
        (root/'queue.json').write_text(json.dumps(state, indent=2)+'\n')
    save()
    try:
        for job in JOBS:
            save(job=job, stage='train_and_evaluate')
            subprocess.run([sys.executable, '-u', 'scripts/run_training_study.py', '--group', GROUP, '--jobs', job], check=True)
            save(stage='audit')
            # Retry only read-only W&B retrieval after completed training; never
            # retry or duplicate a training job because an API call failed.
            for attempt in range(5):
                try:
                    export_provenance(root/job)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(30)
            reward, seed = job.split('-')
            checked_run(root/job, 0.0 if reward == 'binary' else 0.05, int(seed), 'local')
        save(stage='aggregate')
        subprocess.run([sys.executable, 'scripts/analyze_controlled_study.py', str(root), '--profile', 'local',
                        '--destination', 'results/study/controlled-local', '--wandb'], check=True)
        save(state='complete')
    except Exception as error:
        save(state='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
