"""Bounded isolated five-step trial, with guaranteed normal-exit queue resume.

Launch as a detached process with the existing W&B environment. The main queue
is suspended only during the trial, and elapsed pause time is recorded explicitly.
No package is installed into the main training environment.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def interrupted(signum, frame):
    raise SystemExit(f'Trial interrupted by signal {signum}')


signal.signal(signal.SIGTERM, interrupted)
state = json.loads(Path('/content/study-process.json').read_text())
if state['group'] != 'controlled-l4-v1':
    raise RuntimeError('Expected the original L4 study')
pid = state['pid']
command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
if 'run_training_study.py' not in command or state['group'] not in command or os.getpgid(pid) != pid:
    raise RuntimeError('Main queue identity does not match')
root = Path(state['root'])
work = root / 'outputs/kernel-fast-validation'
if work.exists():
    raise RuntimeError('Trial already exists; inspect it before retrying')
work.mkdir()
active = [p for p in (root/'outputs'/state['group']).glob('*/status.json')
          if json.loads(p.read_text()).get('state') == 'running']
if len(active) != 1 or json.loads(active[0].read_text()).get('stage') != 'train':
    raise RuntimeError('Main queue must be in a known training stage')
pause_record = active[0].parent / 'training/runtime-pauses.jsonl'
environment = {**os.environ, 'PYTHONPATH': f'/content/countdown-kernel-probe:{root}/src'}
started = time.monotonic()
exit_code = None
os.killpg(pid, signal.SIGSTOP)
try:
    with (work/'training.log').open('w') as log:
        result = subprocess.run([
            sys.executable,'-u','-m','countdown_grpo.cli','train',
            '--config','configs/study/binary-42.toml','--steps','5',
            '--output-dir','outputs/kernel-fast-validation/training',
            '--wandb','--group','kernel-fast-validation','--run-name','five-step-fla-trial',
        ],cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        exit_code = result.returncode
finally:
    os.killpg(pid, signal.SIGCONT)
    record = {'reason': 'Isolated five-step FLA end-to-end validation',
              'pause_seconds': time.monotonic()-started, 'trial_exit_code': exit_code,
              'trial_directory': str(work)}
    with pause_record.open('a') as handle:
        handle.write(json.dumps(record)+'\n')
    (work/'disposition.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
if exit_code:
    raise SystemExit(exit_code)
