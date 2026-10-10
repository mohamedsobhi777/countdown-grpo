"""Archive a specifically identified, superseded runtime trial.

Interrupt the owned queue, wait for shutdown, and archive its incomplete evidence.
This never labels an interrupted run as a completed training experiment.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time

metadata = json.loads(Path('/content/study-process.json').read_text())
reasons = {
    'controlled-a100-v1': 'Initial A100 throughput was only about 3% faster than L4 with the fallback backend; release the A100.',
    'controlled-l4-v1': 'Five-step FLA validation passed and warm steps were about 26% faster; restart all six comparison runs consistently in controlled-l4-fla-v1.',
}
group = metadata['group']
if group not in reasons:
    raise RuntimeError('This archival script only accepts the two superseded runtime-trial groups')
pid = metadata['pid']
command_path = Path(f'/proc/{pid}/cmdline')
if command_path.exists():
    command = command_path.read_bytes().decode().replace('\0', ' ')
    if 'run_training_study.py' not in command or group not in command:
        raise RuntimeError('PID does not identify the expected training queue')
    if os.getpgid(pid) != pid:
        raise RuntimeError('Queue is not its own process-group leader')
    os.killpg(pid, signal.SIGINT)
for _ in range(20):
    processes = subprocess.run(['ps', '-g', str(pid), '-o', 'stat='], capture_output=True, text=True)
    if not any(line.strip() and not line.strip().startswith('Z') for line in processes.stdout.splitlines()):
        break
    time.sleep(1)
else:
    raise RuntimeError('Queue has not stopped; inspect before archiving or stopping the VM')
root = Path(metadata['root'])
metrics_path = root / 'outputs' / group / 'binary-42/training/metrics.jsonl'
metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
steps = [row for row in metrics if 'reward' in row]
note = {'state': 'interrupted_runtime_trial', 'reason': reasons[group],
        'completed_optimizer_steps': steps[-1]['step'] if steps else 0,
        'seconds_per_step': steps[-1]['seconds_per_step'] if steps else None,
        'included_in_controlled_results': False}
trial = root / 'outputs' / group
(trial / 'trial-disposition.json').write_text(json.dumps(note, indent=2) + '\n')
archive = Path('/content') / f'{group}-runtime-trial.tar.gz'
with tarfile.open(archive, 'w:gz') as handle:
    handle.add(trial, arcname=group)
    handle.add('/content/countdown-study.log', arcname='countdown-study.log')
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, **note}, indent=2))
