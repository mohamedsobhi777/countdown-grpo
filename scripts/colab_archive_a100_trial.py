"""Run only on the A100 runtime trial after verifying the L4 queue is live.

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
if metadata['group'] != 'controlled-a100-v1':
    raise RuntimeError('This archival script is restricted to the A100 runtime trial')
pid = metadata['pid']
command_path = Path(f'/proc/{pid}/cmdline')
if command_path.exists():
    command = command_path.read_bytes().decode().replace('\0', ' ')
    if 'run_training_study.py' not in command or 'controlled-a100-v1' not in command:
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
metrics_path = root / 'outputs/controlled-a100-v1/binary-42/training/metrics.jsonl'
metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
steps = [row for row in metrics if 'reward' in row]
note = {'state': 'interrupted_runtime_trial', 'reason': 'Initial A100 throughput was only about 3% faster than L4 with the current backend. Retain the established L4 grid and release the A100.',
        'completed_optimizer_steps': steps[-1]['step'] if steps else 0,
        'seconds_per_step': steps[-1]['seconds_per_step'] if steps else None,
        'included_in_controlled_results': False}
trial = root / 'outputs/controlled-a100-v1'
(trial / 'trial-disposition.json').write_text(json.dumps(note, indent=2) + '\n')
archive = Path('/content/countdown-a100-runtime-trial.tar.gz')
with tarfile.open(archive, 'w:gz') as handle:
    handle.add(trial, arcname='controlled-a100-v1')
    handle.add('/content/countdown-study.log', arcname='countdown-study.log')
print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, **note}, indent=2))
