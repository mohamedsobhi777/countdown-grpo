"""Observe the bounded trial and the temporarily suspended main queue."""
import json
from pathlib import Path
import subprocess

main = json.loads(Path('/content/study-process.json').read_text())
trial_pid = Path('/content/kernel-training-trial.pid').read_text().strip()
print(subprocess.run(['ps','-p',f"{main['pid']},{trial_pid}",'-o','pid,etime,stat,args'],capture_output=True,text=True).stdout)
work = Path(main['root']) / 'outputs/kernel-fast-validation'
for name in ('/content/kernel-training-trial.log', str(work/'training.log')):
    path = Path(name)
    if path.exists():
        print(name, '\n'.join(path.read_text(errors='replace').splitlines()[-8:]))
metrics = work / 'training/metrics.jsonl'
if metrics.exists():
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    print('Trial metrics', [{k: r.get(k) for k in ('step','step_time','seconds_per_step','reward','train_runtime')} for r in rows])
if (work/'disposition.json').exists():
    print('Disposition', (work/'disposition.json').read_text())
