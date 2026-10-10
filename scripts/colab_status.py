"""Read-only observation of the remote study; safe to repeat after timeouts."""
import json
from pathlib import Path
import subprocess


state=json.loads(Path('/content/study-process.json').read_text())
print(subprocess.run(['ps','-p',str(state['pid']),'-o','pid,etime,stat,args'],capture_output=True,text=True).stdout)
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
log=Path(state.get('log','/content/countdown-study.log'))
if log.exists():
    print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
group_root = Path(state['root']) / 'outputs' / state['group']
for path in sorted(group_root.glob('*/status.json')):
    print(path.parent.name,json.loads(path.read_text()))
for path in sorted(group_root.glob('*/training/metrics.jsonl')):
    lines = path.read_text().splitlines()
    if not lines:
        continue
    row=json.loads(lines[-1])
    print(path.parents[1].name,{k:row.get(k) for k in ('step','reward','seconds_per_step','estimated_remaining_seconds')})
