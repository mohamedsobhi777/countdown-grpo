"""Read-only observation of the remote study; safe to repeat after timeouts."""
import json
from pathlib import Path
import subprocess


state=json.loads(Path('/content/study-process.json').read_text())
print(subprocess.run(['ps','-p',str(state['pid']),'-o','pid,etime,stat,args'],capture_output=True,text=True).stdout)
print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
log=Path('/content/countdown-study.log')
if log.exists():
    print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
for path in sorted(Path(state['root']).glob('outputs/controlled-l4-v1/*/status.json')):
    print(path.parent.name,json.loads(path.read_text()))
for path in sorted(Path(state['root']).glob('outputs/controlled-l4-v1/*/training/metrics.jsonl')):
    row=json.loads(path.read_text().splitlines()[-1])
    print(path.parents[1].name,{k:row.get(k) for k in ('step','reward','seconds_per_step','estimated_remaining_seconds')})
