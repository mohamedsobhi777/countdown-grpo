"""Start the consistent FLA study after archiving the stopped fallback trial."""
import json
from pathlib import Path
import subprocess

old = json.loads(Path('/content/study-process.json').read_text())
if old['group'] != 'controlled-l4-v1':
    raise RuntimeError('Expected the stopped fallback trial, not another live study')
states = subprocess.run(['ps','-g',str(old['pid']),'-o','stat='],capture_output=True,text=True).stdout.splitlines()
if any(state.strip() and not state.strip().startswith('Z') for state in states):
    raise RuntimeError('Fallback process group is still active')
if not Path('/content/controlled-l4-v1-runtime-trial.tar.gz').exists():
    raise RuntimeError('Archive the fallback evidence first')
if 'environment' not in globals() or not environment.get('WANDB_API_KEY'):
    raise RuntimeError('Expected the authenticated bootstrap environment in this kernel')
root = Path(old['root'])
if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
    raise RuntimeError('Remote checkout contains changes; inspect before pulling')
subprocess.run(['git','pull','--ff-only'],cwd=root,check=True)
group = 'controlled-l4-fla-v1'
if (root/'outputs'/group).exists():
    raise RuntimeError('Fast study output already exists; inspect before restarting')
command = (
    'uv sync --locked --extra train --extra fast && '
    '.venv/bin/python -u scripts/run_training_study.py '
    f'--group {group} --jobs binary-42 shaped-42 binary-43 shaped-43 binary-44 shaped-44'
)
with Path('/content/countdown-fast-study.log').open('w') as log:
    process = subprocess.Popen(['bash','-lc',command],cwd=root,env=environment,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path('/content/study-process.json').write_text(json.dumps({
    'pid': process.pid, 'root': str(root), 'group': group,
    'log': '/content/countdown-fast-study.log', 'backend': 'flash-linear-attention==0.5.2',
}))
print('Started consistent FLA study queue', process.pid)
