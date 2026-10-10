"""Run via colab exec; provision the pinned repo environment and a training queue.

Authentication is read from a separately uploaded, mode-600 temporary file outside
the repository and removed immediately. No credential appears in this script or
notebook history. Inspect /content/countdown-study.log and study-process.json.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path('/content/countdown-grpo')
if root.exists():
    raise RuntimeError('Existing checkout; inspect the live process before restarting')
subprocess.run(['git','clone','https://github.com/mohamedsobhi777/countdown-grpo.git',str(root)],check=True)
credential = Path('/content/.countdown-wandb-key')
credential.chmod(0o600)
environment = dict(os.environ)
environment['WANDB_API_KEY'] = credential.read_text().strip()
credential.unlink()
environment.update(OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
command = (
    f'{sys.executable} -m pip install -q uv && '
    'uv sync --locked --extra train && '
    '.venv/bin/python -u scripts/run_training_study.py '
    '--group controlled-l4-v1 --jobs binary-42 shaped-42 binary-43 shaped-43 binary-44 shaped-44'
)
with open('/content/countdown-study.log','w') as log:
    process = subprocess.Popen(['bash','-lc',command],cwd=root,env=environment,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
Path('/content/study-process.json').write_text(json.dumps({'pid':process.pid,'root':str(root),'group':'controlled-l4-v1'}))
print('Started setup and training queue; PID',process.pid)
