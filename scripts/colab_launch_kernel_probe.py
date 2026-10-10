"""Install a separate optional-kernel directory and launch a bounded probe."""
import os
from pathlib import Path
import subprocess

directory = Path('/content/countdown-kernel-probe')
if directory.exists():
    raise RuntimeError('Probe directory already exists; inspect the existing probe before restarting')
command = (
    'uv pip install --target /content/countdown-kernel-probe --no-deps '
    'flash-linear-attention==0.5.2 fla-core==0.5.2 einops==0.8.1 && '
    'PYTHONPATH=/content/countdown-kernel-probe timeout 300 '
    '.venv/bin/python -u /content/benchmark_delta_kernels.py'
)
with Path('/content/kernel-probe.log').open('w') as log:
    probe = subprocess.Popen(['bash','-lc',command], cwd='/content/countdown-grpo',
                             env={**os.environ, 'OMP_NUM_THREADS': '8'},
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
Path('/content/kernel-probe.pid').write_text(str(probe.pid))
print('Started isolated kernel probe', probe.pid)
