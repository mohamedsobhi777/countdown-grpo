"""Evaluate the base and pilot adapter on the frozen challenge suites."""
import json
from pathlib import Path
import subprocess
import sys


root=Path('outputs/generalization-pilot-v1')
root.mkdir(parents=True,exist_ok=False)
for model in ('base','pilot'):
    for suite in ('id_three','id_four','large_inputs','large_targets'):
        name=f'{model}-{suite}'
        (root/'status.json').write_text(json.dumps({'state':'running','job':name})+'\n')
        command=[sys.executable,'-u','-m','countdown_grpo.cli','evaluate',
                 '--puzzles',f'data/suites/v1/{suite}.json','--result',str(root/f'{name}.json'),
                 '--wandb','--group','generalization-pilot-v1','--run-name',name]
        if model=='pilot':
            command.extend(['--adapter','outputs/countdown-20260914-01/training/adapter'])
        result=subprocess.run(command)
        if result.returncode:
            (root/'status.json').write_text(json.dumps({'state':'failed','job':name,'exit_code':result.returncode})+'\n')
            raise SystemExit(result.returncode)
(root/'status.json').write_text(json.dumps({'state':'complete'})+'\n')
