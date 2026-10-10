"""Persistently run the prespecified 2-model x 3-output-budget evaluation grid."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default='study-budgets-20261010')
    parser.add_argument('--adapter', default='outputs/countdown-20260914-01/training/adapter')
    args = parser.parse_args()
    root = Path('outputs') / args.group
    root.mkdir(parents=True, exist_ok=False)
    jobs = [(model, budget) for budget in (256,512,1024) for model in ('base','adapter')]
    completed = []
    for model, budget in jobs:
        name = f'{model}-{budget}'
        output = root / f'{name}.json'
        state = {'state':'running','job':name,'completed':completed,'total_jobs':len(jobs)}
        (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        command = [sys.executable,'-u','-m','countdown_grpo.cli','evaluate',
                   '--completion-length',str(budget),'--result',str(output),
                   '--wandb','--group',args.group,'--run-name',name]
        if model == 'adapter':
            command.extend(['--adapter',args.adapter])
        code = subprocess.run(command).returncode
        if code:
            state.update(state='failed',exit_code=code)
            (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
            raise SystemExit(code)
        completed.append(name)
    (root/'status.json').write_text(json.dumps({'state':'complete','completed':completed},indent=2)+'\n')


if __name__ == '__main__':
    main()
