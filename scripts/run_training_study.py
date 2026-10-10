"""Run a prespecified subset of study jobs sequentially, with W&B and evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--jobs', nargs='+', required=True,
                        help='Config stems, e.g. binary-43 shaped-42 three-only-42')
    parser.add_argument('--group',default='countdown-controlled-v1')
    args=parser.parse_args()
    root=Path('outputs')/args.group
    root.mkdir(parents=True,exist_ok=True)
    for name in args.jobs:
        config=Path('configs/study')/f'{name}.toml'
        if not config.is_file():
            raise FileNotFoundError(config)
        work=root/name
        work.mkdir(exist_ok=False)
        for stage,command in [
            ('train',['train','--output-dir',str(work/'training')]),
            ('evaluate',['evaluate','--adapter',str(work/'training/adapter'),'--result',str(work/'evaluation.json')]),
        ]:
            status={'job':name,'stage':stage,'state':'running'}
            (work/'status.json').write_text(json.dumps(status)+'\n')
            result=subprocess.run([sys.executable,'-u','-m','countdown_grpo.cli',*command,
                                   '--config',str(config),'--wandb','--group',args.group,
                                   '--run-name',f'{name}-{stage}'])
            if result.returncode:
                status.update(state='failed',exit_code=result.returncode)
                (work/'status.json').write_text(json.dumps(status)+'\n')
                raise SystemExit(result.returncode)
        (work/'status.json').write_text(json.dumps({'job':name,'state':'complete'})+'\n')


if __name__=='__main__':
    main()
