"""Archive the current compiled Overleaf report in W&B with source and figures."""
import argparse
import json
from pathlib import Path

from countdown_grpo.config import Config
from countdown_grpo.tracking import session


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='study-report-initial')
    args=parser.parse_args()
    import wandb
    with session(Config(),'report',args.name,'countdown-study-v1') as run:
        artifact=wandb.Artifact('countdown-study-report',type='report')
        for path in [Path('report/main.tex'),Path('report/main.pdf'),*Path('report/figures').glob('*')]:
            if not path.is_file():
                raise FileNotFoundError(path)
            artifact.add_file(str(path),name=str(path.relative_to('report')))
        artifact.add_file('docs/research-plan.md',name='research-plan.md')
        artifact.add_file('docs/study-ledger.md',name='study-ledger.md')
        run.log_artifact(artifact)
        run.summary['overleaf_url']='https://www.overleaf.com/project/6aca3d97baaac8402a980db8'
        Path('results/study/report-links.json').write_text(json.dumps({
            'overleaf':run.summary['overleaf_url'],'wandb_report_run':run.url,
            'artifact':'mohamedsobhi777/countdown-grpo/countdown-study-report:latest'},indent=2)+'\n')


if __name__=='__main__':
    main()
