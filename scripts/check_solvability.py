import json
import argparse
from pathlib import Path
from countdown_grpo.solver import solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    source = Path('results/2026-09-14/after.json')
    report = json.loads(source.read_text())
    results = []
    for index, row in enumerate(report['results']):
        witness = solve(row['nums'], row['target'])
        results.append({'index':index,'nums':row['nums'],'target':row['target'],
                        'solvable':witness is not None,'witness':witness,'model_correct':row['correct']})
    summary = {'samples':len(results),'solvable':sum(r['solvable'] for r in results),
               'model_correct':sum(r['model_correct'] for r in results)}
    assert all(not r['model_correct'] or r['solvable'] for r in results)
    output = Path('results/study/audit/solvability.json')
    output.write_text(json.dumps({'summary':summary,'results':results},indent=2)+'\n')
    if args.wandb:
        from countdown_grpo.config import Config
        from countdown_grpo.tracking import session, log_files
        with session(Config(), 'audit', 'holdout-solvability-and-review', 'countdown-study-v1') as run:
            run.summary.update(summary)
            log_files(run, 'solvability-review', 'evaluation', files=[output, output.parent/'review.md'])
            (output.parent/'solvability-wandb.json').write_text(json.dumps({'url':run.url},indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
