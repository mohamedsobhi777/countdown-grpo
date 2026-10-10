"""Audit saved generations without querying a model or changing the strict reward."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from countdown_grpo.config import Config
from countdown_grpo.diagnostics import analyze_completion
from countdown_grpo.tracking import log_files, session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('results/2026-09-14'))
    parser.add_argument('--output', type=Path, default=Path('results/study/audit'))
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name in ('baseline', 'after'):
        path = args.source / f'{name}.json'
        report = json.loads(path.read_text())
        audited = []
        for index, row in enumerate(report['results']):
            diagnostic = analyze_completion(row['completion'], row['nums'], row['target'], row.get('truncated'))
            assert diagnostic['strict_correct'] == row['correct']
            assert diagnostic['strict_valid'] == row['valid']
            audited.append({'index': index, **row, 'diagnostic': diagnostic})
        n = len(audited)
        summary = {
            'samples': n,
            'strict_correct': sum(r['diagnostic']['strict_correct'] for r in audited),
            'diagnostic_correct': sum(r['diagnostic']['diagnostic_correct'] for r in audited),
            'format_only_misses': sum(r['diagnostic']['diagnostic_correct'] and not r['correct'] for r in audited),
            'failures': dict(Counter(r['diagnostic']['failure'] for r in audited)),
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'diagnostic_policy': 'declared-expression-v1',
            'truncation_available': any(r['diagnostic']['truncated'] is not None for r in audited),
        }
        (args.output / f'{name}.json').write_text(json.dumps({'summary': summary, 'results': audited}, indent=2)+'\n')
        reports[name] = summary
    (args.output / 'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    config = Config(**json.loads((args.source/'baseline.json').read_text())['config'])
    context = session(config, 'audit', 'archived-pilot-format-audit', 'countdown-study-v1') if args.wandb else nullcontext(None)
    with context as run:
        if run:
            import wandb
            for name, summary in reports.items():
                for key in ('strict_correct','diagnostic_correct','format_only_misses'):
                    run.summary[f'{name}/{key}'] = summary[key]
                run.log({f'{name}/failure_categories': wandb.Table(columns=['category','count'], data=list(summary['failures'].items()))})
            log_files(run, 'diagnostic-audit', 'evaluation', directory=args.output)
            (args.output / 'wandb.json').write_text(json.dumps({'url': run.url, 'id': run.id}, indent=2)+'\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
