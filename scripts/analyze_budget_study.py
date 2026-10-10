"""Validate and archive the completed six-cell output-budget experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from countdown_grpo.diagnostics import analyze_completion
from countdown_grpo.statistics import paired_outcomes, wilson_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('outputs/study-budgets-20261010'))
    parser.add_argument('--destination', type=Path, default=Path('results/study/budgets'))
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    reports, summaries = {}, {}
    expected = [f'{model}-{budget}' for budget in (256, 512, 1024) for model in ('base', 'adapter')]
    for name in expected:
        path = args.source / f'{name}.json'
        report = json.loads(path.read_text())  # Missing cells are an error, never silently omitted.
        rows = report['results']
        if report['samples'] != len(rows) or not rows:
            raise ValueError(f'{name}: invalid sample count')
        budget = int(name.split('-')[1])
        if report['config']['max_completion_length'] != budget or report['decoding'] != 'greedy':
            raise ValueError(f'{name}: unexpected evaluation settings')
        for row in rows:
            diagnostic = analyze_completion(row['completion'], row['nums'], row['target'], row['truncated'])
            if diagnostic != row['diagnostic'] or diagnostic['strict_correct'] != row['correct']:
                raise ValueError(f'{name}: scores differ from verifier')
        correct = sum(row['correct'] for row in rows)
        if correct / len(rows) != report['solve_rate']:
            raise ValueError(f'{name}: aggregate accuracy differs')
        reports[name] = report
        summaries[name] = {
            'samples': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
            'accuracy_wilson_95': wilson_interval(correct, len(rows)),
            'diagnostic_correct': sum(row['diagnostic']['diagnostic_correct'] for row in rows),
            'truncated': sum(row['truncated'] for row in rows),
            'mean_generated_tokens': sum(row['generated_tokens'] for row in rows) / len(rows),
            'generation_seconds': report['generation_seconds'],
            'failures': dict(Counter(row['diagnostic']['failure'] for row in rows)),
            'wandb_url': report['wandb_url'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    # Only token budget and adapter may vary within this grid.
    reference = dict(reports[expected[0]]['config'])
    reference.pop('max_completion_length')
    for name, report in reports.items():
        settings = dict(report['config'])
        settings.pop('max_completion_length')
        if settings != reference:
            raise ValueError(f'{name}: evaluation config differs beyond token budget')
        if bool(report['adapter']) != name.startswith('adapter'):
            raise ValueError(f'{name}: unexpected adapter identity')
    if len({r['adapter'] for n, r in reports.items() if n.startswith('adapter')}) != 1:
        raise ValueError('Adapter checkpoints differ')
    pairs = [(f'base-{b}', f'adapter-{b}') for b in (256, 512, 1024)]
    pairs += [(f'{m}-256', f'{m}-{b}') for m in ('base', 'adapter') for b in (512, 1024)]
    comparisons = {f'{a}_to_{b}': paired_outcomes(reports[a]['results'], reports[b]['results']) for a, b in pairs}
    summary = {'cells': summaries, 'paired_comparisons': comparisons,
               'uncertainty_scope': 'Wilson intervals describe puzzle-sampling uncertainty for a fixed checkpoint; they do not measure training-seed variation. Exact paired p-values are exploratory and unadjusted for multiple comparisons.'}
    args.destination.mkdir(parents=True, exist_ok=False)
    for name in expected:
        shutil.copyfile(args.source / f'{name}.json', args.destination / f'{name}.json')
    (args.destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Output-budget experiment', '', summary['uncertainty_scope'], '',
             '| Model / cap | Solved | Accuracy (95% Wilson) | Truncated | Mean tokens | Seconds |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, cell in summaries.items():
        lo, hi = cell['accuracy_wilson_95']
        lines.append(f"| [{name}]({cell['wandb_url']}) | {cell['correct']}/{cell['samples']} | {100*cell['accuracy']:.1f}% ({100*lo:.1f}–{100*hi:.1f}) | {cell['truncated']} | {cell['mean_generated_tokens']:.1f} | {cell['generation_seconds']:.1f} |")
    lines += ['', 'All cells use the same puzzles and greedy decoding. Timings cover batched generation and scoring, excluding model loading; they are not single-request latency.', '',
              'Raw outputs, canonical-puzzle paired gains/losses, unadjusted exact McNemar tests, and source hashes are included in [summary.json](summary.json).', '',
              'Wilson method: [NIST statistical reference](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).']
    (args.destination / 'README.md').write_text('\n'.join(lines) + '\n')
    if args.wandb:
        from countdown_grpo.config import Config
        from countdown_grpo.tracking import session, log_files
        with session(Config(), 'analysis', 'budget-study-analysis', 'countdown-study-v1') as run:
            for name, cell in summaries.items():
                run.summary[f'{name}/accuracy'] = cell['accuracy']
            log_files(run, 'budget-analysis', 'evaluation', directory=args.destination)
            (args.destination / 'wandb.json').write_text(json.dumps({'url': run.url}, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
