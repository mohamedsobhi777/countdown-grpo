"""Validate the complete eight-cell base/pilot generalization grid."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from countdown_grpo.config import Config
from countdown_grpo.data import puzzle_key
from countdown_grpo.diagnostics import analyze_completion
from countdown_grpo.statistics import paired_outcomes, wilson_interval


def validate_report(report, puzzles, checksum, adapter, expected_config=Config()):
    if report['suite_sha256'] != checksum:
        raise ValueError('Evaluation suite hash differs from frozen manifest')
    rows = report['results']
    keys = [puzzle_key(row) for row in rows]
    if (len(rows) != 256 or len(set(keys)) != len(keys)
            or keys != [puzzle_key(row) for row in puzzles] or report['samples'] != len(rows)):
        raise ValueError('Evaluation puzzles differ from the frozen ordered suite')
    if report['adapter'] != adapter or report['decoding'] != 'greedy':
        raise ValueError('Unexpected model adapter or decoding method')
    if Config(**report['config']) != expected_config:
        raise ValueError('Evaluation settings differ from the expected configuration')
    for row in rows:
        diagnostic = analyze_completion(row['completion'], row['nums'], row['target'], row['truncated'])
        if diagnostic != row['diagnostic'] or diagnostic['strict_correct'] != row['correct']:
            raise ValueError('Archived score differs from recomputed verifier')
    correct = sum(row['correct'] for row in rows)
    if correct / len(rows) != report['solve_rate']:
        raise ValueError('Aggregate accuracy differs from per-answer results')
    return {'samples': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
            'accuracy_wilson_95': wilson_interval(correct, len(rows)),
            'diagnostic_correct': sum(row['diagnostic']['diagnostic_correct'] for row in rows),
            'truncated': sum(row['truncated'] for row in rows),
            'failures': dict(Counter(row['diagnostic']['failure'] for row in rows)),
            'wandb_url': report['wandb_url'], 'suite_sha256': checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/generalization-pilot-v1'))
    parser.add_argument('--suites', type=Path, default=Path('data/suites/v1'))
    parser.add_argument('--training', type=Path, default=Path('outputs/countdown-20260914-01/training'))
    parser.add_argument('--destination', type=Path, default=Path('results/study/generalization-pilot'))
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    if json.loads((args.source / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('Generalization grid is incomplete')
    manifest = json.loads((args.suites / 'manifest.json').read_text())
    train = json.loads((args.training / 'train-puzzles.json').read_text())
    training_keys = set(map(puzzle_key, train))
    cells, reports, comparisons = {}, {}, {}
    for suite in ('id_three', 'id_four', 'large_inputs', 'large_targets'):
        raw = (args.suites / f'{suite}.json').read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        if checksum != manifest['suites'][suite]['sha256']:
            raise ValueError(f'{suite}: frozen input file differs from manifest')
        puzzles = json.loads(raw)
        if training_keys.intersection(map(puzzle_key, puzzles)):
            raise ValueError(f'{suite}: training overlap')
        for model in ('base', 'pilot'):
            name = f'{model}-{suite}'
            source = args.source / f'{name}.json'
            report = json.loads(source.read_text())
            cells[name] = validate_report(report, puzzles, checksum, str(args.training / 'adapter') if model == 'pilot' else None)
            cells[name]['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
            reports[name] = report
        comparisons[suite] = paired_outcomes(reports[f'base-{suite}']['results'], reports[f'pilot-{suite}']['results'])
    summary = {'scope': 'Base versus original mixed-arity pilot only; three-number-only transfer training is a separate pending study.',
               'cells': cells, 'paired_comparisons': comparisons,
               'interpretation': 'Each suite contains 256 puzzles. ID suites are arity-stratified held-out dataset samples; shifted suites are constructed, solver-verified distributions. Do not pool suites or claim general reasoning. Wilson intervals concern puzzle sampling for one checkpoint, not training-seed variation. Paired tests are exploratory and unadjusted.'}
    args.destination.mkdir(parents=True, exist_ok=False)
    for name in reports:
        shutil.copyfile(args.source / f'{name}.json', args.destination / f'{name}.json')
    shutil.copyfile(args.suites / 'manifest.json', args.destination / 'suite-manifest.json')
    (args.destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Pilot generalization', '', summary['scope'], '', summary['interpretation'], '',
             '| Suite | Base solved | GRPO solved | GRPO accuracy (95% Wilson) | GRPO truncated |',
             '| --- | ---: | ---: | ---: | ---: |']
    for suite in comparisons:
        base, pilot = cells[f'base-{suite}'], cells[f'pilot-{suite}']
        low, high = pilot['accuracy_wilson_95']
        lines.append(f"| {suite} | {base['correct']}/256 | [{pilot['correct']}/256]({pilot['wandb_url']}) | {100*pilot['accuracy']:.1f}% ({100*low:.1f}–{100*high:.1f}) | {pilot['truncated']}/256 |")
    lines += ['', 'Input generation rules and witnesses: [frozen suites](../../../data/suites/v1/README.md). Every output was re-scored, matched to its frozen suite, and checked against the pilot training puzzles before aggregation. See [summary.json](summary.json) for failure counts, hashes, and paired comparisons.']
    (args.destination / 'README.md').write_text('\n'.join(lines) + '\n')
    if args.wandb:
        from countdown_grpo.tracking import session, log_files
        with session(Config(), 'analysis', 'pilot-generalization-analysis', 'countdown-study-v1') as run:
            for name, cell in cells.items():
                run.summary[f'{name}/accuracy'] = cell['accuracy']
            log_files(run, 'generalization-analysis', 'evaluation', directory=args.destination)
            (args.destination / 'wandb.json').write_text(json.dumps({'url': run.url}, indent=2) + '\n')
    print(json.dumps(cells, indent=2))


if __name__ == '__main__':
    main()
