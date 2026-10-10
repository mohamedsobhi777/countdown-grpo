"""Audit all three three-number-only runs and their five evaluation suites."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean, stdev

from analyze_generalization import validate_report
from countdown_grpo.config import Config
from countdown_grpo.data import puzzle_key
from countdown_grpo.statistics import paired_outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/three-only-local-v1'))
    parser.add_argument('--suites', type=Path, default=Path('data/suites/v1'))
    parser.add_argument('--pilot-evidence', type=Path, default=Path('results/study/generalization-pilot'))
    parser.add_argument('--destination', type=Path, default=Path('results/study/transfer'))
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.suites / 'manifest.json').read_text())
    suites, hashes = {}, {}
    for name in ('id_three', 'id_four', 'large_inputs', 'large_targets'):
        raw = (args.suites / f'{name}.json').read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if hashes[name] != manifest['suites'][name]['sha256']:
            raise ValueError(f'{name}: frozen suite changed')
        suites[name] = json.loads(raw)
    models, scores, comparisons = {}, {}, {}
    reference_training = reference_holdout = None
    for seed in (42, 43, 44):
        work = args.source / f'three-only-{seed}'
        if json.loads((work / 'status.json').read_text())['state'] != 'complete':
            raise ValueError(f'{seed}: transfer run is incomplete')
        expected = Config(seed=seed, split_seed=42, train_num_count=3)
        config = Config(**json.loads((work / 'training/config.json').read_text()))
        if config != replace(expected, output_dir=config.output_dir):
            raise ValueError(f'{seed}: training settings differ')
        train = json.loads((work / 'training/train-puzzles.json').read_text())
        holdout = json.loads((work / 'training/holdout.json').read_text())
        train_keys, holdout_keys = list(map(puzzle_key, train)), list(map(puzzle_key, holdout))
        if len(train) != 20000 or len(set(train_keys)) != len(train) or any(len(row['nums']) != 3 for row in train):
            raise ValueError(f'{seed}: invalid three-number-only training data')
        if reference_training is None:
            reference_training, reference_holdout = train_keys, holdout_keys
        elif train_keys != reference_training or holdout_keys != reference_holdout:
            raise ValueError(f'{seed}: puzzle lists/order differ between seeds')
        metrics = [json.loads(line) for line in (work / 'training/metrics.jsonl').read_text().splitlines()]
        steps = [r for r in metrics if 'reward' in r]
        if [r['step'] for r in steps] != list(range(1, 301)) or 'train_runtime' not in metrics[-1]:
            raise ValueError(f'{seed}: all 300 steps must finish')
        run = json.loads((work / 'training/wandb-run.json').read_text())
        models[str(seed)] = {'train_runtime_seconds': metrics[-1]['train_runtime'],
                             'training_url': run['url'], 'first_positive_reward_step': next((r['step'] for r in steps if r['reward'] > 0), None)}
        scores[str(seed)], comparisons[str(seed)] = {}, {}
        for name, puzzles in {'original_holdout': holdout, **suites}.items():
            if set(train_keys).intersection(map(puzzle_key, puzzles)):
                raise ValueError(f'{seed}/{name}: training overlap')
            path = work / ('evaluation.json' if name == 'original_holdout' else f'{name}.json')
            report = json.loads(path.read_text())
            scores[str(seed)][name] = validate_report(report, puzzles, hashes.get(name), str(Path(config.output_dir) / 'adapter'), expected)
            scores[str(seed)][name]['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if name != 'original_holdout':
                pilot = json.loads((args.pilot_evidence / f'pilot-{name}.json').read_text())
                validate_report(pilot, puzzles, hashes[name], 'outputs/countdown-20260914-01/training/adapter')
                comparisons[str(seed)][name] = paired_outcomes(pilot['results'], report['results'])
    grouped = {}
    for name in ('original_holdout', *suites):
        values = [scores[str(seed)][name]['accuracy'] for seed in (42, 43, 44)]
        grouped[name] = {'mean_accuracy': mean(values), 'sample_std_across_seeds': stdev(values)}
    summary = {'training_runs': models, 'per_seed': scores, 'seed_summary': grouped,
               'paired_mixed_pilot_to_three_only': comparisons,
               'interpretation': 'Each model trained on 20,000 three-number puzzles for 300 steps with a fixed split. Per-seed comparisons against the single mixed-arity pilot are descriptive, not a balanced multi-seed training-distribution ablation. Three seeds share the same test puzzles; do not pool observations as independent. Pretraining exposure and format transfer remain possible. Shifted suites are constructed, not difficulty-matched.'}
    args.destination.mkdir(parents=True, exist_ok=False)
    copied_hashes = {}
    for seed in (42, 43, 44):
        name = f'three-only-{seed}'
        for relative in ('status.json', 'evaluation.json', *(f'{s}.json' for s in suites),
                         'training/config.json', 'training/metrics.jsonl', 'training/train-puzzles.json',
                         'training/holdout.json', 'training/wandb-run.json'):
            source = args.source / name / relative
            target = args.destination / name / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied_hashes[f'{name}/{relative}'] = hashlib.sha256(source.read_bytes()).hexdigest()
    summary['source_sha256'] = copied_hashes
    (args.destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Three-number-only training and transfer', '', summary['interpretation'], '',
             '| Evaluation | Seed 42 | Seed 43 | Seed 44 | Mean ± seed SD |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name, aggregate in grouped.items():
        counts = ' | '.join(f"{scores[str(seed)][name]['correct']}/256" for seed in (42, 43, 44))
        lines.append(f"| {name} | {counts} | {100*aggregate['mean_accuracy']:.1f}% ± {100*aggregate['sample_std_across_seeds']:.1f} pp |")
    lines += ['', 'Per-checkpoint Wilson intervals, diagnostic scores, failure counts, links, and paired comparisons are in [summary.json](summary.json).']
    (args.destination / 'README.md').write_text('\n'.join(lines) + '\n')
    if args.wandb:
        from countdown_grpo.tracking import session, log_files
        with session(Config(), 'analysis', 'three-only-transfer-analysis', 'countdown-study-v1') as run:
            for name, row in grouped.items():
                run.summary[f'{name}/mean_accuracy'] = row['mean_accuracy']
                run.summary[f'{name}/seed_std'] = row['sample_std_across_seeds']
            log_files(run, 'transfer-analysis', 'evaluation', directory=args.destination)
            (args.destination / 'wandb.json').write_text(json.dumps({'url': run.url}, indent=2) + '\n')
    print(json.dumps(grouped, indent=2))


if __name__ == '__main__':
    main()
