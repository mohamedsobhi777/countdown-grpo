"""Audit all six prespecified reward/seed runs before reporting their comparison."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean, stdev

from countdown_grpo.config import Config
from countdown_grpo.data import puzzle_key
from countdown_grpo.diagnostics import analyze_completion
from countdown_grpo.statistics import paired_outcomes, wilson_interval


def checked_run(path, reward, seed):
    provenance = json.loads((path / 'training/provenance.json').read_text())
    kernels = {'flash-linear-attention':'0.5.2', 'fla-core':'0.5.2', 'einops':'0.8.1', 'causal-conv1d':None}
    environment = provenance['config']
    requirements = (path/'training/requirements.txt').read_text()
    if environment['requirements_sha256'] != hashlib.sha256('\n'.join(sorted(requirements.splitlines())).encode()).hexdigest():
        raise ValueError(f'{path}: archived requirements differ from provenance')
    if (provenance['state'] != 'finished' or provenance['group'] != 'controlled-l4-fla-v1'
            or environment['optional_kernel_packages'] != kernels or environment['git_dirty'] is not False
            or provenance['hardware']['gpu'] != 'NVIDIA L4' or provenance['hardware']['gpu_count'] != 1):
        raise ValueError(f'{path}: backend, hardware, or source provenance differs from the controlled study')
    config = json.loads((path / 'training/config.json').read_text())
    expected = Config(seed=seed, split_seed=42, format_reward=reward).to_dict()
    expected['output_dir'] = config['output_dir']
    if config != expected:
        raise ValueError(f'{path}: training settings differ from the prespecified study')
    if json.loads((path / 'status.json').read_text())['state'] != 'complete':
        raise ValueError(f'{path}: run is incomplete')
    train = json.loads((path / 'training/train-puzzles.json').read_text())
    holdout = json.loads((path / 'training/holdout.json').read_text())
    report = json.loads((path / 'evaluation.json').read_text())
    rows = report['results']
    train_keys = [puzzle_key(row) for row in train]
    holdout_keys = [puzzle_key(row) for row in holdout]
    if len(train) != config['train_samples']:
        raise ValueError(f'{path}: training sample count differs')
    if len(set(train_keys)) != len(train_keys) or len(set(holdout_keys)) != len(holdout_keys):
        raise ValueError(f'{path}: duplicate puzzles')
    if set(train_keys) & set(holdout_keys) or holdout_keys != [puzzle_key(row) for row in rows]:
        raise ValueError(f'{path}: train/eval overlap or evaluation puzzle mismatch')
    evaluation_config = dict(report['config'])
    evaluation_config['output_dir'] = config['output_dir']
    if (evaluation_config != config or report['decoding'] != 'greedy'
            or report['adapter'] != str(Path(config['output_dir']) / 'adapter')):
        raise ValueError(f'{path}: evaluation settings differ from training manifest')
    if len(rows) != 256 or report['samples'] != len(rows):
        raise ValueError(f'{path}: incorrect holdout size')
    for row in rows:
        diagnostic = analyze_completion(row['completion'], row['nums'], row['target'], row['truncated'])
        if diagnostic != row['diagnostic'] or diagnostic['strict_correct'] != row['correct']:
            raise ValueError(f'{path}: archived scores differ from verifier')
    correct = sum(row['correct'] for row in rows)
    if correct / len(rows) != report['solve_rate']:
        raise ValueError(f'{path}: aggregate score differs')
    metrics = [json.loads(line) for line in (path / 'training/metrics.jsonl').read_text().splitlines()]
    steps = [row for row in metrics if 'reward' in row]
    if [row['step'] for row in steps] != list(range(1, 301)) or 'train_runtime' not in metrics[-1]:
        raise ValueError(f'{path}: training did not finish all 300 steps')
    run = json.loads((path / 'training/wandb-run.json').read_text())
    summary = {'seed': seed, 'format_reward': reward, 'correct': correct, 'samples': len(rows),
               'accuracy': correct / len(rows), 'accuracy_wilson_95': wilson_interval(correct, len(rows)),
               'diagnostic_correct': sum(r['diagnostic']['diagnostic_correct'] for r in rows),
               'failures': dict(Counter(r['diagnostic']['failure'] for r in rows)),
               'truncated': sum(r['truncated'] for r in rows),
               'train_runtime_seconds': metrics[-1]['train_runtime'],
               'first_positive_reward_step': next((r['step'] for r in steps if r['reward'] > 0), None),
               'last_30_steps_mean_reward': mean(r['reward'] for r in steps[-30:]),
               'training_url': f"https://wandb.ai/{run['entity']}/{run['project']}/runs/{run['id']}",
               'evaluation_url': report['wandb_url'],
               'adapter_artifact': f"{run['entity']}/{run['project']}/{run['id']}-adapter:latest"}
    summary['environment'] = environment
    summary['hardware'] = provenance['hardware']
    return summary, rows, train_keys, holdout_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--destination', type=Path, default=Path('results/study/controlled'))
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    summaries, records = {}, {}
    reference_train = reference_holdout = None
    reference_environment = None
    for reward_name, reward in [('binary', 0.0), ('shaped', 0.05)]:
        for seed in (42, 43, 44):
            name = f'{reward_name}-{seed}'
            summary, rows, train, holdout = checked_run(args.source / name, reward, seed)
            environment = summary['environment']
            if reference_environment is None:
                reference_environment = environment
            elif environment != reference_environment:
                raise ValueError(f'{name}: source commit, Python, packages, or optional kernels differ between runs')
            if reference_train is None:
                reference_train, reference_holdout = train, holdout
            elif train != reference_train or holdout != reference_holdout:
                raise ValueError(f'{name}: puzzle lists/order differ between seeds or rewards')
            summaries[name], records[name] = summary, rows
    comparisons = {str(seed): paired_outcomes(records[f'binary-{seed}'], records[f'shaped-{seed}']) for seed in (42, 43, 44)}
    grouped = {}
    for name in ('binary', 'shaped'):
        values = [summaries[f'{name}-{s}']['accuracy'] for s in (42, 43, 44)]
        grouped[name] = {'mean_accuracy': mean(values), 'sample_std_across_seeds': stdev(values), 'seeds': [42, 43, 44]}
    differences = [r['accuracy_difference'] for r in comparisons.values()]
    summary = {'runs': summaries, 'reward_groups': grouped, 'paired_by_seed': comparisons,
               'mean_paired_accuracy_difference': mean(differences),
               'sample_std_paired_accuracy_difference': stdev(differences),
               'interpretation': 'Three training seeds share one held-out puzzle set. Do not pool 768 observations as independent puzzles. Seed standard deviations and per-checkpoint Wilson intervals describe different uncertainty. Paired exact p-values are exploratory and unadjusted. Shaped reward values are not directly comparable to binary reward values.'}
    args.destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in summaries:
        destination = args.destination / name
        destination.mkdir()
        for relative in ['evaluation.json', 'status.json', 'collection.json', 'training/config.json', 'training/metrics.jsonl',
                         'training/wandb-run.json', 'training/train-puzzles.json', 'training/holdout.json',
                         'training/provenance.json', 'training/requirements.txt']:
            source = args.source / name / relative
            target = destination / relative
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            hashes[f'{name}/{relative}'] = hashlib.sha256(source.read_bytes()).hexdigest()
    summary['source_sha256'] = hashes
    (args.destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Controlled reward study', '', summary['interpretation'], '',
             '| Reward / seed | Solved | Accuracy | Training minutes |', '| --- | ---: | ---: | ---: |']
    for name, row in summaries.items():
        lines.append(f"| [{name}]({row['training_url']}) | {row['correct']}/256 | {100*row['accuracy']:.1f}% | {row['train_runtime_seconds']/60:.1f} |")
    lines += ['', 'Reward-group means, sample standard deviations, per-seed paired comparisons, checkpoint links, and source hashes: [summary.json](summary.json).', '',
              'All six runs passed configuration, 300-step completion, score recomputation, duplicate/overlap, identical ordered puzzle-list, single-L4 hardware, pinned FLA backend, and identical clean source/Python/package checks. Exported W&B provenance is archived with each run.']
    (args.destination / 'README.md').write_text('\n'.join(lines) + '\n')
    if args.wandb:
        from countdown_grpo.tracking import session, log_files
        with session(Config(), 'analysis', 'controlled-study-analysis', 'countdown-study-v1') as run:
            for name, row in grouped.items():
                run.summary[f'{name}/mean_accuracy'] = row['mean_accuracy']
                run.summary[f'{name}/seed_std'] = row['sample_std_across_seeds']
            log_files(run, 'controlled-analysis', 'evaluation', directory=args.destination)
            (args.destination / 'wandb.json').write_text(json.dumps({'url': run.url}, indent=2) + '\n')
    print(json.dumps(grouped, indent=2))


if __name__ == '__main__':
    main()
