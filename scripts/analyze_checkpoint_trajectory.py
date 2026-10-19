"""Training-dynamics follow-up to the laptop controlled grid.

`fetch` caches the 32 sampled completions that each training step logged to W&B.
`analyze` re-scores them, checks recomputed rewards against the logged ones,
combines them with the held-out checkpoint evaluations from
`run_checkpoint_trajectory.py`, enforces the step-300 reproduction gate, and
writes results/study/checkpoint-trajectory.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

from countdown_grpo.diagnostics import analyze_completion
from countdown_grpo.reward import score_completion
from countdown_grpo.statistics import wilson_interval

SOURCE = Path('results/study/controlled-local')
WORK = Path('outputs/checkpoint-trajectory-v1')
DESTINATION = Path('results/study/checkpoint-trajectory')
JOBS = [f'{reward}-{seed}' for seed in (42, 43, 44) for reward in ('binary', 'shaped')]
CHECKPOINTS = (50, 100, 150, 200, 250, 300)
TRAINING_STEPS = 300
SAMPLES_PER_STEP = 32
PROMPT = re.compile(r'Use the numbers \[([\d,\s]+)\] exactly once each to make (\d+)\.')
OUTCOMES = ('correct', 'wrong_value', 'number_mismatch', 'invalid_syntax', 'division_by_zero',
            'no_declared_expression', 'ambiguous_or_incomplete_answer', 'invalid_answer_block')


def parse_prompt(prompt):
    match = PROMPT.search(prompt)
    if not match:
        raise ValueError('Prompt does not state the puzzle')
    return [int(n) for n in match.group(1).split(',')], int(match.group(2))


def rescore(prompt, completion, format_reward):
    """Strict reward exactly as in training, plus the diagnostic outcome."""
    nums, target = parse_prompt(prompt)
    score = score_completion(completion, nums, target)
    # Training samples carry no token metadata, so truncation is unknown here.
    diagnostic = analyze_completion(completion, nums, target, None)
    return {'reward': 1.0 if score.correct else format_reward if score.valid else 0.0,
            'strict_correct': score.correct, 'bonus_only': score.valid and not score.correct,
            'outcome': outcome(diagnostic)}


def outcome(diagnostic):
    label = 'correct' if diagnostic['diagnostic_correct'] else diagnostic['failure']
    if label not in OUTCOMES:
        raise ValueError(f'Unknown diagnostic outcome {label!r}')
    return label


def reproduction_mismatches(published, replay):
    """Indices where a re-evaluation differs from the published evaluation."""
    if len(published) != len(replay):
        return ['length']
    keys = ('nums', 'target', 'completion', 'correct', 'valid', 'truncated')
    return [i for i, (a, b) in enumerate(zip(published, replay)) if any(a[k] != b[k] for k in keys)]


def first_step(series, predicate):
    return next((step for step, value in series if predicate(value)), None)


def summarize_evaluation(report):
    results = report['results']
    counts = Counter(outcome(r['diagnostic']) for r in results)
    correct = sum(r['correct'] for r in results)
    low, high = wilson_interval(correct, len(results))
    return {'strict_correct': correct, 'samples': len(results), 'accuracy': correct / len(results),
            'wilson_95': [low, high], 'diagnostic_correct': counts['correct'],
            'strict_valid': sum(r['valid'] for r in results),
            'bonus_only': sum(r['valid'] and not r['correct'] for r in results),
            'truncated': sum(r['truncated'] for r in results),
            'mean_generated_tokens': sum(r['generated_tokens'] for r in results) / len(results),
            'outcomes': {k: counts[k] for k in OUTCOMES}}


def fetch(args):
    import wandb
    api = wandb.Api(timeout=60)

    def download(job, index):
        reference = json.loads((SOURCE / job / 'training/wandb-run.json').read_text())
        path = WORK / 'training-samples' / job / f'step-{index + 1:03d}.json'
        if path.exists():
            return
        artifact = api.artifact(f"{reference['entity']}/{reference['project']}/run-{reference['id']}-completions:v{index}")
        if artifact.logged_by().id != reference['id']:
            raise ValueError(f'{artifact.name} was not logged by {reference["id"]}')
        table = json.loads(Path(artifact.get_entry('completions.table.json').download(
            root=str(WORK / 'downloads' / job / str(index)))).read_text())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table))

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(download, job, i) for job in args.jobs for i in range(TRAINING_STEPS)]
        for done, future in enumerate(futures, 1):
            future.result()
            if done % 100 == 0:
                print(f'{done}/{len(futures)} tables', flush=True)


def training_trajectory(job, format_reward):
    rows, mismatched = [], 0
    for step in range(1, TRAINING_STEPS + 1):
        table = json.loads((WORK / 'training-samples' / job / f'step-{step:03d}.json').read_text())
        columns = table['columns']
        samples = [dict(zip(columns, row)) for row in table['data']]
        if len(samples) != SAMPLES_PER_STEP or {s['step'] for s in samples} != {step}:
            raise ValueError(f'{job} step {step}: expected {SAMPLES_PER_STEP} samples from that step')
        scored = [rescore(s['prompt'], s['completion'], format_reward) for s in samples]
        mismatched += sum(abs(x['reward'] - s['countdown_reward']) > 1e-6 for x, s in zip(scored, samples))
        counts = Counter(x['outcome'] for x in scored)
        rows.append({'step': step, 'strict_correct': sum(x['strict_correct'] for x in scored),
                     'bonus_only': sum(x['bonus_only'] for x in scored),
                     'outcomes': {k: counts[k] for k in OUTCOMES}})
    return rows, mismatched


def window(rows, key, end, width=25):
    selected = [r for r in rows if end - width < r['step'] <= end]
    return sum(r[key] for r in selected) / (SAMPLES_PER_STEP * len(selected))


def analyze(args):
    summary = {'protocol': 'greedy, 256-token cap, original 256-puzzle holdout, declared-expression-v1',
               'source_grid': 'controlled-local-v2', 'checkpoints': list(CHECKPOINTS), 'runs': {}}
    base = json.loads((WORK / 'base/evaluation.json').read_text())
    summary['base'] = summarize_evaluation(base)
    holdout = [(r['nums'], r['target']) for r in base['results']]
    for job in JOBS:
        reward, seed = job.split('-')
        format_reward = 0.0 if reward == 'binary' else 0.05
        published = json.loads((SOURCE / job / 'evaluation.json').read_text())['results']
        trajectory = {}
        for step in CHECKPOINTS:
            report = json.loads((WORK / job / f'step-{step}' / 'evaluation.json').read_text())
            if [(r['nums'], r['target']) for r in report['results']] != holdout:
                raise ValueError(f'{job} step {step}: puzzle list differs from the base evaluation')
            trajectory[step] = {**summarize_evaluation(report), 'wandb_url': report['wandb_url'],
                                'sha256': hashlib.sha256((WORK / job / f'step-{step}' / 'evaluation.json').read_bytes()).hexdigest()}
        replay = json.loads((WORK / job / 'step-300/evaluation.json').read_text())['results']
        mismatches = reproduction_mismatches(published, replay)
        training, reward_mismatches = training_trajectory(job, format_reward)
        if reward_mismatches:
            raise ValueError(f'{job}: {reward_mismatches} recomputed training rewards differ from logged rewards')
        metrics = [json.loads(line) for line in (SOURCE / job / 'training/metrics.jsonl').read_text().splitlines()]
        metrics = {m['step']: m for m in metrics if 'step' in m and 'completions/clipped_ratio' in m}
        for row in training:
            row['clipped_ratio'] = metrics[row['step']]['completions/clipped_ratio']
            row['mean_length'] = metrics[row['step']]['completions/mean_length']
        series = [(r['step'], r) for r in training]
        accuracy = [(0, summary['base']['accuracy'])] + [(s, trajectory[s]['accuracy']) for s in CHECKPOINTS]
        peak_step, peak = max(accuracy, key=lambda item: (item[1], -item[0]))
        summary['runs'][job] = {
            'reward': reward, 'seed': int(seed), 'heldout': {str(k): v for k, v in trajectory.items()},
            'reproduction': {'step300_matches_published': not mismatches, 'mismatched_indices': mismatches},
            'training_samples': {'recomputed_rewards_match_logged': True, 'per_step': training},
            'milestones': {
                'peak_heldout_step': peak_step, 'peak_heldout_accuracy': peak,
                'final_minus_peak_pp': 100 * (trajectory[300]['accuracy'] - peak),
                'first_training_step_with_correct_sample': first_step(series, lambda r: r['strict_correct'] > 0),
                'first_training_step_bonus_majority': first_step(series, lambda r: r['bonus_only'] > SAMPLES_PER_STEP / 2),
                'training_correct_rate_last_25_steps': window(training, 'strict_correct', TRAINING_STEPS),
                'training_bonus_only_rate_last_25_steps': window(training, 'bonus_only', TRAINING_STEPS),
            }}
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for job in JOBS:
        for step in CHECKPOINTS:
            target = DESTINATION / job / f'step-{step}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((WORK / job / f'step-{step}' / 'evaluation.json').read_bytes())
    (DESTINATION / 'base.json').write_bytes((WORK / 'base/evaluation.json').read_bytes())
    print(json.dumps({job: {**run['milestones'], 'reproduced': run['reproduction']['step300_matches_published'],
                            'heldout': [run['heldout'][str(s)]['strict_correct'] for s in CHECKPOINTS]}
                      for job, run in summary['runs'].items()}, indent=2))
    if args.wandb:
        log_to_wandb(summary)


def log_to_wandb(summary):
    import wandb
    from countdown_grpo.config import Config
    from countdown_grpo.tracking import log_files, session
    with session(Config(), 'analysis', 'checkpoint-trajectory-analysis', 'checkpoint-trajectory-v1') as run:
        columns = ['job', 'step', 'strict_correct', 'bonus_only', 'truncated', 'mean_generated_tokens']
        table = wandb.Table(columns=columns, data=[
            [job, int(step), cell['strict_correct'], cell['bonus_only'], cell['truncated'], cell['mean_generated_tokens']]
            for job, data in summary['runs'].items() for step, cell in data['heldout'].items()])
        run.log({'checkpoint_trajectory': table})
        run.summary.update({f'{job}/reproduced': data['reproduction']['step300_matches_published']
                            for job, data in summary['runs'].items()})
        log_files(run, 'checkpoint-trajectory', 'analysis', directory=DESTINATION)
        summary['wandb_url'] = run.url
    (DESTINATION / 'wandb.json').write_text(json.dumps({'analysis_run': summary['wandb_url']}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    fetch_parser = commands.add_parser('fetch')
    fetch_parser.add_argument('--jobs', nargs='+', default=JOBS)
    fetch_parser.add_argument('--workers', type=int, default=8)
    analyze_parser = commands.add_parser('analyze')
    analyze_parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    fetch(args) if args.command == 'fetch' else analyze(args)


if __name__ == '__main__':
    main()
