"""Plot complete audited studies, showing seeds separately from their mean/SD."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SEEDS = ('42', '43', '44')
COLORS = ('#0f766e', '#c57524', '#6e68a8')
SUITES = ('original_holdout', 'id_three', 'id_four', 'large_inputs', 'large_targets')


def checked_cells(summary, study):
    if study == 'controlled':
        names = ('binary', 'shaped')
        expected = {f'{name}-{seed}' for name in names for seed in SEEDS}
        if set(summary['runs']) != expected:
            raise ValueError('All six controlled runs are required')
        cells = {name: [summary['runs'][f'{name}-{seed}'] for seed in SEEDS] for name in names}
        groups = summary['reward_groups']
    else:
        names = SUITES
        if set(summary['per_seed']) != set(SEEDS):
            raise ValueError('All three transfer seeds are required')
        if any(set(summary['per_seed'][seed]) != set(SUITES) for seed in SEEDS):
            raise ValueError('All five evaluations per transfer seed are required')
        cells = {name: [summary['per_seed'][seed][name] for seed in SEEDS] for name in names}
        groups = summary['seed_summary']
    for name, rows in cells.items():
        for row in rows:
            if (row['samples'] != 256 or type(row['correct']) is not int
                    or not 0 <= row['correct'] <= 256
                    or not math.isclose(row['accuracy'], row['correct'] / 256, abs_tol=1e-12)):
                raise ValueError(f'{name}: inconsistent count/accuracy')
            low, high = row['accuracy_wilson_95']
            if not (math.isfinite(low) and math.isfinite(high)
                    and 0 <= low <= high <= 1
                    and low - 1e-12 <= row['accuracy'] <= high + 1e-12):
                raise ValueError(f'{name}: invalid checkpoint interval')
        values = [row['accuracy'] for row in rows]
        if (not math.isclose(groups[name]['mean_accuracy'], mean(values), abs_tol=1e-12)
                or not math.isclose(groups[name]['sample_std_across_seeds'], stdev(values), abs_tol=1e-12)):
            raise ValueError(f'{name}: aggregate does not match individual seeds')
    return names, cells, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', choices=('controlled', 'transfer'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    source = args.source or Path(f'results/study/{args.study}/summary.json')
    destination = args.destination or source.parent
    raw = source.read_bytes()
    names, cells, groups = checked_cells(json.loads(raw), args.study)
    labels = (['Binary correctness', 'Correctness + validity bonus'] if args.study == 'controlled'
              else ['Original holdout', 'Held-out: 3 numbers', 'Held-out: 4 numbers',
                    'Larger inputs*', 'Larger targets*'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained', sharey=True)
    for index, (seed, color) in enumerate(zip(SEEDS, COLORS)):
        rows = [cells[name][index] for name in names]
        values = [100 * row['accuracy'] for row in rows]
        # Endpoint roundoff can put Wilson's upper bound just below 1 at 256/256.
        errors = [[100 * max(0, row['accuracy'] - row['accuracy_wilson_95'][0]) for row in rows],
                  [100 * max(0, row['accuracy_wilson_95'][1] - row['accuracy']) for row in rows]]
        axes[0].errorbar(values, [i + (index - 1) * .18 for i in range(len(names))],
                         xerr=errors, fmt='o', color=color, capsize=3, label=f'Seed {seed}')
        axes[1].scatter(values, range(len(names)), color=color, marker='|', s=150, zorder=3)
    axes[1].errorbar([100 * groups[name]['mean_accuracy'] for name in names], range(len(names)),
                     xerr=[100 * groups[name]['sample_std_across_seeds'] for name in names],
                     fmt='D', color='#1f2937', capsize=5, label='Mean ± sample SD', zorder=2)
    axes[0].set_title('Individual checkpoints · 95% Wilson intervals')
    axes[1].set_title('Across three seeds · mean ± sample SD')
    axes[0].set(yticks=range(len(names)), yticklabels=labels, ylim=(len(names)-.5, -.5))
    for ax in axes:
        ax.set(xlim=(-5, 105), xlabel='Strict exact-solve accuracy (%)')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), frameon=False, ncol=3)
    # Mean +/- SD is descriptive and can extend beyond the probability bounds.
    axes[1].set_xlim(
        min(-5, min(100 * (groups[n]['mean_accuracy'] - groups[n]['sample_std_across_seeds']) for n in names) - 3),
        max(105, max(100 * (groups[n]['mean_accuracy'] + groups[n]['sample_std_across_seeds']) for n in names) + 3))
    title = ('Controlled reward comparison' if args.study == 'controlled'
             else 'Three-number-only training and transfer')
    fig.suptitle(f'{title} · 300 training steps · greedy decoding · 256-token cap', fontsize=12)
    note = 'Seeds share the same 256 puzzles per evaluation; observations are not pooled. SD is not a confidence interval.'
    if args.study == 'transfer':
        note += '\n* Constructed shifts, not difficulty-matched; this is not a balanced mixed- versus three-only training comparison.'
    fig.supxlabel(note, fontsize=9)
    destination.mkdir(parents=True, exist_ok=True)
    stem = f'{args.study}-seed-comparison'
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(destination / f'{stem}.{extension}', dpi=180)
    plt.close(fig)
    (destination / f'{stem}-metadata.json').write_text(json.dumps({
        'source': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'script': 'scripts/plot_seed_studies.py', 'matplotlib': matplotlib.__version__,
        'seeds': list(SEEDS), 'intervals': 'Left: per-checkpoint Wilson 95%. Right: sample SD across three seeds, not CI.',
        'scope': 'Consumes complete audited analyzer output; count/aggregate checks do not replace raw-output audits.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
