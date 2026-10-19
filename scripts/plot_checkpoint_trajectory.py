"""Plot held-out checkpoints and training samples over the laptop controlled grid."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = Path('results/study/checkpoint-trajectory/summary.json')
SEEDS = ('42', '43', '44')
COLORS = ('#0f766e', '#c57524', '#6e68a8')
STYLES = {'binary': dict(linestyle='-', marker='o'), 'shaped': dict(linestyle='--', marker='s')}
WIDTH = 25


def rolling(rows, key):
    steps, values = [], []
    for end in range(WIDTH, len(rows) + 1):
        selected = rows[end - WIDTH:end]
        steps.append(selected[-1]['step'])
        values.append(100 * sum(r[key] for r in selected) / (32 * WIDTH))
    return steps, values


def main():
    raw = SOURCE.read_bytes()
    summary = json.loads(raw)
    if not all(run['reproduction']['step300_matches_published'] for run in summary['runs'].values()):
        raise ValueError('Plot only a trajectory whose step-300 evaluations reproduce the published ones')
    base = summary['base']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained', sharex=True)
    for reward in ('binary', 'shaped'):
        for seed, color in zip(SEEDS, COLORS):
            run = summary['runs'][f'{reward}-{seed}']
            steps = [0] + [int(s) for s in run['heldout']]
            cells = [base] + list(run['heldout'].values())
            label = f'{reward.capitalize()} · seed {seed}'
            axes[0, 0].plot(steps, [100 * c['accuracy'] for c in cells], color=color, label=label, **STYLES[reward])
            axes[0, 1].plot(steps, [100 * c['bonus_only'] / c['samples'] for c in cells], color=color, **STYLES[reward])
            training = run['training_samples']['per_step']
            axes[1, 0].plot(*rolling(training, 'strict_correct'), color=color, linestyle=STYLES[reward]['linestyle'])
            axes[1, 1].plot(*rolling(training, 'bonus_only'), color=color, linestyle=STYLES[reward]['linestyle'])
    titles = (('Held-out: exact solves (greedy)', 'Held-out: valid but wrong (bonus-earning)'),
              (f'Training samples: correct ({WIDTH}-step rolling)', f'Training samples: valid but wrong ({WIDTH}-step rolling)'))
    for row in range(2):
        for column in range(2):
            ax = axes[row, column]
            ax.set_title(titles[row][column], fontsize=11)
            ax.set(ylim=(-3, 103), ylabel='% of answers')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(alpha=.15)
            ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set(xlabel='Training step', xlim=(-5, 305))
    axes[0, 0].legend(loc='upper left', frameon=False, ncol=2, fontsize=8.5)
    fig.suptitle('How the validity bonus takes over · laptop controlled grid · checkpoints every 50 steps', fontsize=12)
    fig.supxlabel('Top: 256 fixed held-out puzzles; step 0 is the base model. Bottom: 32 sampled completions per step, '
                  'scored with the training verifier.\nSolid = binary reward, dashed = +0.05 validity bonus. '
                  'Per-checkpoint Wilson intervals are in summary.json.', fontsize=9)
    destination = SOURCE.parent
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(destination / f'checkpoint-trajectory.{extension}', dpi=180)
    plt.close(fig)
    (destination / 'checkpoint-trajectory-metadata.json').write_text(json.dumps({
        'source': str(SOURCE), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'script': 'scripts/plot_checkpoint_trajectory.py', 'matplotlib': matplotlib.__version__,
        'rolling_window_steps': WIDTH}, indent=2) + '\n')


if __name__ == '__main__':
    main()
