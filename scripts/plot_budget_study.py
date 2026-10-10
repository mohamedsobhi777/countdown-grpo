"""Plot the validated budget grid; requires Matplotlib (tested with 3.6.3)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('results/study/budgets/summary.json'))
    parser.add_argument('--destination', type=Path, default=Path('results/study/budgets'))
    args = parser.parse_args()
    cells = json.loads(args.source.read_text())['cells']
    budgets = [256, 512, 1024]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for model, label, color, marker in [('base', 'Base', '#64748b', 's'), ('adapter', 'After GRPO', '#0f766e', 'o')]:
        values = [cells[f'{model}-{b}'] for b in budgets]
        accuracy = [100 * row['accuracy'] for row in values]
        errors = [[100 * (r['accuracy'] - r['accuracy_wilson_95'][0]) for r in values],
                  [100 * (r['accuracy_wilson_95'][1] - r['accuracy']) for r in values]]
        axes[0].errorbar(budgets, accuracy, yerr=errors, capsize=4, color=color, marker=marker, label=label)
        axes[1].plot(budgets, [100 * r['truncated'] / r['samples'] for r in values], color=color, marker=marker)
        axes[2].plot(budgets, [r['generation_seconds'] / 60 for r in values], color=color, marker=marker)
    axes[0].set(title='Held-out exact solutions', ylabel='Accuracy (%)', ylim=(-2, 50))
    axes[0].legend(frameon=False)
    axes[1].set(title='Answers reaching the token cap', ylabel='Truncated (%)', ylim=(0, 100))
    axes[2].set(title='Evaluation wall time', ylabel='Minutes for 256 puzzles', ylim=(0, None))
    for ax in axes:
        ax.set(xlabel='Maximum new tokens', xticks=budgets)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.15)
    fig.suptitle('One checkpoint, identical 256 puzzles, greedy decoding', fontsize=12)
    args.destination.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(args.destination / f'budget-comparison.{suffix}', dpi=180)
    plt.close(fig)
    (args.destination / 'plot-metadata.json').write_text(json.dumps({
        'matplotlib': matplotlib.__version__, 'source': str(args.source),
        'script': 'scripts/plot_budget_study.py',
        'interval': '95% Wilson score, fixed checkpoint; not training-seed variation',
    }, indent=2) + '\n')
    print('Saved PNG, SVG, and PDF. Error bars: 95% Wilson intervals, not seed variation.')


if __name__ == '__main__':
    main()
