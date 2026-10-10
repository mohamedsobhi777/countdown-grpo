"""Plot audited pilot evidence, keeping strict accuracy and diagnostic outcomes distinct."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/study/generalization-pilot/summary.json'))
    parser.add_argument('--destination', type=Path, default=Path('results/study/generalization-pilot'))
    args = parser.parse_args()
    raw = args.source.read_bytes()
    cells = json.loads(raw)['cells']
    suites = ['id_three', 'id_four', 'large_inputs', 'large_targets']
    labels = ['Held-out: 3 numbers', 'Held-out: 4 numbers', 'Larger inputs*', 'Larger targets*']
    categories = [
        ('correct', 'Correct expression', '#0f766e'),
        ('number_mismatch', 'Numbers reused / omitted', '#c57524'),
        ('wrong_value', 'Wrong value', '#9f3b53'),
        ('invalid_syntax', 'Invalid syntax', '#6e68a8'),
        ('no_declared_expression', 'No declared expression', '#64748b'),
        ('ambiguous_or_incomplete_answer', 'Ambiguous / incomplete', '#b3bdc6'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout='constrained', gridspec_kw={'width_ratios':[1,1.15]})
    for model, label, color, offset in [('base','Base','#64748b',-.1),('pilot','After GRPO','#0f766e',.1)]:
        rows = [cells[f'{model}-{suite}'] for suite in suites]
        values = [100*r['accuracy'] for r in rows]
        errors = [[100*(r['accuracy']-r['accuracy_wilson_95'][0]) for r in rows],
                  [100*(r['accuracy_wilson_95'][1]-r['accuracy']) for r in rows]]
        axes[0].errorbar(values, [i+offset for i in range(4)], xerr=errors,
                         fmt='o', color=color, capsize=4, label=label)
    axes[0].set(title='Strict exact-solve accuracy', xlabel='Accuracy (%) · 95% Wilson intervals',
                xlim=(-3,100), yticks=range(4), yticklabels=labels, ylim=(3.5,-.5))
    axes[0].legend(frameon=False, loc='lower right')
    left = [0.0]*4
    for key, label, color in categories:
        values = []
        for suite in suites:
            cell = cells[f'pilot-{suite}']
            if sum(cell['failures'].values()) != cell['samples']:
                raise ValueError('Diagnostic counts do not cover every puzzle')
            if set(cell['failures']) - {c[0] for c in categories}:
                raise ValueError('Unmapped diagnostic category; update legend explicitly')
            values.append(100*cell['failures'].get(key,0)/cell['samples'])
        axes[1].barh(range(4), values, left=left, color=color, label=label, height=.65)
        left = [a+b for a,b in zip(left,values)]
    axes[1].set(title='After GRPO: declared-expression outcomes', xlabel='Share of all 256 answers (%)',
                xlim=(0,100), yticks=range(4), yticklabels=[], ylim=(3.5,-.5))
    axes[1].legend(loc='upper center', bbox_to_anchor=(.45,-.19), ncol=2, frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('One mixed-arity pilot checkpoint · greedy decoding · 256-token cap', fontsize=12)
    fig.supxlabel('* Constructed shifts, not difficulty-matched. Truncation can overlap any outcome and is not stacked.', fontsize=9)
    args.destination.mkdir(parents=True, exist_ok=True)
    for extension in ('png','svg','pdf'):
        fig.savefig(args.destination/f'generalization-comparison.{extension}', dpi=180)
    plt.close(fig)
    (args.destination/'plot-metadata.json').write_text(json.dumps({
        'source':str(args.source), 'source_sha256':hashlib.sha256(raw).hexdigest(),
        'matplotlib':matplotlib.__version__, 'script':'scripts/plot_generalization_pilot.py',
        'interval':'95% Wilson intervals per checkpoint; not training-seed variation',
        'outcomes':'Diagnostic categories include correct answers. Truncation is a separate overlapping flag.'
    },indent=2)+'\n')


if __name__ == '__main__':
    main()
