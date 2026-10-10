"""Evaluation uncertainty; training-seed variation must be reported separately."""
import math
from statistics import NormalDist

from .data import puzzle_key


def wilson_interval(successes, total, confidence=0.95):
    """Wilson score interval for an individual checkpoint's binomial accuracy."""
    if not 0 <= successes <= total or total <= 0 or not 0 < confidence < 1:
        raise ValueError('Invalid counts or confidence')
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def paired_outcomes(before, after):
    """Match canonical puzzles; reject missing/duplicate inputs rather than zip."""
    def index(rows):
        mapped = {puzzle_key(row): row for row in rows}
        if not rows or len(mapped) != len(rows):
            raise ValueError('Empty or duplicate evaluation puzzles')
        if any(type(row['correct']) is not bool for row in rows):
            raise ValueError('Correctness must be Boolean')
        return mapped
    left, right = index(before), index(after)
    if left.keys() != right.keys():
        raise ValueError('Paired evaluations must contain identical puzzles')
    gains = sum(not left[k]['correct'] and right[k]['correct'] for k in left)
    losses = sum(left[k]['correct'] and not right[k]['correct'] for k in left)
    both = sum(left[k]['correct'] and right[k]['correct'] for k in left)
    # Exact two-sided binomial test on discordant pairs (exact McNemar).
    discordant = gains + losses
    p_value = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(gains, losses) + 1)) / 2**discordant)
    return {'samples': len(left), 'gained': gains, 'lost': losses,
            'both_correct': both, 'both_incorrect': len(left) - both - gains - losses,
            'accuracy_difference': (gains - losses) / len(left),
            'exact_mcnemar_p_unadjusted': p_value}
