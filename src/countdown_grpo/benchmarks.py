"""Deterministic, solvable distribution-shift suites with separate witnesses."""
from fractions import Fraction
import random

from .data import puzzle_key
from .solver import reachable_expressions


def shifted_suite(name, count=256, seed=20261010):
    if name not in ('large_inputs','large_targets'):
        raise ValueError('Unknown constructed distribution')
    if count < 2 or count % 2:
        raise ValueError('Use an even count to balance three/four-number puzzles')
    rng = random.Random(f'{seed}:{name}')
    rows, witnesses, seen = [], [], set()
    number_range = (101,200) if name == 'large_inputs' else (1,99)
    target_range = (10,100) if name == 'large_inputs' else (101,200)
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 100:
            raise RuntimeError('Could not construct enough distinct solvable puzzles')
        size = 3 if len(rows) < count // 2 else 4
        nums = [rng.randint(*number_range) for _ in range(size)]
        expressions = reachable_expressions(nums)
        targets = sorted(int(t) for t in expressions if t.denominator == 1 and target_range[0] <= t <= target_range[1])
        if not targets:
            continue
        target = rng.choice(targets)
        row = {'nums':nums,'target':target}
        key = puzzle_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        witnesses.append(expressions[Fraction(target)])
    order = list(range(count))
    rng.shuffle(order)
    return [rows[i] for i in order], [witnesses[i] for i in order]
