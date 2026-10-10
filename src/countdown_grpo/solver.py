"""Exact subset dynamic programming oracle for 3/4-number Countdown puzzles."""
from fractions import Fraction


def reachable_expressions(nums):
    if not 1 <= len(nums) <= 4 or any(type(n) is not int or not 0 < n <= 10000 for n in nums):
        raise ValueError('Expected one to four positive integers <= 10000')
    tables = {}
    for mask in range(1, 1 << len(nums)):
        if mask & (mask - 1) == 0:
            index = mask.bit_length() - 1
            tables[mask] = {Fraction(nums[index]): str(nums[index])}
            continue
        values = {}
        left = (mask - 1) & mask
        while left:
            right = mask ^ left
            if right and left < right:
                for a, x in tables[left].items():
                    for b, y in tables[right].items():
                        candidates = [(a+b,f'({x}+{y})'),(a*b,f'({x}*{y})'),
                                      (a-b,f'({x}-{y})'),(b-a,f'({y}-{x})')]
                        if b:
                            candidates.append((a/b,f'({x}/{y})'))
                        if a:
                            candidates.append((b/a,f'({y}/{x})'))
                        for value, expr in candidates:
                            values.setdefault(value, expr)
            left = (left - 1) & mask
        tables[mask] = values
    return tables[(1 << len(nums)) - 1]


def solve(nums, target):
    return reachable_expressions(nums).get(Fraction(target))
