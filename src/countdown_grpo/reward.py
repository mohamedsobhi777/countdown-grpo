"""Bounded AST interpretation with exact rational arithmetic; never executes code."""
import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import re


@dataclass(frozen=True)
class Score:
    valid: bool
    correct: bool
    expression: str | None = None


def score_completion(text, nums, target):
    if not isinstance(text, str) or len(text) > 32768:
        return Score(False, False)
    # Reject ambiguous multiple answers, nested tags, and partial answers.
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return Score(False, False)
    match = re.search(r"<answer>([^<>]+)</answer>", text)
    if not match:
        return Score(False, False)
    expression = match.group(1).strip()
    if not expression or len(expression) > 128 or not re.fullmatch(r"[0-9+*/()\s-]+", expression):
        return Score(False, False)
    used = []

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 < node.value <= 100:
            used.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("Only positive integer literals and + - * / are allowed")

    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 32:
            return Score(False, False)
        value = visit(tree.body)
        if Counter(used) != Counter(nums):
            return Score(False, False)
        return Score(True, value == Fraction(target), expression)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, OverflowError):
        return Score(False, False)


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    return "".join(message.get("content", "") for message in completion)


def make_reward(format_reward=0.0):
    def countdown_reward(completions, nums, target, **kwargs):
        rewards = []
        for completion, numbers, goal in zip(completions, nums, target, strict=True):
            score = score_completion(completion_text(completion), numbers, goal)
            rewards.append(1.0 if score.correct else format_reward if score.valid else 0.0)
        return rewards
    return countdown_reward
