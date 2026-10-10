"""Conservative post-hoc diagnostics; never used as the training reward.

Use the sole answer block when present, otherwise the LAST standalone arithmetic
line (optionally prefixed 'Answer:' or 'Final answer:'). For an equation, evaluate
only its leftmost expression, not its claimed right-hand value. No target-directed
search, selection among multiple solutions, or fallback from a wrong tagged answer.
"""
import ast
from collections import Counter
from fractions import Fraction
import re

from .reward import score_completion


def extract_declared_expression(text):
    opens, closes = text.count("<answer>"), text.count("</answer>")
    if opens or closes:
        if opens != 1 or closes != 1:
            return None, "answer_block", "ambiguous_or_incomplete_answer"
        match = re.search(r"<answer>([^<>]+)</answer>", text)
        if not match:
            return None, "answer_block", "invalid_answer_block"
        return match.group(1).strip().split("=")[0].strip(), "answer_block", None
    candidates = []
    for line in text.splitlines():
        line = re.sub(r"^(?:final answer|answer)\s*:\s*", "", line.strip(), flags=re.I)
        line = line.strip("`$").strip()
        if len(line) <= 256 and re.fullmatch(r"[0-9+*/()\s=.-]+", line) and re.search(r"[+*/-]", line):
            candidates.append(line.split("=")[0].strip())
    if not candidates:
        return None, "none", "no_declared_expression"
    return candidates[-1], "last_arithmetic_line", None


def analyze_completion(text, nums, target, truncated=None):
    strict = score_completion(text, nums, target)
    expression, source, error = extract_declared_expression(text)
    result = {"strict_correct": strict.correct, "strict_valid": strict.valid,
              "diagnostic_correct": False, "diagnostic_valid": False,
              "expression": expression, "answer_source": source,
              "failure": error, "truncated": truncated,
              "missing_numbers": [], "extra_numbers": []}
    if expression is None:
        return result
    if len(expression) > 128 or not re.fullmatch(r"[0-9+*/()\s-]+", expression):
        result["failure"] = "invalid_syntax"
        return result
    used = []

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and 0 < node.value <= 10000:
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
        raise ValueError("Unsupported arithmetic")

    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 32:
            raise ValueError("Too many nodes")
        value = visit(tree.body)
        result["missing_numbers"] = list((Counter(nums) - Counter(used)).elements())
        result["extra_numbers"] = list((Counter(used) - Counter(nums)).elements())
        if result["missing_numbers"] or result["extra_numbers"]:
            result["failure"] = "number_mismatch"
        else:
            result["diagnostic_valid"] = True
            result["diagnostic_correct"] = value == Fraction(target)
            result["failure"] = "correct" if result["diagnostic_correct"] else "wrong_value"
            result["value"] = str(value)
    except ZeroDivisionError:
        result["failure"] = "division_by_zero"
    except (SyntaxError, ValueError, OverflowError, RecursionError):
        result["failure"] = "invalid_syntax"
    return result
