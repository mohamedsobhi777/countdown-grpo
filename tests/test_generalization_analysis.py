import importlib.util
from pathlib import Path
import unittest

from countdown_grpo.config import Config
from countdown_grpo.diagnostics import analyze_completion

spec = importlib.util.spec_from_file_location('generalization', Path(__file__).resolve().parents[1] / 'scripts/analyze_generalization.py')
generalization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generalization)


class GeneralizationAuditTests(unittest.TestCase):
    def setUp(self):
        self.puzzles = [{'nums': [1, 2, 3], 'target': i} for i in range(1, 257)]
        rows = []
        for puzzle in self.puzzles:
            text = '<answer>1+2+3</answer>'
            diagnostic = analyze_completion(text, **puzzle, truncated=False)
            rows.append({**puzzle, 'completion': text, 'truncated': False,
                         'correct': diagnostic['strict_correct'], 'diagnostic': diagnostic})
        self.report = {'suite_sha256': 'frozen', 'results': rows, 'adapter': None,
                       'config': Config().to_dict(), 'samples': 256, 'solve_rate': 1/256,
                       'decoding': 'greedy', 'wandb_url': None}

    def test_valid_and_changed_hash(self):
        self.assertEqual(generalization.validate_report(self.report, self.puzzles, 'frozen', None)['correct'], 1)
        with self.assertRaisesRegex(ValueError, 'hash'):
            generalization.validate_report(self.report, self.puzzles, 'changed', None)

    def test_changed_order_and_budget(self):
        with self.assertRaisesRegex(ValueError, 'ordered suite'):
            generalization.validate_report(self.report, self.puzzles[::-1], 'frozen', None)
        self.report['config']['max_completion_length'] = 1024
        with self.assertRaisesRegex(ValueError, 'settings'):
            generalization.validate_report(self.report, self.puzzles, 'frozen', None)
