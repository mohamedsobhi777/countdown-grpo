import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('budgets', Path(__file__).resolve().parents[1] / 'scripts/analyze_budget_study.py')
budgets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budgets)


class BudgetSchemaTests(unittest.TestCase):
    def test_only_historical_optional_defaults_are_normalized(self):
        old = {'seed': 42, 'max_completion_length': 256}
        new = {**old, 'max_completion_length': 512, 'split_seed': None, 'train_num_count': None}
        self.assertEqual(budgets.comparable_config(old), budgets.comparable_config(new))
        for change in ({'split_seed': 43}, {'train_num_count': 3}, {'seed': 43}, {'unknown': None}):
            self.assertNotEqual(budgets.comparable_config(old), budgets.comparable_config({**new, **change}))
        self.assertNotIn('split_seed', old)
