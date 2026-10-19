"""Training-sample re-scoring and reproduction gates for the trajectory follow-up."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('trajectory', Path(__file__).resolve().parents[1] / 'scripts/analyze_checkpoint_trajectory.py')
trajectory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trajectory)

PROMPT = ('user\nUse the numbers [42, 53, 68, 50] exactly once each to make 78. Use only +, -, *, / '
          'and parentheses.\nassistant\n')


class TrajectoryTests(unittest.TestCase):
    def test_parses_logged_prompt(self):
        self.assertEqual(trajectory.parse_prompt(PROMPT), ([42, 53, 68, 50], 78))
        with self.assertRaises(ValueError):
            trajectory.parse_prompt('no puzzle here')

    def test_rescore_matches_training_reward(self):
        correct = trajectory.rescore(PROMPT, '<answer>68 + 50 - 42 + 53 - 51</answer>', 0.05)
        self.assertEqual(correct['outcome'], 'number_mismatch')
        self.assertEqual(correct['reward'], 0.0)
        bonus = trajectory.rescore(PROMPT, '<answer>(42 / 53) * 68 / 50</answer>', 0.05)
        self.assertEqual((bonus['reward'], bonus['bonus_only'], bonus['outcome']), (0.05, True, 'wrong_value'))
        self.assertEqual(trajectory.rescore(PROMPT, '<answer>(42 / 53) * 68 / 50</answer>', 0.0)['reward'], 0.0)
        solved = trajectory.rescore(PROMPT, '<answer>(68 - 42) * (53 - 50)</answer>', 0.05)
        self.assertEqual((solved['reward'], solved['strict_correct'], solved['outcome']), (1.0, True, 'correct'))

    def test_reproduction_gate_reports_differences(self):
        row = {'nums': [1, 2], 'target': 3, 'completion': 'x', 'correct': False, 'valid': False, 'truncated': False}
        self.assertEqual(trajectory.reproduction_mismatches([row, row], [row, dict(row)]), [])
        self.assertEqual(trajectory.reproduction_mismatches([row, row], [row, {**row, 'completion': 'y'}]), [1])
        self.assertEqual(trajectory.reproduction_mismatches([row], [row, row]), ['length'])

    def test_first_step_and_unknown_outcome(self):
        self.assertEqual(trajectory.first_step([(1, 0), (2, 3), (3, 5)], lambda v: v > 2), 2)
        self.assertIsNone(trajectory.first_step([(1, 0)], lambda v: v > 2))
        with self.assertRaises(ValueError):
            trajectory.outcome({'diagnostic_correct': False, 'failure': 'unexpected'})


if __name__ == '__main__':
    unittest.main()
