import unittest
from countdown_grpo.solver import solve, reachable_expressions
from countdown_grpo.diagnostics import analyze_completion


class SolverTests(unittest.TestCase):
    def test_fraction_required(self):
        expression = solve([3,3,8,8],24)
        self.assertIsNotNone(expression)
        self.assertTrue(analyze_completion(f'<answer>{expression}</answer>',[3,3,8,8],24)['diagnostic_correct'])

    def test_impossible(self):
        self.assertIsNone(solve([1,1,1],100))

    def test_all_witnesses_obey_verifier(self):
        for value, expression in reachable_expressions([2,3,7]).items():
            if value.denominator == 1:
                self.assertTrue(analyze_completion(f'<answer>{expression}</answer>',[2,3,7],int(value))['diagnostic_correct'])

    def test_permutation_invariant_reachability(self):
        self.assertEqual(set(reachable_expressions([2,3,7])),set(reachable_expressions([7,2,3])))


if __name__ == '__main__':
    unittest.main()
