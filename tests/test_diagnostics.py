import unittest

from countdown_grpo.diagnostics import analyze_completion


class DiagnosticTests(unittest.TestCase):
    def test_ignores_claimed_rhs(self):
        r = analyze_completion('(9+31)*(67/9)=45', [9, 31, 67], 45)
        self.assertFalse(r['diagnostic_correct'])
        self.assertEqual(r['failure'], 'number_mismatch')

    def test_extracts_untagged_correct_answer(self):
        r = analyze_completion('Answer: (67-31)+9 = 45', [9, 31, 67], 45)
        self.assertTrue(r['diagnostic_correct'])
        self.assertFalse(r['strict_correct'])

    def test_does_not_search_for_correct_intermediate(self):
        r = analyze_completion('(67-31)+9 = 45\n(67+31)+9 = 107', [9, 31, 67], 45)
        self.assertEqual(r['failure'], 'wrong_value')

    def test_wrong_tag_prevents_fallback(self):
        r = analyze_completion('(67-31)+9 = 45\n<answer>67+31+9</answer>', [9, 31, 67], 45)
        self.assertEqual(r['failure'], 'wrong_value')

    def test_rejects_multiple_tags(self):
        r = analyze_completion('<answer>1+2+3</answer><answer>1*2*3</answer>', [1,2,3],6)
        self.assertFalse(r['diagnostic_correct'])

    def test_duplicate_omission(self):
        r = analyze_completion('<answer>3+33</answer>', [3,3,33],36)
        self.assertEqual(r['missing_numbers'], [3])

    def test_unsafe_syntax(self):
        for expression in ["__import__('os')", '2**3+4', '2//3+4', '2.0+3+4']:
            with self.subTest(expression=expression):
                self.assertEqual(analyze_completion(f'<answer>{expression}</answer>', [2,3,4],9)['failure'], 'invalid_syntax')

    def test_division_zero_and_fraction(self):
        self.assertEqual(analyze_completion('<answer>8/(3-3)</answer>',[8,3,3],24)['failure'],'division_by_zero')
        self.assertTrue(analyze_completion('<answer>8/(3-8/3)</answer>',[8,3,8,3],24)['diagnostic_correct'])

    def test_truncation_is_independent(self):
        r=analyze_completion('<answer>1+2+3</answer>',[1,2,3],6,truncated=True)
        self.assertTrue(r['diagnostic_correct'])
        self.assertTrue(r['truncated'])


if __name__ == '__main__':
    unittest.main()
