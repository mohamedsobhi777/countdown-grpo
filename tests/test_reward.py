import unittest

from countdown_grpo.reward import make_reward, score_completion


class RewardTests(unittest.TestCase):
    def test_exact_solution_with_scratchpad(self):
        score = score_completion("44 + 19 is 63.\n<answer>(44+19)+35</answer>", [44, 19, 35], 98)
        self.assertTrue(score.correct)

    def test_fractional_intermediates(self):
        self.assertTrue(score_completion("<answer>8/(3-8/3)</answer>", [8, 3, 8, 3], 24).correct)

    def test_negative_intermediates(self):
        self.assertTrue(score_completion("<answer>(2-5)*3</answer>", [2, 5, 3], -9).correct)

    def test_duplicate_multiplicity(self):
        self.assertTrue(score_completion("<answer>3+3+33</answer>", [3, 33, 3], 39).correct)
        self.assertFalse(score_completion("<answer>3+33</answer>", [3, 33, 3], 36).valid)

    def test_valid_but_wrong(self):
        score = score_completion("<answer>44+19-35</answer>", [44, 19, 35], 98)
        self.assertTrue(score.valid)
        self.assertFalse(score.correct)

    def test_invalid_and_exploit_attempts(self):
        cases = [
            "<answer>98</answer>", "98", "<answer>44+19+35=98</answer>",
            "<answer>44+19+35</answer><answer>98</answer>",
            "<answer><answer>44+19+35</answer></answer>",
            "<answer>44+19+35", "<answer>44**19+35</answer>",
            "<answer>44//19+35</answer>", "<answer>44%19+35</answer>",
            "<answer>44.0+19+35</answer>", "<answer>+44+19+35</answer>",
            "<answer>44+19+35+0</answer>", "<answer>__import__('os').system('id')</answer>",
            "<answer>" + "(" * 200 + "98" + ")" * 200 + "</answer>",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(score_completion(text, [44, 19, 35], 98).valid)

    def test_division_by_zero(self):
        self.assertFalse(score_completion("<answer>8/(3-3)</answer>", [8, 3, 3], 24).valid)

    def test_trl_string_and_chat_completions(self):
        completions = ["<answer>44+19+35</answer>", [{"role": "assistant", "content": "<answer>44+19-35</answer>"}]]
        self.assertEqual(make_reward()(completions, [[44, 19, 35]] * 2, [98] * 2), [1.0, 0.0])
        self.assertEqual(make_reward(0.05)(completions, [[44, 19, 35]] * 2, [98] * 2), [1.0, 0.05])

    def test_batch_mismatch_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            make_reward()(["anything"], [], [])


if __name__ == "__main__":
    unittest.main()
