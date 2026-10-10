from dataclasses import replace
import unittest

from countdown_grpo.config import Config, load_config
from countdown_grpo.data import puzzle_key, split_rows, render_prompt


class DataTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(eval_fraction=0.2, train_samples=1000, eval_samples=1000)
        self.rows = [{"target": n, "nums": [1, 2, 3]} for n in range(100)]

    def test_duplicate_and_permutation_isolation(self):
        rows = self.rows + [{"target": n, "nums": [3, 2, 1]} for n in range(100)] + self.rows
        train, test = split_rows(rows, self.config)
        train_keys, test_keys = set(map(puzzle_key, train)), set(map(puzzle_key, test))
        self.assertFalse(train_keys & test_keys)
        self.assertEqual(len(train) + len(test), 100)
        self.assertEqual(len(train_keys), len(train))
        self.assertEqual(len(test_keys), len(test))

    def test_membership_stable_under_reordering(self):
        a = split_rows(self.rows, self.config)
        b = split_rows(list(reversed(self.rows)), self.config)
        self.assertEqual(a, b)

    def test_training_limit_does_not_change_holdout(self):
        a = split_rows(self.rows, self.config)
        b = split_rows(self.rows, replace(self.config, train_samples=5))
        self.assertEqual(a[1], b[1])
        self.assertEqual(len(b[0]), 5)

    def test_empty_dataset_fails(self):
        with self.assertRaises(ValueError):
            split_rows([], self.config)

    def test_split_is_fixed_across_training_seeds(self):
        a=split_rows(self.rows, replace(self.config, seed=42, split_seed=42))
        b=split_rows(self.rows, replace(self.config, seed=43, split_seed=42))
        self.assertEqual(a,b)

    def test_training_arity_filter_does_not_change_evaluation(self):
        rows=self.rows+[{'target':n,'nums':[1,2,3,4]} for n in range(100)]
        a=split_rows(rows,self.config)
        b=split_rows(rows,replace(self.config,train_num_count=3))
        self.assertEqual(a[1],b[1])
        self.assertTrue(all(len(r['nums'])==3 for r in b[0]))

    def test_template_requests_generation_prefix(self):
        class Tokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                assert not tokenize and add_generation_prompt
                return messages[0]["content"]
        text = render_prompt(Tokenizer(), self.rows[0])
        self.assertIn("[1, 2, 3]", text)
        self.assertIn("<answer>", text)


class ConfigTests(unittest.TestCase):
    def test_committed_config(self):
        self.assertEqual(load_config("configs/local.toml"), Config())

    def test_invalid_hyperparameters(self):
        for change in [{"num_generations": 1}, {"gradient_accumulation_steps": 7},
                       {"max_steps": 0}, {"eval_fraction": 0}, {"format_reward": 0.5},
                       {"learning_rate": float("nan")}, {"lora_rank": -1}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(Config(), **change)


if __name__ == "__main__":
    unittest.main()
