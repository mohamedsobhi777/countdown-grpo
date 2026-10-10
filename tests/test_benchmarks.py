import unittest
from countdown_grpo.benchmarks import shifted_suite
from countdown_grpo.diagnostics import analyze_completion


class BenchmarkTests(unittest.TestCase):
    def test_deterministic_and_solvable(self):
        for name in ('large_inputs','large_targets'):
            rows,witnesses=shifted_suite(name,count=10)
            self.assertEqual((rows,witnesses),shifted_suite(name,count=10))
            self.assertEqual(sum(len(r['nums'])==3 for r in rows),5)
            for row,witness in zip(rows,witnesses,strict=True):
                self.assertTrue(analyze_completion(f'<answer>{witness}</answer>',row['nums'],row['target'])['diagnostic_correct'])
                if name=='large_inputs':
                    self.assertTrue(all(101<=n<=200 for n in row['nums']))
                    self.assertTrue(10<=row['target']<=100)
                else:
                    self.assertTrue(all(1<=n<=99 for n in row['nums']))
                    self.assertTrue(101<=row['target']<=200)


if __name__ == '__main__':
    unittest.main()
