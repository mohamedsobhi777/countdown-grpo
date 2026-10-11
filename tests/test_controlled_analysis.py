"""Synthetic manifests test rejection gates, not model performance."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from countdown_grpo.config import Config
from countdown_grpo.diagnostics import analyze_completion

spec = importlib.util.spec_from_file_location('controlled', Path(__file__).resolve().parents[1] / 'scripts/analyze_controlled_study.py')
controlled = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controlled)


class ControlledAuditTests(unittest.TestCase):
    def test_accepts_complete_and_rejects_corrupt_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'training').mkdir()
            def save(name, value):
                (root / name).write_text(json.dumps(value))
            config = Config(split_seed=42).to_dict()
            provenance = {'state':'finished','group':'controlled-l4-fla-v1',
                          'config':{'git_dirty':False,'git_commit':'synthetic','python_version':'synthetic',
                                    'packages':{},'optional_kernel_packages':{'flash-linear-attention':'0.5.2',
                                    'fla-core':'0.5.2','einops':'0.8.1','causal-conv1d':None}},
                          'hardware':{'gpu':'NVIDIA L4','gpu_count':1}}
            requirements = 'synthetic-fixture==1\n'
            (root/'training/requirements.txt').write_text(requirements)
            provenance['config']['requirements_sha256'] = hashlib.sha256(requirements.strip().encode()).hexdigest()
            save('training/provenance.json', provenance)
            save('training/config.json', config)
            save('status.json', {'state': 'complete'})
            save('training/train-puzzles.json', [{'nums': [1, 2, 3], 'target': i} for i in range(1001, 21001)])
            holdout = [{'nums': [1, 2, 3], 'target': i} for i in range(1, 257)]
            save('training/holdout.json', holdout)
            rows = []
            for puzzle in holdout:
                text = '<answer>1+2+3</answer>'
                diagnostic = analyze_completion(text, **puzzle, truncated=False)
                rows.append({**puzzle, 'completion': text, 'correct': diagnostic['strict_correct'],
                             'truncated': False, 'diagnostic': diagnostic})
            evaluation = {'config': config, 'adapter': 'outputs/countdown/adapter', 'decoding': 'greedy',
                          'results': rows, 'samples': 256, 'solve_rate': 1/256, 'wandb_url': None}
            save('evaluation.json', evaluation)
            metrics = [{'step': i, 'reward': 0} for i in range(1, 301)] + [{'train_runtime': 100}]
            def write_metrics(values):
                (root / 'training/metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
            write_metrics(metrics)
            save('training/wandb-run.json', {'entity': 'test', 'project': 'test', 'id': 'synthetic'})
            summary, *_ = controlled.checked_run(root, 0, 42)
            self.assertEqual(summary['correct'], 1)
            self.assertEqual(summary['failures'], {'wrong_value':255, 'correct':1})
            self.assertIsNone(summary['first_positive_reward_step'])
            write_metrics(metrics[1:])
            with self.assertRaisesRegex(ValueError, '300 steps'):
                controlled.checked_run(root, 0, 42)
            write_metrics(metrics)
            evaluation['results'][0]['correct'] = True
            save('evaluation.json', evaluation)
            with self.assertRaisesRegex(ValueError, 'verifier'):
                controlled.checked_run(root, 0, 42)
            evaluation['results'][0]['correct'] = False
            save('evaluation.json', evaluation)
            save('status.json', {'state': 'running'})
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                controlled.checked_run(root, 0, 42)
            save('status.json', {'state':'complete'})
            provenance['group'] = 'controlled-local-v2'
            provenance['hardware']['gpu'] = 'NVIDIA GeForce RTX 4090 Laptop GPU'
            provenance['config']['optional_kernel_packages'] = controlled.PROFILES['local']['kernels'].copy()
            save('training/provenance.json', provenance)
            self.assertEqual(controlled.checked_run(root, 0, 42, 'local')[0]['correct'], 1)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                controlled.checked_run(root, 0, 42, 'colab')
            provenance['config']['optional_kernel_packages']['fla-core'] = '0.5.2'
            save('training/provenance.json', provenance)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                controlled.checked_run(root, 0, 42, 'local')
            provenance['config']['optional_kernel_packages']['fla-core'] = None
            save('training/provenance.json', provenance)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                controlled.checked_run(root, 0, 42)
