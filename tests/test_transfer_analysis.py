"""Exercise the transfer audit with synthetic files, never model performance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from countdown_grpo.config import Config
from countdown_grpo.diagnostics import analyze_completion


class TransferAuditIntegrationTests(unittest.TestCase):
    def test_complete_grid_and_four_number_training_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, suites, pilot = root/'source', root/'suites', root/'pilot'
            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            def rows_for(puzzles):
                rows = []
                for puzzle in puzzles:
                    text = '<answer>1+2+3</answer>'
                    diagnostic = analyze_completion(text, **puzzle, truncated=False)
                    rows.append({**puzzle, 'completion': text, 'truncated': False,
                                 'correct': diagnostic['strict_correct'], 'diagnostic': diagnostic})
                return rows
            def report(puzzles, config, checksum, adapter):
                rows = rows_for(puzzles)
                return {'config': config, 'suite_sha256': checksum, 'adapter': adapter,
                        'decoding': 'greedy', 'samples': 256, 'results': rows,
                        'solve_rate': sum(r['correct'] for r in rows)/256, 'wandb_url': None}
            definitions = {'id_three': [1,2,3], 'id_four': [1,2,3,4],
                           'large_inputs': [101,102,103], 'large_targets': [1,3,5]}
            puzzles, manifest = {}, {'suites': {}}
            for name, nums in definitions.items():
                puzzles[name] = [{'nums': nums, 'target': i} for i in range(1,257)]
                path = suites/f'{name}.json'
                save(path, puzzles[name])
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                manifest['suites'][name] = {'sha256': checksum}
                save(pilot/f'pilot-{name}.json', report(puzzles[name], Config().to_dict(), checksum,
                     'outputs/countdown-20260914-01/training/adapter'))
            save(suites/'manifest.json', manifest)
            train = [{'nums': [1,2,3], 'target': i} for i in range(10001,30001)]
            for seed in (42,43,44):
                work = source/f'three-only-{seed}'
                expected = Config(seed=seed, split_seed=42, train_num_count=3).to_dict()
                training_output = f'outputs/synthetic-{seed}/training'
                save(work/'status.json', {'state':'complete'})
                save(work/'training/config.json', {**expected, 'output_dir': training_output})
                save(work/'training/train-puzzles.json', train)
                save(work/'training/holdout.json', puzzles['id_three'])
                save(work/'training/wandb-run.json', {'url':f'https://example.invalid/{seed}'})
                metrics = [{'step':i,'reward':0} for i in range(1,301)] + [{'train_runtime':100}]
                (work/'training/metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metrics))
                save(work/'evaluation.json', report(puzzles['id_three'], expected, None, training_output+'/adapter'))
                for name in definitions:
                    save(work/f'{name}.json', report(puzzles[name], expected,
                         manifest['suites'][name]['sha256'], training_output+'/adapter'))
            script = Path(__file__).resolve().parents[1]/'scripts/analyze_transfer_study.py'
            command = [sys.executable,str(script),'--source',str(source),'--suites',str(suites),
                       '--pilot-evidence',str(pilot),'--destination',str(root/'result')]
            result = subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            summary = json.loads((root/'result/summary.json').read_text())
            self.assertEqual(summary['seed_summary']['id_three']['mean_accuracy'],1/256)
            self.assertEqual(summary['seed_summary']['id_four']['mean_accuracy'],0)
            self.assertEqual(summary['seed_summary']['id_three']['sample_std_across_seeds'],0)
            self.assertEqual(len(summary['per_seed']),3)
            self.assertTrue((root/'result/three-only-44/large_inputs.json').exists())
            train[0] = {'nums':[1,2,3,4], 'target':10001}
            save(source/'three-only-43/training/train-puzzles.json',train)
            command[-1] = str(root/'rejected-result')
            failed = subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('invalid three-number-only training data',failed.stderr)
            self.assertFalse((root/'rejected-result').exists())
