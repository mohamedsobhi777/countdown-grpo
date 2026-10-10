"""Freeze challenge inputs, oracle witnesses, and checksums before model evaluation."""
import hashlib
import json
from pathlib import Path

from countdown_grpo.benchmarks import shifted_suite
from countdown_grpo.config import Config
from countdown_grpo.data import load_rows, puzzle_key
from countdown_grpo.diagnostics import analyze_completion
from countdown_grpo.solver import solve


def main():
    root = Path('data/suites/v1')
    if root.exists():
        raise FileExistsError('Suite is already frozen; create a new version instead')
    root.mkdir(parents=True)
    train, holdout = load_rows(Config(eval_samples=1000000))
    train_keys = set(map(puzzle_key,train))
    manifest = {'version':'v1','seed':20261010,'split_seed':42,'count_per_suite':256,
                'dataset_revision':Config().dataset_revision,'suites':{},
                'generation_rule':'For shifts: sample inputs uniformly, then choose uniformly among sorted reachable integer targets in range; balance 3/4 numbers.'}
    all_witnesses = {}
    for name in ('id_three','id_four','large_inputs','large_targets'):
        if name.startswith('id_'):
            size = 3 if name == 'id_three' else 4
            rows = [r for r in holdout if len(r['nums']) == size][:256]
            witnesses = [solve(r['nums'],r['target']) for r in rows]
        else:
            rows,witnesses = shifted_suite(name)
        assert len(rows) == 256
        assert len(set(map(puzzle_key,rows))) == 256
        assert not train_keys.intersection(map(puzzle_key,rows))
        assert all(w is not None and analyze_completion(f'<answer>{w}</answer>',r['nums'],r['target'])['diagnostic_correct']
                   for r,w in zip(rows,witnesses,strict=True))
        path = root/f'{name}.json'
        path.write_text(json.dumps(rows,indent=2)+'\n')
        manifest['suites'][name] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'samples':len(rows)}
        all_witnesses[name] = witnesses
    (root/'witnesses.json').write_text(json.dumps(all_witnesses,indent=2)+'\n')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
