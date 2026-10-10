"""Exercise the real live-demo CUDA path after other GPU work has finished."""
import argparse
import json
from pathlib import Path

from countdown_grpo.config import Config
from countdown_grpo.demo import LiveComparison, validate_puzzle
from countdown_grpo.diagnostics import analyze_completion
from countdown_grpo.tracking import session, log_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default='outputs/countdown-20260914-01/training/adapter')
    parser.add_argument('--destination', type=Path, default=Path('results/study/demo-live.json'))
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    comparison = LiveComparison(args.adapter)
    puzzle = validate_puzzle({'nums': [67, 54, 98, 19], 'target': 66})
    with session(Config(), 'verification', 'live-demo-smoke-test', 'countdown-study-v1') as run:
        result = comparison.generate(puzzle, 256)
        for name in ('base', 'adapter'):
            answer = result[name]
            if not answer['completion'].strip():
                raise ValueError(f'{name}: empty generation')
            if answer['diagnostic'] != analyze_completion(answer['completion'], **puzzle, truncated=answer['diagnostic']['truncated']):
                raise ValueError(f'{name}: diagnostic mismatch')
        # A second request proves the loaded model remains usable after toggling
        # adapters. Greedy decoding of the identical input should be repeatable.
        repeated = comparison.generate(puzzle, 256)
        if repeated != result:
            raise ValueError('Repeated greedy comparison changed after toggling adapters')
        record = {'state': 'passed', 'adapter': args.adapter, 'max_new_tokens': 256,
                  'wandb_url': run.url, 'comparison': result,
                  'scope': 'Real base/adapter generation and repeatability on one puzzle, not a benchmark or proof of arithmetic accuracy.'}
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        args.destination.write_text(json.dumps(record, indent=2) + '\n')
        run.summary['live_cuda_check'] = 'passed'
        log_files(run, 'live-demo-verification', 'evaluation', files=[args.destination])
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
