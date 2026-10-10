"""Exercise the real live-demo CUDA path after other GPU work has finished."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from urllib.request import Request, urlopen

from countdown_grpo.config import Config
from countdown_grpo.demo import make_server, validate_puzzle
from countdown_grpo.diagnostics import analyze_completion
from countdown_grpo.tracking import session, log_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default='outputs/countdown-20260914-01/training/adapter')
    parser.add_argument('--destination', type=Path, default=Path('results/study/demo-live.json'))
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    puzzle = validate_puzzle({'nums': [67, 54, 98, 19], 'target': 66})
    server = make_server(Path('results/2026-09-14'), port=0, adapter=args.adapter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    def generate():
        request = Request(url + '/api/generate', json.dumps({**puzzle, 'budget':256}).encode(),
                          headers={'Content-Type':'application/json', 'Origin':url})
        with urlopen(request, timeout=600) as response:
            return json.load(response)
    try:
        verify(args, puzzle, generate, url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def verify(args, puzzle, generate, url):
    with session(Config(), 'verification', 'live-demo-smoke-test', 'countdown-study-v1') as run:
        result = generate()
        for name in ('base', 'adapter'):
            answer = result[name]
            if not answer['completion'].strip():
                raise ValueError(f'{name}: empty generation')
            if answer['diagnostic'] != analyze_completion(answer['completion'], **puzzle, truncated=answer['diagnostic']['truncated']):
                raise ValueError(f'{name}: diagnostic mismatch')
        # A second request proves the loaded model remains usable after toggling
        # adapters. Greedy decoding of the identical input should be repeatable.
        repeated = generate()
        if repeated != result:
            raise ValueError('Repeated greedy comparison changed after toggling adapters')
        with tempfile.TemporaryDirectory(prefix='countdown-live-browser-') as temporary:
            prefix = Path(temporary)/'demo-live'
            subprocess.run(['uv','run','--no-project','--with','playwright','python',
                            'scripts/check_demo_browser.py','--live','--url',url,
                            '--output-prefix',str(prefix)], check=True, timeout=900)
            browser_result = json.loads(prefix.with_suffix('.json').read_text())
            if browser_result != result:
                raise ValueError('Browser generation differs from the repeated HTTP comparisons')
            log_files(run, 'live-demo-browser', 'verification', files=list(Path(temporary).glob('*')))
        record = {'state': 'passed', 'adapter': args.adapter, 'max_new_tokens': 256,
                  'wandb_url': run.url, 'comparison': result,
                  'checks': ['HTTP generation', 'repeated greedy outputs', 'browser generation button',
                             'rendered completions and verifier details', 'mobile layout', 'no browser JavaScript errors'],
                  'scope': 'Real base/adapter generation through HTTP and browser on one puzzle, not a benchmark or proof of arithmetic accuracy.'}
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        args.destination.write_text(json.dumps(record, indent=2) + '\n')
        run.summary['live_cuda_check'] = 'passed'
        run.summary['live_browser_check'] = 'passed'
        log_files(run, 'live-demo-verification', 'evaluation', files=[args.destination])
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
