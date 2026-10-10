"""Local, dependency-free archived comparison UI with optional CUDA inference."""
import argparse
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from .data import puzzle_key, render_prompt
from .diagnostics import analyze_completion
from .statistics import paired_outcomes


def validate_puzzle(value):
    nums, target = value.get('nums'), value.get('target')
    if (not isinstance(nums, list) or len(nums) not in (3, 4)
            or any(type(n) is not int or not 1 <= n <= 200 for n in nums)
            or type(target) is not int or not 1 <= target <= 1000):
        raise ValueError('Use three or four integers from 1 to 200 and a target from 1 to 1000.')
    return {'nums': nums, 'target': target}


def archive_pairs(directory):
    before = json.loads((directory / 'baseline.json').read_text())
    after = json.loads((directory / 'after.json').read_text())
    paired_outcomes(before['results'], after['results'])
    indexed = {puzzle_key(r): r for r in after['results']}
    pairs = []
    for row in before['results']:
        pair = {'nums': row['nums'], 'target': row['target']}
        for name, record in [('base', row), ('adapter', indexed[puzzle_key(row)])]:
            diagnostic = analyze_completion(record['completion'], row['nums'], row['target'], record.get('truncated'))
            if diagnostic['strict_correct'] != record['correct']:
                raise ValueError('Archived correctness differs from the current verifier')
            pair[name] = {'completion': record['completion'], 'diagnostic': diagnostic}
        pairs.append(pair)
    return pairs


class LiveComparison:
    def __init__(self, adapter):
        self.adapter = adapter
        self.model = None
        self.lock = threading.Lock()

    def generate(self, puzzle, budget):
        if not self.lock.acquire(blocking=False):
            raise ValueError('Generation is already running. Please wait for it to finish.')
        try:
            import torch
            from .config import Config
            from .model import load_model
            if self.model is None:
                from peft import PeftModel
                base, self.tokenizer = load_model(Config())
                self.model = PeftModel.from_pretrained(base, self.adapter)
                self.model.eval()
            result = dict(puzzle)
            inputs = self.tokenizer(render_prompt(self.tokenizer, puzzle), return_tensors='pt',
                                    add_special_tokens=False).to('cuda')
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            for name in ('base', 'adapter'):
                context = self.model.disable_adapter() if name == 'base' else nullcontext()
                with context, torch.inference_mode():
                    tokens = self.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                                                use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
                generated = tokens[0, inputs.input_ids.shape[1]:].tolist()
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                truncated = len(generated) >= budget and not any(t in eos for t in generated)
                result[name] = {'completion': text, 'diagnostic': analyze_completion(text, puzzle['nums'], puzzle['target'], truncated)}
            return result
        finally:
            self.lock.release()


def make_server(directory, port=8765, adapter=None):
    examples = archive_pairs(directory)
    live = LiveComparison(adapter) if adapter else None
    html = Path(__file__).with_name('demo.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, content_type='application/json'):
            content = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == '/':
                self.reply(200, html, 'text/html; charset=utf-8')
            elif self.path == '/api/examples':
                self.reply(200, {'examples': examples, 'live_enabled': live is not None})
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            # The service is local-only; do not accept cross-site browser requests.
            if self.headers.get('Origin') not in (None, f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
                self.reply(403, {'error': 'Cross-origin requests are not accepted'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('Request must be between 1 and 16384 bytes')
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError('Expected a JSON object')
                puzzle = validate_puzzle(value)
                if self.path == '/api/verify':
                    completion = value.get('completion')
                    if not isinstance(completion, str) or len(completion) > 8192:
                        raise ValueError('Answer must be text of at most 8192 characters')
                    self.reply(200, analyze_completion(completion, **puzzle))
                elif self.path == '/api/generate':
                    if live is None:
                        self.reply(409, {'error': 'Live generation requires starting the server with --adapter PATH.'})
                        return
                    budget = value.get('budget', 256)
                    if type(budget) is not int or budget not in (256, 512, 1024):
                        raise ValueError('Output budget must be 256, 512, or 1024')
                    self.reply(200, live.generate(puzzle, budget))
                else:
                    self.reply(404, {'error': 'Not found'})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {'error': str(error)})
            except Exception as error:
                self.log_error('Generation failed: %s', error)
                self.reply(500, {'error': 'Generation failed. Check the server log for details.'})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('results/2026-09-14'))
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--adapter', help='Enable live CUDA generation with this local LoRA adapter')
    args = parser.parse_args()
    if args.adapter and not (Path(args.adapter) / 'adapter_config.json').is_file():
        parser.error('--adapter must name a local adapter directory')
    server = make_server(args.archive, args.port, args.adapter)
    print(f'Countdown demo: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
