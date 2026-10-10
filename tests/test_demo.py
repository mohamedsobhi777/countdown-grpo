import json
from pathlib import Path
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from countdown_grpo.demo import make_server, validate_puzzle


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(Path(__file__).resolve().parents[1] / 'results/2026-09-14', port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, path, body, headers=None):
        return urlopen(Request(self.url + path, data=json.dumps(body).encode(),
                               headers=headers or {'Content-Type': 'application/json'}), timeout=3)

    def test_archive_contains_all_successes_and_failures(self):
        with urlopen(self.url + '/api/examples', timeout=3) as response:
            data = json.load(response)
        self.assertFalse(data['live_enabled'])
        self.assertEqual(len(data['examples']), 256)
        self.assertEqual(sum(r['adapter']['diagnostic']['strict_correct'] for r in data['examples']), 85)
        self.assertEqual(sum(r['base']['diagnostic']['strict_correct'] for r in data['examples']), 0)

    def test_verifier_and_generation_disabled(self):
        puzzle = {'nums': [1, 2, 3], 'target': 6}
        with self.post('/api/verify', {**puzzle, 'completion': '<answer>1+2+3</answer>'}) as response:
            self.assertTrue(json.load(response)['strict_correct'])
        with self.assertRaises(HTTPError) as error:
            self.post('/api/generate', puzzle)
        self.assertEqual(error.exception.code, 409)

    def test_cross_origin_and_invalid_input(self):
        with self.assertRaises(HTTPError) as error:
            self.post('/api/verify', {}, {'Origin': 'https://example.com'})
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error:
            self.post('/api/verify', {'nums': [True, 2, 3], 'target': 6})
        self.assertEqual(error.exception.code, 400)
        for nums in ([1, 2], [0, 2, 3], [1, 2, 3.5], [1, 2, 201]):
            with self.assertRaises(ValueError):
                validate_puzzle({'nums': nums, 'target': 6})

    def test_no_arbitrary_file_serving(self):
        with urlopen(self.url, timeout=3) as response:
            self.assertIn(b'Before and after RL', response.read())
        with self.assertRaises(HTTPError) as error:
            urlopen(self.url + '/../../.netrc', timeout=3)
        self.assertEqual(error.exception.code, 404)
