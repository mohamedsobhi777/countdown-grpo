"""Browser smoke check against a running demo; requires Playwright.

Run with: uv run --no-project --with playwright python scripts/check_demo_browser.py
Use --browser for a locally installed Chromium executable.
Use --live only when the server has a real adapter and the GPU is available.
"""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8765')
parser.add_argument('--browser', default='/snap/bin/chromium')
parser.add_argument('--live', action='store_true', help='Exercise real generation through the browser')
parser.add_argument('--output-prefix', type=Path, default=Path('outputs/demo'))
args = parser.parse_args()
args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=args.browser, headless=True)
    page = browser.new_page(viewport={'width': 1280, 'height': 1100})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(args.url)
    # Native select options exist in the DOM without being visually expanded.
    page.wait_for_selector('#examples option', state='attached')
    assert page.locator('#examples option').count() == 256
    page.select_option('#filter', 'correct')
    assert page.locator('#examples option').count() == 85
    page.select_option('#filter', 'failed')
    assert page.locator('#examples option').count() == 171
    page.select_option('#filter', 'all')
    page.select_option('#examples', '0')
    page.locator('details').last.locator('summary').click()
    page.click('#verify')
    page.wait_for_function("document.getElementById('verification').textContent.includes('Strict solution: yes')")
    if args.live:
        assert page.locator('#generate').is_enabled()
        with page.expect_response(lambda response: response.url.endswith('/api/generate'), timeout=600000) as pending:
            page.click('#generate')
        response = pending.value
        assert response.status == 200, response.status
        result = response.json()
        page.wait_for_function("document.getElementById('status').textContent === 'Both answers generated and verified.'")
        assert page.locator('#generate').is_enabled()
        assert page.locator('#mode').inner_text().startswith('Live greedy generation.')
        for index, name in enumerate(('base', 'adapter')):
            card = page.locator('#comparison > section').nth(index)
            assert card.locator('pre').first.inner_text() == result[name]['completion']
            card.locator('summary').click()
            assert 'Strict solution:' in card.locator('pre').last.inner_text()
        args.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        assert page.locator('#generate').is_disabled()
    page.screenshot(path=str(args.output_prefix) + '-desktop.png', full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.screenshot(path=str(args.output_prefix) + '-mobile.png', full_page=True)
    assert not errors, errors
    browser.close()
print(f'Browser checks passed: 256 examples, filters, exact verification, {"live" if args.live else "archive"} mode, mobile layout, no JS errors.')
