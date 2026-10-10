"""Browser smoke check against a running archived demo; requires Playwright.

Run with: uv run --no-project --with playwright python scripts/check_demo_browser.py
Use --browser for a locally installed Chromium executable.
"""
import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://127.0.0.1:8765')
parser.add_argument('--browser', default='/snap/bin/chromium')
args = parser.parse_args()
Path('outputs').mkdir(exist_ok=True)
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
    assert page.locator('#generate').is_disabled()
    page.screenshot(path='outputs/demo-desktop.png', full_page=True)
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    page.screenshot(path='outputs/demo-mobile.png', full_page=True)
    assert not errors, errors
    browser.close()
print('Browser checks passed: 256 examples, filters, exact verification, archive mode, mobile layout, no JS errors.')
