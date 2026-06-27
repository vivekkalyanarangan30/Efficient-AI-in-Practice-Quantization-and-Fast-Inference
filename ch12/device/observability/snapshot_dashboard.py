"""Capture the ch12_device_vision Grafana dashboard as a PNG via Playwright.

This is the workaround for grafana-image-renderer 3.x not shipping a
darwin-arm64 build. Same idea, fewer moving parts: a headless Chromium
authenticates against the local Grafana, waits for panels to render,
then takes a viewport screenshot.

Secrets:
  * admin password read from observability/secrets/grafana.admin (0600)
  * never logged, never placed in URL, never echoed in argv
  * login goes through the standard Grafana /login form over loopback
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent
SECRETS = BASE / "secrets" / "grafana.admin"


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot the Grafana dashboard")
    ap.add_argument("--url", default="http://127.0.0.1:3000",
                    help="Grafana base URL (loopback only)")
    ap.add_argument("--uid", default="ch12-device-vision",
                    help="dashboard UID")
    ap.add_argument("--width",  type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--out", default=str(BASE.parent / "exports" / "grafana_m3.png"))
    args = ap.parse_args()

    if not args.url.startswith("http://127.0.0.1") and not args.url.startswith("https://127.0.0.1"):
        # Refuse to drive a headless browser against anything but loopback —
        # avoids inadvertent screenshot of a public dashboard from this script.
        print("URL must be loopback (127.0.0.1).", file=sys.stderr)
        return 2

    if not SECRETS.exists():
        print(f"missing {SECRETS}; cannot log in.", file=sys.stderr)
        return 2
    pw_value = SECRETS.read_text().strip()
    if not pw_value:
        print("empty admin password file.", file=sys.stderr)
        return 2

    qs = urlencode({
        "orgId": "1",
        "kiosk": "tv",          #A hides side menu, keeps a thin top bar
        "refresh": "",          #B halt auto-refresh during snapshot
        "from": "now-15m",
        "to":   "now",
    })
    dash_url = f"{args.url}/d/{args.uid}/{args.uid}?{qs}"

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                viewport={"width": args.width, "height": args.height},
                device_scale_factor=2,            #A retina-quality PNG
                ignore_https_errors=False,
            )
            page = context.new_page()
            # Standard form login (cookie-based; no creds in URL or argv).
            page.goto(f"{args.url}/login", wait_until="networkidle", timeout=20000)
            page.fill('input[name="user"]', "admin")
            page.fill('input[name="password"]', pw_value)
            page.click('button[type="submit"]')
            page.wait_for_load_state("networkidle", timeout=20000)

            # Some Grafana installs prompt for a password change after first
            # login. We don't want that flow — skip if present.
            try:
                skip = page.locator("text=Skip").first
                if skip and skip.is_visible(timeout=1500):
                    skip.click()
                    page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            page.goto(dash_url, wait_until="networkidle", timeout=30000)

            # Give the canvas/SVG renderers a beat to settle after data loads.
            time.sleep(4)

            page.screenshot(path=str(out), full_page=False)
        finally:
            try:
                pw_value = "x" * len(pw_value)  # scrub
            finally:
                browser.close()

    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
