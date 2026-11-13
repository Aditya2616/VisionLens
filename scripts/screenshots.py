"""Capture the README screenshots from the real page with a headless Chromium of its own.

    python scripts/screenshots.py --mode static --out docs/assets   # the two phone screenshots (recorded demo)
    python scripts/screenshots.py --mode live --out docs/assets     # live-mode.png (runs the real pipeline)

Phone images are real 390x844 viewports at device scale factor 2; the desktop image is 1440x900. Every shot is taken
at the end state of its animation (scanner finished, typewriter finished, fonts loaded). Starts its own server on port
8106 and stops it afterwards. The live shot loads the detector and the captioner, so run it on an idle machine.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8106
BASE = f"http://127.0.0.1:{PORT}/"

PHONE = dict(
    viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True
)
DESKTOP = dict(viewport={"width": 1440, "height": 900})


def start_server(mode: str) -> subprocess.Popen:
    if mode == "static":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "static_server.py"),
            str(PORT),
            str(ROOT / "docs"),
        ]
    else:
        cmd = [sys.executable, "-m", "visionlens.cli", "serve", "--port", str(PORT)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            urllib.request.urlopen(BASE, timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit("server did not start")


def settle(page) -> None:
    """Fonts loaded and no pending typewriter text."""
    page.evaluate("document.fonts.ready")
    page.wait_for_function(
        "() => { const t = document.querySelector('#d-text'); return t && t.textContent.length > 40"
        " && !t.querySelector('.caret, .cursor'); }",
        timeout=60000,
    )
    last = ""
    for _ in range(40):  # the description is stable for half a second
        cur = page.inner_text("#d-text")
        if cur == last:
            break
        last = cur
        page.wait_for_timeout(500)


def pick_track(page) -> None:
    box = page.evaluate(
        "() => { const h = (window.__vl.overlay.hit || []).slice().sort("
        "(a, b) => (b.x2 - b.x1) * (b.y2 - b.y1) - (a.x2 - a.x1) * (a.y2 - a.y1))[0];"
        " return h ? [h.x1, h.y1, h.x2, h.y2] : null; }"
    )
    if not box:
        raise SystemExit("no box to select")
    ov = page.locator("#overlay").bounding_box()
    page.mouse.click(ov["x"] + (box[0] + box[2]) / 2, ov["y"] + (box[1] + box[3]) / 2)
    page.wait_for_timeout(500)


def card_clip(page, top_sel: str, bottom_sel: str, width: int, height: int) -> dict:
    """Scroll top_sel to the top and return a clip from its card's top edge to the bottom edge of bottom_sel's card."""
    page.evaluate(
        f"window.scrollTo(0, document.querySelector('{top_sel}').getBoundingClientRect().top + scrollY - 6)"
    )
    page.wait_for_timeout(500)
    top, bottom = page.evaluate(
        f"""(() => {{ const t = document.querySelector('{top_sel}').getBoundingClientRect().top;
        const e = document.querySelector('{bottom_sel}'); const c = e.closest('.card') || e;
        return [t, c.getBoundingClientRect().bottom]; }})()"""
    )
    y = max(0, top)
    return {"x": 0, "y": y, "width": width, "height": min(height - y, bottom + 4 - y)}


def phone_shots(browser, out: Path) -> None:
    ctx = browser.new_context(**PHONE)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_selector("#media video, #media canvas, #media img", timeout=15000)

    # synthetic clip with the ground-truth layer, scanner finished, mid-clip
    page.click('.clip[data-id="synthetic"]')
    page.wait_for_timeout(1800)
    page.click("#bigplay")
    page.click('[data-layer="gt"]')
    page.wait_for_timeout(3500)
    page.evaluate("document.fonts.ready")
    clip = card_clip(page, "#strip", "#timeline", 390, 844)
    page.screenshot(path=str(out / "phone-synthetic-ground-truth.png"), clip=clip)

    # the village-road clip, one track selected with its whole trail
    page.click('.clip[data-id="traffic"]')
    page.wait_for_timeout(1800)
    page.click("#bigplay")
    page.wait_for_timeout(4500)
    pick_track(page)
    clip = card_clip(page, "#strip", "#timeline", 390, 844)
    page.screenshot(path=str(out / "phone-selected-track.png"), clip=clip)
    ctx.close()
    if errors:
        print("console errors:", errors)


def live_shot(browser, out: Path) -> None:
    ctx = browser.new_context(**DESKTOP)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_selector("#live-on:not([hidden])", timeout=10000)
    page.select_option("#opt-caption", "on")
    page.set_input_files("#file", str(ROOT / "docs" / "demo" / "traffic.webm"))
    page.wait_for_selector("#busy:not([hidden])", timeout=5000)
    page.wait_for_selector("#busy", state="hidden", timeout=300000)
    page.wait_for_selector("#d-scene:not([hidden])", timeout=10000)
    settle(page)
    page.wait_for_timeout(3000)  # the scanner sweep has finished
    # the clip chooser through the counters card, whole cards only on the right-hand side
    page.evaluate(
        "window.scrollTo(0, document.querySelector('#clips').getBoundingClientRect().top + scrollY - 10)"
    )
    page.wait_for_timeout(600)
    bottom = page.evaluate(
        "document.querySelector('#c-now').closest('.counters').getBoundingClientRect().bottom"
    )
    page.screenshot(
        path=str(out / "live-mode.png"),
        clip={"x": 0, "y": 0, "width": 1440, "height": min(900, bottom + 8)},
    )
    ctx.close()
    if errors:
        print("console errors:", errors)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "live"], default="static")
    ap.add_argument("--out", default=str(ROOT / "docs" / "assets"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = start_server(args.mode)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
            if args.mode == "static":
                phone_shots(browser, out)
            else:
                live_shot(browser, out)
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
