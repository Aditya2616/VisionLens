"""Headless Chromium check of the UI: screenshots at desktop and phone size, console errors,
horizontal-overflow and tap-target audits, the sprite fallback, and (with --live) the upload,
sample-run and webcam paths against the real local server.

    python scripts/ui_check.py --mode static --out docs/assets
    python scripts/ui_check.py --mode live --out scratch/live

Starts its own server on port 8106 and stops it afterwards.
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


AUDIT_JS = """
() => {
  const small = [];
  for (const el of document.querySelectorAll('button, a.btn, label.btn, select, .scrub, .tab, .chip, .clip')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (r.height < 43.5 || r.width < 43.5) small.push(`${el.tagName}.${el.className} ${Math.round(r.width)}x${Math.round(r.height)}`);
  }
  return {
    overflowX: document.documentElement.scrollWidth - window.innerWidth,
    small,
    video: !!document.querySelector('#media video'),
    sprite: !!document.querySelector('#media canvas'),
  };
}
"""


def check(page, label: str, out: Path, problems: list[str]) -> None:
    a = page.evaluate(AUDIT_JS)
    if a["overflowX"] > 1:
        problems.append(f"{label}: horizontal overflow {a['overflowX']}px")
    if a["small"]:
        problems.append(f"{label}: small tap targets {a['small']}")


def run(mode: str, out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    logs: list[str] = []
    proc = start_server(mode)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args=[
                    "--use-fake-device-for-media-stream",
                    "--use-fake-ui-for-media-stream",
                    "--autoplay-policy=no-user-gesture-required",
                ]
            )
            for name, kw in [
                ("desktop", dict(viewport={"width": 1440, "height": 900})),
                (
                    "phone",
                    dict(
                        viewport={"width": 390, "height": 844},
                        device_scale_factor=2,
                        is_mobile=True,
                        has_touch=True,
                    ),
                ),
            ]:
                ctx = browser.new_context(permissions=["camera"], **kw)
                page = ctx.new_page()
                page.on(
                    "console",
                    lambda m, n=name: (
                        logs.append(f"[{n}] {m.type}: {m.text}")
                        if m.type in ("error", "warning")
                        else None
                    ),
                )
                page.on("pageerror", lambda e, n=name: logs.append(f"[{n}] pageerror: {e}"))
                page.goto(BASE, wait_until="networkidle")
                page.wait_for_selector("#media video, #media canvas, #media img", timeout=15000)
                page.wait_for_timeout(1500)
                page.screenshot(path=out / f"{name}-1-loaded.png")
                check(page, f"{name} loaded", out, problems)

                page.click("#bigplay")
                page.wait_for_timeout(900)
                page.locator("#stage").screenshot(path=out / f"{name}-2-scanning.png")
                page.wait_for_timeout(3500)
                page.screenshot(path=out / f"{name}-3-playing.png")
                page.click('[data-layer="dets"]')
                page.wait_for_timeout(500)
                page.locator("#stage").screenshot(path=out / f"{name}-4-raw-vs-tracks.png")
                page.click('[data-tab="model"]')
                page.wait_for_timeout(1500)
                page.screenshot(path=out / f"{name}-5-model-tab.png", full_page=True)
                check(page, f"{name} after interaction", out, problems)

                # each clip loads and draws
                for cid in ["traffic", "synthetic", "still"]:
                    sel = f'.clip[data-id="{cid}"]'
                    if page.locator(sel).count():
                        page.click(sel)
                        page.wait_for_timeout(1800)
                        if cid == "synthetic":
                            page.click("#bigplay")
                            page.click('[data-layer="gt"]')
                            page.wait_for_timeout(2500)
                        page.screenshot(path=out / f"{name}-6-{cid}.png")
                        check(page, f"{name} {cid}", out, problems)

                # timeline scrub by pointer
                page.click('.clip[data-id="intersection"]')
                page.wait_for_timeout(1500)
                page.locator("#timeline").scroll_into_view_if_needed()
                tl = page.locator("#tl-canvas").bounding_box()
                if tl:
                    page.mouse.click(tl["x"] + tl["width"] * 0.6, tl["y"] + 60)
                    page.wait_for_timeout(600)
                    t_label = page.inner_text("#time")
                    if t_label.startswith("0:00 /"):
                        problems.append(f"{name}: timeline click did not seek ({t_label})")
                ctx.close()

            # sprite fallback (what a browser without VP9 WebM gets)
            ctx = browser.new_context(
                viewport={"width": 390, "height": 844},
                device_scale_factor=2,
                is_mobile=True,
                has_touch=True,
            )
            page = ctx.new_page()
            page.on(
                "console",
                lambda m: (
                    logs.append(f"[sprite] {m.type}: {m.text}")
                    if m.type in ("error", "warning")
                    else None
                ),
            )
            page.goto(BASE + "?player=sprite", wait_until="networkidle")
            page.wait_for_selector("#media canvas", timeout=15000)
            page.wait_for_timeout(800)
            page.click("#bigplay")
            page.wait_for_timeout(2500)
            page.screenshot(path=out / "phone-7-sprite-fallback.png")
            if page.locator("#media video").count():
                problems.append("sprite mode still created a <video>")
            ctx.close()

            if mode == "live":
                live_checks(browser, out, problems, logs)
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    for line in logs:
        print(line)
    for pr in problems:
        print("PROBLEM:", pr)
    print("console messages:", len(logs), "problems:", len(problems))
    return 1 if (problems or any("error" in line for line in logs)) else 0


def live_checks(browser, out: Path, problems: list[str], logs: list[str]) -> None:
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, permissions=["camera"])
    page = ctx.new_page()
    page.on(
        "console",
        lambda m: (
            logs.append(f"[live] {m.type}: {m.text}") if m.type in ("error", "warning") else None
        ),
    )
    page.goto(BASE, wait_until="networkidle")
    page.wait_for_selector("#live-on:not([hidden])", timeout=10000)
    mode = page.inner_text("#mode")
    if "Live" not in mode:
        problems.append(f"live badge missing: {mode!r}")
    # upload a real video (the traffic sample), which exercises the whole server path
    page.set_input_files("#file", str(ROOT / "docs" / "demo" / "traffic.webm"))
    page.wait_for_selector("#busy:not([hidden])", timeout=5000)
    page.wait_for_timeout(1500)
    page.screenshot(path=out / "live-1-busy.png")
    page.wait_for_selector("#busy", state="hidden", timeout=240000)
    page.wait_for_timeout(2500)
    page.screenshot(path=out / "live-2-result.png")
    if "Live result" not in page.inner_text("#d-tag"):
        problems.append("live result tag missing")
    # image upload
    page.set_input_files("#file", str(ROOT / "docs" / "demo" / "still.jpg"))
    page.wait_for_selector("#busy:not([hidden])", timeout=5000)
    page.wait_for_selector("#busy", state="hidden", timeout=120000)
    page.wait_for_timeout(1500)
    page.screenshot(path=out / "live-3-image.png")
    # webcam (Chromium fake device)
    page.click("#cam")
    page.wait_for_timeout(5000)
    page.screenshot(path=out / "live-4-webcam.png")
    page.click("#cam")
    page.wait_for_timeout(800)
    ctx.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "live"], default="static")
    ap.add_argument("--out", default=str(ROOT / "scratch" / "ui"))
    args = ap.parse_args()
    raise SystemExit(run(args.mode, Path(args.out)))
