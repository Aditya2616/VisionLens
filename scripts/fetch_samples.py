"""Download CC0 clips from Wikimedia Commons, verify the licence, trim and re-encode them.

Writes docs/demo/<name>.webm (VP9, 480 wide, 10 fps, kept small on purpose), docs/demo/<name>_sprite.jpg and a
provenance entry in data/samples.json. Run:  python scripts/fetch_samples.py [name ...]

Commons throttles anonymous bulk downloads (HTTP 429), so requests send a descriptive
User-Agent and pause between calls. For very large sources only the first bytes are fetched
with an HTTP Range request (the decoder copes with the truncated tail).
"""

from __future__ import annotations

import html
import json
import re
import sys
import tempfile
import time
from pathlib import Path

import requests

from visionlens.sprite import write_sprite
from visionlens.video import WebmWriter, iter_frames, probe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "demo"
WIDTH, FPS = 480, 10
UA = "VisionLens/0.1 (https://github.com/Aditya2616/VisionLens; portfolio project) python-requests"

SAMPLES = [
    {
        "name": "intersection",
        "title": "Straßenverkehr in Düsseldorf.ogv",
        "label": "Street corner, fixed camera",
        "blurb": "Cars and small, distant pedestrians at a tram junction.",
        "range_bytes": 20_000_000,
        "start": 3.0,
        "seconds": 8.0,
    },
    {
        "name": "traffic",
        "title": "Gau-Bischofsheim - Steigstraße, Ampelschaltung auf Rot.webm",
        "label": "Village road, no people",
        "blurb": "A car, a bus and a van pass a fixed camera; there are no people in the shot.",
        "range_bytes": None,
        "start": 2.0,
        "seconds": 8.0,
    },
]


def commons_info(title: str) -> dict:
    r = requests.get(
        "https://commons.wikimedia.org/w/api.php",
        headers={"User-Agent": UA},
        params={
            "action": "query",
            "titles": "File:" + title,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "format": "json",
        },
        timeout=60,
    ).json()
    ii = next(iter(r["query"]["pages"].values()))["imageinfo"][0]
    meta = ii["extmetadata"]
    lic = meta["LicenseShortName"]["value"]
    if lic not in ("CC0", "Public domain"):
        raise SystemExit(f"{title}: licence is {lic!r}, refusing to use it")
    artist = html.unescape(re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", ""))).strip()
    return {
        "url": ii["url"],
        "license": lic,
        "author": artist or "unknown",
        "page": ii["descriptionurl"],
    }


def download(url: str, dest: Path, range_bytes: int | None) -> None:
    headers = {"User-Agent": UA}
    if range_bytes:
        headers["Range"] = f"bytes=0-{range_bytes}"
    for attempt in range(5):
        r = requests.get(url, headers=headers, timeout=600)
        if r.status_code in (200, 206):
            dest.write_bytes(r.content)
            return
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"download failed: {url} ({r.status_code})")


def build(spec: dict, manifest: dict) -> None:
    info = commons_info(spec["title"])
    time.sleep(2)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / Path(spec["title"]).name
        download(info["url"], src, spec["range_bytes"])
        src_info = probe(src)
        out = OUT / f"{spec['name']}.webm"
        frames = []
        size = None
        w = None
        for _i, t, frame in iter_frames(
            src,
            max_width=WIDTH,
            target_fps=FPS,
            start_seconds=spec["start"],
            max_seconds=spec["seconds"],
        ):
            if w is None:
                size = (frame.shape[1], frame.shape[0])
                w = WebmWriter(out, float(FPS), size)
            w.write(frame)
            frames.append((t, frame))
        if w is None:
            raise SystemExit("no frames decoded")
        w.close()
        sprite = write_sprite(
            frames, OUT / f"{spec['name']}_sprite.jpg", fps=5.0, thumb_w=224, quality=50
        )
    manifest[spec["name"]] = {
        "label": spec["label"],
        "blurb": spec["blurb"],
        "file": out.name,
        "sprite": f"{spec['name']}_sprite.jpg",
        "sprite_info": sprite,
        "width": size[0],
        "height": size[1],
        "fps": float(FPS),
        "frames": len(frames),
        "source": {
            "title": spec["title"],
            "page": info["page"],
            "author": info["author"],
            "license": info["license"],
            "trimmed_from": src_info.fps and f"{spec['start']}s",
            "seconds": spec["seconds"],
        },
    }
    print(
        spec["name"],
        out.stat().st_size // 1000,
        "KB webm,",
        (OUT / f"{spec['name']}_sprite.jpg").stat().st_size // 1000,
        "KB sprite",
    )


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "data" / "samples.json"
    manifest = json.loads(path.read_text("utf-8")) if path.exists() else {}
    for spec in SAMPLES:
        if names and spec["name"] not in names:
            continue
        build(spec, manifest)
        time.sleep(3)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
