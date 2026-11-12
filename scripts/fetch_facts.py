"""Collect the facts about the models that the README quotes and write data/facts.json.

    python scripts/fetch_facts.py

Licence, parameter count and download size come from the Hugging Face hub API (the download size is
the sum of every file in the repository, which is what `visionlens fetch-models` downloads). The
published COCO AP figures are not measured here: each one is looked up in the page it is cited from
and the script refuses to write a number that it cannot find there.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionlens.describe.llm import DEFAULT_LLM  # noqa: E402
from visionlens.detect.hf import MODELS  # noqa: E402

UA = {"User-Agent": "visionlens-facts/1.0"}
DFINE_README = "https://raw.githubusercontent.com/Peterande/D-FINE/master/README.md"
DFINE_PAGE = "https://github.com/Peterande/D-FINE"
RTDETR_CARD = "https://huggingface.co/PekingU/rtdetr_r18vd"

# key -> (published AP, where it is cited, regex that must match on that page)
PUBLISHED_AP = {
    "dfine-nano": (42.8, DFINE_PAGE, r"D&#8209;FINE&#8209;N\*\* \| COCO \| \*\*42\.8\*\*"),
    "dfine-small": (48.5, DFINE_PAGE, r"D&#8209;FINE&#8209;S\*\* \| COCO \| \*\*48\.5\*\*"),
    "rtdetr-r18": (
        46.5,
        RTDETR_CARD,
        r"\| RT-DETR-R18\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*46\.5\s*\|",
    ),
}
LLMS = {"qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct"}


def fetch(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def main() -> None:
    api = HfApi()
    out: dict = {"fetched_on": dt.date.today().isoformat(), "models": {}}
    pages = {
        DFINE_PAGE: fetch(DFINE_README),
        RTDETR_CARD: open(
            hf_hub_download("PekingU/rtdetr_r18vd", "README.md"), encoding="utf-8"
        ).read(),
    }
    repos = {**{k: v for k, v in MODELS.items()}, **LLMS}
    for key, repo in repos.items():
        info = api.model_info(repo, files_metadata=True)
        rec = {
            "repo": repo,
            "licence": (info.card_data.get("license") if info.card_data else None) or "unknown",
            "params": info.safetensors.total if info.safetensors else None,
            "download_bytes": sum(s.size or 0 for s in info.siblings),
            "revision": info.sha,
        }
        if key in PUBLISHED_AP:
            ap, src, pattern = PUBLISHED_AP[key]
            if not re.search(pattern, pages[src]):
                raise SystemExit(f"published AP {ap} for {key} was not found at {src}")
            rec.update(published_ap=ap, ap_source=src)
        out["models"][key] = rec
    out["default_llm"] = DEFAULT_LLM
    path = ROOT / "data" / "facts.json"
    path.write_text(json.dumps(out, indent=2) + "\n", "utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
