"""The README may only say what the code, the data files and the recorded results say.

Generated blocks (tables and the sentences that carry numbers) are checked by regenerating them
(`scripts/update_readme.py --check`). The facts in the prose (commands, flags, versions, ports, file
names, model ids) are checked here against the code and files they describe.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text("utf-8")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


update_readme = _load("update_readme", ROOT / "scripts" / "update_readme.py")
RESULTS = json.loads((ROOT / "results" / "results.json").read_text("utf-8"))
FACTS = json.loads((ROOT / "data" / "facts.json").read_text("utf-8"))
INDEX = json.loads((ROOT / "docs" / "demo" / "index.json").read_text("utf-8"))
SAMPLES = json.loads((ROOT / "data" / "samples.json").read_text("utf-8"))


def test_generated_blocks_are_in_sync_with_their_sources():
    assert update_readme.check(ROOT) == []


def _subparsers():
    from visionlens.cli import build_parser

    return build_parser()._subparsers._group_actions[0].choices


def _commands(text: str):
    """Every `visionlens <sub> ...` line in the README's code blocks and in the notes."""
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        for line in block.splitlines():
            line = line.split("#")[0].strip()
            if line.startswith("visionlens "):
                yield shlex.split(line)


def test_every_readme_command_and_flag_exists():
    subs = _subparsers()
    seen = 0
    for argv in _commands(README):
        seen += 1
        assert argv[1] in subs, f"unknown subcommand in README: {argv}"
        known = {o for a in subs[argv[1]]._actions for o in a.option_strings}
        for tok in argv[2:]:
            if tok.startswith("--"):
                assert tok in known, f"{argv[1]} has no flag {tok}"
    assert seen >= 5


def test_readme_command_inputs_and_scripts_exist():
    for argv in _commands(README):
        for tok in argv[2:]:
            if tok.startswith("docs/"):
                assert (ROOT / tok).exists(), tok
    for path in re.findall(r"`(?:python )?(scripts/\w+\.py)", README):
        assert (ROOT / path).exists(), path
    for name in re.findall(
        r"\b(?:fetch_samples|fetch_facts|ui_check|static_server|update_readme)\.py", README
    ):
        assert (ROOT / "scripts" / name).exists(), name


def test_every_readme_subcommand_mention_is_real():
    subs = set(_subparsers())
    for sub in re.findall(r"`visionlens ([a-z-]+)", README):
        assert sub in subs, sub


def test_port_and_python_version_match_the_code():
    serve = _subparsers()["serve"]
    port = next(a.default for a in serve._actions if "--port" in a.option_strings)
    assert f"127.0.0.1:{port}" in README
    pyproject = (ROOT / "pyproject.toml").read_text("utf-8")
    required = re.search(r'requires-python = ">=(\d+\.\d+)"', pyproject).group(1)
    assert f"Python {required} or later" in README
    dev = re.search(r"developed on (\d+\.\d+)", README).group(1)
    assert RESULTS["machine"]["python"].startswith(dev + ".")


def test_model_ids_in_the_readme_exist_in_code_and_facts():
    from visionlens.describe.llm import DEFAULT_LLM
    from visionlens.detect.hf import MODELS

    repos = {m["repo"] for m in FACTS["models"].values()}
    assert set(MODELS.values()) <= repos
    assert DEFAULT_LLM in repos and FACTS["default_llm"] == DEFAULT_LLM
    for rid in re.findall(r"`((?:ustc-community|PekingU|Qwen)/[\w.-]+)`", README):
        assert rid in repos, rid
    # the model the rewrite prose picks is the model the code defaults to
    assert update_readme.best_model(RESULTS)["model"] == DEFAULT_LLM


def test_licences_in_the_readme_match_the_hub():
    assert {m["licence"] for m in FACTS["models"].values()} == {"apache-2.0"}
    assert "are Apache-2.0" in " ".join(README.split())  # the YOLO paragraph
    assert "AGPL" in README and "MIT" in README


def test_benchmark_parameter_counts_agree_with_the_hub_counts():
    for row in RESULTS["detectors"]:
        hub = FACTS["models"][row["model"]]["params"] / 1e6
        assert abs(row["params_m"] - hub) < 0.1, row["model"]


def test_published_ap_has_a_source():
    for key in ("dfine-nano", "dfine-small", "rtdetr-r18"):
        f = FACTS["models"][key]
        assert f["published_ap"] > 0 and f["ap_source"].startswith("https://")


def test_viewports_and_fallback_rate_quoted_in_the_readme_are_the_ones_used():
    check = (ROOT / "scripts" / "ui_check.py").read_text("utf-8")
    assert '"width": 1440, "height": 900' in check
    assert '"width": 390, "height": 844' in check
    assert "1440x900" in README and "390x844" in README
    rates = {c["sprite_info"]["fps"] for c in INDEX["clips"] if c.get("sprite_info")}
    assert len(rates) == 1
    assert f"sprite sheet of {rates.pop():g} frames per second" in README


def test_recorded_demo_files_agree_with_each_other():
    demo = ROOT / "docs" / "demo"
    total_video = 0
    for clip in INDEX["clips"]:
        r = json.loads((demo / clip["json"]).read_text("utf-8"))
        assert (r["meta"]["width"], r["meta"]["height"]) == (clip["width"], clip["height"]) or (
            clip["kind"] == "image"
        )
        assert (demo / clip["video"]).exists()
        if clip["kind"] == "video":
            total_video += (demo / clip["video"]).stat().st_size
            assert r["meta"]["stride"] == INDEX["stride"]
            assert r["meta"]["detector"].startswith(INDEX["detector"]) or clip.get("simulated")
        if not clip.get("simulated"):
            assert r["timings"]["device"] in INDEX["device"].replace("cuda", "cuda cpu")
            assert INDEX["device"].startswith(r["timings"]["device"])
        if clip.get("sprite"):
            assert clip["sprite_info"] == SAMPLES.get(clip["id"], clip)["sprite_info"]
    assert total_video <= 3_000_000  # the committed sample clips stay about 3 MB in total


def test_sample_credits_are_cc0_with_a_source_url():
    for sm in SAMPLES.values():
        assert sm["source"]["license"] == "CC0"
        assert sm["source"]["page"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert sm["source"]["author"] in README


def test_bench_clip_is_the_readme_clip():
    assert (ROOT / "docs" / "demo" / RESULTS["pipeline"]["clip"]).exists()
    assert RESULTS["pipeline"]["clip"] in README


def test_notes_only_mention_real_subcommands():
    subs = set(_subparsers())
    for note in (ROOT / "docs" / "notes").glob("*.md"):
        for sub in re.findall(r"`visionlens ([a-z-]+)", note.read_text("utf-8")):
            assert sub in subs, f"{note.name}: {sub}"


@pytest.mark.parametrize("word", ["colour", "behaviour", "licence", "summarise"])
def test_british_spelling_is_used_in_the_readme(word):
    # a few words that are easy to get wrong; the American forms must not appear
    american = {
        "colour": "color",
        "behaviour": "behavior",
        "licence": "license",
        "summarise": "summarize",
    }
    text = re.sub(r"`[^`]*`|https?://\S+|\| CC0.*", "", README)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(
        r"<!-- RESULTS:credits:start -->.*?<!-- RESULTS:credits:end -->", "", text, flags=re.S
    )
    assert not re.search(r"\b" + american[word] + r"\b", text.lower())
