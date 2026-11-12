from __future__ import annotations

import json
import shutil
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionlens.api import Settings, create_app
from visionlens.synth import NoiseConfig, NoisyOracle


class BlobDetector:
    """Finds the bright rectangle in a synthetic frame: enough to exercise the live paths."""

    name = "blob"
    input_size = 0

    def detect(self, frame, index=0):
        from visionlens.schemas import Detection

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero(gray > 200)
        if len(xs) < 20:
            return []
        return [
            Detection(
                box=(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                score=0.9,
                cls="person",
            )
        ]


@pytest.fixture()
def docs(tmp_path, scene_video):
    demo = tmp_path / "docs" / "demo"
    demo.mkdir(parents=True)
    shutil.copy(scene_video, demo / "clip.webm")
    (demo / "index.json").write_text(
        json.dumps(
            {
                "clips": [
                    {"id": "clip", "video": "clip.webm", "simulated": False},
                    {"id": "fake", "video": "x.webm", "simulated": True},
                ]
            }
        )
    )
    (tmp_path / "docs" / "index.html").write_text(
        '<html><meta name="vl-server" content="0">ui</html>'
    )
    return tmp_path / "docs"


@pytest.fixture()
def client(docs, small_scene):
    det = NoisyOracle(small_scene, NoiseConfig.level(0))
    app = create_app(det, Settings(docs_dir=str(docs), max_upload_mb=1, max_seconds=4.0))
    with TestClient(app) as c:
        yield c


def wait_done(client, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["state"] in ("done", "error"):
            return st
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_health_and_static_ui(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["live"] is True and h["limits"]["max_upload_mb"] == 1
    # the served page tells the front end that a local server is present
    assert 'name="vl-server" content="1"' in client.get("/").text


def test_samples_list(client):
    clips = client.get("/api/samples").json()["clips"]
    assert [c["id"] for c in clips] == ["clip", "fake"]


def test_analyze_sample_streams_events_and_returns_result(client):
    r = client.post("/api/analyze", data={"sample": "clip", "stride": "1"})
    assert r.status_code == 200
    job = r.json()["job_id"]
    assert wait_done(client, job)["state"] == "done"
    with client.stream("GET", f"/api/jobs/{job}/stream") as s:
        body = "".join(s.iter_text())
    for name in ("status", "token", "done"):
        assert f"event: {name}" in body
    res = client.get(f"/api/jobs/{job}/result").json()
    assert res["meta"]["n_frames"] > 20 and res["description"]["template"]["text"]
    video = client.get(f"/api/jobs/{job}/video")
    assert video.status_code == 200 and video.headers["content-type"] == "video/webm"
    assert len(video.content) > 1000


def test_simulated_samples_cannot_be_run_live(client):
    assert client.post("/api/analyze", data={"sample": "fake"}).status_code == 404
    assert client.post("/api/analyze", data={"sample": "nope"}).status_code == 404


def test_upload_is_analysed_and_the_original_is_deleted(client, scene_video):
    with scene_video.open("rb") as fh:
        r = client.post(
            "/api/analyze", files={"file": ("my.webm", fh, "video/webm")}, data={"stride": "1"}
        )
    job = r.json()["job_id"]
    assert wait_done(client, job)["state"] == "done"
    res = client.get(f"/api/jobs/{job}/result").json()
    assert res["meta"]["name"] == "your upload"
    # only the playback copy remains in the job folder
    import tempfile
    from pathlib import Path

    folder = next(p for p in Path(tempfile.gettempdir()).glob("visionlens-*") if (p / job).exists())
    assert sorted(f.name for f in (folder / job).iterdir()) == ["playback.webm"]
    assert client.delete(f"/api/jobs/{job}").status_code == 204
    assert client.get(f"/api/jobs/{job}").status_code == 404


def test_upload_limits_and_type_checks(client):
    big = b"x" * (1024 * 1024 + 10)
    r = client.post("/api/analyze", files={"file": ("big.mp4", big, "video/mp4")})
    assert r.status_code == 413
    r = client.post(
        "/api/analyze", files={"file": ("evil.exe", b"abc", "application/octet-stream")}
    )
    assert r.status_code == 415
    assert client.post("/api/analyze", data={}).status_code == 422
    assert client.post("/api/analyze", data={"sample": "clip", "stride": "99"}).status_code == 422
    assert client.post("/api/analyze", data={"sample": "clip", "describer": "x"}).status_code == 422


def test_corrupt_upload_reports_an_error_not_a_crash(client):
    r = client.post("/api/analyze", files={"file": ("bad.mp4", b"not a video at all", "video/mp4")})
    job = r.json()["job_id"]
    st = wait_done(client, job)
    assert st["state"] == "error" and st["error"]
    assert client.get(f"/api/jobs/{job}/result").status_code == 409


def test_image_upload(client):
    img = np.zeros((90, 160, 3), np.uint8)
    cv2.rectangle(img, (40, 20), (80, 70), (255, 255, 255), -1)
    ok, jpg = cv2.imencode(".jpg", img)
    r = client.post("/api/analyze", files={"file": ("a.jpg", jpg.tobytes(), "image/jpeg")})
    job = r.json()["job_id"]
    assert wait_done(client, job)["state"] == "done"
    res = client.get(f"/api/jobs/{job}/result").json()
    assert res["meta"]["kind"] == "image" and len(res["frames"]) == 1
    assert client.get(f"/api/jobs/{job}/video").headers["content-type"] == "image/jpeg"


def test_unknown_job(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/nope/stream").status_code == 404


def test_webcam_frames_keep_identity_across_calls(docs):
    app = create_app(BlobDetector(), Settings(docs_dir=str(docs), max_sessions=1))
    with TestClient(app) as c:
        ids = []
        for k in range(6):
            img = np.zeros((240, 320, 3), np.uint8)
            x = 40 + 8 * k
            cv2.rectangle(img, (x, 60), (x + 50, 180), (255, 255, 255), -1)
            ok, jpg = cv2.imencode(".jpg", img)
            r = c.post("/api/frame?session=abcdefgh", content=jpg.tobytes())
            assert r.status_code == 200
            ids += [t["id"] for t in r.json()["tracks"]]
        assert ids and set(ids) == {1}
        assert c.post("/api/frame?session=otherses1", content=jpg.tobytes()).status_code == 429
        assert c.post("/api/frame?session=abcdefgh", content=b"junk").status_code == 415
        assert c.post("/api/frame?session=abcdefgh", content=b"").status_code == 422


def test_caption_is_opt_in_and_comes_back_as_a_separate_scene_line(docs, small_scene):

    from visionlens.caption import SceneCaptioner

    class Fake(SceneCaptioner):
        def __init__(self):
            super().__init__(device="cpu")

        def generate(self, frame_bgr: np.ndarray) -> str:
            return "A street with a bus."

    det = NoisyOracle(small_scene, NoiseConfig.level(0))
    app = create_app(
        det, Settings(docs_dir=str(docs), max_upload_mb=1, max_seconds=4.0), captioner=Fake()
    )
    with TestClient(app) as c:
        plain = c.post("/api/analyze", data={"sample": "clip", "stride": "1"}).json()["job_id"]
        assert wait_done(c, plain)["state"] == "done"
        assert c.get(f"/api/jobs/{plain}/result").json()["scene"] is None
        job = c.post(
            "/api/analyze", data={"sample": "clip", "stride": "1", "caption": "true"}
        ).json()["job_id"]
        assert wait_done(c, job)["state"] == "done"
        res = c.get(f"/api/jobs/{job}/result").json()
        assert res["scene"]["text"] == "A street with a bus."
        assert res["scene"]["unseen"] == ["bus"] and res["scene"]["flagged"] is True
        assert "A street with a bus" not in res["description"]["template"]["text"]
