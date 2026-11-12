"""FastAPI service: health, sample list, analysis jobs with server-sent events, webcam frames.

Uploads go to a temp folder and the original upload is deleted as soon as it has been analysed;
the resized playback copy is kept for a short time so the browser can fetch it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .describe import Describer, make_describer
from .detect.base import Detector
from .events import EventConfig, extract_events, summarize
from .pipeline import PipelineConfig
from .runner import analyze_file
from .schemas import Result, TrackOut
from .track import ByteTracker, TrackerConfig
from .video import IMAGE_EXT, VIDEO_EXT

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


class Settings(BaseModel):
    max_upload_mb: int = 40
    max_seconds: float = 20.0  # analysed length; longer uploads are truncated
    max_width: int = 640
    fps: float = 15.0
    stride: int = 3
    keep_seconds: int = 20 * 60
    keep_jobs: int = 6
    session_ttl: int = 120
    max_sessions: int = 4
    docs_dir: str = str(DOCS)


class HealthResponse(BaseModel):
    status: str
    version: str
    live: bool = True
    detector: str
    detector_loaded: bool
    limits: dict[str, Any]
    busy: bool


class AnalyzeResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    state: str
    error: str | None = None


class FrameResponse(BaseModel):
    session: str
    width: int
    height: int
    tracks: list[TrackOut]
    ms: float
    count: int
    description: str | None = None


class _Job:
    def __init__(self) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.msgs: list[tuple[str, dict]] = []
        self.state = "queued"
        self.error: str | None = None
        self.result: Result | None = None
        self.media: Path | None = None
        self.media_type = "video/webm"
        self.created = time.time()
        self.lock = threading.Lock()

    def emit(self, name: str, data: dict) -> None:
        with self.lock:
            self.msgs.append((name, data))


class _Session:
    def __init__(self) -> None:
        self.tracker = ByteTracker(TrackerConfig(min_hits=2, max_misses=6, coast_display=2))
        self.frames: list = []
        self.calls = 0
        self.last = time.time()
        self.t0 = time.time()


def _sse(name: str, data: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def create_app(detector: Detector | None = None, settings: Settings | None = None) -> FastAPI:
    cfg = settings or Settings()
    state: dict[str, Any] = {"detector": detector, "jobs": {}, "sessions": {}}
    det_lock = threading.Lock()  # one analysis at a time: the CPU is the bottleneck
    tmp = Path(tempfile.mkdtemp(prefix="visionlens-"))

    def get_detector() -> Detector:
        if state["detector"] is None:
            from .detect import load_detector

            state["detector"] = load_detector(os.environ.get("VISIONLENS_DETECTOR", "dfine-nano"))
        return state["detector"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        shutil.rmtree(tmp, ignore_errors=True)

    app = FastAPI(title="VisionLens", version=__version__, lifespan=lifespan)

    # ------------------------------------------------------------------ helpers
    def purge() -> None:
        now = time.time()
        jobs: dict[str, _Job] = state["jobs"]
        old = [j for j in jobs.values() if now - j.created > cfg.keep_seconds]
        extra = sorted(jobs.values(), key=lambda j: j.created)[: max(0, len(jobs) - cfg.keep_jobs)]
        for j in {*old, *extra}:
            if j.state in ("done", "error"):
                shutil.rmtree(tmp / j.id, ignore_errors=True)
                jobs.pop(j.id, None)
        sess: dict[str, _Session] = state["sessions"]
        for sid in [k for k, s in sess.items() if now - s.last > cfg.session_ttl]:
            sess.pop(sid, None)

    def run_job(
        job: _Job, src: Path, describer: Describer | None, stride: int, delete_source: bool
    ) -> None:
        folder = tmp / job.id
        try:
            job.emit("status", {"state": "queued"})
            with det_lock:
                job.state = "running"
                job.emit("status", {"state": "running"})
                det = get_detector()
                is_img = src.suffix.lower() in IMAGE_EXT
                playback = folder / ("playback.jpg" if is_img else "playback.webm")
                if is_img:
                    img = cv2.imread(str(src))
                    if img is None:
                        raise ValueError("could not read the image")
                    from .video import resize_to_width

                    cv2.imwrite(
                        str(playback), resize_to_width(img, 960), [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    job.media_type = "image/jpeg"

                last_report = [0.0]

                def on_frame(fr, total):
                    now = time.time()
                    if now - last_report[0] > 0.25:
                        last_report[0] = now
                        job.emit("progress", {"i": fr.i, "total": total, "tracks": len(fr.tracks)})

                res = analyze_file(
                    playback if is_img else src,
                    det,
                    PipelineConfig(stride=stride),
                    describer,
                    playback_path=None if is_img else playback,
                    max_width=cfg.max_width,
                    target_fps=cfg.fps,
                    max_seconds=cfg.max_seconds,
                    on_frame=on_frame,
                    on_token=lambda t: job.emit("token", {"text": t}),
                    on_model_token=lambda t: job.emit("model_token", {"text": t}),
                    name=src.stem if not delete_source else "your upload",
                )
                job.media = playback
                job.result = res
            job.state = "done"
            job.emit("done", {"job_id": job.id})
        except Exception as exc:  # noqa: BLE001 - reported to the client
            job.state, job.error = "error", str(exc)
            job.emit("error", {"message": str(exc)})
        finally:
            if delete_source:
                src.unlink(missing_ok=True)

    # ------------------------------------------------------------------- routes
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            detector=os.environ.get("VISIONLENS_DETECTOR", "dfine-nano"),
            detector_loaded=state["detector"] is not None,
            limits={
                "max_upload_mb": cfg.max_upload_mb,
                "max_seconds": cfg.max_seconds,
                "max_width": cfg.max_width,
            },
            busy=det_lock.locked(),
        )

    @app.get("/api/samples")
    def samples() -> JSONResponse:
        p = Path(cfg.docs_dir) / "demo" / "index.json"
        if not p.exists():
            return JSONResponse({"clips": []})
        return JSONResponse(json.loads(p.read_text("utf-8")))

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(
        file: UploadFile | None = File(default=None),
        sample: str | None = Form(default=None),
        describer: str = Form(default="template"),
        stride: int = Form(default=cfg.stride),
    ) -> AnalyzeResponse:
        purge()
        if not 1 <= stride <= 10:
            raise HTTPException(422, "stride must be between 1 and 10")
        if describer not in ("template", "llm", "openai", "anthropic"):
            raise HTTPException(422, "unknown describer")
        job = _Job()
        folder = tmp / job.id
        folder.mkdir(parents=True)
        if sample:
            idx = Path(cfg.docs_dir) / "demo" / "index.json"
            clips = json.loads(idx.read_text("utf-8"))["clips"] if idx.exists() else []
            match = next((c for c in clips if c["id"] == sample and not c.get("simulated")), None)
            if match is None:
                raise HTTPException(404, "unknown sample")
            src = Path(cfg.docs_dir) / "demo" / match["video"]
            delete_source = False
        elif file is not None:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in VIDEO_EXT | IMAGE_EXT:
                raise HTTPException(415, f"unsupported file type {ext or '(none)'}")
            src = folder / f"upload{ext}"
            size, limit = 0, cfg.max_upload_mb * 1024 * 1024
            with src.open("wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        fh.close()
                        shutil.rmtree(folder, ignore_errors=True)
                        raise HTTPException(413, f"file is larger than {cfg.max_upload_mb} MB")
                    fh.write(chunk)
            delete_source = True
        else:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(422, "send a file or a sample name")
        try:
            desc = make_describer(describer)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, str(exc)) from exc
        state["jobs"][job.id] = job
        threading.Thread(
            target=run_job, args=(job, src, desc, stride, delete_source), daemon=True
        ).start()
        return AnalyzeResponse(job_id=job.id)

    def get_job(job_id: str) -> _Job:
        job = state["jobs"].get(job_id)
        if job is None:
            raise HTTPException(404, "unknown or expired job")
        return job

    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str) -> JobStatus:
        j = get_job(job_id)
        return JobStatus(job_id=j.id, state=j.state, error=j.error)

    @app.get("/api/jobs/{job_id}/stream")
    async def stream(job_id: str, request: Request) -> StreamingResponse:
        job = get_job(job_id)

        async def gen():
            i = 0
            while True:
                with job.lock:
                    batch = job.msgs[i:]
                for name, data in batch:
                    yield _sse(name, data)
                i += len(batch)
                if job.state in ("done", "error") and not batch:
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.08)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str) -> JSONResponse:
        j = get_job(job_id)
        if j.result is None:
            raise HTTPException(409, f"job is {j.state}")
        return JSONResponse(json.loads(j.result.model_dump_json()))

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str) -> FileResponse:
        j = get_job(job_id)
        if j.media is None or not j.media.exists():
            raise HTTPException(409, f"job is {j.state}")
        return FileResponse(j.media, media_type=j.media_type)

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        j = get_job(job_id)
        shutil.rmtree(tmp / j.id, ignore_errors=True)
        state["jobs"].pop(j.id, None)

    @app.post("/api/frame", response_model=FrameResponse)
    async def frame(
        request: Request, session: str = Query(min_length=8, max_length=64)
    ) -> FrameResponse:
        """Webcam mode: the browser posts one JPEG at a time; tracker state lives per session."""
        purge()
        body = await request.body()
        if not body or len(body) > 3 * 1024 * 1024:
            raise HTTPException(413 if body else 422, "send one JPEG under 3 MB")
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(415, "not a decodable image")
        sessions: dict[str, _Session] = state["sessions"]
        if session not in sessions:
            if len(sessions) >= cfg.max_sessions:
                raise HTTPException(429, "too many live sessions")
            sessions[session] = _Session()
        sess = sessions[session]
        from .schemas import FrameResult
        from .video import resize_to_width

        img = resize_to_width(img, 480)
        h, w = img.shape[:2]
        t0 = time.perf_counter()

        def work() -> list[TrackOut]:
            with det_lock:
                dets = get_detector().detect(img, sess.calls)
            return sess.tracker.step(dets, sess.calls)

        tracks = await asyncio.to_thread(work)
        ms = (time.perf_counter() - t0) * 1000
        sess.last = time.time()
        sess.frames.append(
            FrameResult(i=sess.calls, t=round(sess.last - sess.t0, 2), tracks=tracks)
        )
        del sess.frames[:-400]
        sess.calls += 1
        text = None
        if sess.calls % 12 == 0 and len(sess.frames) > 8:
            span = max(sess.frames[-1].t - sess.frames[0].t, 1e-3)
            fps = len(sess.frames) / span
            ev = extract_events(sess.frames, w, h, fps, EventConfig())
            from .describe import Facts, TemplateDescriber

            facts = Facts(span, w, h, ev, summarize(sess.frames, fps))
            text = TemplateDescriber().describe(facts).text
        return FrameResponse(
            session=session,
            width=w,
            height=h,
            tracks=tracks,
            ms=round(ms, 1),
            count=len(tracks),
            description=text,
        )

    docs = Path(cfg.docs_dir)
    if docs.exists():
        index_html = docs / "index.html"

        @app.get("/", include_in_schema=False)
        def home() -> HTMLResponse:
            # The page only probes for a server when this flag says one is serving it, so the
            # static copy on GitHub Pages never makes a request that would 404.
            text = index_html.read_text("utf-8").replace(
                'name="vl-server" content="0"', 'name="vl-server" content="1"'
            )
            return HTMLResponse(text, headers={"Cache-Control": "no-cache"})

        app.mount("/", StaticFiles(directory=docs, html=True), name="ui")
    return app
