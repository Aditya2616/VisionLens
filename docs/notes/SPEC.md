# VisionLens: specification

A three-stage pipeline, Detect -> Track -> Describe, that turns video into labelled boxes, stable identities and a
plain-language description, with a UI that is the full-size version of the portfolio's scanner simulation.

## 1. Data flow

```
frames (video / image / webcam)
  -> Detector  (HF D-FINE nano, every `stride` frames, COCO classes, low score floor 0.15)
  -> Tracker   (own ByteTrack-style: Kalman + IoU + Hungarian; predict-only between detections)
  -> Events    (enter / leave / move / linger / count_change / proximity)
  -> Describer (template baseline | local LLM with regenerate-on-rejection | hosted adapter) -> streamed text
  -> Scene captioner (optional, opt-in: a small vision-language model on three key frames; separate labelled line)
  -> Result JSON {meta, frames:[{t, dets, tracks}], events, description, timings}
```

Result JSON (pydantic models in `schemas.py`) is the one contract between core, CLI, API, front
end and the static demo.

## 2. Package layout (`src/visionlens/`)

| Module | Responsibility |
|---|---|
| `schemas.py` | `Detection`, `TrackOut`, `FrameResult`, `Event`, `Sentence`, `Description`, `Timings`, `Result` |
| `video.py` | sequential OpenCV reader with CFR timestamps, VP9 WebM writer, resize, probe |
| `detect/` | `Detector` protocol; HF adapter (D-FINE, RT-DETR); oracle/noisy detector for evaluation |
| `track/` | box Kalman filter, IoU, two-stage association, lifecycle |
| `events.py` | track history to typed events, thresholds in a config dataclass |
| `caption.py` | optional scene captioner (SmolVLM), key-frame choice, cross-check of named classes against the detector |
| `models.py` | which models `fetch-models` downloads |
| `describe/` | `Describer` interface; grouped template; local causal LM; `guard.py` faithfulness guard; OpenAI/Anthropic adapters |
| `pipeline.py` | orchestration as a generator (progress and partial results for streaming) |
| `synth.py` | deterministic synthetic scenes with exact ground truth |
| `metrics.py` | MOTA, IDF1, ID switches, fragmentation, HOTA/DetA/AssA |
| `bench.py` | reproducible benchmarks (detector table, tracker sweeps and ablations, end-to-end, rewrite evaluation) |
| `api.py`, `cli.py` | FastAPI service; `visionlens run / annotate / bench / serve / fetch-models / make-synthetic / record-demo` |

## 3. Decisions

- Stride detection with Kalman prediction in between; IDF1/HOTA vs stride evaluated.
- Own tracker with real unit tests: Kalman convergence, assignment, birth/death, occlusion recovery, ID stability on
  crossing paths, class gating, second pass beating single pass on noisy input.
- Events are deterministic and are the source of truth; the LLM only rephrases, with a faithfulness guard and fallback.
- Browser draws overlays; the server returns original video (CFR WebM) plus Result JSON. Same for static demo.
- Models fetched by `visionlens fetch-models` (the default detector; `--llm` adds the rewrite model; `--caption` adds
  the scene captioner). Sizes are in the README. Nothing is committed.

## 4. API (FastAPI, port 8106)

- `GET /health`: status, whether the detector is loaded, the upload limits.
- `GET /api/samples`; `POST /api/analyze` (video or image upload, or a bundled sample; options stride, describer and caption)
  returns `job_id`; `GET /api/jobs/{id}` is the status; `GET /api/jobs/{id}/stream` is SSE (`status`, `progress`,
  `token`, `model_token`, `done`, `error`); `GET /api/jobs/{id}/result` and `/video` return the result and the media. The describers and the captioner are
  created once and reused.
- `POST /api/frame`: single-frame detect+track for webcam live mode (browser `getUserMedia`, JPEG at ~3-5 FPS,
  per-session tracker with expiry).
- Typed models, upload size/duration/resolution caps, uploads deleted after processing.

## 5. UI concept

Vanilla ES modules + GSAP (CDN), portfolio tokens (void #07080a, lime #d7ff3f, orange only for warnings; Inter Tight,
Instrument Serif italic, JetBrains Mono). The sources are in `docs/`; the API serves the same files.

Desktop: video stage with canvas overlay on the left; description panel (streaming text, tabs Template and Model)
and counters on the right; an **event timeline** below (a lane per track id, glyphs for enter/leave/linger/
proximity, playhead synced to video, click to seek). Phone: stacked, 44 px controls, timeline scrolls inside its own
box, nothing hover-dependent.

Motion, all from real data:
- **Scanner sweep:** a lime scan line crosses the frame; each box appears as the line passes its object, with label and
  confidence counting up. It plays on the first tap and on demand; reduced motion shows everything at once.
- **Stage toggles:** Raw detections (thin dashed, flickering between detector frames) / Tracks (solid, stable id chips,
  smoothed; chips are compact and nudged apart where boxes are crowded; tapping a box selects the track) / Trails / Events (pulses at event positions). Flipping raw vs tracks shows what the tracker adds.
- **Stage strip** Detect -> Track -> Describe with the real per-stage milliseconds and the device the detector ran on;
  the end-to-end FPS is in its tooltip.
- **Streaming description** token by token; sentences cite events, and tapping a sentence highlights its tracks and
  timeline marks.
- **Modes:** Demo (recorded runs, static-hostable, labelled "Recorded run, not live") and Live (backend detected via
  `/health`: upload video or image, bundled samples, webcam).
- Canvas DPR capped at 2, overlay drawn in a `requestAnimationFrame` loop from the video's current time, paused
  off-screen.

## 6. Test plan

Fast and offline by default: Kalman, IoU/assignment, tracker lifecycle and ID stability, events from hand-built tracks,
grouped-template tests, metrics against hand-computed cases, synth determinism, schemas, VP9 write/read round trip, API
with a fake detector (health, analyze, SSE, limits, bad file), hosted adapters with stubbed HTTP, LLM faithfulness guard
with a stub model and with deliberately corrupted rewrites, the CLI output location, and a test that the README's
generated blocks and prose facts match their sources. `slow` marker: real D-FINE inference, real LLM generation. CI: ruff + fast tests.
`scripts/ui_check.py` (own headless Chromium) captures 1440x900 and 390x844 touch screenshots and console errors; the
webcam path is tested with Chromium's fake-device flags.

## 7. Results to report (all from `visionlens bench`)

1. Detector table: ms/frame and FPS on the CPU and on the GPU (median, p95, every frame the clip yields up to 40 after
   a 3-frame warm-up, load sampled); AP only cited from the sources.
2. Tracker vs synthetic ground truth: MOTA/IDF1/HOTA/IDsw across noise levels and detector stride; ablations
   (single pass vs second pass, with/without OC-SORT re-update, IoU-only vs class-gated) and a naive per-frame
   baseline.
3. End-to-end per-stage latency on the real clip; guard acceptance and seconds per rewrite for each local model.
4. Caveats stated: synthetic is easier than real; the real clip has no ground truth; timings depend on load.

## 8. Done means

The tests and lint pass, the UI is checked at desktop and phone size, and: tracker beats the naive baseline on the synthetic set with the margin shown; real clip
shows stable IDs by visual check; description works with the template alone, with the local LLM, and fails safe;
static demo plays with scanner reveal, toggles and timeline over a plain static server; uploaded video and image work in
live mode; webcam works under fake-device Chromium; 390 px layout verified; repo size reported.

## 9. Risks

- CPU contention makes timings noisy; the benchmark samples the load and the README states it.
- D-FINE nano is weaker on small or occluded objects; D-FINE small is selectable and the low-score pass helps.
- OpenCV WebM seeking is unreliable: sequential read and CFR re-encode.
- No H.264 encoding here: annotated-video export is VP9 WebM; the UI draws overlays itself so does not need it.
- The local model is slow on a CPU and may paraphrase badly: template first, guard and fallback.
- Static GitHub Pages has no backend, so demo mode replays recorded results; upload and webcam need the local server.
- Wikimedia CC0 clips are large: fetched and trimmed by script; only a small trimmed clip committed.
