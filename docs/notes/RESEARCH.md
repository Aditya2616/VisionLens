# VisionLens: research notes

Notes I made before building. Items marked **[measured]** were quick runs on my machine (Ryzen 5 8500G, 12 threads,
CPU-only PyTorch 2.11, transformers 5.7, opencv-python 5.0.0) while the CPU was busy with other jobs, so absolute
timings are pessimistic and noisy. The numbers in the README come from `visionlens bench`, not from this file. Items marked
**[claimed]** come from papers or model cards and are verified before they reach the README.

## 1. Detector (stage 1)

| Model (HF id) | Params | Licence | Weights | Notes |
|---|---|---|---|---|
| D-FINE nano (`ustc-community/dfine-nano-coco`) | 3.8M | Apache-2.0 | 15 MB | 42.8 COCO AP [claimed, paper]. Native in transformers 5.7. |
| D-FINE small (`ustc-community/dfine-small-coco`) | 10.4M | Apache-2.0 | 42 MB | 48.5 AP [claimed] |
| RT-DETR r18vd (`PekingU/rtdetr_r18vd`) | 20.2M | Apache-2.0 | 81 MB | 46.5 AP [claimed] |
| RT-DETRv2 r18vd (`PekingU/rtdetr_v2_r18vd`) | 20.2M | Apache-2.0 | 81 MB | ~47.9 AP [claimed] |
| YOLOS-tiny (`hustvl/yolos-tiny`) | 6.5M | Apache-2.0 | 52 MB | ~28.7 AP [claimed]; variable-size input |
| DETR ResNet-50 (`facebook/detr-resnet-50`) | 41.5M | Apache-2.0 | 160 MB | 42 AP [claimed]; slowest |
| OWL-ViT / OWLv2 base | ~150M | Apache-2.0 | 1.2 GB | open-vocabulary; too slow for per-frame use, rejected for the main path |

Excluded on licence: Ultralytics YOLOv5/8/11 (AGPL-3.0, would force the repo to AGPL or a commercial licence). That is
why a DETR-family model is used.

**Speed.** My first quick runs on a busy CPU ranked the models in the order D-FINE nano, D-FINE small, RT-DETR r18vd (fastest
to slowest) and put YOLOS-tiny, RT-DETRv2 and DETR R50 behind them. The numbers I quote now come only from
`visionlens bench` (see the README), with the load stated; this file does not repeat any.

**Choice:** D-FINE nano default (smallest, fastest, Apache-2.0, 15 MB), D-FINE small as the "accurate" option, RT-DETR
r18vd as a reference row in the comparison table. Detection runs every Nth frame (stride); tracker prediction covers
the gaps, so a detector that only manages a few frames per second can still drive a smoother overlay, which is why a tracker is
needed.

Sanity check on real footage [measured]: D-FINE nano on the CC0 pedestrian clip (1920x1080) finds persons, bicycles,
bags and a stop sign sensibly, but gives nested near-duplicate boxes on heavily occluded people and some
handbag/backpack false positives. The tracker's low-score pass and class handling have to cope with that.

## 2. Tracking (stage 2)

- **SORT** (Bewley 2016): constant-velocity Kalman filter per box, IoU cost, Hungarian assignment. Fast, no appearance
  model, ID switches under occlusion.
- **DeepSORT** (Wojke 2017): adds a CNN appearance embedding and cascade matching. Fewer ID switches, but needs a
  re-ID network (more weights and CPU).
- **ByteTrack** (Zhang 2021): SORT-style, with a second association pass that matches leftover tracks to
  low-confidence detections, recovering occluded objects a threshold would discard. No re-ID.
- **OC-SORT** (Cao 2022): observation-centric re-update of the Kalman state after occlusion plus a velocity-direction
  consistency term; more robust to non-linear motion and gaps.

**Choice:** implement a ByteTrack-style two-stage association myself on a SORT Kalman core (state centre x/y, area,
aspect ratio and their velocities), Hungarian assignment via `scipy.optimize.linear_sum_assignment`, class-aware
gating, lifecycle (tentative, confirmed, lost, removed), plus an OC-SORT-style re-update on re-association as an
ablation toggle. Prediction-only on non-detector frames. No tracker library is used or vendored.

## 3. Evaluation

Metrics: **MOTA** (FP/FN/ID-switch aggregate, dominated by detection), **IDF1** (identity-level F1 from a global
trajectory match; Ristani 2016), **HOTA** (Luiten 2020; geometric mean of detection and association accuracy over IoU
thresholds). TrackEval is the reference implementation but is large with extra dependencies; I implement MOTA, IDF1,
ID switches, fragmentation and HOTA (DetA/AssA over 19 thresholds) in ~200 lines and check them against hand-computed
toy cases in tests (optionally cross-checked locally against `motmetrics`, not a dependency).

Ground truth:
- MOT17/MOT20 are **CC BY-NC-SA 3.0** (non-commercial, share-alike) per the secondary sources I found; not
  redistributable under MIT, so they are not used or committed.
- **Synthetic sequences with exact ground truth, rendered by me** (primary): agents on linear, curved and crossing
  paths, static occluders, entering/leaving, scripted near-collisions. Two configurations: (a) oracle detections with
  injected noise (jitter, misses, false positives, score noise) to isolate tracker quality across a noise sweep;
  (b) a real detector on rendered frames, which I did not run: the rendered silhouettes are not natural images, so
  it would not be meaningful.
- Real footage has no ground truth: only descriptive statistics (track count, mean track length), labelled as not an
  accuracy measure.

## 4. Describe (stage 3)

1. **Event extraction** (pure Python, deterministic, testable): `enter`, `leave` (with edge), `move` (direction and
   speed bucket), `linger` (slow for N seconds), `count_change`, `proximity` (two tracks close for N frames, with
   start/end). Typed records with time, track ids, class.
2. **Language:** a deterministic **template describer** is baseline and fallback. An optional **small local LLM**
   rewrites the event list into fluent text, constrained to the supplied facts.

LLM candidates (Apache-2.0): Qwen2.5-0.5B-Instruct, SmolLM2-360M-Instruct and Qwen2.5-1.5B-Instruct. I built the rewrite
around the Qwen2.5 models and compared the 0.5B and 1.5B ones on the same facts (the results are in the README: guard
acceptance and seconds per rewrite). Generation runs once per clip in a worker thread and is streamed, while the
template text shows instantly.

Faithfulness guard: after generation, check the text sentence by sentence against the event list (objects, counts, times,
directions, event verbs); otherwise fall back to the template and flag it in the UI. The rules are in
`src/visionlens/describe/guard.py`.

**Optional scene captioner:** SmolVLM-256M-Instruct (Apache-2.0, 513 MB) or Florence-2-base (MIT, 463 MB in the
`florence-community` native conversion). Moondream2 is Apache-2.0 on the hub (one source suggested BSL; check card) but
~2B params / 3.9 GB: over budget. The caption adds scene context the events cannot ("a pedestrian street with shops").
**Cut:** I did not build the captioner. Detection, tracking and events already give the description its facts, and a
third model on a shared CPU was not worth the latency.

Hosted adapters (OpenAI, optionally Anthropic) sit behind the same `Describer` interface, selected by env var, tested
only against a stubbed HTTP transport.

## 5. Video I/O here [measured]

opencv-python 5.0.0 (PyPI wheel with bundled FFmpeg), via write-then-read-back tests:
- `VP90` into `.webm`: works (20/20 frames read back). Chromium plays VP9 WebM. **Chosen for video output.**
- `avc1`/`H264`: `isOpened()` says true but encoding fails (needs `openh264-2.5.0-win64.dll`, absent). Unusable.
- `mp4v`: encodes, but Chromium does not play MPEG-4 Part 2. Unusable for the browser.
- Reading WebM: sequential reads work. **Seeking is unreliable**: Wikimedia WebM reported FPS 1000 and a wrong frame
  count, and `set(POS_FRAMES)` returned only one frame. So: sequential reads, timestamps from frame index and declared
  FPS, and a constant-frame-rate re-encode step for sample clips.

Design consequence: the UI does not depend on a server-rendered annotated video. The server (and the static demo)
supplies the original video plus per-frame JSON; the browser draws boxes, IDs, trails and the scanner reveal on a
canvas synced with `requestVideoFrameCallback`. Toggles then cost nothing and it works as a static site. OpenCV still
does decode, resize, CFR transcoding, synthetic rendering and an `annotate` CLI command that writes an annotated WebM.

## 6. Sample data (licence-clean)

Wikimedia Commons files marked CC0 (checked through the Commons API licence field):
- Candidates I looked at first: "Video Codec Test pedestrian area 1080p25" (Taurus Media Technik, CC0, 15 s, 1920x1080,
  people and bicycles) and "DiagonalCrosswalkYongeDundas" (CC0, busy crosswalk, a 73 MB source). I did not ship them:
  I preferred wide shots in which individual people cannot be recognised.
- Shipped instead: two wide fixed-camera shots, one with distant pedestrians and one without people (credits in
  `data/samples.json` and the README), cut to short excerpts by `scripts/fetch_samples.py`.
- Generated clips from my scene generator, with ground truth, committed (small).
- Attribution is kept in `data/samples.json` and the README even though CC0 does not require it. Wikimedia throttles bare downloads
  (HTTP 429): the fetch script sends a descriptive User-Agent and retries.

## 7. Prior art (behaviour only, nothing copied)

Roboflow `supervision`/`trackers` (MIT), `ifzhang/ByteTrack`, `noahcao/OC_SORT`, `abewley/sort` (GPL-3.0, read only),
`JonathonLuiten/TrackEval` (MIT), HF transformers D-FINE/RT-DETR docs, SmolVLM paper. Ultralytics tracking (AGPL) avoided.

## Sources

- SORT https://arxiv.org/abs/1602.00763 , DeepSORT https://arxiv.org/abs/1703.07402 , ByteTrack https://arxiv.org/abs/2110.06864 , OC-SORT https://arxiv.org/abs/2203.14360
- HOTA https://arxiv.org/abs/2009.07736 , MOTChallenge https://arxiv.org/abs/2010.07548
- DETR https://arxiv.org/abs/2005.12872 , YOLOS https://arxiv.org/abs/2106.00666 , RT-DETR https://arxiv.org/abs/2304.08069 , RT-DETRv2 https://arxiv.org/abs/2407.17140 , D-FINE https://arxiv.org/abs/2410.13842 , OWL-ViT https://arxiv.org/abs/2205.06230
- SmolVLM https://arxiv.org/abs/2504.05299 , Florence-2 https://arxiv.org/abs/2311.06242
- https://huggingface.co/docs/transformers/model_doc/rt_detr_v2 , https://huggingface.co/ustc-community/dfine-nano-coco
- Licences read from the Hugging Face hub API for every model listed (Apache-2.0, except Florence-2, MIT).
- The arXiv ids and the [claimed] AP figures come from the papers and model cards; the README quotes them only as claims.
