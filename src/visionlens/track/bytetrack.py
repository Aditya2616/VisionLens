"""A ByteTrack-style multi-object tracker built on a SORT Kalman core.

Per detector step:
  1. every track is predicted forward (this also happens on frames the detector skips);
  2. confirmed tracks are matched to high-score detections (Hungarian on IoU);
  3. tentative tracks are matched to the high-score detections that are left;
  4. confirmed tracks that were matched last step and are still unmatched get a second
     chance against low-score detections (the ByteTrack idea: occluded objects score low);
  5. unmatched high-score detections above `new_thr` start tentative tracks; a track gets
     its public id once it has been matched `min_hits` times;
  6. confirmed tracks that stay unmatched coast on the Kalman prediction and are dropped
     after `max_misses` detector steps.

Optional `ocr_update` re-runs the filter along an interpolated path over a gap when a lost
track is re-found (the observation-centric re-update idea from OC-SORT).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from ..schemas import Detection, TrackOut
from .iou import associate, iou_matrix
from .kalman import BoxKalman, xyxy_to_z


@dataclass
class TrackerConfig:
    high_thr: float = 0.45  # detections at or above this are "high score"
    low_thr: float = 0.15  # detections below this are ignored
    new_thr: float = 0.5  # a track may only start from a detection this confident
    match_iou: float = 0.3  # minimum IoU for the first association
    second_iou: float = 0.3  # minimum IoU for the low-score association
    max_misses: int = 6  # detector steps a confirmed track may coast before removal
    min_hits: int = 2  # matches needed before a track gets an id
    coast_display: int = 3  # detector steps a coasting track stays visible in the output
    class_gate: bool = True  # never match across classes
    two_stage: bool = True  # use the low-score second pass
    ocr_update: bool = False  # observation-centric re-update on re-association

    def as_dict(self) -> dict:
        return asdict(self)


class _Track:
    __slots__ = (
        "kf",
        "votes",
        "score",
        "hits",
        "misses",
        "id",
        "confirmed",
        "matched_now",
        "last_obs_frame",
        "last_obs_box",
        "snap",
    )

    def __init__(self, det: Detection, frame_idx: int) -> None:
        self.kf = BoxKalman(det.box)
        self.votes: dict[str, float] = defaultdict(float)
        self.votes[det.cls] += det.score
        self.score = det.score
        self.hits = 1
        self.misses = 0
        self.id = 0
        self.confirmed = False
        self.matched_now = True
        self.last_obs_frame = frame_idx
        self.last_obs_box = det.box
        self.snap = self.kf.snapshot()

    @property
    def cls(self) -> str:
        return max(self.votes.items(), key=lambda kv: kv[1])[0]


class ByteTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._frame = -1

    # ------------------------------------------------------------------ public
    def step(self, dets: list[Detection] | None, frame_idx: int | None = None) -> list[TrackOut]:
        """Advance one frame. Pass dets=None on frames the detector skipped."""
        self._frame = self._frame + 1 if frame_idx is None else frame_idx
        for tr in self._tracks:
            tr.kf.predict()
            tr.matched_now = False
        if dets is not None:
            self._associate(dets)
        return self._output(detector_frame=dets is not None)

    @property
    def n_confirmed(self) -> int:
        return sum(1 for t in self._tracks if t.confirmed)

    # ----------------------------------------------------------------- internals
    def _sim(self, tracks: list[_Track], dets: list[Detection]) -> np.ndarray:
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))
        tb = np.array([t.kf.box for t in tracks])
        db = np.array([d.box for d in dets])
        sim = iou_matrix(tb, db)
        if self.cfg.class_gate:
            for i, t in enumerate(tracks):
                c = t.cls
                for j, d in enumerate(dets):
                    if d.cls != c:
                        sim[i, j] = 0.0
        return sim

    def _apply(self, tr: _Track, det: Detection) -> None:
        gap = self._frame - tr.last_obs_frame
        if self.cfg.ocr_update and tr.misses > 0 and gap > 1:
            self._reupdate(tr, det.box, gap)
        else:
            tr.kf.update(det.box)
        tr.votes[det.cls] += det.score
        tr.score = det.score
        tr.hits += 1
        tr.misses = 0
        tr.matched_now = True
        tr.last_obs_frame = self._frame
        tr.last_obs_box = det.box
        tr.snap = tr.kf.snapshot()
        if not tr.confirmed and tr.hits >= self.cfg.min_hits:
            tr.confirmed = True
            tr.id = self._next_id
            self._next_id += 1

    def _reupdate(self, tr: _Track, new_box: tuple, gap: int) -> None:
        """Replay the filter over the gap along a straight line between the two observations."""
        tr.kf.restore(tr.snap)
        z0 = xyxy_to_z(tr.last_obs_box)
        z1 = xyxy_to_z(new_box)
        for k in range(1, gap):
            tr.kf.predict()
            zk = z0 + (z1 - z0) * (k / gap)
            w = float(np.sqrt(max(zk[2] * zk[3], 1e-6)))
            h = float(max(zk[2], 1e-6) / max(w, 1e-6))
            tr.kf.update((zk[0] - w / 2, zk[1] - h / 2, zk[0] + w / 2, zk[1] + h / 2))
        tr.kf.predict()
        tr.kf.update(new_box)

    def _associate(self, dets: list[Detection]) -> None:
        cfg = self.cfg
        high = [d for d in dets if d.score >= cfg.high_thr]
        low = [d for d in dets if cfg.low_thr <= d.score < cfg.high_thr]
        confirmed = [t for t in self._tracks if t.confirmed]
        tentative = [t for t in self._tracks if not t.confirmed]

        # 1. confirmed tracks against high-score detections
        m1, un_t1, un_h1 = associate(self._sim(confirmed, high), cfg.match_iou)
        for ti, di in m1:
            self._apply(confirmed[ti], high[di])
        left_high = [high[j] for j in un_h1]

        # 2. tentative tracks against the remaining high-score detections
        m2, un_tent, un_h2 = associate(self._sim(tentative, left_high), cfg.match_iou)
        for ti, di in m2:
            self._apply(tentative[ti], left_high[di])
        new_candidates = [left_high[j] for j in un_h2]

        # 3. ByteTrack second pass: still-unmatched, recently matched tracks vs low detections
        un_confirmed = [confirmed[i] for i in un_t1]
        if cfg.two_stage and low:
            pool = [t for t in un_confirmed if t.misses == 0]
            m3, _, _ = associate(self._sim(pool, low), cfg.second_iou)
            for ti, di in m3:
                self._apply(pool[ti], low[di])

        # 4. misses and removal
        for t in un_confirmed:
            if not t.matched_now:
                t.misses += 1
        drop = {id(tentative[i]) for i in un_tent}  # unconfirmed tracks get one chance only
        self._tracks = [
            t
            for t in self._tracks
            if id(t) not in drop and not (t.confirmed and t.misses > cfg.max_misses)
        ]

        # 5. births
        for d in new_candidates:
            if d.score >= cfg.new_thr:
                tr = _Track(d, self._frame)
                if cfg.min_hits <= 1:
                    tr.confirmed = True
                    tr.id = self._next_id
                    self._next_id += 1
                self._tracks.append(tr)

    def _output(self, detector_frame: bool) -> list[TrackOut]:
        out: list[TrackOut] = []
        for t in self._tracks:
            if not t.confirmed or t.misses > self.cfg.coast_display:
                continue
            b = t.kf.box
            out.append(
                TrackOut(
                    id=t.id,
                    box=(round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)),
                    cls=t.cls,
                    score=round(t.score, 3),
                    detected=bool(detector_frame and t.matched_now),
                )
            )
        out.sort(key=lambda o: o.id)
        return out
