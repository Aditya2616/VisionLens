// Canvas overlay: raw detections, tracks, trails, events, ground truth, and the scanner reveal.
// Every pixel it draws comes from the Result JSON; nothing here is decorative state.
import { clamp, easeOut, lerp, reduced } from './util.js';

const LIME = '#d7ff3f';
const BONE = '#edece6';
const EMBER = '#ff7a3d';
const MONO = '"JetBrains Mono", ui-monospace, Consolas, monospace';
const SCAN_MS = 1700;
const TRAIL = 28; // frames
const INITIAL = { person: 'P', car: 'C', bus: 'B', truck: 'T', bicycle: 'Y', motorcycle: 'M', dog: 'D', cat: 'A', bird: 'I', horse: 'H', boat: 'O', train: 'R' };

export class Overlay {
  constructor(canvas) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = 1;
    this.W = 2;
    this.H = 2;
    this.data = null;
    this.scan = null; // {t0}
    this.scanDone = true;
    this.revealed = new Map(); // track id -> performance.now() when first drawn
    this.selected = new Set();
    this.found = 0;
    this.visible = true;
  }

  resize() {
    const r = this.c.getBoundingClientRect();
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.W = Math.max(2, Math.round(r.width * this.dpr));
    this.H = Math.max(2, Math.round(r.height * this.dpr));
    if (this.c.width !== this.W || this.c.height !== this.H) {
      this.c.width = this.W;
      this.c.height = this.H;
    }
  }

  setData(result, liveMode = false) {
    this.data = result;
    if (!liveMode) {
      this.revealed.clear();
      this.selected.clear();
      this.found = 0;
      this.scan = null;
      this.scanDone = true;
    }
    this.live = liveMode;
    if (!result) return;
    const frames = result.frames;
    // per-id position history, plus the most recent frame that carried raw detections
    this.hist = new Map();
    this.posIdx = frames.map(() => new Map());
    this.lastDet = new Array(frames.length).fill(-1);
    let ld = -1;
    frames.forEach((fr, i) => {
      if (fr.dets) ld = i;
      this.lastDet[i] = ld;
      for (const t of fr.tracks) {
        if (!this.hist.has(t.id)) this.hist.set(t.id, []);
        const arr = this.hist.get(t.id);
        arr.push([i, (t.box[0] + t.box[2]) / 2, t.box[3]]);
        this.posIdx[i].set(t.id, arr.length - 1);
      }
    });
    this.events = result.events || [];
  }

  startScan() {
    if (reduced()) { this.scanDone = true; this.scan = null; return; }
    this.scan = { t0: performance.now() };
    this.scanDone = false;
    this.revealed.clear();
    this.found = 0;
  }

  frameIndex(t) {
    const d = this.data;
    if (!d || !d.frames.length) return 0;
    return clamp(Math.floor(t * d.meta.fps + 1e-6), 0, d.frames.length - 1);
  }

  draw(t, layers, now = performance.now()) {
    const { ctx, W, H, data } = this;
    ctx.clearRect(0, 0, W, H);
    if (!data || !data.frames.length) return;
    const fi = this.frameIndex(t);
    const fr = data.frames[fi];
    const sx = W / data.meta.width;
    const sy = H / data.meta.height;
    const k = this.dpr;

    // scanner progress
    let scanX = -1;
    if (this.scan) {
      const p = (now - this.scan.t0) / SCAN_MS;
      if (p >= 1) { this.scan = null; this.scanDone = true; } else scanX = p * W;
    }

    this.labelRects = [];
    if (layers.gt && data.ground_truth) this._gt(data.ground_truth[fi] || [], sx, sy, k);
    if (layers.dets) this._dets(data.frames[this.lastDet[fi]]?.dets || [], sx, sy, k, scanX);
    if (layers.trails) this._trails(fr, fi, sx, sy, k, scanX);
    if (layers.tracks) this._tracks(fr, sx, sy, k, scanX, now);
    if (layers.events) this._events(t, sx, sy, k);
    if (scanX >= 0) this._scanner(scanX, k);

    ctx.font = `500 ${10 * k}px ${MONO}`;
    ctx.fillStyle = 'rgba(237,236,230,0.55)';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(`FRAME ${String(fi).padStart(4, '0')}`, 12 * k, 20 * k);
    if (scanX >= 0) {
      ctx.fillStyle = LIME;
      ctx.fillText(`SCAN ${this.found}`, 12 * k, 34 * k);
    }
  }

  _gt(list, sx, sy, k) {
    const { ctx } = this;
    ctx.save();
    ctx.lineWidth = 1 * k;
    ctx.strokeStyle = 'rgba(237,236,230,0.85)';
    ctx.fillStyle = 'rgba(237,236,230,0.85)';
    ctx.font = `500 ${8.5 * k}px ${MONO}`;
    for (const g of list) {
      const [x1, y1, x2, y2] = g.box;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      ctx.fillText(`gt${g.id}`, x1 * sx + 2 * k, y2 * sy - 3 * k);
    }
    ctx.restore();
  }

  _dets(list, sx, sy, k, scanX) {
    const { ctx } = this;
    ctx.save();
    ctx.lineWidth = 1 * k;
    ctx.setLineDash([4 * k, 3 * k]);
    ctx.strokeStyle = 'rgba(237,236,230,0.7)';
    ctx.fillStyle = 'rgba(237,236,230,0.75)';
    ctx.font = `500 ${8.5 * k}px ${MONO}`;
    for (const d of list) {
      const [x1, y1, x2, y2] = d.box;
      if (scanX >= 0 && x1 * sx > scanX) continue;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      ctx.fillText(d.score.toFixed(2), x1 * sx + 2 * k, y1 * sy + 10 * k);
    }
    ctx.restore();
  }

  _trails(fr, fi, sx, sy, k, scanX) {
    const { ctx } = this;
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const t of fr.tracks) {
      if (!this._shown(t, sx, scanX)) continue;
      const arr = this.hist.get(t.id);
      const p = this.posIdx[fi].get(t.id);
      const sel = this.selected.has(t.id);
      const quiet = this.selected.size > 0 && !sel ? 0.35 : 1;
      const from = sel ? 0 : Math.max(0, p - TRAIL); // a selected track shows its whole path
      for (let q = from + 1; q <= p; q++) {
        const a = (q - from) / (p - from + 1);
        ctx.strokeStyle = sel ? `rgba(255,255,255,${(0.25 + 0.7 * a).toFixed(3)})` : `rgba(215,255,63,${((0.08 + 0.55 * a) * quiet).toFixed(3)})`;
        ctx.lineWidth = (sel ? 1.6 + 1.6 * a : 0.8 + 1.4 * a) * k;
        ctx.beginPath();
        ctx.moveTo(arr[q - 1][1] * sx, arr[q - 1][2] * sy);
        ctx.lineTo(arr[q][1] * sx, arr[q][2] * sy);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  _shown(t, sx, scanX) {
    if (!this.scan && this.scanDone) return true;
    if (this.revealed.has(t.id)) return true;
    return false;
  }

  _tracks(fr, sx, sy, k, scanX, now) {
    const { ctx } = this;
    const items = [];
    this.hit = [];
    this.animating = false;
    for (const t of fr.tracks) {
      const [x1, y1, x2, y2] = t.box;
      const cx = ((x1 + x2) / 2) * sx;
      if (!this.revealed.has(t.id)) {
        if (this.scan && cx > scanX) continue; // the line has not reached it yet
        this.revealed.set(t.id, now);
        if (this.scan) this.found++;
      }
      const born = now - this.revealed.get(t.id);
      const e = reduced() ? 1 : easeOut(born / 380);
      if (born < 400) this.animating = true; // keep redrawing until the box has finished appearing
      const X1 = x1 * sx, Y1 = y1 * sy, X2 = x2 * sx, Y2 = y2 * sy;
      const pad = lerp(10 * k, 0, e);
      const bx = X1 - pad, by = Y1 - pad, bw = X2 - X1 + pad * 2, bh = Y2 - Y1 + pad * 2;
      const sel = this.selected.has(t.id);
      const dim = this.selected.size > 0 && !sel;
      const alpha = (t.detected ? 1 : 0.62) * e * (dim ? 0.55 : 1);
      this.hit.push({ id: t.id, x1: X1 / k, y1: Y1 / k, x2: X2 / k, y2: Y2 / k });
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = sel ? '#ffffff' : LIME;
      ctx.lineWidth = (sel ? 2.4 : 1.7) * k;
      ctx.lineJoin = 'round';
      const cl = clamp(Math.min(bw, bh) * 0.26, 5 * k, 16 * k);
      ctx.beginPath();
      for (const [px, py, dx, dy] of [[bx, by, 1, 1], [bx + bw, by, -1, 1], [bx, by + bh, 1, -1], [bx + bw, by + bh, -1, -1]]) {
        ctx.moveTo(px + dx * cl, py);
        ctx.lineTo(px, py);
        ctx.lineTo(px, py + dy * cl);
      }
      ctx.stroke();
      if (sel) {
        ctx.globalAlpha = 0.16 * e;
        ctx.fillStyle = LIME;
        ctx.fillRect(bx, by, bw, bh);
      }
      ctx.restore();
      items.push({ t, e, alpha, sel, bx, by, bw, bh });
    }
    this._labels(items, k);
  }

  // Labels: compact chips (class initial + id) by default, the full label with confidence for the
  // selected track and for large boxes. Chips are placed greedily without overlapping each other:
  // above the box, then stacked upward, then inside it; a thin leader line shows which box a
  // nudged chip belongs to.
  _labels(items, k) {
    const { ctx } = this;
    const ch = 16 * k;
    const gap = 2 * k;
    // the FRAME / SCAN readout in the top-left corner is reserved so chips never sit under it
    const placed = [{ x: 0, y: 0, w: 118 * k, h: 40 * k }];
    const hit = (r) => placed.some((p) => r.x < p.x + p.w + gap && r.x + r.w + gap > p.x && r.y < p.y + p.h + gap && r.y + r.h + gap > p.y);
    ctx.font = `600 ${10 * k}px ${MONO}`;
    ctx.textBaseline = 'middle';
    const order = [...items].sort((a, b) => (b.sel - a.sel) || (b.bw * b.bh - a.bw * a.bh));
    for (const it of order) {
      const { t, e, sel } = it;
      const big = it.bw / k >= 120 && it.bh / k >= 70;
      const full = `${t.cls} ${t.id}  ${(t.score * e).toFixed(2)}`;
      const compact = `${INITIAL[t.cls] || t.cls[0].toUpperCase()}${t.id}`;
      const variants = sel || big ? [full, compact] : [compact];
      let spot = null;
      let text = '';
      for (const v of variants) {
        const w = ctx.measureText(v).width + 10 * k;
        const xs = [it.bx, it.bx + it.bw - w, it.bx + it.bw / 2 - w / 2].map((x) => clamp(x, 2, this.W - w - 2));
        for (let row = 0; row < 5 && !spot; row++) {
          for (const x of xs) {
            for (const y of [it.by - (ch + gap) * (row + 1), it.by + gap + (ch + gap) * row]) {
              const r = { x, y, w, h: ch, row, above: y < it.by };
              if (r.y < 1 || r.y + ch > this.H - 1 || hit(r)) continue;
              spot = r;
              text = v;
              break;
            }
            if (spot) break;
          }
        }
        if (spot) break;
      }
      ctx.save();
      ctx.globalAlpha = it.alpha;
      if (!spot) { // no room: a dot at the box corner keeps the box identifiable by position
        ctx.fillStyle = sel ? '#ffffff' : LIME;
        ctx.beginPath();
        ctx.arc(it.bx + 3 * k, it.by + 3 * k, 2.5 * k, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        continue;
      }
      placed.push(spot);
      (this.labelRects ||= []).push(spot);
      const edgeY = spot.above ? spot.y + spot.h : spot.y;
      const gapPx = spot.above ? it.by - (spot.y + spot.h) : spot.y - it.by;
      if (gapPx > 4 * k) { // displaced: leader line to the box corner
        ctx.strokeStyle = sel ? '#ffffff' : LIME;
        ctx.lineWidth = 1 * k;
        ctx.globalAlpha = it.alpha * 0.7;
        ctx.beginPath();
        ctx.moveTo(spot.x + 6 * k, edgeY);
        ctx.lineTo(it.bx + 2 * k, it.by);
        ctx.stroke();
        ctx.globalAlpha = it.alpha;
      }
      ctx.fillStyle = sel ? '#ffffff' : LIME;
      roundRect(ctx, spot.x, spot.y, spot.w, spot.h, 3 * k);
      ctx.fill();
      ctx.fillStyle = '#07080a';
      ctx.fillText(text, spot.x + 5 * k, spot.y + spot.h / 2 + 0.5 * k);
      ctx.restore();
    }
  }

  // The track under a point (CSS pixels relative to the canvas), smallest box first, with a
  // little slop so thumbs can hit small boxes.
  pick(x, y) {
    const slop = 8;
    let best = null;
    let bestA = Infinity;
    for (const h of this.hit || []) {
      if (x >= h.x1 - slop && x <= h.x2 + slop && y >= h.y1 - slop && y <= h.y2 + slop) {
        const a = (h.x2 - h.x1) * (h.y2 - h.y1);
        if (a < bestA) { best = h.id; bestA = a; }
      }
    }
    return best;
  }

  _events(t, sx, sy, k) {
    const { ctx } = this;
    if (reduced()) return;
    for (const e of this.events) {
      if (!e.pos) continue;
      const age = t - e.t;
      if (age < 0 || age > 1.3) continue;
      const p = age / 1.3;
      const r = (8 + 34 * easeOut(p)) * k;
      const warn = e.kind === 'leave' || e.kind === 'proximity';
      ctx.save();
      ctx.globalAlpha = (1 - p) * 0.9;
      ctx.strokeStyle = warn ? EMBER : LIME;
      ctx.lineWidth = 1.5 * k;
      ctx.beginPath();
      ctx.arc(e.pos[0] * sx, e.pos[1] * sy, r, 0, Math.PI * 2);
      ctx.stroke();
      if (p < 0.7) {
        ctx.font = `600 ${8.5 * k}px ${MONO}`;
        const word = e.kind.toUpperCase().replace('_', ' ');
        const tx = e.pos[0] * sx + r + 4 * k;
        const ty = e.pos[1] * sy + 3 * k;
        const box = { x: tx - 2 * k, y: ty - 10 * k, w: ctx.measureText(word).width + 4 * k, h: 14 * k };
        // the word is dropped where it would run over a label chip or off the picture, the ring stays
        const clear = box.x + box.w < this.W - 2 && !(this.labelRects || []).some(
          (q) => box.x < q.x + q.w && box.x + box.w > q.x && box.y < q.y + q.h && box.y + box.h > q.y);
        if (clear) {
          ctx.fillStyle = warn ? EMBER : LIME;
          ctx.fillText(word, tx, ty);
        }
      }
      ctx.restore();
    }
  }

  _scanner(x, k) {
    const { ctx, H } = this;
    const g = ctx.createLinearGradient(x - 90 * k, 0, x, 0);
    g.addColorStop(0, 'rgba(215,255,63,0)');
    g.addColorStop(1, 'rgba(215,255,63,0.2)');
    ctx.fillStyle = g;
    ctx.fillRect(x - 90 * k, 0, 90 * k, H);
    ctx.strokeStyle = 'rgba(215,255,63,0.95)';
    ctx.lineWidth = 1.5 * k;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
