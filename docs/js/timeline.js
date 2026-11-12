// Event timeline: a count sparkline, one lane per track, event glyphs, a playhead.
// Dragging scrubs the video; tapping a glyph selects the event.
import { clamp, fmtT } from './util.js';

const LIME = '#d7ff3f';
const EMBER = '#ff7a3d';
const MONO = '"JetBrains Mono", ui-monospace, Consolas, monospace';
const GUT = 78; // label gutter (css px)
const LANE = 20;
const SPARK = 44;
const AXIS = 18;

export class Timeline {
  constructor(wrap, canvas, { onSeek, onPick }) {
    this.wrap = wrap;
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.onSeek = onSeek;
    this.onPick = onPick;
    this.data = null;
    this.t = 0;
    this.sel = null;
    this._drag = null;
    this.ro = new ResizeObserver(() => this.layout());
    this.ro.observe(wrap);
    canvas.addEventListener('pointerdown', (e) => this._down(e));
    canvas.addEventListener('pointermove', (e) => this._move(e));
    canvas.addEventListener('pointerup', (e) => this._up(e));
    canvas.addEventListener('pointercancel', () => { this._drag = null; });
  }

  setData(result) {
    this.data = result;
    this.sel = null;
    if (!result) { this.lanes = []; this.layout(); return; }
    const spans = new Map();
    for (const fr of result.frames) {
      for (const t of fr.tracks) {
        if (!t.detected) continue;
        const s = spans.get(t.id) || { id: t.id, cls: t.cls, a: fr.t, b: fr.t };
        s.b = fr.t;
        spans.set(t.id, s);
      }
    }
    const ids = new Set(result.events.flatMap((e) => e.ids));
    this.lanes = [...spans.values()].filter((s) => ids.has(s.id) || result.events.length === 0).sort((x, y) => x.a - y.a || x.id - y.id);
    this.laneOf = new Map(this.lanes.map((s, i) => [s.id, i]));
    this.counts = result.frames.map((f) => f.tracks.filter((t) => this.laneOf.has(t.id)).length);
    this.layout();
  }

  get dur() { return this.data ? Math.max(this.data.meta.duration, 0.1) : 1; }

  layout() {
    const w = this.wrap.clientWidth || 300;
    const lanes = this.lanes ? this.lanes.length : 0;
    const h = AXIS + SPARK + Math.max(lanes, 1) * LANE + 8;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cssW = w;
    this.cssH = h;
    this.c.style.height = `${h}px`;
    this.c.width = Math.round(w * this.dpr);
    this.c.height = Math.round(h * this.dpr);
    this.draw(this.t);
  }

  x(t) { return GUT + (t / this.dur) * (this.cssW - GUT - 12); }
  tAt(px) { return clamp((px - GUT) / (this.cssW - GUT - 12)) * this.dur; }

  draw(t) {
    this.t = t;
    const { ctx, dpr, cssW, cssH } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    if (!this.data) return;
    const dur = this.dur;
    ctx.font = `500 9.5px ${MONO}`;
    ctx.textBaseline = 'middle';
    // axis ticks
    const step = dur > 40 ? 10 : dur > 14 ? 2 : 1;
    ctx.fillStyle = 'rgba(237,236,230,0.42)';
    ctx.strokeStyle = 'rgba(237,236,230,0.07)';
    ctx.lineWidth = 1;
    for (let s = 0; s <= dur; s += step) {
      const x = Math.round(this.x(s)) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, AXIS); ctx.lineTo(x, cssH); ctx.stroke();
      ctx.textAlign = 'center';
      ctx.fillText(fmtT(s), x, 8);
    }
    // count sparkline
    const maxC = Math.max(1, ...this.counts);
    const y0 = AXIS + SPARK - 4;
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(237,236,230,0.5)';
    ctx.fillText('in view', 8, AXIS + 12);
    ctx.fillText(`max ${maxC}`, 8, AXIS + 26);
    ctx.beginPath();
    this.counts.forEach((c, i) => {
      const x = this.x(this.data.frames[i].t);
      const y = y0 - (c / maxC) * (SPARK - 12);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'rgba(215,255,63,0.8)';
    ctx.lineWidth = 1.4;
    ctx.stroke();
    ctx.lineTo(this.x(dur), y0); ctx.lineTo(this.x(0), y0); ctx.closePath();
    ctx.fillStyle = 'rgba(215,255,63,0.08)';
    ctx.fill();
    // lanes
    const top = AXIS + SPARK;
    this.lanes.forEach((l, i) => {
      const y = top + i * LANE + LANE / 2;
      const selected = this.sel && this.sel.ids.includes(l.id);
      ctx.fillStyle = selected ? '#fff' : 'rgba(237,236,230,0.62)';
      ctx.textAlign = 'left';
      ctx.fillText(`${l.cls} ${l.id}`.slice(0, 12), 8, y);
      const xa = this.x(l.a), xb = Math.max(this.x(l.b), xa + 3);
      ctx.fillStyle = selected ? 'rgba(215,255,63,0.55)' : 'rgba(215,255,63,0.22)';
      ctx.fillRect(xa, y - 3.5, xb - xa, 7);
      ctx.strokeStyle = 'rgba(215,255,63,0.7)';
      ctx.lineWidth = 1;
      ctx.strokeRect(xa + 0.5, y - 3.5, xb - xa - 1, 7);
    });
    // events
    this.glyphs = [];
    for (const e of this.data.events) {
      if (e.kind === 'count_change') {
        const x = this.x(e.t);
        ctx.fillStyle = 'rgba(237,236,230,0.5)';
        ctx.fillRect(x - 0.5, AXIS + 4, 1, SPARK - 8);
        this.glyphs.push({ e, x, y: AXIS + SPARK / 2 });
        continue;
      }
      for (const id of e.ids) {
        const li = this.laneOf.get(id);
        if (li === undefined) continue;
        const y = top + li * LANE + LANE / 2;
        const x = this.x(e.t);
        const hot = this.sel && this.sel.id === e.id;
        this._glyph(ctx, e.kind, x, y, hot);
        if (e.t_end && (e.kind === 'linger' || e.kind === 'proximity')) {
          ctx.strokeStyle = e.kind === 'proximity' ? EMBER : 'rgba(237,236,230,0.8)';
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(x, y + 6.5); ctx.lineTo(this.x(e.t_end), y + 6.5); ctx.stroke();
        }
        this.glyphs.push({ e, x, y });
      }
    }
    // playhead
    const px = this.x(t);
    ctx.strokeStyle = LIME;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, AXIS - 2); ctx.lineTo(px, cssH); ctx.stroke();
    ctx.fillStyle = LIME;
    ctx.beginPath(); ctx.moveTo(px - 4, AXIS - 2); ctx.lineTo(px + 4, AXIS - 2); ctx.lineTo(px, AXIS + 4); ctx.closePath(); ctx.fill();
  }

  _glyph(ctx, kind, x, y, hot) {
    ctx.save();
    ctx.translate(x, y);
    const col = kind === 'leave' || kind === 'proximity' ? EMBER : LIME;
    ctx.fillStyle = hot ? '#fff' : col;
    ctx.strokeStyle = '#07080a';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (kind === 'enter') { ctx.moveTo(-4, -5); ctx.lineTo(5, 0); ctx.lineTo(-4, 5); }
    else if (kind === 'leave') { ctx.moveTo(4, -5); ctx.lineTo(-5, 0); ctx.lineTo(4, 5); }
    else if (kind === 'move') { ctx.arc(0, 0, 3, 0, Math.PI * 2); }
    else if (kind === 'linger') { ctx.rect(-4, -4, 8, 8); }
    else { ctx.moveTo(0, -5); ctx.lineTo(5, 0); ctx.lineTo(0, 5); ctx.lineTo(-5, 0); }
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
    ctx.restore();
  }

  select(sel) { this.sel = sel; this.draw(this.t); }

  _pos(e) { const r = this.c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  _down(e) {
    const [x, y] = this._pos(e);
    this._drag = { x0: x, y0: y, moved: false, id: e.pointerId };
    if (x >= GUT - 6) { try { this.c.setPointerCapture(e.pointerId); } catch { /* ignore */ } this.onSeek(this.tAt(x)); }
  }
  _move(e) {
    if (!this._drag) return;
    const [x, y] = this._pos(e);
    if (Math.abs(x - this._drag.x0) > 5 || Math.abs(y - this._drag.y0) > 5) this._drag.moved = true;
    if (this._drag.moved) this.onSeek(this.tAt(x));
  }
  _up(e) {
    const d = this._drag;
    this._drag = null;
    if (!d || d.moved) return;
    const [x, y] = this._pos(e);
    let best = null, bd = 18;
    for (const g of this.glyphs || []) {
      const dist = Math.hypot(g.x - x, g.y - y);
      if (dist < bd) { bd = dist; best = g; }
    }
    if (best) this.onPick(best.e);
  }
}
