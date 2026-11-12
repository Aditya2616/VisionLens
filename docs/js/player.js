// Players share one small interface so the overlay and timeline do not care what is underneath:
//   el, time, duration, paused, play(), pause(), seek(t), onEnd(cb), destroy()

export function canPlayVp9() {
  const v = document.createElement('video');
  return !!v.canPlayType && v.canPlayType('video/webm; codecs="vp9"') !== '';
}

export class VideoPlayer {
  constructor(container, url, duration) {
    this.kind = 'video';
    this.video = document.createElement('video');
    const v = this.video;
    v.muted = true;
    v.playsInline = true;
    v.setAttribute('playsinline', '');
    v.preload = 'auto';
    v.src = url;
    v.setAttribute('aria-label', 'Analysed video');
    container.replaceChildren(v);
    this._dur = duration;
    this._end = [];
    v.addEventListener('ended', () => this._end.forEach((f) => f()));
    this.ready = new Promise((res, rej) => {
      v.addEventListener('loadeddata', () => res(), { once: true });
      v.addEventListener('error', () => rej(new Error('video failed to load')), { once: true });
    });
  }
  get time() { return this.video.currentTime; }
  get duration() { return this.video.duration && isFinite(this.video.duration) ? this.video.duration : this._dur; }
  get paused() { return this.video.paused; }
  play() { return this.video.play().catch(() => {}); }
  pause() { this.video.pause(); }
  seek(t) { this.video.currentTime = Math.max(0, Math.min(t, this.duration - 0.001)); }
  onEnd(cb) { this._end.push(cb); }
  destroy() { this.video.pause(); this.video.removeAttribute('src'); this.video.load(); }
}

// Fallback for browsers that cannot play VP9 WebM: a grid sprite of low-rate frames on a canvas.
export class SpritePlayer {
  constructor(container, spriteUrl, info, duration, width, height) {
    this.kind = 'sprite';
    this.canvas = document.createElement('canvas');
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', 'Analysed video (frame-sequence fallback)');
    container.replaceChildren(this.canvas);
    this.info = info;
    this._dur = duration;
    this._t = 0;
    this._paused = true;
    this._last = 0;
    this._end = [];
    this.w = width;
    this.h = height;
    this.img = new Image();
    this.ready = new Promise((res, rej) => {
      this.img.onload = () => { this._fit(); this._draw(); res(); };
      this.img.onerror = () => rej(new Error('sprite failed to load'));
    });
    this.img.src = spriteUrl;
    this._raf = requestAnimationFrame((n) => this._tick(n));
    this._ro = new ResizeObserver(() => { this._fit(); this._draw(); });
    this._ro.observe(container);
  }
  _fit() {
    const r = this.canvas.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.max(2, Math.round(r.width * dpr));
    this.canvas.height = Math.max(2, Math.round(r.height * dpr));
  }
  _draw() {
    const { info, img } = this;
    if (!img.complete || !img.naturalWidth) return;
    const idx = Math.min(info.count - 1, Math.floor(this._t * info.fps));
    const col = idx % info.cols;
    const row = Math.floor(idx / info.cols);
    const ctx = this.canvas.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, col * info.thumb_w, row * info.thumb_h, info.thumb_w, info.thumb_h, 0, 0, this.canvas.width, this.canvas.height);
  }
  _tick(now) {
    if (!this._paused) {
      this._t += (now - this._last) / 1000;
      if (this._t >= this._dur) { this._t = this._dur - 0.001; this._paused = true; this._end.forEach((f) => f()); }
      this._draw();
    }
    this._last = now;
    this._raf = requestAnimationFrame((n) => this._tick(n));
  }
  get time() { return this._t; }
  get duration() { return this._dur; }
  get paused() { return this._paused; }
  play() { if (this._t >= this._dur - 0.01) this._t = 0; this._paused = false; this._last = performance.now(); return Promise.resolve(); }
  pause() { this._paused = true; }
  seek(t) { this._t = Math.max(0, Math.min(t, this._dur - 0.001)); this._draw(); }
  onEnd(cb) { this._end.push(cb); }
  destroy() { cancelAnimationFrame(this._raf); this._ro.disconnect(); }
}

// A still image: no timeline, the overlay draws once.
export class StillPlayer {
  constructor(container, url) {
    this.kind = 'still';
    this.img = document.createElement('img');
    this.img.alt = 'Analysed image';
    container.replaceChildren(this.img);
    this.ready = new Promise((res, rej) => {
      this.img.onload = () => res();
      this.img.onerror = () => rej(new Error('image failed to load'));
    });
    this.img.src = url;
  }
  get time() { return 0; }
  get duration() { return 0; }
  get paused() { return true; }
  play() { return Promise.resolve(); }
  pause() {}
  seek() {}
  onEnd() {}
  destroy() {}
}

// No playable media: a dark canvas so the overlay still has somewhere to draw.
export class BlankPlayer extends SpritePlayer {
  constructor(container, duration, width, height) {
    super(container, 'data:image/gif;base64,R0lGODlhAQABAAAAACw=', { count: 1, fps: 1, cols: 1, thumb_w: 1, thumb_h: 1 }, duration, width, height);
    this.kind = 'blank';
  }
  _draw() {
    const ctx = this.canvas.getContext('2d');
    ctx.fillStyle = '#0a0b0e';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }
}
