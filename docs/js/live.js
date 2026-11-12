// Live mode: talks to the local FastAPI server. Nothing here runs on the static site.
import { fetchJSON } from './util.js';

export async function probeServer(timeoutMs = 1500) {
  const flag = document.querySelector('meta[name="vl-server"]');
  if (!flag || flag.content !== '1') return null; // static copy: nothing to ask
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch('health', { signal: ctl.signal, cache: 'no-store' });
    if (!r.ok) return null;
    const j = await r.json();
    return j && j.status === 'ok' && j.live ? j : null;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Start an analysis job (file or sample) and stream its events. Resolves with the finished result. */
export function analyze({ file, sample, describer, stride, caption }, h) {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    if (file) fd.append('file', file);
    if (sample) fd.append('sample', sample);
    fd.append('describer', describer);
    fd.append('stride', String(stride));
    fd.append('caption', caption ? 'true' : 'false');
    fetch('api/analyze', { method: 'POST', body: fd })
      .then(async (r) => {
        if (!r.ok) {
          let msg = `${r.status}`;
          try { msg = (await r.json()).detail || msg; } catch { /* not json */ }
          throw new Error(msg);
        }
        return r.json();
      })
      .then(({ job_id: id }) => {
        const es = new EventSource(`api/jobs/${id}/stream`);
        const on = (name, fn) => es.addEventListener(name, (e) => fn(JSON.parse(e.data)));
        on('status', (d) => h.onStatus && h.onStatus(d));
        on('progress', (d) => h.onProgress && h.onProgress(d));
        on('token', (d) => h.onToken && h.onToken(d.text));
        on('model_token', (d) => h.onModelToken && h.onModelToken(d.text));
        on('error', (d) => { es.close(); reject(new Error(d.message || 'analysis failed')); });
        on('done', async () => {
          es.close();
          try {
            const result = await fetchJSON(`api/jobs/${id}/result`);
            resolve({ id, result, mediaUrl: `api/jobs/${id}/video` });
          } catch (e) { reject(e); }
        });
        es.onerror = () => { if (es.readyState === EventSource.CLOSED) reject(new Error('lost the connection to the server')); };
      })
      .catch(reject);
  });
}

/** Webcam: started only from a tap. Posts a JPEG about 4 times a second and reports tracks. */
export class Webcam {
  constructor(videoEl, onResult, onError) {
    this.video = videoEl;
    this.onResult = onResult;
    this.onError = onError;
    this.session = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now()).replace(/-/g, '').slice(0, 24);
    this.stream = null;
    this.running = false;
    this.canvas = document.createElement('canvas');
  }
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 } }, audio: false });
    this.video.srcObject = this.stream;
    this.video.muted = true;
    this.video.playsInline = true;
    await this.video.play();
    this.running = true;
    this._loop();
  }
  async _loop() {
    while (this.running) {
      const t0 = performance.now();
      try {
        const v = this.video;
        if (v.videoWidth) {
          const w = 480;
          const hgt = Math.round((w * v.videoHeight) / v.videoWidth);
          this.canvas.width = w;
          this.canvas.height = hgt;
          this.canvas.getContext('2d').drawImage(v, 0, 0, w, hgt);
          const blob = await new Promise((res) => this.canvas.toBlob(res, 'image/jpeg', 0.7));
          const r = await fetch(`api/frame?session=${this.session}`, { method: 'POST', body: blob, headers: { 'Content-Type': 'image/jpeg' } });
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `server said ${r.status}`);
          this.onResult(await r.json());
        }
      } catch (e) {
        this.running = false;
        this.onError(e);
        break;
      }
      const wait = Math.max(0, 220 - (performance.now() - t0));
      await new Promise((res) => setTimeout(res, wait));
    }
  }
  stop() {
    this.running = false;
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.video.srcObject = null;
  }
}
