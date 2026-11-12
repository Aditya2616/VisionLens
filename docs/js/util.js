export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const easeOut = (t) => 1 - Math.pow(1 - clamp(t), 3);
export const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export function fmtT(t) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

export async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null) el.append(kid.nodeType ? kid : document.createTextNode(kid));
  return el;
}

/** Tween a number into an element's text. Uses GSAP when present, otherwise rAF. */
export function countTo(el, to, { dur = 0.8, fmt = (v) => String(Math.round(v)) } = {}) {
  if (reduced()) { el.textContent = fmt(to); return; }
  const from = parseFloat(el.dataset.v || '0') || 0;
  const t0 = performance.now();
  el.dataset.v = to;
  const step = (now) => {
    const p = clamp((now - t0) / (dur * 1000));
    el.textContent = fmt(lerp(from, to, easeOut(p)));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
