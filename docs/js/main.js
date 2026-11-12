import { DescriptionPanel } from './describe.js';
import { analyze, probeServer, Webcam } from './live.js';
import { Overlay } from './overlay.js';
import { BlankPlayer, canPlayVp9, SpritePlayer, StillPlayer, VideoPlayer } from './player.js';
import { Timeline } from './timeline.js';
import { $, $$, clamp, countTo, fetchJSON, fmtT, h, reduced } from './util.js';

const state = {
  index: null,
  clip: null,
  result: null,
  player: null,
  server: null, // health payload when a local server answers
  layers: { dets: false, tracks: true, trails: true, events: true, gt: false },
  visible: true,
  busy: false,
  cam: null,
  lastT: -1,
};

const el = {
  media: $('#media'), overlay: $('#overlay'), stage: $('#stage'), bigplay: $('#bigplay'),
  play: $('#play'), scrub: $('#scrub'), fill: $('#scrub-fill'), thumb: $('#scrub-thumb'),
  time: $('#time'), note: $('#note'), busy: $('#busy'), busyLabel: $('#busy-label'),
  busyBar: $('#busy-bar'), busySub: $('#busy-sub'), mode: $('#mode'),
};

const overlay = new Overlay(el.overlay);
window.__vl = { overlay, state }; // handles for the headless UI check
const desc = new DescriptionPanel({ onPickEvents: (ids, seek) => pickEvents(ids, seek) });
const timeline = new Timeline($('#timeline'), $('#tl-canvas'), {
  onSeek: (t) => seekTo(t),
  onPick: (e) => pickEvents([e.id], false, e),
});

const forceSprite = new URLSearchParams(location.search).get('player') === 'sprite';

// ---------------------------------------------------------------- boot
init().catch((e) => {
  console.error(e);
  showNote(`Could not start: ${e.message}`, true);
});

async function init() {
  wireUi();
  const [server, index] = await Promise.all([probeServer(), fetchJSON('demo/index.json')]);
  state.server = server;
  state.index = index;
  setMode();
  renderClips();
  wireClipFade();
  renderCredits();
  await loadClip(index.clips[0]);
  requestAnimationFrame(frame);
  if (window.gsap && !reduced()) {
    gsap.from('.intro > *', { y: 18, opacity: 0, duration: 0.9, stagger: 0.08, ease: 'expo.out' });
    gsap.from('.clips .clip', { y: 14, opacity: 0, duration: 0.7, stagger: 0.05, ease: 'expo.out', delay: 0.15 });
  }
}

function setMode() {
  const m = el.mode;
  const idx = state.index;
  if (state.server) {
    m.className = 'mode mono live';
    m.querySelector('span').textContent = 'Live · local server';
    $('#live-on').hidden = false;
    const l = state.server.limits;
    $('#live-limits').textContent = `Up to ${l.max_upload_mb} MB; the first ${l.max_seconds} s are analysed at ${l.max_width}px wide. Clips run one at a time.`;
  } else {
    m.className = 'mode mono rec';
    m.querySelector('span').textContent = `Recorded run · ${idx.recorded_on}`;
    m.title = `Recorded on ${idx.cpu}, ${idx.threads} CPU threads; detector and model on ${idx.device || 'the CPU'}${idx.llm_dtype ? ` (${idx.llm_dtype})` : ''}. ${idx.note || ''}`;
    $('#live-demo').hidden = false;
  }
}

function renderClips() {
  const box = $('#clips');
  box.replaceChildren();
  for (const c of state.index.clips) {
    const kind = c.kind === 'image' ? 'image' : c.simulated ? 'synthetic' : 'real footage';
    const b = h('button', { class: 'clip', type: 'button', 'data-id': c.id }, h('b', { text: c.label }), h('span', { class: 'mono', text: kind }));
    b.addEventListener('click', () => loadClip(c));
    box.append(b);
  }
}

// Edge fades on the clip picker show there is more to scroll to, and hide when there is not.
function wireClipFade() {
  const box = $('#clips');
  const update = () => {
    box.classList.toggle('more-r', box.scrollLeft + box.clientWidth < box.scrollWidth - 6);
    box.classList.toggle('more-l', box.scrollLeft > 6);
  };
  box.addEventListener('scroll', update, { passive: true });
  new ResizeObserver(update).observe(box);
  update();
}

function renderCredits() {
  const body = $('#credits-body');
  body.replaceChildren();
  for (const c of state.index.clips) {
    const s = c.source || {};
    body.append(h('p', { class: 'src' },
      h('b', { text: c.label }), ' · ',
      s.page ? h('a', { href: s.page, rel: 'noopener', text: s.title || 'source' }) : (s.title || ''),
      ` · ${s.author || ''} · ${s.license || ''}`));
  }
  $('#credits-link').addEventListener('click', (e) => { e.preventDefault(); $('#credits').showModal(); });
}

// ---------------------------------------------------------------- clip loading
async function loadClip(clip, preset) {
  stopCam();
  state.clip = clip;
  $$('.clip').forEach((b) => b.setAttribute('aria-current', String(b.dataset.id === clip.id)));
  const result = preset || (await fetchJSON(`demo/${clip.json}`));
  const media = preset ? preset.mediaUrl : `demo/${clip.video}`;
  await showResult(result, { clip, media, label: preset ? 'Live result. ' : 'Recorded run. ' });
}

async function showResult(result, { clip, media, label }) {
  if (state.player) state.player.destroy();
  state.result = result;
  const m = result.meta;
  el.stage.style.setProperty('--ar', `${m.width} / ${m.height}`);
  hideNote();
  el.bigplay.hidden = true;
  el.play.disabled = true;

  let player;
  const dur = m.duration || (result.frames.length / (m.fps || 1));
  try {
    if (m.kind === 'image') {
      player = new StillPlayer(el.media, media);
    } else if (canPlayVp9() && !(forceSprite && clip && clip.sprite)) {
      player = new VideoPlayer(el.media, media, dur);
    } else if (clip && clip.sprite) {
      player = new SpritePlayer(el.media, `demo/${clip.sprite}`, clip.sprite_info, dur, m.width, m.height);
    } else {
      player = new BlankPlayer(el.media, dur, m.width, m.height);
      showNote('This browser cannot play VP9 WebM, so only the overlay is shown.', true);
    }
    await player.ready;
  } catch (e) {
    console.warn(e);
    player = new BlankPlayer(el.media, dur, m.width, m.height);
    showNote('The video could not be loaded; showing the analysis only.', true);
  }
  state.player = player;
  player.onEnd(() => syncPlayButton());

  overlay.setData(result);
  overlay.resize();
  overlay.scanDone = m.kind === 'image';
  state.layers.gt = false;
  const gtChip = $('[data-layer="gt"]');
  gtChip.hidden = !result.ground_truth;
  setChip('gt', false);
  timeline.setData(result);
  $('#tl-hint').textContent = m.kind === 'image' ? 'single image: no timeline' : 'drag to scrub · tap a marker for the event';

  desc.setResult(result, { stream: true, label });
  renderStats(result, label);

  if (m.note && !state.cam) showNote(m.note);
  const still = m.kind === 'image';
  el.bigplay.hidden = still;
  el.play.disabled = still;
  $('#rescan').disabled = still;
  if (still) { overlay.scanDone = true; }
  state.lastT = -1;
  syncPlayButton();
  syncScrub();
}

function renderStats(result, label) {
  const t = result.timings;
  const m = result.meta;
  const s = result.summary || {};
  const dev = t.device === 'cuda' ? 'GPU' : t.device === 'cpu' ? 'CPU' : '';
  $('#s-detect-k').textContent = dev ? `01 DETECT · ${dev}` : '01 DETECT';
  $('#s-detect').textContent = t.detect_ms_mean ? `${t.detect_ms_mean} ms` : '–';
  $('#s-detect-sub').textContent = `${m.detector.split(' ')[0]} · ${m.input_size || '–'}px · every ${m.stride}f`;
  $('#s-track').textContent = m.kind === 'image' ? 'n/a' : `${t.track_ms_mean} ms`;
  $('#s-track-sub').textContent = m.kind === 'image' ? 'single frame' : `${s.tracks || 0} tracks`;
  const d = result.description;
  $('#s-desc').textContent = `${result.events.length} ev`;
  $('#s-desc-sub').textContent = d ? (d.model && d.model.accepted ? 'template + model' : 'template') : '';
  $('#strip').title = `${t.cpu || ''}${t.threads ? `, ${t.threads} threads` : ''}. ${t.machine_note || ''} End to end: ${t.fps_end_to_end || '–'} fps.`;
  countTo($('#c-total'), s.tracks || 0, { dur: 0.9 });
  countTo($('#c-events'), result.events.length, { dur: 0.9 });
  $('#c-classes').replaceChildren(...Object.entries(s.by_class || {}).map(([k, v]) => h('span', { text: `${v} ${k}` })));
}

// ---------------------------------------------------------------- UI wiring
function wireUi() {
  el.bigplay.addEventListener('click', () => startWithScan());
  el.play.addEventListener('click', () => togglePlay());
  $('#rescan').addEventListener('click', () => startWithScan());
  $$('#layers .chip[data-layer]').forEach((b) => b.addEventListener('click', () => {
    const k = b.dataset.layer;
    state.layers[k] = !state.layers[k];
    setChip(k, state.layers[k]);
    state.lastT = -1;
  }));

  // scrub bar (pointer events so a thumb drag works)
  let dragging = false;
  const at = (e) => { const r = el.scrub.getBoundingClientRect(); return clamp((e.clientX - r.left) / r.width); };
  el.scrub.addEventListener('pointerdown', (e) => { dragging = true; el.scrub.setPointerCapture(e.pointerId); seekTo(at(e) * dur()); });
  el.scrub.addEventListener('pointermove', (e) => { if (dragging) seekTo(at(e) * dur()); });
  el.scrub.addEventListener('pointerup', () => { dragging = false; });
  el.scrub.addEventListener('pointercancel', () => { dragging = false; });
  el.scrub.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') seekTo((state.player?.time || 0) + 1);
    if (e.key === 'ArrowLeft') seekTo((state.player?.time || 0) - 1);
    if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  });

  // tap a box to select its track: trail, timeline lane, its events and the sentences that cite it
  el.stage.addEventListener('click', (e) => {
    if (e.target.closest('button, .busy, .note') || !state.result || state.cam) return;
    const r = el.overlay.getBoundingClientRect();
    selectTrack(overlay.pick(e.clientX - r.left, e.clientY - r.top));
  });

  new IntersectionObserver((es) => { state.visible = es[0].isIntersecting; }, { threshold: 0.05 }).observe(el.stage);
  new ResizeObserver(() => { overlay.resize(); state.lastT = -1; }).observe(el.stage);

  // live controls (only visible when a server answered)
  $('#file').addEventListener('change', (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (f) runLive({ file: f }, `Your file: ${f.name}`);
  });
  $('#run-sample').addEventListener('click', () => {
    const c = state.clip;
    if (!c || c.simulated) { showNote('Pick a real-footage clip above to run it live.', true); return; }
    runLive({ sample: c.id }, `Clip: ${c.label}`);
  });
  $('#cam').addEventListener('click', () => (state.cam ? stopCam(true) : startCam()));
}

const dur = () => (state.player ? state.player.duration : 0) || 1;

function setChip(k, on) {
  const b = $(`[data-layer="${k}"]`);
  if (b) b.setAttribute('aria-pressed', String(on));
}

function showNote(text, warn = false) {
  el.note.textContent = text;
  el.note.classList.toggle('warn', warn);
  el.note.hidden = false;
}
function hideNote() { el.note.hidden = true; }

function startWithScan() {
  const p = state.player;
  if (!p || p.kind === 'still') return;
  el.bigplay.hidden = true;
  hideNote();
  p.seek(0);
  overlay.startScan();
  p.play();
  syncPlayButton();
}

function togglePlay() {
  const p = state.player;
  if (!p || p.kind === 'still') return;
  el.bigplay.hidden = true;
  if (p.paused) {
    if (p.time >= p.duration - 0.05) p.seek(0);
    p.play();
  } else p.pause();
  syncPlayButton();
}

function seekTo(t) {
  const p = state.player;
  if (!p || p.kind === 'still') return;
  overlay.scan = null;
  overlay.scanDone = true;
  p.seek(clamp(t, 0, p.duration));
  state.lastT = -1;
}

function syncPlayButton() {
  const p = state.player;
  const playing = !!p && !p.paused;
  el.play.classList.toggle('playing', playing);
  el.play.setAttribute('aria-label', playing ? 'Pause' : 'Play');
}

function syncScrub() {
  const p = state.player;
  const f = p && p.duration ? clamp(p.time / p.duration) : 0;
  el.fill.style.width = `${f * 100}%`;
  el.thumb.style.left = `${f * 100}%`;
  el.scrub.setAttribute('aria-valuenow', String(Math.round(f * 100)));
  el.time.textContent = `${fmtT(p ? p.time : 0)} / ${fmtT(p ? p.duration : 0)}`;
}

// ---------------------------------------------------------------- events <-> description <-> timeline
function pickEvents(ids, seek, ev) {
  const r = state.result;
  if (!r) return;
  const events = r.events.filter((e) => ids.includes(e.id));
  if (!events.length) return;
  const first = events[0];
  overlay.selected = new Set(events.flatMap((e) => e.ids));
  timeline.select({ id: first.id, ids: [...overlay.selected] });
  desc.highlight(ids);
  const tip = $('#tl-tip');
  const text = describeEvent(first);
  tip.textContent = text;
  tip.hidden = false;
  clearTimeout(pickEvents._t);
  pickEvents._t = setTimeout(() => { tip.hidden = true; overlay.selected = new Set(); timeline.select(null); desc.highlight([]); state.lastT = -1; }, 4500);
  if (seek || ev) seekTo(Math.max(0, first.t - 0.2));
  state.lastT = -1;
}

function selectTrack(id) {
  const r = state.result;
  clearTimeout(pickEvents._t);
  const tip = $('#tl-tip');
  if (id == null || !r || overlay.selected.has(id)) {
    overlay.selected = new Set();
    timeline.select(null);
    desc.highlight([]);
    tip.hidden = true;
    state.lastT = -1;
    return;
  }
  const evs = r.events.filter((e) => e.ids.includes(id));
  const cls = (evs[0] && evs[0].classes[evs[0].ids.indexOf(id)]) || 'track';
  overlay.selected = new Set([id]);
  timeline.select({ id: null, ids: [id], eventIds: evs.map((e) => e.id) });
  desc.highlight(evs.map((e) => e.id));
  const shown = evs.filter((e) => e.kind !== 'count_change').slice(0, 3).map(describeEvent);
  tip.textContent = `${cls} ${id}${shown.length ? ' · ' + shown.join(' | ') : ' · no events'}${evs.length > 3 ? ` | +${evs.length - 3}` : ''}`;
  tip.hidden = false;
  state.lastT = -1;
}

function describeEvent(e) {
  const who = e.ids.map((id, i) => `${e.classes[i] || ''} ${id}`.trim()).join(' + ');
  const when = fmtT(e.t) + (e.t_end ? `–${fmtT(e.t_end)}` : '');
  const extra = e.data.edge ? ` · ${e.data.edge}` : e.data.direction ? ` · ${e.data.direction}` : e.data.count != null ? ` · ${e.data.previous}→${e.data.count}` : '';
  return `${when}  ${e.kind.replace('_', ' ')}  ${who}${extra}`;
}

// ---------------------------------------------------------------- render loop
function frame(now) {
  requestAnimationFrame(frame);
  if (state.cam) return;
  const p = state.player;
  if (!p || !state.visible) return;
  const t = p.time;
  const scanning = !!overlay.scan;
  if (t !== state.lastT || scanning || !p.paused) {
    overlay.draw(t, state.layers, now);
    timeline.draw(t);
    syncScrub();
    updateNow(t);
    state.lastT = t;
  }
  if (p.kind !== 'still') syncPlayButton();
}

let lastNow = -1;
function updateNow(t) {
  const r = state.result;
  if (!r) return;
  const n = r.frames[overlay.frameIndex(t)]?.tracks.length ?? 0;
  if (n !== lastNow) { $('#c-now').textContent = n; lastNow = n; }
}

// ---------------------------------------------------------------- live: upload, sample, webcam
async function runLive(req, label) {
  if (state.busy) return;
  stopCam();
  state.busy = true;
  const describer = $('#opt-desc').value;
  const stride = Number($('#opt-stride').value);
  if (state.player) state.player.pause();
  el.bigplay.hidden = true;
  el.busy.hidden = false;
  el.busyBar.style.width = '2%';
  el.busyLabel.textContent = 'Queued…';
  el.busySub.textContent = label;
  desc.beginLive();
  $$('.strip .cell').forEach((c) => c.classList.remove('on'));
  $('.strip [data-s="detect"]').classList.add('on');
  try {
    const out = await analyze({ ...req, describer, stride }, {
      onStatus: (d) => { if (d.state === 'running') el.busyLabel.textContent = 'Detecting and tracking…'; },
      onProgress: (d) => {
        if (d.total) el.busyBar.style.width = `${Math.min(98, (d.i / d.total) * 100)}%`;
        el.busySub.textContent = `${label} · frame ${d.i}${d.total ? ` of ~${d.total}` : ''} · ${d.tracks} tracked`;
      },
      onToken: (t) => {
        $$('.strip .cell').forEach((c) => c.classList.remove('on'));
        $('.strip [data-s="describe"]').classList.add('on');
        el.busyLabel.textContent = 'Describing…';
        desc.pushToken(t);
      },
      onModelToken: (t) => desc.pushModelToken(t),
    });
    el.busyBar.style.width = '100%';
    const clip = { id: 'live', kind: out.result.meta.kind, sprite: null };
    state.clip = state.clip && state.clip.id !== 'live' ? state.clip : clip;
    out.result.mediaUrl = out.mediaUrl;
    $$('.clip').forEach((b) => b.setAttribute('aria-current', 'false'));
    await showResult(out.result, { clip: { ...clip }, media: out.mediaUrl, label: 'Live result. ' });
    if (out.result.meta.truncated) showNote(`Only the first ${state.server.limits.max_seconds} s were analysed.`);
    if (out.result.meta.kind === 'video') startWithScan();
  } catch (e) {
    showNote(`Analysis failed: ${e.message}`, true);
    if (state.result) desc.setResult(state.result, { stream: false });
  } finally {
    el.busy.hidden = true;
    state.busy = false;
    $$('.strip .cell').forEach((c) => c.classList.remove('on'));
  }
}

async function startCam() {
  if (state.busy) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showNote('This browser does not offer camera access on this page (it needs HTTPS or localhost).', true);
    return;
  }
  if (state.player) state.player.pause();
  const video = h('video', { muted: true, playsinline: true, autoplay: true });
  el.media.replaceChildren(video);
  el.bigplay.hidden = true;
  const data = { meta: { width: 480, height: 360, fps: 1, duration: 0, name: 'webcam' }, frames: [], events: [], ground_truth: null };
  const cam = new Webcam(video, (r) => {
    el.stage.style.setProperty('--ar', `${r.width} / ${r.height}`);
    data.meta.width = r.width;
    data.meta.height = r.height;
    data.frames.push({ i: data.frames.length, t: data.frames.length, dets: null, tracks: r.tracks });
    if (data.frames.length > 60) data.frames.shift(), data.frames.forEach((f, i) => { f.i = i; f.t = i; });
    overlay.setData(data, true);
    overlay.scanDone = true;
    overlay.resize();
    overlay.draw(data.frames.length - 1, state.layers, performance.now());
    $('#c-now').textContent = r.count;
    $('#s-detect').textContent = `${r.ms} ms`;
    if (r.description) { $('#d-text').textContent = r.description; $('#d-tag').textContent = 'Live description, refreshed every few seconds.'; }
  }, (e) => {
    showNote(`Webcam stopped: ${e.message}`, true);
    stopCam(true);
  });
  state.cam = cam;
  try {
    desc.clear('Webcam: tracking runs on your machine; no frames are stored.');
    $('#c-total').textContent = '0';
    $('#c-events').textContent = '0';
    $('#c-classes').replaceChildren();
    await cam.start();
    $('#cam').textContent = 'Stop webcam';
  } catch (e) {
    state.cam = null;
    showNote(e.name === 'NotAllowedError' ? 'Camera permission was not granted.' : `Could not open the camera: ${e.message}`, true);
    if (state.clip) loadClip(state.clip);
  }
}

function stopCam(reload = false) {
  if (!state.cam) return;
  state.cam.stop();
  state.cam = null;
  $('#cam').textContent = 'Use webcam';
  overlay.setData(null);
  if (reload && state.clip) loadClip(state.clip);
}
