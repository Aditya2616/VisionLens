// The description panel: template and model tabs, token streaming, event linking.
import { $, $$, reduced } from './util.js';

export class DescriptionPanel {
  constructor({ onPickEvents }) {
    this.text = $('#d-text');
    this.tag = $('#d-tag');
    this.guard = $('#d-guard');
    this.tabs = $$('#d-tabs .tab');
    this.scene = $('#d-scene');
    this.onPickEvents = onPickEvents;
    this.tab = 'template';
    this.result = null;
    this._job = 0;
    this.tabs.forEach((b) => b.addEventListener('click', () => this.showTab(b.dataset.tab)));
    this.text.addEventListener('click', (e) => {
      const s = e.target.closest('.s');
      if (!s) return;
      const ids = (s.dataset.ev || '').split(' ').filter(Boolean);
      this.onPickEvents(ids, true);
    });
  }

  // The vision-language caption of the setting: its own labelled block, never mixed into the
  // event sentences. Objects it names that the detector never saw are flagged.
  _scene(sc) {
    this.scene.hidden = !sc;
    if (!sc) return;
    $('#d-scene-text').textContent = sc.text;
    $('#d-scene-meta').textContent = `${sc.model.split('/').pop()} · ${sc.device === 'cuda' ? 'GPU' : 'CPU'} ${sc.dtype} · ${sc.frames.length} key frames in ${sc.seconds} s`;
    const flag = $('#d-scene-flag');
    flag.hidden = !sc.flagged;
    if (sc.flagged) flag.textContent = `Check: the caption names ${sc.unseen.join(', ')}, which the detector never saw in this clip.`;
  }

  clear(message = '') {
    this._job++;
    this.scene.hidden = true;
    this.text.replaceChildren();
    this.guard.hidden = true;
    if (message) this.tag.textContent = message;
  }

  setResult(result, { stream = true, label = '' } = {}) {
    this.result = result;
    this.label = label;
    const d = result.description;
    this.tabs.forEach((b) => b.classList.toggle('used', !!d && ((b.dataset.tab === 'model') === (d.used === 'model'))));
    // The template is the default describer, so it is what opens; the model tab is one tap away.
    this.showTab('template', { stream });
    this._scene(result.scene);
  }

  showTab(name, { stream = false } = {}) {
    this.tab = name;
    this.tabs.forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
    const d = this.result && this.result.description;
    this._job++;
    const job = this._job;
    this.text.replaceChildren();
    this.guard.hidden = true;
    if (!d) return;
    if (name === 'template') {
      this.tag.textContent = `${this.label}Deterministic template over the detected events. Tap a sentence to find it on the timeline.`;
      this._type(d.template.sentences, job, stream);
    } else if (!d.model) {
      this.tag.textContent = `${this.label}The local model rewrite was not run for this clip.`;
      this.text.textContent = 'In local mode, choose “template + local model” and run a clip to see the rewrite and whether the guard accepted it.';
    } else {
      const m = d.model;
      this.tag.textContent = `${this.label}${m.model || 'local model'} rewrote the template facts${m.seconds ? ` in ${m.seconds} s` : ''}.`;
      this._type([{ text: m.text, event_ids: [] }], job, stream);
      this.guard.hidden = false;
      this.guard.classList.toggle('ok', m.accepted);
      const tries = m.attempts || 1;
      const earlier = (m.attempt_log || []).slice(0, m.accepted ? tries - 1 : tries);
      this.guard.textContent = m.accepted
        ? `Guard: accepted${tries > 1 ? ` on attempt ${tries} (earlier attempts were rejected: ${earlier.join('; ')})` : ' on the first attempt'}. Objects, counts, times, directions and event verbs were checked against the event list.`
        : `Guard: rejected after ${tries} ${tries === 1 ? 'attempt' : 'attempts'} (${m.rejected_reason}). The template description is used instead.`;
    }
  }

  _type(sentences, job, stream) {
    const spans = sentences.map((s) => {
      const el = document.createElement('span');
      el.className = 's';
      el.dataset.ev = (s.event_ids || []).join(' ');
      this.text.append(el, ' ');
      return { el, text: s.text };
    });
    if (!stream || reduced()) {
      spans.forEach(({ el, text }) => { el.textContent = text; });
      return;
    }
    const caret = document.createElement('span');
    caret.className = 'caret';
    let si = 0, wi = 0;
    const tick = () => {
      if (job !== this._job) return;
      if (si >= spans.length) { caret.remove(); return; }
      const { el, text } = spans[si];
      const words = text.split(' ');
      wi++;
      el.textContent = words.slice(0, wi).join(' ');
      el.append(caret);
      if (wi >= words.length) { si++; wi = 0; }
      setTimeout(tick, 34);
    };
    tick();
  }

  // live streaming (server-sent tokens)
  beginLive() {
    this._job++;
    this.result = null;
    this.text.replaceChildren();
    this.guard.hidden = true;
    this.tag.textContent = 'Streaming from the local pipeline…';
    this.tabs.forEach((b) => b.classList.remove('used'));
    this.showTabOnly('template');
    this._live = null;
  }
  showTabOnly(name) {
    this.tab = name;
    this.tabs.forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
  }
  pushToken(text) {
    if (!this._live) {
      this._live = document.createElement('span');
      this._live.className = 's';
      this.text.append(this._live);
    }
    this._live.textContent += text;
  }
  pushModelToken(text) {
    if (this.tab !== 'model') {
      this.showTabOnly('model');
      this.text.replaceChildren();
      this._live = null;
      this.tag.textContent = 'The local model is rewriting the facts…';
    }
    this.pushToken(text);
  }

  highlight(eventIds) {
    const set = new Set(eventIds);
    $$('.s', this.text).forEach((s) => {
      const ids = (s.dataset.ev || '').split(' ').filter(Boolean);
      s.classList.toggle('sel', ids.some((i) => set.has(i)));
    });
    const first = $('.s.sel', this.text);
    if (first && this.tab === 'template') first.scrollIntoView({ block: 'nearest', behavior: reduced() ? 'auto' : 'smooth' });
  }
}
