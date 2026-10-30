// Speak & Learn: single-page voice-first lesson UI.
import { api, ApiError, MOCK } from './api.js';
import { Recorder, micSupported, speak, cancelSpeech, isMuted, setMuted, store } from './audio.js';
import { h, clear, randomHex, humanize } from './ui.js';

const $app = document.getElementById('app');
const $live = document.getElementById('live');
const $mute = document.getElementById('muteBtn');
const $menuBtn = document.getElementById('menuBtn');
const $menu = document.getElementById('menu');
const AUTO_STOP_MS = 8000;

// ---- State ----
let learnerId = store.get('sl_learner_id');
if (!learnerId) { learnerId = 'l-' + randomHex(8); store.set('sl_learner_id', learnerId); }

const S = {
  view: 'loading',     // loading | course | home | lesson | results | expired
  courses: [], course: null,
  sttOk: micSupported, // flipped off by /api/health or a 503
  scenarios: [],
  request: '',         // home text box
  plan: null,          // {source, options, note}
  build: null,         // {option} while building a lesson
  buildError: null,    // {detail, fallbacks}
  lesson: null,        // current LessonState
  typed: false,        // typed input forced for the lesson
  notice: '',          // learner-facing message
  busy: null,          // "Thinking…" label while a request is in flight
};
let activeRec = null;  // current Recorder, if recording

// ---- Header ----
function syncMute() {
  const m = isMuted();
  $mute.textContent = m ? '🔇' : '🔊';
  $mute.setAttribute('aria-pressed', String(m));
  $mute.setAttribute('aria-label', m ? 'Unmute voice' : 'Mute voice');
}
$mute.addEventListener('click', () => { setMuted(!isMuted()); syncMute(); });
$menuBtn.addEventListener('click', () => {
  $menu.hidden = !$menu.hidden;
  $menuBtn.setAttribute('aria-expanded', String(!$menu.hidden));
});
document.addEventListener('click', e => {
  if (!$menu.hidden && !$menu.contains(e.target) && e.target !== $menuBtn) { $menu.hidden = true; $menuBtn.setAttribute('aria-expanded', 'false'); }
});
document.getElementById('changeCourse').addEventListener('click', () => { $menu.hidden = true; go('course'); });
document.getElementById('brand').addEventListener('click', () => { if (S.course) go('home'); });
syncMute();

// ---- Navigation / rendering ----
function go(view) {
  cancelSpeech();
  S.view = view; S.notice = ''; S.busy = null;
  if (view === 'home') { S.build = null; S.buildError = null; loadScenarios(); }
  render();
  window.scrollTo(0, 0);
}

function render() {
  if (activeRec) { activeRec.cancel(); activeRec = null; }
  clear($app);
  const views = { loading: viewLoading, course: viewCourse, home: viewHome, lesson: viewLesson, results: viewResults, expired: viewExpired };
  $app.append(...[].concat((views[S.view] || viewLoading)()));
}

function announce(t) { $live.textContent = ''; setTimeout(() => { $live.textContent = t; }, 30); }

// Run an async request with a busy label; blocks input while in flight.
async function run(label, fn) {
  if (S.busy) return;
  S.busy = label; render();
  try { return await fn(); }
  finally { S.busy = null; }
}

const errText = e => (e instanceof ApiError ? e.detail : 'Something went wrong. Please try again.');

// ---- Shared components ----
const notice = () => S.notice ? h('p', { class: 'notice', role: 'alert' }, S.notice) : null;
const status = () => S.busy ? h('p', { class: 'status' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), S.busy) : null;

// Mic button: tap to start, tap to stop, auto-stop after ~8 s.
function micButton({ onBlob, hint = 'Tap to speak', big = true }) {
  const disabled = !!S.busy;
  const label = h('span', { class: 'mic-label' }, disabled ? (S.busy || '') : hint);
  const btn = h('button', { class: 'mic' + (big ? '' : ' mic-small'), type: 'button', disabled, 'aria-label': 'Record your voice' }, h('span', { class: 'mic-icon', 'aria-hidden': 'true' }, '🎤'));
  const wrap = h('div', { class: 'mic-wrap' }, btn, label);
  let timer = null;

  const stop = async () => {
    clearTimeout(timer);
    const rec = activeRec; activeRec = null;
    if (!rec) return;
    btn.disabled = true;
    const blob = await rec.stop();
    onBlob(blob);
  };
  btn.addEventListener('click', async () => {
    if (activeRec) return stop();
    cancelSpeech(); // never record over TTS
    const rec = new Recorder(l => btn.style.setProperty('--lvl', l.toFixed(2)));
    try { await rec.start(); }
    catch (e) {
      S.typed = true; S.notice = 'I can\'t use the microphone. You can type instead.';
      S.sttOk = false; render(); return;
    }
    activeRec = rec;
    btn.classList.add('recording');
    btn.setAttribute('aria-label', 'Stop recording');
    label.textContent = 'Listening… tap to stop';
    announce('Listening');
    timer = setTimeout(stop, AUTO_STOP_MS);
  });
  return wrap;
}

function textInput({ placeholder, onSend, value = '', onInput, button = 'Send', autofocus = false }) {
  const input = h('input', { type: 'text', class: 'text', placeholder, value, autocomplete: 'off', autocapitalize: 'sentences', 'aria-label': placeholder, disabled: !!S.busy });
  if (onInput) input.addEventListener('input', () => onInput(input.value));
  const send = () => { const v = input.value.trim(); if (v) onSend(v); };
  input.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); send(); } });
  const btn = h('button', { class: 'btn primary', type: 'button', disabled: !!S.busy, onclick: send }, button);
  if (autofocus) setTimeout(() => input.focus(), 0);
  return h('div', { class: 'text-row' }, input, btn);
}

const speakerBtn = (text, lang, label = 'Hear this') =>
  h('button', { class: 'speaker', type: 'button', 'aria-label': label, onclick: () => speak(text, lang, true) }, '🔈');

// ---- Loading / course pick ----
function viewLoading() { return [h('p', { class: 'status' }, h('span', { class: 'spinner', 'aria-hidden': 'true' }), 'Loading…'), notice()]; }

function viewCourse() {
  return [
    h('h1', null, 'What do you want to learn?'),
    h('p', { class: 'lead' }, 'Pick a course. You can change it later from the menu.'),
    notice(),
    h('div', { class: 'cards' }, S.courses.map(c =>
      h('button', { class: 'card tap', type: 'button', onclick: () => chooseCourse(c) },
        h('strong', null, c.title), h('span', { class: 'chip' }, c.cefr_level)))),
  ];
}

function chooseCourse(c) {
  S.course = c; store.set('sl_course', c.course_id);
  S.plan = null; S.request = '';
  go('home');
}

// ---- Home ----
async function loadScenarios() {
  try { S.scenarios = await api.scenarios(S.course.course_id, learnerId); } catch (e) { S.scenarios = S.scenarios || []; }
  if (S.view === 'home' && !S.busy && !S.build) render();
}

function viewHome() {
  const out = [h('h1', null, 'What do you want to learn today?')];
  const building = S.build;
  const disabled = !!S.busy || !!building;
  if (S.sttOk) {
    out.push(micButton({ hint: 'Tap and tell me', onBlob: onHomeBlob }));
  }
  out.push(textInput({
    placeholder: 'e.g. I want to ask for directions', value: S.request, button: 'Find lessons',
    onInput: v => { S.request = v; }, onSend: v => { S.request = v; findLessons(); },
  }));
  out.push(notice());
  if (!building) out.push(status());

  if (building) {
    out.push(h('div', { class: 'building', role: 'status' },
      h('span', { class: 'spinner big', 'aria-hidden': 'true' }),
      h('strong', null, 'Building your lesson…'),
      h('span', null, `“${building.title}”. This can take up to 40 seconds.`)));
  } else if (S.buildError) {
    out.push(h('div', { class: 'notice', role: 'alert' }, S.buildError.detail));
    if (S.buildError.fallbacks.length) {
      out.push(h('h2', null, 'Try one of these instead'));
      out.push(h('div', { class: 'cards' }, S.buildError.fallbacks.map(id => {
        const sc = S.scenarios.find(x => x.scenario_id === id);
        return scenarioCard(sc || { scenario_id: id, title: humanize(id), can_do: '' }, startLesson);
      })));
    }
  } else if (S.plan) {
    if (S.plan.note) out.push(h('p', { class: 'note' }, S.plan.note));
    out.push(h('h2', null, 'Pick a lesson'));
    out.push(h('div', { class: 'cards' }, S.plan.options.map(o => optionCard(o, disabled))));
  }

  out.push(h('h2', null, 'Your lessons'));
  out.push(S.scenarios.length
    ? h('div', { class: 'cards' }, S.scenarios.map(sc => scenarioCard(sc, startLesson, disabled)))
    : h('p', { class: 'muted' }, 'Lessons will show up here.'));
  return out;
}

function optionCard(o, disabled) {
  return h('button', { class: 'card tap option', type: 'button', disabled, onclick: () => pickOption(o) },
    h('strong', null, o.title),
    o.situation && h('span', { class: 'sub' }, o.situation),
    o.can_do && h('span', { class: 'cando' }, '✔ ' + o.can_do),
    (o.phrases_preview || []).length ? h('span', { class: 'chips' }, o.phrases_preview.map(p => h('span', { class: 'chip' }, p))) : null,
    o.why && h('span', { class: 'why' }, o.why),
    o.est_minutes ? h('span', { class: 'meta' }, `About ${o.est_minutes} min`) : null);
}

function scenarioCard(sc, onPick, disabled = !!S.busy) {
  return h('button', { class: 'card tap', type: 'button', disabled, onclick: () => onPick(sc.scenario_id) },
    h('span', { class: 'row' }, h('strong', null, sc.title), sc.source === 'generated' ? h('span', { class: 'badge gen' }, 'generated') : null),
    sc.can_do && h('span', { class: 'sub' }, sc.can_do),
    sc.item_count ? h('span', { class: 'meta' }, `${sc.item_count} phrases`) : null);
}

async function onHomeBlob(blob) {
  S.notice = '';
  const r = await run('Listening…', async () => {
    try { return await api.stt(blob, S.course.stt_lang); }
    catch (e) { S.notice = e.status === 503 ? 'Voice isn\'t available right now. Please type instead.' : errText(e); if (e.status === 503) S.sttOk = false; return null; }
  });
  if (r && r.transcript) S.request = r.transcript; else if (r) S.notice = 'I didn\'t catch that.';
  render();
}

async function findLessons() {
  if (S.busy) return;
  S.notice = ''; S.buildError = null; S.plan = null;
  const plan = await run('Finding lessons…', async () => {
    try { return await api.plan(S.course.course_id, learnerId, S.request); }
    catch (e) { S.notice = errText(e); return null; }
  });
  if (plan) { S.plan = plan; announce(`${plan.options.length} lessons found`); }
  render();
}

async function pickOption(o) {
  if (S.busy || S.build) return;
  if (o.scenario_id) return startLesson(o.scenario_id);
  S.build = o; S.notice = ''; S.buildError = null; S.busy = 'Building your lesson…'; render();
  try {
    const sc = await api.buildScenario(S.course.course_id, learnerId, o.option_id);
    S.build = null; S.busy = null;
    await startLesson(sc.scenario_id);
  } catch (e) {
    S.build = null; S.busy = null;
    if (e.status === 422) S.buildError = { detail: e.detail, fallbacks: (e.body && e.body.fallback_scenario_ids) || [] };
    else S.notice = errText(e);
    render();
  }
}

// ---- Lesson ----
async function startLesson(scenario_id) {
  if (S.busy && !S.build) return;
  S.busy = 'Starting your lesson…'; S.notice = ''; render();
  try {
    const L = await api.startLesson(S.course.course_id, learnerId, scenario_id);
    S.busy = null; S.typed = !S.sttOk;
    applyLesson(L);
  } catch (e) {
    S.busy = null; S.view = 'home'; S.notice = errText(e); render();
  }
}

// Take a new LessonState: choose the view, speak the reply.
function applyLesson(L) {
  S.lesson = L;
  const ex = L.reply.expects;
  S.view = (ex === 'done' || ex === 'held') ? 'results' : 'lesson';
  S.busy = null;
  render();
  window.scrollTo(0, 0);
  announce(L.reply.text);
  speak(L.reply.text, L.tts_lang);
  if (S.view === 'lesson' && L.reply.expects === 'answer' && S.typed) {
    const i = $app.querySelector('input.text'); if (i) i.focus();
  }
}

// Shared error handling for /turn and /audio.
function lessonError(e) {
  S.busy = null;
  if (e.status === 404) { S.view = 'expired'; render(); return; }
  if (e.status === 503) { S.typed = true; S.sttOk = false; S.notice = 'Voice isn\'t available right now. Please type your answer.'; }
  else S.notice = e.status === 422 ? e.detail : errText(e);
  render();
  if (S.typed) { const i = $app.querySelector('input.text'); if (i) i.focus(); }
}

async function sendText(text) {
  if (S.busy) return;
  cancelSpeech(); S.notice = ''; S.busy = 'Thinking…'; render();
  try { applyLesson(await api.turn(S.lesson.session_id, text)); } catch (e) { lessonError(e); }
}

async function sendAudio(blob) {
  if (S.busy) return;
  S.notice = ''; S.busy = 'Thinking…'; render();
  try { applyLesson(await api.audio(S.lesson.session_id, blob)); } catch (e) { lessonError(e); }
}

function progressBar(L) {
  const p = L.progress, n = p.stages.length;
  const inItems = L.reply.stage === 'guided_practice' && p.item_count;
  const frac = inItems ? Math.min(1, p.item_index / p.item_count) : 0;
  const pct = Math.round(((p.stage_index + frac) / n) * 100);
  return h('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': String(pct), 'aria-label': 'Lesson progress' },
    h('div', { class: 'progress-fill', style: `width:${pct}%` }));
}

const LEARNER_NAMES = new Set(['you', 'learner', 'user', 'student', 'me']);
function bubbles(lines, lang) {
  const wrap = h('div', { class: 'dialogue' }, (lines || []).map(l => {
    const me = LEARNER_NAMES.has(String(l.speaker).trim().toLowerCase());
    return h('div', { class: 'bubble-row ' + (me ? 'me' : 'them') },
      h('div', { class: 'bubble' }, h('span', { class: 'who' }, l.speaker), h('span', null, l.text)),
      speakerBtn(l.text, lang, 'Hear ' + l.speaker));
  }));
  setTimeout(() => { const last = wrap.lastElementChild; if (last && last.scrollIntoView) last.scrollIntoView({ block: 'nearest' }); }, 50);
  return wrap;
}

const VERDICT_CLASS = { correct: 'fb-good', surface_error: 'fb-warn', meaning_error: 'fb-neutral', off_topic: 'fb-neutral' };

function hintDots(level) {
  return h('span', { class: 'dots', role: 'img', 'aria-label': `Help level ${level} of 3` },
    [0, 1, 2, 3].map(i => h('span', { class: 'dot' + (i < level ? ' on' : '') })));
}

function stageBody(L) {
  const r = L.reply, d = L.display || {}, lang = L.tts_lang;
  const heading = d.heading || L.scenario.title;
  const replay = h('button', { class: 'link', type: 'button', onclick: () => speak(r.text, lang, true) }, '🔈 Hear again');
  switch (r.stage) {
    case 'intro':
      return [h('h1', null, heading), L.scenario.can_do && h('p', { class: 'cando big' }, '✔ ' + L.scenario.can_do), h('p', { class: 'lead' }, r.text), replay];
    case 'model_dialogue':
      return [h('h1', null, heading), h('p', { class: 'lead' }, r.text), bubbles(d.dialogue, lang)];
    case 'useful_language':
      return [h('h1', null, heading), h('p', { class: 'lead' }, r.text),
        h('div', { class: 'cards' }, (d.phrases || []).map(p =>
          h('button', { class: 'card tap phrase', type: 'button', onclick: () => speak(p.phrase, lang, true), 'aria-label': `${p.phrase}. Tap to hear.` },
            h('strong', null, p.phrase), h('span', { class: 'sub' }, p.meaning))))];
    case 'guided_practice': {
      const hint = d.hint, lvl = hint ? hint.level : 0;
      const prompt = r.prompt || (hint && hint.text) || r.text;
      return [
        h('h2', { class: 'stage-title' }, heading),
        S.lesson.transcript ? h('p', { class: 'said' }, 'You said: ', h('q', null, L.transcript)) : null,
        r.feedback ? h('p', { class: 'feedback ' + (VERDICT_CLASS[L.verdict] || 'fb-neutral') }, r.feedback) : null,
        h('div', { class: 'prompt-card' },
          h('p', { class: 'prompt' }, prompt),
          hint ? h('div', { class: 'hint' }, hintDots(lvl), lvl > 0 && hint.text !== prompt ? h('span', { class: 'hint-text' }, hint.text) : null) : null),
        replay,
      ];
    }
    case 'roleplay':
      return [h('h2', { class: 'stage-title' }, heading), bubbles(d.dialogue, lang)];
    default:
      return [h('h1', null, heading), h('p', { class: 'lead' }, r.text)];
  }
}

function viewLesson() {
  const L = S.lesson;
  const parts = [progressBar(L), h('div', { class: 'stage' }, stageBody(L)), notice(), status()];
  const guided = L.reply.stage === 'guided_practice';
  // For guided practice, show "you said" only when the last turn was spoken or typed
  if (L.reply.expects === 'continue') {
    parts.push(h('button', { class: 'btn primary xl', type: 'button', disabled: !!S.busy, onclick: () => sendText('') }, 'Continue'));
    if (S.sttOk && !S.typed) parts.push(micButton({ onBlob: sendAudio, hint: 'or say “next”', big: false }));
  } else if (L.reply.expects === 'answer') {
    if (S.sttOk && !S.typed) {
      parts.push(micButton({ onBlob: sendAudio, hint: guided ? 'Tap and say it' : 'Tap and answer' }));
      parts.push(h('button', { class: 'link', type: 'button', disabled: !!S.busy, onclick: () => { S.typed = true; render(); const i = $app.querySelector('input.text'); if (i) i.focus(); } }, 'Type instead'));
    } else {
      parts.push(textInput({ placeholder: 'Type your answer', onSend: sendText, autofocus: true }));
      if (S.sttOk) parts.push(h('button', { class: 'link', type: 'button', disabled: !!S.busy, onclick: () => { S.typed = false; render(); } }, 'Use microphone'));
    }
  }
  return parts;
}

function viewExpired() {
  return [h('h1', null, 'This lesson expired'),
    h('p', { class: 'lead' }, 'It was closed after a while or the server restarted. Start again from the beginning.'),
    h('button', { class: 'btn primary xl', type: 'button', onclick: () => go('home') }, 'Back home')];
}

// ---- Results ----
function ring(done, total) {
  const R = 52, C = 2 * Math.PI * R, frac = total ? done / total : 0;
  const ns = 'http://www.w3.org/2000/svg';
  const mk = (t, a) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  const svg = mk('svg', { viewBox: '0 0 120 120', class: 'ring', role: 'img', 'aria-label': `${done} of ${total} from memory` });
  svg.append(mk('circle', { cx: 60, cy: 60, r: R, class: 'ring-bg' }),
    mk('circle', { cx: 60, cy: 60, r: R, class: 'ring-fg', 'stroke-dasharray': `${(C * frac).toFixed(1)} ${C.toFixed(1)}`, transform: 'rotate(-90 60 60)' }),
    Object.assign(mk('text', { x: 60, y: 68, 'text-anchor': 'middle', class: 'ring-text' }), { textContent: `${done}/${total}` }));
  return svg;
}

function viewResults() {
  const L = S.lesson, res = L.result;
  const out = [h('h1', null, L.reply.expects === 'held' ? 'Lesson paused' : 'Lesson complete')];
  if (res) {
    out.push(h('div', { class: 'ring-wrap' }, ring(res.independent_count, res.item_count), h('span', { class: 'muted' }, 'said from memory')));
    out.push(h('p', { class: 'lead' }, res.message));
    out.push(h('ul', { class: 'items' }, (res.items || []).map(it =>
      h('li', { class: 'item' }, h('span', { class: 'item-prompt' }, it.prompt || humanize(it.item_id)),
        h('span', { class: 'badge ' + (it.independent ? 'good' : 'help') }, it.independent ? 'From memory' : humanize(it.outcome))))));
  } else out.push(h('p', { class: 'lead' }, L.reply.text));
  out.push(h('div', { class: 'actions' },
    h('button', { class: 'btn primary xl', type: 'button', onclick: () => startLesson(L.scenario.scenario_id) }, 'Practice again'),
    h('button', { class: 'btn xl', type: 'button', onclick: () => go('home') }, 'Home')));
  out.push(notice());
  return out;
}

// ---- Boot ----
async function boot() {
  render();
  try {
    S.courses = await api.courses();
  } catch (e) { S.notice = errText(e); S.view = 'course'; render(); return; }
  try {
    const hl = await api.health();
    if (hl && hl.stt && hl.stt.available === false) S.sttOk = false;
  } catch (e) { /* health is optional */ }
  const saved = store.get('sl_course');
  S.course = S.courses.find(c => c.course_id === saved) || null;
  go(S.course ? 'home' : 'course');
}
if (MOCK) document.title += ' (mock)';
boot();
