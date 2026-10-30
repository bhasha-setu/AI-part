// API client. Uses the mock when the page is opened with ?mock=1.
import * as mock from './mock.js';

export const MOCK = new URLSearchParams(location.search).has('mock');

export class ApiError extends Error {
  constructor(status, detail, body) { super(detail); this.status = status; this.detail = detail; this.body = body || {}; }
}

async function call(method, path, { json, form } = {}) {
  const opts = { method, headers: {} };
  if (json !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(json); }
  if (form) opts.body = form;
  let res;
  try { res = await fetch(path, opts); }
  catch (e) { throw new ApiError(0, 'Can\'t reach the server. Check your connection.'); }
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON */ }
  if (!res.ok) {
    const d = body && body.detail;
    throw new ApiError(res.status, typeof d === 'string' ? d : 'Something went wrong.', body);
  }
  return body;
}

function audioForm(blob) {
  const t = (blob.type || '').toLowerCase();
  const ext = t.includes('ogg') ? 'ogg' : t.includes('mp4') || t.includes('aac') ? 'm4a' : t.includes('wav') ? 'wav' : 'webm';
  const f = new FormData();
  f.append('audio', blob, `speech.${ext}`);
  return f;
}

const q = encodeURIComponent;
const real = {
  health: () => call('GET', 'api/health'),
  courses: () => call('GET', 'api/courses'),
  scenarios: (cid, lid) => call('GET', `api/courses/${q(cid)}/scenarios?learner_id=${q(lid)}`),
  plan: (course_id, learner_id, request) => call('POST', 'api/plan', { json: { course_id, learner_id, request } }),
  buildScenario: (course_id, learner_id, option_id) => call('POST', 'api/scenarios', { json: { course_id, learner_id, option_id } }),
  startLesson: (course_id, learner_id, scenario_id) => call('POST', 'api/lessons', { json: { course_id, learner_id, scenario_id } }),
  turn: (sid, text) => call('POST', `api/lessons/${q(sid)}/turn`, { json: { text } }),
  audio: (sid, blob) => call('POST', `api/lessons/${q(sid)}/audio`, { form: audioForm(blob) }),
  stt: (blob, lang) => call('POST', 'api/stt' + (lang ? `?lang=${q(lang)}` : ''), { form: audioForm(blob) }),
};

export const api = MOCK ? mock.api : real;
