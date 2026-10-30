// Mic recording, level meter and speech synthesis helpers.

export const micSupported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);

const MUTE_KEY = 'sl_muted';
export const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* ignore */ } },
};

// ---- Text to speech ----
let muted = store.get(MUTE_KEY) === '1';
export const isMuted = () => muted;
export function setMuted(m) { muted = m; store.set(MUTE_KEY, m ? '1' : '0'); if (m) cancelSpeech(); }

const synth = window.speechSynthesis;
export const ttsSupported = !!synth && typeof SpeechSynthesisUtterance !== 'undefined';
let speakingCount = 0;
export const isSpeaking = () => ttsSupported && (synth.speaking || speakingCount > 0);

function pickVoice(lang) {
  const voices = synth.getVoices();
  const l = lang.toLowerCase();
  return voices.find(v => v.lang.toLowerCase() === l)
    || voices.find(v => v.lang.toLowerCase().startsWith(l.split('-')[0]));
}

export function cancelSpeech() { if (ttsSupported) { synth.cancel(); speakingCount = 0; } }

// force=true ignores mute (used for the per-line speaker buttons).
export function speak(text, lang, force = false) {
  if (!ttsSupported || !text || (muted && !force)) return;
  cancelSpeech();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang || 'en-US';
  const v = pickVoice(u.lang);
  if (v) u.voice = v;
  u.rate = 0.92;
  speakingCount++;
  const done = () => { speakingCount = Math.max(0, speakingCount - 1); };
  u.onend = done; u.onerror = done;
  synth.speak(u);
}

// ---- Recording ----
export class Recorder {
  constructor(onLevel) { this.onLevel = onLevel; }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.chunks = [];
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
    const mt = types.find(t => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t));
    this.rec = mt ? new MediaRecorder(this.stream, { mimeType: mt }) : new MediaRecorder(this.stream);
    this.rec.ondataavailable = e => { if (e.data && e.data.size) this.chunks.push(e.data); };
    this.rec.start();
    this._meter();
  }

  // Resolves with a Blob once the recorder has flushed.
  stop() {
    return new Promise(resolve => {
      const finish = () => {
        this._cleanup();
        resolve(new Blob(this.chunks, { type: (this.rec && this.rec.mimeType) || 'audio/webm' }));
      };
      if (!this.rec || this.rec.state === 'inactive') return finish();
      this.rec.onstop = finish;
      this.rec.stop();
    });
  }

  cancel() { try { if (this.rec && this.rec.state !== 'inactive') { this.rec.onstop = null; this.rec.stop(); } } catch (e) { /* */ } this._cleanup(); }

  _meter() {
    if (!this.onLevel) return;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
      const an = this.ctx.createAnalyser(); an.fftSize = 256;
      this.ctx.createMediaStreamSource(this.stream).connect(an);
      const buf = new Uint8Array(an.fftSize);
      const tick = () => {
        an.getByteTimeDomainData(buf);
        let peak = 0; for (const b of buf) peak = Math.max(peak, Math.abs(b - 128));
        this.onLevel(Math.min(1, peak / 64));
        this.raf = requestAnimationFrame(tick);
      };
      tick();
    } catch (e) { /* meter is optional */ }
  }

  _cleanup() {
    cancelAnimationFrame(this.raf);
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.ctx) { try { this.ctx.close(); } catch (e) { /* */ } }
    this.stream = null; this.ctx = null;
    if (this.onLevel) this.onLevel(0);
  }
}
