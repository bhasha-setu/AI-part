// Mock API for ?mock=1. Returns contract-shaped data and walks a fake lesson
// through all six stages. STT always returns MOCK_TRANSCRIPT.
export const MOCK_TRANSCRIPT = 'can I have the menu please';
const delay = (ms = 350) => new Promise(r => setTimeout(r, ms));

const COURSES = [
  { course_id: 'english_for_hindi_a1', title: 'English for Hindi speakers', target_language: 'en', learner_language: 'hi', cefr_level: 'A1', tts_lang: 'en-IN', stt_lang: 'en' },
  { course_id: 'hindi_for_english_a1', title: 'Hindi for English speakers', target_language: 'hi', learner_language: 'en', cefr_level: 'A1', tts_lang: 'hi-IN', stt_lang: 'hi' },
];
const SCENARIOS = [
  { scenario_id: 'ordering_food', title: 'Ordering food', can_do: 'Order food: ask for the menu, choose a dish and pay.', item_count: 2, source: 'authored' },
];
const generated = [];
const sessions = {};

const DIALOGUE = [
  { speaker: 'Waiter', text: 'Welcome! Here is the menu.' },
  { speaker: 'You', text: 'Thank you. Can I have water, please?' },
  { speaker: 'Waiter', text: 'Of course.' },
];
const PHRASES = [
  { phrase: 'Can I have the menu, please?', meaning: 'a polite way to ask' },
  { phrase: 'I would like a tea.', meaning: 'to order a drink' },
];
const PROGRESS_STAGES = ['intro', 'model_dialogue', 'useful_language', 'guided_practice', 'roleplay', 'assessment'];
const SCEN = { scenario_id: 'ordering_food', title: 'Ordering food', can_do: SCENARIOS[0].can_do };

function state(s, over) {
  const stageIdx = PROGRESS_STAGES.indexOf(over.stage);
  const st = {
    session_id: s.id, scenario: SCEN, tts_lang: s.tts_lang, transcript: null,
    reply: { text: '', stage: over.stage, expects: over.expects, feedback: null, prompt: null, item_id: null, ...(over.reply || {}) },
    frame: { stage: over.stage, expects: over.expects, item_id: null, hint_level: 0, corrections_used: 0, lesson_status: 'in_progress' },
    verdict: over.verdict || null,
    progress: { stages: PROGRESS_STAGES, stage_index: stageIdx, item_index: over.item_index || 0, item_count: 2 },
    display: { heading: over.heading || '', dialogue: [], phrases: [], hint: null, ...(over.display || {}) },
    result: over.result || null,
  };
  return st;
}

// step: 0 intro, 1 dialogue, 2 phrases, 3 item 1, 4 item 2, 5/6 roleplay, 7 done
function render(s, transcript) {
  let st;
  switch (s.step) {
    case 0: st = state(s, { stage: 'intro', expects: 'continue', heading: 'Ordering food', reply: { text: 'Today you will learn to order food. Ready?' } }); break;
    case 1: st = state(s, { stage: 'model_dialogue', expects: 'continue', heading: 'Listen to this conversation', display: { dialogue: DIALOGUE }, reply: { text: 'Listen to this conversation.' } }); break;
    case 2: st = state(s, { stage: 'useful_language', expects: 'continue', heading: 'Useful phrases', display: { phrases: PHRASES }, reply: { text: 'Here are two useful phrases.' } }); break;
    case 3: case 4: {
      const first = s.step === 3;
      const prompt = first ? 'You sit down at a restaurant. Ask for the menu.' : 'Order a cup of tea.';
      const lvl = first ? s.hint : 0;
      const hints = [prompt, 'Menu → ? (ask politely)', 'Can I have ...', 'Can I have the menu, please?'];
      const fb = s.feedback || null;
      st = state(s, {
        stage: 'guided_practice', expects: 'answer', verdict: s.verdict, item_index: s.step - 3,
        heading: 'Your turn: say it from memory',
        reply: { text: (fb ? fb + ' ' : '') + 'Try again: ' + prompt, feedback: fb, prompt, item_id: first ? 'food_menu' : 'food_tea' },
        display: { hint: { level: lvl, text: hints[lvl] } },
      });
      st.frame.hint_level = lvl; st.frame.item_id = st.reply.item_id;
      if (!fb) st.reply.text = prompt;
      break;
    }
    case 5: case 6:
      st = state(s, { stage: 'roleplay', expects: 'answer', heading: 'Roleplay: you are at the restaurant', item_index: 2,
        display: { dialogue: s.dialogue }, reply: { text: s.dialogue[s.dialogue.length - 1].text } });
      break;
    default:
      st = state(s, { stage: 'assessment', expects: 'done', item_index: 2, heading: 'Well done',
        display: { dialogue: s.dialogue },
        reply: { text: 'You said 1 of 2 from memory. Next time: food_friend_diet.' },
        result: { status: 'complete',
          items: [
            { item_id: 'food_menu', prompt: 'You sit down at a restaurant. Ask for the menu.', outcome: 'produced_with_help', independent: false, solved_level: 1 },
            { item_id: 'food_tea', prompt: 'Order a cup of tea.', outcome: 'produced', independent: true, solved_level: 0 },
          ],
          independent_count: 1, item_count: 2, roleplay_turns: 2,
          message: 'You said 1 of 2 from memory. Next time: food_friend_diet.' } });
  }
  st.transcript = transcript || null;
  s.verdict = null; s.feedback = null;
  return st;
}

function advance(s, text) {
  const step = s.step;
  if (step <= 2) s.step++;
  else if (step === 3) {
    if (!s.tried) { s.tried = true; s.hint = 1; s.verdict = 'surface_error'; s.feedback = 'Almost. Check your sentence.'; }
    else { s.step = 4; s.verdict = 'correct'; s.feedback = 'Great, that is right!'; }
  } else if (step === 4) { s.step = 5; s.verdict = 'correct'; s.feedback = 'Nice one!'; s.dialogue = [{ speaker: 'Waiter', text: 'What would you like to eat today?' }]; }
  else if (step === 5) { s.dialogue.push({ speaker: 'You', text: text || '...' }, { speaker: 'Waiter', text: 'Good choice. Anything to drink?' }); s.step = 6; }
  else if (step === 6) { s.dialogue.push({ speaker: 'You', text: text || '...' }, { speaker: 'Waiter', text: 'Enjoy your meal!' }); s.step = 7; }
}

export const api = {
  health: async () => ({ judge: 'stub', generator: 'stub', roleplay: 'stub', stt: { available: true, engine: 'mock', model: 'mock' }, tts: 'browser' }),
  courses: async () => { await delay(); return COURSES; },
  scenarios: async () => { await delay(); return [...SCENARIOS, ...generated]; },
  plan: async (course_id, learner_id, request) => {
    await delay(700);
    return { source: 'llm', note: '', options: [
      { option_id: 'opt-aaa111', title: 'Asking the way to the station', can_do: 'Ask where a place is and understand left, right and straight on.', situation: 'You are new in town and need the railway station.', phrases_preview: ['Where is the station?', 'Go straight on.'], why: 'Short, high-frequency questions you will say on day one.', est_minutes: 6, scenario_id: null },
      { option_id: 'opt-bbb222', title: 'Ordering food', can_do: SCENARIOS[0].can_do, situation: 'You are hungry at a small restaurant.', phrases_preview: ['Can I have the menu?', 'I would like tea.'], why: 'You already have this one ready.', est_minutes: 4, scenario_id: 'ordering_food' },
    ] };
  },
  buildScenario: async (course_id, learner_id, option_id) => {
    await delay(2500);
    const s = { scenario_id: 'gen_asking_the_way_aaa111', title: 'Asking the way to the station', can_do: 'Ask where a place is.', item_count: 2, source: 'generated', attempts: 1 };
    if (!generated.length) generated.push(s);
    return s;
  },
  startLesson: async (course_id, learner_id, scenario_id) => {
    await delay();
    const id = 'srv-' + Math.random().toString(16).slice(2, 10);
    const tts_lang = (COURSES.find(c => c.course_id === course_id) || COURSES[0]).tts_lang;
    const s = sessions[id] = { id, step: 0, hint: 0, tts_lang, dialogue: [] };
    return render(s);
  },
  turn: async (sid, text) => {
    await delay();
    const s = sessions[sid];
    if (!s) { const { ApiError } = await import('./api.js'); throw new ApiError(404, 'Lesson not found'); }
    advance(s, text);
    return render(s);
  },
  audio: async (sid, blob) => {
    await delay(700);
    const s = sessions[sid];
    if (!s) { const { ApiError } = await import('./api.js'); throw new ApiError(404, 'Lesson not found'); }
    advance(s, MOCK_TRANSCRIPT);
    return render(s, MOCK_TRANSCRIPT);
  },
  stt: async () => { await delay(500); return { transcript: 'I want to ask for directions' }; },
};
