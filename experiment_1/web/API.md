# Web API contract (v1)

The browser app (`web/static/`) talks to the FastAPI server (`web/server.py`) only
through these endpoints. All bodies are JSON unless marked multipart. Errors are
`{"detail": "<message>"}` with a 4xx/5xx status.

There are no accounts. The browser makes up a `learner_id` once (for example
`"l-" + 8 random hex chars`), keeps it in `localStorage`, and sends it with every call.

Voice is the main input. The browser records the mic with `MediaRecorder` (webm/opus
or whatever the browser gives) and uploads the blob. The server transcribes it with a
local Whisper model. Typed text is always accepted as a fallback. Replies are spoken in
the browser with `speechSynthesis`, using `tts_lang`.

## GET /api/health

```json
{"judge": "gemini", "generator": "gemini", "roleplay": "gemini",
 "stt": {"available": true, "engine": "faster-whisper", "model": "small"},
 "tts": "browser"}
```
`judge`/`generator`/`roleplay` are `"gemini"` or `"stub"`/`"off"`. When `stt.available` is
false, the UI hides the mic and uses typed input.

## GET /api/courses

```json
[{"course_id": "english_for_hindi_a1", "title": "English for Hindi speakers",
  "target_language": "en", "learner_language": "hi", "cefr_level": "A1",
  "tts_lang": "en-IN", "stt_lang": "en"}]
```

## GET /api/courses/{course_id}/scenarios?learner_id=...

The authored scenarios plus the ones generated for this learner.
```json
[{"scenario_id": "ordering_food", "title": "Ordering food",
  "can_do": "Order food: ask for the menu, ...", "item_count": 5,
  "source": "authored"}]
```
`source` is `"authored"` or `"generated"`.

## POST /api/plan

The learner says or types what they want to learn. The server answers with 2-3 lesson
options shaped by the pedagogy, so the learner chooses the task and the server chooses
the method.
Request: `{"course_id": "...", "learner_id": "...", "request": "I want to ask for directions"}`
```json
{"source": "llm",
 "options": [
   {"option_id": "opt-3f2a1c", "title": "Asking the way to the station",
    "can_do": "Ask where a place is and understand left, right and straight on.",
    "situation": "You are new in town and need the railway station.",
    "phrases_preview": ["Where is the station?", "Go straight on."],
    "why": "Short, high-frequency questions you will say on day one.",
    "est_minutes": 6,
    "scenario_id": null}
 ],
 "note": ""}
```
- `source` is `"llm"` or `"fallback"`. On a fallback, the options are authored
  scenarios and `note` says why, in one learner-facing sentence.
- If `scenario_id` is non-null, the option is an existing scenario: skip
  `/api/scenarios` and start the lesson directly.
- Options last 1 hour, per learner.

## POST /api/scenarios

Turns the chosen option into a validated lesson. This takes 10-40 s, so the UI must
show progress ("Building your lesson...") and allow no double submit.
Request: `{"course_id": "...", "learner_id": "...", "option_id": "opt-3f2a1c"}`
```json
{"scenario_id": "gen_asking_the_way_3f2a1c", "title": "Asking the way to the station",
 "can_do": "...", "item_count": 4, "source": "generated", "attempts": 1}
```
On a generation failure (the content didn't pass validation twice, or the LLM is
unavailable), the status is 422 with
`{"detail": "...", "fallback_scenario_ids": ["ordering_food", ...]}`.

## POST /api/lessons

Request: `{"course_id": "...", "learner_id": "...", "scenario_id": "..."}`
Response: a **LessonState** (below).

## POST /api/lessons/{session_id}/turn

Request: `{"text": "can i have the menu please"}`. Response: a **LessonState**.
For stages where `expects == "continue"`, send `{"text": ""}` to move on.

## POST /api/lessons/{session_id}/audio  (multipart)

Form field `audio` holds the recorded blob. The server transcribes it, then handles it
exactly like `/turn`. Response: a **LessonState** with `transcript` set. If STT isn't
available the status is 503. An empty transcript gives 422
`{"detail": "I didn't catch that."}`, and the lesson state is unchanged.

## POST /api/stt  (multipart)

Field `audio`, plus an optional query `?lang=en`. Response: `{"transcript": "..."}`. Used
for the "what do you want to learn?" box, by voice.

## GET /api/lessons/{session_id}

Response: the current **LessonState**, with `transcript: null`. A server restart loses
live lessons, so the status is 404 and the UI offers to start again.

## LessonState

```json
{
  "session_id": "srv-1a2b3c4d",
  "scenario": {"scenario_id": "ordering_food", "title": "Ordering food", "can_do": "..."},
  "tts_lang": "en-IN",
  "transcript": "can i have menu please",
  "reply": {"text": "Almost. Check your sentence. Try again: ...",
            "stage": "guided_practice", "expects": "answer",
            "feedback": "Almost. Check your sentence.",
            "prompt": "Try again: You sit down at a restaurant. Ask for the menu.",
            "item_id": "food_menu"},
  "frame": {"stage": "guided_practice", "expects": "answer", "item_id": "food_menu",
            "hint_level": 0, "corrections_used": 0, "lesson_status": "in_progress"},
  "verdict": "surface_error",
  "progress": {"stages": ["intro", "model_dialogue", "useful_language",
                          "guided_practice", "roleplay", "assessment"],
               "stage_index": 3, "item_index": 0, "item_count": 4},
  "display": {
    "heading": "Your turn: say it from memory",
    "dialogue": [{"speaker": "Waiter", "text": "Welcome! Here is the menu."}],
    "phrases": [{"phrase": "I agree.", "meaning": "to agree with someone"}],
    "hint": {"level": 1, "text": "Menu → ? (ask politely)"}
  },
  "result": null
}
```

- `reply.text` is what is spoken (TTS). `reply.expects` is one of:
  - `continue`: show a "Continue" button. Voice ("ok", "next") also works; it sends any
    text.
  - `answer`: mic and text input.
  - `done` or `held`: the lesson is over. Show `result`.
- `verdict` is `null` except on guided-practice turns: `correct`, `surface_error`,
  `meaning_error` or `off_topic`. Use it for colour (green, amber, grey). Never show the
  raw word to the learner.
- `display` fields are present or empty depending on the stage:
  - `intro`: `heading`, plus `can_do` in `scenario`.
  - `model_dialogue`: `dialogue`.
  - `useful_language`: `phrases`.
  - `guided_practice`: `hint`, the current hint-ladder text. At levels 0 to 3 it is
    prompt, cue, first words, model answer.
  - `roleplay`: `dialogue` holds the conversation so far, partner and learner.
- `result` is non-null only when `expects` is `done` or `held`:
  ```json
  {"status": "complete",
   "items": [{"item_id": "food_menu", "prompt": "...", "outcome": "produced",
              "independent": true, "solved_level": 0}],
   "independent_count": 3, "item_count": 4,
   "roleplay_turns": 3,
   "message": "You said 3 of 4 from memory. Next time: food_friend_diet."}
  ```
