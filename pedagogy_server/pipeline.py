"""The turn pipeline: transcript in, reply out.

    PedagogyServer.start() -> Reply          first stage of the lesson
    PedagogyServer.handle(transcript) -> Reply

Per turn: build the frame; in guided practice, judge the transcript (TurnJudge) and
hand the guarded verdict to the existing Lesson/Session, which decide the hint level,
retries, the item outcome, the next stage and the end of the lesson. Presentation
stages advance on any utterance. Every turn and every policy decision is written to
``decisions.jsonl`` with its reason. The Session's own turn log (``turns.jsonl``) is
unchanged and still feeds the summary and assessment.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from guided_practice.lesson import STAGE_GUIDED_PRACTICE, Lesson
from guided_practice.policy import HoldPolicy
from guided_practice.session import render_prompt

from .config import load_judge_config, load_reply_strings
from .models import Frame, LogRecord, Reply, ServerSession, Turn
from .turn_judge import TurnJudge

REDACTED = "[redacted]"


class RecordingPolicy:
    """Wraps the lesson policy and keeps each decision so the server can log it."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self._pending: list = []

    def after_stage(self, stage, record, history):
        decision = self.inner.after_stage(stage, record, history)
        self._pending.append((stage, record, decision))
        return decision

    def drain(self) -> list:
        out, self._pending = self._pending, []
        return out


class DecisionLogger:
    """JSON Lines. As a last line of defence, the API key's value (if set in the
    environment) is replaced before a line is written. Nothing should contain it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: LogRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=False)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            line = line.replace(key, REDACTED)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    @staticmethod
    def read(path: str | Path) -> list:
        path = Path(path)
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class PedagogyServer:
    def __init__(self, course, scenario_id: str, learner_id: str, state_dir: str | Path,
                 judge, *, clock=None, policy=None, session_id: Optional[str] = None,
                 judge_config=None) -> None:
        self.course = course
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or ("srv-" + uuid.uuid4().hex[:8])
        self.policy = RecordingPolicy(policy or HoldPolicy())
        self.lesson = Lesson(course, scenario_id, learner_id, self.state_dir / "learners.json",
                             self.state_dir / "turns.jsonl", clock=clock, policy=self.policy,
                             lesson_id=self.session_id)
        self.clock = self.lesson.clock
        self.strings = load_reply_strings(course.ui.code)
        self.turn_judge = TurnJudge(course, judge, judge_config or load_judge_config(), self.strings)
        self.state = ServerSession(self.session_id, learner_id, course.course_id, scenario_id)
        self.log = DecisionLogger(self.state_dir / "decisions.jsonl")
        self._view = None

    # --- public API -----------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._view is not None and self._view.expects in ("done", "held")

    def frame(self) -> Frame:
        view = self._view
        session = self.lesson.guided_session
        item = session.current_item if session is not None else None
        return Frame(
            stage=self.lesson.current_stage if self.lesson.status == "in_progress" else view.stage,
            expects=view.expects if view else "start",
            item_id=item.id if item else "",
            hint_level=session.hint_level if item else None,
            awaiting_follow_up=bool(session and session.awaiting_follow_up),
            corrections_used=self.state.corrections_used,
            corrected_error_ids=tuple(sorted(self.state.corrected_error_ids)),
            lesson_status=self.lesson.status,
        )

    def start(self) -> Reply:
        if self._view is not None:
            raise RuntimeError("server already started")
        frame = Frame(stage=self.lesson.current_stage, expects="start")
        view = self.lesson.start()
        return self._after(frame, None, view, feedback="", judged=None)

    def handle(self, transcript: str) -> Reply:
        if self._view is None:
            raise RuntimeError("call start() before handle()")
        if self.done:
            raise RuntimeError(f"lesson is {self.lesson.status}")
        frame = self.frame()
        judged, prompt_override = None, ""
        if frame.expects == "continue":
            view = self.lesson.continue_()
            feedback = ""
        elif frame.stage == STAGE_GUIDED_PRACTICE and not frame.awaiting_follow_up:
            session = self.lesson.guided_session
            judged = self.turn_judge.judge_turn(self.lesson.scenario, session.current_item,
                                                session.hint_level, transcript, self.state)
            view = self.lesson.submit(transcript, evaluation=judged.evaluation)
            feedback = judged.evaluation.feedback
            prompt_override = self._retry_prompt(frame, judged)
        else:
            # A follow-up answer (unscored) or a roleplay turn: no judge call.
            view = self.lesson.submit(transcript)
            feedback = view.feedback
        return self._after(frame, transcript, view, feedback, judged, prompt_override)

    # --- internals --------------------------------------------------------------

    def _retry_prompt(self, frame: Frame, judged) -> str:
        """Session's surface-error retry prompt assumes a correction was shown. When none
        was, and the learner stays on the same item and level, use the neutral line."""
        if judged.evaluation.verdict != "surface_error" or judged.correction_shown:
            return ""
        session = self.lesson.guided_session
        item = session.current_item if session is not None else None
        if item is None or item.id != frame.item_id or session.hint_level != frame.hint_level:
            return ""
        return self.strings.retry_without_correction.format(
            prompt=render_prompt(item, frame.hint_level, self.course.ui))

    def _reply(self, view, feedback: str, prompt_override: str = "") -> Reply:
        parts = [feedback]
        prompt = ""
        if view.expects == "continue":
            parts += view.lines
        elif view.expects == "answer":
            prompt = prompt_override or view.prompt
            parts += view.lines + [prompt]
        elif view.expects == "done":
            parts.append(self.strings.stage_lines["complete"])
        elif view.expects == "held":
            parts.append(self.strings.stage_lines["held"])
        text = " ".join(p for p in parts if p)
        return Reply(text=text, stage=view.stage, expects=view.expects, feedback=feedback,
                     prompt=prompt, item_id=view.item_id)

    def _record(self, kind: str, stage: str, reason: str, data: dict) -> None:
        self.log.log(LogRecord(timestamp=self.clock().isoformat(), session_id=self.session_id,
                               turn_index=len(self.state.turns), kind=kind, stage=stage,
                               reason=reason, data=data))

    def _after(self, frame: Frame, transcript, view, feedback: str, judged,
               prompt_override: str = "") -> Reply:
        self._view = view
        reply = self._reply(view, feedback, prompt_override)
        turn = Turn(index=len(self.state.turns), frame=frame, transcript_raw=transcript or "",
                    reply=reply)

        for stage, record, decision in self.policy.drain():
            self._record("policy", stage, decision.reason, {
                "action": decision.action, "stage_status": record.status,
                "attempt": record.attempt, "over_budget": record.over_budget,
            })

        if judged is not None:
            reason = f"{judged.source}: {judged.evaluation.verdict}"
            data = {"frame": asdict(frame), "transcript": transcript, "judge": judged.log_data(),
                    "request": judged.request.to_payload(), "reply": asdict(reply)}
        else:
            reason = "no judge call: " + ("start" if transcript is None else
                                          "continue" if frame.expects == "continue" else
                                          "follow-up or roleplay")
            data = {"frame": asdict(frame), "transcript": transcript, "reply": asdict(reply)}
        self._record("turn", frame.stage, reason, data)

        if view.expects in ("done", "held"):
            self._record("lesson_end", view.stage, view.reason or self.lesson.status,
                         {"status": self.lesson.status,
                          "corrections_used": self.state.corrections_used})
        self.state.turns.append(turn)
        return reply
