"""Learner profiles, curriculum progress and the tutor menu. Offline."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cascade.learners import (COMPLETE, MOVED_ON, Curriculum, LearnerProgress, safe_username,
                              write_json_atomic)
from cascade.tutor_cli import TutorApp
from cascade.tutor_brain import TutorFactory

LESSONS = Path(__file__).resolve().parent.parent / "curriculum" / "english_for_hindi" / "lessons"


@pytest.fixture
def curriculum(tmp_path):
    """Stage 1: introducing_yourself + a lesson with no file yet; stage 2: ordering_food."""
    root = tmp_path / "cur"
    (root / "lessons").mkdir(parents=True)
    for lid in ("introducing_yourself", "ordering_food"):
        shutil.copy(LESSONS / f"{lid}.yaml", root / "lessons")
    (root / "curriculum.yaml").write_text(yaml.safe_dump({
        "course": "english_for_hindi_a1",
        "stages": [{"name": "Basics", "goal": "g1", "lessons": ["introducing_yourself", "unwritten"]},
                   {"name": "Everyday", "goal": "g2", "lessons": ["ordering_food"]}]}))
    return Curriculum(root / "curriculum.yaml")


def summary(achieved):
    items = [{"item_id": f"i{n}", "achieved": a, "solved_level": 0, "used_hindi": False}
             for n, a in enumerate(achieved)]
    return SimpleNamespace(items=items, from_memory=0.5, hindi_turns=0,
                           reviewed=[{"said": "x", "fixed": "y", "type": "article"}])


def test_usernames_become_safe_folder_names():
    assert safe_username("  Himanshu P! ") == "himanshu-p"
    assert safe_username("../../etc") == "etc"
    with pytest.raises(ValueError):
        safe_username("!!!")


def test_curriculum_numbers_stages_and_lessons(curriculum):
    ref = curriculum.ref("ordering_food")
    assert (ref.stage, ref.index, ref.stage_size) == (2, 1, 1)
    assert ref.label == "Stage 2 · Everyday · lesson 1 of 1"


def test_complete_needs_every_phrase(curriculum, tmp_path):
    p = LearnerProgress(tmp_path, "ravi")
    assert p.is_new and p.next_lesson(curriculum).id == "introducing_yourself"
    attempt = p.record("introducing_yourself", summary([True, True, False, True]))
    assert not p.is_new
    assert not attempt["all_achieved"] and attempt["missed"] == ["i2"]
    assert p.status("introducing_yourself") == "in_progress"
    p.record("introducing_yourself", summary([True] * 4))
    assert p.status("introducing_yourself") == COMPLETE
    # a later weaker redo never undoes "complete"
    p.record("introducing_yourself", summary([False] * 4))
    assert p.status("introducing_yourself") == COMPLETE
    assert len(p.attempts("introducing_yourself")) == 3


def test_move_on_counts_as_done_and_progress_survives_reload(curriculum, tmp_path):
    p = LearnerProgress(tmp_path, "Ravi")
    p.record("introducing_yourself", summary([True, False]))
    p.move_on("introducing_yourself")
    again = LearnerProgress(tmp_path, "ravi")
    assert not again.is_new and again.status("introducing_yourself") == MOVED_ON
    assert again.next_lesson(curriculum).id == "unwritten"


def test_atomic_write_leaves_no_temp_files(tmp_path):
    write_json_atomic(tmp_path / "a" / "x.json", {"k": "v"})
    assert json.loads((tmp_path / "a" / "x.json").read_text()) == {"k": "v"}
    assert [f.name for f in (tmp_path / "a").iterdir()] == ["x.json"]


# --- the menu ----------------------------------------------------------------------------

def app_with(curriculum, tmp_path, answers):
    factory = TutorFactory({"overrides": {"hindi_script": "devanagari"}}, {"providers": []}, offline=True)
    progress = LearnerProgress(tmp_path / "learners", "ravi")
    out = []
    it = iter(answers)
    return TutorApp(curriculum, factory, progress, ask=lambda q: next(it), out=out.append), out


def test_menu_continue_skips_lessons_without_files(curriculum, tmp_path):
    app, out = app_with(curriculum, tmp_path, [""])
    app.progress.record("introducing_yourself", summary([True]))
    assert app.next_lesson().id == "ordering_food"     # "unwritten" has no file yet
    assert app.menu() == "ordering_food"


def test_menu_redo_and_choose_stage(curriculum, tmp_path):
    app, out = app_with(curriculum, tmp_path, ["r", "1", "s", "2", "q"])
    app.progress.record("introducing_yourself", summary([True, False]))
    assert app.menu() == "introducing_yourself"          # redo: started lessons are listed
    assert app.menu() == "ordering_food"                 # stage 2's first open lesson
    assert app.menu() is None


def test_show_progress_keeps_two_progressions_apart(curriculum, tmp_path):
    app, out = app_with(curriculum, tmp_path, [])
    app.show_progress()
    text = "\n".join(out)
    assert "Hindi help:" in text and "Stage 1 · Basics (0/2 done)" in text
    assert "support stage" not in text
    assert "<- next" in text and "not written yet" in text


class FinishedBrain:
    def __init__(self, achieved):
        self.summary = summary(achieved)

    def finish(self):
        return ["report"]


def test_after_lesson_choices(curriculum, tmp_path):
    app, out = app_with(curriculum, tmp_path, ["m", "r"])
    assert app.after_lesson("introducing_yourself", FinishedBrain([True, False])) is None
    assert app.progress.status("introducing_yourself") == MOVED_ON
    assert app.after_lesson("ordering_food", FinishedBrain([False])) == "ordering_food"   # redo now
    assert app.after_lesson("ordering_food", FinishedBrain([True])) is None              # complete
    assert app.progress.status("ordering_food") == COMPLETE


# --- extra practice (stand-in generator; the real one is the tutor's) ----------------

class FakeGenerator:
    def __init__(self):
        self.requests = []

    def plan(self, request, library):
        self.requests.append(request)
        opt = SimpleNamespace(title="At the bank", can_do="Open an account.", scenario_id=None)
        existing = SimpleNamespace(title="Ordering food", can_do="x", scenario_id="ordering_food")
        return SimpleNamespace(options=[existing, opt])

    def build(self, option, library, request):
        text = (LESSONS / "introducing_yourself.yaml").read_text().replace(
            "id: introducing_yourself", "id: gen_at_the_bank_abc123")
        library.generated_dir.mkdir(parents=True, exist_ok=True)
        (library.generated_dir / "gen_at_the_bank_abc123.yaml").write_text(text)
        return SimpleNamespace(scenario_id="gen_at_the_bank_abc123")


def test_extra_practice_is_planned_from_the_stage_and_past_mistakes(curriculum, tmp_path):
    app, out = app_with(curriculum, tmp_path, ["x", "1", "1"])
    app.generator = FakeGenerator()
    app.progress.record("introducing_yourself", summary([True]))   # reviewed: "y"
    key = app.menu()
    assert key == "extra:gen_at_the_bank_abc123"
    req = app.generator.requests[0]
    assert "Basics" in req and "g1" in req and "Practise again: y" in req
    assert "Not these: Introducing yourself" in req
    assert "Ordering food" not in "\n".join(out)          # existing lessons aren't offered
    brain = app.start_lesson(key)
    assert brain.scenario.id == "gen_at_the_bank_abc123"
    assert app.after_lesson(key, FinishedBrain([True, False])) is None   # no redo prompt
    assert app.progress.status("introducing_yourself") == COMPLETE        # curriculum untouched
    assert app.progress.attempts(key)


def test_extra_practice_offline_is_explained(curriculum, tmp_path):
    app, out = app_with(curriculum, tmp_path, ["x", "q"])
    assert app.menu() is None and "isn't available offline" in "\n".join(out)
