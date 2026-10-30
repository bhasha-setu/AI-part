"""Every lesson in the curriculum must be safe to teach from (the tutor's rules).

Run on its own while writing lessons:  python -m pytest tests/test_curriculum.py
"""

import re
from pathlib import Path

import pytest
import yaml

from cascade.learners import Curriculum
from cascade.tutor_brain import use_tutor_packages

use_tutor_packages()

from guided_practice.content import validate_scenario  # noqa: E402
from guided_practice.course import load_course  # noqa: E402
from guided_practice.normalise import normalise  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "curriculum" / "english_for_hindi"
CURRICULUM = Curriculum(ROOT / "curriculum.yaml")
LANG = load_course(CURRICULUM.course_id).lang
SLOT = re.compile(r"\{(\w+)\}")
LESSON_IDS = [ref.id for ref in CURRICULUM.lessons]


def fill(template: str) -> str:
    return SLOT.sub("placeholderval", template)


@pytest.fixture(scope="module", params=LESSON_IDS)
def lesson(request):
    return request.param, CURRICULUM.load(request.param, LANG)


def test_every_listed_lesson_has_a_file():
    for lid in LESSON_IDS:
        assert CURRICULUM.path(lid).exists(), lid
    on_disk = {p.stem for p in (ROOT / "lessons").glob("*.yaml")}
    assert on_disk == set(LESSON_IDS), f"files not in curriculum.yaml: {on_disk - set(LESSON_IDS)}"


def test_passes_scenario_validation(lesson):
    lid, scn = lesson
    assert scn.id == lid
    assert validate_scenario(scn) == []


def test_correct_answers_are_always_accepted(lesson):
    """the tutor's core guarantee: a correct sentence is never marked wrong. Its rule
    verdict checks the accepted list before the error patterns, so every correct form
    must be matched by that list (then an overlapping error pattern can't fire)."""
    _, scn = lesson
    for item in scn.items:
        for form in [item.model_answer] + list(item.accepted):
            norm = normalise(fill(form), LANG)
            assert item.match_accepted(norm) is not None, f"{item.id}: '{norm}' not accepted"


def test_every_item_is_taught_and_labelled(lesson):
    _, scn = lesson
    taught = {i for p in scn.useful_language for i in p.used_by}
    ids = {it.id for it in scn.items}
    assert ids <= taught, f"items without a useful_language phrase: {ids - taught}"
    assert taught <= ids, f"used_by points at unknown items: {taught - ids}"
    assert all(it.scenario == scn.id for it in scn.items)
    assert 3 <= len(scn.items) <= 5


def test_prompt_and_cue_do_not_give_the_answer_away(lesson):
    _, scn = lesson
    for item in scn.items:
        answer = normalise(SLOT.sub(" ", item.model_answer), LANG)
        first = normalise(item.first_words, LANG)
        for text in (item.prompt, item.cue):
            t = normalise(text, LANG)
            assert answer not in t, f"{item.id}: '{text}' contains the answer"
            if len(first.split()) >= 2:
                assert first not in t, f"{item.id}: '{text}' contains the first words"


def test_files_are_marked_draft():
    for lid in LESSON_IDS:
        head = CURRICULUM.path(lid).read_text(encoding="utf-8")[:400]
        assert "DRAFT" in head, lid


def test_item_ids_are_unique_across_the_curriculum():
    """Spaced review is keyed by item id: two lessons must never share one."""
    seen = {}
    for lid in LESSON_IDS:
        raw = yaml.safe_load(CURRICULUM.path(lid).read_text(encoding="utf-8"))
        for item in raw["scenario"]["items"]:
            assert item["id"] not in seen, f"{item['id']} in {lid} and {seen.get(item['id'])}"
            seen[item["id"]] = lid
