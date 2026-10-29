"""Config loading, threshold sourcing, and the A1 compatibility values."""

import pytest
import yaml

from guided_practice import config
from guided_practice.config import (
    CONFIG_DIR,
    ConfigError,
    audit_thresholds,
    load_language,
    load_level,
    load_profile,
)
from guided_practice.course import load_course


def _all_config_files():
    return sorted(CONFIG_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("path", _all_config_files(), ids=lambda p: p.relative_to(CONFIG_DIR).as_posix())
def test_every_threshold_has_a_source(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert audit_thresholds(data) == []


def test_audit_rejects_bare_number():
    assert audit_thresholds({"cefr": "A1", "max_items": 5}) != []


def test_audit_rejects_missing_source_and_scope():
    problems = audit_thresholds({"x": {"value": 3}})
    assert any("source" in p for p in problems)
    problems = audit_thresholds({"x": {"value": 3, "source": "hypothesis"}})
    assert any("evidence_scope" in p for p in problems)


def test_audit_accepts_sourced_leaf():
    assert audit_thresholds({"x": {"value": 3, "source": "hypothesis", "evidence_scope": "none"}}) == []


def test_unsupported_tokenisation_is_refused(tmp_path, monkeypatch):
    lang_dir = tmp_path / "languages"
    lang_dir.mkdir()
    (lang_dir / "th.yaml").write_text(
        "language: th\nstatus: DRAFT\ntokenisation: dictionary\ncased: false\n"
        "apostrophe_variants: []\ncontractions: {}\nleading_fillers: []\n"
        "titlecase_slots: []\ntypo_min_token_length:\n  value: 4\n"
        "  source: hypothesis\n  evidence_scope: none\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    load_language.cache_clear()
    try:
        with pytest.raises(ConfigError, match="not supported"):
            load_language("th")
    finally:
        load_language.cache_clear()


def test_a1_values_match_the_previous_hard_coded_values():
    # These were constants in code before this refactor. Changing them is a
    # product decision, so this test makes the change visible.
    level = load_level("A1")
    assert level.max_items_per_session == 5
    assert level.max_retries_per_level == 2
    assert level.independent_max_level == 1
    assert level.leitner_box_intervals_days == {1: 1, 2: 2, 3: 4, 4: 7, 5: 15}
    assert level.stage_budgets_seconds["guided_practice"] == 180


def test_a2_is_marked_placeholder():
    level = load_level("A2")
    assert level.status.startswith("DRAFT")
    assert "PLACEHOLDER" in level.status


def test_language_config_matches_previous_english_rules():
    en = load_language("en")
    assert en.contractions["i'm"] == "i am"
    assert "um" in en.leading_fillers
    assert en.typo_min_token_length == 4
    assert en.cased is True


def test_profiles_reference_existing_configs():
    for course_id in ("english_for_hindi_a1", "hindi_for_english_a1"):
        profile = load_profile(course_id)
        course = load_course(course_id)
        assert course.lang.code == profile.target_language
        assert course.level.cefr == profile.cefr_level
        assert profile.scenarios_dir.is_dir()


def test_profile_course_id_must_match_folder(tmp_path, monkeypatch):
    course_dir = tmp_path / "wrong_folder"
    course_dir.mkdir()
    (course_dir / "profile.yaml").write_text(
        "course_id: some_other_id\nstatus: DRAFT\ntarget_language: en\n"
        "learner_language: hi\ncefr_level: A1\nui_strings: en\nscenarios: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "COURSES_DIR", tmp_path)
    load_profile.cache_clear()
    try:
        with pytest.raises(ConfigError, match="declares course_id"):
            load_profile("wrong_folder")
    finally:
        load_profile.cache_clear()


def test_core_has_no_hard_coded_language_strings():
    """Learner-facing strings and English contractions must live in config, not code."""
    core = [p for p in config.PROJECT_ROOT.joinpath("guided_practice").glob("*.py")
            if p.name != "demo.py"]
    forbidden = ["Correct!", "Let's try", "Say:", "Now say the correct form", "I'm from"]
    hits = []
    for path in core:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in text:
                hits.append(f"{path.name}: {needle}")
    assert hits == []
