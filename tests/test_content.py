"""Content-integrity tests for the seed scenarios."""

import pytest

from guided_practice.content import SCENARIOS_DIR, load_scenario, validate_scenario


SCENARIO_FILES = sorted(SCENARIOS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_scenario_validates(path):
    scenario = load_scenario(path)
    problems = validate_scenario(scenario)
    assert problems == [], f"{path.name}: {problems}"


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_draft_header(path):
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    assert "DRAFT" in first_line


def test_seed_scenarios_present():
    stems = {p.stem for p in SCENARIO_FILES}
    assert {"introducing_yourself", "ordering_food"} <= stems


def test_each_scenario_has_four_items():
    for path in SCENARIO_FILES:
        assert len(load_scenario(path).items) == 4


def test_all_six_error_categories_present():
    ids = set()
    for path in SCENARIO_FILES:
        for item in load_scenario(path).items:
            ids.update(ce.id for ce in item.common_errors)
    required = {
        "dropped_article",
        "missing_copula",
        "wrong_verb_after_i_am",
        "missing_from_preposition",
        "question_word_order",
        "he_she_agreement",
    }
    assert required <= ids, f"missing: {required - ids}"
