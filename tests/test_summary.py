"""Summary / independence-score computation from JSON Lines logs."""

from guided_practice.summary import (
    independence_score,
    items_attempted,
    review_items,
    solved_level_per_item,
    summarise,
)


def rec(item_id, level, verdict, turn_type="scored", sid="s1"):
    return {
        "session_id": sid,
        "item_id": item_id,
        "hint_level": level,
        "verdict": verdict,
        "turn_type": turn_type,
    }


def test_independence_counts_level_0_and_1():
    records = [
        rec("a", 0, "correct"),
        rec("b", 1, "correct"),
        rec("c", 2, "correct"),
        rec("d", 3, "correct"),
    ]
    # a, b independent; c (L2) and d (L3 repeat) not.
    assert independence_score(records) == 0.5


def test_follow_up_turns_excluded():
    records = [
        rec("a", 0, "correct"),
        rec("a", 0, "follow_up", turn_type="follow_up"),
    ]
    assert items_attempted(records) == ["a"]
    assert independence_score(records) == 1.0


def test_not_achieved_stays_in_denominator():
    records = [
        rec("a", 0, "correct"),
        rec("b", 0, "meaning_error"),
        rec("b", 1, "meaning_error"),
        rec("b", 2, "meaning_error"),
        rec("b", 3, "meaning_error"),
    ]
    levels = solved_level_per_item(records)
    assert levels == {"a": 0, "b": None}
    assert independence_score(records) == 0.5   # a solved, b not achieved
    assert review_items(records) == ["b"]


def test_first_correct_level_is_recorded():
    # Surface-error retries precede the correct answer at the same level.
    records = [
        rec("a", 0, "surface_error"),
        rec("a", 0, "correct"),
    ]
    assert solved_level_per_item(records) == {"a": 0}


def test_review_items_include_level_2_and_3():
    records = [
        rec("a", 0, "correct"),
        rec("b", 2, "correct"),
        rec("c", 3, "correct"),
    ]
    assert review_items(records) == ["b", "c"]


def test_empty_log():
    assert independence_score([]) == 0.0
    assert summarise([]).items_attempted == []


def test_session_id_filter():
    records = [
        rec("a", 0, "correct", sid="s1"),
        rec("x", 2, "correct", sid="s2"),
    ]
    s = summarise(records, session_id="s1")
    assert s.items_attempted == ["a"]
    assert s.independence_score == 1.0
