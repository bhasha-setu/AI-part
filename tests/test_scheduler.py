"""Scheduler and learner-model tests."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from guided_practice.learner_model import ItemState, LearnerModel
from guided_practice.scheduler import select_items


@dataclass
class FakeItem:
    id: str


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


NOW = datetime(2025, 1, 10, 9, 0, 0, tzinfo=timezone.utc)


def make_learner(states):
    lm = LearnerModel("l1")
    lm.states = states
    return lm


def test_due_reviews_before_new_items():
    items = [FakeItem(f"i{n}") for n in range(6)]
    # i0, i2 are due; i1 not yet due; i3,i4,i5 are new (no state).
    states = {
        "i0": ItemState("i0", 1, _iso(NOW - timedelta(days=2)), _iso(NOW)),
        "i2": ItemState("i2", 1, _iso(NOW - timedelta(days=1)), _iso(NOW)),
        "i1": ItemState("i1", 2, _iso(NOW + timedelta(days=3)), _iso(NOW)),
    }
    selected = select_items(items, make_learner(states), NOW, cap=5)
    ids = [it.id for it in selected]
    # due (earliest first) then new, i1 excluded (not due)
    assert ids == ["i0", "i2", "i3", "i4", "i5"]


def test_cap_is_respected():
    items = [FakeItem(f"i{n}") for n in range(7)]
    selected = select_items(items, make_learner({}), NOW, cap=5)
    assert len(selected) == 5
    assert [it.id for it in selected] == ["i0", "i1", "i2", "i3", "i4"]


def test_smaller_cap():
    items = [FakeItem(f"i{n}") for n in range(7)]
    states = {
        "i5": ItemState("i5", 1, _iso(NOW - timedelta(days=1)), _iso(NOW)),
    }
    selected = select_items(items, make_learner(states), NOW, cap=3)
    ids = [it.id for it in selected]
    assert len(selected) == 3
    assert ids[0] == "i5"  # the due review comes first even with a tight cap


def test_not_due_items_excluded():
    items = [FakeItem("a"), FakeItem("b")]
    states = {
        "a": ItemState("a", 3, _iso(NOW + timedelta(days=4)), _iso(NOW)),
        "b": ItemState("b", 3, _iso(NOW + timedelta(days=4)), _iso(NOW)),
    }
    selected = select_items(items, make_learner(states), NOW, cap=5)
    assert selected == []


# --- learner model box promotion/reset -------------------------------------

def test_independent_solve_promotes_box():
    lm = LearnerModel("l1")
    st = lm.record_result("x", solved_level=0, achieved=True, now=NOW)
    assert st.box == 2  # 1 -> 2


def test_not_achieved_resets_box():
    lm = LearnerModel("l1")
    lm.record_result("x", solved_level=0, achieved=True, now=NOW)   # box 2
    lm.record_result("x", solved_level=1, achieved=True, now=NOW)   # box 3
    st = lm.record_result("x", solved_level=None, achieved=False, now=NOW)
    assert st.box == 1


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "learners.json"
    lm = LearnerModel("l1")
    lm.record_result("x", solved_level=0, achieved=True, now=NOW)
    lm.save(path)
    reloaded = LearnerModel.load(path, "l1")
    assert reloaded.state_for("x").box == 2
