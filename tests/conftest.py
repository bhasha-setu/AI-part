import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guided_practice.content import SCENARIOS_DIR, load_scenario  # noqa: E402


@pytest.fixture
def intro():
    return load_scenario(SCENARIOS_DIR / "introducing_yourself.yaml")


@pytest.fixture
def food():
    return load_scenario(SCENARIOS_DIR / "ordering_food.yaml")


@pytest.fixture
def course():
    from guided_practice.course import load_course
    return load_course("english_for_hindi_a1")


@pytest.fixture
def fixed_clock():
    base = datetime(2025, 1, 6, 9, 0, 0, tzinfo=timezone.utc)
    counter = {"n": 0}

    def clock():
        counter["n"] += 1
        return base + timedelta(seconds=counter["n"])

    return clock
