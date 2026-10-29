"""A Course bundles everything the core needs for one (target, learner, level).

    course = load_course("english_for_hindi_a1")
    scenario = course.load_scenario("introducing_yourself")

Code takes a Course and never names a language or level. ``default_course()``
exists only so older call sites keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .config import (
    DEFAULT_COURSE,
    LanguageConfig,
    LevelConfig,
    Profile,
    UIStrings,
    load_language,
    load_level,
    load_profile,
    load_ui,
)
from .content import Scenario, load_scenario


@dataclass(frozen=True)
class Course:
    profile: Profile
    lang: LanguageConfig
    level: LevelConfig
    ui: UIStrings

    @property
    def course_id(self) -> str:
        return self.profile.course_id

    def scenario_ids(self) -> tuple:
        return self.profile.scenarios

    def load_scenario(self, scenario_id: str) -> Scenario:
        if scenario_id not in self.profile.scenarios:
            raise KeyError(f"scenario '{scenario_id}' is not part of course '{self.course_id}'")
        return load_scenario(self.profile.scenarios_dir / f"{scenario_id}.yaml", self.lang)


@lru_cache(maxsize=None)
def load_course(course_id: str = DEFAULT_COURSE) -> Course:
    profile = load_profile(course_id)
    return Course(
        profile=profile,
        lang=load_language(profile.target_language),
        level=load_level(profile.cefr_level),
        ui=load_ui(profile.ui_strings),
    )


def default_course() -> Course:
    return load_course(DEFAULT_COURSE)
