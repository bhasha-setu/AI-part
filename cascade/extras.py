"""Extra practice: a new lesson generated inside the learner's current stage.

Uses pedagogy_server's ScenarioGenerator. It plans 2-3 lesson options from a
request, then builds the chosen one and checks it with its own rules (error patterns
built by the server and never matching an accepted answer, answers not given away in
prompts, length and count limits) before anything is taught; a build that fails twice
is refused. Generated lessons are stamped GENERATED_DRAFT and written to the learner's
folder (runs/learners/<user>/generated/), never into courses/.

The request names the stage's goal and what this learner found hard, so the extra
lesson practises the same kind of language in a new situation.
"""

from __future__ import annotations

import os
from typing import Optional

from .tutor_brain import JSONChain, use_tutor_packages

use_tutor_packages()

from pedagogy_server.generator import (GenerationFailed, ScenarioGenerator,  # noqa: E402
                                       learner_course, load_generator_config)
from pedagogy_server.llm import GeminiModel  # noqa: E402

REQUEST_MAX = 300   # pedagogy_server's generator cuts requests at this length


def gemini_chain(model: str, providers: list, env=None, *, thinking: Optional[str] = None,
                 timeout: float = 60) -> Optional[JSONChain]:
    """One GeminiModel per Gemini key (GEMINI_API_KEY, _2, ...), tried in order."""
    env = os.environ if env is None else env
    backends = []
    for p in providers:
        if p["type"] != "gemini":
            continue
        for i in range(1, 10):
            var = p["key_env"] if i == 1 else f"{p['key_env']}_{i}"
            if env.get(var):
                m = GeminiModel(model, thinking_level=thinking or None, timeout_seconds=timeout,
                                env={"GEMINI_API_KEY": env[var]})
                m.name = f"gemini:{model}[key{i}]"
                backends.append(m)
    return JSONChain(backends) if backends else None


def build_generator(providers: list, env=None) -> ScenarioGenerator:
    cfg = load_generator_config()
    plan = gemini_chain(cfg.plan_model, providers, env, timeout=cfg.plan_timeout_seconds)
    build = gemini_chain(cfg.build_model, providers, env, thinking=cfg.build_thinking_level,
                         timeout=cfg.build_timeout_seconds)
    backup = (gemini_chain(cfg.build_fallback_model, providers, env,
                           timeout=cfg.build_timeout_seconds) if cfg.build_fallback_model else None)
    return ScenarioGenerator(plan, build, cfg, build_fallback_model=backup)


def extra_request(stage: dict, hard: list[str], known_titles: list[str]) -> str:
    """What to ask the planner for, within its length limit."""
    parts = [f"A new speaking lesson for '{stage['name']}': {stage['goal']}"]
    if hard:
        parts.append("Practise again: " + "; ".join(hard))
    if known_titles:
        parts.append("Not these: " + ", ".join(known_titles))
    text = ". ".join(parts)
    return text if len(text) <= REQUEST_MAX else text[: REQUEST_MAX - 1] + "…"


def hard_points(progress, lesson_ids: list[str], scenarios: dict) -> list[str]:
    """Phrases this learner missed and forms from their reviews, newest first, short."""
    points = []
    for lid in lesson_ids:
        for att in reversed(progress.attempts(lid)):
            scn = scenarios.get(lid)
            for item_id in att.get("missed", []):
                item = scn.item_by_id(item_id) if scn else None
                if item and item.model_answer not in points:
                    points.append(item.model_answer)
            for r in att.get("reviewed", []):
                if r.get("fixed") and r["fixed"] not in points:
                    points.append(r["fixed"])
    return points[:4]


def learner_library(course, progress):
    """the tutor's per-learner course view; generated lessons go to the learner's folder."""
    return learner_course(course, progress.dir, progress.username)


__all__ = ["GenerationFailed", "build_generator", "extra_request", "hard_points", "learner_library"]
