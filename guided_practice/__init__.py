"""Guided-practice core for a language-learning app.

The core is course-agnostic. A course is a profile (target language, learner
language, CEFR level, scenarios) plus per-language and per-level config. Code
takes a Course and never names a language or level.

Main entry points:
    guided_practice.course.load_course          -- profile + configs -> Course
    guided_practice.session.Session             -- turn-driven recall session
    guided_practice.lesson.Lesson               -- six-stage lesson loop
    guided_practice.policy.Decision / Policy    -- what happens after a stage
    guided_practice.summary.summarise           -- end-of-session summary from logs
"""

__all__ = [
    "app",
    "assessment",
    "config",
    "content",
    "course",
    "evaluator",
    "learner_model",
    "lesson",
    "matching",
    "normalise",
    "policy",
    "roleplay",
    "scheduler",
    "session",
    "summary",
    "turn_log",
]
