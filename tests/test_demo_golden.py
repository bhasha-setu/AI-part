"""Golden test: the committed sample transcript must match a fresh demo run."""

from pathlib import Path

from guided_practice.demo import render_markdown, run_demo

SAMPLE_MD = Path(__file__).resolve().parent.parent / "samples" / "sample_session.md"


def test_demo_runs_and_summarises(tmp_path):
    lines, summary, judge = run_demo(tmp_path / "s.jsonl")
    assert summary.items_attempted == [
        "intro_name", "intro_origin", "intro_work", "intro_interest",
    ]
    assert summary.independence_score == 0.75
    assert summary.review_items == ["intro_work"]
    # The LLM seam was exercised (meaning_error branch) but never for scoring.
    assert len(judge.calls) >= 1


def test_sample_markdown_is_up_to_date(tmp_path):
    lines, summary, _ = run_demo(tmp_path / "s.jsonl")
    fresh = render_markdown(lines, summary)
    committed = SAMPLE_MD.read_text(encoding="utf-8")
    assert fresh.strip() == committed.strip(), (
        "samples/sample_session.md is stale; regenerate with "
        "`python -m guided_practice.demo`"
    )


def test_lesson_sample_is_up_to_date(tmp_path):
    from guided_practice.demo import render_lesson_markdown, run_lesson_demo

    lines, lesson = run_lesson_demo(tmp_path)
    fresh = render_lesson_markdown(lines, lesson)
    committed = (SAMPLE_MD.parent / "sample_lesson.md").read_text(encoding="utf-8")
    assert fresh.strip() == committed.strip(), (
        "samples/sample_lesson.md is stale; regenerate with `python -m guided_practice.demo`"
    )
