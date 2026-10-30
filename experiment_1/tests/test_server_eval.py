"""Step 8: the eval set and the eval runner (stub judge only; real runs are manual)."""

import csv
import importlib.util

import pytest

from guided_practice.config import PROJECT_ROOT
from pedagogy_server.eval_runner import (
    DEFAULT_ANSWERS,
    RowResult,
    format_report,
    load_rows,
    main,
    run_eval,
    summarise,
)


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_eval_set", PROJECT_ROOT / "scripts" / "build_eval_set.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_eval_set_matches_the_generator():
    builder = _builder()
    from guided_practice.course import load_course
    fresh = builder.rows_for(load_course(builder.COURSE_ID))
    with DEFAULT_ANSWERS.open(encoding="utf-8", newline="") as fh:
        committed = list(csv.DictReader(fh))
    assert [{k: str(v) for k, v in r.items()} for r in fresh] == committed, \
        "eval/answers.csv is stale; regenerate with `python3 scripts/build_eval_set.py`"


def test_every_row_has_a_provenance():
    for row in load_rows(DEFAULT_ANSWERS):
        assert row.label_source.split(":")[0] in ("content", "rule", "definition")


def test_stub_eval_runs_the_whole_set():
    rows = load_rows(DEFAULT_ANSWERS)
    report = run_eval(rows, "stub")
    assert report.n == len(rows)
    assert report.fallback_rate == 0.0
    assert report.error_id_agreement == 1.0
    assert report.ladder_agreement == 1.0
    # The rules can't tell off_topic from meaning_error (README decision 5).
    assert {d["label_verdict"] for d in report.disagreements} <= {"off_topic"}
    assert "verdict agreement" in format_report(report)


def test_summary_counts():
    def r(label, got, source="judge", rej=()):
        return RowResult("r", "i", "t", label, got, "", "", source, list(rej), [])

    report = summarise([r("correct", "correct"), r("off_topic", "meaning_error"),
                        r("correct", "meaning_error", "fallback", ["malformed_json"])], "x")
    assert report.verdict_agreement == round(1 / 3, 4)
    assert report.ladder_agreement == round(2 / 3, 4)
    assert report.fallback_rate == round(1 / 3, 4)
    assert report.rejection_counts == {"malformed_json": 1}


def test_bad_label_is_refused(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("row_id,course_id,scenario_id,item_id,hint_level,transcript,label_verdict,"
                    "label_error_id\nr1,c,s,i,0,x,great,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="great"):
        load_rows(path)


def test_real_judge_eval_refuses_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("pedagogy_server.eval_runner.load_dotenv", lambda *a, **k: [])
    assert main(["--judge", "gemini", "--limit", "1"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
