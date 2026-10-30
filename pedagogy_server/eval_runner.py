"""Eval runner: feed eval/answers.csv through the judge path and report agreement.

Each row is judged by the same TurnJudge the server uses (request, judge, validation,
fallback, guardrails), with a fresh session state per row, at the row's hint level.
The ladder itself is not run: a row is one turn, and verdict agreement doesn't depend
on what the ladder does next.

    python3 -m pedagogy_server.eval_runner                    # stub judge (CI-safe)
    python3 -m pedagogy_server.eval_runner --judge gemini     # real judge, manual, uses quota

With the stub judge the labels and the judge share the content's rules, so the number
checks the plumbing, not judgement quality. Real-judge runs are the measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from guided_practice.config import PROJECT_ROOT
from guided_practice.course import load_course

from .judge import JUDGE_KINDS, JudgeConfigError, make_judge
from .llm import load_dotenv
from .models import VERDICTS, ServerSession
from .turn_judge import TurnJudge

DEFAULT_ANSWERS = PROJECT_ROOT / "eval" / "answers.csv"
REQUIRED_COLUMNS = ("row_id", "course_id", "scenario_id", "item_id", "hint_level",
                    "transcript", "label_verdict", "label_error_id")
# Both verdicts climb the hint ladder, so mixing them up costs the learner nothing.
LADDER_CLASS = {"correct": "correct", "surface_error": "retry",
                "meaning_error": "advance", "off_topic": "advance"}


@dataclass(frozen=True)
class EvalRow:
    row_id: str
    course_id: str
    scenario_id: str
    item_id: str
    hint_level: int
    transcript: str
    label_verdict: str
    label_error_id: str
    label_source: str = ""


@dataclass
class RowResult:
    row_id: str
    item_id: str
    transcript: str
    label_verdict: str
    verdict: str
    label_error_id: str
    error_id: str
    source: str
    rejections: list
    guardrail_actions: list


@dataclass
class EvalReport:
    judge: str
    n: int
    verdict_agreement: float
    ladder_agreement: float
    error_id_agreement: float
    surface_rows: int
    fallback_rate: float
    confusion: dict = field(default_factory=dict)
    rejection_counts: dict = field(default_factory=dict)
    unavailable_reasons: dict = field(default_factory=dict)   # why the judge gave nothing
    model_rows: int = 0                  # rows the model's own verdict decided
    model_verdict_agreement: float = 0.0  # agreement on those rows only
    guardrail_counts: dict = field(default_factory=dict)
    disagreements: list = field(default_factory=list)


def load_rows(path: str | Path) -> list:
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for raw in reader:
            if raw["label_verdict"] not in VERDICTS:
                raise ValueError(f"{path}: row {raw['row_id']} has label '{raw['label_verdict']}'")
            rows.append(EvalRow(
                row_id=raw["row_id"], course_id=raw["course_id"], scenario_id=raw["scenario_id"],
                item_id=raw["item_id"], hint_level=int(raw["hint_level"] or 0),
                transcript=raw["transcript"], label_verdict=raw["label_verdict"],
                label_error_id=raw["label_error_id"] or "", label_source=raw.get("label_source", ""),
            ))
    return rows


def _ratio(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def run_eval(rows, judge_kind: str = "stub", delay: float = 0.0, sleep=time.sleep) -> EvalReport:
    """``delay`` seconds between rows keeps a rate-limited key (e.g. a free tier) from
    turning most rows into fallbacks."""
    judges, scenarios, results = {}, {}, []
    for n, row in enumerate(rows):
        if delay and n:
            sleep(delay)
        if row.course_id not in judges:
            course = load_course(row.course_id)
            judges[row.course_id] = TurnJudge(course, make_judge(judge_kind, course))
        tj = judges[row.course_id]
        key = (row.course_id, row.scenario_id)
        if key not in scenarios:
            scenarios[key] = tj.course.load_scenario(row.scenario_id)
        scenario = scenarios[key]
        item = scenario.item_by_id(row.item_id)
        if item is None:
            raise ValueError(f"row {row.row_id}: unknown item {row.item_id}")
        state = ServerSession(f"eval-{row.row_id}", "eval", row.course_id, row.scenario_id)
        turn = tj.judge_turn(scenario, item, row.hint_level, row.transcript, state)
        results.append(RowResult(
            row_id=row.row_id, item_id=row.item_id, transcript=row.transcript,
            label_verdict=row.label_verdict, verdict=turn.evaluation.verdict,
            label_error_id=row.label_error_id, error_id=turn.evaluation.matched_error_id,
            source=turn.source, rejections=list(turn.rejections),
            guardrail_actions=list(turn.guardrail_actions),
        ))
    return summarise(results, judge_kind)


def summarise(results, judge_kind: str) -> EvalReport:
    n = len(results)
    agree = sum(r.verdict == r.label_verdict for r in results)
    ladder = sum(LADDER_CLASS[r.verdict] == LADDER_CLASS[r.label_verdict] for r in results)
    surface = [r for r in results if r.label_verdict == "surface_error"]
    err_agree = sum(r.verdict == "surface_error" and r.error_id == r.label_error_id for r in surface)
    confusion = defaultdict(Counter)
    for r in results:
        confusion[r.label_verdict][r.verdict] += 1
    rejections = Counter(code.split(":")[0] for r in results for code in r.rejections)
    guards = Counter(a.split(":")[0] for r in results for a in r.guardrail_actions)
    unavailable = Counter(code.split(":", 1)[1] for r in results for code in r.rejections
                          if code.startswith("judge_unavailable:"))
    by_model = [r for r in results if r.source != "fallback"]
    return EvalReport(
        judge=judge_kind, n=n,
        verdict_agreement=_ratio(agree, n),
        ladder_agreement=_ratio(ladder, n),
        error_id_agreement=_ratio(err_agree, len(surface)),
        surface_rows=len(surface),
        fallback_rate=_ratio(sum(r.source == "fallback" for r in results), n),
        confusion={k: dict(v) for k, v in confusion.items()},
        rejection_counts=dict(rejections),
        unavailable_reasons=dict(unavailable),
        model_rows=len(by_model),
        model_verdict_agreement=_ratio(sum(r.verdict == r.label_verdict for r in by_model), len(by_model)),
        guardrail_counts=dict(guards),
        disagreements=[asdict(r) for r in results if r.verdict != r.label_verdict],
    )


def format_report(report: EvalReport) -> str:
    lines = [
        f"judge: {report.judge}   rows: {report.n}",
        f"verdict agreement:  {report.verdict_agreement:.1%}",
        f"ladder agreement:   {report.ladder_agreement:.1%}   (meaning_error = off_topic)",
        f"error_id agreement: {report.error_id_agreement:.1%}   over {report.surface_rows} surface_error rows",
        f"fallback rate:      {report.fallback_rate:.1%}",
        f"model-decided rows: {report.model_rows}, verdict agreement on them: "
        f"{report.model_verdict_agreement:.1%}",
        "confusion (label -> predicted):",
    ]
    for label in VERDICTS:
        if label in report.confusion:
            lines.append(f"  {label:<14} {report.confusion[label]}")
    lines.append(f"rejections: {report.rejection_counts or 'none'}")
    lines.append(f"guardrails: {report.guardrail_counts or 'none'}")
    if report.unavailable_reasons:
        lines.append(f"judge unavailable: {report.unavailable_reasons}")
    for d in report.disagreements[:20]:
        lines.append(f"  x {d['row_id']} {d['item_id']}: {d['transcript']!r} "
                     f"label={d['label_verdict']} got={d['verdict']} via {d['source']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")   # keys only; never printed
    load_dotenv(PROJECT_ROOT.parent / ".env")   # shared key at the repo root
    parser = argparse.ArgumentParser(description="Run eval/answers.csv through the judge path.")
    parser.add_argument("--answers", default=str(DEFAULT_ANSWERS))
    parser.add_argument("--judge", choices=JUDGE_KINDS, default="stub")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows")
    parser.add_argument("--json-out", default=None, help="also write the full report as JSON")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds between rows, for rate-limited keys")
    args = parser.parse_args(argv)

    rows = load_rows(args.answers)[: args.limit]
    try:
        report = run_eval(rows, args.judge, delay=args.delay)
    except JudgeConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
