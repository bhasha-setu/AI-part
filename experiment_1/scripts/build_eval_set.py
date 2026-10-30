"""Build eval/answers.csv from course content. DRAFT seed set.

Every row is a sentence the content already contains, labelled by a mechanical rule.
No label is a human or native-speaker judgement:

    content:model_answer         the item's model answer, slots filled      -> correct
    content:accepted             an accepted form with no slots             -> correct
    content:correction_example   a listed error's corrected form            -> correct
    content:common_error         a listed error's example                   -> surface_error
    rule:empty_transcript        "" (silence or an STT miss)                -> off_topic
    definition:other_item_answer the next item's model answer, kept only
                                 when this item's rules don't accept it     -> off_topic

Accepted forms with slots are not filled: "i am an {job}" with job=student would make
an ungrammatical row the content never reviewed. Slot values are the demo learner's
(guided_practice/demo.py). Replace or extend this set with real, labelled learner
answers before trusting any agreement number.

    python3 scripts/build_eval_set.py          # writes eval/answers.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from guided_practice.course import load_course  # noqa: E402
from guided_practice.normalise import normalise  # noqa: E402

COURSE_ID = "english_for_hindi_a1"
SLOT_SAMPLES = {"name": "Ravi", "place": "Dehradun", "job": "student", "interest": "cricket"}
COLUMNS = ["row_id", "course_id", "scenario_id", "item_id", "hint_level", "transcript",
           "label_verdict", "label_error_id", "label_source"]
OUT = ROOT / "eval" / "answers.csv"


def fill(template: str) -> str:
    for slot, value in SLOT_SAMPLES.items():
        template = template.replace("{" + slot + "}", value)
    return template


def rows_for(course) -> list:
    rows = []
    for scenario_id in course.scenario_ids():
        scenario = course.load_scenario(scenario_id)
        items = scenario.items
        for i, item in enumerate(items):
            seen = set()

            def add(text, verdict, source, error_id=""):
                key = normalise(text, course.lang)
                if key in seen and text:
                    return
                seen.add(key)
                rows.append({"course_id": course.course_id, "scenario_id": scenario_id,
                             "item_id": item.id, "hint_level": 0, "transcript": text,
                             "label_verdict": verdict, "label_error_id": error_id,
                             "label_source": source})

            add(fill(item.model_answer), "correct", "content:model_answer")
            for form in item.accepted:
                if "{" not in form:
                    add(form, "correct", "content:accepted")
            for ce in item.common_errors:
                add(ce.example, "surface_error", "content:common_error", ce.id)
                add(ce.correction_example, "correct", "content:correction_example")
            add("", "off_topic", "rule:empty_transcript")

            other = fill(items[(i + 1) % len(items)].model_answer)
            norm = normalise(other, course.lang)
            if (item.match_accepted(norm) is None
                    and not any(ce.match(norm) for ce in item.common_errors)):
                add(other, "off_topic", "definition:other_item_answer")
    for n, row in enumerate(rows, start=1):
        row["row_id"] = f"r{n:03d}"
    return rows


def main() -> int:
    rows = rows_for(load_course(COURSE_ID))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
