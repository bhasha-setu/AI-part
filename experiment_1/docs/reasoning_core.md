# Reasoning core: rules, thresholds and their justification

Status: **decisions recorded; partly built.** The user settled Q1–Q9 (see "Decisions"
at the end). Built: R1 (with skipped stages), R2 (one repeat, not-achieved items only),
the learner-model commit once per lesson (R7). Not built yet:
R3 (drop to hint level 2), the correction cap (C5/C6, Q5), the delayed-recall metric
(C1), and Hindi UI strings (Q8). R4 is off, per Q4. Today's code still uses
`HoldPolicy`, a placeholder that advances completed and skipped stages and holds on
everything else.

Each rule gives its inputs (signals already in the turn log or learner model),
the action, the threshold and where it lives in config, the justification, and
the reason string it will log. Thresholds are read from `config/levels/<CEFR>.yaml`.

Justification labels:
- **research: <citation>** supports the rule. The citation was checked in the
  review papers in `docs/` (see "Citation check" below).
- **framework: §n** comes from `docs/Language Learning Product Pedagogy and Interaction Framework.pdf`.
- **hypothesis** is a design guess for the eval set to test.

Evidence scope: **EFL** means the cited evidence comes from English-as-a-foreign-language
learners. **general** means it spans languages. **none** means no study backs the number.

---

## Signals available

From each scored turn (`turn_log`): `verdict`, `hint_level`, `matched_error_id`,
`surface_errors_before_correct`, `outcome`, `timestamp`, `course_id`.
From each stage (`lesson.StageRecord`): `status`, `attempt`, `elapsed_seconds`, `over_budget`.
From the learner model: the Leitner `box` and `due` per item.

Not available: grit, age, self-report, prior course history across lessons beyond
the Leitner state. See conflict C2.

---

## Rules

### R1. Advance when a stage completes
- **Inputs:** stage `status` in {completed, stubbed, skipped}. A skipped stage (nothing
  due) advances, per Q7. Holding would block a learner with nothing to review.
- **Action:** `advance`.
- **Threshold:** none.
- **Justification:** structural. Framework §4 gives the loop order; a stage that
  finished has nothing left to repeat. Evidence scope: none.
- **Reason:** `"R1: <stage> <status>"`.

### R2. Repeat a failed guided-practice stage
- **Inputs:** guided-practice `status` = failed; `not_achieved` item ids; `attempt`.
- **Action:** `repeat`, re-presenting only the items that were `not_achieved` in the
  last attempt (Q9). Solved items are not re-presented. Repeats don't go back through
  the scheduler, which would see nothing due and skip.
- **Threshold:** `max_stage_attempts` = 2, meaning one repeat (Q3; hypothesis, scope
  none). This is the cap that R3 takes over when reached. A cap of 3 was considered
  and rejected: the third attempt adds little over spaced retrievals (C3) and risks
  frustration.
- **Justification:** a loop guard, so a policy can't repeat forever. It is not a
  learning claim. In-lesson repeats are massed practice, and the evidence favours
  spacing over massing (framework §3: benefits were "larger when spaced"). The
  framework's "4 or more repetitions" is about spaced retrievals of an item across
  sessions, and R7's Leitner boxes already give about that many (see C3).
- **Reason:** `"R2: repeat <stage>, attempt <n> of <cap>, not achieved: <ids>"`.

### R3. Drop to easier content when repeats run out
- **Inputs:** R2's cap reached; the items still `not_achieved`.
- **Action:** `drop_easier`. **Not implemented.** The lesson holds today.
- **Decided (Q2):** restart each failed item at a higher hint level (start at
  level `drop_start_hint_level` = 2, first words shown). This keeps the same item and
  the same meaning and adds support. Swapping in a simpler item is deferred until
  content has difficulty tags. Revisit once the eval set shows hint escalation
  rescues failures.
- **Threshold:** `drop_start_hint_level` = 2 (hypothesis; scope none).
- **Justification:** framework §6 "graduated hints, not one-click rewrites" argues for
  adding support rather than replacing the target. Prompts before direct answers
  raised uptake and self-repair for lower-proficiency learners (framework: §6), which
  the prompts-vs-recasts review also reports (Alzi'abi 2026; Ito 2021; Ammar & Spada).
  Evidence scope: EFL, per the framework's §11 caveat.
- **Reason:** `"R3: drop <ids> to hint level <n>"`.

### R4. Bring an item back early, within the session
- **Inputs:** an item solved at `review_min_level` (2) or above, or with
  `surface_errors_before_correct` ≥ 2 (hypothesis).
- **Decided (Q4): off.** Return is next-day only, per framework §4 step 7. The Leitner
  box handles it (R7). This rule is not built and stays off by default. The
  `early_return_after_items` = 2 value is kept for reference only.
- **Justification:** spaced retrieval supports consolidation (research: Hanson & Brown,
  2019; EFL). **Conflict C4:** framework §4 step 7 says missed items return "over
  following days", so an in-session return goes beyond the framework. The
  next-day return is already handled by the Leitner box (R7).
- **Reason:** `"R4: return <id> early after <n> items (solved at level <l>)"`.

### R5. End the session early for fatigue
- **Inputs:** consecutive non-correct scored turns; guided stage `over_budget`.
- **Action:** `end`, which holds the lesson and logs the reason.
- **Threshold:** `max_consecutive_failed_turns` = 4 (hypothesis; scope none).
  The 180 s budget overrun is recorded but does not cut the stage (decision 13).
- **Justification:** attrition is the binding constraint (research: Sudina et al.,
  2025, on L2 learners in mobile-assisted learning). The fatigue threshold itself
  is an unvalidated guess.
- **Reason:** `"R5: <n> consecutive non-correct turns; ending for fatigue"`.

### R6. Hold after repeated lessons with not-achieved results
- **Inputs:** number of consecutive lessons whose guided stage ended `failed`.
- **Action:** `end`, with a review flag for the teacher.
- **Threshold:** `max_failed_lessons_in_a_row` = 2 (hypothesis; scope none).
- **Justification:** attrition again (Sudina et al., 2025). Repeated failure is the
  kind of frustration that predicts dropout. Murphy (2025) and He & Loewen (2022)
  both argue that streaks and habits support retention through attrition, not memory.
  **Caveat:** the lesson does not persist per-lesson history yet. This rule needs
  a per-lesson log first.
- **Reason:** `"R6: <n> failed lessons in a row; holding for review"`.

### R7. Leitner box update (already built; listed for completeness)
- **Inputs:** `outcome`, `hint_level`.
- **Action:** the box promotes on a level ≤ 1 solve, holds on level 2, and resets to
  box 1 on a level 3 repeat or not-achieved.
- **Thresholds:** `promote_max_level` = 1, `hold_max_level` = 2, intervals
  {1, 2, 4, 7, 15} days (all hypothesis). The direction of spacing is from Hanson &
  Brown (2019), EFL.
- **Reason:** `"R7: box <from> -> <to>, next due in <days> d"`.
- **Decided (Q6), built:** the learner model is committed once, after the lesson's
  guided stage is final, not on each attempt. Each item keeps its latest outcome, so
  an item solved on attempt 1 keeps that outcome. If a lesson is abandoned mid-attempt,
  its pending outcomes are not saved.

### R8. Count a level-3 repeat as recall, not production
- **Inputs:** `outcome` = repeated.
- **Action:** counts as not independent (already built) and is logged with
  `outcome="repeated"`, distinct from "produced".
- **Justification:** framework §4 step 4 (recall). The brief defines this as a
  repeat, not a production. Evidence scope: none.
- **Reason:** `"R8: item <id> repeated at level 3, not independent"`.

---

## Conflicts raised, not fixed

**C1. Model dialogue shown before recall inflates independence.**
The model dialogue in `introducing_yourself` contains the model answers verbatim
("My name is Ravi.", "I am from Dehradun.", "I am a student."). Those lines appear
before recall. A level-0 solve in this course may measure memory of the dialogue,
not recall. The same problem arose earlier with the follow-up question. Framework
§4 puts "learn the key phrases" before "recall", so the framework's own loop causes
this. Independence measures immediate recall after exposure, not retention. Retention
needs the delayed-recall analysis (framework §12, item 5). **Decided (Q1):** keep the
framework's order (dialogue before recall). Report independence as "immediate recall
after exposure", not retention. Add a separate delayed-recall metric (framework §12)
for retention. Not built yet.

**C2. Sudina et al. (2025) names grit and age as attrition predictors. The learner
model has no field for either.** R5 and R6 use what we have. Adding grit or age would
need a consent and privacy decision first, and possibly a self-report instrument.

**C3. Massed in-lesson repeats against the spacing evidence.** Framework §3 says
retrieval benefits "needed about 4 or more repetitions" and were "larger when spaced".
Checking the Leitner arithmetic (R7): a new item solved independently is retrieved on
day 0, then due at +2, +6, +13 and +28 days. That is four spaced retrievals after the
first solve, so the framework's threshold is met across sessions, not violated. The real
tension is the opposite one. R2 (repeat the stage in one lesson) and R4 (return an item
early in the session) are massed practice, and the evidence favours spacing. So both
should stay minimal, and R4 should be used sparingly.

**C4. In-session return goes beyond the framework.** Framework §4 step 7 says
"over following days". R4 is an in-session extension (see R4).

**C5. Surface errors are retried, but the framework says to judge communication, not
perfection.** Framework §3 "Judge on communication, not perfection" is "partly
supported": withholding corrections during speaking protects fluency. Every surface
error in guided practice currently costs a retry. A guardrail that caps corrections per
turn (Step 4) reduces this but doesn't remove it. **Decided (Q5), not built:** keep
corrections in guided recall, since form accuracy is the drill's target. Apply the
C6 cap (2–3 per session) and show a given correction at most once per error type. Skip
corrections only in the roleplay stage, which does not exist yet.

**C6. The framework prescribes limiting feedback to 2 or 3 errors per session**
(framework §6, §4 step 6). Nothing enforces that yet. Step 4's guardrail is where it
goes.

**C7. Only the drill half of the loop exists.** Tomita (2025), Nguyen (2026) and Nawaz
et al. (2025) find that task-based practice builds fluency, and focused drills hold the
accuracy edge. Guided recall is the drill half. The task-based half is the roleplay
stub. So independence measures drill accuracy, not fluency, and a high independence score
does not show the learner can hold a conversation. Until roleplay exists, the lesson can
report drill accuracy and not fluency.

---

## Citation check

The citations in this brief were checked against the PDFs in `docs/`. Each author and
year appears in the review papers' reference lists or in-text citations. **I did not
read the primary papers.** A figure or claim is only as good as the review that
reports it.

| Brief citation | Found in | Notes |
|---|---|---|
| Murphy 2025 | streaks review | "streak conditions … sustained participation over a four-week trial" as the review reports it |
| He & Loewen 2022 | streaks review, prompts review | cited for retention through attrition |
| Hanson & Brown 2019 | streaks review | spaced retrieval for vocabulary |
| Tomita 2025 | task-based review | task-based vs. PPP fluency |
| Nguyen 2026 | task-based review | "Nguyen, T. T. H. (2026)" in the reference list |
| Nawaz et al. 2025 | task-based review | "Nawaz, K., Haq, Z., & Mehmood, S. (2025)" in the reference list |
| Pasha 2026 | LLM-feedback review | LLM feedback accuracy |
| Craven & Fredrick 2026 | LLM-feedback review | overcorrection / hallucination |
| Lin 2024 | LLM-feedback review | LLM feedback |
| Cameron et al. 2005 | competence-rewards review | informational vs. controlling rewards |
| Al-Hoorie 2025 | competence-rewards review | rewards and intrinsic motivation |
| Sudina et al. 2025 | streaks review | title confirmed: "L2 grit and age as predictors of attrition in mobile-assisted language learning" |
| Lyu et al. 2024 | AI-chatbots review | division of labour |

Not checked against the reference lists: the claim attributed to "Hanson & Brown
2019" that learners find spaced retrieval effortful. The streaks review cites Hanson
& Brown only for spaced retrieval supporting durable vocabulary, not for effort.
Treat the effort claim as unverified.

---

## Decisions

Settled by the user. Each one is applied in the rules above or still open in the build
list at the top.

- **Q1. Dialogue before recall (C1).** Keep the framework's order. Report independence
  as "immediate recall after exposure". Add a separate delayed-recall metric
  (framework §12) for retention. *Not built.*
- **Q2. "Easier content" (R3).** Same item at hint level 2. Item swapping waits for
  difficulty tags. Revisit after the eval set. *Not built.*
- **Q3. In-lesson repeats (R2, C3).** Cap at one repeat (2 attempts). The third attempt
  is not allowed. *Built:* `max_stage_attempts` = 2 in `config/levels/A1.yaml`.
- **Q4. In-session return (R4, C4).** Next-day only, per framework §4 step 7. R4 is off.
  The Leitner box does the next-day return. *Not built, by design.*
- **Q5. Surface-error retries (C5).** Keep corrections in guided recall. Cap at 2–3 per
  session (C6). Show each correction at most once per error type. Skip them only in the
  roleplay stage. *Built in the pedagogy server* (`pedagogy_server/guardrails.py`,
  `corrections_per_session` = 3 and `corrections_per_error_id` = 1 in
  `config/judge.yaml`). `Session` used alone, without the server, does not cap.
- **Q6. Double updates on repeat (R7).** Commit the learner model once, after the final
  attempt, and call `finish_session` once at lesson end. *Built:* `commit_outcomes` in
  `guided_practice/app.py`, called from `Lesson._commit_guided`.
- **Q7. Skipped practice.** When nothing is due, advance. The skip reason is logged in
  the stage record. *Built:* `HoldPolicy` advances on `skipped`.
- **Q8. Hindi UI.** Draft the Hindi strings now, marked DRAFT, and gate shipping on
  native review. *Not built.*
- **Q9. Repeat scope (R2).** Re-present only the `not_achieved` items. *Built:*
  `Lesson._start_guided` filters to the open ids.

Test coverage for the built items: `tests/test_lesson.py` (commit once on repeat, skipped
advances, repeat cap) and the golden sample in `samples/sample_lesson.md`, regenerated
after the clock-read change. Elapsed time in the sample shifted from 65 s to 60 s because
the commit no longer reads the clock inside the stage timer. That is a fake-clock
artifact, not a behavior change.
