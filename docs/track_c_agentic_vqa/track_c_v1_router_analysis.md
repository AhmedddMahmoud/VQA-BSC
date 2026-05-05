# Track C v1 Router Analysis (50-Case Verifier-OFF)

## Purpose

This report analyzes the current 50-case Track C v1 paired failure-case run to identify routing weaknesses and define conservative router-only improvements.

The scope is intentionally limited to deterministic routing behavior (no Track A changes, no Track B core logic changes, no verifier/selector redesign).

## Source Artifacts Used

- Paired JSONL:
  - `outputs/predictions/track_c_agentic_paired_failures_track_c_fast50_verifier_off_20260426_172422.jsonl`
- Metrics JSON:
  - `outputs/logs/track_c_agentic_metrics_failures_track_c_fast50_verifier_off_20260426_172422.json`
- Summary markdown:
  - `outputs/logs/track_c_agentic_summary_failures_track_c_fast50_verifier_off_20260426_172422.md`
- Router implementation reviewed:
  - `src/agents/router.py`
- Route policy document reviewed:
  - `docs/track_c_agentic_vqa/route_policy_v1.md`

## Summary of Current 50-Case Results

- Rows loaded: `100`
- Rows evaluated: `50`
- Track B exact: `0.0000`
- Track C exact: `0.0400`
- Average exact delta: `+0.0400`
- Track B soft: `0.5333`
- Track C soft: `0.5400`
- Average soft delta: `+0.0067`
- Recovery count: `2`
- Recovery rate: `0.0400`
- Improved / Tied / Worsened: `3 / 46 / 1`

Route distribution:

```json
{
  "spatial_or_relation_sensitive": 4,
  "localization_or_missing_object": 13,
  "general": 29,
  "ocr_or_text_sensitive": 1,
  "counting": 3
}
```

## Improved Cases Analysis

Three cases improved by VQA soft score (all `+0.3333`):

1. `question_id=393277005` (route=`counting`)
   - Question: "What time is it on the clock?"
   - Track B: `7:35` (soft `0.0`)
   - Track C: `7:45` (soft `0.3333`)
   - Majority: `8:35`
   - Interpretation: improvement is partial and plausible for numeric-sensitive route, but not exact recovery.

2. `question_id=393277009` (route=`localization_or_missing_object`)
   - Question: "What color is the building the background?"
   - Track B: `white` (soft `0.6667`)
   - Track C: `gray` (soft `1.0`, exact recovered)
   - Majority: `gray`
   - Interpretation: real routing benefit on localized attribute query.

3. `question_id=393282008` (route=`spatial_or_relation_sensitive`)
   - Question: "What is in front of the giraffe?"
   - Track B: `grass` (soft `0.0`)
   - Track C: `camera` (soft `0.3333`)
   - Majority: `dirt`
   - Interpretation: partial gain only; still not correct, but route-specific prompting nudged output toward a listed annotator answer.

Recovered exact failures by route:

```json
{
  "localization_or_missing_object": 2
}
```

Note: one exact recovery (`question_id=262161016`, "What kind of a show does this look like it is despite the motorcycle?") improved exact from `car` to `car show`, while soft stayed `1.0` in both. This supports exact formatting gains from route prompting.

## Worsened Case Analysis

One case worsened:

- `question_id=131159000` (route=`localization_or_missing_object`)
- Question: "Where is the bear sitting?"
- Track B: `floorboard` (soft `0.6667`)
- Track C: `car` (soft `0.0`)
- Majority: `car floor`

Interpretation:
- This looks more like a **prompt-side miss on a localization question** than a route-class mismatch.
- The route itself (`localization_or_missing_object`) is reasonable for this question.
- Risk: localization prompts can sometimes over-generalize object category ("car") and lose fine-grained relation/location detail ("car floor").

## Route Distribution Analysis

Observed route usage is strongly skewed to `general`:

- `general`: 29/50 (`58%`)
- Non-general total: 21/50 (`42%`)

This suggests the current regex rules are too narrow for many failure questions that are still deterministic-route candidates.

## Why `general` Is Overused

From the 50-case paired rows (questions routed to `general`), several question forms are currently under-captured:

- `where was ...` form (not covered by current `where is|where are` regex), e.g.:
  - `question_id=262175001`: "Where was the picture taken of the man?"
- localized detail/action questions lacking current keywords, e.g.:
  - `question_id=393274004`: "What is the background metal structure?"
  - `question_id=74002`: "What is the dog doing?"
  - `question_id=393284008`: "What is the person wearing?"
- potential text/label-like forms not captured by current OCR regex, e.g.:
  - `question_id=393277003`: "What year is the car?"

Current issue is therefore mostly **coverage gaps** in deterministic keyword patterns, not architecture-level failure.

## Proposed Router Improvements

Conservative improvements (same routes, deterministic, explainable):

1. **Counting route**
   - Add `total number`, `how many ... are there`, `amount of`, `quantity of`.
   - Keep counting before other routes.

2. **OCR/text-sensitive route**
   - Extend with explicit text-reading cues: `label`, `brand name`, `street name`, `what does .* read`, `logo says`, `text on`.
   - Keep OCR before spatial/localization to catch explicit reading tasks.

3. **Spatial/relation route**
   - Extend with `beside`, `alongside`, `on top of`, `touching`, `inside`, `outside`, `toward`, `facing`, `across from`, `adjacent`.

4. **Localization route**
   - Extend with missed forms: `where was`, `where were`.
   - Add common localized detail forms: `what is in the background`, `what is the background`, `what is the .* doing`, `what is the .* wearing`, `what is the .* holding`, `what is the .* made of`.

5. **Priority and determinism**
   - Keep fixed priority order and route toggles unchanged.
   - Avoid broad catch-all rules that would collapse routing into localization.

## Risk Notes

- Over-expanding localization can absorb too many borderline questions and reduce route precision.
- OCR expansion must avoid swallowing generic numeric questions.
- Spatial expansion should stay relation-specific to avoid false positives in generic object queries.
- Changes should be validated on the same 50-case benchmark first (verifier OFF) before any further scope.

## Next Validation Plan

1. Rerun the same benchmark setup:
   - `50` cases
   - verifier OFF
   - same input artifact and split style
2. Compare against baseline run `track_c_fast50_verifier_off_20260426_172422` on:
   - improved/tied/worsened counts
   - average soft delta
   - route distribution (especially `general` usage)
3. Accept router update only if it keeps regressions controlled and shows cleaner hard-case routing.

---

## Router v2 Validation Result (50-case, Verifier OFF)

New run artifacts:

- Paired JSONL:
  - `outputs/predictions/track_c_agentic_paired_failures_track_c_fast50_verifier_off_router_v2_20260502_005550.jsonl`
- Metrics JSON:
  - `outputs/logs/track_c_agentic_metrics_failures_track_c_fast50_verifier_off_router_v2_20260502_005550.json`
- Summary markdown:
  - `outputs/logs/track_c_agentic_summary_failures_track_c_fast50_verifier_off_router_v2_20260502_005550.md`

Headline metrics (router v2):

- Track B soft: `0.5333`
- Track C soft: `0.5600`
- Avg soft delta: `+0.0267`
- Improved/Tied/Worsened: `4 / 45 / 1`

Route distribution (router v2):

```json
{
  "spatial_or_relation_sensitive": 5,
  "localization_or_missing_object": 19,
  "general": 22,
  "ocr_or_text_sensitive": 1,
  "counting": 3
}
```

Comparison vs router v1 (same 50-case setup):

- `general`: `29 → 22`
- Avg soft delta: `+0.0067 → +0.0267`
- Improved/Tied/Worsened: `3/46/1 → 4/45/1`

Conclusion:
The conservative router expansion reduced `general` overuse and improved soft gains without increasing regressions. This supports keeping the router v2 update as the default heuristic set for Track C v1.

---

## Selector v2 Validation Result (Router v2 + Selector v2)

New run artifacts:

- Paired JSONL:
  - `outputs/predictions/track_c_agentic_paired_failures_track_c_fast50_verifier_off_router_v2_selector_v2_20260503_152938.jsonl`
- Metrics JSON:
  - `outputs/logs/track_c_agentic_metrics_failures_track_c_fast50_verifier_off_router_v2_selector_v2_20260503_152938.json`
- Summary markdown:
  - `outputs/logs/track_c_agentic_summary_failures_track_c_fast50_verifier_off_router_v2_selector_v2_20260503_152938.md`

Headline metrics (router v2 + selector v2):

- Track B soft: `0.5333`
- Track C soft: `0.5733`
- Avg soft delta: `+0.0400`
- Improved/Tied/Worsened: `4 / 46 / 0`

Route distribution remains:

```json
{
  "spatial_or_relation_sensitive": 5,
  "localization_or_missing_object": 19,
  "general": 22,
  "ocr_or_text_sensitive": 1,
  "counting": 3
}
```

Comparison vs router v2 without selector:

- Avg soft delta: `+0.0267 → +0.0400`
- Worsened: `1 → 0`

Conclusion:
The selector v2 layer improved soft gains and eliminated the single regression without changing route distribution. This supports keeping selector v2 for Track C v2 candidate selection.
