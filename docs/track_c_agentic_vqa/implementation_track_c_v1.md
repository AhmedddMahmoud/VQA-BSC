# Track C v1 Implementation Summary

## Scope Completed

Implemented the first minimal Track C prototype as a lightweight agentic wrapper around existing Track B PaliGemma inference, with failure-case-only paired evaluation.

## Files Implemented

- `src/agents/agentic_vqa.py`
- `src/agents/router.py`
- `src/agents/prompt_policies.py`
- `src/agents/selector.py`
- `src/analysis/eval_agentic_vqa_failures.py`
- `configs/track_c_agentic_vqa_failures.json`
- `docs/track_c_agentic_vqa/route_policy_v1.md`

Additionally, this handover file:

- `docs/track_c_agentic_vqa/implementation_track_c_v1.md`

## What Was Implemented

### 1) Deterministic Router

- Added rule-based route classification from question text.
- Supported labels:
  - `general`
  - `counting`
  - `localization_or_missing_object`
  - `ocr_or_text_sensitive`
  - `spatial_or_relation_sensitive`
- Added route toggles through config.

### 2) Route-Specific Prompt Policies

- Added one prompt template per route.
- Kept `general` prompt close to Track B baseline (`<image> answer {question}`).
- Added focused prompt variants for counting, OCR/text-sensitive, spatial/relation, and localization/detail.

### 3) Minimal Agentic Wrapper

- Input: image + question.
- Router assigns route.
- Route prompt is built and sent to existing Track B runner.
- Optional verifier stage runs only on configured routes.
- Final selector chooses first-pass or verifier answer using conservative deterministic rules.
- Returns final answer plus route trace metadata.

### 4) Failure-Case-Only Evaluation

- Added evaluator that reads existing Track B failure-case JSONL.
- Runs Track C on the same rows only.
- Produces paired comparison between:
  - original Track B answer,
  - Track C routed answer.
- Computes required metrics:
  - exact/normalized match,
  - VQA soft score,
  - failure recovery count,
  - failure recovery rate,
  - average score delta.
- Saves outputs:
  - paired JSONL,
  - metrics JSON,
  - markdown summary.

## Intentionally Left Out of v1

- No RL.
- No external OCR tools.
- No object detectors.
- No retrieval.
- No new models.
- No training.
- No full-dataset rerun.

## Reuse of Existing Components

- Reused `PaliGemmaRunner` (`src/models/vlm/paligemma_runner.py`) as core answerer.
- Reused normalization/soft scoring helpers from `src/utils/metrics.py`.
- Did not modify Track A or Track B base logic.
