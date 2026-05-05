# Track C v3 — Multi-Agent Prompt Framework (Lightweight)

## Purpose

Track C v3 upgrades Track C into a lightweight **multi-agent prompt framework** while keeping Track B as the core model. It reuses the existing PaliGemma runner, router v2, and selector v2, but structures inference as explicit agents with trace logging.

This is **not** a new model and does not add external tools, RL, retrieval, or full-dataset evaluation.

## Architecture Overview

### 1) Base Answer Agent

- Uses the original Track B answer from the failure JSONL when available.
- If no base answer is present, it generates one using the normal Track B prompt.

### 2) Router Agent

- Uses **router v2** without changes.
- Outputs one of the existing route labels:
  - `general`
  - `counting`
  - `localization_or_missing_object`
  - `ocr_or_text_sensitive`
  - `spatial_or_relation_sensitive`

### 3) Specialist Agents

Each specialist agent builds a route-specific prompt and calls the same PaliGemma runner:

- **GeneralAgent** → `general`
- **NumericAgent** → `counting`
- **DetailAgent** → `localization_or_missing_object`
- **RelationAgent** → `spatial_or_relation_sensitive`
- **TextAgent** → `ocr_or_text_sensitive`

Each specialist returns:

- `specialist_answer`
- `specialist_agent`
- `specialist_prompt`

### 4) Selector / Risk Agent

- Reuses **selector v2** rules.
- Compares **base** vs **specialist** answer.
- Chooses final answer conservatively to avoid regressions.
- Logs `selector_choice` and `selector_reason`.

## Trace Logging (Paired JSONL)

Each output row records:

- `question_id`, `question_text`
- `route`
- `specialist_agent`
- `base_answer`
- `specialist_answer`
- `final_answer`
- `selector_choice`, `selector_reason`
- `base_vqa_soft_score`
- `specialist_vqa_soft_score`
- `final_vqa_soft_score`
- `final_delta_vs_base`

## Evaluation Scope

- Failure-case only (no full validation run).
- Typical runs:
  - 10-case smoke
  - 50-case comparison
  - 100-case final failure-set evaluation

## Rationale

This design makes the agentic behavior **auditable** and **modular** while preserving the strong Track B backbone. It also ensures that weak specialist suggestions do not override strong base answers.
