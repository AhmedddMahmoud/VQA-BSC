# Track C v2 Selector Policy (Base vs Routed)

## Purpose

This policy defines deterministic selection rules for Track C v2. The selector chooses between:

- `base_answer`: original Track B answer from failure JSONL.
- `routed_answer`: new answer generated with router v2 prompt.

The selector does not use ground truth or VQA scores for selection. Scores are computed after selection for evaluation only.

## General Principles

- Preserve Track B answer by default.
- Accept routed answer only when it matches route-specific validity checks.
- Avoid broad, generic, or placeholder routed answers.

## Route-Specific Rules

### `general`
- Always choose `base_answer`.

### `counting`
- Choose `routed_answer` only if it is numeric or time-like.

Examples accepted:
- `7`, `2`, `8:35`, `7:45`, `50 minutes`, `11 hours`.

Otherwise choose `base_answer`.

### `ocr_or_text_sensitive`
- Choose `routed_answer` only if it is non-empty and not a generic placeholder.
- Reject: `text`, `word`, `sign`, `logo`, `unknown`.
- Reject yes/no answers when the question is not yes/no.

### `localization_or_missing_object`

If the question contains **"where"**:
- Choose `base_answer` when `routed_answer` is generic or broader than `base_answer`.
- Reject routed answers like: `car`, `room`, `area`, `object`, `thing`, `person`, `man`, `woman`, `building`.
- This protects against regressions such as:
  - question: "Where is the bear sitting?"
  - base: `floorboard`
  - routed: `car`

If the question asks **color/detail**:
- Choose routed answer if it is concise and non-generic.

If the question asks **action** ("what is X doing", "wearing", "holding", "made of"):
- Choose routed answer if it is a plausible action verb/phrase and non-generic.

Otherwise:
- Choose routed answer only if non-generic; else base.

### `spatial_or_relation_sensitive`
- Choose routed answer only if it is not generic and not a yes/no response for non-yes/no questions.

## Output Fields (Paired JSONL)

The evaluator writes:

- `base_answer`
- `routed_answer`
- `final_answer`
- `selector_choice` (`base` or `routed`)
- `selector_reason`
- `base_exact`, `routed_exact`, `final_exact`
- `base_vqa_soft_score`, `routed_vqa_soft_score`, `final_vqa_soft_score`
- `routed_delta_vs_base`, `final_delta_vs_base`

## Scope Guardrails

- Deterministic, rule-based only.
- No LLM judge, no external tools, no ground-truth usage for selection.
- No change to router v2 rules or prompt policies.
