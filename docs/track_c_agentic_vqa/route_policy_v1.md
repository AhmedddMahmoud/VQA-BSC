# Track C Route Policy v1

## Purpose

This document defines the deterministic routing and prompt policy used by the first Track C prototype. Track C v1 wraps Track B and does not replace it.

## Route Labels

- `general`
- `counting`
- `localization_or_missing_object`
- `ocr_or_text_sensitive`
- `spatial_or_relation_sensitive`

## Deterministic Routing Rules (v1)

Routing is rule-based and text-only from the question string, with fixed precedence:

1. `counting`
2. `ocr_or_text_sensitive`
3. `spatial_or_relation_sensitive`
4. `localization_or_missing_object`
5. fallback `general`

### Rule Intuition

- **counting**: keywords like “how many”, “number of”, “total number”, “count”, “amount/quantity of”, “what time”, “how long”.
- **ocr/text**: keywords like “what does ... say/read”, “written”, “text on”, “label”, “brand name”, “street name”, “license plate”, “logo says”, “jersey number”, “number on sign/jersey/clock”.
- **spatial/relation**: words such as “left”, “right”, “which side”, “behind”, “in front of”, “between”, “under”, “above”, “next to”, “on top of”, “beside”, “touching”, “inside/outside”, “across from”.
- **localization/detail**: patterns like “where is/are/was/were”, “find”, “locate”, “do you see”, “is there”, plus localized detail forms such as “what is in/the background”, “what is the ... doing/wearing/holding/made of”, and “what color/type/kind/object”.

## Route-Specific Prompt Templates

- `general`: `<image> answer {question}`
- `counting`: `<image> answer with a short number only. {question}`
- `localization_or_missing_object`: `<image> inspect the relevant object/details carefully and answer briefly. {question}`
- `ocr_or_text_sensitive`: `<image> if text is visible, read it carefully and answer briefly. {question}`
- `spatial_or_relation_sensitive`: `<image> focus on object positions and relations, then answer briefly. {question}`

## Verifier Policy (v1)

Verifier is optional and enabled only for selected routes in config:

- `counting`
- `spatial_or_relation_sensitive`
- `ocr_or_text_sensitive`

Verifier prompt uses the question + first answer and asks for a short re-checked output.

## Selection Policy (v1)

- Keep first-pass answer if verifier output is empty, same after normalization, or too long.
- Accept verifier answer only when it is short and meaningfully different.

This keeps Track C behavior conservative and deterministic.
