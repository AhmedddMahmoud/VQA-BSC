# Version 1 Evaluation Metrics & Outputs — Implementation Report

Date: March 2026  
Scope: **Version 1 only** (classical multi-hop VQA baseline)

---

## 1) What Was Changed

### Files Modified

- `src/utils/metrics.py`
- `src/train/eval_vqa.py`

No architecture changes were made.  
No retrieval/OCR/external knowledge/generative components were added.

---

## 2) Implemented Features

## 2.1 `src/utils/metrics.py`

Added reusable evaluation utilities:

- `compute_top1_accuracy(pred_indices, gold_indices) -> float`
  - Deterministic exact index-match accuracy.
  - Validates matching lengths.

- `compute_vqa_soft_accuracy(pred_answers, all_gold_answers) -> float`
  - Implements VQA-style soft scoring per sample:
    - $\text{score} = \min(\frac{\#\text{matching human answers}}{3}, 1.0)$
  - Averages across samples.
  - Uses normalized string matching.

- `compute_answer_type_breakdown(pred_answers, gold_answers_or_annotations, answer_types=None) -> dict`
  - Returns per-type metrics for:
    - `yes/no`
    - `number`
    - `other`
  - For each type reports:
    - `count`
    - `exact_match_accuracy`
    - `vqa_soft_accuracy`
  - Uses provided `answer_type` if available; otherwise derives minimally.

- `normalize_answer_for_eval(answer: str) -> str`
  - Reuses existing normalization from `src.utils.vocab.normalize_answer`.

Added helper functions:

- `derive_answer_type(...)`
- `exact_match_after_normalization(...)`
- `indices_to_answers(...)`

Compatibility preserved:

- Existing `top1_accuracy(...)` still exists and now delegates to `compute_top1_accuracy`.
- Existing training helpers (`update_accuracy_counters`, `compute_accuracy`, `save_jsonl`, `build_failure_rows`) remain available.

---

## 2.2 `src/train/eval_vqa.py`

Upgraded evaluation behavior to produce full Version 1 reporting.

### Inputs and Loading

- Loads config via `V1Config.from_json(...)`.
- Loads checkpoint from:
  - `--checkpoint` if provided, else
  - default `outputs/checkpoints/best_model.pt`.
- Loads vocab and answer vocab from checkpoint.
- Builds val dataset/dataloader (synthetic or real).
- For real mode, loads annotation map from `config.val_annotations_path`.

### Metrics Computed

Script now computes and prints:

- `sample_count`
- `eval_loss`
- `top1_accuracy`
- `vqa_soft_accuracy`
- answer-type breakdown (count, exact acc, soft acc per type)

### Prediction Outputs Saved

Saved to `outputs/predictions/` with timestamped names:

- `eval_predictions_<split>_<timestamp>.jsonl`
- `qualitative_failures_eval_<split>_<timestamp>.jsonl`

Each prediction row includes:

- `question_id`
- `question_text`
- `predicted_answer`
- `predicted_index`
- `gold_majority_answer`
- `gold_index`
- `all_gold_answers`
- `answer_type`
- `image_path`
- `exact_match_correct`
- `vqa_soft_score`
- `topk_predicted_answers`
- `topk_logits`
- `annotation_majority_answer`

### Metrics Summary Saved

Saved to `outputs/logs/`:

- `eval_metrics_<split>_<timestamp>.json`

Includes:

- checkpoint/config references
- sample counts
- filtered-sample info from dataset stats
- eval loss
- top-1 accuracy
- VQA soft accuracy
- answer-type breakdown
- paths to prediction/failure files

### Qualitative Failures

- Built from non-exact-match predictions.
- Saved as JSONL and truncated by `config.qualitative_failures_to_save`.

### Defensive Error Handling Added

- Clear error if checkpoint missing.
- Clear error if annotation format missing required keys.
- Clear wrapped error if checkpoint state dict mismatches model/config settings (notably `use_precomputed_features`).

---

## 3) Assumptions About Data Format

The implementation assumes VQA-style annotation JSON format:

- top-level `annotations` list
- each annotation has `question_id`
- each annotation usually has `answers` list with `answer` fields
- `answer_type` may exist (`yes/no`, `number`, `other`)

If `answer_type` is absent, it is minimally derived from normalized gold answers:

- `yes/no` if majority is `yes` or `no`
- `number` if numeric token or simple number word (`zero`–`ten`)
- otherwise `other`

If `answers` is missing/empty, fallback is `multiple_choice_answer` or majority-label fallback.

Important evaluation scope note:

- Model predicts over top-K labels.
- Dataset filtering still applies (majority answer must be in top-K).
- Metrics are computed on this filtered subset and explicitly reported via:
  - `dataset_loaded_samples`
  - `dataset_filtered_out_not_in_topk`

---

## 4) Commands to Run

## 4.1 Syntax check

```bash
cd /Users/ahmedmahmoud/Documents/VQA-VERSION1
source .venv/bin/activate
python -m py_compile src/utils/metrics.py src/train/eval_vqa.py
```

## 4.2 Real evaluation run

```bash
cd /Users/ahmedmahmoud/Documents/VQA-VERSION1
source .venv/bin/activate
python -m src.train.eval_vqa --config configs/version1_real.json --checkpoint outputs/checkpoints/best_model.pt --split-name val
```

## 4.3 Synthetic evaluation sanity run

```bash
cd /Users/ahmedmahmoud/Documents/VQA-VERSION1
source .venv/bin/activate
python -m src.train.eval_vqa --config configs/version1_vqa.json --checkpoint outputs/checkpoints/overfit_tiny_batch.pt --synthetic --split-name val
```

---

## 5) Validation Performed During Implementation

- Syntax compile passed:
  - `src/utils/metrics.py`
  - `src/train/eval_vqa.py`

- End-to-end synthetic eval execution completed and produced:
  - printed loss/top1/soft/type breakdown
  - predictions JSONL
  - qualitative failures JSONL
  - metrics summary JSON

Observed synthetic sample output included values like:

- `sample_count=128`
- `top1_accuracy=0.0078`
- `vqa_soft_accuracy=0.0026`

(These values are expectedly low for that synthetic checkpoint/config combination.)

---

## 6) Likely Failure Points to Watch

- Checkpoint/config mismatch (especially `use_precomputed_features`) causes state-dict load mismatch.
- Missing or malformed `val_annotations_path` blocks full VQA soft/type evaluation.
- Nonstandard annotation schema (missing `annotations` or `question_id`) raises explicit errors.
- Image-path layout differences may cause unresolved `image_path` values in rows.
- Metrics reflect **filtered top-K subset**, not all raw validation questions.

---

## 7) Scope Guardrails Honored

This implementation intentionally did **not**:

- change model architecture,
- change training objective,
- add retrieval/OCR/external knowledge,
- introduce BLIP/BERT/CLIP/ViT/transformer upgrades,
- refactor unrelated files.

All changes are evaluation-focused and Version-1-compatible.
