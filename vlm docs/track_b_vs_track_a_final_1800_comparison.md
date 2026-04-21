# Track B vs Track A — Final 1800-Sample Comparison

## Scope

This report summarizes the final comparison between:

- **Track A (locked classical baseline)** reference metrics on 1800 validation samples.
- **Track B (PaliGemma VLM inference path)** evaluation on 1800 validation samples (`val_1800_kaggle`).

The goal is to answer whether Track B beats Track A on the locked primary metric.

---

## Track A Locked Reference (Control)

From the locked baseline documentation:

- `sample_count`: `1800`
- `top1_accuracy`: `0.2983`
- `vqa_soft_accuracy`: `0.3715`

Answer-type breakdown (Track A reference):
- `yes/no`: exact `0.5192`, soft `0.6497`
- `number`: exact `0.2564`, soft `0.3291`
- `other`: exact `0.0764`, soft `0.0887`

---

## Track B Result (Kaggle, 1800)

From user-provided run output:

- `sample_count`: `1800`
- `top1_accuracy`: `0.8556`
- `vqa_soft_accuracy`: `0.9287`

Answer-type breakdown (Track B, 1800):
- `yes/no`: count `722`, exact `0.9543`, soft `0.9935`
- `number`: count `235`, exact `0.7787`, soft `0.8752`
- `other`: count `843`, exact `0.7924`, soft `0.8881`

Saved artifacts (Track B 1800):
- Metrics JSON: `outputs/logs/eval_track_b_paligemma_metrics_val_1800_kaggle_20260421_143715.json`
- Predictions JSONL: `outputs/predictions/eval_track_b_paligemma_predictions_val_1800_kaggle_20260421_143715.jsonl`
- Failures JSONL: `outputs/predictions/eval_track_b_paligemma_failures_val_1800_kaggle_20260421_143715.jsonl`

---

## Direct Metric Comparison (1800 vs 1800)

| Metric | Track A (Locked) | Track B (PaliGemma) | Delta (Track B - Track A) |
|---|---:|---:|---:|
| `top1_accuracy` | 0.2983 | 0.8556 | +0.5573 |
| `vqa_soft_accuracy` | 0.3715 | 0.9287 | +0.5572 |

---

## Required Decision

**Question:** Does Track B beat Track A on VQA soft accuracy?

**Answer:** **Yes.**

- Track B VQA soft: `0.9287`
- Track A VQA soft: `0.3715`
- Improvement: `+0.5572`

---

## Notes

- This comparison is at the same reported evaluation scale (`sample_count=1800`).
- The `SiglipImageProcessor` fast-processor warning is informational and non-blocking.
- No claim is made beyond the evaluated setup and reported artifacts.
