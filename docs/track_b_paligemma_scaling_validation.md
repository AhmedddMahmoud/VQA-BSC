# Track B PaliGemma Scaling Validation

## Purpose

This report checks whether the strong Track B PaliGemma performance remains stable when evaluation is scaled beyond the 1800-sample comparison setting.

## Model

- `google/paligemma-3b-ft-vqav2-224`
- Prompt: `<image> answer {question}`
- Inference-only
- Hardware: Kaggle GPU

## Verification Status

- **Artifact-verified:** `1800`, `5000`, and `10000` runs.

Verified source files used for scaling entries:
- `logs/eval_track_b_paligemma_metrics_val_1800_kaggle_20260421_143715.json`
- `vlm docs/track_b_5000_results/logs/eval_track_b_paligemma_metrics_val_5000_kaggle_20260421_164641.json`
- `vlm docs/track_b_10000_results/logs/eval_track_b_paligemma_metrics_val_10000_kaggle_20260421_171722.json`

## Results

| Evaluation Size | Top1 / Exact | VQA Soft |
|---:|---:|---:|
| 1800 | 0.8556 | 0.9287 |
| 5000 | 0.8474 | 0.9242 |
| 10000 | 0.8446 | 0.9225 |

## 10000-Sample Answer-Type Breakdown

| Answer Type | Count | Exact | VQA Soft |
|---|---:|---:|---:|
| yes/no | 3669 | 0.9572 | 0.9973 |
| number | 1329 | 0.7795 | 0.8799 |
| other | 5002 | 0.7793 | 0.8790 |

## Supplementary Metrics at 10000 Samples

| Metric | Score |
|---|---:|
| BLEU-1 | 0.9782 |
| BLEU-2 | 0.9310 |
| BLEU-4 | 0.8286 |
| ROUGE-L | 0.9776 |
| METEOR | 0.5123 |

## Interpretation

The 10000-sample run indicates that Track B remains highly stable when evaluated on a much larger validation subset. VQA soft accuracy decreases only slightly from `0.9287` at 1800 samples to `0.9225` at 10000 samples.

This supports the conclusion that Track B’s improvement over the classical Track A baseline is not a small-subset artifact.

## Note for Thesis Rigor

This report is now backed by attached metric artifacts for all listed scaling points (`1800`, `5000`, `10000`).

For cleaner repository organization, a later housekeeping step can copy `5000`/`10000` artifacts into top-level `logs/` and `predictions/` folders, but this is not required for validity of the reported values.
