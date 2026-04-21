# Experiment Ledger

This ledger consolidates the key evaluation runs referenced in the Track A vs Track B analysis.

## Master Table

| Track | Run Label | Split | Sample Count | Top1 / Exact | VQA Soft | Metrics Artifact | Predictions Artifact |
|---|---|---|---:|---:|---:|---|---|
| Track A (locked classical) | `track_a_reference_1800` | `val` | 1800 | 0.2983 | 0.3715 | documented in `docs/version_1_full_handover_report.md` and `vlm docs/track_b_vs_track_a_final_1800_comparison.md` | n/a (legacy reference) |
| Track B (PaliGemma) | `val_1800_kaggle_20260421_143715` | `val_1800_kaggle` | 1800 | 0.8556 | 0.9287 | `logs/eval_track_b_paligemma_metrics_val_1800_kaggle_20260421_143715.json` | `predictions/eval_track_b_paligemma_predictions_val_1800_kaggle_20260421_143715.jsonl` |
| Track B (PaliGemma) | `val_5000_kaggle_20260421_164641` | `val_5000_kaggle` | 5000 | 0.8474 | 0.9242 | `vlm docs/track_b_5000_results/logs/eval_track_b_paligemma_metrics_val_5000_kaggle_20260421_164641.json` | `vlm docs/track_b_5000_results/predictions/eval_track_b_paligemma_predictions_val_5000_kaggle_20260421_164641.jsonl` |
| Track B (PaliGemma) | `val_10000_kaggle_20260421_171722` | `val_10000_kaggle` | 10000 | 0.8446 | 0.9225 | `vlm docs/track_b_10000_results/logs/eval_track_b_paligemma_metrics_val_10000_kaggle_20260421_171722.json` | `vlm docs/track_b_10000_results/predictions/eval_track_b_paligemma_predictions_val_10000_kaggle_20260421_171722.jsonl` |

## Notes

- Track A values are treated as locked control metrics for comparison.
- Track B values are taken from saved metrics JSON files and cross-referenced in `docs/track_b_paligemma_scaling_validation.md`.
- The 5000/10000 artifacts are currently stored under `vlm docs/` and are valid for reporting.
