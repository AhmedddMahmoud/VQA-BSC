# Version 1 Larger Run (MacBook Air M4) — Training + Evaluation Record

## Scope and Intent
This document records the completed larger benchmark-oriented Version 1 run and its evaluation.

Scope remains Version 1 only:
- frozen ResNet-50 image encoder
- GRU question encoder
- 2-hop attention
- top-1000 answer classification
- existing train/eval pipeline

No architecture changes were introduced.

---

## Run Configuration
Training config used:
- `configs/version1_real_larger_run_m4.json`

Key values:
- `device`: `cpu`
- `train_subset_size`: `9000`
- `val_subset_size`: `1800`
- `batch_size`: `16`
- `epochs`: `4`
- `learning_rate`: `0.001`
- `freeze_image_encoder`: `true`
- `use_precomputed_features`: `false`

Output root:
- `outputs/final_runs/version1_m4_larger_run/`

---

## Training Command
```zsh
cd /Users/ahmedmahmoud/Documents/VQA-VERSION1
.venv/bin/python -m src.train.train_vqa --config configs/version1_real_larger_run_m4.json
```

---

## Training Outcome
Key terminal outputs:
- `best_checkpoint_updated epoch=3 val_soft=0.3715 path=outputs/final_runs/version1_m4_larger_run/checkpoints/best.pt`
- `epoch=4 train_loss=3.2535 val_loss=3.0477 val_top1=0.2711 val_soft=0.3552`
- `best_epoch=3`
- `best_val_vqa_soft=0.3715`

Checkpoints and logs:
- `latest_checkpoint=outputs/final_runs/version1_m4_larger_run/checkpoints/latest.pt`
- `best_checkpoint=outputs/final_runs/version1_m4_larger_run/checkpoints/best.pt`
- `history_json=outputs/final_runs/version1_m4_larger_run/logs/training_history.json`
- `history_csv=outputs/final_runs/version1_m4_larger_run/logs/training_history.csv`
- `final_eval_predictions=outputs/final_runs/version1_m4_larger_run/predictions/final_epoch_predictions_val_final_20260404_163133.jsonl`
- `final_eval_failures=outputs/final_runs/version1_m4_larger_run/predictions/final_epoch_failures_val_final_20260404_163133.jsonl`

---

## Best Checkpoint Evaluation
Evaluation command used:

```zsh
cd /Users/ahmedmahmoud/Documents/VQA-VERSION1
.venv/bin/python -m src.train.eval_vqa \
  --config configs/version1_real_larger_run_m4.json \
  --checkpoint outputs/final_runs/version1_m4_larger_run/checkpoints/best.pt \
  --split-name val_best_larger_run
```

Primary metrics:
- `sample_count=1800`
- `eval_loss=3.2182`
- `top1_accuracy=0.2983`
- `vqa_soft_accuracy=0.3715`

Answer-type breakdown:
- `yes/no`: `count=807`, `exact=0.5192`, `soft=0.6497`
- `number`: `count=234`, `exact=0.2564`, `soft=0.3291`
- `other`: `count=759`, `exact=0.0764`, `soft=0.0887`

Supplementary text-overlap metrics:
- `bleu_1=0.4310`
- `bleu_2=0.0000` (rounded display)
- `bleu_4=0.0000` (rounded display)
- `rouge_l=0.4262`
- `meteor=0.2119`

Primary benchmark note:
- `vqa_soft_accuracy` remains the primary benchmark metric.

---

## Evaluation Artifacts Saved
- Metrics summary JSON:
  - `outputs/final_runs/version1_m4_larger_run/logs/eval_metrics_val_best_larger_run_20260405_160653.json`
- Predictions JSONL:
  - `outputs/final_runs/version1_m4_larger_run/predictions/eval_predictions_val_best_larger_run_20260405_160653.jsonl`
- Qualitative failures JSONL:
  - `outputs/final_runs/version1_m4_larger_run/predictions/eval_failures_val_best_larger_run_20260405_160653.jsonl`

---

## Quick Interpretation
- This larger run improved to a best validation soft score of `0.3715` at epoch 3.
- Final epoch (4) was not best, and checkpoint selection worked correctly.
- Pattern remains consistent with prior runs:
  - strong `yes/no`
  - weaker `other`
  - moderate `number`

This run serves as a larger, laptop-safe benchmark-oriented Version 1 reference.
