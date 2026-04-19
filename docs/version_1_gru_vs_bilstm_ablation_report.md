# Version 1 Encoder Ablation Report: GRU vs BiLSTM

## Scope
This report summarizes the controlled encoder ablation for Version 1:
- reference: existing GRU baseline (locked)
- comparison: BiLSTM run under matched settings
- everything else kept fixed (image encoder freeze, answer vocab, subsets, training/eval pipeline)

---

## Matched Setup Summary

| Setting | GRU Reference | BiLSTM Run |
|---|---:|---:|
| Question encoder | GRU | BiLSTM |
| Bidirectional | True | True |
| Train subset size | 9000 | 9000 |
| Val subset size | 1800 | 1800 |
| Answer vocab size | 1000 | 1000 |
| Learning rate | 0.001 | 0.001 |
| Epochs | 4 | 4 |
| Batch size | 16 | 16 |
| Frozen image encoder | True | True |
| Soft targets | False | False |

---

## Full Comparison Table (Best Checkpoint Eval)

| Metric | GRU | BiLSTM | Delta (BiLSTM - GRU) |
|---|---:|---:|---:|
| sample_count | 1800 | 1800 | 0 |
| eval_loss | 3.2182 | 3.2785 | +0.0603 |
| top1_accuracy | 0.2983 | 0.2767 | -0.0217 |
| vqa_soft_accuracy (primary) | 0.3715 | 0.3537 | -0.0178 |
| yes/no exact_match_accuracy | 0.5192 | 0.4957 | -0.0235 |
| yes/no vqa_soft_accuracy | 0.6497 | 0.6336 | -0.0161 |
| number exact_match_accuracy | 0.2564 | 0.2094 | -0.0470 |
| number vqa_soft_accuracy | 0.3291 | 0.2806 | -0.0484 |
| other exact_match_accuracy | 0.0764 | 0.0646 | -0.0118 |
| other vqa_soft_accuracy | 0.0887 | 0.0786 | -0.0101 |
| bleu_1 | 0.4310 | 0.4156 | -0.0155 |
| bleu_2 | 0.00002069 | 0.00002032 | -0.00000037 |
| bleu_4 | 0.0000001434 | 0.0000001421 | -0.0000000013 |
| rouge_l | 0.4262 | 0.4092 | -0.0170 |
| meteor | 0.2119 | 0.2032 | -0.0088 |

---

## Training Best Summary

| Item | GRU | BiLSTM |
|---|---:|---:|
| best_epoch | 3 | 1 |
| best_val_vqa_soft | 0.3715 | 0.3537 |
| best_val_top1 | 0.2983 | 0.2767 |

---

## Conclusion
Under a matched Version 1 setup, GRU outperformed BiLSTM on the primary benchmark metric:
- `vqa_soft_accuracy`: `0.3715` (GRU) vs `0.3537` (BiLSTM)

BiLSTM did not beat the locked GRU reference in this ablation.

---

## Artifact References

### GRU reference artifacts
- Metrics JSON:
  - `outputs/final_runs/version1_m4_larger_run/logs/eval_metrics_val_best_larger_run_20260405_160653.json`
- Predictions JSONL:
  - `outputs/final_runs/version1_m4_larger_run/predictions/eval_predictions_val_best_larger_run_20260405_160653.jsonl`
- Failures JSONL:
  - `outputs/final_runs/version1_m4_larger_run/predictions/eval_failures_val_best_larger_run_20260405_160653.jsonl`
- Training history:
  - `outputs/final_runs/version1_m4_larger_run/logs/training_history.json`

### BiLSTM ablation artifacts
- Best checkpoint:
  - `outputs/encoder_ablation/bilstm/checkpoints/best.pt`
- Metrics JSON:
  - `outputs/encoder_ablation/bilstm/logs/eval_metrics_val_best_bilstm_20260413_220306.json`
- Predictions JSONL:
  - `outputs/encoder_ablation/bilstm/predictions/eval_predictions_val_best_bilstm_20260413_220306.jsonl`
- Failures JSONL:
  - `outputs/encoder_ablation/bilstm/predictions/eval_failures_val_best_bilstm_20260413_220306.jsonl`
- Training history:
  - `outputs/encoder_ablation/bilstm/logs/training_history.json`
- Auto summary:
  - `outputs/encoder_ablation/gru_vs_bilstm_summary.json`
