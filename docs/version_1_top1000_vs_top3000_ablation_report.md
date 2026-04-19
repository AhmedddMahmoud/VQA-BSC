# Version 1 Answer-Space Ablation Report: Top-1000 vs Top-3000

## Scope
This report summarizes the controlled Version 1 experiment comparing:
- **Reference baseline:** GRU hard-label run with **top-1000** answers
- **New experiment:** same GRU hard-label setup with **top-3000** answers

All other major factors were held constant:
- frozen pretrained ResNet-50 image encoder
- GRU question encoder
- 2-hop attention
- same optimizer family and core training pipeline
- same evaluation pipeline and metrics

---

## Run Context

| Item | Top-1000 (Reference) | Top-3000 (Experiment) |
|---|---:|---:|
| Question encoder | GRU | GRU |
| Image encoder frozen | True | True |
| Train subset size | 9000 | 9000 |
| Val subset size | 1800 | 1800 |
| Answer top-K | 1000 | 3000 |
| Primary metric | vqa_soft_accuracy | vqa_soft_accuracy |

---

## Full Comparison Table (Best Checkpoint Eval)

| Metric | Top-1000 | Top-3000 | Delta (Top-3000 - Top-1000) |
|---|---:|---:|---:|
| sample_count | 1800 | 1800 | 0 |
| dataset_filtered_out_not_in_topk | 238 | 112 | -126 |
| eval_loss | 3.2182 | 3.5870 | +0.3688 |
| top1_accuracy | 0.2983 | 0.2639 | -0.0344 |
| vqa_soft_accuracy (primary) | 0.3715 | 0.3370 | -0.0344 |
| yes/no count | 807 | 768 | -39 |
| yes/no exact_match_accuracy | 0.5192 | 0.4948 | -0.0244 |
| yes/no vqa_soft_accuracy | 0.6497 | 0.6302 | -0.0195 |
| number count | 234 | 235 | +1 |
| number exact_match_accuracy | 0.2564 | 0.2128 | -0.0436 |
| number vqa_soft_accuracy | 0.3291 | 0.2851 | -0.0440 |
| other count | 759 | 797 | +38 |
| other exact_match_accuracy | 0.0764 | 0.0565 | -0.0200 |
| other vqa_soft_accuracy | 0.0887 | 0.0698 | -0.0189 |
| bleu_1 | 0.4310 | 0.4019 | -0.0291 |
| bleu_2 | 0.00002069 | 0.00001997 | -0.00000072 |
| bleu_4 | 0.0000001434 | 0.0000001408 | -0.0000000026 |
| rouge_l | 0.4262 | 0.3943 | -0.0319 |
| meteor | 0.2119 | 0.1956 | -0.0163 |

---

## Result Summary
- On this controlled experiment, **top-3000 underperformed top-1000** on the primary benchmark metric.
- Primary metric delta:
  - `vqa_soft_accuracy`: **-0.0344**
- Top-1 delta:
  - `top1_accuracy`: **-0.0344**

---

## Interpretation Notes
- The top-3000 run includes more answers in-vocab, which reduced `filtered_out_not_in_topk`.
- Category counts changed (`yes/no`, `number`, `other`) because expanding answer space changes which samples survive top-K filtering and how majority labels map to the vocabulary.
- Therefore, direct per-category comparisons should be interpreted with this distribution shift in mind.

---

## Artifact References

### Top-1000 reference
- Metrics JSON:
  - `outputs/final_runs/version1_m4_larger_run/logs/eval_metrics_val_best_larger_run_20260405_160653.json`
- Predictions JSONL:
  - `outputs/final_runs/version1_m4_larger_run/predictions/eval_predictions_val_best_larger_run_20260405_160653.jsonl`
- Failures JSONL:
  - `outputs/final_runs/version1_m4_larger_run/predictions/eval_failures_val_best_larger_run_20260405_160653.jsonl`

### Top-3000 experiment
- Metrics JSON:
  - `outputs/topk_runs/version1_top3000/logs/eval_metrics_val_best_top3000_20260414_163755.json`
- Predictions JSONL:
  - `outputs/topk_runs/version1_top3000/predictions/eval_predictions_val_best_top3000_20260414_163755.jsonl`
- Failures JSONL:
  - `outputs/topk_runs/version1_top3000/predictions/eval_failures_val_best_top3000_20260414_163755.jsonl`
