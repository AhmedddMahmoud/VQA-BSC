# Version 1 Full Handover Report

Date: April 17, 2026  
Project: `VQA-VERSION1`

---

## 1) Project Status Summary

This repository is in a strong **Version 1 completed-baseline state** with controlled ablations implemented and benchmarked.

Current implemented scope:
- frozen `ResNet-50` image encoder
- RNN question encoder (`GRU` baseline + `BiLSTM` ablation support)
- 2-hop spatial attention
- top-K answer classification (now config-driven)
- train/eval/debug pipelines with saved artifacts and reports

Out-of-scope components (still excluded):
- retrieval
- OCR
- external knowledge
- generative answering

---

## 2) Core Architecture and Pipeline

### 2.1 Configuration Contract
File: `src/utils/config.py`

`V1Config` is the runtime source of truth. Key fields include:
- model/training settings
- dataset paths
- output folders
- experiment toggles

Notable current capabilities:
- `answer_top_k` support
- backward-compatible sync between `answer_vocab_size` and `answer_top_k`
- optional `use_soft_targets`
- `question_encoder_type` alias support (`gru`, `bilstm`)

### 2.2 Dataset Layer
File: `src/datasets/vqa_dataset.py`

Implements:
- `VQAv2Dataset` for real VQA data
- `SyntheticVQADataset` for smoke/debug
- `vqa_collate_fn`

Behavior highlights:
- majority-answer filtering to in-vocab top-K labels
- optional soft-target vectors from annotator answers
- supports raw image mode and precomputed feature mode

### 2.3 Model Layer
Files:
- `src/models/image_encoder.py`
- `src/models/question_encoder.py`
- `src/models/attention.py`
- `src/models/multihop_vqa.py`

Model composition:
1. visual tokens from frozen `ResNet-50`
2. question vector from RNN encoder
3. two attention hops
4. hop state updates
5. fusion + classifier head

Debug outputs include attention maps and intermediate tensors.

### 2.4 Training/Evaluation
Files:
- `src/train/train_vqa.py`
- `src/train/eval_vqa.py`

Training supports:
- hard-label CE loss
- soft-target CE loss
- checkpointing (`latest.pt`, `best.pt`)
- run summary + history artifacts

Evaluation supports:
- `top1_accuracy`
- `vqa_soft_accuracy` (primary benchmark metric)
- answer-type breakdown (`yes/no`, `number`, `other`)
- supplementary lexical metrics (BLEU/ROUGE-L/METEOR)
- predictions/failures/metrics JSON outputs

---

## 3) Experiments Completed (Verified from Artifacts)

### 3.1 Reference Larger Run (GRU, hard labels, top-1000)
Config: `configs/version1_real_larger_run_m4.json`  
Primary eval artifact:
- `outputs/final_runs/version1_m4_larger_run/logs/eval_metrics_val_best_larger_run_20260405_160653.json`

Best-checkpoint metrics:
- `sample_count = 1800`
- `top1_accuracy = 0.2983`
- `vqa_soft_accuracy = 0.3715`  ← primary

### 3.2 Soft-Target Objective Run
Config: `configs/version1_real_soft_targets.json`  
Primary eval artifact:
- `outputs/soft_target_runs/version1_soft_targets/logs/eval_metrics_val_best_soft_targets_20260405_181516.json`

Best-checkpoint metrics:
- `top1_accuracy = 0.2889`
- `vqa_soft_accuracy = 0.3641`

Result vs reference:
- underperformed reference on primary metric

### 3.3 Encoder Ablation: GRU vs BiLSTM
Config: `configs/version1_bilstm_ablation.json`  
Primary eval artifact:
- `outputs/encoder_ablation/bilstm/logs/eval_metrics_val_best_bilstm_20260413_220306.json`

BiLSTM best-checkpoint metrics:
- `top1_accuracy = 0.2767`
- `vqa_soft_accuracy = 0.3537`

Result vs reference:
- BiLSTM underperformed locked GRU baseline

### 3.4 Answer Space Ablation: Top-1000 vs Top-3000
Config: `configs/version1_real_top3000.json`  
Primary eval artifact:
- `outputs/topk_runs/version1_top3000/logs/eval_metrics_val_best_top3000_20260414_163755.json`

Top-3000 best-checkpoint metrics:
- `top1_accuracy = 0.2639`
- `vqa_soft_accuracy = 0.3370`
- `dataset_filtered_out_not_in_topk = 112` (vs 238 for top-1000 reference)

Result vs reference:
- despite lower filtering, top-3000 underperformed on primary metric

### 3.5 Earlier Baseline Reference (smaller subset run)
Artifact:
- `outputs/logs/eval_metrics_val_best_20260324_195101.json`

Metrics:
- `sample_count = 1000`
- `top1_accuracy = 0.2950`
- `vqa_soft_accuracy = 0.3693`

---

## 4) Key Reports and Documentation

Main docs currently present:
- `docs/version_1_classical_multihop_vqa.md`
- `docs/version_1_architecture_review_full.md`
- `docs/version_1_implementation_detailed_log.md`
- `docs/version_1_evaluation_metrics_implementation.md`
- `docs/version_1_first_real_baseline_experiment.md`
- `docs/version_1_baseline_best_checkpoint_failure_analysis.md`
- `docs/version_1_larger_run_m4_benchmark_results.md`
- `docs/version_1_gru_vs_bilstm_ablation_report.md`
- `docs/version_1_top1000_vs_top3000_ablation_report.md`
- `docs/version_1_supplementary_text_overlap_metrics.md`

---

## 5) Known Gaps / Cautions

1. **README drift**  
   `README.md` still describes strict top-1000 framing, while code now supports config-driven top-K.

2. **Run-summary overwrite risk**  
   `run_summary.json` may reflect a later run context; timestamped eval-metrics JSON files are the canonical comparison source.

3. **Sweep output consistency**  
   `outputs/sweeps/smoke_sweep_check/` is complete; `outputs/sweeps/v1_lr_epoch_sweep_m4/` appears partial at top level.

---

## 6) Artifact Map (Most Important Paths)

### Reference benchmark
- `outputs/final_runs/version1_m4_larger_run/checkpoints/best.pt`
- `outputs/final_runs/version1_m4_larger_run/logs/eval_metrics_val_best_larger_run_20260405_160653.json`

### Soft targets
- `outputs/soft_target_runs/version1_soft_targets/checkpoints/best.pt`
- `outputs/soft_target_runs/version1_soft_targets/logs/eval_metrics_val_best_soft_targets_20260405_181516.json`

### BiLSTM ablation
- `outputs/encoder_ablation/bilstm/checkpoints/best.pt`
- `outputs/encoder_ablation/bilstm/logs/eval_metrics_val_best_bilstm_20260413_220306.json`

### Top-3000 ablation
- `outputs/topk_runs/version1_top3000/checkpoints/best.pt`
- `outputs/topk_runs/version1_top3000/logs/eval_metrics_val_best_top3000_20260414_163755.json`

### Historical baseline
- `outputs/checkpoints/best.pt`
- `outputs/logs/eval_metrics_val_best_20260324_195101.json`

---

## 7) Recommended Next Steps (Version-1-Safe)

1. Update `README.md` to reflect:
   - `answer_top_k` support
   - current canonical benchmark artifact paths

2. Add one canonical frozen summary file (for example `docs/version_1_canonical_results.md`) that pins exact comparison metrics and artifact paths.

3. Continue single-factor controlled experiments only, keeping the locked reference unchanged:
   - GRU
   - hard labels
   - top-1000
   - frozen image encoder

4. If doing additional tuning, keep it strict and measurable:
   - LR schedule variants
   - grad clipping values
   - regularization adjustments
   - subset scaling with identical eval protocol

---

## 8) Handover Conclusion

The codebase is production-ready for **Version 1 classical baseline research iteration** with:
- clear module boundaries
- reproducible config-driven runs
- robust eval artifacts
- documented controlled ablations

Current evidence supports keeping the locked reference configuration (`GRU`, hard labels, top-1000) as the primary baseline for all further Version 1 comparisons.
