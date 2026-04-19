# Version 1 First Real Baseline Experiment Record

## Scope Lock (Version 1 Only)
This repository is currently locked to the Version 1 classical multi-hop VQA baseline.

- Architecture in scope:
  - pretrained ResNet-50 image encoder
  - embedding + GRU/BiLSTM question encoder
  - 2-hop attention over visual regions
  - top-K answer classification (top-1000)
- Explicitly out of scope:
  - retrieval
  - OCR
  - external knowledge
  - transformers (BERT/BLIP/CLIP/ViT, etc.)
  - generative answering
  - Version 2 / Version 3 upgrades

---

## What Was Verified Before Baseline Run

### 1) Real Data Pipeline
Verified on real VQA data:
- questions load correctly
- annotations/answers load correctly
- images load correctly
- question/answer/image alignment is correct
- answer vocabulary mapping is correct
- decoded answers align with expected labels

### 2) Real Forward Pass
Verified end-to-end tensor flow on real batches:
- images -> ResNet feature map
- feature map -> flattened visual tokens (49 regions)
- questions -> token IDs -> embeddings -> final question vector
- hop-1 attention works
- hop-2 attention works
- fused representation shape is correct
- logits shape matches answer vocabulary
- attention softmax sums are valid

### 3) Tiny Real Overfit Check
A tiny real subset (32 samples) was overfit successfully:
- loss dropped near zero
- tiny-set accuracy reached 100%

This validated:
- optimizer/updates work
- gradients flow correctly
- labels are aligned
- training loop is functionally healthy

### 4) Evaluation Pipeline
Implemented and verified:
- top-1 exact-match accuracy
- VQA soft accuracy
- answer-type breakdown (`yes/no`, `number`, `other`)
- saved predictions JSONL
- saved qualitative failures JSONL
- saved metrics summary JSON

### 5) Training Pipeline
Implemented and verified:
- real training support
- subset train/val sizes
- epoch-end validation
- `latest.pt` and `best.pt` checkpoint saving
- training history JSON/CSV
- final/best prediction outputs
- final/best failure outputs

### 6) Freeze Behavior (Real Config)
Verified and logged explicitly:
- `freeze_image_encoder = true`
- image encoder is present in real-image mode
- all image encoder params are frozen
- trainable params are on question/attention/classifier side

---

## Reproducible Config Used for First Real Baseline
Config file:
- `configs/version1_real_train_subset.json`

Key values:
- `seed`: 42
- `device`: `cpu`
- `answer_vocab_size`: 1000
- `num_hops`: 2
- `freeze_image_encoder`: true
- `use_precomputed_features`: false
- `train_subset_size`: 5000
- `val_subset_size`: 1000
- `batch_size`: 16
- `epochs`: 4
- `learning_rate`: 0.001
- `weight_decay`: 1e-05
- `grad_clip_norm`: 1.0

---

## First Real Baseline Training Outcome
Observed during training:
- `epoch=1 train_loss=3.9215 val_loss=3.5045 val_top1=0.2710 val_soft=0.3410`
- `epoch=2 train_loss=3.3709 val_loss=3.3094 val_top1=0.2950 val_soft=0.3693`
- `epoch=3 train_loss=3.2929 val_loss=3.2733 val_top1=0.2670 val_soft=0.3417`
- `epoch=4 train_loss=3.2033 val_loss=3.3801 val_top1=0.2700 val_soft=0.3497`

Run summary:
- `best_epoch=2`
- `best_val_vqa_soft=0.3693`
- best checkpoint selected correctly by validation VQA soft accuracy
- final epoch was not the best epoch

---

## Best Checkpoint Evaluation (Reference Baseline)
Evaluated checkpoint:
- `outputs/checkpoints/best.pt`

Evaluation command used:
- `.venv/bin/python -m src.train.eval_vqa --config configs/version1_real_train_subset.json --checkpoint outputs/checkpoints/best.pt --split-name val_best`

Metrics:
- `sample_count=1000`
- `eval_loss=3.3094`
- `top1_accuracy=0.2950`
- `vqa_soft_accuracy=0.3693`

Answer-type breakdown:
- `yes/no`: `count=467`, `exact=0.5118`, `soft=0.6410`
- `number`: `count=122`, `exact=0.2541`, `soft=0.3115`
- `other`: `count=411`, `exact=0.0608`, `soft=0.0779`

Saved artifacts:
- predictions: `outputs/predictions/eval_predictions_val_best_20260324_195101.jsonl`
- failures: `outputs/predictions/eval_failures_val_best_20260324_195101.jsonl`
- metrics: `outputs/logs/eval_metrics_val_best_20260324_195101.json`

---

## Baseline Reference Decision
For Version 1 controlled experiments, treat this as the baseline reference:
- Reference checkpoint: `outputs/checkpoints/best.pt` (epoch 2)
- Reference validation VQA soft accuracy: `0.3693`

This project is now past pipeline sanity-check phase and in controlled improvement phase.
All future experiments should compare against this baseline with reproducible configs and clear logging.
