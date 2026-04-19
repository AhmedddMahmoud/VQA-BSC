# Version 1 Classical Multi-Hop VQA

This repository implements **Version 1 only**:
- frozen pretrained `ResNet-50`
- `GRU`/`BiLSTM` question encoder
- strict **2-hop spatial attention**
- **top-1000 answer classification**
- no retrieval, no OCR, no external knowledge, no generative answering

## Project Layout

- `src/datasets/vqa_dataset.py`: VQA v2 data loading + synthetic smoke dataset
- `src/models/image_encoder.py`: frozen ResNet-50 feature extraction
- `src/models/question_encoder.py`: GRU/BiLSTM question encoder
- `src/models/attention.py`: spatial attention hop module
- `src/models/multihop_vqa.py`: full multi-hop VQA model (returns attention weights)
- `src/train/train_vqa.py`: training entry point
- `src/train/eval_vqa.py`: evaluation entry point
- `src/utils/config.py`: all core constants and runtime settings
- `configs/version1_vqa.json`: default Version 1 config

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Smoke test (no dataset required)

Runs a synthetic end-to-end training pass to verify model/data/training loop:

```bash
python -m src.train.train_vqa --config configs/version1_vqa.json --synthetic
```

Then evaluate the saved checkpoint:

```bash
python -m src.train.eval_vqa --config configs/version1_vqa.json --checkpoint outputs/checkpoints/best_model.pt --synthetic
```

## 2) VQA v2 subset training (real data)

1. Put files in `data/vqa_v2/`:
   - `v2_OpenEnded_mscoco_train2014_questions.json`
   - `v2_mscoco_train2014_annotations.json`
   - `v2_OpenEnded_mscoco_val2014_questions.json`
   - `v2_mscoco_val2014_annotations.json`
   - `train2014/` images
   - `val2014/` images
2. Keep `configs/version1_vqa.json` with:
   - `answer_vocab_size = 1000`
   - `num_hops = 2`
   - small subset sizes for first run

Train:

```bash
python -m src.train.train_vqa --config configs/version1_vqa.json
```

Evaluate:

```bash
python -m src.train.eval_vqa --config configs/version1_vqa.json --checkpoint outputs/checkpoints/best_model.pt
```

## Outputs

- Best checkpoint: `outputs/checkpoints/best_model.pt`
- Vocab snapshots: `outputs/logs/question_vocab.json`, `outputs/logs/answer_vocab.json`
- Qualitative failures with attention-debug traces:
  - `outputs/predictions/qualitative_failures_epoch_*.jsonl`
  - `outputs/predictions/qualitative_failures_eval.jsonl`

## Notes for Laptop Compute

- Keep `freeze_image_encoder = true` first.
- If training is heavy, set `use_precomputed_features = true` and provide `(49, 2048)` `.npy` files under `data/vqa_v2/features/` keyed by image id.
- Start with the configured subset sizes before scaling up.
