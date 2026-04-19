# Version 1 Supplementary Text-Overlap Metrics Integration

## Purpose
This document records the recent evaluation-layer extension that adds supplementary lexical/text-overlap metrics to the existing Version 1 classical multi-hop VQA baseline.

Important scope note:
- This change **does not** modify model architecture or training dynamics.
- Existing VQA metrics remain intact and primary.

---

## What Was Added
Supplementary metrics are now computed during evaluation and reported **in addition to** existing metrics:

- BLEU-1
- BLEU-2
- BLEU-4
- ROUGE-L
- METEOR (deterministic, lightweight implementation)

CIDEr status:
- **Not added** in this change to avoid heavy/fragile dependency overhead and keep the Version 1 pipeline lightweight and laptop-safe.

---

## Files Modified

### 1) `src/utils/metrics.py`
Added reusable supplementary metric utilities:
- tokenization helper for normalized overlap scoring
- n-gram counting helpers
- corpus BLEU computation:
  - `compute_corpus_bleu(..., max_n=1|2|4)`
- ROUGE-L computation:
  - `compute_rouge_l(...)`
- METEOR-style deterministic computation:
  - `compute_meteor(...)`
- unified wrapper:
  - `compute_supplementary_text_overlap_metrics(...)`

Implementation properties:
- deterministic behavior
- no external metric dependencies
- safe handling of short answers and empty token sequences

### 2) `src/train/eval_vqa.py`
Integrated supplementary metrics into the evaluation output path:
- computes `supplementary_text_overlap_metrics` from predictions and references
- adds a `primary_metric` field with value `vqa_soft_accuracy`
- prints supplementary metrics in CLI output
- saves supplementary metrics into the metrics summary JSON

Existing primary metrics were preserved unchanged:
- `eval_loss`
- `top1_accuracy`
- `vqa_soft_accuracy`
- `answer_type_breakdown`

---

## Reference Handling Assumptions
For supplementary metrics (BLEU/ROUGE-L/METEOR):
- each prediction is compared against **all available normalized gold answers** for that sample when available
- this enables multi-reference scoring behavior
- for synthetic/fallback paths, single normalized majority-answer reference is used

This is separate from primary VQA scoring:
- `vqa_soft_accuracy` remains the benchmark metric for model comparison

---

## Output Schema Changes
Metrics summary JSON (saved by eval) now includes:

- `primary_metric`: `"vqa_soft_accuracy"`
- `supplementary_text_overlap_metrics`:
  - `bleu_1`
  - `bleu_2`
  - `bleu_4`
  - `rouge_l`
  - `meteor`

No existing metric fields were removed.

---

## Verification Performed
The following checks were run successfully:

1. Python compile check:
- `src/utils/metrics.py`
- `src/train/eval_vqa.py`

2. Evaluation execution check (synthetic path):
- eval run completed
- CLI printed supplementary metrics
- saved metrics JSON included `supplementary_text_overlap_metrics`

---

## How to Run Evaluation
Real checkpoint evaluation (recommended):

```zsh
cd /Users/ahmedmahmoud/Documents/VQA-VERSION1
.venv/bin/python -m src.train.eval_vqa \
  --config configs/version1_real_train_subset.json \
  --checkpoint outputs/checkpoints/best.pt \
  --split-name val_best
```

Synthetic quick-check path:

```zsh
cd /Users/ahmedmahmoud/Documents/VQA-VERSION1
.venv/bin/python -m src.train.eval_vqa \
  --config configs/version1_real_train_subset.json \
  --checkpoint outputs/checkpoints/best.pt \
  --synthetic \
  --split-name val_synth_overlap_check
```

---

## Interpretation Caveats (Short-Answer VQA)
When reading supplementary lexical metrics for short-answer VQA:

- BLEU-2 and BLEU-4 can be near-zero due to short output length.
- ROUGE-L on one-token answers often behaves close to exact token overlap.
- METEOR on short answers can vary sharply with small lexical changes.
- Lexical metrics do not fully capture annotator-consensus behavior.

Therefore:
- use `vqa_soft_accuracy` as the **primary** benchmark metric
- use BLEU/ROUGE-L/METEOR as supplementary diagnostics only

---

## Scope Compliance
This change keeps Version 1 constraints intact:
- no architecture changes
- no retrieval/OCR/external knowledge
- no transformer/generative additions
- no dataset semantics or normalization behavior replacement
- no removal/replacement of existing VQA metrics
