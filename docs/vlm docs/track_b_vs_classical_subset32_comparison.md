# Track B vs Classical Baseline — Subset-32 Comparison Report

## 1) Goal

This report records the direct comparison between:

- **Track A (classical baseline)** subset-32 evaluation artifact, and
- **Track B (PaliGemma VLM)** subset-32 evaluation artifact.

It captures exact file sources, metric values, deltas, interpretation, and fairness caveats.

---

## 2) Source Artifacts Used

## 2.1 Classical (Track A) subset-32 artifact

- Metrics file: `outputs/logs/eval_metrics_val_verify_20260324_184427.json`
- Key metadata:
  - `split`: `val_verify`
  - `sample_count`: `32`
  - `dataset_loaded_samples`: `32`
  - `dataset_filtered_out_not_in_topk`: `3`

## 2.2 Track B (PaliGemma) subset-32 artifact

- Metrics file: `outputs/logs/eval_track_b_paligemma_metrics_val_20260418_005106.json`
- Key metadata:
  - `split`: `val`
  - `track`: `track_b_vlm`
  - `model_name_or_path`: `google/paligemma-3b-ft-vqav2-224`
  - `sample_count`: `32`
  - `dataset_loaded_samples`: `32`
  - `dataset_filtered_out_not_in_topk`: `0`

---

## 3) Headline Metrics Comparison (Subset=32)

| Metric | Classical Track A | Track B (PaliGemma) | Delta (Track B - Track A) |
|---|---:|---:|---:|
| `top1_accuracy` | 0.03125 | 0.90625 | +0.87500 |
| `vqa_soft_accuracy` | 0.03125 | 0.9479166667 | +0.9166666667 |

Interpretation:
- On this subset-32 run, Track B strongly outperforms the historical classical subset-32 artifact.

---

## 4) Answer-Type Breakdown Comparison

## 4.1 yes/no

| Metric | Classical Track A | Track B |
|---|---:|---:|
| count | 17 | 14 |
| exact_match_accuracy | 0.0588235294 | 0.9285714286 |
| vqa_soft_accuracy | 0.0588235294 | 1.0000000000 |

## 4.2 number

| Metric | Classical Track A | Track B |
|---|---:|---:|
| count | 5 | 5 |
| exact_match_accuracy | 0.0000000000 | 1.0000000000 |
| vqa_soft_accuracy | 0.0000000000 | 1.0000000000 |

## 4.3 other

| Metric | Classical Track A | Track B |
|---|---:|---:|
| count | 10 | 13 |
| exact_match_accuracy | 0.0000000000 | 0.8461538462 |
| vqa_soft_accuracy | 0.0000000000 | 0.8717948718 |

Interpretation:
- Track B exceeds Track A across all answer types in this subset-32 comparison.
- Differences in type counts indicate the evaluated subsets are not guaranteed identical.

---

## 5) Supplementary Text Metrics (Track B report)

Track B metrics file includes:

- `bleu_1`: `0.9705882353`
- `bleu_2`: `0.9851843661`
- `bleu_4`: `0.0000313876`
- `rouge_l`: `0.96875`
- `meteor`: `0.51171875`

Classical subset-32 artifact used for this comparison does not include the same supplementary block, so only Track B values are reported here.

---

## 6) Fairness / Apples-to-Apples Caveat (Important)

This is a **strong practical comparison**, but not a perfect paired evaluation yet.

Why:
1. Classical `VQAv2Dataset` applies top-K answer-vocab filtering.
2. Classical artifact reports `dataset_filtered_out_not_in_topk=3`.
3. Track B path has no top-K filtering (`0` filtered).

Result:
- Even though both report `sample_count=32`, the exact question set may differ.
- Therefore this comparison is **indicative**, not a strict paired benchmark.

---

## 7) Conclusion

Based on available subset-32 artifacts:

- Track B currently shows much higher performance than the historical classical subset-32 run.
- The result is directionally compelling and consistent with expectations for a strong pretrained VLM.
- For strict scientific fairness, run a **paired comparison on identical `question_id`s** across Track A and Track B.

---

## 8) Next Action (requested and pending)

Next step to implement after this report:

- Build a small paired-evaluation comparator that:
  1. aligns both runs by exact `question_id`,
  2. computes shared metrics on the aligned set,
  3. outputs a strict apples-to-apples comparison JSON + markdown summary.

This will remove subset mismatch ambiguity and provide final fair comparison numbers.
