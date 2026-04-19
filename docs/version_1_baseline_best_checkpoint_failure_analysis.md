# Version 1 Baseline Failure Pattern Analysis (Best Checkpoint)

## Context
This report analyzes error patterns for the **Version 1 baseline best checkpoint**:
- Checkpoint: `outputs/checkpoints/best.pt`
- Eval split tag: `val_best`
- Full predictions file: `outputs/predictions/eval_predictions_val_best_20260324_195101.jsonl`
- Qualitative failures file: `outputs/predictions/eval_failures_val_best_20260324_195101.jsonl`

Architecture/scope remain unchanged (classical Version 1 only).

---

## High-Level Result Snapshot
From the evaluated best checkpoint:
- `sample_count=1000`
- `top1_accuracy=0.2950`
- `vqa_soft_accuracy=0.3693`

From full prediction analysis:
- total wrong predictions: `705`
- wrong rate: `0.705`

---

## 1) Too Many Wrong Answers in `other`
Full wrong-answer distribution by answer type:
- `other`: `386`
- `yes/no`: `228`
- `number`: `91`

Finding:
- `other` is the largest error bucket by a wide margin.

Representative `other` errors:
- Q: "Where is he looking?" -> pred `white`, gold `down`
- Q: "What are the people in the background doing?" -> pred `white`, gold `watching`
- Q: "What is to the right of the soup?" -> pred `white`, gold `chopsticks`
- Q: "What is the man doing in the street?" -> pred `white`, gold `walking`

---

## 2) Overprediction of Common Answers
Top predicted answers among wrong samples (full file):
- `white`: `382`
- `yes`: `228`
- `1`: `80`
- `3`: `6`
- `red`: `6`

Finding:
- Strong collapse toward frequent/common outputs, especially `white`, `yes`, and `1`.
- This matches the observed qualitative behavior for color/frequent-token overprediction.

---

## 3) Counting Mistakes in `number`
For wrong samples with `answer_type=number`:
- count: `91`
- top wrong predicted answers:
  - `1`: `77`
  - `white`: `6`
  - `3`: `4`
  - `yes`: `3`

Additional subset check (from qualitative failures file):
- number failures in qualitative subset: `13`
- when both pred/gold parse as integers in subset: `12` pairs
  - average absolute error: `3.000`
  - within ±1: `6/12` (`0.500`)

Representative counting errors:
- Q: "How many pictures are there?" -> pred `1`, gold `7`
- Q: "How many frames are on the wall?" -> pred `1`, gold `7`
- Q: "How many pictures on the wall?" -> pred `1`, gold `7`
- Q: "How many people are on the field?" -> pred `1`, gold `3`
- Q: "How many signs?" -> pred `yes`, gold `4`

---

## 4) Plausible but Not Majority Label
Using full predictions among wrong samples:
- wrong with `vqa_soft_score > 0`: `119`
- rate within wrong: `0.169`

Using qualitative failures subset (100 rows):
- plausible-but-not-majority count: `16`
- soft-score histogram in subset:
  - `0.333`: `7`
  - `0.667`: `4`
  - `1.000`: `5`

Interpretation:
- A non-trivial fraction of "wrong" predictions are still partially supported by annotator answers.
- This is especially common in yes/no ambiguity and some open-ended `other` questions.

Representative plausible mismatches:
- Q: "Are these twin mattresses?" -> pred `yes`, majority `no`, soft `0.333`
- Q: "Could this be a hotel room?" -> pred `yes`, majority `no`, soft `0.333`
- Q: "Can you see the hook up for the train?" -> pred `yes`, majority `no`, soft `0.667`
- Q: "What color is the plant?" -> pred `white`, majority `green`, soft `0.333`
- Q: "How many cats are in the image?" -> pred `1`, majority `2`, soft `0.667`

---

## Conclusion
The saved failures show clear baseline error structure:
1. largest error concentration is in `other`
2. heavy overprediction bias toward common answers (`white`, `yes`, `1`)
3. systematic weakness on counting (`number` often collapsing to `1`)
4. meaningful share of plausible-but-not-majority answers (`soft > 0`) among exact-match errors

These findings should be used as the reference error profile for controlled Version 1 baseline improvement experiments.
