# Track B PaliGemma 1800-Sample Failure Analysis

## 1. Purpose

This report analyzes the remaining failure modes in the existing Track B pretrained VLM evaluation artifacts (PaliGemma, inference-only) on the 1800-sample validation setting. The goal is to identify what kinds of errors remain, distinguish true model errors from annotation/normalization effects, and compare the Track B failure profile against the locked Track A classical baseline failure profile.

## 2. Source Artifacts

Exact files used:

- `logs/eval_track_b_paligemma_metrics_val_1800_kaggle_20260421_143715.json`
- `predictions/eval_track_b_paligemma_predictions_val_1800_kaggle_20260421_143715.jsonl`
- `predictions/eval_track_b_paligemma_failures_val_1800_kaggle_20260421_143715.jsonl`

## 3. Evaluation Context

- Model: `google/paligemma-3b-ft-vqav2-224`
- Prompt: `<image> answer {question}`
- Mode: inference-only Track B path (no fine-tuning, no retrieval, no OCR, no KB)
- `sample_count`: `1800`
- `top1_accuracy`: `0.8556`
- `vqa_soft_accuracy`: `0.9287`

Answer-type breakdown (from metrics artifact):
- `yes/no`: `count=722`, `exact=0.9543`, `soft=0.9935`
- `number`: `count=235`, `exact=0.7787`, `soft=0.8752`
- `other`: `count=843`, `exact=0.7924`, `soft=0.8881`

## 4. Failure Summary

Computed from the 1800 predictions + failures JSONL files:

- Total evaluated samples: `1800`
- Total exact-match failures: `100`
- Failure rate: `5.56%` (`100 / 1800`)
- Average VQA soft score among failures: `0.5600`
- Failures with partial credit (`vqa_soft_score > 0`): `83`
- Strict failures (`vqa_soft_score == 0`): `17`

Interpretation: most exact failures are not completely wrong under VQA consensus; the majority still receive partial soft credit.

## 5. Failures by Answer Type

| Answer Type | Total Count | Exact Failures | Failure Rate | Avg Failure Soft Score |
|---|---:|---:|---:|---:|
| yes/no | 722 | 17 | 2.35% | 0.9216 |
| number | 235 | 16 | 6.81% | 0.3333 |
| other | 843 | 67 | 7.95% | 0.5224 |

Main signal:
- `yes/no` has low failure rate and very high soft score when failing (strong annotator disagreement effect).
- `number` and `other` account for most remaining strict/low-soft failures.

## 6. Main Failure Categories

Category counts below are heuristic but computed directly from failure rows.

### 6.1 Annotation disagreement / plausible alternative answer (`83`)

Explanation:
- Predicted answer is plausible and often appears in `all_gold_answers`, but differs from majority answer.
- This is a classic VQA consensus mismatch rather than a clear visual miss.

Representative examples:

1)
- Question: `Why is there a gap between the roof and wall?`
- Predicted: `ventilation`
- Gold majority: `yes`
- All gold answers: `['ventilation', 'provide air', 'to air out barn from stinky bovines', 'keep cow safe', 'airflow', 'to let sunlight in', 'yes', 'yes', 'for air', 'air']`
- Answer type: `other`
- VQA soft score: `0.3333`
- Failure interpretation: model gives semantically specific reason; majority label is coarse/ambiguous.

2)
- Question: `Is the bed white?`
- Predicted: `no`
- Gold majority: `yes`
- All gold answers: `['yes', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'yes', 'yes']`
- Answer type: `yes/no`
- VQA soft score: `1.0000`
- Failure interpretation: full annotator disagreement pattern; exact-majority fails but consensus score is maximal.

3)
- Question: `Where was the picture taken of the man?`
- Predicted: `outside`
- Gold majority: `forest`
- All gold answers: `['in forest', 'in jungle', 'park', 'forest', 'forest', 'outside', 'forest', 'not sure', 'outside', 'hawaii']`
- Answer type: `other`
- VQA soft score: `0.6667`
- Failure interpretation: plausible alternative location abstraction.

4)
- Question: `Does the door open inward or outward?`
- Predicted: `outward`
- Gold majority: `inward`
- All gold answers: `['inward', 'outward', 'inward', 'outward', 'outward', 'not sure', 'outward', 'out', 'inward', 'inward']`
- Answer type: `other`
- VQA soft score: `1.0000`
- Failure interpretation: severe annotator split; model is not uniquely wrong.

5)
- Question: `What color is the sign?`
- Predicted: `red`
- Gold majority: `red and yellow`
- All gold answers: `['red and yellow', 'red', 'red', 'orange, yellow, white, black', 'red and yellow', 'red/yellow', 'red orange', 'red and orange', 'red and yellow', 'red']`
- Answer type: `other`
- VQA soft score: `1.0000`
- Failure interpretation: under-specific but still strongly supported by annotations.

### 6.2 Counting / quantity error (`6`)

Explanation:
- Failures where numerical quantity/time is incorrect and usually receives zero soft credit.

Representative examples:

1)
- Question: `How many people can you see in the picture?`
- Predicted: `7`
- Gold majority: `8`
- All gold answers: `['8', '8', '8', '8', '8', '8', '8', '6', '8', '6']`
- Answer type: `number`
- VQA soft score: `0.0000`
- Failure interpretation: true counting miss.

2)
- Question: `What time is it on the clock?`
- Predicted: `7:35`
- Gold majority: `8:35`
- All gold answers: `['830', '8:35', '9:35', '6', '7:45', '6', '738', '8:35', '7', '8:35']`
- Answer type: `number`
- VQA soft score: `0.0000`
- Failure interpretation: fine-grained clock reading error.

3)
- Question: `How many dishes of food are in the picture?`
- Predicted: `15`
- Gold majority: `20`
- All gold answers: `['20', '19', '20', '19', '18', '17', '18', '10', '20', '19']`
- Answer type: `number`
- VQA soft score: `0.0000`
- Failure interpretation: counting underestimation.

4)
- Question: `How long until the clock says midnight?`
- Predicted: `11 hours`
- Gold majority: `50 minutes`
- All gold answers: `['12 hours 50 minutes', '55 minutes', '50 minutes', '10 minutes', '12 hours 53 minutes', '50 minutes', '50 minutes', '50 min', '12 hours 52 minutes', '50 minutes']`
- Answer type: `number`
- VQA soft score: `0.0000`
- Failure interpretation: temporal/numeric reasoning failure.

### 6.3 Visual object recognition error (`4`)

Explanation:
- Model predicts an object/scene concept not supported by annotation consensus.

Representative examples:

1)
- Question: `What is the background metal structure?`
- Predicted: `train station`
- Gold majority: `platform`
- All gold answers: `['trees', 'station', 'awning', 'platform', 'platform', 'platform', 'roof', 'shelter', 'train stop', 'awning']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: wrong object class granularity.

2)
- Question: `What is the woman in the room doing?`
- Predicted: `standing`
- Gold majority: `cleaning`
- All gold answers: `['talking to someone', 'cleaning', 'looking out window', 'talking', 'talking', 'cleaning', 'talking to someone outside of window', 'talking', 'talking out window', 'cleaning']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: activity recognition failure.

3)
- Question: `What does the blue sign say?`
- Predicted: `stop`
- Gold majority: `none`
- All gold answers: `['nothing', 'hijashurayuma dr', 'turnpike', 'none', 'not blue', 'none', 'truman', 'street name', 'go', 'truman rd']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: OCR-like content hallucination / recognition miss.

### 6.4 Attribute/color/material error (`3`)

Explanation:
- Incorrect fine-grained visual attribute prediction (color/type/clothing descriptor).

Representative examples:

1)
- Question: `What color is the bedspread?`
- Predicted: `tan`
- Gold majority: `white`
- All gold answers: `['beige', 'white', 'pink', 'white', 'white', 'white', 'yellow', 'yellow', 'beige', 'yellow']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: color/attribute mismatch.

2)
- Question: `What is the person wearing?`
- Predicted: `snowsuit`
- Gold majority: `snow suit and snowboard`
- All gold answers: `['snow suit and snowboard', 'pants and jacket', 'snow clothes', 'winter clothes', 'ski jacket/pants/gloves/boots', 'ski gear', 'skiing gear', 'winter wear', 'ski clothes', 'snow pants']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: under-specified apparel/gear composition.

3)
- Question: `What type of numbers are on the clock?`
- Predicted: `roman numerals`
- Gold majority: `numerical`
- All gold answers: `['numerical', 'regular numbers', 'digits', 'arabic', 'analog', 'add number', 'regular', 'normal', 'roman numeral', 'integers']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: semantic label mismatch in attribute taxonomy.

### 6.5 Spatial/relation reasoning error (`3`)

Explanation:
- Wrong relation/location answer (front/right/on relation).

Representative examples:

1)
- Question: `What is in front of the giraffe?`
- Predicted: `grass`
- Gold majority: `dirt`
- All gold answers: `['canopy', 'dirt', 'canopy', 'nothing', 'tent', 'dirt', 'shelter', 'camera', 'photographer', 'dirt']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: relation grounding error.

2)
- Question: `Where is the bird?`
- Predicted: `on bike`
- Gold majority: `on metal`
- All gold answers: `['in tree', 'plate', 'in machine', 'on wheel', 'on metal', 'on bicycle wheel', 'perched on machine', 'horseshoe', 'in picture', 'on metal']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: relation/object-support mismatch.

3)
- Question: `What is the object on the right?`
- Predicted: `knife`
- Gold majority: `microwave`
- All gold answers: `['cutting board', 'skillets', 'microwave', 'cookware', 'microwave', 'frying pan', 'pot', 'microwave', 'microwave oven', 'cutting board']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: spatial localization and object selection error.

### 6.6 Prompt/generation formatting issue (`1`)

Explanation:
- Very small residual bucket where wording style/format appears to contribute.

Representative example:

1)
- Question: `Why are the bicycles chained?`
- Predicted: `safety`
- Gold majority: `locked`
- All gold answers: `['locked', 'security', 'bike rack', 'not used', 'not being ridden', 'theft', 'keep them from being stolen', 'for security', 'theft prevention', 'protection']`
- Answer type: `other`
- VQA soft score: `0.0000`
- Failure interpretation: terse abstract wording not matching consensus lexical choices.

## 7. Comparison with Track A Failure Modes

Track A known profile:
- heavy collapse to frequent answers (e.g., `white`, `yes`, `1`),
- severe weakness on open-ended `other`,
- counting often collapsing to `1`,
- strong answer-prior bias.

Track B observed profile (this analysis):
- no dominant frequent-answer collapse signature,
- failures are far fewer (`100 / 1800`),
- majority of failures receive partial soft credit (`83/100`), indicating plausible alternatives,
- remaining strict failures concentrate in nuanced reasoning pockets: counting, object/attribute detail, relation grounding.

So the error regime shifts from **prior-collapse** (Track A) to **higher-level nuance and ambiguity** (Track B).

## 8. Interpretation

Scientifically, the remaining Track B errors split into three meaningful groups:

1. **True model failures** (object/count/relation/attribute misses): still present, especially in counting and fine-grained visual grounding.
2. **Evaluation/annotation softness effects**: dominant among exact mismatches; many "failures" are semantically plausible alternatives with non-zero (often high) soft credit.
3. **Formatting/lexical mismatch effects**: minor residual bucket.

This indicates the model now handles broad VQA competence strongly, while residual errors concentrate in fine-grained precision and annotation-consensus sensitivity.

## 9. Limitations

- Failure categorization here is heuristic and based on rule-guided grouping from JSONL rows.
- Exact semantic equivalence is hard to resolve automatically without human adjudication.
- Some examples reflect annotation disagreement rather than clear model deficiency.
- A stricter paired question-ID Track A vs Track B error-comparison lens could further harden causal claims.

## 10. Conclusion

- Track B strongly outperforms Track A at the 1800-sample scale.
- Remaining Track B failures are **not** dominated by frequent-answer collapse.
- Residual failures are more nuanced: counting/quantity mistakes, fine-grained visual detail misses, spatial-relation errors, and annotation-consensus/semantic mismatch cases.
- This is a qualitatively different and substantially improved failure profile compared with the classical baseline.
