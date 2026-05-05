# Track C v1 Failure-Recovery Report

## Scope

This report records staged fast-validation runs of Track C v1 on limited subsets of Track B failure cases.

- Evaluation mode: failure-case only (not full validation)
- Stage 1: 25 cases, verifier OFF (`track_c_fast25_verifier_off`)
- Stage 2: 50 cases, verifier OFF (`track_c_fast50_verifier_off`)

## Stage 1 Command (25, Verifier OFF)

```bash
cd "/Users/ahmedmahmoud/Documents/VQA-VERSION1"
./.venv/bin/python -m src.analysis.eval_agentic_vqa_failures \
  --config configs/track_c_agentic_vqa_failures.json \
  --max-cases 25 \
  --split-name track_c_fast25_verifier_off
```

## Input / Output Artifacts

### Input

- Track B failures source:
  - `predictions/eval_track_b_paligemma_failures_val_1800_kaggle_20260421_143715.jsonl`

### Outputs produced by Stage 1

- Paired rows JSONL:
  - `outputs/predictions/track_c_agentic_paired_failures_track_c_fast25_verifier_off_20260426_162757.jsonl`
- Metrics JSON:
  - `outputs/logs/track_c_agentic_metrics_failures_track_c_fast25_verifier_off_20260426_162757.json`
- Auto summary markdown:
  - `outputs/logs/track_c_agentic_summary_failures_track_c_fast25_verifier_off_20260426_162757.md`

## Stage 1 Metrics (25, Verifier OFF)

- Rows loaded: `100`
- Rows evaluated: `25`
- Samples: `25`

### Exact-style results

- Track B exact: `0.0000`
- Track C exact: `0.0400`
- Average exact delta (Track C - Track B): `+0.0400`

### VQA soft results

- Track B soft: `0.5200`
- Track C soft: `0.5600`
- Average soft delta (Track C - Track B): `+0.0400`

### Recovery signal

- Recovered failures: `1`
- Recovery rate: `0.0400`

### Comparison counts

- Improved: `3`
- Tied: `22`
- Worsened: `0`

### Route distribution

- `general`: `14`
- `localization_or_missing_object`: `4`
- `spatial_or_relation_sensitive`: `3`
- `counting`: `3`
- `ocr_or_text_sensitive`: `1`

## Stage 1 Interpretation

This first pass is a **conservative positive signal**:

1. Track C shows a small but real gain on both exact and soft metrics (`+0.04`).
2. No regressions were observed in this slice (`worsened = 0`).
3. Most cases remained tied, which is expected for a minimal wrapper designed to be low-risk.
4. The run is still too small (`n=25`) to claim robust general improvement.

## Stage 1 Limitations

- Small subset size (`25` out of `100` failure rows).
- Verifier disabled by design for speed.
- Results should be treated as **screening evidence**, not final conclusion.

## Stage 2 Command (50, Verifier OFF)

```bash
cd "/Users/ahmedmahmoud/Documents/VQA-VERSION1"
./.venv/bin/python -m src.analysis.eval_agentic_vqa_failures \
  --config configs/track_c_agentic_vqa_failures.json \
  --max-cases 50 \
  --split-name track_c_fast50_verifier_off
```

## Stage 2 Artifacts (50, Verifier OFF)

- Paired rows JSONL:
  - `outputs/predictions/track_c_agentic_paired_failures_track_c_fast50_verifier_off_20260426_172422.jsonl`
- Metrics JSON:
  - `outputs/logs/track_c_agentic_metrics_failures_track_c_fast50_verifier_off_20260426_172422.json`
- Auto summary markdown:
  - `outputs/logs/track_c_agentic_summary_failures_track_c_fast50_verifier_off_20260426_172422.md`

## Stage 3 Command (50, Verifier OFF, Router v2)

```bash
cd "/Users/ahmedmahmoud/Documents/VQA-VERSION1"
./.venv/bin/python -m src.analysis.eval_agentic_vqa_failures \
  --config configs/track_c_agentic_vqa_failures.json \
  --max-cases 50 \
  --split-name track_c_fast50_verifier_off_router_v2
```

## Stage 3 Artifacts (50, Verifier OFF, Router v2)

- Paired rows JSONL:
  - `outputs/predictions/track_c_agentic_paired_failures_track_c_fast50_verifier_off_router_v2_20260502_005550.jsonl`
- Metrics JSON:
  - `outputs/logs/track_c_agentic_metrics_failures_track_c_fast50_verifier_off_router_v2_20260502_005550.json`
- Auto summary markdown:
  - `outputs/logs/track_c_agentic_summary_failures_track_c_fast50_verifier_off_router_v2_20260502_005550.md`

## Stage 4 Command (50, Verifier OFF, Router v2 + Selector v2)

```bash
cd "/Users/ahmedmahmoud/Documents/VQA-VERSION1"
./.venv/bin/python -m src.analysis.eval_agentic_vqa_failures \
  --config configs/track_c_agentic_vqa_failures.json \
  --max-cases 50 \
  --split-name track_c_fast50_verifier_off_router_v2_selector_v2
```

## Stage 4 Artifacts (50, Verifier OFF, Router v2 + Selector v2)

- Paired rows JSONL:
  - `outputs/predictions/track_c_agentic_paired_failures_track_c_fast50_verifier_off_router_v2_selector_v2_20260503_152938.jsonl`
- Metrics JSON:
  - `outputs/logs/track_c_agentic_metrics_failures_track_c_fast50_verifier_off_router_v2_selector_v2_20260503_152938.json`
- Auto summary markdown:
  - `outputs/logs/track_c_agentic_summary_failures_track_c_fast50_verifier_off_router_v2_selector_v2_20260503_152938.md`

## Stage 2 Metrics (50, Verifier OFF)

- Rows loaded: `100`
- Rows evaluated: `50`
- Samples: `50`

### Exact-style results

- Track B exact: `0.0000`
- Track C exact: `0.0400`
- Average exact delta (Track C - Track B): `+0.0400`

### VQA soft results

- Track B soft: `0.5333`
- Track C soft: `0.5400`
- Average soft delta (Track C - Track B): `+0.0067`

### Recovery signal

- Recovered failures: `2`
- Recovery rate: `0.0400`

### Comparison counts

- Improved: `3`
- Tied: `46`
- Worsened: `1`

### Route distribution

- `general`: `29`
- `localization_or_missing_object`: `13`
- `spatial_or_relation_sensitive`: `4`
- `counting`: `3`
- `ocr_or_text_sensitive`: `1`

## Stage 3 Metrics (50, Verifier OFF, Router v2)

- Rows loaded: `100`
- Rows evaluated: `50`
- Samples: `50`

### Exact-style results

- Track B exact: `0.0000`
- Track C exact: `0.0400`
- Average exact delta (Track C - Track B): `+0.0400`

### VQA soft results

- Track B soft: `0.5333`
- Track C soft: `0.5600`
- Average soft delta (Track C - Track B): `+0.0267`

### Recovery signal

- Recovered failures: `2`
- Recovery rate: `0.0400`

### Comparison counts

- Improved: `4`
- Tied: `45`
- Worsened: `1`

### Route distribution

- `general`: `22`
- `localization_or_missing_object`: `19`
- `spatial_or_relation_sensitive`: `5`
- `counting`: `3`
- `ocr_or_text_sensitive`: `1`

## Stage 4 Metrics (50, Verifier OFF, Router v2 + Selector v2)

- Rows loaded: `100`
- Rows evaluated: `50`
- Samples: `50`

### Exact-style results

- Track B exact: `0.0000`
- Track C exact: `0.0400`
- Average exact delta (Track C - Track B): `+0.0400`

### VQA soft results

- Track B soft: `0.5333`
- Track C soft: `0.5733`
- Average soft delta (Track C - Track B): `+0.0400`

### Recovery signal

- Recovered failures: `2`
- Recovery rate: `0.0400`

### Comparison counts

- Improved: `4`
- Tied: `46`
- Worsened: `0`

### Route distribution

- `general`: `22`
- `localization_or_missing_object`: `19`
- `spatial_or_relation_sensitive`: `5`
- `counting`: `3`
- `ocr_or_text_sensitive`: `1`

## Stage 1 vs Stage 2 Snapshot

| Metric | 25 OFF | 50 OFF |
|---|---:|---:|
| Track C exact | 0.0400 | 0.0400 |
| Avg exact delta (C-B) | +0.0400 | +0.0400 |
| Track C soft | 0.5600 | 0.5400 |
| Avg soft delta (C-B) | +0.0400 | +0.0067 |
| Improved/Tied/Worsened | 3/22/0 | 3/46/1 |
| Recovery rate | 0.0400 | 0.0400 |

## Stage 2 vs Stage 3 Snapshot (Router v2 impact)

| Metric | 50 OFF (v1 router) | 50 OFF (v2 router) |
|---|---:|---:|
| Track C exact | 0.0400 | 0.0400 |
| Avg exact delta (C-B) | +0.0400 | +0.0400 |
| Track C soft | 0.5400 | 0.5600 |
| Avg soft delta (C-B) | +0.0067 | +0.0267 |
| Improved/Tied/Worsened | 3/46/1 | 4/45/1 |
| Recovery rate | 0.0400 | 0.0400 |
| Route: general | 29 | 22 |

## Stage 3 vs Stage 4 Snapshot (Selector v2 impact)

| Metric | 50 OFF (router v2) | 50 OFF (router v2 + selector v2) |
|---|---:|---:|
| Track C soft | 0.5600 | 0.5733 |
| Avg soft delta (C-B) | +0.0267 | +0.0400 |
| Improved/Tied/Worsened | 4/45/1 | 4/46/0 |
| Recovery rate | 0.0400 | 0.0400 |

## Updated Interpretation

The router v2 update keeps exact gains stable while **improving soft gains** and **reducing `general` routing** (29 → 22). The selector v2 then removes the remaining regression (worsened: 1 → 0) while further improving soft gains.

## Recommended Immediate Next Step

Optional: run a 50-case verifier-on test to check whether re-checking improves soft delta without increasing regressions.

```bash
cd "/Users/ahmedmahmoud/Documents/VQA-VERSION1"
./.venv/bin/python -m src.analysis.eval_agentic_vqa_failures \
  --config configs/track_c_agentic_vqa_failures.json \
  --max-cases 50 \
  --split-name track_c_fast50_verifier_on
```

Before running this command, set `"use_verifier": true` in `configs/track_c_agentic_vqa_failures.json`.

## Conclusion (Current Status)

Track C v1 shows **limited but real targeted recovery** in fast-validation mode. It is not yet strong enough for broad claims and should remain a scoped prototype pending the 50-case verifier-on comparison.
