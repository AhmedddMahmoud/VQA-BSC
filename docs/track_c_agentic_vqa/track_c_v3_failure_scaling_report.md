# Track C v3 Failure-Subset Scaling Report

## 1. Purpose

Track C is evaluated **only on Track B failure cases**, not full validation, because it is designed as a failure-recovery wrapper. This report documents scaling tests using failures from the **Track B 10000-sample run**.

## 2. Source Artifacts

### Track B 10000 failure artifact

- `docs/vlm docs/track_b_10000_results/predictions/eval_track_b_paligemma_failures_val_10000_kaggle_20260421_171722.jsonl`

### Track C config

- `configs/track_c_agentic_vqa_failures_10k.json`

### Output artifacts

#### 250-failure run

- Paired JSONL: `outputs/predictions/track_c_agentic_paired_failures_track_c_v3_failures250_from_10k_<timestamp>.jsonl`
- Metrics JSON: `outputs/logs/track_c_agentic_metrics_failures_track_c_v3_failures250_from_10k_<timestamp>.json`
- Summary MD: `outputs/logs/track_c_agentic_summary_failures_track_c_v3_failures250_from_10k_<timestamp>.md`

#### 500-failure run (if executed)

- Paired JSONL: `outputs/predictions/track_c_agentic_paired_failures_track_c_v3_failures500_from_10k_<timestamp>.jsonl`
- Metrics JSON: `outputs/logs/track_c_agentic_metrics_failures_track_c_v3_failures500_from_10k_<timestamp>.json`
- Summary MD: `outputs/logs/track_c_agentic_summary_failures_track_c_v3_failures500_from_10k_<timestamp>.md`

## 3. Track C v3 Architecture Reminder

- **Base Answer Agent** (reuse Track B answer from failure JSONL)
- **Router Agent** (router v2)
- **Specialist Agents** (General, Detail, Relation, Numeric, Text)
- **Selector / Risk Agent** (selector v2)
- **Trace logging** of base/specialist/final answers

## 4. Results Table

| Run | Failure Source | Rows Evaluated | Track B Soft | Track C Soft | Avg Soft Delta | Track C Exact | Recovery Count | Recovery Rate | Improved/Tied/Worsened | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| v3-50 (baseline) | 1800 failures | 50 | 0.5333 | 0.5733 | +0.0400 | 0.0400 | 2 | 0.0400 | 4/46/0 | Router v2 + selector v2 |
| v3-250 | 10000 failures | 250 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Fill after run |
| v3-500 | 10000 failures | 500 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Fill if run |

## 5. Route Distribution

- v3-250: TBD
- v3-500: TBD

## 6. Selector / Safety Analysis

Report:
- `worsened` count (should remain low)
- `selector_choice` distribution (base vs specialist)
- whether regressions are controlled

## 7. Interpretation

Discuss:
- whether v3 gains persist as failure subset scales
- whether gains shrink or remain meaningful
- whether architecture should be frozen or improved

## 8. Limitations

- Failure-subset evaluation is **not** full validation.
- Results depend on Track B failure artifacts.
- No external tools are used.
- Improvements are expected to be modest because Track B is already strong.

## 9. Conclusion

Provide a conservative conclusion based on the 250/500 runs:
- Whether Track C v3 shows scalable targeted recovery
- Or remains a modest prototype
