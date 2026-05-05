# Evaluation Plan — Track C Agentic VQA

## Why Full-Dataset Evaluation Is Not the First Step

Immediate full-dataset reruns are unnecessary for v1 because Track C is intended as a targeted recovery mechanism, not a full replacement model. Early proof should focus on whether routing helps where Track B currently fails.

## Why Track B Failure Cases Are the Right First Benchmark

Failure-case benchmarking provides the cleanest first test of utility:

- same images/questions,
- known weak points,
- direct before/after comparison,
- lower compute and faster iteration.

## Proposed Evaluation Inputs

- Track B failure-case subset (from existing failure artifacts).
- Matched metadata: question, answer type, original Track B answer, gold answers.
- Optional tagged subsets by category (counting, relation, OCR-sensitive, etc.).

## Proposed Comparison Metrics

- exact / normalized match,
- VQA soft accuracy,
- failure recovery rate,
- improvement over original Track B answer on the same cases,
- qualitative side-by-side examples.

## Proposed Outputs / Artifacts

- Paired JSON/JSONL with:
  - question id,
  - Track B original answer,
  - Track C routed answer,
  - route label,
  - score deltas.
- Summary metrics report (markdown + JSON).
- Compact qualitative report with representative recovered and unrecovered cases.

## Required Paired Comparison

Track C must be compared against the **original Track B answer on the exact same failure-case subset**. This paired setup avoids sample-mismatch bias and is the scientifically clean first step.

## Interpretation Rule

If paired failure-case gains are small or inconsistent, Track C should remain future work. Expansion to broader evaluation should only occur after stable evidence of targeted recovery.
