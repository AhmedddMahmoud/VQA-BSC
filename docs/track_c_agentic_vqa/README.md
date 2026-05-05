# Track C — Agentic VQA Research and Planning

## What Track C Is

Track C is a **planned, exploratory agentic wrapper** around the already-strong Track B VLM path. It is not implemented yet. This folder defines the research rationale, scope boundaries, prototype design, and evaluation strategy before any coding starts.

## Why Consider Agentic VQA After Track B

Track B (`google/paligemma-3b-ft-vqav2-224`, inference-only) already performs strongly:

- 1800 samples: exact/top1 `0.8556`, VQA soft `0.9287`
- 5000 samples: exact/top1 `0.8474`, VQA soft `0.9242`
- 10000 samples: exact/top1 `0.8446`, VQA soft `0.9225`

Because Track B is strong, Track C is framed as a **targeted extension** for hard cases (e.g., counting, ambiguity, detail-sensitive questions), not a replacement program.

## Track Relationship (A → B → C)

- **Track A:** locked classical baseline (frozen ResNet-50 + GRU + 2-hop attention + top-1000 classification), 1800 result: exact/top1 `0.2983`, soft `0.3715`.
- **Track B:** modern pretrained VLM inference path, currently main high-performing branch.
- **Track C:** future research prototype to test whether lightweight routing/re-check behavior can recover a subset of Track B failures.

Track C should be treated as a controlled research extension built on top of Track B.

## Folder Contents

- `literature_scan.md`: paper-grounded research scan and practical takeaways.
- `problem_statement.md`: precise definition of what Track C should and should not solve.
- `prototype_plan.md`: conservative first prototype plan and phased roadmap.
- `system_design.md`: modular architecture and data-flow proposal for this repository.
- `evaluation_plan.md`: failure-case-first evaluation strategy and artifact plan.
- `risks_and_scope_control.md`: risk register and strict scope-control rules.
- `next_steps.md`: actionable checklist for moving from planning to implementation.
- `implementation_handover.md`: detailed record of what documentation was created in this task.
