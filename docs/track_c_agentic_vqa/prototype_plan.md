# Prototype Plan — Track C (Lightweight Agentic VQA)

## Prototype Goal

Build a small, testable Track C wrapper that keeps Track B as the base VLM answerer and adds minimal routing/re-check behavior for selected hard cases.

## Why We Are Keeping It Small

- Track B already performs strongly; destabilizing it is unnecessary.
- A bachelor-project timeline requires measurable, controlled scope.
- A small prototype is easier to evaluate scientifically on known failure subsets.

## Proposed Prototype Modules

1. **Planner / Router**
   - Classifies incoming question into route types.
2. **Base VLM Answerer (Track B existing)**
   - Calls current PaliGemma inference path.
3. **Specialized Prompt Route(s)**
   - Applies route-specific prompt templates.
4. **Optional Verifier / Re-check Stage**
   - Performs a constrained second pass for uncertain cases.
5. **Final Answer Selector**
   - Chooses between first-pass and routed/rechecked output, with trace.

## Suggested Initial Route Types

- `general`
- `counting`
- `localization_or_missing_object`
- `ocr_or_text_sensitive`
- `spatial_or_relation_sensitive`

## What We Will NOT Build in v1

- No RL training loop.
- No large external tool ecosystem.
- No heavy multi-agent orchestration framework.
- No immediate full-dataset reruns as first validation step.

## Evaluation Recommendation for First Prototype

Evaluate Track C first on **Track B failure cases**, not on the full validation set. This gives a precise answer to whether routing helps where Track B currently struggles.

## Proposed Implementation Order

1. Define failure-case subset and route taxonomy.
2. Implement minimal router with deterministic rules.
3. Add route-specific prompt templates.
4. Add optional re-check policy for selected routes.
5. Run paired comparison on same failure-case subset.
6. Expand only if measurable gains are observed.

## Staged Roadmap

- **Phase 1: Research docs**
  - Finalize planning documents and design constraints.
- **Phase 2: Lightweight routing prototype**
  - Implement minimal Track C inference wrapper.
- **Phase 3: Targeted evaluation on Track B failures**
  - Run paired, case-matched comparisons.
- **Phase 4: Optional future extension**
  - Consider broader routes/tools only if Phase 3 shows clear value.
