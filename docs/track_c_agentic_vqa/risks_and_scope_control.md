# Risks and Scope Control — Track C Agentic VQA

## Main Risks

- Agentic VQA can quickly expand into a separate large project.
- Complex orchestration can consume time without measurable gains.
- Research ambition may exceed available compute and timeline.

## Scope Creep Risks

- Adding many tools/routes before validating a minimal baseline.
- Expanding to end-to-end RL too early.
- Attempting full benchmark reruns before proving failure-case recovery.

## Engineering Risks

- Increased system complexity can reduce reproducibility.
- Route-specific prompts can become hard to maintain.
- Weak trace logging can make debugging and analysis unclear.

## Evaluation Risks

- Non-paired comparisons can produce misleading conclusions.
- Cherry-picked examples can overstate improvements.
- Ignoring failure subtypes can hide where the method breaks.

## How We Control Scope

- Keep Track B as the primary branch and stable baseline.
- Build one minimal router-first prototype only.
- Evaluate first on paired Track B failure cases.
- Require explicit evidence before adding new modules.
- Preserve repository cleanliness and documentation discipline.

## Strict “Do Not” List

- Do not replace Track B.
- Do not start with RL.
- Do not build five tools at once.
- Do not rerun everything as the first experiment.
- Do not destroy repo cleanliness.

## Positioning Rule

Track C should be treated as a **small prototype / future-work extension** unless it demonstrates clear, repeatable improvement under paired evaluation.
