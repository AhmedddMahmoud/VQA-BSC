# System Design — Track C Agentic VQA

## Design Objective

Define a realistic, modular architecture for a future Track C prototype that reuses Track B components and adds lightweight routing logic.

## High-Level Modules

1. **Input Interface**
   - Input: image + question
   - Output: normalized request payload

2. **Planner / Router**
   - Input: question text (and optional first-pass confidence proxies)
   - Output: route label (`general`, `counting`, `localization_or_missing_object`, `ocr_or_text_sensitive`, `spatial_or_relation_sensitive`)

3. **Base Answerer (Track B Reuse)**
   - Input: image + routed prompt
   - Output: candidate answer from existing PaliGemma runner

4. **Optional Verifier / Second Pass**
   - Input: first answer + route context
   - Output: verified/revised candidate answer

5. **Final Answer Selector**
   - Input: one or more candidates + route metadata
   - Output: final answer + route trace artifact

## Suggested Data Flow

1. Receive image/question.
2. Router assigns route.
3. Build route-specific prompt.
4. Call existing Track B answerer.
5. Optionally run re-check for selected routes.
6. Select final answer and save trace.

## Pseudo-Workflow

```text
input(image, question)
  -> route = planner_router(question)
  -> prompt = prompt_builder(route, question)
  -> ans_1 = paligemma_runner(image, prompt)
  -> if needs_verification(route, ans_1):
       ans_2 = verifier_pass(image, question, ans_1, route)
       final = selector(ans_1, ans_2, route)
     else:
       final = ans_1
  -> return final, route_trace
```

## Track B Components to Reuse

- Existing VLM runner logic in `src/models/vlm/paligemma_runner.py`.
- Existing evaluation conventions and artifact style from Track B runs.
- Existing normalization/metrics pipeline for paired case analysis.

## Suggested Future Paths (When Implementation Starts)

- `src/agents/` for router, prompt policies, selector, trace schema.
- `src/analysis/` for failure-subset preparation and paired comparisons.
- `configs/track_c_*.json` for route toggles and prompt configuration.
- `docs/track_c_agentic_vqa/` for evolving design and experiment notes.

This document is design-only and intentionally does not add implementation code.
