from __future__ import annotations

from typing import Dict


DEFAULT_ROUTE_PROMPTS: Dict[str, str] = {
    "general": "<image> answer {question}",
    "counting": "<image> answer the question with a short number only. {question}",
        "localization_or_missing_object": "<image> inspect the specific object or region carefully. answer briefly but precisely. prefer the most specific visible location/detail rather than a broad category. {question}",
    "ocr_or_text_sensitive": "<image> if visible text matters, read it carefully and answer briefly. {question}",
    "spatial_or_relation_sensitive": "<image> reason about positions and relations between objects, then answer briefly. {question}",
}


DEFAULT_VERIFY_PROMPT = (
    "<image> verify the previous answer for this question and return only the final short answer. "
    "question: {question} previous_answer: {first_answer}"
)


def merged_route_prompts(overrides: Dict[str, str] | None = None) -> Dict[str, str]:
    prompts = dict(DEFAULT_ROUTE_PROMPTS)
    if overrides:
        for route, template in overrides.items():
            if isinstance(template, str) and template.strip():
                prompts[route] = template
    return prompts


def build_route_prompt(question: str, route: str, route_prompts: Dict[str, str]) -> str:
    template = route_prompts.get(route) or route_prompts.get("general", DEFAULT_ROUTE_PROMPTS["general"])
    return template.format(question=question)


def build_verify_prompt(
    question: str,
    first_answer: str,
    verify_template: str | None,
) -> str:
    template = verify_template or DEFAULT_VERIFY_PROMPT
    return template.format(question=question, first_answer=first_answer)
