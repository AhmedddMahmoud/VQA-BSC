from __future__ import annotations

from dataclasses import dataclass
import re

from src.utils.metrics import normalize_answer_for_eval


@dataclass
class SelectionResult:
    final_answer: str
    chosen_from: str
    reason: str


def select_final_answer(first_answer: str, verifier_answer: str | None) -> SelectionResult:
    first = (first_answer or "").strip()
    verify = (verifier_answer or "").strip()

    if not verify:
        return SelectionResult(final_answer=first, chosen_from="first_pass", reason="verifier_empty")

    if normalize_answer_for_eval(verify) == normalize_answer_for_eval(first):
        return SelectionResult(final_answer=first, chosen_from="first_pass", reason="same_normalized_answer")

    verify_token_count = len(verify.split())
    if verify_token_count == 0:
        return SelectionResult(final_answer=first, chosen_from="first_pass", reason="verifier_no_tokens")
    if verify_token_count > 6:
        return SelectionResult(final_answer=first, chosen_from="first_pass", reason="verifier_too_long")

    return SelectionResult(final_answer=verify, chosen_from="verifier", reason="accepted_short_verifier_answer")


_GENERIC_ANSWERS = {
    "object",
    "thing",
    "person",
    "man",
    "woman",
    "people",
    "car",
    "room",
    "area",
    "building",
    "sign",
    "logo",
    "text",
    "word",
    "unknown",
}

_ACTION_VERBS = {
    "talking",
    "walking",
    "standing",
    "sitting",
    "running",
    "playing",
    "cleaning",
    "holding",
    "wearing",
    "looking",
    "eating",
    "drinking",
    "sleeping",
    "working",
    "riding",
    "driving",
    "cooking",
}

_NUMERIC_LIKE = re.compile(r"^(\d+|\d+[:.]\d+|\d+\s*(minutes|minute|hours|hour|hrs|hr))$", re.IGNORECASE)


def _normalize(text: str) -> str:
    return normalize_answer_for_eval(text or "")


def _token_count(text: str) -> int:
    return len(_normalize(text).split())


def _is_yes_no_question(question: str) -> bool:
    q = _normalize(question)
    return bool(re.match(r"^(is|are|was|were|do|does|did|has|have|can|could|should|will)\b", q))


def _is_yes_no_answer(answer: str) -> bool:
    return _normalize(answer) in {"yes", "no"}


def _is_generic(answer: str) -> bool:
    return _normalize(answer) in _GENERIC_ANSWERS


def _is_numeric_or_time(answer: str) -> bool:
    a = _normalize(answer)
    return bool(_NUMERIC_LIKE.match(a))


def _is_action_like(answer: str) -> bool:
    a = _normalize(answer)
    if not a:
        return False
    first = a.split()[0]
    return first in _ACTION_VERBS or a.endswith("ing")


def _is_concise(answer: str, max_tokens: int = 3) -> bool:
    return _token_count(answer) <= max_tokens


def _is_more_specific(base: str, routed: str) -> bool:
    if not base or not routed:
        return False
    if _is_generic(routed) and not _is_generic(base):
        return True
    if _token_count(base) > _token_count(routed) and not _is_generic(base) and _is_generic(routed):
        return True
    return False


def select_candidate(route: str, question: str, base_answer: str, routed_answer: str) -> SelectionResult:
    base = (base_answer or "").strip()
    routed = (routed_answer or "").strip()
    q_norm = _normalize(question)

    if route == "general":
        return SelectionResult(final_answer=base, chosen_from="base", reason="general_route_base")

    if route == "counting":
        if _is_numeric_or_time(routed):
            return SelectionResult(final_answer=routed, chosen_from="routed", reason="counting_numeric_routed")
        return SelectionResult(final_answer=base, chosen_from="base", reason="counting_non_numeric_routed")

    if route == "ocr_or_text_sensitive":
        if not routed:
            return SelectionResult(final_answer=base, chosen_from="base", reason="ocr_empty_routed")
        if _is_generic(routed):
            return SelectionResult(final_answer=base, chosen_from="base", reason="ocr_generic_routed")
        if _is_yes_no_answer(routed) and not _is_yes_no_question(q_norm):
            return SelectionResult(final_answer=base, chosen_from="base", reason="ocr_yesno_non_yesno_question")
        return SelectionResult(final_answer=routed, chosen_from="routed", reason="ocr_routed_ok")

    if route == "localization_or_missing_object":
        if "where" in q_norm:
            if _is_more_specific(base, routed):
                return SelectionResult(final_answer=base, chosen_from="base", reason="where_base_more_specific")
            if _is_generic(routed):
                return SelectionResult(final_answer=base, chosen_from="base", reason="where_generic_routed")
            return SelectionResult(final_answer=routed, chosen_from="routed", reason="where_routed_ok")

        if "what color" in q_norm or "color" in q_norm:
            if routed and _is_concise(routed) and not _is_generic(routed):
                return SelectionResult(final_answer=routed, chosen_from="routed", reason="color_concise_routed")
            return SelectionResult(final_answer=base, chosen_from="base", reason="color_routed_not_concise")

        if re.search(r"what is the .* (doing|wearing|holding|made of)", q_norm):
            if routed and (_is_action_like(routed) or _is_concise(routed)) and not _is_generic(routed):
                return SelectionResult(final_answer=routed, chosen_from="routed", reason="action_routed_ok")
            return SelectionResult(final_answer=base, chosen_from="base", reason="action_routed_not_action")

        if routed and not _is_generic(routed):
            return SelectionResult(final_answer=routed, chosen_from="routed", reason="localization_routed_ok")
        return SelectionResult(final_answer=base, chosen_from="base", reason="localization_routed_generic")

    if route == "spatial_or_relation_sensitive":
        if not routed:
            return SelectionResult(final_answer=base, chosen_from="base", reason="spatial_empty_routed")
        if _is_generic(routed):
            return SelectionResult(final_answer=base, chosen_from="base", reason="spatial_generic_routed")
        if _is_yes_no_answer(routed) and not _is_yes_no_question(q_norm):
            return SelectionResult(final_answer=base, chosen_from="base", reason="spatial_yesno_non_yesno_question")
        return SelectionResult(final_answer=routed, chosen_from="routed", reason="spatial_routed_ok")

    return SelectionResult(final_answer=base, chosen_from="base", reason="fallback_base")
