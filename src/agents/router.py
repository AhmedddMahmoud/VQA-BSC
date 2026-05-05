from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict


VALID_ROUTES = {
    "general",
    "counting",
    "localization_or_missing_object",
    "ocr_or_text_sensitive",
    "spatial_or_relation_sensitive",
}


@dataclass
class RouteDecision:
    route: str
    rule: str


class DeterministicRouter:
    def __init__(self, route_toggles: Dict[str, bool] | None = None) -> None:
        self.route_toggles = dict(route_toggles or {})

        self._counting = re.compile(
            r"\b(how many|number of|total number|count|amount of|quantity of|what time|how long)\b",
            flags=re.IGNORECASE,
        )
        self._ocr = re.compile(
            r"\b(what does .* (say|read)|what is written|read|text on|word|words|letter|letters|label|brand name|street name|sign|license plate|logo says|jersey number|number on (sign|jersey|clock))\b",
            flags=re.IGNORECASE,
        )
        self._spatial = re.compile(
            r"\b(left|right|which side|behind|in front of|next to|between|under|below|above|over|on top of|near|far|closest|farthest|relative to|position|beside|alongside|touching|inside|outside|toward|facing|across from|adjacent)\b",
            flags=re.IGNORECASE,
        )
        self._localization = re.compile(
            r"\b(where is|where are|where was|where were|which .* is|find|locate|do you see|is there|missing|what color|what type|what kind|what object|what is in the background|what is the background|what is the .* doing|what is the .* wearing|what is the .* holding|what is the .* made of)\b",
            flags=re.IGNORECASE,
        )

    def _enabled(self, route: str) -> bool:
        if route not in VALID_ROUTES:
            return False
        return self.route_toggles.get(route, True)

    def route(self, question: str) -> RouteDecision:
        if not isinstance(question, str):
            question = str(question)

        if self._enabled("counting") and self._counting.search(question):
            return RouteDecision(route="counting", rule="counting_regex")

        if self._enabled("ocr_or_text_sensitive") and self._ocr.search(question):
            return RouteDecision(route="ocr_or_text_sensitive", rule="ocr_regex")

        if self._enabled("spatial_or_relation_sensitive") and self._spatial.search(question):
            return RouteDecision(route="spatial_or_relation_sensitive", rule="spatial_regex")

        if self._enabled("localization_or_missing_object") and self._localization.search(question):
            return RouteDecision(route="localization_or_missing_object", rule="localization_regex")

        return RouteDecision(route="general", rule="default_general")
