from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from PIL import Image

from src.models.vlm.paligemma_runner import PaliGemmaRunner


@dataclass
class SpecialistResult:
    agent_name: str
    prompt: str
    answer: str


class SpecialistAgent:
    def __init__(self, agent_name: str, route_key: str) -> None:
        self.agent_name = agent_name
        self.route_key = route_key

    def run(self, runner: PaliGemmaRunner, image: Image.Image, question: str, prompt_template: str) -> SpecialistResult:
        original_template = runner.prompt_template
        try:
            runner.prompt_template = prompt_template
            answer = runner.generate_answers([image], [question])[0].strip()
        finally:
            runner.prompt_template = original_template

        return SpecialistResult(agent_name=self.agent_name, prompt=prompt_template, answer=answer)


def build_specialist_agents() -> Dict[str, SpecialistAgent]:
    return {
        "general": SpecialistAgent(agent_name="GeneralAgent", route_key="general"),
        "localization_or_missing_object": SpecialistAgent(agent_name="DetailAgent", route_key="localization_or_missing_object"),
        "spatial_or_relation_sensitive": SpecialistAgent(agent_name="RelationAgent", route_key="spatial_or_relation_sensitive"),
        "counting": SpecialistAgent(agent_name="NumericAgent", route_key="counting"),
        "ocr_or_text_sensitive": SpecialistAgent(agent_name="TextAgent", route_key="ocr_or_text_sensitive"),
    }
