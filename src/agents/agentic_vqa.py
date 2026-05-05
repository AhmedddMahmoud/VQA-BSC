from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set

import torch
from PIL import Image

from src.agents.prompt_policies import build_route_prompt, build_verify_prompt, merged_route_prompts
from src.agents.router import DeterministicRouter
from src.agents.selector import SelectionResult, select_candidate, select_final_answer
from src.agents.specialists import SpecialistResult, build_specialist_agents
from src.models.vlm.paligemma_runner import GenerationConfig, PaliGemmaRunner
from src.utils.metrics import normalize_answer_for_eval


@dataclass
class AgenticVQAConfig:
    model_name_or_path: str
    device: str = "cuda"
    generation_max_new_tokens: int = 8
    generation_do_sample: bool = False
    generation_num_beams: int = 1
    route_toggles: Dict[str, bool] | None = None
    prompt_templates: Dict[str, str] | None = None
    verifier_enabled: bool = True
    verifier_routes: Iterable[str] | None = None
    verifier_prompt_template: str | None = None


class AgenticVQASystem:
    def __init__(self, config: AgenticVQAConfig) -> None:
        device = torch.device(config.device if (config.device == "cpu" or torch.cuda.is_available()) else "cpu")
        generation = GenerationConfig(
            max_new_tokens=config.generation_max_new_tokens,
            do_sample=config.generation_do_sample,
            num_beams=config.generation_num_beams,
        )

        self.runner = PaliGemmaRunner(
            model_name_or_path=config.model_name_or_path,
            device=device,
            generation=generation,
            prompt_template="<image> answer {question}",
        )
        self.router = DeterministicRouter(route_toggles=config.route_toggles)
        self.route_prompts = merged_route_prompts(config.prompt_templates)
        self.specialists = build_specialist_agents()
        self.verifier_enabled = bool(config.verifier_enabled)
        self.verifier_routes: Set[str] = set(config.verifier_routes or [])
        self.verifier_prompt_template = config.verifier_prompt_template

    def _generate_with_template(self, image: Image.Image, question: str, template: str) -> str:
        original_template = self.runner.prompt_template
        try:
            self.runner.prompt_template = template
            return self.runner.generate_answers([image], [question])[0]
        finally:
            self.runner.prompt_template = original_template

    def answer(self, image: Image.Image, question: str) -> Dict[str, Any]:
        route_decision = self.router.route(question)
        route_prompt = build_route_prompt(
            question=question,
            route=route_decision.route,
            route_prompts=self.route_prompts,
        )

        first_answer_raw = self._generate_with_template(image=image, question=question, template=route_prompt)
        first_answer = first_answer_raw.strip()

        verifier_answer = None
        verify_prompt = None
        if self.verifier_enabled and route_decision.route in self.verifier_routes:
            verify_prompt = build_verify_prompt(
                question=question,
                first_answer=first_answer,
                verify_template=self.verifier_prompt_template,
            )
            verifier_answer = self._generate_with_template(image=image, question=question, template=verify_prompt).strip()

        selection: SelectionResult = select_final_answer(first_answer=first_answer, verifier_answer=verifier_answer)

        return {
            "final_answer": selection.final_answer,
            "final_answer_normalized": normalize_answer_for_eval(selection.final_answer),
            "route": route_decision.route,
            "route_rule": route_decision.rule,
            "route_prompt": route_prompt,
            "first_answer": first_answer,
            "verifier_enabled": self.verifier_enabled,
            "verifier_used": verify_prompt is not None,
            "verifier_prompt": verify_prompt,
            "verifier_answer": verifier_answer,
            "selected_from": selection.chosen_from,
            "selection_reason": selection.reason,
        }

    def answer_with_base(self, image: Image.Image, question: str, base_answer: str) -> Dict[str, Any]:
        route_decision = self.router.route(question)
        route_prompt = build_route_prompt(
            question=question,
            route=route_decision.route,
            route_prompts=self.route_prompts,
        )

        specialist = self.specialists.get(route_decision.route, self.specialists["general"])
        specialist_result: SpecialistResult = specialist.run(
            runner=self.runner,
            image=image,
            question=question,
            prompt_template=route_prompt,
        )

        selection: SelectionResult = select_candidate(
            route=route_decision.route,
            question=question,
            base_answer=base_answer,
            routed_answer=specialist_result.answer,
        )

        return {
            "base_answer": base_answer,
            "specialist_answer": specialist_result.answer,
            "final_answer": selection.final_answer,
            "final_answer_normalized": normalize_answer_for_eval(selection.final_answer),
            "selector_choice": selection.chosen_from,
            "selector_reason": selection.reason,
            "route": route_decision.route,
            "route_rule": route_decision.rule,
            "route_prompt": route_prompt,
            "specialist_agent": specialist_result.agent_name,
            "specialist_prompt": specialist_result.prompt,
        }

    def answer_with_base_or_generate(self, image: Image.Image, question: str, base_answer: str | None) -> Dict[str, Any]:
        base = (base_answer or "").strip()
        if not base:
            base_prompt = build_route_prompt(
                question=question,
                route="general",
                route_prompts=self.route_prompts,
            )
            base = self._generate_with_template(image=image, question=question, template=base_prompt).strip()
        return self.answer_with_base(image=image, question=question, base_answer=base)
