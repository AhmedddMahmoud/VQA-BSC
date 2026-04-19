from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor

try:
    from transformers import PaliGemmaForConditionalGeneration
except ImportError:
    PaliGemmaForConditionalGeneration = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 8
    do_sample: bool = False
    num_beams: int = 1


class PaliGemmaRunner:
    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device,
        generation: GenerationConfig,
        prompt_template: str = "<image> answer {question}",
    ) -> None:
        if PaliGemmaForConditionalGeneration is None:
            raise ImportError(
                "PaliGemmaForConditionalGeneration is unavailable in this transformers build. "
                "Please ensure transformers>=4.40.0 is installed."
            )

        self.device = device
        self.prompt_template = prompt_template
        self.generation = generation

        try:
            self.processor = AutoProcessor.from_pretrained(model_name_or_path)

            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=dtype,
            ).to(self.device)
        except OSError as exc:
            raise RuntimeError(
                "Failed to load the VLM checkpoint. "
                "If using a gated Hugging Face model, authenticate first with `hf auth login` "
                "and ensure your account is approved for access to the repository."
            ) from exc
        self.model.eval()

    def _make_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question)

    @torch.no_grad()
    def generate_answers(self, images: Sequence[Image.Image], questions: Sequence[str]) -> List[str]:
        if len(images) != len(questions):
            raise ValueError(f"images/questions length mismatch: {len(images)} vs {len(questions)}")

        prompts = [self._make_prompt(question) for question in questions]
        processor_inputs = self.processor(
            images=list(images),
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        input_ids = processor_inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Processor inputs are missing input_ids")

        processor_inputs = {key: value.to(self.device) for key, value in processor_inputs.items()}

        generated = self.model.generate(
            **processor_inputs,
            max_new_tokens=self.generation.max_new_tokens,
            do_sample=self.generation.do_sample,
            num_beams=self.generation.num_beams,
        )

        prompt_lengths = (input_ids != self.processor.tokenizer.pad_token_id).sum(dim=1).tolist()
        answers: List[str] = []
        for row_index, prompt_length in enumerate(prompt_lengths):
            completion_ids = generated[row_index, int(prompt_length) :]
            decoded = self.processor.decode(completion_ids, skip_special_tokens=True)
            answers.append(decoded.strip())

        return answers
