from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from src.utils.vocab import extract_majority_answer, normalize_answer


class VQAv2VLMEvalDataset(Dataset):
    def __init__(
        self,
        questions_path: str,
        annotations_path: str,
        images_dir: str,
        max_samples: Optional[int] = None,
    ) -> None:
        self.images_dir = Path(images_dir)

        with Path(questions_path).open("r", encoding="utf-8") as file:
            question_payload = json.load(file)
        with Path(annotations_path).open("r", encoding="utf-8") as file:
            annotation_payload = json.load(file)

        question_by_id = {item["question_id"]: item for item in question_payload["questions"]}
        annotations = annotation_payload["annotations"]

        self.samples: List[dict] = []
        for annotation in annotations:
            question_id = annotation["question_id"]
            question_obj = question_by_id.get(question_id)
            if question_obj is None:
                continue

            all_gold_answers = [normalize_answer(item.get("answer", "")) for item in annotation.get("answers", [])]
            if len(all_gold_answers) == 0:
                all_gold_answers = [normalize_answer(annotation.get("multiple_choice_answer", ""))]

            sample = {
                "question_id": int(question_id),
                "question_text": question_obj["question"],
                "image_id": int(question_obj["image_id"]),
                "all_gold_answers": all_gold_answers,
                "gold_majority_answer": extract_majority_answer(annotation),
                "answer_type": annotation.get("answer_type", ""),
            }
            self.samples.append(sample)

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if len(self.samples) == 0:
            raise ValueError("No VQA samples available for VLM evaluation.")

        self.stats = {
            "loaded": len(self.samples),
            "skipped_not_in_topk": 0,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_id: int) -> Path:
        train_style = self.images_dir / f"COCO_train2014_{image_id:012d}.jpg"
        if train_style.exists():
            return train_style
        val_style = self.images_dir / f"COCO_val2014_{image_id:012d}.jpg"
        if val_style.exists():
            return val_style
        return train_style

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image_path = self._resolve_image_path(sample["image_id"])
        with Image.open(image_path).convert("RGB") as image:
            image_rgb = image.copy()

        return {
            "question_id": sample["question_id"],
            "question_text": sample["question_text"],
            "image_id": sample["image_id"],
            "image_path": str(image_path),
            "image": image_rgb,
            "all_gold_answers": sample["all_gold_answers"],
            "gold_majority_answer": sample["gold_majority_answer"],
            "answer_type": sample["answer_type"],
        }


def vqa_vlm_eval_collate_fn(batch: List[dict]) -> dict:
    return {
        "question_ids": [int(item["question_id"]) for item in batch],
        "question_texts": [item["question_text"] for item in batch],
        "image_ids": [int(item["image_id"]) for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "images": [item["image"] for item in batch],
        "all_gold_answers": [item["all_gold_answers"] for item in batch],
        "gold_majority_answers": [item["gold_majority_answer"] for item in batch],
        "answer_types": [item["answer_type"] for item in batch],
    }
