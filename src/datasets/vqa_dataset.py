from __future__ import annotations

import json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.vocab import Vocab, extract_majority_answer, normalize_answer, tokenize_question


class VQAv2Dataset(Dataset):
    def __init__(
        self,
        questions_path: str,
        annotations_path: str,
        images_dir: str,
        vocab: Vocab,
        answer_to_idx: Dict[str, int],
        max_question_length: int,
        image_size: int,
        max_samples: Optional[int] = None,
        use_precomputed_features: bool = False,
        precomputed_features_dir: Optional[str] = None,
        use_soft_targets: bool = False,
    ) -> None:
        self.vocab = vocab
        self.answer_to_idx = answer_to_idx
        self.max_question_length = max_question_length
        self.use_precomputed_features = use_precomputed_features
        self.use_soft_targets = use_soft_targets
        self.precomputed_features_dir = Path(precomputed_features_dir) if precomputed_features_dir else None
        self.images_dir = Path(images_dir)
        self.answer_vocab_size = len(answer_to_idx)

        with Path(questions_path).open("r", encoding="utf-8") as file:
            question_payload = json.load(file)
        with Path(annotations_path).open("r", encoding="utf-8") as file:
            annotation_payload = json.load(file)

        question_by_id = {item["question_id"]: item for item in question_payload["questions"]}
        annotations = annotation_payload["annotations"]

        self.samples: List[dict] = []
        skipped = 0
        for annotation in annotations:
            question_id = annotation["question_id"]
            question_obj = question_by_id.get(question_id)
            if question_obj is None:
                continue

            majority_answer = extract_majority_answer(annotation)
            if majority_answer not in answer_to_idx:
                skipped += 1
                continue

            self.samples.append(
                {
                    "question_id": question_id,
                    "question": question_obj["question"],
                    "image_id": question_obj["image_id"],
                    "answer_idx": answer_to_idx[majority_answer],
                    "soft_answer_counts": self._build_soft_answer_counts(annotation, answer_to_idx, answer_to_idx[majority_answer]),
                }
            )

            if max_samples is not None and len(self.samples) >= max_samples:
                break

        if len(self.samples) == 0:
            raise ValueError("No VQA samples available after top-K answer filtering.")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.stats = {"loaded": len(self.samples), "skipped_not_in_topk": skipped}

    @staticmethod
    def _build_soft_answer_counts(annotation: dict, answer_to_idx: Dict[str, int], fallback_idx: int) -> Dict[int, int]:
        counts: Counter[int] = Counter()
        for answer_obj in annotation.get("answers", []):
            normalized = normalize_answer(answer_obj.get("answer", ""))
            answer_idx = answer_to_idx.get(normalized)
            if answer_idx is not None:
                counts[answer_idx] += 1

        if len(counts) == 0:
            counts[fallback_idx] = 1
        return dict(counts)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_tensor(self, image_id: int) -> torch.Tensor:
        image_path = self.images_dir / f"COCO_train2014_{image_id:012d}.jpg"
        if not image_path.exists():
            image_path = self.images_dir / f"COCO_val2014_{image_id:012d}.jpg"
        with Image.open(image_path).convert("RGB") as image:
            return self.image_transform(image)

    def _load_precomputed_features(self, image_id: int) -> torch.Tensor:
        if self.precomputed_features_dir is None:
            raise ValueError("precomputed_features_dir must be set when use_precomputed_features=True")
        feature_path = self.precomputed_features_dir / f"{image_id:012d}.npy"
        array = np.load(feature_path)
        return torch.from_numpy(array).float()

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        question_tokens = tokenize_question(sample["question"])
        question_ids = self.vocab.encode(question_tokens, max_length=self.max_question_length)
        question_length = min(len(question_tokens), self.max_question_length)

        output = {
            "question_id": int(sample["question_id"]),
            "question_text": sample["question"],
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "question_length": question_length,
            "label": torch.tensor(sample["answer_idx"], dtype=torch.long),
            "image_id": int(sample["image_id"]),
        }

        if self.use_soft_targets:
            soft_target = torch.zeros(self.answer_vocab_size, dtype=torch.float32)
            total_count = 0.0
            for answer_idx, count in sample["soft_answer_counts"].items():
                soft_target[int(answer_idx)] = float(count)
                total_count += float(count)

            if total_count <= 0:
                soft_target[int(sample["answer_idx"])] = 1.0
                total_count = 1.0
            output["soft_target"] = soft_target / total_count

        if self.use_precomputed_features:
            output["image_features"] = self._load_precomputed_features(sample["image_id"])
        else:
            output["image"] = self._load_image_tensor(sample["image_id"])

        return output


class SyntheticVQADataset(Dataset):
    def __init__(
        self,
        size: int,
        question_length: int,
        answer_vocab_size: int,
        use_precomputed_features: bool,
        image_size: int,
        use_soft_targets: bool = False,
    ) -> None:
        self.size = size
        self.question_length = question_length
        self.answer_vocab_size = answer_vocab_size
        self.use_precomputed_features = use_precomputed_features
        self.image_size = image_size
        self.use_soft_targets = use_soft_targets

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        question_ids = torch.randint(low=2, high=100, size=(self.question_length,), dtype=torch.long)
        label = torch.tensor(index % self.answer_vocab_size, dtype=torch.long)
        sample = {
            "question_id": index,
            "question_text": f"synthetic question {index}",
            "question_ids": question_ids,
            "question_length": self.question_length,
            "label": label,
            "image_id": index,
        }

        if self.use_soft_targets:
            soft_target = torch.zeros(self.answer_vocab_size, dtype=torch.float32)
            soft_target[int(label.item())] = 1.0
            sample["soft_target"] = soft_target

        if self.use_precomputed_features:
            sample["image_features"] = torch.randn(49, 2048)
        else:
            sample["image"] = torch.randn(3, self.image_size, self.image_size)
        return sample


def vqa_collate_fn(batch: List[dict]) -> dict:
    output = defaultdict(list)
    for item in batch:
        for key, value in item.items():
            output[key].append(value)

    collated = {
        "question_ids": torch.stack(output["question_ids"], dim=0),
        "question_lengths": torch.tensor(output["question_length"], dtype=torch.long),
        "labels": torch.stack(output["label"], dim=0),
        "question_ids_list": output["question_id"],
        "question_texts": output["question_text"],
        "image_ids": output["image_id"],
    }

    if "image" in output:
        collated["images"] = torch.stack(output["image"], dim=0)
    if "image_features" in output:
        collated["image_features"] = torch.stack(output["image_features"], dim=0)
    if "soft_target" in output:
        collated["soft_targets"] = torch.stack(output["soft_target"], dim=0)

    return collated
