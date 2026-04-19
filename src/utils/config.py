from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class V1Config:
    seed: int = 42
    device: str = "cuda"

    image_size: int = 224
    question_max_length: int = 20
    question_vocab_size: int = 12000
    answer_vocab_size: int = 1000
    answer_top_k: int = 1000

    word_embedding_dim: int = 300
    hidden_dim: int = 512
    rnn_type: str = "gru"
    question_encoder_type: str = ""
    bidirectional: bool = True
    num_hops: int = 2

    batch_size: int = 16
    num_workers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 5
    grad_clip_norm: float = 0.0

    scheduler_step_size: int = 0
    scheduler_gamma: float = 0.1

    train_subset_size: int = 10000
    val_subset_size: int = 5000
    freeze_image_encoder: bool = True
    use_precomputed_features: bool = False
    use_soft_targets: bool = False

    data_root: str = "data/vqa_v2"
    train_questions_path: str = "data/vqa_v2/v2_OpenEnded_mscoco_train2014_questions.json"
    train_annotations_path: str = "data/vqa_v2/v2_mscoco_train2014_annotations.json"
    val_questions_path: str = "data/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json"
    val_annotations_path: str = "data/vqa_v2/v2_mscoco_val2014_annotations.json"
    train_images_dir: str = "data/vqa_v2/train2014"
    val_images_dir: str = "data/vqa_v2/val2014"
    precomputed_features_dir: str = "data/vqa_v2/features"

    output_root: str = "outputs"
    checkpoints_dir: str = "outputs/checkpoints"
    logs_dir: str = "outputs/logs"
    predictions_dir: str = "outputs/predictions"

    qualitative_failures_to_save: int = 100

    @classmethod
    def from_json(cls, path: str | Path) -> "V1Config":
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)

        has_answer_vocab_size = "answer_vocab_size" in data
        has_answer_top_k = "answer_top_k" in data

        if has_answer_vocab_size and has_answer_top_k:
            if int(data["answer_vocab_size"]) != int(data["answer_top_k"]):
                raise ValueError(
                    "Config mismatch: answer_vocab_size and answer_top_k must match when both are provided."
                )
        elif has_answer_vocab_size and not has_answer_top_k:
            data["answer_top_k"] = int(data["answer_vocab_size"])
        elif has_answer_top_k and not has_answer_vocab_size:
            data["answer_vocab_size"] = int(data["answer_top_k"])

        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_default_config() -> V1Config:
    return V1Config()
