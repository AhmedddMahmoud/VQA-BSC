from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.datasets.vqa_dataset import VQAv2Dataset
from src.train.train_vqa import enforce_v1_contracts
from src.utils.config import V1Config
from src.utils.vocab import build_answer_vocab, build_question_vocab, extract_majority_answer, tokenize_question


def load_json(path: str) -> dict:
    payload_path = Path(path)
    if not payload_path.exists():
        raise FileNotFoundError(f"Required file not found: {payload_path}")
    with payload_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_question_annotation_maps(questions_path: str, annotations_path: str) -> tuple[Dict[int, dict], Dict[int, dict], int]:
    question_payload = load_json(questions_path)
    annotation_payload = load_json(annotations_path)

    question_map = {int(item["question_id"]): item for item in question_payload["questions"]}
    annotation_map = {int(item["question_id"]): item for item in annotation_payload["annotations"]}
    total_annotations = len(annotation_payload["annotations"])
    return question_map, annotation_map, total_annotations


def resolve_image_path(images_dir: Path, image_id: int) -> Path:
    train_style = images_dir / f"COCO_train2014_{image_id:012d}.jpg"
    if train_style.exists():
        return train_style
    val_style = images_dir / f"COCO_val2014_{image_id:012d}.jpg"
    if val_style.exists():
        return val_style
    return train_style


def build_split_dataset(
    split: str,
    config: V1Config,
    answer_to_idx: Dict[str, int],
    vocab,
) -> tuple[VQAv2Dataset, Dict[int, dict], int]:
    if split == "train":
        questions_path = config.train_questions_path
        annotations_path = config.train_annotations_path
        images_dir = config.train_images_dir
        max_samples = config.train_subset_size
    elif split == "val":
        questions_path = config.val_questions_path
        annotations_path = config.val_annotations_path
        images_dir = config.val_images_dir
        max_samples = config.val_subset_size
    else:
        raise ValueError(f"Unsupported split: {split}")

    _, annotation_map, total_annotations = load_question_annotation_maps(questions_path, annotations_path)

    dataset = VQAv2Dataset(
        questions_path=questions_path,
        annotations_path=annotations_path,
        images_dir=images_dir,
        vocab=vocab,
        answer_to_idx=answer_to_idx,
        max_question_length=config.question_max_length,
        image_size=config.image_size,
        max_samples=max_samples,
        use_precomputed_features=config.use_precomputed_features,
        precomputed_features_dir=config.precomputed_features_dir,
    )
    return dataset, annotation_map, total_annotations


def print_split_summary(
    split: str,
    dataset: VQAv2Dataset,
    total_annotations: int,
    answer_vocab_size: int,
    warn_drop_ratio: float,
) -> None:
    loaded = dataset.stats.get("loaded", len(dataset))
    skipped = dataset.stats.get("skipped_not_in_topk", 0)
    denominator = loaded + skipped
    drop_ratio = (skipped / denominator) if denominator > 0 else 0.0

    print("=" * 100)
    print(f"Split: {split}")
    print(f"- Total annotations in source file: {total_annotations}")
    print(f"- Dataset size (len(dataset)): {len(dataset)}")
    print(f"- Valid samples after filtering (loaded): {loaded}")
    print(f"- Filtered out / OOV (not in top-K): {skipped}")
    print(f"- Answer vocab size: {answer_vocab_size}")
    print(f"- Drop ratio: {drop_ratio:.4f}")
    if drop_ratio > warn_drop_ratio:
        print(
            "WARNING: High dropped-sample ratio detected. "
            "This can indicate answer-vocab mismatch or aggressive filtering."
        )


def inspect_examples(
    split: str,
    dataset: VQAv2Dataset,
    annotation_map: Dict[int, dict],
    answer_vocab: List[str],
    max_examples: int,
    fail_on_missing_image: bool,
) -> None:
    num_examples = min(max_examples, len(dataset))
    if num_examples == 0:
        raise RuntimeError(f"No samples available to inspect for split='{split}'.")

    print(f"\nInspecting {num_examples} examples from split='{split}'")

    for index in range(num_examples):
        sample_meta = dataset.samples[index]

        question_id = int(sample_meta["question_id"])
        annotation = annotation_map.get(question_id)
        if annotation is None:
            raise KeyError(f"Missing annotation for question_id={question_id}")

        raw_answers = [entry.get("answer", "") for entry in annotation.get("answers", [])]
        majority_answer = extract_majority_answer(annotation)

        label_idx = int(sample_meta["answer_idx"])
        if not (0 <= label_idx < len(answer_vocab)):
            raise AssertionError(
                f"Label index out of answer vocab range at sample {index}: "
                f"label_idx={label_idx}, answer_vocab_size={len(answer_vocab)}"
            )

        decoded_answer = answer_vocab[label_idx]
        if decoded_answer != majority_answer:
            raise AssertionError(
                "Label mapping mismatch detected: "
                f"question_id={question_id}, majority='{majority_answer}', decoded='{decoded_answer}', label_idx={label_idx}"
            )

        image_path = resolve_image_path(dataset.images_dir, int(sample_meta["image_id"]))
        if not image_path.exists() and not dataset.use_precomputed_features:
            print(f"WARNING: image path does not exist: {image_path}")
            if fail_on_missing_image:
                raise FileNotFoundError(f"Image file missing for question_id={question_id}: {image_path}")

        question_tokens = tokenize_question(sample_meta["question"])
        question_ids = dataset.vocab.encode(question_tokens, max_length=dataset.max_question_length)
        question_length = min(len(question_tokens), dataset.max_question_length)

        image_tensor_shape = "<not loaded>"
        if image_path.exists() and not dataset.use_precomputed_features:
            item = dataset[index]
            image_tensor_shape = str(tuple(item["image"].shape))

        print("-" * 100)
        print(f"sample_index: {index}")
        print(f"image_path: {image_path}")
        print(f"question_id: {question_id}")
        print(f"raw_question_text: {sample_meta['question']}")
        print(f"tokenized_question_ids: {question_ids}")
        print(f"question_length: {question_length}")
        print(f"raw_answers: {raw_answers}")
        print(f"normalized_majority_answer: {majority_answer}")
        print(f"answer_label_index: {label_idx}")
        print(f"decoded_answer_from_idx: {decoded_answer}")

        if image_path.exists() and not dataset.use_precomputed_features:
            print(f"image_tensor_shape: {image_tensor_shape}")
        else:
            if dataset.use_precomputed_features:
                print("image_tensor_shape: <not loaded, using precomputed features>")
            else:
                print("image_tensor_shape: <not loaded, image file missing>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect VQA dataset samples before training.")
    parser.add_argument("--config", type=str, default="configs/version1_vqa.json")
    parser.add_argument("--split", choices=["train", "val", "both"], default="train")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument(
        "--warn-drop-ratio",
        type=float,
        default=0.30,
        help="Warn when filtered_out/(loaded+filtered_out) exceeds this threshold.",
    )
    parser.add_argument(
        "--fail-on-missing-image",
        action="store_true",
        help="Raise FileNotFoundError when a sample image path is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = V1Config.from_json(args.config)
    enforce_v1_contracts(config)

    if args.num_examples < 1:
        raise ValueError("--num-examples must be >= 1")

    vocab = build_question_vocab([config.train_questions_path], max_vocab_size=config.question_vocab_size)
    answer_to_idx, answer_vocab = build_answer_vocab(config.train_annotations_path, top_k=config.answer_vocab_size)

    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        dataset, annotation_map, total_annotations = build_split_dataset(
            split=split,
            config=config,
            answer_to_idx=answer_to_idx,
            vocab=vocab,
        )
        print_split_summary(
            split=split,
            dataset=dataset,
            total_annotations=total_annotations,
            answer_vocab_size=len(answer_vocab),
            warn_drop_ratio=args.warn_drop_ratio,
        )
        inspect_examples(
            split=split,
            dataset=dataset,
            annotation_map=annotation_map,
            answer_vocab=answer_vocab,
            max_examples=args.num_examples,
            fail_on_missing_image=args.fail_on_missing_image,
        )


if __name__ == "__main__":
    main()
