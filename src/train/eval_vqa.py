from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.vqa_dataset import SyntheticVQADataset, VQAv2Dataset, vqa_collate_fn
from src.models.multihop_vqa import MultiHopVQAModel
from src.utils.config import V1Config
from src.utils.metrics import (
    compute_answer_type_breakdown,
    compute_supplementary_text_overlap_metrics,
    compute_top1_accuracy,
    compute_vqa_soft_accuracy,
    derive_answer_type,
    exact_match_after_normalization,
    normalize_answer_for_eval,
    save_jsonl,
)
from src.utils.vocab import Vocab, extract_majority_answer


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_question_encoder_type(config: V1Config) -> str:
    encoder_type = config.question_encoder_type.strip().lower() if isinstance(config.question_encoder_type, str) else ""
    if encoder_type == "":
        encoder_type = config.rnn_type.strip().lower()
    if encoder_type == "bilstm":
        return "lstm"
    if encoder_type in {"gru", "lstm"}:
        return encoder_type
    raise ValueError(f"Unsupported question encoder type '{encoder_type}'. Supported values: gru, bilstm (or lstm).")


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    answer_vocab: List[str],
    split_name: str,
    synthetic: bool,
    annotation_path: str | None,
    images_dir: str | None,
    qualitative_limit: int,
    save_outputs: bool,
    predictions_dir: str,
    logs_dir: str,
    file_prefix: str,
) -> dict:
    model.eval()

    annotation_map: Dict[int, dict] | None = None
    resolved_images_dir = Path(images_dir) if images_dir else Path(".")
    if not synthetic:
        if annotation_path is None:
            raise ValueError("annotation_path is required for real evaluation mode")
        annotation_map = _load_annotation_map(annotation_path)

    losses: List[float] = []
    all_pred_indices: List[int] = []
    all_gold_indices: List[int] = []
    all_pred_answers: List[str] = []
    all_gold_answers: List[List[str]] = []
    all_answer_type_records: List[dict] = []

    prediction_rows: List[dict] = []
    failure_rows: List[dict] = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                "question_ids": batch["question_ids"].to(device),
                "question_lengths": batch["question_lengths"].to(device),
            }
            if "images" in batch:
                inputs["images"] = batch["images"].to(device)
            if "image_features" in batch:
                inputs["image_features"] = batch["image_features"].to(device)

            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            logits = outputs["logits"]
            losses.append(float(criterion(logits, labels).item()))

            predictions = logits.argmax(dim=1)
            topk_values, topk_indices = torch.topk(logits, k=min(5, logits.size(1)), dim=1)

            labels_list = labels.detach().cpu().tolist()
            pred_list = predictions.detach().cpu().tolist()
            question_ids = [int(item) for item in batch["question_ids_list"]]
            question_texts = list(batch["question_texts"])
            image_ids = [int(item) for item in batch["image_ids"]]

            for index, question_id in enumerate(question_ids):
                pred_idx = int(pred_list[index])
                gold_idx = int(labels_list[index])
                pred_answer = answer_vocab[pred_idx]
                gold_majority_answer = answer_vocab[gold_idx]

                if annotation_map is not None:
                    annotation = annotation_map.get(question_id)
                    if annotation is None:
                        raise KeyError(f"Missing annotation for question_id={question_id}")

                    all_gold = [normalize_answer_for_eval(item.get("answer", "")) for item in annotation.get("answers", [])]
                    if len(all_gold) == 0:
                        all_gold = [normalize_answer_for_eval(annotation.get("multiple_choice_answer", gold_majority_answer))]

                    annotation_majority = extract_majority_answer(annotation)
                    answer_type = annotation.get("answer_type")
                    if not isinstance(answer_type, str) or answer_type.strip() == "":
                        answer_type = derive_answer_type(all_gold)
                else:
                    all_gold = [normalize_answer_for_eval(gold_majority_answer)]
                    annotation_majority = normalize_answer_for_eval(gold_majority_answer)
                    answer_type = derive_answer_type(all_gold)

                soft_score = compute_vqa_soft_accuracy([pred_answer], [all_gold])
                exact_match = exact_match_after_normalization(pred_answer, gold_majority_answer)
                image_path = _resolve_image_path(resolved_images_dir, image_ids[index]) if not synthetic else ""

                top_answers = []
                top_answer_scores = []
                for top_idx, top_value in zip(topk_indices[index].detach().cpu().tolist(), topk_values[index].detach().cpu().tolist()):
                    top_answers.append(answer_vocab[int(top_idx)])
                    top_answer_scores.append(float(top_value))

                row = {
                    "question_id": question_id,
                    "question_text": question_texts[index],
                    "predicted_answer": pred_answer,
                    "predicted_index": pred_idx,
                    "gold_majority_answer": normalize_answer_for_eval(gold_majority_answer),
                    "gold_index": gold_idx,
                    "all_gold_answers": all_gold,
                    "answer_type": answer_type,
                    "image_path": image_path,
                    "exact_match_correct": exact_match,
                    "vqa_soft_score": soft_score,
                    "topk_predicted_answers": top_answers,
                    "topk_logits": top_answer_scores,
                    "annotation_majority_answer": annotation_majority,
                }
                prediction_rows.append(row)

                if not exact_match:
                    failure_rows.append(row)

                all_pred_indices.append(pred_idx)
                all_gold_indices.append(gold_idx)
                all_pred_answers.append(pred_answer)
                all_gold_answers.append(all_gold)
                all_answer_type_records.append(
                    {
                        "gold_majority_answer": row["gold_majority_answer"],
                        "all_gold_answers": all_gold,
                        "answer_type": answer_type,
                    }
                )

    mean_loss = sum(losses) / len(losses) if losses else 0.0
    top1 = compute_top1_accuracy(all_pred_indices, all_gold_indices)
    vqa_soft = compute_vqa_soft_accuracy(all_pred_answers, all_gold_answers)
    answer_type_breakdown = compute_answer_type_breakdown(
        pred_answers=all_pred_answers,
        gold_answers_or_annotations=all_answer_type_records,
    )
    supplementary_text_metrics = compute_supplementary_text_overlap_metrics(
        pred_answers=all_pred_answers,
        all_gold_answers=all_gold_answers,
    )

    dataset = loader.dataset
    sample_count = len(all_pred_indices)
    dataset_loaded = getattr(dataset, "stats", {}).get("loaded", len(dataset)) if hasattr(dataset, "stats") else len(dataset)
    dataset_filtered = (
        getattr(dataset, "stats", {}).get("skipped_not_in_topk", 0) if hasattr(dataset, "stats") else 0
    )

    outputs = {
        "split": split_name,
        "synthetic": synthetic,
        "sample_count": sample_count,
        "dataset_loaded_samples": dataset_loaded,
        "dataset_filtered_out_not_in_topk": dataset_filtered,
        "eval_loss": mean_loss,
        "primary_metric": "vqa_soft_accuracy",
        "top1_accuracy": top1,
        "vqa_soft_accuracy": vqa_soft,
        "answer_type_breakdown": answer_type_breakdown,
        "supplementary_text_overlap_metrics": supplementary_text_metrics,
        "predictions": prediction_rows,
        "failures": failure_rows,
    }

    if save_outputs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_dir_path = Path(predictions_dir)
        logs_dir_path = Path(logs_dir)
        predictions_dir_path.mkdir(parents=True, exist_ok=True)
        logs_dir_path.mkdir(parents=True, exist_ok=True)

        predictions_path = predictions_dir_path / f"{file_prefix}_predictions_{split_name}_{timestamp}.jsonl"
        failures_path = predictions_dir_path / f"{file_prefix}_failures_{split_name}_{timestamp}.jsonl"
        metrics_path = logs_dir_path / f"{file_prefix}_metrics_{split_name}_{timestamp}.json"

        save_jsonl(predictions_path, prediction_rows)
        save_jsonl(failures_path, failure_rows[:qualitative_limit])

        metrics_summary = {
            key: value
            for key, value in outputs.items()
            if key not in {"predictions", "failures"}
        }
        metrics_summary["predictions_file"] = str(predictions_path)
        metrics_summary["failures_file"] = str(failures_path)

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics_summary, file, indent=2)

        outputs["predictions_file"] = str(predictions_path)
        outputs["failures_file"] = str(failures_path)
        outputs["metrics_file"] = str(metrics_path)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Version-1 classical multi-hop VQA baseline")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--config", type=str, default="configs/version1_vqa.json")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--split-name", type=str, default="val")
    return parser.parse_args()


def _load_annotation_map(annotation_path: str) -> Dict[int, dict]:
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("Invalid annotation format: expected key 'annotations' with a list")

    annotation_map: Dict[int, dict] = {}
    for entry in annotations:
        question_id = entry.get("question_id")
        if question_id is None:
            raise ValueError("Invalid annotation format: annotation missing 'question_id'")
        annotation_map[int(question_id)] = entry
    return annotation_map


def _resolve_image_path(images_dir: Path, image_id: int) -> str:
    train_style = images_dir / f"COCO_train2014_{image_id:012d}.jpg"
    if train_style.exists():
        return str(train_style)
    val_style = images_dir / f"COCO_val2014_{image_id:012d}.jpg"
    if val_style.exists():
        return str(val_style)
    return str(train_style)


def main() -> None:
    args = parse_args()
    config = V1Config.from_json(args.config)
    resolved_encoder_type = resolve_question_encoder_type(config)
    device = resolve_device(config.device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(config.checkpoints_dir) / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    question_vocab = checkpoint["question_vocab"]
    answer_vocab = checkpoint["answer_vocab"]
    print(f"eval_answer_top_k_config={config.answer_top_k}")
    print(f"eval_checkpoint_answer_vocab_size={len(answer_vocab)}")

    vocab = Vocab(question_vocab)
    answer_to_idx = {answer: index for index, answer in enumerate(answer_vocab)}

    if args.synthetic:
        dataset = SyntheticVQADataset(
            size=min(config.val_subset_size, 128),
            question_length=config.question_max_length,
            answer_vocab_size=len(answer_vocab),
            use_precomputed_features=config.use_precomputed_features,
            image_size=config.image_size,
        )
    else:
        dataset = VQAv2Dataset(
            questions_path=config.val_questions_path,
            annotations_path=config.val_annotations_path,
            images_dir=config.val_images_dir,
            vocab=vocab,
            answer_to_idx=answer_to_idx,
            max_question_length=config.question_max_length,
            image_size=config.image_size,
            max_samples=config.val_subset_size,
            use_precomputed_features=config.use_precomputed_features,
            precomputed_features_dir=config.precomputed_features_dir,
        )

    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=vqa_collate_fn)

    annotation_map: Dict[int, dict] | None = None
    images_dir = Path(config.val_images_dir)
    if not args.synthetic:
        annotation_map = _load_annotation_map(config.val_annotations_path)

    model = MultiHopVQAModel(
        question_vocab_size=len(vocab),
        answer_vocab_size=len(answer_vocab),
        embedding_dim=config.word_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_hops=config.num_hops,
        rnn_type=resolved_encoder_type,
        bidirectional=config.bidirectional,
        freeze_image_encoder=config.freeze_image_encoder,
        use_precomputed_features=config.use_precomputed_features,
        pad_idx=vocab.pad_idx,
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint into evaluation model. "
            "Check that config model flags (especially use_precomputed_features) match the checkpoint training setup."
        ) from exc
    model.eval()

    criterion = nn.CrossEntropyLoss()
    outputs = evaluate_model_on_loader(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        answer_vocab=answer_vocab,
        split_name=args.split_name,
        synthetic=args.synthetic,
        annotation_path=None if args.synthetic else config.val_annotations_path,
        images_dir=None if args.synthetic else config.val_images_dir,
        qualitative_limit=config.qualitative_failures_to_save,
        save_outputs=True,
        predictions_dir=config.predictions_dir,
        logs_dir=config.logs_dir,
        file_prefix="eval",
    )

    print(f"sample_count={outputs['sample_count']}")
    print(f"eval_loss={outputs['eval_loss']:.4f}")
    print(f"top1_accuracy={outputs['top1_accuracy']:.4f}")
    print(f"vqa_soft_accuracy={outputs['vqa_soft_accuracy']:.4f}")
    print("answer_type_breakdown=")
    for answer_type, values in outputs["answer_type_breakdown"]["types"].items():
        print(
            f"  {answer_type}: count={values['count']} "
            f"exact={values['exact_match_accuracy']:.4f} soft={values['vqa_soft_accuracy']:.4f}"
        )
    print("supplementary_text_overlap_metrics=")
    print(f"  bleu_1={outputs['supplementary_text_overlap_metrics']['bleu_1']:.4f}")
    print(f"  bleu_2={outputs['supplementary_text_overlap_metrics']['bleu_2']:.4f}")
    print(f"  bleu_4={outputs['supplementary_text_overlap_metrics']['bleu_4']:.4f}")
    print(f"  rouge_l={outputs['supplementary_text_overlap_metrics']['rouge_l']:.4f}")
    print(f"  meteor={outputs['supplementary_text_overlap_metrics']['meteor']:.4f}")
    print("note=vqa_soft_accuracy remains the primary benchmark metric")

    print(f"Saved predictions to {outputs['predictions_file']}")
    print(f"Saved qualitative failures to {outputs['failures_file']}")
    print(f"Saved metrics summary to {outputs['metrics_file']}")


if __name__ == "__main__":
    main()
