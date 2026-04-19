from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.vqa_vlm_eval_dataset import VQAv2VLMEvalDataset, vqa_vlm_eval_collate_fn
from src.models.vlm import PaliGemmaRunner
from src.models.vlm.paligemma_runner import GenerationConfig
from src.utils.metrics import (
    compute_answer_type_breakdown,
    compute_supplementary_text_overlap_metrics,
    compute_vqa_soft_accuracy,
    exact_match_after_normalization,
    normalize_answer_for_eval,
    save_jsonl,
)


@dataclass
class TrackBVLMEvalConfig:
    seed: int = 42
    device: str = "cuda"

    model_name_or_path: str = "google/paligemma-3b-ft-vqav2-224"
    prompt_template: str = "<image> answer {question}"

    generation_max_new_tokens: int = 8
    generation_do_sample: bool = False
    generation_num_beams: int = 1

    batch_size: int = 1
    num_workers: int = 0
    val_subset_size: int = 64

    val_questions_path: str = "datasets/train2014/v2_OpenEnded_mscoco_val2014_questions.json"
    val_annotations_path: str = "datasets/train2014/v2_mscoco_val2014_annotations.json"
    val_images_dir: str = "datasets/train2014/val2014"

    logs_dir: str = "outputs/logs"
    predictions_dir: str = "outputs/predictions"
    qualitative_failures_to_save: int = 100
    save_outputs: bool = True
    compute_supplementary_text_overlap: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "TrackBVLMEvalConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        return cls(**payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Track-B pretrained VLM on VQAv2 subset")
    parser.add_argument("--config", type=str, default="configs/track_b_paligemma_vqav2_eval.json")
    parser.add_argument("--subset-val", type=int, default=0)
    parser.add_argument("--split-name", type=str, default="val")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_vlm_on_loader(
    runner: PaliGemmaRunner,
    loader: DataLoader,
    split_name: str,
    qualitative_limit: int,
    save_outputs: bool,
    predictions_dir: str,
    logs_dir: str,
    file_prefix: str,
    compute_supplementary_text_overlap: bool,
) -> Dict[str, Any]:
    all_pred_answers: List[str] = []
    all_gold_answers: List[List[str]] = []
    all_answer_type_records: List[dict] = []
    prediction_rows: List[dict] = []
    failure_rows: List[dict] = []

    for batch in loader:
        generated_answers = runner.generate_answers(images=batch["images"], questions=batch["question_texts"])

        for index, generated_answer in enumerate(generated_answers):
            raw_pred_answer = generated_answer.strip()
            pred_answer = normalize_answer_for_eval(raw_pred_answer)
            gold_majority_answer = normalize_answer_for_eval(batch["gold_majority_answers"][index])
            all_gold = [normalize_answer_for_eval(item) for item in batch["all_gold_answers"][index]]
            answer_type = batch["answer_types"][index]

            row = {
                "question_id": int(batch["question_ids"][index]),
                "question_text": batch["question_texts"][index],
                "predicted_answer": pred_answer,
                "predicted_answer_raw": raw_pred_answer,
                "gold_majority_answer": gold_majority_answer,
                "all_gold_answers": all_gold,
                "answer_type": answer_type,
                "image_id": int(batch["image_ids"][index]),
                "image_path": batch["image_paths"][index],
                "exact_match_correct": exact_match_after_normalization(pred_answer, gold_majority_answer),
                "vqa_soft_score": compute_vqa_soft_accuracy([pred_answer], [all_gold]),
            }
            prediction_rows.append(row)

            if not row["exact_match_correct"]:
                failure_rows.append(row)

            all_pred_answers.append(pred_answer)
            all_gold_answers.append(all_gold)
            all_answer_type_records.append(
                {
                    "gold_majority_answer": gold_majority_answer,
                    "all_gold_answers": all_gold,
                    "answer_type": answer_type,
                }
            )

    sample_count = len(all_pred_answers)
    exact_top1 = (
        sum(1 for pred, record in zip(all_pred_answers, all_answer_type_records) if exact_match_after_normalization(pred, record["gold_majority_answer"]))
        / sample_count
        if sample_count > 0
        else 0.0
    )
    vqa_soft = compute_vqa_soft_accuracy(all_pred_answers, all_gold_answers)
    answer_type_breakdown = compute_answer_type_breakdown(
        pred_answers=all_pred_answers,
        gold_answers_or_annotations=all_answer_type_records,
    )

    supplementary_text_metrics = (
        compute_supplementary_text_overlap_metrics(pred_answers=all_pred_answers, all_gold_answers=all_gold_answers)
        if compute_supplementary_text_overlap
        else {}
    )

    dataset = loader.dataset
    outputs: Dict[str, Any] = {
        "split": split_name,
        "track": "track_b_vlm",
        "model_name_or_path": runner.model.name_or_path,
        "sample_count": sample_count,
        "dataset_loaded_samples": getattr(dataset, "stats", {}).get("loaded", len(dataset)),
        "dataset_filtered_out_not_in_topk": getattr(dataset, "stats", {}).get("skipped_not_in_topk", 0),
        "primary_metric": "vqa_soft_accuracy",
        "top1_accuracy": exact_top1,
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

        metrics_summary = {key: value for key, value in outputs.items() if key not in {"predictions", "failures"}}
        metrics_summary["predictions_file"] = str(predictions_path)
        metrics_summary["failures_file"] = str(failures_path)

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics_summary, file, indent=2)

        outputs["predictions_file"] = str(predictions_path)
        outputs["failures_file"] = str(failures_path)
        outputs["metrics_file"] = str(metrics_path)

    return outputs


def main() -> None:
    args = parse_args()
    config = TrackBVLMEvalConfig.from_json(args.config)

    if args.subset_val > 0:
        config.val_subset_size = int(args.subset_val)

    set_seed(config.seed)
    device = resolve_device(config.device)

    dataset = VQAv2VLMEvalDataset(
        questions_path=config.val_questions_path,
        annotations_path=config.val_annotations_path,
        images_dir=config.val_images_dir,
        max_samples=config.val_subset_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=vqa_vlm_eval_collate_fn,
    )

    runner = PaliGemmaRunner(
        model_name_or_path=config.model_name_or_path,
        device=device,
        generation=GenerationConfig(
            max_new_tokens=config.generation_max_new_tokens,
            do_sample=config.generation_do_sample,
            num_beams=config.generation_num_beams,
        ),
        prompt_template=config.prompt_template,
    )

    outputs = evaluate_vlm_on_loader(
        runner=runner,
        loader=loader,
        split_name=args.split_name,
        qualitative_limit=config.qualitative_failures_to_save,
        save_outputs=config.save_outputs,
        predictions_dir=config.predictions_dir,
        logs_dir=config.logs_dir,
        file_prefix="eval_track_b_paligemma",
        compute_supplementary_text_overlap=config.compute_supplementary_text_overlap,
    )

    print(f"sample_count={outputs['sample_count']}")
    print(f"top1_accuracy={outputs['top1_accuracy']:.4f}")
    print(f"vqa_soft_accuracy={outputs['vqa_soft_accuracy']:.4f}")
    print("answer_type_breakdown=")
    for answer_type, values in outputs["answer_type_breakdown"]["types"].items():
        print(
            f"  {answer_type}: count={values['count']} "
            f"exact={values['exact_match_accuracy']:.4f} soft={values['vqa_soft_accuracy']:.4f}"
        )
    if outputs["supplementary_text_overlap_metrics"]:
        print("supplementary_text_overlap_metrics=")
        print(f"  bleu_1={outputs['supplementary_text_overlap_metrics']['bleu_1']:.4f}")
        print(f"  bleu_2={outputs['supplementary_text_overlap_metrics']['bleu_2']:.4f}")
        print(f"  bleu_4={outputs['supplementary_text_overlap_metrics']['bleu_4']:.4f}")
        print(f"  rouge_l={outputs['supplementary_text_overlap_metrics']['rouge_l']:.4f}")
        print(f"  meteor={outputs['supplementary_text_overlap_metrics']['meteor']:.4f}")

    if "predictions_file" in outputs:
        print(f"Saved predictions to {outputs['predictions_file']}")
        print(f"Saved qualitative failures to {outputs['failures_file']}")
        print(f"Saved metrics summary to {outputs['metrics_file']}")


if __name__ == "__main__":
    main()
