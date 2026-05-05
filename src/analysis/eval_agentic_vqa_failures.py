from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from src.agents.agentic_vqa import AgenticVQAConfig, AgenticVQASystem
from src.utils.metrics import (
    compute_vqa_soft_accuracy,
    exact_match_after_normalization,
    normalize_answer_for_eval,
    save_jsonl,
)


@dataclass
class TrackCFailureEvalConfig:
    track_b_failures_jsonl: str
    model_name_or_path: str
    device: str = "cuda"
    generation_max_new_tokens: int = 8
    generation_do_sample: bool = False
    generation_num_beams: int = 1
    max_cases: int | None = None
    route_toggles: Dict[str, bool] | None = None
    prompt_templates: Dict[str, str] | None = None
    use_verifier: bool | None = None
    verifier_enabled: bool = True
    verifier_routes: List[str] | None = None
    verifier_prompt_template: str | None = None
    fallback_images_dir: str | None = None
    logs_dir: str = "outputs/logs"
    predictions_dir: str = "outputs/predictions"
    split_name: str = "track_b_failures"
    save_outputs: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "TrackCFailureEvalConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if "use_verifier" in payload and "verifier_enabled" not in payload:
            payload["verifier_enabled"] = bool(payload["use_verifier"])

        return cls(**payload)


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_image_path(row: Dict[str, Any], fallback_images_dir: str | None) -> Path:
    image_path = row.get("image_path")
    if isinstance(image_path, str) and image_path.strip():
        candidate = Path(image_path)
        if candidate.exists():
            return candidate

    image_id = int(row.get("image_id"))
    if fallback_images_dir:
        fallback_dir = Path(fallback_images_dir)
        candidate = fallback_dir / f"COCO_val2014_{image_id:012d}.jpg"
        if candidate.exists():
            return candidate
        candidate = fallback_dir / f"COCO_train2014_{image_id:012d}.jpg"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image for question_id={row.get('question_id')} image_id={row.get('image_id')}. "
        "Provide valid image_path rows or set fallback_images_dir in config."
    )


def _safe_soft(pred_answer: str, all_gold_answers: List[str]) -> float:
    return float(compute_vqa_soft_accuracy([pred_answer], [all_gold_answers]))


def _safe_exact(pred_answer: str, majority_answer: str) -> bool:
    return bool(exact_match_after_normalization(pred_answer, majority_answer))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _build_markdown_summary(metrics: Dict[str, Any], split_name: str, timestamp: str) -> str:
    return "\n".join(
        [
            "# Track C Failure-Case Paired Evaluation Summary",
            "",
            f"- Split: `{split_name}`",
            f"- Timestamp: `{timestamp}`",
            "",
            "## Headline Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| total_rows_loaded | {metrics['total_rows_loaded']} |",
            f"| total_rows_evaluated | {metrics['total_rows_evaluated']} |",
            f"| sample_count | {metrics['sample_count']} |",
            f"| track_b_exact_accuracy | {metrics['track_b_exact_accuracy']:.6f} |",
            f"| track_c_exact_accuracy | {metrics['track_c_exact_accuracy']:.6f} |",
            f"| average_exact_delta | {metrics['average_exact_delta']:.6f} |",
            f"| track_b_vqa_soft_accuracy | {metrics['track_b_vqa_soft_accuracy']:.6f} |",
            f"| track_c_vqa_soft_accuracy | {metrics['track_c_vqa_soft_accuracy']:.6f} |",
            f"| failure_recovery_count | {metrics['failure_recovery_count']} |",
            f"| failure_recovery_rate | {metrics['failure_recovery_rate']:.6f} |",
            f"| average_vqa_soft_delta | {metrics['average_vqa_soft_delta']:.6f} |",
            "",
            "## Comparison Counts",
            "",
            f"- improved_soft_count: `{metrics['improved_soft_count']}`",
            f"- tied_soft_count: `{metrics['tied_soft_count']}`",
            f"- worsened_soft_count: `{metrics['worsened_soft_count']}`",
            "",
            "## Route Usage",
            "",
            "```json",
            json.dumps(metrics["route_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Selection Usage",
            "",
            "```json",
            json.dumps(metrics["selection_counts"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )


def evaluate_failures(config: TrackCFailureEvalConfig, max_cases: int | None = None, split_name: str | None = None) -> Dict[str, Any]:
    all_failures = _load_jsonl(config.track_b_failures_jsonl)
    total_rows_loaded = len(all_failures)

    effective_max_cases = max_cases if (max_cases is not None and max_cases > 0) else config.max_cases
    if effective_max_cases is not None and effective_max_cases > 0:
        failures = all_failures[:effective_max_cases]
    else:
        failures = all_failures

    active_split = split_name or config.split_name

    effective_use_verifier = config.use_verifier if config.use_verifier is not None else config.verifier_enabled

    agentic_config = AgenticVQAConfig(
        model_name_or_path=config.model_name_or_path,
        device=config.device,
        generation_max_new_tokens=config.generation_max_new_tokens,
        generation_do_sample=config.generation_do_sample,
        generation_num_beams=config.generation_num_beams,
        route_toggles=config.route_toggles,
        prompt_templates=config.prompt_templates,
        verifier_enabled=bool(effective_use_verifier),
        verifier_routes=config.verifier_routes,
        verifier_prompt_template=config.verifier_prompt_template,
    )
    system = AgenticVQASystem(agentic_config)

    paired_rows: List[Dict[str, Any]] = []
    route_counts: Dict[str, int] = {}
    selection_counts: Dict[str, int] = {}

    total_b_exact = 0
    total_c_exact = 0
    total_b_soft = 0.0
    total_c_soft = 0.0
    improved_soft_count = 0
    tied_soft_count = 0
    worsened_soft_count = 0
    recovery_count = 0

    for row in failures:
        question = str(row.get("question_text", "")).strip()
        image_path = _resolve_image_path(row, config.fallback_images_dir)
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        base_answer = str(
            row.get("predicted_answer")
            or row.get("track_b_original_answer")
            or ""
        ).strip()

        route_result = system.answer_with_base_or_generate(image=image, question=question, base_answer=base_answer)
        specialist_answer = str(route_result["specialist_answer"]).strip()
        final_answer = str(route_result["final_answer"]).strip()

        all_gold_answers = row.get("all_gold_answers", [])
        if not isinstance(all_gold_answers, list) or len(all_gold_answers) == 0:
            majority_fallback = str(row.get("gold_majority_answer", ""))
            all_gold_answers = [majority_fallback]
        all_gold_answers = [str(answer) for answer in all_gold_answers]

        gold_majority = str(row.get("gold_majority_answer", ""))
        if not gold_majority:
            normalized_gold = [normalize_answer_for_eval(answer) for answer in all_gold_answers]
            gold_majority = normalized_gold[0] if normalized_gold else ""

        base_exact = _safe_exact(base_answer, gold_majority)
        routed_exact = _safe_exact(specialist_answer, gold_majority)
        final_exact = _safe_exact(final_answer, gold_majority)
        base_soft = _safe_soft(base_answer, all_gold_answers)
        routed_soft = _safe_soft(specialist_answer, all_gold_answers)
        final_soft = _safe_soft(final_answer, all_gold_answers)

        if final_soft > base_soft:
            improved_soft_count += 1
        elif final_soft < base_soft:
            worsened_soft_count += 1
        else:
            tied_soft_count += 1

        if (not base_exact) and final_exact:
            recovery_count += 1

        total_b_exact += int(base_exact)
        total_c_exact += int(final_exact)
        total_b_soft += base_soft
        total_c_soft += final_soft

        route = str(route_result["route"])
        selected_from = str(route_result["selector_choice"])
        route_counts[route] = route_counts.get(route, 0) + 1
        selection_counts[selected_from] = selection_counts.get(selected_from, 0) + 1

        paired_rows.append(
            {
                "question_id": row.get("question_id"),
                "image_id": row.get("image_id"),
                "question_text": question,
                "answer_type": row.get("answer_type"),
                "all_gold_answers": all_gold_answers,
                "gold_majority_answer": gold_majority,
                "base_answer": base_answer,
                "specialist_answer": specialist_answer,
                "final_answer": final_answer,
                "selector_choice": selected_from,
                "selector_reason": route_result.get("selector_reason"),
                "base_exact": base_exact,
                "routed_exact": routed_exact,
                "final_exact": final_exact,
                "base_vqa_soft_score": base_soft,
                "routed_vqa_soft_score": routed_soft,
                "final_vqa_soft_score": final_soft,
                "routed_delta_vs_base": routed_soft - base_soft,
                "final_delta_vs_base": final_soft - base_soft,
                "failure_recovered": (not base_exact) and final_exact,
                "route": route,
                "route_rule": route_result.get("route_rule"),
                "route_prompt": route_result.get("route_prompt"),
                "specialist_agent": route_result.get("specialist_agent"),
                "specialist_prompt": route_result.get("specialist_prompt"),
                "selected_from": selected_from,
            }
        )

    sample_count = len(paired_rows)
    if sample_count == 0:
        raise ValueError("No samples found for Track C failure-case evaluation.")

    metrics = {
        "track": "track_c_agentic_vqa_v1",
        "split": active_split,
        "total_rows_loaded": total_rows_loaded,
        "total_rows_evaluated": sample_count,
        "sample_count": sample_count,
        "track_b_exact_accuracy": total_b_exact / sample_count,
        "track_c_exact_accuracy": total_c_exact / sample_count,
        "average_exact_delta": (total_c_exact - total_b_exact) / sample_count,
        "track_b_vqa_soft_accuracy": total_b_soft / sample_count,
        "track_c_vqa_soft_accuracy": total_c_soft / sample_count,
        "failure_recovery_count": recovery_count,
        "failure_recovery_rate": recovery_count / sample_count,
        "average_vqa_soft_delta": (total_c_soft - total_b_soft) / sample_count,
        "improved_soft_count": improved_soft_count,
        "tied_soft_count": tied_soft_count,
        "worsened_soft_count": worsened_soft_count,
        "route_counts": route_counts,
        "selection_counts": selection_counts,
        "route_toggles": dict(config.route_toggles or {}),
        "input_failure_jsonl": str(config.track_b_failures_jsonl),
        "model_name_or_path": config.model_name_or_path,
        "use_verifier": bool(effective_use_verifier),
        "verifier_enabled": bool(effective_use_verifier),
        "verifier_routes": list(config.verifier_routes or []),
        "max_cases": effective_max_cases,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.save_outputs:
        predictions_dir = Path(config.predictions_dir)
        logs_dir = Path(config.logs_dir)
        predictions_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        paired_path = predictions_dir / f"track_c_agentic_paired_failures_{active_split}_{timestamp}.jsonl"
        metrics_path = logs_dir / f"track_c_agentic_metrics_failures_{active_split}_{timestamp}.json"
        summary_md_path = logs_dir / f"track_c_agentic_summary_failures_{active_split}_{timestamp}.md"

        save_jsonl(paired_path, paired_rows)
        _save_json(metrics_path, metrics)
        summary_md_path.write_text(_build_markdown_summary(metrics, active_split, timestamp), encoding="utf-8")

        metrics["paired_rows_path"] = str(paired_path)
        metrics["metrics_path"] = str(metrics_path)
        metrics["summary_md_path"] = str(summary_md_path)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Track C agentic wrapper on existing Track B failure cases")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/track_c_agentic_vqa_failures.json",
        help="Path to Track C failure evaluation config",
    )
    parser.add_argument("--max-cases", type=int, default=None, help="Optional max rows to evaluate from failures JSONL")
    parser.add_argument("--max-samples", type=int, default=None, help="Deprecated alias for --max-cases")
    parser.add_argument("--split-name", type=str, default=None, help="Override split name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrackCFailureEvalConfig.from_json(args.config)
    effective_max_cases = args.max_cases if args.max_cases is not None else args.max_samples
    metrics = evaluate_failures(config=config, max_cases=effective_max_cases, split_name=args.split_name)

    print("\n=== Track C Failure-Case Paired Evaluation ===")
    print(f"Rows loaded: {metrics['total_rows_loaded']}")
    print(f"Rows evaluated: {metrics['total_rows_evaluated']}")
    print(f"Samples: {metrics['sample_count']}")
    print(f"Track B exact: {metrics['track_b_exact_accuracy']:.4f}")
    print(f"Track C exact: {metrics['track_c_exact_accuracy']:.4f}")
    print(f"Avg exact delta (C-B): {metrics['average_exact_delta']:.4f}")
    print(f"Track B soft:  {metrics['track_b_vqa_soft_accuracy']:.4f}")
    print(f"Track C soft:  {metrics['track_c_vqa_soft_accuracy']:.4f}")
    print(f"Recovery count: {metrics['failure_recovery_count']}")
    print(f"Recovery rate:  {metrics['failure_recovery_rate']:.4f}")
    print(f"Avg soft delta (C-B): {metrics['average_vqa_soft_delta']:.4f}")
    print(f"Improved/Tied/Worsened soft: {metrics['improved_soft_count']}/{metrics['tied_soft_count']}/{metrics['worsened_soft_count']}")
    print(f"Route distribution: {json.dumps(metrics['route_counts'], ensure_ascii=False)}")

    if "paired_rows_path" in metrics:
        print(f"Paired rows: {metrics['paired_rows_path']}")
        print(f"Metrics JSON: {metrics['metrics_path']}")
        print(f"Summary MD: {metrics['summary_md_path']}")


if __name__ == "__main__":
    main()
