from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.utils.metrics import (
    compute_answer_type_breakdown,
    compute_supplementary_text_overlap_metrics,
    compute_vqa_soft_accuracy,
    exact_match_after_normalization,
    normalize_answer_for_eval,
    save_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two VQA prediction JSONL files on the exact shared question_id set"
    )
    parser.add_argument("--classical-predictions", type=str, required=True)
    parser.add_argument("--vlm-predictions", type=str, required=True)
    parser.add_argument("--split-name", type=str, default="paired")
    parser.add_argument("--output-dir", type=str, default="outputs/logs")
    parser.add_argument("--output-prefix", type=str, default="paired_compare_track_a_vs_track_b")
    return parser.parse_args()


def _load_jsonl(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text == "":
                continue
            rows.append(json.loads(text))
    return rows


def _index_by_question_id(rows: List[dict]) -> Dict[int, dict]:
    index: Dict[int, dict] = {}
    for row in rows:
        question_id = row.get("question_id")
        if question_id is None:
            raise ValueError("Prediction row missing 'question_id'")
        key = int(question_id)
        if key in index:
            raise ValueError(f"Duplicate question_id detected: {key}")
        index[key] = row
    return index


def _safe_all_gold(row: dict) -> List[str]:
    all_gold = row.get("all_gold_answers", [])
    if not isinstance(all_gold, list) or len(all_gold) == 0:
        majority = normalize_answer_for_eval(str(row.get("gold_majority_answer", "")))
        return [majority]
    return [normalize_answer_for_eval(str(item)) for item in all_gold]


def _safe_majority(row: dict) -> str:
    return normalize_answer_for_eval(str(row.get("gold_majority_answer", "")))


def _safe_answer_type(row: dict) -> str:
    value = row.get("answer_type", "")
    return str(value) if isinstance(value, str) else ""


def _compute_model_metrics(pred_answers: List[str], answer_type_records: List[dict]) -> Dict[str, Any]:
    all_gold_answers = [record["all_gold_answers"] for record in answer_type_records]
    sample_count = len(pred_answers)

    exact_top1 = (
        sum(
            1
            for pred, record in zip(pred_answers, answer_type_records)
            if exact_match_after_normalization(pred, record["gold_majority_answer"])
        )
        / sample_count
        if sample_count > 0
        else 0.0
    )
    vqa_soft = compute_vqa_soft_accuracy(pred_answers, all_gold_answers)
    answer_type_breakdown = compute_answer_type_breakdown(
        pred_answers=pred_answers,
        gold_answers_or_annotations=answer_type_records,
    )
    supplementary = compute_supplementary_text_overlap_metrics(pred_answers, all_gold_answers)

    return {
        "sample_count": sample_count,
        "top1_accuracy": exact_top1,
        "vqa_soft_accuracy": vqa_soft,
        "answer_type_breakdown": answer_type_breakdown,
        "supplementary_text_overlap_metrics": supplementary,
    }


def _build_markdown_report(summary: Dict[str, Any]) -> str:
    cls = summary["classical_metrics"]
    vlm = summary["vlm_metrics"]
    paired = summary["paired_set"]
    delta_top1 = vlm["top1_accuracy"] - cls["top1_accuracy"]
    delta_soft = vlm["vqa_soft_accuracy"] - cls["vqa_soft_accuracy"]

    return "\n".join(
        [
            "# Paired Comparison — Track A vs Track B",
            "",
            "## Inputs",
            f"- Classical predictions: `{summary['inputs']['classical_predictions']}`",
            f"- VLM predictions: `{summary['inputs']['vlm_predictions']}`",
            "",
            "## Paired Set",
            f"- Classical total rows: `{paired['classical_total']}`",
            f"- VLM total rows: `{paired['vlm_total']}`",
            f"- Shared question_ids: `{paired['shared_count']}`",
            f"- Only classical: `{paired['only_classical_count']}`",
            f"- Only VLM: `{paired['only_vlm_count']}`",
            "",
            "## Headline Metrics",
            "| Metric | Classical | VLM | Delta (VLM-Classical) |",
            "|---|---:|---:|---:|",
            f"| top1_accuracy | {cls['top1_accuracy']:.6f} | {vlm['top1_accuracy']:.6f} | {delta_top1:+.6f} |",
            f"| vqa_soft_accuracy | {cls['vqa_soft_accuracy']:.6f} | {vlm['vqa_soft_accuracy']:.6f} | {delta_soft:+.6f} |",
            "",
            "## Pairwise Outcome (Shared IDs)",
            f"- VLM better by soft score: `{summary['pairwise_outcome']['vlm_better_soft_count']}`",
            f"- Classical better by soft score: `{summary['pairwise_outcome']['classical_better_soft_count']}`",
            f"- Tied soft score: `{summary['pairwise_outcome']['tied_soft_count']}`",
            f"- VLM better by exact: `{summary['pairwise_outcome']['vlm_better_exact_count']}`",
            f"- Classical better by exact: `{summary['pairwise_outcome']['classical_better_exact_count']}`",
            f"- Tied exact: `{summary['pairwise_outcome']['tied_exact_count']}`",
            "",
            "## Files",
            f"- JSON summary: `{summary['artifacts']['summary_json']}`",
            f"- Paired rows JSONL: `{summary['artifacts']['paired_rows_jsonl']}`",
            f"- Markdown report: `{summary['artifacts']['report_md']}`",
        ]
    )


def main() -> None:
    args = parse_args()

    classical_rows = _load_jsonl(args.classical_predictions)
    vlm_rows = _load_jsonl(args.vlm_predictions)

    classical_map = _index_by_question_id(classical_rows)
    vlm_map = _index_by_question_id(vlm_rows)

    classical_ids = set(classical_map.keys())
    vlm_ids = set(vlm_map.keys())
    shared_ids = sorted(classical_ids.intersection(vlm_ids))
    only_classical_ids = sorted(classical_ids.difference(vlm_ids))
    only_vlm_ids = sorted(vlm_ids.difference(classical_ids))

    if len(shared_ids) == 0:
        raise ValueError("No shared question_id values found between the two prediction files")

    answer_type_records: List[dict] = []
    classical_pred_answers: List[str] = []
    vlm_pred_answers: List[str] = []
    paired_rows: List[dict] = []

    vlm_better_soft_count = 0
    classical_better_soft_count = 0
    tied_soft_count = 0

    vlm_better_exact_count = 0
    classical_better_exact_count = 0
    tied_exact_count = 0

    for question_id in shared_ids:
        classical_row = classical_map[question_id]
        vlm_row = vlm_map[question_id]

        gold_majority = _safe_majority(classical_row)
        all_gold_answers = _safe_all_gold(classical_row)
        answer_type = _safe_answer_type(classical_row)

        classical_pred = normalize_answer_for_eval(str(classical_row.get("predicted_answer", "")))
        vlm_pred = normalize_answer_for_eval(str(vlm_row.get("predicted_answer", "")))

        classical_soft = compute_vqa_soft_accuracy([classical_pred], [all_gold_answers])
        vlm_soft = compute_vqa_soft_accuracy([vlm_pred], [all_gold_answers])

        classical_exact = exact_match_after_normalization(classical_pred, gold_majority)
        vlm_exact = exact_match_after_normalization(vlm_pred, gold_majority)

        if vlm_soft > classical_soft:
            vlm_better_soft_count += 1
        elif classical_soft > vlm_soft:
            classical_better_soft_count += 1
        else:
            tied_soft_count += 1

        if int(vlm_exact) > int(classical_exact):
            vlm_better_exact_count += 1
        elif int(classical_exact) > int(vlm_exact):
            classical_better_exact_count += 1
        else:
            tied_exact_count += 1

        answer_type_records.append(
            {
                "gold_majority_answer": gold_majority,
                "all_gold_answers": all_gold_answers,
                "answer_type": answer_type,
            }
        )
        classical_pred_answers.append(classical_pred)
        vlm_pred_answers.append(vlm_pred)

        paired_rows.append(
            {
                "question_id": question_id,
                "question_text": classical_row.get("question_text", vlm_row.get("question_text", "")),
                "answer_type": answer_type,
                "gold_majority_answer": gold_majority,
                "all_gold_answers": all_gold_answers,
                "classical_predicted_answer": classical_pred,
                "vlm_predicted_answer": vlm_pred,
                "classical_exact_match": bool(classical_exact),
                "vlm_exact_match": bool(vlm_exact),
                "classical_vqa_soft_score": classical_soft,
                "vlm_vqa_soft_score": vlm_soft,
            }
        )

    classical_metrics = _compute_model_metrics(classical_pred_answers, answer_type_records)
    vlm_metrics = _compute_model_metrics(vlm_pred_answers, answer_type_records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / f"{args.output_prefix}_metrics_{args.split_name}_{timestamp}.json"
    paired_rows_jsonl_path = output_dir / f"{args.output_prefix}_paired_rows_{args.split_name}_{timestamp}.jsonl"
    report_md_path = output_dir / f"{args.output_prefix}_report_{args.split_name}_{timestamp}.md"

    summary: Dict[str, Any] = {
        "split": args.split_name,
        "inputs": {
            "classical_predictions": args.classical_predictions,
            "vlm_predictions": args.vlm_predictions,
        },
        "paired_set": {
            "classical_total": len(classical_rows),
            "vlm_total": len(vlm_rows),
            "shared_count": len(shared_ids),
            "only_classical_count": len(only_classical_ids),
            "only_vlm_count": len(only_vlm_ids),
            "only_classical_question_ids": only_classical_ids,
            "only_vlm_question_ids": only_vlm_ids,
        },
        "pairwise_outcome": {
            "vlm_better_soft_count": vlm_better_soft_count,
            "classical_better_soft_count": classical_better_soft_count,
            "tied_soft_count": tied_soft_count,
            "vlm_better_exact_count": vlm_better_exact_count,
            "classical_better_exact_count": classical_better_exact_count,
            "tied_exact_count": tied_exact_count,
        },
        "classical_metrics": classical_metrics,
        "vlm_metrics": vlm_metrics,
        "artifacts": {
            "summary_json": str(summary_json_path),
            "paired_rows_jsonl": str(paired_rows_jsonl_path),
            "report_md": str(report_md_path),
        },
    }

    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_jsonl(paired_rows_jsonl_path, paired_rows)

    report_text = _build_markdown_report(summary)
    with report_md_path.open("w", encoding="utf-8") as file:
        file.write(report_text + "\n")

    print(f"paired_shared_count={summary['paired_set']['shared_count']}")
    print(f"classical_top1={classical_metrics['top1_accuracy']:.6f}")
    print(f"classical_vqa_soft={classical_metrics['vqa_soft_accuracy']:.6f}")
    print(f"vlm_top1={vlm_metrics['top1_accuracy']:.6f}")
    print(f"vlm_vqa_soft={vlm_metrics['vqa_soft_accuracy']:.6f}")
    print(f"Saved summary JSON to {summary_json_path}")
    print(f"Saved paired rows JSONL to {paired_rows_jsonl_path}")
    print(f"Saved markdown report to {report_md_path}")


if __name__ == "__main__":
    main()
