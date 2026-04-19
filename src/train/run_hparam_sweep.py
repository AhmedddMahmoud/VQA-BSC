from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import torch

from src.utils.config import V1Config

FIXED_DROPOUT = 0.2


def parse_float_list(raw: str) -> List[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) == 0:
        return []
    return [float(value) for value in values]


def parse_int_list(raw: str) -> List[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) == 0:
        return []
    return [int(value) for value in values]


def to_float(value: Any, default: float = -1.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def format_lr(value: float) -> str:
    if value == 0:
        return "0"
    text = f"{value:.0e}"
    return text.replace("+", "").replace("e-0", "e-")


def find_latest_matching(logs_dir: Path, pattern: str) -> str:
    matches = sorted(logs_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if len(matches) == 0:
        return ""
    return str(matches[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small, laptop-safe hyperparameter sweep for Version-1 VQA")
    parser.add_argument("--base-config", type=str, default="configs/version1_real_train_subset.json")
    parser.add_argument("--sweep-root", type=str, default="outputs/sweeps")
    parser.add_argument("--sweep-name", type=str, default="")

    parser.add_argument("--learning-rates", type=str, default="1e-3,5e-4,2e-4")
    parser.add_argument("--epochs-list", type=str, default="3,4,6")
    parser.add_argument("--batch-sizes", type=str, default="")

    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--quick", action="store_true")

    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--val-subset", type=int, default=0)
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--synthetic", action="store_true")
    return parser.parse_args()


def build_run_grid(
    base_config: V1Config,
    learning_rates: List[float],
    epochs_list: List[int],
    batch_sizes: List[int],
    run_name_filter: str,
    max_runs: int,
) -> List[Dict[str, Any]]:
    if len(batch_sizes) == 0:
        batch_sizes = [base_config.batch_size]

    runs: List[Dict[str, Any]] = []
    for learning_rate, epochs, batch_size in product(learning_rates, epochs_list, batch_sizes):
        run_name = f"lr_{format_lr(learning_rate)}_drop_fixed_0p2_ep_{epochs}_bs_{batch_size}"
        run = {
            "run_name": run_name,
            "learning_rate": learning_rate,
            "dropout": FIXED_DROPOUT,
            "epochs": epochs,
            "batch_size": batch_size,
        }
        runs.append(run)

    if run_name_filter:
        runs = [run for run in runs if run["run_name"] == run_name_filter]

    if max_runs > 0:
        runs = runs[:max_runs]

    return runs


def run_single_experiment(
    run: Dict[str, Any],
    base_config: V1Config,
    sweep_dir: Path,
    train_subset_override: int,
    val_subset_override: int,
    device_override: str,
    synthetic: bool,
) -> Dict[str, Any]:
    run_name = run["run_name"]
    run_dir = sweep_dir / run_name
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    predictions_dir = run_dir / "predictions"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    run_config = V1Config(**asdict(base_config))
    run_config.learning_rate = run["learning_rate"]
    run_config.epochs = run["epochs"]
    run_config.batch_size = run["batch_size"]

    if train_subset_override > 0:
        run_config.train_subset_size = train_subset_override
    if val_subset_override > 0:
        run_config.val_subset_size = val_subset_override
    if device_override:
        run_config.device = device_override

    run_config.output_root = str(run_dir)
    run_config.checkpoints_dir = str(checkpoints_dir)
    run_config.logs_dir = str(logs_dir)
    run_config.predictions_dir = str(predictions_dir)

    config_path = run_dir / "config_used.json"
    run_config.to_json(config_path)

    print("-" * 80)
    print(f"run_start name={run_name}")
    print(f"subset_train={run_config.train_subset_size} subset_val={run_config.val_subset_size}")
    print(f"learning_rate={run_config.learning_rate} dropout_fixed={FIXED_DROPOUT}")
    print(f"epochs={run_config.epochs} batch_size={run_config.batch_size}")
    print(f"freeze_image_encoder={run_config.freeze_image_encoder}")

    command = [sys.executable, "-m", "src.train.train_vqa", "--config", str(config_path)]
    if synthetic:
        command.append("--synthetic")

    completed = subprocess.run(command, check=False)

    summary_row: Dict[str, Any] = {
        "run_name": run_name,
        "status": "success" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "learning_rate": run_config.learning_rate,
        "dropout": FIXED_DROPOUT,
        "epochs": run_config.epochs,
        "batch_size": run_config.batch_size,
        "train_subset_size": run_config.train_subset_size,
        "val_subset_size": run_config.val_subset_size,
        "freeze_image_encoder": run_config.freeze_image_encoder,
        "config_snapshot": str(config_path),
        "best_epoch": "",
        "best_val_soft": "",
        "best_val_top1": "",
        "final_val_soft": "",
        "final_val_top1": "",
        "best_checkpoint_path": str(checkpoints_dir / "best.pt"),
        "latest_checkpoint_path": str(checkpoints_dir / "latest.pt"),
        "metrics_log_path": "",
        "history_json_path": str(logs_dir / "training_history.json"),
        "history_csv_path": str(logs_dir / "training_history.csv"),
        "predictions_path": find_latest_matching(predictions_dir, "final_epoch_predictions_val_final_*.jsonl"),
        "failures_path": find_latest_matching(predictions_dir, "final_epoch_failures_val_final_*.jsonl"),
    }

    if completed.returncode != 0:
        print(f"run_end name={run_name} status=failed return_code={completed.returncode}")
        return summary_row

    history_path = logs_dir / "training_history.json"
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
        if isinstance(history, list) and len(history) > 0:
            last_row = history[-1]
            summary_row["final_val_soft"] = last_row.get("val_vqa_soft_accuracy", "")
            summary_row["final_val_top1"] = last_row.get("val_top1_accuracy", "")

    best_checkpoint_path = checkpoints_dir / "best.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        best_epoch = checkpoint.get("epoch", "")
        summary_row["best_epoch"] = best_epoch
        summary_row["best_val_soft"] = checkpoint.get("best_val_vqa_soft", "")

        if history_path.exists() and summary_row["best_epoch"] != "":
            with history_path.open("r", encoding="utf-8") as file:
                history = json.load(file)
            if isinstance(history, list):
                for row in history:
                    if row.get("epoch") == best_epoch:
                        summary_row["best_val_top1"] = row.get("val_top1_accuracy", "")
                        break

    best_metrics_log = find_latest_matching(logs_dir, "best_epoch_metrics_val_best_*.json")
    final_metrics_log = find_latest_matching(logs_dir, "final_epoch_metrics_val_final_*.json")
    summary_row["metrics_log_path"] = best_metrics_log if best_metrics_log else final_metrics_log

    print(
        f"run_end name={run_name} status=success "
        f"best_epoch={summary_row['best_epoch']} "
        f"best_val_soft={summary_row['best_val_soft']} "
        f"best_checkpoint={summary_row['best_checkpoint_path']}"
    )

    return summary_row


def save_summaries(sweep_dir: Path, rows: List[Dict[str, Any]]) -> tuple[Path, Path]:
    def sort_key(row: Dict[str, Any]) -> tuple[int, float]:
        if row.get("status") != "success":
            return (1, float("-inf"))
        value = to_float(row.get("best_val_soft", ""), default=float("-inf"))
        return (0, value)

    ranked = sorted(rows, key=sort_key, reverse=False)
    successful = [row for row in ranked if row.get("status") == "success"]
    failed = [row for row in ranked if row.get("status") != "success"]
    successful = sorted(successful, key=lambda row: to_float(row.get("best_val_soft", -1.0), default=-1.0), reverse=True)
    ranked_rows = successful + failed

    summary_json = sweep_dir / "sweep_summary.json"
    summary_csv = sweep_dir / "sweep_summary.csv"

    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(ranked_rows, file, indent=2)

    fieldnames = [
        "run_name",
        "status",
        "return_code",
        "learning_rate",
        "dropout",
        "epochs",
        "batch_size",
        "train_subset_size",
        "val_subset_size",
        "freeze_image_encoder",
        "best_epoch",
        "best_val_soft",
        "best_val_top1",
        "final_val_soft",
        "final_val_top1",
        "best_checkpoint_path",
        "metrics_log_path",
        "history_json_path",
        "history_csv_path",
        "predictions_path",
        "failures_path",
        "config_snapshot",
    ]

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return summary_json, summary_csv


def main() -> None:
    args = parse_args()

    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    base_config = V1Config.from_json(base_config_path)

    learning_rates = parse_float_list(args.learning_rates)
    epochs_list = parse_int_list(args.epochs_list)
    batch_sizes = parse_int_list(args.batch_sizes)

    if args.quick:
        learning_rates = [1e-3, 5e-4]
        epochs_list = [3, 4]
        if len(batch_sizes) == 0:
            batch_sizes = [base_config.batch_size]

    if len(learning_rates) == 0:
        raise ValueError("No learning rates provided")
    if len(epochs_list) == 0:
        raise ValueError("No epoch values provided")

    sweep_name = args.sweep_name if args.sweep_name else f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = Path(args.sweep_root) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    run_grid = build_run_grid(
        base_config=base_config,
        learning_rates=learning_rates,
        epochs_list=epochs_list,
        batch_sizes=batch_sizes,
        run_name_filter=args.run_name,
        max_runs=args.max_runs,
    )

    if len(run_grid) == 0:
        raise ValueError("No runs selected. Check --run-name or grid arguments.")

    print("=" * 80)
    print(f"sweep_start name={sweep_name}")
    print(f"base_config={base_config_path}")
    print(f"runs_planned={len(run_grid)}")
    print(f"learning_rates={learning_rates}")
    print(f"epochs_list={epochs_list}")
    print(f"batch_sizes={batch_sizes if batch_sizes else [base_config.batch_size]}")
    print(f"dropout=not_configurable_fixed_{FIXED_DROPOUT}")
    print("execution_mode=sequential")

    rows: List[Dict[str, Any]] = []
    for run in run_grid:
        row = run_single_experiment(
            run=run,
            base_config=base_config,
            sweep_dir=sweep_dir,
            train_subset_override=args.train_subset,
            val_subset_override=args.val_subset,
            device_override=args.device,
            synthetic=args.synthetic,
        )
        rows.append(row)

    summary_json, summary_csv = save_summaries(sweep_dir=sweep_dir, rows=rows)

    successful = [row for row in rows if row.get("status") == "success"]
    ranked = sorted(successful, key=lambda row: to_float(row.get("best_val_soft", -1.0), default=-1.0), reverse=True)

    print("=" * 80)
    print("sweep_ranked_results")
    for rank, row in enumerate(ranked, start=1):
        print(
            f"rank={rank} run_name={row['run_name']} "
            f"best_val_soft={row.get('best_val_soft', '')} "
            f"best_epoch={row.get('best_epoch', '')}"
        )

    print(f"sweep_summary_json={summary_json}")
    print(f"sweep_summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
