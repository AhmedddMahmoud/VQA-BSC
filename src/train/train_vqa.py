from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.vqa_dataset import SyntheticVQADataset, VQAv2Dataset, vqa_collate_fn
from src.models.multihop_vqa import MultiHopVQAModel
from src.train.eval_vqa import evaluate_model_on_loader
from src.utils.config import V1Config, get_default_config
from src.utils.vocab import Vocab, build_answer_vocab, build_question_vocab


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def enforce_v1_contracts(config: V1Config) -> None:
    if config.answer_top_k <= 0:
        raise ValueError("Version 1 contract: answer_top_k must be positive.")
    if config.answer_vocab_size <= 0:
        raise ValueError("Version 1 contract: answer_vocab_size must be positive.")
    if config.answer_vocab_size != config.answer_top_k:
        raise ValueError("Version 1 contract: answer_vocab_size and answer_top_k must match.")
    if config.num_hops != 2:
        raise ValueError("Version 1 contract: num_hops must be exactly 2.")
    if config.train_subset_size <= 0 or config.val_subset_size <= 0:
        raise ValueError("Version 1 contract: subset sizes must be positive.")


def resolve_answer_top_k(config: V1Config) -> int:
    answer_top_k = int(config.answer_top_k)
    if answer_top_k <= 0:
        raise ValueError(f"Invalid answer_top_k: {answer_top_k}")

    config.answer_top_k = answer_top_k
    config.answer_vocab_size = answer_top_k
    return answer_top_k


def resolve_question_encoder_type(config: V1Config) -> str:
    encoder_type = config.question_encoder_type.strip().lower() if isinstance(config.question_encoder_type, str) else ""
    if encoder_type == "":
        encoder_type = config.rnn_type.strip().lower()

    if encoder_type == "bilstm":
        return "lstm"
    if encoder_type in {"gru", "lstm"}:
        return encoder_type
    raise ValueError(
        f"Unsupported question encoder type '{encoder_type}'. Supported values: gru, bilstm (or lstm)."
    )


def make_synthetic_loaders(config: V1Config) -> tuple[DataLoader, DataLoader, Vocab, List[str]]:
    token_to_idx = {Vocab.pad_token: 0, Vocab.unk_token: 1}
    for idx in range(2, config.question_vocab_size):
        token_to_idx[f"tok_{idx}"] = idx
    vocab = Vocab(token_to_idx)
    answer_vocab = [f"answer_{i}" for i in range(config.answer_vocab_size)]

    train_dataset = SyntheticVQADataset(
        size=min(config.train_subset_size, 512),
        question_length=config.question_max_length,
        answer_vocab_size=config.answer_vocab_size,
        use_precomputed_features=config.use_precomputed_features,
        image_size=config.image_size,
        use_soft_targets=config.use_soft_targets,
    )
    val_dataset = SyntheticVQADataset(
        size=min(config.val_subset_size, 128),
        question_length=config.question_max_length,
        answer_vocab_size=config.answer_vocab_size,
        use_precomputed_features=config.use_precomputed_features,
        image_size=config.image_size,
        use_soft_targets=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=vqa_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=vqa_collate_fn,
    )
    return train_loader, val_loader, vocab, answer_vocab


def make_real_loaders(config: V1Config) -> tuple[DataLoader, DataLoader, Vocab, List[str]]:
    vocab = build_question_vocab(
        question_paths=[config.train_questions_path],
        max_vocab_size=config.question_vocab_size,
    )
    answer_to_idx, answer_vocab = build_answer_vocab(
        annotation_path=config.train_annotations_path,
        top_k=config.answer_vocab_size,
    )

    train_dataset = VQAv2Dataset(
        questions_path=config.train_questions_path,
        annotations_path=config.train_annotations_path,
        images_dir=config.train_images_dir,
        vocab=vocab,
        answer_to_idx=answer_to_idx,
        max_question_length=config.question_max_length,
        image_size=config.image_size,
        max_samples=config.train_subset_size,
        use_precomputed_features=config.use_precomputed_features,
        precomputed_features_dir=config.precomputed_features_dir,
        use_soft_targets=config.use_soft_targets,
    )
    val_dataset = VQAv2Dataset(
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
        use_soft_targets=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=vqa_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=vqa_collate_fn,
    )

    return train_loader, val_loader, vocab, answer_vocab


def save_run_artifacts(config: V1Config, vocab: Vocab, answer_vocab: List[str]) -> None:
    Path(config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
    Path(config.predictions_dir).mkdir(parents=True, exist_ok=True)

    with Path(config.logs_dir, "question_vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab.token_to_idx, file)
    with Path(config.logs_dir, "answer_vocab.json").open("w", encoding="utf-8") as file:
        json.dump(answer_vocab, file)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
    use_soft_targets: bool,
) -> float:
    model.train()
    losses: List[float] = []

    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        inputs = {
            "question_ids": batch["question_ids"].to(device),
            "question_lengths": batch["question_lengths"].to(device),
        }
        if "images" in batch:
            inputs["images"] = batch["images"].to(device)
        if "image_features" in batch:
            inputs["image_features"] = batch["image_features"].to(device)

        logits = model(**inputs)["logits"]
        labels = batch["labels"].to(device)
        if use_soft_targets:
            if "soft_targets" not in batch:
                raise RuntimeError("use_soft_targets=True but batch does not contain soft_targets")
            soft_targets = batch["soft_targets"].to(device)
            loss = soft_target_cross_entropy(logits, soft_targets)
        else:
            loss = criterion(logits, labels)

        if torch.isnan(loss):
            raise RuntimeError("NaN loss encountered during training; stopping run.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        losses.append(float(loss.item()))
        progress.set_postfix({"loss": f"{np.mean(losses):.4f}"})

    return float(np.mean(losses)) if losses else 0.0


def soft_target_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 2 or soft_targets.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors for logits and soft_targets, got {tuple(logits.shape)} and {tuple(soft_targets.shape)}"
        )
    if logits.shape != soft_targets.shape:
        raise ValueError(
            f"Logits and soft_targets shape mismatch: {tuple(logits.shape)} vs {tuple(soft_targets.shape)}"
        )

    log_probs = torch.log_softmax(logits, dim=1)
    loss_per_sample = -(soft_targets * log_probs).sum(dim=1)
    return loss_per_sample.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Version-1 classical multi-hop VQA baseline")
    parser.add_argument("--config", type=str, default="configs/version1_vqa.json")
    parser.add_argument("--synthetic", action="store_true", help="Run an end-to-end smoke train on synthetic data")
    parser.add_argument("--subset-train", type=int, default=0)
    parser.add_argument("--subset-val", type=int, default=0)
    return parser.parse_args()


def _print_parameter_summary(model: nn.Module, config: V1Config) -> None:
    print(f"config_question_encoder_type={config.question_encoder_type}")
    print(f"config_rnn_type={config.rnn_type}")
    print(f"config_use_soft_targets={config.use_soft_targets}")
    print(f"config_freeze_image_encoder={config.freeze_image_encoder}")
    print(f"config_use_precomputed_features={config.use_precomputed_features}")

    image_encoder_present = hasattr(model, "image_encoder") and getattr(model, "image_encoder") is not None
    print(f"image_encoder_present={image_encoder_present}")

    if image_encoder_present:
        image_encoder_params = list(model.image_encoder.parameters())
        image_encoder_trainable = sum(parameter.numel() for parameter in image_encoder_params if parameter.requires_grad)
        image_encoder_frozen = sum(parameter.numel() for parameter in image_encoder_params if not parameter.requires_grad)
        image_encoder_all_frozen = image_encoder_trainable == 0
        print(f"image_encoder_trainable_params={image_encoder_trainable}")
        print(f"image_encoder_frozen_params={image_encoder_frozen}")
        print(f"image_encoder_all_frozen={image_encoder_all_frozen}")
    else:
        print("image_encoder_trainable_params=0")
        print("image_encoder_frozen_params=0")
        print("image_encoder_all_frozen=not_applicable")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = total - trainable
    print(f"parameters_total={total}")
    print(f"parameters_trainable={trainable}")
    print(f"parameters_frozen={frozen}")

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    print(f"trainable_param_tensors={len(trainable_names)}")
    print(f"frozen_param_tensors={len(frozen_names)}")
    if len(trainable_names) > 0:
        print(f"trainable_examples={trainable_names[:5]}")
    if len(frozen_names) > 0:
        print(f"frozen_examples={frozen_names[:5]}")

    classifier_output_dim = model.classifier[-1].out_features if hasattr(model, "classifier") else -1
    print(f"classifier_output_dim={classifier_output_dim}")


def _warn_if_tiny_filtered_subset(loader: DataLoader, requested_size: int, split_name: str) -> None:
    dataset = loader.dataset
    loaded = getattr(dataset, "stats", {}).get("loaded", len(dataset)) if hasattr(dataset, "stats") else len(dataset)
    skipped = getattr(dataset, "stats", {}).get("skipped_not_in_topk", 0) if hasattr(dataset, "stats") else 0
    print(f"{split_name}_loaded_samples={loaded}")
    print(f"{split_name}_filtered_out_not_in_topk={skipped}")

    if requested_size >= 100 and loaded < int(requested_size * 0.5):
        print(
            f"WARNING: {split_name} loaded subset is unexpectedly small after filtering "
            f"(requested={requested_size}, loaded={loaded})."
        )


def _print_run_configuration(config: V1Config, answer_vocab_size_effective: int, resolved_encoder_type: str) -> None:
    print(f"run_question_encoder_type={resolved_encoder_type}")
    print(f"run_use_soft_targets={config.use_soft_targets}")
    print(f"run_answer_top_k={config.answer_top_k}")
    print(f"run_answer_vocab_size_config={config.answer_vocab_size}")
    print(f"run_answer_vocab_size_effective={answer_vocab_size_effective}")
    print(f"run_train_subset_size={config.train_subset_size}")
    print(f"run_val_subset_size={config.val_subset_size}")
    print(f"run_freeze_image_encoder={config.freeze_image_encoder}")
    print(f"run_learning_rate={config.learning_rate}")
    print(f"run_epochs={config.epochs}")
    print(f"run_batch_size={config.batch_size}")


def main() -> None:
    args = parse_args()
    config = V1Config.from_json(args.config) if Path(args.config).exists() else get_default_config()

    if args.subset_train > 0:
        config.train_subset_size = args.subset_train
    if args.subset_val > 0:
        config.val_subset_size = args.subset_val

    resolve_answer_top_k(config)

    enforce_v1_contracts(config)
    resolved_encoder_type = resolve_question_encoder_type(config)
    set_seed(config.seed)
    device = resolve_device(config.device)

    if args.synthetic:
        train_loader, val_loader, vocab, answer_vocab = make_synthetic_loaders(config)
    else:
        train_loader, val_loader, vocab, answer_vocab = make_real_loaders(config)

    _warn_if_tiny_filtered_subset(train_loader, config.train_subset_size, "train")
    _warn_if_tiny_filtered_subset(val_loader, config.val_subset_size, "val")

    save_run_artifacts(config, vocab, answer_vocab)
    _print_run_configuration(config, answer_vocab_size_effective=len(answer_vocab), resolved_encoder_type=resolved_encoder_type)

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
    _print_parameter_summary(model, config)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = None
    if config.scheduler_step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )

    history: List[dict] = []
    best_val_soft = -1.0
    best_val_top1 = -1.0
    best_epoch = -1

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            grad_clip_norm=config.grad_clip_norm,
            use_soft_targets=config.use_soft_targets,
        )

        val_outputs = evaluate_model_on_loader(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            answer_vocab=answer_vocab,
            split_name="val",
            synthetic=args.synthetic,
            annotation_path=None if args.synthetic else config.val_annotations_path,
            images_dir=None if args.synthetic else config.val_images_dir,
            qualitative_limit=config.qualitative_failures_to_save,
            save_outputs=False,
            predictions_dir=config.predictions_dir,
            logs_dir=config.logs_dir,
            file_prefix="epoch_val",
        )

        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_outputs["eval_loss"],
            "val_top1_accuracy": val_outputs["top1_accuracy"],
            "val_vqa_soft_accuracy": val_outputs["vqa_soft_accuracy"],
            "val_answer_type_breakdown": val_outputs["answer_type_breakdown"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_row)

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_outputs['eval_loss']:.4f} "
            f"val_top1={val_outputs['top1_accuracy']:.4f} "
            f"val_soft={val_outputs['vqa_soft_accuracy']:.4f}"
        )

        latest_checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config.as_dict(),
            "question_vocab": vocab.token_to_idx,
            "answer_vocab": answer_vocab,
            "epoch": epoch,
            "best_val_vqa_soft": best_val_soft,
            "latest_val_vqa_soft": val_outputs["vqa_soft_accuracy"],
        }
        torch.save(latest_checkpoint, Path(config.checkpoints_dir, "latest.pt"))

        is_best = val_outputs["vqa_soft_accuracy"] > best_val_soft
        if is_best:
            best_val_soft = val_outputs["vqa_soft_accuracy"]
            best_val_top1 = val_outputs["top1_accuracy"]
            best_epoch = epoch
            best_checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": config.as_dict(),
                "question_vocab": vocab.token_to_idx,
                "answer_vocab": answer_vocab,
                "epoch": epoch,
                "best_val_vqa_soft": best_val_soft,
                "best_val_top1": best_val_top1,
            }
            torch.save(best_checkpoint, Path(config.checkpoints_dir, "best.pt"))

            best_outputs = evaluate_model_on_loader(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                answer_vocab=answer_vocab,
                split_name="val_best",
                synthetic=args.synthetic,
                annotation_path=None if args.synthetic else config.val_annotations_path,
                images_dir=None if args.synthetic else config.val_images_dir,
                qualitative_limit=config.qualitative_failures_to_save,
                save_outputs=True,
                predictions_dir=config.predictions_dir,
                logs_dir=config.logs_dir,
                file_prefix="best_epoch",
            )
            print(
                f"best_checkpoint_updated epoch={epoch} "
                f"val_soft={best_outputs['vqa_soft_accuracy']:.4f} path={Path(config.checkpoints_dir, 'best.pt')}"
            )

        if scheduler is not None:
            scheduler.step()

    final_outputs = evaluate_model_on_loader(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        answer_vocab=answer_vocab,
        split_name="val_final",
        synthetic=args.synthetic,
        annotation_path=None if args.synthetic else config.val_annotations_path,
        images_dir=None if args.synthetic else config.val_images_dir,
        qualitative_limit=config.qualitative_failures_to_save,
        save_outputs=True,
        predictions_dir=config.predictions_dir,
        logs_dir=config.logs_dir,
        file_prefix="final_epoch",
    )

    history_path = Path(config.logs_dir, "training_history.json")
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    csv_path = Path(config.logs_dir, "training_history.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "val_loss", "val_top1_accuracy", "val_vqa_soft_accuracy", "learning_rate"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "val_loss": row["val_loss"],
                    "val_top1_accuracy": row["val_top1_accuracy"],
                    "val_vqa_soft_accuracy": row["val_vqa_soft_accuracy"],
                    "learning_rate": row["learning_rate"],
                }
            )

    run_summary = {
        "question_encoder_type": resolved_encoder_type,
        "use_soft_targets": config.use_soft_targets,
        "best_epoch": best_epoch,
        "best_val_vqa_soft": best_val_soft,
        "best_val_top1": best_val_top1,
        "best_checkpoint": str(Path(config.checkpoints_dir, "best.pt")),
        "latest_checkpoint": str(Path(config.checkpoints_dir, "latest.pt")),
    }
    run_summary_path = Path(config.logs_dir, "run_summary.json")
    with run_summary_path.open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2)

    print(f"best_epoch={best_epoch}")
    print(f"best_val_vqa_soft={best_val_soft:.4f}")
    print(f"best_val_top1={best_val_top1:.4f}")
    print(f"latest_checkpoint={Path(config.checkpoints_dir, 'latest.pt')}")
    print(f"best_checkpoint={Path(config.checkpoints_dir, 'best.pt')}")
    print(f"history_json={history_path}")
    print(f"history_csv={csv_path}")
    print(f"run_summary_json={run_summary_path}")
    print(f"final_eval_predictions={final_outputs.get('predictions_file', '')}")
    print(f"final_eval_failures={final_outputs.get('failures_file', '')}")

    if resolved_encoder_type == "lstm":
        ablation_summary_path = Path("outputs/encoder_ablation/gru_vs_bilstm_summary.json")
        ablation_summary_path.parent.mkdir(parents=True, exist_ok=True)
        ablation_summary = {
            "reference_gru_best_val_vqa_soft": 0.3715,
            "reference_gru_best_val_top1": 0.2983,
            "bilstm_best_val_vqa_soft": best_val_soft,
            "bilstm_best_val_top1": best_val_top1,
            "bilstm_best_checkpoint": str(Path(config.checkpoints_dir, "best.pt")),
            "bilstm_run_summary": str(run_summary_path),
            "bilstm_final_eval_metrics": final_outputs.get("metrics_file", ""),
        }
        with ablation_summary_path.open("w", encoding="utf-8") as file:
            json.dump(ablation_summary, file, indent=2)
        print(f"encoder_ablation_summary_json={ablation_summary_path}")


if __name__ == "__main__":
    main()
