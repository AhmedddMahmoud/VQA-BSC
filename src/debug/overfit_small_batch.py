from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.datasets.vqa_dataset import SyntheticVQADataset, VQAv2Dataset, vqa_collate_fn
from src.models.multihop_vqa import MultiHopVQAModel
from src.train.train_vqa import enforce_v1_contracts, resolve_device, set_seed
from src.utils.config import V1Config
from src.utils.metrics import save_jsonl
from src.utils.vocab import Vocab, build_answer_vocab, build_question_vocab


def build_tiny_loader(
    config: V1Config,
    tiny_samples: int,
    synthetic: bool,
) -> tuple[DataLoader, Vocab, List[str]]:
    if synthetic:
        token_to_idx = {Vocab.pad_token: 0, Vocab.unk_token: 1}
        for idx in range(2, config.question_vocab_size):
            token_to_idx[f"tok_{idx}"] = idx
        vocab = Vocab(token_to_idx)
        answer_vocab = [f"answer_{index}" for index in range(config.answer_vocab_size)]

        dataset = SyntheticVQADataset(
            size=tiny_samples,
            question_length=config.question_max_length,
            answer_vocab_size=config.answer_vocab_size,
            use_precomputed_features=config.use_precomputed_features,
            image_size=config.image_size,
        )
    else:
        vocab = build_question_vocab([config.train_questions_path], max_vocab_size=config.question_vocab_size)
        answer_to_idx, answer_vocab = build_answer_vocab(config.train_annotations_path, top_k=config.answer_vocab_size)

        dataset = VQAv2Dataset(
            questions_path=config.train_questions_path,
            annotations_path=config.train_annotations_path,
            images_dir=config.train_images_dir,
            vocab=vocab,
            answer_to_idx=answer_to_idx,
            max_question_length=config.question_max_length,
            image_size=config.image_size,
            max_samples=tiny_samples,
            use_precomputed_features=config.use_precomputed_features,
            precomputed_features_dir=config.precomputed_features_dir,
        )

    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, tiny_samples),
        shuffle=True,
        num_workers=0,
        collate_fn=vqa_collate_fn,
    )
    return loader, vocab, answer_vocab


def evaluate_tiny_set(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    answer_vocab: List[str],
) -> tuple[float, float, List[dict]]:
    model.eval()
    losses: List[float] = []
    correct = 0
    total = 0
    prediction_rows: List[dict] = []

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
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))

            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())

            for i in range(labels.size(0)):
                gold_idx = int(labels[i].item())
                pred_idx = int(predictions[i].item())
                prediction_rows.append(
                    {
                        "question_id": int(batch["question_ids_list"][i]),
                        "question_text": batch["question_texts"][i],
                        "gold_answer": answer_vocab[gold_idx],
                        "predicted_answer": answer_vocab[pred_idx],
                        "gold_idx": gold_idx,
                        "pred_idx": pred_idx,
                    }
                )

    mean_loss = sum(losses) / len(losses) if losses else 0.0
    accuracy = (correct / total) if total > 0 else 0.0
    return mean_loss, accuracy, prediction_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a tiny batch to validate baseline training correctness.")
    parser.add_argument("--config", type=str, default="configs/version1_vqa.json")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic tiny dataset for quick debugging.")
    parser.add_argument("--tiny-samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--checkpoint-name", type=str, default="overfit_tiny_batch.pt")
    parser.add_argument("--predictions-name", type=str, default="overfit_tiny_predictions.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = V1Config.from_json(args.config)
    enforce_v1_contracts(config)

    if args.tiny_samples <= 0:
        raise ValueError("--tiny-samples must be > 0")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.print_every <= 0:
        raise ValueError("--print-every must be > 0")

    set_seed(config.seed)
    device = resolve_device(config.device)

    loader, vocab, answer_vocab = build_tiny_loader(
        config=config,
        tiny_samples=args.tiny_samples,
        synthetic=args.synthetic,
    )

    model = MultiHopVQAModel(
        question_vocab_size=len(vocab),
        answer_vocab_size=len(answer_vocab),
        embedding_dim=config.word_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_hops=config.num_hops,
        rnn_type=config.rnn_type,
        bidirectional=config.bidirectional,
        freeze_image_encoder=config.freeze_image_encoder,
        use_precomputed_features=config.use_precomputed_features,
        pad_idx=vocab.pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    iterator = iter(loader)
    first_batch = next(iterator)
    first_inputs = {
        "question_ids": first_batch["question_ids"].to(device),
        "question_lengths": first_batch["question_lengths"].to(device),
    }
    if "images" in first_batch:
        first_inputs["images"] = first_batch["images"].to(device)
    if "image_features" in first_batch:
        first_inputs["image_features"] = first_batch["image_features"].to(device)

    model.eval()
    with torch.no_grad():
        initial_logits = model(**first_inputs)["logits"]
        initial_loss = criterion(initial_logits, first_batch["labels"].to(device)).item()
    print(f"initial_loss: {initial_loss:.6f}")

    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        inputs = {
            "question_ids": batch["question_ids"].to(device),
            "question_lengths": batch["question_lengths"].to(device),
        }
        if "images" in batch:
            inputs["images"] = batch["images"].to(device)
        if "image_features" in batch:
            inputs["image_features"] = batch["image_features"].to(device)

        labels = batch["labels"].to(device)
        logits = model(**inputs)["logits"]
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.print_every == 0 or step == 1 or step == args.steps:
            batch_accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
            print(f"step={step:04d} loss={loss.item():.6f} batch_acc={batch_accuracy:.4f}")

    final_loss, tiny_accuracy, prediction_rows = evaluate_tiny_set(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        answer_vocab=answer_vocab,
    )
    print(f"final_loss: {final_loss:.6f}")
    print(f"tiny_set_accuracy: {tiny_accuracy:.4f}")

    checkpoint_dir = Path(config.checkpoints_dir)
    prediction_dir = Path(config.predictions_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / args.checkpoint_name
    prediction_path = prediction_dir / args.predictions_name

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.as_dict(),
            "question_vocab": vocab.token_to_idx,
            "answer_vocab": answer_vocab,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "tiny_set_accuracy": tiny_accuracy,
            "steps": args.steps,
            "tiny_samples": args.tiny_samples,
        },
        checkpoint_path,
    )
    save_jsonl(prediction_path, prediction_rows)

    print(f"saved_checkpoint: {checkpoint_path}")
    print(f"saved_predictions: {prediction_path}")

    if not (final_loss < initial_loss * 0.7 or tiny_accuracy >= 0.9):
        print(
            "NOTE: Tiny-set overfit signal is weak. This can indicate a bug in data-label alignment, "
            "loss setup, attention flow, or optimizer updates."
        )


if __name__ == "__main__":
    main()
