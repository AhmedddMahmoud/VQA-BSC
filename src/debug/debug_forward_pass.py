from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.models.multihop_vqa import MultiHopVQAModel
from src.train.train_vqa import (
    enforce_v1_contracts,
    make_real_loaders,
    make_synthetic_loaders,
    resolve_device,
    set_seed,
)
from src.utils.config import V1Config


def print_shape(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        print(f"{name}: <none>")
        return
    print(f"{name}: {tuple(tensor.shape)}")


def assert_expected_shapes(
    config: V1Config,
    batch: dict,
    outputs: dict,
) -> None:
    question_ids = batch["question_ids"]
    question_lengths = batch["question_lengths"]
    logits = outputs["logits"]
    visual_tokens = outputs["visual_tokens"]
    attention = outputs["attention_weights"]
    debug = outputs.get("debug", {})

    batch_size = question_ids.size(0)

    assert question_ids.ndim == 2, f"question_ids must be 2D, got shape {tuple(question_ids.shape)}"
    assert question_lengths.ndim == 1, f"question_lengths must be 1D, got shape {tuple(question_lengths.shape)}"
    assert question_ids.size(1) == config.question_max_length, (
        f"question_ids second dim should match question_max_length={config.question_max_length}, "
        f"got {question_ids.size(1)}"
    )

    assert visual_tokens.ndim == 3, f"visual_tokens must be 3D, got shape {tuple(visual_tokens.shape)}"
    assert visual_tokens.size(0) == batch_size, "visual_tokens batch size mismatch"
    assert visual_tokens.size(1) == 49, f"Expected 49 spatial tokens, got {visual_tokens.size(1)}"
    assert visual_tokens.size(2) == 2048, f"Expected visual dim 2048, got {visual_tokens.size(2)}"

    assert attention.ndim == 3, f"attention_weights must be 3D, got shape {tuple(attention.shape)}"
    assert attention.size(0) == batch_size, "attention batch size mismatch"
    assert attention.size(1) == config.num_hops, (
        f"attention hop dim should be {config.num_hops}, got {attention.size(1)}"
    )
    assert attention.size(2) == 49, f"attention region dim should be 49, got {attention.size(2)}"

    question_embedding = debug.get("question_embedding")
    question_vector = outputs["question_vector"]
    attended_per_hop = debug.get("attended_per_hop")
    fusion = debug.get("fusion")

    assert question_vector.shape == (batch_size, config.hidden_dim), (
        f"question_vector shape expected {(batch_size, config.hidden_dim)}, got {tuple(question_vector.shape)}"
    )

    if question_embedding is not None:
        assert question_embedding.shape[0] == batch_size, "question_embedding batch mismatch"
        assert question_embedding.shape[1] == config.question_max_length, "question_embedding seq length mismatch"

    if attended_per_hop is None:
        raise AssertionError("debug.attended_per_hop not returned in debug mode")
    assert attended_per_hop.shape == (batch_size, config.num_hops, 2048), (
        f"attended_per_hop expected {(batch_size, config.num_hops, 2048)}, got {tuple(attended_per_hop.shape)}"
    )

    if fusion is None:
        raise AssertionError("debug.fusion not returned in debug mode")
    assert fusion.shape == (batch_size, config.hidden_dim + 2048), (
        f"fusion expected {(batch_size, config.hidden_dim + 2048)}, got {tuple(fusion.shape)}"
    )

    assert logits.shape == (batch_size, config.answer_vocab_size), (
        f"logits expected {(batch_size, config.answer_vocab_size)}, got {tuple(logits.shape)}"
    )

    if not config.use_precomputed_features:
        images = batch.get("images")
        if images is None:
            raise AssertionError("images tensor missing while use_precomputed_features=False")
        assert images.ndim == 4, f"images must be 4D, got shape {tuple(images.shape)}"
        assert images.shape[1:] == (3, config.image_size, config.image_size), (
            f"images expected shape (*, 3, {config.image_size}, {config.image_size}), got {tuple(images.shape)}"
        )

        feature_map = debug.get("feature_map")
        if feature_map is None:
            raise AssertionError("debug.feature_map missing while image encoder is used")
        assert feature_map.shape == (batch_size, 2048, 7, 7), (
            f"feature_map expected {(batch_size, 2048, 7, 7)}, got {tuple(feature_map.shape)}"
        )


def print_attention_stats(attention_weights: torch.Tensor) -> None:
    num_hops = attention_weights.size(1)
    for hop_idx in range(num_hops):
        hop = attention_weights[:, hop_idx, :]
        sums = hop.sum(dim=1)
        print(
            f"hop_{hop_idx + 1}_attention: min={hop.min().item():.6f} max={hop.max().item():.6f} "
            f"mean={hop.mean().item():.6f}"
        )
        print(
            f"hop_{hop_idx + 1}_softmax_sum: min={sums.min().item():.6f} "
            f"max={sums.max().item():.6f} mean={sums.mean().item():.6f}"
        )
        if not torch.allclose(sums, torch.ones_like(sums), atol=1e-4, rtol=1e-4):
            raise AssertionError(
                f"Softmax sums are not approximately 1 for hop {hop_idx + 1}. "
                f"Observed sums: {sums.tolist()}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-batch forward pass with explicit shape debugging.")
    parser.add_argument("--config", type=str, default="configs/version1_vqa.json")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data path for quick verification.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = V1Config.from_json(config_path)
    enforce_v1_contracts(config)
    set_seed(config.seed)

    device = resolve_device(config.device)
    if args.synthetic:
        train_loader, val_loader, vocab, _ = make_synthetic_loaders(config)
    else:
        train_loader, val_loader, vocab, _ = make_real_loaders(config)
    loader = train_loader if args.split == "train" else val_loader

    model = MultiHopVQAModel(
        question_vocab_size=len(vocab),
        answer_vocab_size=config.answer_vocab_size,
        embedding_dim=config.word_embedding_dim,
        hidden_dim=config.hidden_dim,
        num_hops=config.num_hops,
        rnn_type=config.rnn_type,
        bidirectional=config.bidirectional,
        freeze_image_encoder=config.freeze_image_encoder,
        use_precomputed_features=config.use_precomputed_features,
        pad_idx=vocab.pad_idx,
    ).to(device)
    model.eval()

    batch = next(iter(loader))
    model_inputs = {
        "question_ids": batch["question_ids"].to(device),
        "question_lengths": batch["question_lengths"].to(device),
        "debug": True,
    }
    if "images" in batch:
        model_inputs["images"] = batch["images"].to(device)
    if "image_features" in batch:
        model_inputs["image_features"] = batch["image_features"].to(device)

    with torch.no_grad():
        outputs = model(**model_inputs)

    debug = outputs.get("debug", {})

    print("==== Forward Pass Tensor Shapes ====")
    print_shape("input_images", model_inputs.get("images"))
    print_shape("cnn_feature_map_before_flatten", debug.get("feature_map"))
    print_shape("flattened_visual_tokens", outputs.get("visual_tokens"))
    print_shape("question_token_tensor", model_inputs.get("question_ids"))
    print_shape("question_lengths", model_inputs.get("question_lengths"))
    print_shape("question_embedding_output", debug.get("question_embedding"))
    print_shape("final_question_vector", outputs.get("question_vector"))
    print_shape("hop_1_attention_weights", outputs["attention_weights"][:, 0, :])
    print_shape("hop_1_attended_visual", debug["attended_per_hop"][:, 0, :])
    print_shape("hop_2_attention_weights", outputs["attention_weights"][:, 1, :])
    print_shape("hop_2_attended_visual", debug["attended_per_hop"][:, 1, :])
    print_shape("fused_representation", debug.get("fusion"))
    print_shape("logits", outputs.get("logits"))

    print("\n==== Attention Sanity Stats ====")
    print_attention_stats(outputs["attention_weights"])

    assert_expected_shapes(config=config, batch=batch, outputs=outputs)
    print("\nAll shape assertions passed.")


if __name__ == "__main__":
    main()
