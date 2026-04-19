from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import SpatialAttentionHop
from src.models.image_encoder import FrozenResNet50Encoder
from src.models.question_encoder import QuestionEncoder


class MultiHopVQAModel(nn.Module):
    def __init__(
        self,
        question_vocab_size: int,
        answer_vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 512,
        num_hops: int = 2,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        freeze_image_encoder: bool = True,
        use_precomputed_features: bool = False,
        visual_dim: int = 2048,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.use_precomputed_features = use_precomputed_features
        self.num_hops = num_hops

        if not use_precomputed_features:
            self.image_encoder = FrozenResNet50Encoder(pretrained=True, freeze=freeze_image_encoder)

        self.question_encoder = QuestionEncoder(
            vocab_size=question_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            pad_idx=pad_idx,
        )

        self.hops = nn.ModuleList(
            [
                SpatialAttentionHop(
                    visual_dim=visual_dim,
                    question_dim=hidden_dim,
                    joint_dim=hidden_dim,
                )
                for _ in range(num_hops)
            ]
        )
        self.hop_updates = nn.ModuleList([nn.Linear(visual_dim, hidden_dim) for _ in range(num_hops)])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, answer_vocab_size),
        )

    def forward(
        self,
        question_ids: torch.Tensor,
        question_lengths: torch.Tensor,
        images: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        debug: bool = False,
    ) -> dict:
        feature_map = None
        if self.use_precomputed_features:
            if image_features is None:
                raise ValueError("image_features must be provided when use_precomputed_features=True")
            visual_tokens = image_features
        else:
            if images is None:
                raise ValueError("images must be provided when use_precomputed_features=False")
            if debug:
                visual_tokens, feature_map = self.image_encoder(images, return_feature_map=True)
            else:
                visual_tokens = self.image_encoder(images)

        question_embedding = None
        if debug:
            question_outputs = self.question_encoder(question_ids, question_lengths, return_intermediate=True)
            question_embedding = question_outputs["embedded"]
            question_vector = question_outputs["question_vector"]
        else:
            question_vector = self.question_encoder(question_ids, question_lengths)

        state = question_vector
        attention_per_hop = []
        attended_per_hop = []
        attended_vector = None

        for hop_index in range(self.num_hops):
            attended_vector, attention_weights = self.hops[hop_index](visual_tokens, state)
            state = state + self.hop_updates[hop_index](attended_vector)
            attention_per_hop.append(attention_weights)
            attended_per_hop.append(attended_vector)

        if attended_vector is None:
            raise RuntimeError("No attention hops executed. num_hops must be >= 1.")

        fusion = torch.cat([state, attended_vector], dim=1)
        logits = self.classifier(fusion)

        outputs = {
            "logits": logits,
            "attention_weights": torch.stack(attention_per_hop, dim=1),
            "visual_tokens": visual_tokens,
            "question_vector": question_vector,
        }

        if debug:
            outputs["debug"] = {
                "feature_map": feature_map,
                "question_embedding": question_embedding,
                "attended_per_hop": torch.stack(attended_per_hop, dim=1),
                "fusion": fusion,
            }

        return outputs
