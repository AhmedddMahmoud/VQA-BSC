from __future__ import annotations

import torch
import torch.nn as nn


class SpatialAttentionHop(nn.Module):
    def __init__(self, visual_dim: int, question_dim: int, joint_dim: int) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, joint_dim)
        self.question_proj = nn.Linear(question_dim, joint_dim)
        self.score_proj = nn.Linear(joint_dim, 1)

    def forward(self, visual_tokens: torch.Tensor, query_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visual_component = self.visual_proj(visual_tokens)
        query_component = self.question_proj(query_vector).unsqueeze(1)
        joint = torch.tanh(visual_component + query_component)
        scores = self.score_proj(joint).squeeze(-1)
        attention_weights = torch.softmax(scores, dim=1)
        attended = torch.sum(attention_weights.unsqueeze(-1) * visual_tokens, dim=1)
        return attended, attention_weights
