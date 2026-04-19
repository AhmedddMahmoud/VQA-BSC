from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class FrozenResNet50Encoder(nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        try:
            backbone = models.resnet50(weights=weights)
        except Exception:
            backbone = models.resnet50(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        if freeze:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor, return_feature_map: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.feature_extractor(images)
        batch_size, channels, height, width = feature_map.shape
        tokens = feature_map.view(batch_size, channels, height * width).transpose(1, 2)
        if return_feature_map:
            return tokens, feature_map
        return tokens
