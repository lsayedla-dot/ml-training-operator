"""AV object classifier model based on ResNet."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.worker.dataset import NUM_CLASSES


class AVObjectClassifier(nn.Module):
    """ResNet-based classifier for AV object detection.

    Classifies cropped images as: car, pedestrian, cyclist, or other.
    Uses pretrained ResNet backbone with custom classification head.
    """

    def __init__(self, model_type: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        self.model_type = model_type

        if model_type == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        elif model_type == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50

            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Replace the final FC layer for our classification task
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, NUM_CLASSES)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier."""
        return self.backbone(x)
