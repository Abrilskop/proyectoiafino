"""Transfer learning con ResNet18 (opcional, mas pesado que el baseline CNN)."""
from __future__ import annotations

import torch.nn as nn


def build_resnet18(num_classes: int = 10, pretrained: bool = False):
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # SVHN es 32x32: adaptamos la primera convolucion y quitamos el maxpool agresivo.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
