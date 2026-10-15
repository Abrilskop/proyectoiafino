"""CNN baseline para SVHN (32x32x3 -> 10 clases)."""
from __future__ import annotations

import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    """Red convolucional ligera: 3 bloques conv + clasificador denso."""

    def __init__(self, num_classes: int = 10, p_dropout: float = 0.3):
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # /2
            )

        self.features = nn.Sequential(
            block(3, 32),    # 32 -> 16
            block(32, 64),   # 16 -> 8
            block(64, 128),  # 8  -> 4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p_dropout),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
