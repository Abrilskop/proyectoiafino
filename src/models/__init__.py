"""Modelos de clasificacion de digitos."""
from __future__ import annotations

from .cnn import DigitCNN

__all__ = ["DigitCNN", "build_model"]


def build_model(name: str = "cnn", num_classes: int = 10, pretrained: bool = False):
    """Fabrica de modelos: ``cnn`` (baseline) o ``resnet18`` (transfer learning)."""
    name = name.lower()
    if name == "cnn":
        return DigitCNN(num_classes=num_classes)
    if name in ("resnet18", "resnet"):
        from .resnet import build_resnet18

        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Modelo desconocido: {name}")
