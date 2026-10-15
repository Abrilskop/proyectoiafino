"""Pruebas de los modelos (O.E.4)."""
from __future__ import annotations

import pytest
import torch

from src.models import build_model


@pytest.mark.parametrize("name", ["cnn", "resnet18"])
def test_forward_shape(name):
    model = build_model(name, num_classes=10, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 10)


def test_cnn_output_is_logits():
    model = build_model("cnn")
    out = model(torch.randn(1, 3, 32, 32))
    assert out.dtype == torch.float32
