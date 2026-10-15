"""Pruebas de la capa de datos (O.E.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.dataset import SVHNTensors, build_transforms

TRAIN_PT = Path("data/processed/train.pt")


def test_transforms_output_normalized_tensor():
    cfg = load_config()
    tf = build_transforms(cfg, train=False)
    img = torch.from_numpy((np.random.rand(3, 32, 32) * 255).astype(np.uint8))
    out = tf(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


@pytest.mark.skipif(not TRAIN_PT.exists(), reason="Ejecuta 'make oe3' primero")
def test_train_dataset_item():
    cfg = load_config()
    ds = SVHNTensors(TRAIN_PT, build_transforms(cfg, train=True))
    img, label = ds[0]
    assert img.shape == (3, cfg["data"]["image_size"], cfg["data"]["image_size"])
    assert 0 <= label < 10
