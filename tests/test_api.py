"""Pruebas de la API de despliegue (O.E.6).

Construye un checkpoint temporal (modelo sin entrenar) para poder probar el
endpoint sin depender de que exista ``models/best.pt``.
"""
from __future__ import annotations

import io

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from deployment.app import create_app
from src.models import build_model


def _make_temp_checkpoint(tmp_path):
    model = build_model("cnn", num_classes=10)
    ckpt_path = tmp_path / "test_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": "cnn",
            "num_classes": 10,
            "epoch": 0,
            "val_acc": 0.1,
            "mean": [0.4377, 0.4438, 0.4728],
            "std": [0.1980, 0.2010, 0.1970],
            "image_size": 32,
        },
        ckpt_path,
    )
    return ckpt_path


def _client(tmp_path):
    return TestClient(create_app(_make_temp_checkpoint(tmp_path)))


def test_health(tmp_path):
    resp = _client(tmp_path).get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_digit(tmp_path):
    client = _client(tmp_path)
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)

    resp = client.post("/predict", files={"file": ("test.png", buf, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["prediction"] <= 9
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 10


def test_predict_rejects_non_image(tmp_path):
    client = _client(tmp_path)
    resp = client.post(
        "/predict", files={"file": ("test.txt", io.BytesIO(b"hi"), "text/plain")}
    )
    assert resp.status_code == 400
