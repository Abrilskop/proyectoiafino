"""Deteccion simple de data drift (ML-Ops).

Compara las estadisticas de color de la imagen entrante contra las estadisticas
de entrenamiento (media/desviacion de SVHN). Si la desviacion absoluta de la
media supera el umbral, se marca ``drift_detected``.
"""
from __future__ import annotations

import numpy as np

# Estadisticas de referencia del dataset de entrenamiento (espacio [0, 1]).
REFERENCE_MEAN = np.array([0.4377, 0.4438, 0.4728])
REFERENCE_STD = np.array([0.1980, 0.2010, 0.1970])


def check_drift(image, threshold: float = 0.25) -> dict:
    arr = np.asarray(image).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[..., :3] / 255.0

    mean = arr.reshape(-1, 3).mean(axis=0)
    drift_score = float(np.abs(mean - REFERENCE_MEAN).mean())
    return {
        "mean_rgb": [round(float(v), 4) for v in mean],
        "drift_score": round(drift_score, 4),
        "threshold": threshold,
        "drift_detected": bool(drift_score > threshold),
    }
