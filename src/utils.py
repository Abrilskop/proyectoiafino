"""Utilidades compartidas."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Fija las semillas para que los resultados sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch no instalado todavia
        pass


def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio (y sus padres) si no existe y lo devuelve."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: str | Path) -> Path:
    """Guarda un diccionario como JSON legible."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return path


def load_json(path: str | Path) -> dict:
    """Carga un JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
