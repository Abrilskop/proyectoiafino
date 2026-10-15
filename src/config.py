"""Carga central de configuracion.

Toda la configuracion vive en ``configs/config.yaml`` (fuente unica de verdad).
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Lee el YAML de configuracion y lo devuelve como diccionario."""
    cfg_path = Path(path) if path else CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(path_str: str) -> Path:
    """Convierte una ruta relativa de la config en ruta absoluta del proyecto."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def get_paths(cfg: dict) -> dict:
    """Devuelve el bloque ``paths`` resuelto a rutas absolutas."""
    return {key: resolve(value) for key, value in cfg["paths"].items()}
