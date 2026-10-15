"""Registro de predicciones (ML-Ops).

Escribe cada inferencia como una linea JSON en ``reports/predictions.jsonl``,
insumo para auditoria y deteccion de drift.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path("reports/predictions.jsonl")


class PredictionLogger:
    def __init__(self, path: str | Path = DEFAULT_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> dict:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
