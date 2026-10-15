"""O.E.1 - Comprension del negocio (Sprint 1).

Documenta el problema, los KPIs y los criterios de exito, y los persiste como
evidencia demostrable en ``reports/oe1_business.json``.
"""
from __future__ import annotations

from .config import get_paths, load_config
from .utils import save_json


def run() -> dict:
    cfg = load_config()
    business = cfg["business"]
    paths = get_paths(cfg)

    evidence = {
        "objetivo_especifico": "O.E.1 - Comprender el negocio (CRISP-ML)",
        "sprint": 1,
        "problema": business["problem"],
        "kpi": {
            "metrica": business["target_metric"],
            "umbral_exito": business["target_value"],
            "objetivo_ambicioso": business["stretch_value"],
            "latencia_max_ms": business["max_latency_ms"],
        },
        "clases": business["classes"],
        "criterio_aceptacion": (
            f"El modelo se considera exitoso si {business['target_metric']} "
            f">= {business['target_value']} en el subconjunto de evaluacion."
        ),
    }

    out = paths["reports_dir"] / "oe1_business.json"
    save_json(evidence, out)

    print("=" * 68)
    print("O.E.1 - COMPRENSION DEL NEGOCIO (Sprint 1)")
    print("=" * 68)
    print(f"Problema : {business['problem']}")
    print(f"KPI      : {business['target_metric']} >= {business['target_value']}")
    print(f"Clases   : {business['classes']}")
    print(f"Evidencia: {out}")
    print("=" * 68)
    return evidence


if __name__ == "__main__":
    run()
