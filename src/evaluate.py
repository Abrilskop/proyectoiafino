"""O.E.5 - Evaluacion (Sprint 3).

Carga el mejor checkpoint y evalua sobre el split de test completo: accuracy,
macro-F1, reporte por clase y matriz de confusion. Persiste
``reports/metrics.json`` y ``reports/figures/confusion_matrix.png``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import get_paths, load_config
from .data.dataset import get_dataloaders
from .engine import evaluate, get_device
from .models import build_model
from .utils import ensure_dir, save_json


def run() -> dict:
    cfg = load_config()
    paths = get_paths(cfg)
    device = get_device()

    print("=" * 68)
    print("O.E.5 - EVALUACION (Sprint 3)")
    print("=" * 68)

    ckpt = torch.load(paths["best_model"], map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, _, test_loader = get_dataloaders(cfg)
    criterion = nn.CrossEntropyLoss()
    loss, acc, preds, labels = evaluate(model, test_loader, criterion, device, collect_preds=True)

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    macro_f1 = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(10)))

    target = cfg["business"]["target_value"]
    evidence = {
        "objetivo_especifico": "O.E.5 - Evaluacion",
        "sprint": 3,
        "n_test": int(len(labels)),
        "loss": round(float(loss), 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "target_value": target,
        "cumple_kpi": bool(acc >= target),
        "per_class": {k: v for k, v in report.items() if k.isdigit()},
        "confusion_matrix": cm.tolist(),
    }
    out = paths["reports_dir"] / "metrics.json"
    save_json(evidence, out)
    _plot_confusion(paths["figures_dir"], cm)

    print(f"Test accuracy = {acc:.4f} | macro-F1 = {macro_f1:.4f} "
          f"| KPI({'OK' if evidence['cumple_kpi'] else 'NO'})")
    print(f"Evidencia: {out}")
    print("=" * 68)
    return evidence


def _plot_confusion(figs_dir, cm) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figs = ensure_dir(figs_dir)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Matriz de confusion - SVHN (test)")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(figs / "confusion_matrix.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
