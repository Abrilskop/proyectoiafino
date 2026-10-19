"""Fase C - Experimentos comparativos de la CNN (actividad).

Corre la matriz de experimentos sobre el subconjunto preparado y compara el
efecto de: Dropout, optimizador (Adam/SGD), Data Augmentation (rotacion/zoom)
y Batch Normalization. Exporta:

  reports/experiments/<nombre>.json   (historial + metricas de test)
  reports/experiments/comparison.csv  (tabla comparativa)
  reports/experiments/comparison.md
  reports/experiments/comparison.png  (grafico de barras)

Uso:
  python -m src.experiments                 # corre la matriz completa
  python -m src.experiments --epochs 4 --max-train 8000
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .config import load_config, resolve
from .data.dataset import SVHNTensors, build_transforms
from .engine import evaluate, get_device, train_one_epoch
from .models.cnn import DigitCNN
from .utils import ensure_dir, save_json, seed_everything

# Matriz experimental: cada fila varia UN factor respecto al baseline.
EXPERIMENTS = [
    {"name": "1_baseline",     "optimizer": "adam", "dropout": 0.3, "augment": True,  "batchnorm": True},
    {"name": "2_sin_dropout",  "optimizer": "adam", "dropout": 0.0, "augment": True,  "batchnorm": True},
    {"name": "3_sgd",          "optimizer": "sgd",  "dropout": 0.3, "augment": True,  "batchnorm": True},
    {"name": "4_sin_augment",  "optimizer": "adam", "dropout": 0.3, "augment": False, "batchnorm": True},
    {"name": "5_sin_batchnorm","optimizer": "adam", "dropout": 0.3, "augment": True,  "batchnorm": False},
]


def make_optimizer(name: str, params, lr: float, wd: float):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def _build_loaders(cfg: dict, augment: bool, max_train: int | None, bs: int):
    processed = resolve(cfg["data"]["processed_dir"])
    cfg_task = copy.deepcopy(cfg)
    cfg_task["training"]["augment"] = augment
    if not augment:
        cfg_task["training"]["augment_rotation"] = False
        cfg_task["training"]["augment_zoom"] = False

    train_ds = SVHNTensors(processed / "train.pt", build_transforms(cfg_task, train=True))
    val_ds = SVHNTensors(processed / "val.pt", build_transforms(cfg_task, train=False))
    test_ds = SVHNTensors(processed / "test.pt", build_transforms(cfg_task, train=False))
    if max_train:
        train_ds = Subset(train_ds, range(min(max_train, len(train_ds))))

    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(val_ds, batch_size=bs, shuffle=False),
        DataLoader(test_ds, batch_size=bs, shuffle=False),
    )


def run_experiment(spec: dict, cfg: dict, epochs: int, max_train: int | None, device):
    seed_everything(cfg["project"]["seed"])
    tcfg = cfg["training"]
    bs = tcfg["batch_size"]

    train_loader, val_loader, test_loader = _build_loaders(
        cfg, spec["augment"], max_train, bs
    )
    model = DigitCNN(
        num_classes=cfg["business"]["num_classes"],
        p_dropout=spec["dropout"],
        use_batchnorm=spec["batchnorm"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(spec["optimizer"], model.parameters(),
                               tcfg["learning_rate"], tcfg["weight_decay"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        print(f"  [{spec['name']}] epoca {epoch}/{epochs} "
              f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    secs = round(time.time() - t0, 1)
    return {
        "name": spec["name"],
        "config": spec,
        "epochs": epochs,
        "n_train": len(train_loader.dataset),
        "history": history,
        "best_val_acc": round(max(history["val_acc"]), 4),
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "test_acc": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "gap_train_val": round(history["train_acc"][-1] - history["val_acc"][-1], 4),
        "seconds": secs,
    }


def _write_table(results: list[dict], out_dir: Path) -> Path:
    headers = ["Experimento", "Dropout", "Optimizador", "Aug (rot/zoom)",
               "BatchNorm", "Train acc", "Val acc", "Test acc", "Gap (overfit)"]
    rows = []
    for r in results:
        c = r["config"]
        rows.append([
            r["name"],
            "si" if c["dropout"] > 0 else "no",
            c["optimizer"].upper(),
            "si" if c["augment"] else "no",
            "si" if c["batchnorm"] else "no",
            f"{r['final_train_acc']:.3f}",
            f"{r['final_val_acc']:.3f}",
            f"{r['test_acc']:.3f}",
            f"{r['gap_train_val']:+.3f}",
        ])

    md = out_dir / "comparison.md"
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("|" + "---|" * len(headers) + "\n")
        for row in rows:
            fh.write("| " + " | ".join(row) + " |\n")

    csv = out_dir / "comparison.csv"
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(",".join(headers) + "\n")
        for row in rows:
            fh.write(",".join(row) + "\n")
    return csv


def _plot_bars(results: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    val = [r["final_val_acc"] for r in results]
    test = [r["test_acc"] for r in results]
    x = range(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - width / 2 for i in x], val, width, label="Val acc", color="#4C72B0")
    ax.bar([i + width / 2 for i in x], test, width, label="Test acc", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fase C - Comparativa de experimentos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=120)
    plt.close(fig)


def run(epochs: int = 4, max_train: int | None = 8000) -> list[dict]:
    cfg = load_config()
    device = get_device()
    out_dir = ensure_dir(resolve(cfg["paths"]["reports_dir"]) / "experiments")

    print("=" * 68)
    print(f"FASE C - EXPERIMENTOS ({len(EXPERIMENTS)} corridas, "
          f"{epochs} epocas, n_train<={max_train}, device={device})")
    print("=" * 68)

    results = []
    for spec in EXPERIMENTS:
        print(f"-> {spec['name']}")
        res = run_experiment(spec, cfg, epochs, max_train, device)
        results.append(res)
        save_json(res, out_dir / f"{spec['name']}.json")
        print(f"   test_acc={res['test_acc']:.4f} ({res['seconds']}s)")

    _write_table(results, out_dir)
    _plot_bars(results, out_dir)
    save_json({"experiments": results}, out_dir / "all_experiments.json")
    print(f"Evidencia: {out_dir}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-train", type=int, default=8000)
    args = ap.parse_args()
    run(epochs=args.epochs, max_train=args.max_train)
