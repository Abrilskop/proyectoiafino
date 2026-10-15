"""O.E.4 - Modelado (Sprint 2).

Entrena el modelo sobre el subconjunto preparado (CPU-friendly), guarda el
mejor checkpoint en ``models/best.pt`` y persiste la curva de aprendizaje en
``reports/figures/training_curves.png`` y un log en ``reports/oe4_training.json``.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from .config import get_paths, load_config
from .data.dataset import get_dataloaders
from .engine import evaluate, get_device, train_one_epoch
from .models import build_model
from .utils import ensure_dir, save_json, seed_everything


def run() -> dict:
    cfg = load_config()
    paths = get_paths(cfg)
    seed_everything(cfg["project"]["seed"])
    device = get_device()

    print("=" * 68)
    print("O.E.4 - MODELADO (Sprint 2)")
    print("=" * 68)
    print(f"Dispositivo: {device}")

    train_loader, val_loader, _ = get_dataloaders(cfg)
    tcfg = cfg["training"]

    model = build_model(
        tcfg["model"], num_classes=cfg["business"]["num_classes"], pretrained=False
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    ensure_dir(paths["model_dir"])

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoca {epoch:02d}/{tcfg['epochs']} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | {time.time() - t0:.1f}s"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": tcfg["model"],
                    "num_classes": cfg["business"]["num_classes"],
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "mean": cfg["data"]["mean"],
                    "std": cfg["data"]["std"],
                    "image_size": cfg["data"]["image_size"],
                },
                paths["best_model"],
            )

    _save_curves(paths["figures_dir"], history)

    evidence = {
        "objetivo_especifico": "O.E.4 - Modelar",
        "sprint": 2,
        "model": tcfg["model"],
        "epochs": tcfg["epochs"],
        "batch_size": tcfg["batch_size"],
        "learning_rate": tcfg["learning_rate"],
        "best_val_acc": round(best_acc, 4),
        "history": history,
        "checkpoint": str(paths["best_model"]),
    }
    out = paths["reports_dir"] / "oe4_training.json"
    save_json(evidence, out)
    print(f"Mejor val_acc={best_acc:.4f} -> {paths['best_model']}")
    print(f"Evidencia: {out}")
    print("=" * 68)
    return evidence


def _save_curves(figs_dir, history) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ensure_dir(figs_dir)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], "o-", label="train")
    axes[0].plot(epochs, history["val_loss"], "s-", label="val")
    axes[0].set_title("Perdida")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], "o-", label="train")
    axes[1].plot(epochs, history["val_acc"], "s-", label="val")
    axes[1].set_title("Exactitud")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figs / "training_curves.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
