"""O.E.3 - Preparacion de los datos (Sprint 2).

Toma los ``.mat`` crudos, construye un subconjunto reproducible del split de
entrenamiento (para poder entrenar en CPU sin GPU), separa validacion y
persiste los tensores crudos (uint8) listos para el pipeline de PyTorch en
``data/processed``.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import get_paths, load_config, resolve
from ..utils import ensure_dir, save_json, seed_everything


def _load_split(split: str):
    from torchvision.datasets import SVHN

    cfg = load_config()
    ds = SVHN(root=str(resolve(cfg["data"]["raw_dir"])), split=split, download=True)
    images = np.ascontiguousarray(ds.data)  # (N, 32, 32, 3) uint8
    labels = np.asarray(ds.labels, dtype=np.int64)
    labels = np.where(labels == 10, 0, labels)
    return images, labels


def run() -> dict:
    cfg = load_config()
    paths = get_paths(cfg)
    seed_everything(cfg["project"]["seed"])

    out_dir = ensure_dir(resolve(cfg["data"]["processed_dir"]))
    frac = float(cfg["data"]["subset_fraction"])
    val_frac = float(cfg["data"]["val_fraction"])
    rng = np.random.default_rng(cfg["project"]["seed"])

    print("=" * 68)
    print("O.E.3 - PREPARACION DE LOS DATOS (Sprint 2)")
    print("=" * 68)

    # --- entrenamiento (subconjunto) ---
    tr_images, tr_labels = _load_split("train")
    order = rng.permutation(len(tr_labels))
    n_subset = int(len(order) * frac)
    subset_idx = order[:n_subset]
    images, labels = tr_images[subset_idx], tr_labels[subset_idx]

    n_val = int(len(labels) * val_frac)
    val_idx, train_idx = subset_idx[:n_val], subset_idx[n_val:]
    train_images, train_labels = tr_images[train_idx], tr_labels[train_idx]
    val_images, val_labels = tr_images[val_idx], tr_labels[val_idx]

    # --- test completo ---
    test_images, test_labels = _load_split("test")

    splits = {
        "train": (train_images, train_labels),
        "val": (val_images, val_labels),
        "test": (test_images, test_labels),
    }

    summary = {"objetivo_especifico": "O.E.3 - Preparacion de los datos", "splits": {}}
    for name, (imgs, lbls) in splits.items():
        torch.save(
            {"images": torch.from_numpy(imgs), "labels": torch.from_numpy(lbls)},
            out_dir / f"{name}.pt",
        )
        counts = np.bincount(lbls, minlength=10).tolist()
        summary["splits"][name] = {"n": int(len(lbls)), "class_counts": counts}
        print(f"[{name:5s}] {len(lbls):>7,} imagenes -> {out_dir / (name + '.pt')}")

    summary["subset_fraction"] = frac
    summary["val_fraction"] = val_frac
    out = paths["reports_dir"] / "oe3_preparation.json"
    save_json(summary, out)
    print(f"Evidencia: {out}")
    print("=" * 68)
    return summary


if __name__ == "__main__":
    run()
