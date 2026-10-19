"""Datasets y DataLoaders de PyTorch.

Carga los tensores crudos generados por ``preprocess.py`` y aplica el pipeline
de transformaciones (aumento ligero en train, normalizacion en todos).
Nota: no se usan volteos horizontales porque invertirian la identidad del
digito (p. ej. un 6 pasaria a parecer un 9).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import load_config, resolve


class SVHNTensors(Dataset):
    """Dataset sobre los tensores guardados en ``data/processed/*.pt``.

    Los tensores se almacenan como ``uint8`` en formato CHW: ``(N, 3, 32, 32)``.
    """

    def __init__(self, path: str | Path, transform=None):
        blob = torch.load(path, weights_only=True)
        self.images = blob["images"].numpy()  # (N, 3, 32, 32) uint8
        self.labels = blob["labels"].numpy().astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx]).clone()  # CHW uint8
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(cfg: dict, train: bool):
    from torchvision import transforms

    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]
    size = cfg["data"]["image_size"]

    steps = [transforms.ConvertImageDtype(torch.float32)]
    if train and cfg["training"].get("augment", True):
        steps.append(transforms.RandomCrop(size, padding=4))
        if cfg["training"].get("augment_rotation", False):
            steps.append(transforms.RandomRotation(15))
        if cfg["training"].get("augment_zoom", False):
            steps.append(transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)))
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def get_dataloaders(cfg: dict | None = None):
    """Devuelve ``(train_loader, val_loader, test_loader)``."""
    cfg = cfg or load_config()
    processed = resolve(cfg["data"]["processed_dir"])
    bs = cfg["training"]["batch_size"]
    nw = cfg["data"]["num_workers"]

    train_ds = SVHNTensors(processed / "train.pt", build_transforms(cfg, train=True))
    val_ds = SVHNTensors(processed / "val.pt", build_transforms(cfg, train=False))
    test_ds = SVHNTensors(processed / "test.pt", build_transforms(cfg, train=False))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader
