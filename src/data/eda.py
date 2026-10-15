"""O.E.2 - Analisis exploratorio de datos (EDA) (Sprint 1).

Genera y persiste las figuras de evidencia en ``reports/figures``:
  * ``eda_class_distribution.png`` - balance de clases por split
  * ``eda_samples.png``           - rejilla de imagenes de ejemplo
  * ``eda_channel_stats.png``     - media/desviacion por canal

Tambien escribe ``reports/oe2_data_summary.json``.
"""
from __future__ import annotations

import numpy as np

from ..config import get_paths, load_config, resolve
from ..utils import ensure_dir, save_json


def _load_split(split: str):
    from torchvision.datasets import SVHN

    cfg = load_config()
    ds = SVHN(root=str(resolve(cfg["data"]["raw_dir"])), split=split, download=True)
    images = np.ascontiguousarray(ds.data)  # (N, 3, 32, 32) uint8 (CHW)
    labels = np.asarray(ds.labels)
    labels = np.where(labels == 10, 0, labels)  # seguridad: 0 <-> 10
    return images, labels


def run(figure: bool = True) -> dict:
    cfg = load_config()
    paths = get_paths(cfg)
    figs = ensure_dir(paths["figures_dir"])

    print("=" * 68)
    print("O.E.2 - COMPRENSION DE LOS DATOS (Sprint 1)")
    print("=" * 68)

    summary = {"objetivo_especifico": "O.E.2 - Comprender los datos", "splits": {}}
    train_images = train_labels = None

    for split in ("train", "test"):
        images, labels = _load_split(split)
        counts = np.bincount(labels, minlength=10).tolist()
        mean = images.mean(axis=(0, 2, 3)) / 255.0
        std = images.std(axis=(0, 2, 3)) / 255.0

        summary["splits"][split] = {
            "n_images": int(len(labels)),
            "shape": list(images.shape[1:]),
            "class_counts": counts,
            "mean_rgb": [round(float(v), 4) for v in mean],
            "std_rgb": [round(float(v), 4) for v in std],
            "imbalance_ratio": round(float(max(counts) / max(min(counts), 1)), 3),
        }
        print(f"[{split}] n={len(labels):,} shape={images.shape[1:]} "
              f"imbalance={summary['splits'][split]['imbalance_ratio']}")
        print(f"        mean_rgb={summary['splits'][split]['mean_rgb']} "
              f"std_rgb={summary['splits'][split]['std_rgb']}")
        if split == "train":
            train_images, train_labels = images, labels

    if figure:
        _make_figures(figs, summary, train_images, train_labels)

    out = paths["reports_dir"] / "oe2_data_summary.json"
    save_json(summary, out)
    print(f"Evidencia EDA: {figs}")
    print(f"Resumen JSON : {out}")
    print("=" * 68)
    return summary


def _make_figures(figs, summary, train_images, train_labels) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) Distribucion de clases
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    x = np.arange(10)
    for i, split in enumerate(("train", "test")):
        counts = summary["splits"][split]["class_counts"]
        norm = np.array(counts) / max(counts)
        ax.bar(x + (i - 0.5) * width, norm, width, label=split)
    ax.set_xticks(x)
    ax.set_title("Distribucion de clases (normalizada) - SVHN")
    ax.set_xlabel("Digito")
    ax.set_ylabel("Frecuencia relativa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "eda_class_distribution.png", dpi=120)
    plt.close(fig)

    # 2) Rejilla de ejemplos
    rng = np.random.default_rng(42)
    idx = rng.choice(len(train_images), size=16, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(np.transpose(train_images[i], (1, 2, 0)))
        ax.set_title(int(train_labels[i]), fontsize=9)
        ax.axis("off")
    fig.suptitle("Ejemplos de imagenes SVHN (32x32)")
    fig.tight_layout()
    fig.savefig(figs / "eda_samples.png", dpi=120)
    plt.close(fig)

    # 3) Media / desviacion por canal
    splits = list(summary["splits"].keys())
    means = np.array([summary["splits"][s]["mean_rgb"] for s in splits])
    stds = np.array([summary["splits"][s]["std_rgb"] for s in splits])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ["R", "G", "B"]
    x = np.arange(3)
    for i, s in enumerate(splits):
        axes[0].bar(x + (i - 0.5) * 0.35, means[i], 0.35, label=s)
        axes[1].bar(x + (i - 0.5) * 0.35, stds[i], 0.35, label=s)
    axes[0].set_title("Media por canal")
    axes[1].set_title("Desviacion por canal")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "eda_channel_stats.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
