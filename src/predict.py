"""O.E.6 - Inferencia y exportacion (Sprint 3).

Incluye:
  * ``Predictor``: carga el checkpoint y predice sobre una imagen (usado por la API).
  * exportacion del modelo a ONNX para despliegue portable.
  * CLI: ``python -m src.predict --export`` y ``python -m src.predict --image ruta.png``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import get_paths, load_config
from .engine import get_device
from .models import build_model


class Predictor:
    """Envoltura de inferencia: imagen PIL/np -> (digito, confianza)."""

    def __init__(self, model_path: str | Path):
        self.device = get_device()
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = build_model(ckpt["model_name"], num_classes=ckpt["num_classes"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.mean = ckpt.get("mean", [0.4377, 0.4438, 0.4728])
        self.std = ckpt.get("std", [0.1980, 0.2010, 0.1970])
        self.image_size = ckpt.get("image_size", 32)
        self.val_acc = ckpt.get("val_acc")

    def _to_tensor(self, image) -> torch.Tensor:
        from PIL import Image
        from torchvision import transforms

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        image = image.convert("RGB").resize((self.image_size, self.image_size))
        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )
        return tf(image).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image) -> dict:
        tensor = self._to_tensor(image)
        probs = F.softmax(self.model(tensor), dim=1).squeeze(0).cpu().numpy()
        pred = int(probs.argmax())
        return {
            "digit": pred,
            "confidence": round(float(probs[pred]), 4),
            "probabilities": [round(float(p), 4) for p in probs],
            "val_acc": self.val_acc,
        }


def export_onnx(model_path: str | Path, onnx_path: str | Path) -> Path:
    device = get_device()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, ckpt.get("image_size", 32), ckpt.get("image_size", 32), device=device)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    return onnx_path


def main() -> None:
    import argparse

    cfg = load_config()
    paths = get_paths(cfg)

    parser = argparse.ArgumentParser(description="Inferencia / exportacion SVHN")
    parser.add_argument("--image", type=str, help="ruta de una imagen para predecir")
    parser.add_argument("--export", action="store_true", help="exportar el modelo a ONNX")
    args = parser.parse_args()

    print("=" * 68)
    print("O.E.6 - DESPLIEGUE / INFERENCIA (Sprint 3)")
    print("=" * 68)

    if args.export:
        out = export_onnx(paths["best_model"], paths["onnx_model"])
        print(f"Modelo exportado a ONNX: {out}")

    if args.image:
        from PIL import Image

        predictor = Predictor(paths["best_model"])
        result = predictor.predict(Image.open(args.image))
        print(f"Imagen : {args.image}")
        print(f"Digito : {result['digit']} (confianza={result['confidence']})")

    if not args.export and not args.image:
        parser.print_help()
    print("=" * 68)


if __name__ == "__main__":
    main()
