"""O.E.6 - Interfaz web (Gradio) para el reconocedor de numeros SVHN.

Dos modos de uso:
  * Subir foto    -> envia la imagen a la API /predict.
  * Dibujar       -> dibuja un digito en el canvas y lo envia a la API.

La interfaz NO carga el modelo: solo consume la API REST (por defecto la
misma instancia montada en FastAPI). Esto mantiene una unica fuente de
verdad para la inferencia, logging y drift (src/predict.py).

Variables de entorno:
  SVHN_API_URL  URL base de la API (default: http://127.0.0.1:8000)
  PORT          puerto cuando se ejecuta `python deployment/gradio_app.py`
"""
from __future__ import annotations

import io
import os

import gradio as gr
import numpy as np
import requests
from PIL import Image

API_URL = os.environ.get("SVHN_API_URL", "http://127.0.0.1:8000").rstrip("/")

AVISO = (
    "**Aviso:** el modelo se entreno con fotos reales de numeros de casa del "
    "dataset SVHN, no con dibujos. El modo *Dibujar* es una demo y puede "
    "acertar poco."
)


def _to_pil(image) -> Image.Image | None:
    """Normaliza la entrada de Gradio (PIL o numpy, con/sin alpha) a RGB."""
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    if image.mode == "RGBA":
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg
    else:
        image = image.convert("RGB")
    return image


def predict(image) -> tuple[str, dict]:
    """Llama a la API /predict y devuelve (mensaje, probabilidades)."""
    pil = _to_pil(image)
    if pil is None:
        return "Sube o dibuja una imagen para comenzar.", {}

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)

    try:
        resp = requests.post(
            f"{API_URL}/predict",
            files={"file": ("input.png", buf, "image/png")},
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as exc:  # API apagada o error de red
        return f"No se pudo contactar la API en {API_URL}: {exc}", {}

    data = resp.json()
    probs = {str(i): float(p) for i, p in enumerate(data["probabilities"])}
    msg = (
        f"Digito reconocido: **{data['prediction']}**  "
        f"(confianza {data['confidence'] * 100:.1f}% | "
        f"drift: {data['drift']['drift_detected']})"
    )
    return msg, probs


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SVHN - Reconocedor de numeros de casa") as demo:
        gr.Markdown(
            "# Reconocedor de numeros de casa (SVHN)\n"
            "Sube una foto de un numero de casa o dibuja un digito. "
            "El modelo devuelve la clase predicha y su distribucion de "
            "probabilidad."
        )

        with gr.Tab("Subir foto"):
            with gr.Row():
                foto = gr.Image(type="pil", label="Imagen del numero")
                with gr.Column():
                    foto_msg = gr.Markdown()
                    foto_out = gr.Label(num_top_classes=10, label="Probabilidades")
            foto_btn = gr.Button("Predecir", variant="primary")
            foto_btn.click(predict, inputs=foto, outputs=[foto_msg, foto_out])

        with gr.Tab("Dibujar"):
            with gr.Row():
                pad = gr.Sketchpad(
                    type="pil",
                    label="Dibuja un digito",
                    height=280,
                    width=280,
                )
                with gr.Column():
                    pad_msg = gr.Markdown()
                    pad_out = gr.Label(num_top_classes=10, label="Probabilidades")
            pad_btn = gr.Button("Predecir", variant="primary")
            pad_btn.click(predict, inputs=pad, outputs=[pad_msg, pad_out])

        gr.Markdown(AVISO)

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
