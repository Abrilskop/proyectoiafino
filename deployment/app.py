"""O.E.6 - API de inferencia (FastAPI).

Expone el modelo entrenado como servicio REST:
  * GET  /health   -> estado del servicio y del modelo
  * POST /predict  -> imagen -> digito reconocido + confianza
Cada peticion se registra (logging) y se le aplica un chequeo de drift.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Permite ejecutar tanto `uvicorn deployment.app:app` como `python deployment/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from monitoring.drift import check_drift  # noqa: E402
from monitoring.logging import PredictionLogger  # noqa: E402
from src.config import get_paths, load_config  # noqa: E402
from src.predict import Predictor  # noqa: E402


def create_app(model_path: str | Path | None = None) -> FastAPI:
    cfg = load_config()
    paths = get_paths(cfg)
    model_path = Path(model_path) if model_path else paths["best_model"]

    app = FastAPI(
        title="SVHN House Number Recognition API",
        description="Clasifica el digito de un numero de casa (dataset SVHN).",
        version=cfg["project"]["version"],
    )

    logger = PredictionLogger()
    predictor = None
    load_error = None
    try:
        predictor = Predictor(model_path)
    except Exception as exc:  # modelo no entrenado aun
        load_error = str(exc)

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok" if predictor else "model_unavailable",
            "model_path": str(model_path),
            "model_loaded": predictor is not None,
            "error": load_error,
        }

    @app.get("/")
    def root() -> dict:
        return {"service": cfg["project"]["name"], "version": cfg["project"]["version"]}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        if predictor is None:
            raise HTTPException(status_code=503, detail=f"Modelo no disponible: {load_error}")
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Se esperaba un archivo de imagen.")

        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")

        result = predictor.predict(image)
        drift = check_drift(image)
        logger.log(
            {
                "filename": file.filename,
                "digit": result["digit"],
                "confidence": result["confidence"],
                "drift_detected": drift["drift_detected"],
                "drift_score": drift["drift_score"],
            }
        )
        return {
            "filename": file.filename,
            "prediction": result["digit"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "drift": drift,
        }

    # Monta opcionalmente la UI de Gradio en /ui.
    # Se activa con SVHN_MOUNT_UI=1 y solo si gradio esta instalado, para que
    # los tests sigan corriendo sin depender de gradio.
    mount_ui = os.environ.get("SVHN_MOUNT_UI") == "1"
    if mount_ui:
        try:
            import gradio as gr

            from deployment.gradio_app import build_ui

            gr.mount_gradio_app(app, build_ui(), path="/ui")
        except Exception as exc:  # noqa: BLE001
            app.state.ui_error = str(exc)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
