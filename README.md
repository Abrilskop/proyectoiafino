# SVHN — Reconocimiento de Numeros de Casa (CRISP-ML)

Proyecto de Machine Learning que **reconoce automaticamente el digito de un
numero de casa** en imagenes de Street View (dataset [SVHN](http://ufldl.stanford.edu/housenumbers/)),
con despliegue funcional como API REST.

## Problema

**¿Para que?** Digitalizar numeros de vivienda desde imagenes de Street View.
**¿Para donde?** Poblar bases de direcciones (GIS/OpenStreetMap), apoyar la
logistica de ultima milla y validar direcciones en navegacion.

- Tarea: clasificacion multiclase de imagenes (10 clases: digitos 0-9).
- Datos: SVHN 32x32 RGB (`torchvision.datasets.SVHN`).
- Metodologia: **CRISP-ML**, organizada en 3 Sprints.

## Objetivo General

> **Desarrollar** un sistema inteligente de clasificacion de digitos en imagenes
> de Street View (SVHN) bajo la metodologia CRISP-ML, con despliegue reproducible
> y monitoreo.

## Objetivos Especificos y Evidencias

| O.E. | Fase CRISP-ML | Sprint | Evidencia demostrable |
|------|---------------|--------|-----------------------|
| O.E.1 | Comprension del negocio | 1 | `reports/oe1_business.json` + `configs/config.yaml` |
| O.E.2 | Comprension de los datos | 1 | `reports/oe2_data_summary.json` + `reports/figures/eda_*.png` |
| O.E.3 | Preparacion de los datos | 2 | `data/processed/*.pt` + `reports/oe3_preparation.json` |
| O.E.4 | Modelado | 2 | `models/best.pt` + `reports/oe4_training.json` + `reports/figures/training_curves.png` |
| O.E.5 | Evaluacion | 3 | `reports/metrics.json` + `reports/figures/confusion_matrix.png` |
| O.E.6 | Despliegue ML-Ops | 3 | API `/predict` + UI web en `/ui` (Gradio) + `models/model.onnx` + `reports/predictions.jsonl` |

Notebooks de sustentacion (una por objetivo): `notebooks/01..05_*.ipynb` y
`notebooks/06_demo_despliegue.ipynb`.

### Resultado de la demo (CPU, subconjunto 20 %)

| Metrica | Valor |
|---------|-------|
| Accuracy test | 0.8399 |
| Macro-F1 test | 0.8240 |
| KPI (>= 0.80) | Cumplido |
| Modelo | `models/best.pt` (CNN, 8 epocas) |
| Latencia / ONNX | `models/model.onnx` |

## Estructura

```
.
├── configs/config.yaml          # KPIs y parametros (fuente unica de verdad)
├── data/{raw,processed}/        # datos crudos y tensores preparados
├── notebooks/                   # evidencia por objetivo
├── src/
│   ├── business.py              # O.E.1
│   ├── data/{download,eda,preprocess,dataset}.py
│   ├── models/{cnn,resnet}.py
│   ├── engine.py  train.py  evaluate.py  predict.py
├── deployment/{app.py,gradio_app.py,Dockerfile,docker-compose.yml}
│   └── hf_space/                # plantilla lista para Hugging Face Spaces
├── monitoring/{logging,drift}.py
├── tests/
├── models/  reports/
```

## Instalacion

```powershell
# (recomendado) entorno virtual
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# PyTorch CPU (mucho mas ligero)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Ejecucion por objetivo (Makefile)

```powershell
make oe1     # O.E.1  Negocio  -> reports/oe1_business.json
make oe2     # O.E.2  Datos    -> descarga SVHN + EDA (figuras)
make oe3     # O.E.3  Preparar -> data/processed/*.pt
make oe4     # O.E.4  Modelar  -> models/best.pt + curvas
make oe5     # O.E.5  Evaluar  -> reports/metrics.json + confusion
make oe6     # O.E.6  Desplegar-> models/model.onnx
make test    # pytest
```

> En Windows puedes invocar los modulos directamente, p. ej.
> `python -m src.train`.

## Despliegue

```powershell
# Local (solo API)
uvicorn deployment.app:app --host 0.0.0.0 --port 8000

# Local (API + interfaz web en http://127.0.0.1:7860/ui)
.\run.ps1 -Ui

# Docker
docker compose -f deployment/docker-compose.yml up --build
```

Tambien puedes correr solo la UI apuntando a una API existente:

```powershell
$env:SVHN_API_URL = "http://127.0.0.1:8000"
python deployment/gradio_app.py     # UI en http://127.0.0.1:7860
```

Probar la API:

```powershell
curl -F "file=@ruta_a_imagen.png" http://localhost:8000/predict
```

Respuesta:

```json
{
  "filename": "sample.png",
  "prediction": 3,
  "confidence": 0.98,
  "probabilities": [ ... 10 valores ... ],
  "drift": { "drift_detected": false, "drift_score": 0.01 }
}
```

## Como probarlo online (Hugging Face Spaces)

La carpeta `deployment/hf_space/` es una plantilla lista para publicar el demo
(sdk `docker`, `app_port` 7860, UI montada en `/ui`):

```powershell
.\deployment\hf_space\sync.ps1                     # copia codigo + models/best.pt
cd deployment\hf_space
git init
git remote add origin https://huggingface.co/spaces/<USUARIO>/<NOMBRE>
git add . ; git commit -m "chore(space): despliegue SVHN"
git push -u origin main
```

Una vez publicado, el demo queda en `https://huggingface.co/spaces/<USUARIO>/<NOMBRE>`.

## Actividad: Diseño y Optimización de una CNN

Entregables de la actividad de laboratorio (Fases A-C):

| Fase | Contenido | Evidencia |
|------|-----------|-----------|
| A | Preparación y preprocesamiento | `configs/config.yaml`, `data/processed/*.pt` |
| B | Modelo base (Conv 3×3 + MaxPool + Softmax + BatchNorm) | `src/models/cnn.py` |
| C | Experimentos (Dropout, SGD vs Adam, Augmentation, BatchNorm) | `src/experiments.py` → `reports/experiments/` |

- **Notebook ejecutado:** `notebooks/actividad_cnn.ipynb`
- **Informe PDF:** `reports/actividad_cnn.pdf`
- Correr experimentos: `python -m src.experiments --epochs 4 --max-train 8000`
- Generar el PDF: `python -m src.report`

## Notas

- Sin GPU: se entrena sobre un **subconjunto** (`data.subset_fraction` en el YAML).
- Reproducible: semilla fija (`project.seed`).
- ML-Ops: cada prediccion se registra en `reports/predictions.jsonl` y se le
  aplica un chequeo de drift (`monitoring/drift.py`).
