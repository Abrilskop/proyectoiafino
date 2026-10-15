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
| O.E.6 | Despliegue ML-Ops | 3 | API `/predict` + `models/model.onnx` + `reports/predictions.jsonl` |

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
├── deployment/{app.py,Dockerfile,docker-compose.yml}
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
# Local
uvicorn deployment.app:app --host 0.0.0.0 --port 8000

# Docker
docker compose -f deployment/docker-compose.yml up --build
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

## Notas

- Sin GPU: se entrena sobre un **subconjunto** (`data.subset_fraction` en el YAML).
- Reproducible: semilla fija (`project.seed`).
- ML-Ops: cada prediccion se registra en `reports/predictions.jsonl` y se le
  aplica un chequeo de drift (`monitoring/drift.py`).
