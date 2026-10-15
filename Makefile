# ===========================================================================
# SVHN - Reconocimiento de numeros de casa (CRISP-ML)
# Cada objetivo especifico (O.E.) tiene su propio target.
# ===========================================================================

PYTHON ?= python

.PHONY: help install oe1 oe2 oe3 oe4 oe5 oe6 test serve clean

help:
	@echo "Targets disponibles:"
	@echo "  make install  - instalar dependencias"
	@echo "  make oe1      - O.E.1 Comprension del negocio"
	@echo "  make oe2      - O.E.2 Comprension de los datos (descarga + EDA)"
	@echo "  make oe3      - O.E.3 Preparacion de los datos"
	@echo "  make oe4      - O.E.4 Modelado (entrenamiento)"
	@echo "  make oe5      - O.E.5 Evaluacion"
	@echo "  make oe6      - O.E.6 Despliegue (export ONNX)"
	@echo "  make serve    - levantar la API FastAPI"
	@echo "  make test     - ejecutar pruebas"

install:
	$(PYTHON) -m pip install -r requirements.txt

oe1:
	$(PYTHON) -m src.business

oe2:
	$(PYTHON) -m src.data.download
	$(PYTHON) -m src.data.eda

oe3:
	$(PYTHON) -m src.data.preprocess

oe4:
	$(PYTHON) -m src.train

oe5:
	$(PYTHON) -m src.evaluate

oe6:
	$(PYTHON) -m src.predict --export

serve:
	$(PYTHON) -m uvicorn deployment.app:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest -q

clean:
	@echo "Limpiando artefactos temporales..."
