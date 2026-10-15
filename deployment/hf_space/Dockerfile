# O.E.6 - Imagen de despliegue para Hugging Face Spaces (sdk: docker).
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    SVHN_MOUNT_UI=1 \
    SVHN_API_URL=http://127.0.0.1:7860 \
    PORT=7860

RUN pip install --no-cache-dir --upgrade pip

# PyTorch en version CPU (mucho mas liviano para un Space).
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "7860"]
