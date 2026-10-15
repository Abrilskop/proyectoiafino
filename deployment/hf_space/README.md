---
title: SVHN House Number Recognition
emoji: 🔢
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Reconocedor de numeros de casa (SVHN)

Demo desplegable en Hugging Face Spaces de un clasificador de digitos
entrenado sobre el dataset SVHN (Street View House Numbers).

- `GET /health` — estado del servicio y del modelo.
- `POST /predict` — imagen -> digito + confianza + probabilidades.
- `/ui` — interfaz web (subir foto + dibujar digito).

**Aviso:** el modelo se entreno con fotos reales del dataset SVHN, no con
dibujos; el modo *Dibujar* es una demo.

## Publicar este Space

```powershell
.\deployment\hf_space\sync.ps1          # copia codigo + modelo desde el proyecto
cd deployment\hf_space
git init
git remote add origin https://huggingface.co/spaces/<USUARIO>/<NOMBRE>
git add .
git commit -m "chore(space): despliegue SVHN"
git push -u origin main
```
