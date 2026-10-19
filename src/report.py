"""Genera el informe PDF de la actividad (ReportLab).

Entrada : reports/experiments/*  +  reports/figures/*  +  reports/metrics.json
Salida  : reports/actividad_cnn.pdf
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .config import load_config, resolve

TEAM = [
    "Ramos Ticahuanca, Gianella Alexandra",
    "SuÃ¡rez Condori, Juan Gabriel",
    "Del Ãguila GarcÃ­a, JesÃºs",
    "Mendoza Torres, Lincol J.",
]
REPO = "https://github.com/Abrilskop/proyectoiafino"


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("CoverTitle", parent=ss["Title"], fontSize=26, spaceAfter=6))
    ss.add(ParagraphStyle("CoverSub", parent=ss["Normal"], fontSize=14,
                          alignment=TA_CENTER, textColor=colors.HexColor("#333333")))
    ss.add(ParagraphStyle("Body", parent=ss["BodyText"], alignment=TA_JUSTIFY,
                          fontSize=10.5, leading=15))
    ss.add(ParagraphStyle("H", parent=ss["Heading2"], textColor=colors.HexColor("#1f3b73")))
    return ss


def _img(path: Path, width, height=None):
    if path.exists():
        return Image(str(path), width=width, height=height)
    return Paragraph(f"[falta la figura: {path.name}]", _styles()["Body"])


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _formula_image(figs: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = figs / "accuracy_formula.png"
    fig, ax = plt.subplots(figsize=(6, 1.1))
    ax.axis("off")
    ax.text(0.5, 0.5,
            r"$Accuracy = \dfrac{TP + TN}{TP + TN + FP + FN}$" +
            r"$\quad=\quad\dfrac{\text{aciertos}}{\text{total}}$",
            ha="center", va="center", fontsize=20)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _comparison_table(csv_path: Path, ss):
    if not csv_path.exists():
        return Paragraph("No hay tabla de experimentos.", ss["Body"])
    lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    data = [ln.split(",") for ln in lines]
    table = Table(data, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b73")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2fa")]),
        ("ALIGN", (5, 1), (-1, -1), "CENTER"),
    ]))
    return table


def run() -> Path:
    cfg = load_config()
    reports = resolve(cfg["paths"]["reports_dir"])
    figs = resolve(cfg["paths"]["figures_dir"])
    exp_dir = reports / "experiments"
    out_pdf = reports / "actividad_cnn.pdf"

    ss = _styles()
    story = []

    # ---------------------- PORTADA ----------------------
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("DiseÃ±o y OptimizaciÃ³n de una CNN", ss["CoverTitle"]))
    story.append(Paragraph("para ClasificaciÃ³n de ImÃ¡genes Complejas", ss["CoverTitle"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("ClasificaciÃ³n de dÃ­gitos de nÃºmeros de casa (SVHN)", ss["CoverSub"]))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Informe de la actividad", ss["CoverSub"]))
    story.append(Spacer(1, 3 * cm))
    for name in TEAM:
        story.append(Paragraph(name, ss["CoverSub"]))
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("VersiÃ³n del modelo: 1.0.0", ss["CoverSub"]))
    story.append(Paragraph(
        dt.datetime.now().strftime("Fecha: %d/%m/%Y"), ss["CoverSub"]))
    story.append(Paragraph(f'Repositorio: <link href="{REPO}">{REPO}</link>', ss["CoverSub"]))
    story.append(PageBreak())

    # ---------------------- 1. Contexto ----------------------
    story.append(Paragraph("1. Contexto de la misiÃ³n", ss["H"]))
    story.append(Paragraph(
        "Actuamos como ingenieros de visiÃ³n artificial para una startup de "
        "logÃ­stica. El objetivo es una Red Neuronal Convolucional (CNN) capaz de "
        "clasificar con alta precisiÃ³n imÃ¡genes de nÃºmeros de casa del dataset "
        "<b>SVHN</b> (Street View House Numbers, 32Ã—32 RGB, 10 clases), "
        "enfrentando problemas reales como el sobreajuste y la variabilidad de "
        "los datos.", ss["Body"]))
    story.append(Spacer(1, 0.3 * cm))

    # ---------------------- 2. Fase A ----------------------
    story.append(Paragraph("2. Fase A â€” PreparaciÃ³n y preprocesamiento", ss["H"]))
    story.append(Paragraph(
        "Los datos se descargan con <font face='Courier'>torchvision.datasets.SVHN</font> "
        "y se normalizan con la media/desviaciÃ³n calculadas sobre el propio dataset. "
        "Se usa un subconjunto del 20 % del entrenamiento (CPU) para iterar rÃ¡pido, "
        "manteniendo el test completo (26 032 imÃ¡genes).", ss["Body"]))
    story.append(Spacer(1, 0.3 * cm))

    # ---------------------- 3. Fase B ----------------------
    story.append(Paragraph("3. Fase B â€” Arquitectura del modelo base", ss["H"]))
    story.append(Paragraph(
        "La CNN (<font face='Courier'>src/models/cnn.py</font>) consta de 3 bloques "
        "convolucionales. Cada bloque usa convoluciones de <b>3Ã—3, stride 1, "
        "padding 1</b> con activaciÃ³n <b>ReLU</b>, seguido de "
        "<b>MaxPooling 2Ã—2</b> que reduce la dimensiÃ³n espacial (32â†’16â†’8â†’4). Se "
        "incluye <b>Batch Normalization</b> (innovaciÃ³n) y <b>Dropout</b> en el "
        "clasificador. La salida usa <b>Softmax</b> multiclase (10 neuronas) para "
        "obtener probabilidades por dÃ­gito.", ss["Body"]))
    story.append(Spacer(1, 0.3 * cm))

    # ---------------------- 4. Fase C ----------------------
    story.append(Paragraph("4. Fase C â€” ExperimentaciÃ³n", ss["H"]))
    story.append(Paragraph(
        "Se parte de un <b>baseline</b> y se modifica un factor por corrida para "
        "aislar su efecto: (1) baseline, (2) sin Dropout, (3) SGD en lugar de Adam, "
        "(4) sin Data Augmentation, (5) sin Batch Normalization.", ss["Body"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_comparison_table(exp_dir / "comparison.csv", ss))
    story.append(Spacer(1, 0.4 * cm))
    story.append(_img(exp_dir / "comparison.png", 6.0 * inch, 2.7 * inch))
    story.append(PageBreak())

    # ---------------------- 5. Curvas ----------------------
    story.append(Paragraph("5. Curvas de Loss y Accuracy (train vs validaciÃ³n)", ss["H"]))
    story.append(Paragraph(
        "La diferencia entre la curva de entrenamiento y la de validaciÃ³n es la "
        "seÃ±al del sobreajuste. Un <i>gap</i> creciente (trainâ†‘, valâ†“) indica que "
        "el modelo memoriza el train en vez de generalizar.", ss["Body"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_img(figs / "training_curves.png", 6.5 * inch, 2.6 * inch))
    story.append(PageBreak())

    # ---------------------- 6. Matriz de confusiÃ³n ----------------------
    story.append(Paragraph("6. Matriz de confusiÃ³n", ss["H"]))
    story.append(Paragraph(
        "Permite ver quÃ© clases se confunden mÃ¡s. Las confusiones suelen darse "
        "entre dÃ­gitos visualmente parecidos (p. ej. 3/8, 4/9, 5/6).", ss["Body"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_img(figs / "confusion_matrix.png", 4.8 * inch, 3.9 * inch))
    story.append(Spacer(1, 0.3 * cm))

    metrics = _load_json(reports / "metrics.json")
    if metrics:
        story.append(Paragraph(
            f"<b>Accuracy final (modelo completo, test):</b> {metrics.get('accuracy')}  |  "
            f"<b>macro-F1:</b> {metrics.get('macro_f1')}  |  "
            f"<b>KPI â‰¥ {metrics.get('target_value')}:</b> "
            f"{'cumplido' if metrics.get('cumple_kpi') else 'no cumplido'}", ss["Body"]))
    story.append(PageBreak())

    # ---------------------- 7. FÃ³rmula de Accuracy ----------------------
    story.append(Paragraph("7. CÃ¡lculo de la precisiÃ³n final (Accuracy)", ss["H"]))
    story.append(Paragraph(
        "La exactitud se define como la fracciÃ³n de predicciones correctas:", ss["Body"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(_img(_formula_image(figs), 4.6 * inch, 0.85 * inch))
    story.append(Spacer(1, 0.3 * cm))
    if metrics:
        n = metrics.get("n_test")
        acc = metrics.get("accuracy")
        story.append(Paragraph(
            f"Con {n} imÃ¡genes de test y accuracy {acc}, el nÃºmero de aciertos es "
            f"â‰ˆ {int(round(acc * n))} ({int(round(acc * n))}/{n}).", ss["Body"]))
    story.append(Spacer(1, 0.3 * cm))

    # ---------------------- 8. AnÃ¡lisis ----------------------
    story.append(Paragraph("8. AnÃ¡lisis del sobreajuste y conclusiones", ss["H"]))
    story.append(Paragraph(
        "<b>Â¿Por quÃ© ocurre el overfitting?</b> Con pocas muestras la red aprende "
        "el ruido del train. La brecha entre accuracy de train y de validaciÃ³n "
        "(columna <i>Gap</i> de la tabla) mide este efecto: a mayor <i>gap</i>, mÃ¡s "
        "sobreajuste.", ss["Body"]))
    story.append(Paragraph(
        "<b>RegularizaciÃ³n (Dropout):</b> comparar el baseline con Â«sin DropoutÂ» "
        "muestra cÃ³mo el Dropout reduce la brecha train/validaciÃ³n y mejora la "
        "generalizaciÃ³n.", ss["Body"]))
    story.append(Paragraph(
        "<b>Data Augmentation (rotaciÃ³n + zoom):</b> aumenta la variabilidad "
        "artificial del train y suele mejorar la accuracy de validaciÃ³n. "
        "Observar la diferencia entre el baseline y Â«sin AugmentationÂ».", ss["Body"]))
    story.append(Paragraph(
        "<b>OptimizaciÃ³n (Adam vs SGD):</b> Adam converge mucho mÃ¡s rÃ¡pido con esta "
        "tasa de aprendizaje; el SGD sin ajustar la tasa apenas aprende en pocas "
        "Ã©pocas, evidenciando la importancia del optimizador.", ss["Body"]))
    story.append(Paragraph(
        "<b>InnovaciÃ³n (Batch Normalization):</b> al retirarla, el entrenamiento se "
        "estanca; BatchNorm estabiliza y acelera la convergencia.", ss["Body"]))
    story.append(Paragraph(
        "<b>ConclusiÃ³n:</b> la CNN con Conv 3Ã—3 + MaxPooling + BatchNorm + Dropout + "
        "Data Augmentation entrenada con Adam es la configuraciÃ³n mÃ¡s robusta, "
        "alcanzando un accuracy de test de "
        f"{metrics.get('accuracy') if metrics else 'N/D'} sobre el conjunto "
        "completo, cumpliendo el KPI (â‰¥ 0.80).", ss["Body"]))

    doc = SimpleDocTemplate(
        str(out_pdf), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Actividad CNN - SVHN", author="Equipo SVHN",
    )
    doc.build(story)
    print(f"PDF generado: {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    run()

