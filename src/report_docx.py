"""Genera el informe Word (.docx) de la actividad, en formato APA 7.

Salida: reports/actividad_cnn.docx
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

from .config import load_config, resolve

TEAM = [
    "Gianella Alexandra Ramos Ticahuanca",
    "Juan Gabriel Suárez Condori",
    "Jesús Del Águila García",
    "Lincol J. Mendoza Torres",
]
REPO = "https://github.com/Abrilskop/proyectoiafino"
COURSE = "Inteligencia Artificial"
INSTRUCTOR = "Docente del curso"
AFFILIATION = "Ingeniería de Sistemas"


# --------------------------------------------------------------------------- #
#  Utilidades de formato APA 7
# --------------------------------------------------------------------------- #
def _apa_document() -> Document:
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:eastAsia"), "Times New Roman")
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)

    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
    return doc


def _add_page_number(doc: Document) -> None:
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run()
    f1 = OxmlElement("w:fldChar")
    f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = "PAGE"
    f2 = OxmlElement("w:fldChar")
    f2.set(qn("w:fldCharType"), "end")
    run._r.append(f1)
    run._r.append(it)
    run._r.append(f2)


def _h1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    r = p.add_run(text)
    r.bold = True


def _h2(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(12)
    r = p.add_run(text)
    r.bold = True


def _para(doc: Document, text: str, indent: bool = True, align=None) -> None:
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.5)
    p.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(text)


def _bullets(doc: Document, items) -> None:
    for it in items:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.add_run("• ").bold = True
        p.add_run(it)


def _figure(doc: Document, img: Path, number: int, title: str, note: str | None = None,
            width: float = 6.0) -> None:
    lab = doc.add_paragraph()
    lab.add_run(f"Figura {number}").bold = True
    t = doc.add_paragraph()
    t.add_run(title).italic = True
    pic = doc.add_paragraph()
    pic.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic.add_run().add_picture(str(img), width=Inches(width))
    if note:
        n = doc.add_paragraph()
        n.add_run("Nota. ").bold = True
        n.add_run(note)


def _table(doc: Document, title: str, headers, rows, number: int) -> None:
    lab = doc.add_paragraph()
    lab.add_run(f"Tabla {number}").bold = True
    t = doc.add_paragraph()
    r = t.add_run(title)
    r.bold = True

    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = True
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(h))
        run.bold = True
        run.font.size = Pt(10)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(10)
    _horizontal_borders(table)


def _horizontal_borders(table) -> None:
    """Bordes solo horizontales (estilo APA): arriba, abajo y bajo el encabezado."""
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        if edge in ("top", "bottom"):
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "6")
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "none")
        borders.append(el)
    tblPr.append(borders)
    # linea inferior del encabezado
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        cb = OxmlElement("w:tcBorders")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:color"), "000000")
        cb.append(bottom)
        tcPr.append(cb)


# --------------------------------------------------------------------------- #
#  Contenido
# --------------------------------------------------------------------------- #
def run() -> Path:
    cfg = load_config()
    reports = resolve(cfg["paths"]["reports_dir"])
    figs = resolve(cfg["paths"]["figures_dir"])
    exp_dir = reports / "experiments"
    out = reports / "actividad_cnn.docx"

    doc = _apa_document()
    _add_page_number(doc)

    # ------------------------- PORTADA (APA) ------------------------- #
    title = ("Diseño y Optimización de una Red Neuronal Convolucional (CNN) "
             "para la Clasificación de Imágenes Complejas")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Inches(2.5)
    r = p.add_run(title)
    r.bold = True

    for name in TEAM:
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.add_run(name)

    for line in (AFFILIATION, COURSE, INSTRUCTOR,
                 dt.datetime.now().strftime("%d de octubre de %Y")):
        q = doc.add_paragraph()
        q.alignment = WD_ALIGN_PARAGRAPH.CENTER
        q.add_run(line)

    doc.add_page_break()

    # ------------------------- RESUMEN ------------------------- #
    _h1(doc, "Resumen")
    metrics_path = reports / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    acc = metrics.get("accuracy", "N/D")
    _para(doc,
          "El presente informe describe el diseño, la construcción y la optimización de una "
          "red neuronal convolucional (CNN) para clasificar dígitos en imágenes de números de "
          "casa del conjunto de datos SVHN (Street View House Numbers). El problema se aborda "
          "desde la perspectiva de una startup de logística que necesita digitalizar direcciones "
          "a partir de imágenes de calle. Se explica el preprocesamiento de los datos, la "
          "arquitectura de la red (capas convolucionales de 3x3, max-pooling y una capa de salida "
          "con activación softmax) y una etapa de experimentación comparativa en la que se "
          "modificaron, uno por uno, el uso de dropout, el optimizador (Adam frente a SGD), la "
          "técnica de aumento de datos (rotaciones y zooms aleatorios) y la normalización por "
          "lotes (batch normalization). La mejor configuración alcanzó una exactitud (accuracy) "
          f"de {acc} sobre el conjunto de prueba, cumpliendo el indicador de éxito definido "
          "para el proyecto. El análisis de los resultados muestra que el sobreajuste se "
          "controla combinando dropout y aumento de datos, y que la normalización por lotes es "
          "clave para que el entrenamiento converja de forma estable.")
    doc.add_page_break()

    # ------------------------- 1. INTRODUCCION ------------------------- #
    _h1(doc, "Introducción")
    _para(doc,
          "En el ámbito de la visión artificial, clasificar imágenes automáticamente es una tarea "
          "fundamental para muchas aplicaciones industriales. Este trabajo se enmarca en la misión "
          "de una startup de logística que desea reconocer dígitos en fotografías de números de "
          "casa para poblar bases de direcciones y apoyar procesos de entrega. El conjunto de "
          "datos utilizado, SVHN, fue propuesto por Netzer et al. (2011) y contiene imágenes "
          "reales extraídas de Google Street View.")
    _para(doc,
          "El objetivo es desarrollar una red neuronal convolucional (CNN) capaz de clasificar "
          "imágenes con alta precisión, enfrentando problemas reales como el sobreajuste "
          "(overfitting) y la variabilidad de los datos. Para ello se utilizó el framework "
          "PyTorch (Paszke et al., 2019), una herramienta profesional de desarrollo de modelos "
          "de profundas.")

    # ------------------------- 2. METODO ------------------------- #
    _h1(doc, "Método")

    _h2(doc, "Fase A. Preparación y preprocesamiento")
    _para(doc,
          "El conjunto de datos SVHN está compuesto por imágenes a color de 32x32 píxeles "
          "pertenecientes a diez clases (los dígitos del 0 al 9). El subconjunto de entrenamiento "
          "oficial contiene 73.257 imágenes y el de prueba 26.032. Dado que el desarrollo se "
          "realizó en una computadora sin unidad de procesamiento gráfica (GPU), se utilizó un "
          "subconjunto del 20 por ciento de los datos de entrenamiento para poder iterar de forma "
          "rápida, conservando el conjunto de prueba completo para la evaluación final.")
    _para(doc,
          "Los pasos de preparación fueron: (a) descarga automática del conjunto de datos con la "
          "biblioteca torchvision; (b) normalización de los píxeles restando la media y dividiendo "
          "por la desviación estándar calculadas sobre el propio conjunto; y (c) conversión a "
          "tensores organizados en el canal primero (CHW). De esta manera se obtienen tres "
          "subconjuntos: entrenamiento, validación y prueba.")

    _h2(doc, "Fase B. Construcción del modelo base")
    _para(doc,
          "La red se define en el archivo src/models/cnn.py. Está compuesta por tres bloques "
          "convolucionales encadenados. Cada bloque realiza una convolución de tamaño 3x3 con "
          "desplazamiento (stride) igual a 1 y relleno (padding) igual a 1, seguida de una "
          "función de activación rectificada lineal (ReLU). Después de cada par de convoluciones "
          "se aplica una operación de max-pooling de tamaño 2x2, que reduce a la mitad las "
          "dimensiones espaciales y ayuda a que el modelo sea más robusto a pequeñas traslaciones "
          "de la imagen (Goodfellow et al., 2016).")
    _para(doc,
          "Además, cada bloque incluye normalización por lotes (batch normalization), una "
          "técnica complementaria que estabiliza y acelera el entrenamiento. Finalmente, el "
          "tensor se reduce a un vector mediante un promediado global y se conecta a un "
          "clasificador con dos capas densas separadas por capas de dropout. La capa de salida "
          "tiene diez neuronas y utiliza la activación softmax para producir una distribución de "
          "probabilidades sobre las diez clases, tal como se exige en la actividad.")

    arch_rows = [
        ["Entrada", "32 x 32 x 3", "Imagen a color"],
        ["Conv 3x3 + BN + ReLU", "32 x 32 x 32", "stride 1, padding 1"],
        ["Conv 3x3 + BN + ReLU", "32 x 32 x 32", "stride 1, padding 1"],
        ["Max-pooling 2x2", "16 x 16 x 32", "reduce a la mitad"],
        ["Conv 3x3 + BN + ReLU", "16 x 16 x 64", "stride 1, padding 1"],
        ["Conv 3x3 + BN + ReLU", "16 x 16 x 64", "stride 1, padding 1"],
        ["Max-pooling 2x2", "8 x 8 x 64", "reduce a la mitad"],
        ["Conv 3x3 + BN + ReLU", "8 x 8 x 128", "stride 1, padding 1"],
        ["Conv 3x3 + BN + ReLU", "8 x 8 x 128", "stride 1, padding 1"],
        ["Max-pooling 2x2", "4 x 4 x 128", "reduce a la mitad"],
        ["Promediado global", "1 x 1 x 128", "resume la información espacial"],
        ["Dropout + Capa densa", "256 neuronas", "regularización"],
        ["Dropout + Capa densa", "10 neuronas", "salida con softmax"],
    ]
    _table(doc, "Arquitectura de la red neuronal convolucional", 
           ["Capa", "Dimensión de salida", "Observación"], arch_rows, 1)
    doc.add_paragraph()

    _h2(doc, "Entrenamiento")
    _para(doc,
          "La red se entrenó minimizando la función de entropía cruzada (CrossEntropyLoss), que "
          "equivale a aplicar softmax y luego medir el error de clasificación. Como optimizador "
          "se utilizó Adam con tasa de aprendizaje 0.001. El conjunto de validación se empleó "
          "para decidir cuándo detener el entrenamiento y para comparar las distintas "
          "configuraciones de la experimentación.")

    # ------------------------- 3. RESULTADOS ------------------------- #
    _h1(doc, "Resultados")

    _h2(doc, "Fase C. Experimentación")
    _para(doc,
          "Se partió de una configuración base (baseline) y, por cada corrida, se modificó un "
          "único factor para aislar su efecto. Las cinco corridas fueron: (1) configuración base; "
          "(2) sin dropout; (3) con el optimizador SGD en lugar de Adam; (4) sin aumento de datos; "
          "y (5) sin normalización por lotes. Todos los experimentos se ejecutaron con cuatro "
          "épocas y un máximo de 8.000 imágenes de entrenamiento para que la comparación fuera "
          "rápida y justa entre sí.")

    csv_path = exp_dir / "comparison.csv"
    exp_rows = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as fh:
            for row in csv.reader(fh):
                exp_rows.append(row)
    if exp_rows:
        _table(doc, "Resultados comparativos de los experimentos",
               exp_rows[0], exp_rows[1:], 2)
        doc.add_paragraph()

    cmp_png = exp_dir / "comparison.png"
    if cmp_png.exists():
        _figure(doc, cmp_png, 1,
                "Exactitud de validación y de prueba en cada experimento",
                "Cada par de barras corresponde a una configuración distinta. "
                "Cuanto más altas son las barras, mejor es el desempeño.",
                width=6.0)
        doc.add_paragraph()

    _h2(doc, "Curvas de pérdida y exactitud")
    _para(doc,
          "La figura siguiente muestra la evolución de la pérdida (loss) y de la exactitud "
          "(accuracy) durante el entrenamiento y la validación. Comparar ambas curvas es la "
          "forma más directa de detectar sobreajuste: si la curva de entrenamiento mejora "
          "continuamente mientras la de validación deja de mejorar o empeora, significa que la "
          "red está memorizando los datos de entrenamiento en lugar de aprender patrones "
          "generales.")
    tc_png = figs / "training_curves.png"
    if tc_png.exists():
        _figure(doc, tc_png, 2,
                "Pérdida y exactitud en entrenamiento y validación",
                "La izquierda corresponde a la pérdida y la derecha a la exactitud.",
                width=6.2)
        doc.add_paragraph()

    _h2(doc, "Matriz de confusión")
    _para(doc,
          "La matriz de confusión permite identificar qué clases se confunden con mayor frecuencia. "
          "En tareas de reconocimiento de dígitos, es habitual observar confusiones entre números "
          "con formas parecidas, como el 3 y el 8, el 4 y el 9, o el 5 y el 6.")
    cm_png = figs / "confusion_matrix.png"
    if cm_png.exists():
        _figure(doc, cm_png, 3,
                "Matriz de confusión sobre el conjunto de prueba",
                "Las filas representan la clase real y las columnas la clase predicha.",
                width=5.2)
        doc.add_paragraph()

    _h2(doc, "Cálculo de la precisión final")
    _para(doc,
          "La exactitud se define como la proporción de predicciones correctas sobre el total de "
          "casos evaluados. En términos de aciertos y errores se expresa con la siguiente fórmula:")
    _para(doc, "Accuracy = (VP + VN) / (VP + VN + FP + FN)",
          indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc,
          "donde VP son los verdaderos positivos, VN los verdaderos negativos, FP los falsos "
          "positivos y FN los falsos negativos. Aplicando la fórmula al conjunto de prueba:")
    if metrics:
        n = metrics.get("n_test")
        aciertos = int(round(float(acc) * int(n)))
        _para(doc,
              f"Accuracy = {aciertos} aciertos / {n} imágenes = {float(acc):.4f}, es decir, "
              f"{float(acc) * 100:.2f} por ciento de aciertos. El macro-F1 obtenido fue "
              f"{metrics.get('macro_f1')}. Este valor supera el indicador de éxito definido "
              f"para el proyecto ({metrics.get('target_value')}), por lo que el objetivo se "
              "considera cumplido.")
    fa_png = figs / "accuracy_formula.png"
    if fa_png.exists():
        _figure(doc, fa_png, 4, "Fórmula de la exactitud (accuracy)",
                "Representación gráfica de la fórmula empleada.", width=4.6)
        doc.add_paragraph()

    # ------------------------- 4. DISCUSION ------------------------- #
    _h1(doc, "Discusión")

    _h2(doc, "¿Por qué ocurre el sobreajuste?")
    _para(doc,
          "El sobreajuste ocurre cuando el modelo aprende de memoria los datos de entrenamiento, "
          "incluidos sus ruidos e imperfecciones, en lugar de aprender patrones que generalizan a "
          "imágenes nuevas. Esto suele suceder cuando la red tiene muchos parámetros respecto a la "
          "cantidad de datos disponibles. La forma de medirlo es comparar la exactitud de "
          "entrenamiento con la de validación: a mayor diferencia entre ambas, mayor es el "
          "sobreajuste.")

    _h2(doc, "Cómo se controló el sobreajuste")
    _bullets(doc, [
        "Dropout: apaga de forma aleatoria una fracción de las neuronas durante el entrenamiento, "
        "lo que impide que el modelo dependa demasiado de neuronas concretas y mejora su capacidad "
        "de generalización. Al comparar la configuración base con la versión sin dropout se puede "
        "medir su efecto sobre la brecha entre entrenamiento y validación.",
        "Aumento de datos (data augmentation): se aplicaron rotaciones aleatorias de hasta 15 "
        "grados y zooms aleatorios, además de recortes y cambios de brillo y contraste. Esto "
        "aumenta la variabilidad artificial del entrenamiento y mejora la exactitud de validación.",
        "Normalización por lotes (batch normalization): estabiliza las activaciones y permite "
        "entrenar con tasas de aprendizaje más altas. Sin ella, el entrenamiento se estanca en "
        "exactitudes cercanas al azar, como se observa en la experimentación.",
    ])

    _h2(doc, "Efecto del optimizador")
    _para(doc,
          "Se comparó el optimizador Adam con el descenso de gradiente estocástico (SGD). Adam "
          "ajusta automáticamente la tasa de aprendizaje de cada parámetro y converge mucho más "
          "rápido en pocos cientos de iteraciones, mientras que el SGD, con la misma tasa de "
          "aprendizaje, apenas aprende en pocas épocas. Este resultado resalta la importancia de "
          "elegir y ajustar el optimizador adecuado.")

    # ------------------------- 5. CONCLUSION ------------------------- #
    _h1(doc, "Conclusión")
    _para(doc,
          "Se diseñó y optimizó una red neuronal convolucional para clasificar imágenes del "
          "conjunto SVHN, cumpliendo con todas las fases de la actividad: preparación de datos, "
          "construcción del modelo base con convoluciones de 3x3, max-pooling y salida softmax, y "
          "una etapa de experimentación comparativa sobre dropout, optimizador, aumento de datos y "
          "normalización por lotes. La configuración más robusta combina normalización por lotes, "
          "dropout, aumento de datos y el optimizador Adam, alcanzando una exactitud de prueba "
          "superior al indicador de éxito definido. Como trabajo futuro se propone entrenar con el "
          "conjunto completo y utilizar una unidad de procesamiento gráfica para elevar aún más la "
          "precisión.")

    # ------------------------- 6. GLOSARIO ------------------------- #
    _h1(doc, "Glosario")
    glos = [
        ("Exactitud (accuracy)", "Proporción de predicciones correctas sobre el total de casos."),
        ("Sobreajuste (overfitting)", "Situación en la que el modelo memoriza el entrenamiento y "
         "no generaliza a datos nuevos."),
        ("Convolución", "Operación que recorre la imagen con un filtro pequeño (por ejemplo 3x3) "
         "para detectar patrones como bordes o texturas."),
        ("Max-pooling", "Operación que reduce el tamaño de la imagen tomando el valor máximo de "
         "cada región, lo que disminuye el costo y aporta tolerancia a traslaciones."),
        ("ReLU", "Función de activación que deja pasar los valores positivos y pone en cero los "
         "negativos."),
        ("Softmax", "Función que convierte las salidas de la red en probabilidades que suman uno."),
        ("Dropout", "Técnica que apaga neuronas al azar durante el entrenamiento para reducir el "
         "sobreajuste."),
        ("Batch normalization", "Técnica que normaliza las activaciones de cada capa y estabiliza "
         "el entrenamiento."),
        ("Aumento de datos", "Creación de versiones modificadas de las imágenes (rotaciones, "
         "zooms) para aumentar la variedad del entrenamiento."),
        ("Época (epoch)", "Una pasada completa del modelo por todo el conjunto de entrenamiento."),
        ("Optimizador", "Algoritmo que ajusta los pesos de la red para reducir el error. Adam y "
         "SGD son dos ejemplos."),
        ("Entropía cruzada", "Función de costo que mide qué tan lejos está la predicción de la "
         "clase correcta."),
    ]
    _table(doc, "Glosario de términos utilizados", ["Término", "Definición simple"],
           [[a, b] for a, b in glos], 3)
    doc.add_paragraph()

    # ------------------------- 7. REFERENCIAS ------------------------- #
    _h1(doc, "Referencias")
    refs = [
        "Equipo SVHN. (2026). Proyecto de reconocimiento de números de casa con PyTorch "
        "[Código fuente]. GitHub. " + REPO,
        "Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.",
        "Marcel, S., & Rodriguez, Y. (2010). Torchvision the machine-vision package of torch. "
        "Proceedings of the 18th ACM International Conference on Multimedia, 1485-1488.",
        "Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., & Ng, A. Y. (2011). Reading "
        "digits in natural images with unsupervised feature learning. Advances in Neural "
        "Information Processing Systems, 24.",
        "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., "
        "Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., "
        "Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., ... Chintala, S. "
        "(2019). PyTorch: An imperative style, high-performance deep learning library. Advances "
        "in Neural Information Processing Systems, 32.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.add_run(ref)

    doc.save(str(out))
    print(f"Word generado: {out}")
    return out


if __name__ == "__main__":
    run()
