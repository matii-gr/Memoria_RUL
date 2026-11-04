# Estimación de la vida útil remanente en sistemas de propulsión aeronáutica mediante arquitecturas híbridas de aprendizaje profundo

Trabajo de título — Departamento de Electrónica, Universidad Técnica Federico Santa María.

- **Estudiante:** Matías Godoy Rojas
- **Profesor guía:** Mauricio Araya L.

Estimación de la vida útil remanente (RUL) de motores turbofán sobre el conjunto N-CMAPSS
(subconjunto DS02), comparando arquitecturas con convolución 1D, LSTM y atención temporal, y
una pérdida asimétrica que penaliza la sobreestimación.

## Resultado principal

Error en la zona final (RUL ≤ 20 ciclos) sobre los motores de prueba oficiales (11, 14 y 15),
mediana [mínimo – máximo] sobre 5 semillas:

| Configuración | RMSE final (ciclos) | Sesgo (ciclos) |
|---|---|---|
| Base: CNN + LSTM, error cuadrático medio | 2,97 [2,47 – 3,30] | +3,00 |
| CNN + atención, error cuadrático medio | 2,16 [1,92 – 2,45] | +1,24 |
| **CNN + atención, pérdida asimétrica (α = 2)** | **1,78** [1,50 – 1,97] | **+0,19** |

En la validación cruzada dejando un motor fuera, la ventaja de la configuración final no es
uniforme: depende de si el modelo base tiende a sobreestimar o a subestimar cada motor. El
detalle está en `Project/ProjectV03/RULv03.ipynb`. ProjectV04 probó agregar *pooling* a la
red y no cambió la configuración final.

## Dónde empezar

La versión vigente es **ProjectV04**, que responde a la Reunión 05: por qué la red no tiene
una capa de *pooling* y una revisión del agrupamiento de regímenes de operación. Los
experimentos principales (ablación de arquitecturas, pérdida asimétrica, validación cruzada
y prueba) están en **ProjectV03**, y V04 parte de su configuración final.

| Archivo | Contenido |
|---|---|
| `Project/ProjectV04/RULv04.ipynb` | Notebook de V04, comentado y ejecutado: pooling y regímenes de operación |
| `Project/ProjectV03/RULv03.ipynb` | Notebook de V03, comentado y ejecutado: ablación, pérdida asimétrica, validación cruzada y prueba |
| `Project/ProjectV04/Plan_ProjectV04.md` | Plan por etapas de V04, decisiones y estado |
| `Project/ProjectV04/Diagramas.md` | Diagramas del pipeline, de A6 con y sin pooling y del análisis de regímenes (Mermaid) |
| `Project/ProjectV04/Guia_Estudio.md` | Explicación del código y de la teoría |
| `Project/ProjectV04/Metodologia_IA.md` | Metodología de uso de inteligencia artificial |
| `Project/ProjectV04/Preparacion_Defensa.md` | Preguntas probables, limitaciones y pendientes |

Las figuras de cada etapa están en la carpeta `results/` de cada versión.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `Dataset/` | Descripción del conjunto de datos. Los archivos `.h5` no se incluyen (ver más abajo) |
| `Project/ProjectV00` | Primera versión del pipeline (con fuga de datos entre conjuntos) |
| `Project/ProjectV01` | Ablación corregida: partición por motor y early stopping |
| `Project/ProjectV02` | Estudio por etapas: arquitecturas, normalización, pérdida y validación cruzada |
| `Project/ProjectV03` | Mismos experimentos que V02, código comentado, análisis de regímenes de operación y documentación |
| `Project/ProjectV04` | Versión vigente: capa de pooling y revisión del agrupamiento de regímenes |

Cada versión se conserva sin modificar; los cambios pedidos en cada reunión se implementan
en una versión nueva.

## Datos

El conjunto N-CMAPSS (*Turbofan Engine Degradation Simulation Data Set-2*) es público y pesa
unos 28 GB, por lo que no se incluye en el repositorio. Se descarga desde el repositorio de
datos de pronóstico de NASA (conjunto 17):

- Espejo de la PHM Society: https://data.phmsociety.org/nasa/
- Descarga directa: https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip

Este trabajo usa solo el archivo **`N-CMAPSS_DS02-006.h5`**, que debe ubicarse en `Dataset/`.

Referencia del conjunto de datos: Chao, M. A., Kulkarni, C., Goebel, K. y Fink, O. (2021).
Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and
Diagnostics. *Data*, 6(1), 5. https://doi.org/10.3390/data6010005

## Cómo ejecutar los notebooks

Los notebooks ya están ejecutados y guardados con sus salidas, así que se pueden revisar sin
volver a correrlos. Para re-ejecutarlos:

1. **Entorno usado:** Python 3.12, PyTorch 2.7.1 (CPU), NumPy 1.26, pandas 2.2, scikit-learn 1.5,
   SciPy 1.13, h5py 3.12, matplotlib 3.9 y seaborn 0.13.
2. **Ruta del dataset:** en la celda de configuración de cada notebook, la variable
   `DATASET_PATH` apunta a la ruta del equipo original
   (`D:/Ramos/Memoria_Titulacion_ELO308/Dataset/N-CMAPSS_DS02-006.h5`). Hay que ajustarla a la
   ubicación local del archivo.
3. **Carpeta de trabajo:** cada notebook usa rutas relativas (`results/`), así que debe
   ejecutarse desde su propia carpeta.
4. **Prueba rápida:** con `SMOKE_TEST = True` en la celda de configuración, el notebook corre
   completo en pocos minutos con datos reducidos.

**Resultados guardados.** `Project/ProjectV02/results/runs/` contiene las 96 corridas de
entrenamiento de V02 (pesos, historial y predicciones). Al re-ejecutar, los notebooks las
cargan desde disco en vez de reentrenar, lo que reduce el tiempo de más de 2 horas a unos
minutos. ProjectV03 y ProjectV04 copian esas corridas al ejecutarse y verifican,
reentrenando algunas desde cero, que su código las reproduce exactamente. Las 20 corridas
nuevas de V04 (A6 con pooling) están en `Project/ProjectV04/results/runs/`.

La primera ejecución lee el archivo `.h5` y genera una caché del dataset submuestreado en
`results/` (unos 2 minutos).

## Archivos no incluidos

Además del dataset, no se incluyen las cachés regenerables, las copias de las corridas de V02
en ProjectV03 y ProjectV04 (se regeneran desde ProjectV02), los informes y artículos de la
carpeta `Documentos/` ni los resúmenes de reuniones. El detalle está en `.gitignore`.

## Uso de inteligencia artificial

El código y la experimentación se desarrollaron con apoyo de Claude Code (Anthropic). La
metodología, las salvaguardas adoptadas y los errores de la herramienta detectados durante el
trabajo están descritos en `Project/ProjectV04/Metodologia_IA.md`.
