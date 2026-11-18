# Plan de implementación — ProjectV06

Plan derivado de la **Reunión 07** con el profesor guía (Mauricio Araya L.), en la que se
presentó ProjectV05. ProjectV06 **consolida** todo lo trabajado en v01–v05 en un solo
notebook, `RULv06.ipynb`, ordenado como la memoria, y apoya la **redacción de la memoria**
en `Memoria/` (LaTeX). Todo el código va comentado en español. Las versiones anteriores no
se modifican.

**Plazo:** el escrito final se entrega a más tardar el **10 de diciembre de 2026**.

## Pedidos de la Reunión 07 y etapa en que se resuelven

| Pedido | Etapa |
|---|---|
| Centrarse al 100 % en escribir el documento, en formato memoria, antes del 10 de diciembre | 4 a 7 |
| Estudiar a fondo el proceso seguido por la IA para poder explicarlo y defenderlo | 7 (y el notebook comentado de las etapas 1 a 3) |
| Exportar los notebooks como PDF y enviárselos al profesor | Hecho por el estudiante con `RULv05`, el que pidió el profesor |

### Orientación del documento, según la reunión

- **Estudio de factibilidad tecnológica**, no un trabajo científico: una propuesta para
  implementar un predictor de RUL en una empresa (el profesor mencionó NASA o Boeing),
  basada en conocimiento científico.
- **Lo más completo posible**: estado del arte y fundamentos científicos incluidos.
- El proceso ELO-308 evalúa un documento con el trabajo terminado en un 90 %; la defensa es
  un proceso aparte, con plazo de hasta un año.
- El profesor estimó que un equipo de tres ingenieros podría construir el predictor en
  cuatro meses. Se usa como punto de partida de la estimación de implementación (etapa 6),
  citándolo como estimación del profesor guía.

## Decisiones confirmadas

| Punto | Decisión |
|---|---|
| Qué significa "consolidar" | Un solo notebook, `RULv06.ipynb`, que recorre el estudio completo en el orden de la memoria (datos, ablación, normalización, α, pooling, validación cruzada, prueba, sesgo y score, regímenes), reutilizando las corridas verificadas, **sin entrenar modelos nuevos**, y que exporta las figuras y tablas para la memoria |
| Papel de la IA en la redacción | La IA redacta un **borrador** de cada capítulo en LaTeX y el estudiante lo revisa y reescribe. Se declara explícitamente en la metodología de la memoria y en `Metodologia_IA.md` |
| Sesgo de la zona final en el conjunto de prueba | Se calcula como métrica **descriptiva** desde las predicciones de prueba ya guardadas, y se declara. Todas las decisiones están cerradas, así que no puede influir en ninguna elección |
| Borradores de la IA | **Aceptados por el profesor guía** (confirmado por el estudiante el 17 nov). Se declaran en la memoria y en `Metodologia_IA.md` |
| PDF para el profesor | **No se envían** los de v03 y v04: el profesor solo pidió el de v05, que el estudiante ya envió. El de `RULv06` se genera solo si el profesor lo pide |
| Comparación con el estado del arte | Se hace **si es posible**: tabla de resultados publicados en DS02 con las diferencias de protocolo declaradas |

## Decisiones y valores elegidos por la IA (confirmados el 17 nov)

| Punto | Valor | Por qué |
|---|---|---|
| Verificación de reproducibilidad | Reentrenar A6 con α = 2 (semilla 0) y A3 con MSE (semilla 0), y comparar con las corridas copiadas | Son la configuración final y la base, los dos modelos que más se citan en la memoria |
| Coherencia con versiones anteriores | Comprobar con `assert` que las cifras clave coinciden con las tablas guardadas de v03 (`prueba_oficial.csv`), v04 (`etapa1_val.csv`) y v05 (`etapa0_metricas_val.csv`) | La memoria no puede citar números que cambien entre versiones |
| Ancho de las figuras | 14,5 cm | Es el ancho del texto: hoja carta de 21,59 cm menos los márgenes de 4 y 3 cm de la plantilla |
| Letra de las figuras | Serif tipo Times, 10 pt | Igual que el texto de la memoria (newtx, Times); 10 pt queda legible a ese ancho sin competir con el texto de 12 pt |
| Formato de las figuras | PDF vectorial (y PNG para el notebook) | El PDF se escala sin perder nitidez en LaTeX |
| Cifras en el texto | El notebook escribe un archivo `cifras.tex` con un `\newcommand` por cifra citada | Si una cifra cambia, el texto se actualiza solo y no se transcriben números a mano |
| Formato de los números exportados (agregado en la etapa 0) | Punto decimal y espacio fino de miles (`16\,251`); signo menos tipográfico | La plantilla de la memoria usa `\decimalpoint`; con punto decimal, un punto de miles sería ambiguo |
| Barrido de K del clustering | Se recalcula en v06 (15 a 30 minutos, con semillas fijas) | v04 no guardó la tabla del barrido; la figura de la silueta según K se necesita en la memoria |
| Medición nueva para la factibilidad | Tiempo de inferencia por ventana en CPU y memoria del modelo; no es entrenamiento | La factibilidad pide saber si el modelo corre en hardware común; hoy solo se conoce el tiempo por época |
| Calendario | El de la última sección | Deja dos días de margen antes del 10 de diciembre |

---

## Etapa 0 — Infraestructura

- Infraestructura de v05 en `RULv06.ipynb`: datos, ventanas, normalización global,
  catálogo A1–A6 y variantes con pooling, métricas ampliadas de v05, `run` reanudable.
- Se agregan las normalizaciones N1a (por régimen) y N1b (por motor) de v02, para cargar sus
  corridas en la etapa 2.
- Se copian las corridas de v02 (96) y de v04 (20), y se reentrenan las dos corridas de
  verificación (diferencia esperada: 0).
- Estilo de figuras para la memoria (ancho, letra, PDF vectorial) y función de exportación de
  figuras, tablas (`booktabs`) y cifras (`cifras.tex`).

## Etapa 1 — Notebook: datos, protocolo y arquitecturas (capítulos de datos y metodología)

- Descripción de N-CMAPSS DS02: variables W, X_s, X_v y T, motores, clases de vuelo `Fc`, y
  por qué solo se usan W y X_s.
- Figuras de exploración: un vuelo típico (altitud, Mach, TRA) y la evolución de algunos
  sensores con la degradación.
- RUL y recorte lineal por tramos en 62 desde el indicador `hs`, ventanas de 30 pasos,
  partición por motor, normalización sin fuga, zonas de evaluación.
- Catálogo de arquitecturas con la forma de cada tensor y el número de parámetros;
  entrenamiento, early stopping y semillas.

## Etapa 2 — Notebook: resultados experimentales (capítulo de resultados)

Cada sección carga las corridas desde disco y presenta la tabla, la figura y la decisión
registrada en su momento:

1. Ablación de arquitecturas A1–A6 (v02/v03, etapa 1).
2. Normalización: global, N1a y N1b (v02, etapa 2).
3. Pérdida asimétrica, α ∈ {1, 2, 4}, y comprobación de la condición de expectiles (v03).
4. Pooling: A6-P1 y A6-P2 (v04).
5. Validación cruzada dejando un motor fuera (v03).
6. Prueba oficial: resultado único de v02, más el **sesgo de la zona final en prueba**,
   declarado como métrica descriptiva calculada en v06.

## Etapa 3 — Notebook: análisis, factibilidad y PDF de v06

- Sesgo y score NASA (v05): descomposición, recorte en 62, sesgo por motor, promedio frente
  a suma.
- Pesos de atención, error por fase de vuelo y por régimen, y relación entre la vida del motor
  y el sesgo (v03 y v04).
- Mediciones para la factibilidad: parámetros, tamaño en disco, tiempo de inferencia en CPU y
  tiempo de entrenamiento.
- Exportación de todas las figuras, tablas y cifras a `Memoria/`.
- PDF de `RULv06`, solo si el profesor lo pide.

## Etapa 4 — Redacción: fundamentos (borradores de la IA)

- **Marco teórico:** mantenimiento predictivo y pronóstico, RUL, convolución 1D,
  BatchNorm, dropout, pooling, LSTM, atención, pérdida asimétrica y expectiles, métricas.
- **Estado del arte:** se parte del informe de ELO-307 (`Documentos/Estado_del_arte`), con
  cada referencia verificada en su fuente. Incluye una tabla de resultados publicados en DS02
  con las diferencias de protocolo declaradas, sin afirmar superioridad: el resumen de ELO-307
  prometió comparar con el estado del arte, pero los protocolos no son comparables.
- **Alternativas de solución y selección:** desde los informes de ELO-307 (criterios C1–C8,
  alternativa A4). Se explica por qué el estudio terminó en A6 (CNN + atención) y no en A5
  (CNN + LSTM + atención), que es la implementación directa de A4.

## Etapa 5 — Redacción: capítulos técnicos (borradores de la IA)

- **Conjunto de datos y protocolo experimental** (etapa 1 del notebook).
- **Diseño experimental y metodología:** arquitecturas, estudio por etapas, criterios
  fijados antes de ver resultados, verificación de reproducibilidad y uso de IA.
- **Resultados** (etapa 2 del notebook).
- **Análisis y discusión** (etapa 3 del notebook), incluidas las correcciones registradas
  en v04 y v05 y las limitaciones.

## Etapa 6 — Redacción: factibilidad y cierre (borradores de la IA)

- **Evaluación de factibilidad tecnológica:**
  - técnica: resultados frente a los requisitos y a los criterios C1–C8;
  - de datos y operación: sensores disponibles en un motor real, dominio de clases de vuelo
    no vistas, datos simulados;
  - computacional: mediciones de la etapa 3;
  - de implementación: equipo, plazo y etapas, partiendo de la estimación del profesor;
  - riesgos.
- **Introducción, Conclusiones** (con trabajo futuro), **Resumen y Abstract** (una página),
  **Glosario** y **Anexos** (enlace al repositorio, instrucciones de reproducción, uso de IA).
- Referencias en el formato del anexo C de la norma.

## Etapa 7 — Revisión, material de defensa y entrega

- Revisión del documento completo con la retroalimentación del profesor sobre los PDF.
- Comprobación de que cada cifra del texto sale de `cifras.tex` y de que cada referencia está
  verificada.
- **Material para estudiar el proceso de la IA:** `Recorrido_del_Proceso.md`, un recorrido
  cronológico de v00 a v06 (qué pidió el profesor, qué hizo la IA, qué decidió el estudiante,
  qué errores se detectaron), más `Guia_Estudio.md`, `Preparacion_Defensa.md` y
  `Metodologia_IA.md` actualizados a v06.
- PDF final de la memoria, `README.md`, `CLAUDE.md` y subida a GitHub, cuando se apruebe.

## Cómo se trabaja cada capítulo

1. La IA redacta el borrador en LaTeX, con las cifras de `cifras.tex` y las figuras de
   `Memoria/Imagenes`, y lo compila.
2. El estudiante lo revisa y lo reescribe con sus palabras.
3. La IA comprueba la versión reescrita: cifras, referencias, numeración y norma de formato.

Cada borrador lleva al inicio un comentario `%% BORRADOR IA` que el estudiante elimina al
reescribirlo.

## Puntos abiertos y riesgos

- **Lineamientos de la USM sobre IA:** siguen sin encontrarse. El profesor guía aceptó que
  la IA redacte borradores, siempre que se declare.
- **Descripción de las variables del conjunto de datos:** el artículo de Arias Chao et al.
  (2021) no pudo descargarse (la editorial rechazó la descarga automática). La etapa 1 usó la
  documentación distribuida con los datos y el notebook de ejemplo de uno de los autores;
  conviene verificarla en el artículo antes de entregar.
- **Duración de los vuelos:** los vuelos de los datos son más cortos que lo que indica la
  definición de las clases de vuelo (etapa 1, sección 1.2). Se informa como observación.
- **Profesor correferente:** la portada lo pide; falta su nombre.
- **Tiempo de reescritura:** el calendario supone que el estudiante reescribe cada capítulo
  en 2 o 3 días. Si se atrasa, se recorta primero el detalle de los anexos, no los capítulos.
- **Comparación con el estado del arte:** si no se encuentran resultados de DS02 con
  protocolo comparable, la memoria lo declara, sin forzar la comparación.
- **Definición original del score** (Saxena et al., 2008): leerla en la fuente antes de
  citarla directamente.

## Calendario propuesto

| Etapa | Fechas | Entregable |
|---|---|---|
| 0 | 17–18 nov | Infraestructura de v06 verificada |
| 1 | 18–20 nov | Notebook: datos, protocolo y arquitecturas |
| 2 | 20–23 nov | Notebook: resultados |
| 3 | 23–25 nov | Notebook completo |
| 4 | 19–27 nov (en paralelo) | Marco teórico, estado del arte y alternativas |
| 5 | 25 nov–2 dic | Datos, metodología, resultados y discusión |
| 6 | 1–5 dic | Factibilidad, introducción, conclusiones, resumen y anexos |
| 7 | 5–9 dic | Revisión final, material de defensa y PDF; 9 dic de margen |
| Entrega | 10 dic | Escrito final al profesor |

## Costo computacional estimado (CPU)

| Etapa | Cómputo | Tiempo aproximado |
|---|---|---|
| 0 | 2 reentrenamientos de verificación | 3 min |
| 1–2 | Sin entrenamiento (corridas guardadas) | minutos |
| 3 | Barrido de K del clustering y medición de inferencia | 20–35 min |
| 4–7 | Compilación de LaTeX | segundos por pasada |

## Estado

| Etapa | Estado |
|---|---|
| 0 | Completada (17 nov): 96 corridas de v02 y 20 de v04 copiadas; el reentrenamiento de A6 con α = 2 y de A3 con MSE (semilla 0) coincide exactamente (diferencia 0); las 35 corridas analizadas en v05 dan las mismas métricas (diferencia máxima 7e-15); normalizaciones N1a y N1b de v02 portadas y comentadas; funciones `save_fig`, `save_table` y `cifra` exportan a `Memoria/Imagenes`, `Memoria/Tablas` y `Memoria/cifras.tex`, comprobadas compilando con el preámbulo de la memoria |
| 1 | Completada (17 nov): variables, motores (ciclos de vida y de transición comprobados contra la documentación), vuelo típico, degradación en los sensores (W explica el 99,9 % o más de cada sensor; descontada, T48 sube con ρ = 0,97), recorte PwL, partición y ventanas, arquitecturas e hiperparámetros. Exportadas 3 figuras, 7 tablas y 22 cifras |
| 2 | Completada (17 nov): ablación, normalización, α, pooling, validación cruzada y prueba, cargadas desde disco; cada regla registrada se reaplica y un `assert` comprueba que la decisión y sus cifras coinciden con v02–v04, y las 12 métricas de prueba con v03 (diferencia máxima 7e-15). **Sesgo de la zona final en prueba (nuevo, declarado):** +0,72 [+0,21 – +1,36] en la configuración final, positivo en las 5 semillas (base +2,24); el global citado hasta v05 (+0,19) daba una imagen más favorable. La relación vida–sesgo de v03 se refuerza con el sesgo de la zona final (Spearman −0,94 en ambas configuraciones). Exportadas 7 figuras, 8 tablas y 48 cifras nuevas, comprobadas compilando con el preámbulo de la memoria |
| 3 | Completada (17 nov): sesgo por zona y recorte en 62 (v05), score NASA (promedio frente a suma, asimetría; Spearman 0,95 con el RMSE y 0,99 con el MAE asimétrico), vida del motor y sesgo, pesos de atención, fases y regímenes (PCA de W, barrido de K de 2 a 20 guardado en `results/`), error por fase y mediciones de factibilidad. Hallazgos nuevos: en A6 la atención es uniforme en los pasos 3 a 28 y los bordes varían entre semillas (no son siempre picos, como se describía); A4 y A5 no son uniformes; en el descenso la configuración final sobreestima (sesgo final +1,49); el mayor Rand ajustado frente a las fases es 0,37 (Spectral 18D, K = 3). Factibilidad: 14 497 parámetros (57 KiB), unos 0,5 ms por inferencia en CPU, 52 s por entrenamiento. Exportadas 5 figuras, 4 tablas y unas 50 cifras más (123 en total) |
| 4 | En curso: Marco teórico redactado (18 nov; borrador de la IA, 10 páginas, 21 ecuaciones): RUL y etiqueta PwL, aprendizaje supervisado, Adam, regularización, BatchNorm, convolución 1D y pooling, LSTM, agregación y atención, pérdida asimétrica y expectiles (con la figura nueva `fig_perdidas`, RULv06 sección 3.7), métricas y agrupamiento. 14 referencias nuevas, 11 verificadas en Crossref, arXiv, PMLR, JMLR o NeurIPS; quedan con VERIFICAR MacQueen, Ng et al., Ester et al. (escritas de memoria) y detalles menores de Goodfellow, Vaswani y Kaufman. Ejemplos de formato de la plantilla movidos a `2_Desarrollo/EjemplosFormato.tex`. Agregada (18 nov, a pedido del estudiante) la sección 1.1 sobre el motor turbofán (unas 2,5 páginas, con un esquema en TikZ con la ubicación de los sensores), basada en la guía de usuario de C-MAPSS (NASA/TM-2007-215026, leída en el PDF de NTRS) y en dos páginas de NASA Glenn. Estado del arte redactado (18 nov; borrador de la IA, unas 9 páginas, 3 tablas): conjuntos de referencia, redes convolucionales, recurrentes, con atención, Transformers y grafos, tratamiento de la sobreestimación, resultados publicados en DS02 (Li et al. 2022; Wang et al. 2024, arXiv 2409.19629; Wen et al. 2025, arXiv 2405.04336) con sus diferencias de protocolo y sin afirmar superioridad, y síntesis. 7 referencias nuevas, verificadas en los PDF, Crossref o arXiv. Para la comparación, RULv06 (sección 2.6) exporta ahora el RMSE y el MAE globales de prueba y el RMSE global por motor de la configuración final (15 cifras nuevas, 138 en total), calculados con las predicciones guardadas de v02 y comprobados contra v03. Falta: alternativas |
| 5 | Pendiente |
| 6 | En curso: Resumen redactado como borrador de la IA (17 nov; cabe en una página, cifras desde `cifras.tex`, que `main.tex` ahora carga). Abstract redactado como traducción literal del Resumen (17 nov; una página, mismas macros). Introducción redactada (18 nov; borrador de la IA, unas 4,5 páginas): objetivos del formulario de tema, antecedentes con 9 referencias verificadas en sus PDF o DOI (4 datos marcados VERIFICAR), y estructura en 8 capítulos (archivos `Parte1` a `Parte8` creados con su título y contenido previsto). Correferente agregado a la portada (Patricio Olivares R., según el formulario). Falta: factibilidad, conclusiones, glosario y anexos. Cuando el estudiante reescriba el Resumen, hay que volver a traducir el Abstract |
| 7 | Pendiente |
