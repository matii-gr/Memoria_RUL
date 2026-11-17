# Plan de implementación — ProjectV05

Plan derivado de la **Reunión 06** con el profesor guía (Mauricio Araya L.), en la que se
presentó ProjectV04. Todas las etapas se implementan en un único notebook, `RULv05.ipynb`,
con **todo el código comentado en español**. ProjectV04 no se modifica.

V05 es una versión de **análisis y aclaración**: no hay experimentos nuevos ni cambia la
configuración final (A6 + normalización global + α = 2). Se reanalizan las predicciones ya
guardadas de v02 y v04. Los únicos entrenamientos son los de la verificación de la etapa 0.

## Pedidos de la Reunión 06 y etapa en que se resuelven

| Pedido | Etapa |
|---|---|
| Revisar por qué el score es tan bajo a pesar de los resultados de RMSE de P1 y P2 | 2 |
| Revisar el sesgo global: qué significa matemáticamente y cuánto se sobreestima o subestima | 1 |
| Preguntar a la IA por qué incluyó el sesgo global y con qué argumentos lo usó | 3 |
| Subir el trabajo a GitHub | Hecho para v04 (commit "Se actualiza a projectV04"); v05 se sube al terminar, en la etapa 4 |
| Seguir avanzando en la escritura de la memoria | Fuera de v05 (decisión confirmada): la redacción se lleva por separado |

### Aclaraciones sobre lo conversado en la reunión

El resumen automático de la reunión registra algunas afirmaciones que no coinciden con el
código o con los resultados. La etapa 3 las aclara con cifras:

- **"El score tiene un rango máximo de 100 y vale 67".** El score NASA no tiene máximo: va
  de 0 a infinito, y en v04 vale entre 0,11 y 0,19 en la zona final. La única métrica de 0
  a 100 es el porcentaje de ventanas sobreestimadas (26 a 80 en v04). No se recuerda a qué
  valor correspondía el 67, así que la etapa 2 documenta el rango y el valor de cada
  métrica. (Etapa 4: muy probablemente es el 67,4 % de ventanas sobreestimadas en
  la zona final en la prueba oficial de v03.)
- **"El dropout se aplica después de la convolución y antes del max pooling".** En el
  código el orden es convolución, BatchNorm, ReLU, max pooling y dropout. El dropout va
  **después** del pooling, que es la secuencia tradicional que describió el profesor.
- **"Max pooling con 15 canales".** 15 son pasos de tiempo, no canales. El pooling reduce
  el tiempo (30 a 15 o 7) y mantiene los canales (32 o 64).
- **"P1 sobreestima demasiado y P2 subestima demasiado".** Depende de α y de la zona de
  RUL. En la zona final, con α = 2, ambas variantes subestiman (sesgo −0,44 y −0,62).
- **Por qué hay dropout tras la primera convolución y no tras la segunda.** Viene del
  diseño de v01, donde la red tenía una convolución seguida de dropout y una LSTM. Al crear
  A1 y A6 en v02, la LSTM se reemplazó por una segunda convolución de 32 a 64 canales, sin
  dropout. Así, cada arquitectura tiene un único dropout. No fue una elección ajustada
  experimentalmente, y se declara como tal (decisión confirmada: se explica, sin
  experimento nuevo).

## Decisiones confirmadas

| Punto | Decisión |
|---|---|
| Experimentos | Ninguno nuevo; se reanalizan las predicciones de v02 y v04 |
| Dropout en la segunda convolución | Se explica el origen del diseño, sin entrenar variantes |
| Valor "67" de la reunión | No se recuerda; se documenta el rango y el valor actual de cada métrica |
| Escritura de la memoria | Fuera de v05 |
| Organización | Un solo notebook con todas las etapas |

---

## Etapa 0 — Infraestructura

- Infraestructura de v04, con el mismo código comentado.
- Se copian las corridas de v02 (A1–A6) y las de v04 (A6-P1 y A6-P2).
- **Verificación:** se reentrenan desde cero A6 (semilla 0, MSE) y A6-P2 (semilla 0, α = 2)
  y se comprueba que coinciden exactamente con las copiadas. La segunda verifica que las
  corridas de v04 también son reproducibles.
- **Métricas nuevas** en `compute_metrics`, calculadas sobre predicciones ya guardadas, sin
  reentrenar:
  - sesgo por zona de RUL (final, intermedia y temprana);
  - sobreestimación media y subestimación media, por zona;
  - aporte de la sobreestimación y de la subestimación al score NASA.
- Configuraciones que se analizan, todas en validación (motores 18 y 20): base A3 con MSE;
  A6 con MSE; final, A6 con α = 2; A6-P1 y A6-P2 con α = 1 y α = 2.

## Etapa 1 — El sesgo: qué es y cuánto se sobreestima o subestima

1. **Definición.** $\text{Sesgo} = \frac{1}{N}\sum_i (\hat{y}_i - y_i)$, en ciclos;
   positivo significa que el modelo sobreestima en promedio. Incluye un ejemplo de cómo los
   errores de signo opuesto se cancelan.
2. **Descomposición en sobreestimación y subestimación:**
   $\text{Sesgo} = \overline{\max(d, 0)} - \overline{\max(-d, 0)}$. Responde directamente
   "cuánto se sobreestima y cuánto se subestima": dos magnitudes en ciclos y el porcentaje
   de ventanas en cada caso.
3. **Por zona de RUL.** Tabla para cada configuración: sesgo, sobreestimación media,
   subestimación media y porcentaje de ventanas sobreestimadas, en la zona final (≤ 20),
   intermedia (20–40) y temprana (> 40). Muestra que el sesgo global está dominado por la
   zona temprana, donde la etiqueta está recortada en 62.
4. **Por motor de validación** (18 y 20), para la configuración final y la base.
5. **Relación con el RMSE:** $\text{RMSE}^2 = \text{Sesgo}^2 + \text{varianza del error}$,
   comprobada numéricamente. Indica qué parte del error es sistemática y qué parte varía de
   una ventana a otra.
6. **Figura:** distribución del error con signo por zona de RUL, para la base y la
   configuración final.

## Etapa 2 — El score NASA: por qué es tan bajo

1. **Definición usada en este trabajo:** promedio por ventana de
   $s(d) = e^{-d/13} - 1$ si $d < 0$ y $e^{d/10} - 1$ si $d \ge 0$ (Saxena et al., 2008).
   Rango de 0 a infinito; 0 es perfecto.
2. **Por qué los valores son pequeños.** Tabla de $s(d)$ para errores de ±1, ±2, ±5, ±10,
   ±20 y ±30 ciclos. Para errores pequeños, $s(d)$ crece casi en línea recta
   ($\approx d/10$ o $|d|/13$). Como en la zona final los errores típicos son de unos 2
   ciclos, el promedio queda cerca de 0,1 a 0,2.
3. **Promedio frente a suma.** En la formulación original (Saxena et al., 2008) el score
   se **suma** sobre las predicciones evaluadas, y en C-MAPSS suele reportarse con una sola
   predicción por motor, lo que da valores de cientos o miles. Aquí se promedia sobre miles
   de ventanas para que conjuntos de distinto tamaño sean comparables. Los números no son
   comparables con los publicados (verificado en la etapa 2 con Sateesh Babu et al., 2016;
   Li, Ding y Sun, 2018, y Li, Wang y Li, 2022).
4. **Por qué el score no sigue al RMSE en P1 y P2.** El score castiga menos la
   subestimación (divisor 13) que la sobreestimación (divisor 10); el RMSE castiga igual
   ambas. Una variante que empeora porque subestima, como A6-P2 con α = 2, empeora mucho en
   RMSE y poco en score. Se cuantifica con:
   - el aporte de la sobreestimación y de la subestimación al score de cada configuración;
   - la correlación de rangos (Spearman) entre RMSE final y score final sobre todas las
     corridas, y los casos en que ambas métricas ordenan distinto.
5. **Tabla de rangos de todas las métricas** con su valor actual en la configuración final:
   RMSE, MAE, sesgo, score NASA y porcentaje de sobreestimación.

## Etapa 3 — Por qué se incluyó el sesgo global, y correcciones de v04

1. **Respuesta de la IA, registrada en el notebook:**
   - el sesgo estaba en las métricas principales desde v02, y la IA lo agregó a la figura
     del pooling para mostrar la dirección del error;
   - lo usó para explicar por qué el pooling mejora con MSE y empeora con α = 2;
   - el argumento era débil: el sesgo global mezcla las tres zonas, la decisión se toma en
     la zona final, y el 39 % de las ventanas de validación está en la zona temprana.
2. **Comprobación con el sesgo de la zona final:**
   - el mecanismo se sostiene para A6-P2: con MSE, el sesgo final pasa de +0,70 a −0,03;
     con α = 2, de −0,04 a −0,62;
   - las cifras usadas en v04 (−1,32 y −1,83) exageraban el efecto;
   - no se generaliza a A6-P1.
3. **Discusión de la etapa 1 de v04, corregida en v05** con el sesgo de la zona final. v04
   no se modifica.
4. **Aclaraciones de la reunión** (lista de la sección anterior), con cifras y con el orden
   real de las capas.

## Etapa 4 — Documentación y GitHub

- `Guia_Estudio.md`: secciones nuevas sobre el sesgo (descomposición, relación con el RMSE)
  y el score NASA (escala, promedio frente a suma, asimetría).
- `Preparacion_Defensa.md`: preguntas sobre el score, el sesgo, el dropout y el orden de las
  capas, y la explicación corregida del pooling.
- `Metodologia_IA.md`: registro del error (argumento basado en el sesgo global, detectado por
  el profesor) y salvaguarda nueva: los argumentos se apoyan en métricas de la zona que
  decide.
- `Diagramas.md`: orden real de las capas con dropout y pooling, y descomposición del sesgo.
- `README.md` de la raíz apuntando a v05, y `CLAUDE.md` actualizado.
- Commit y subida a GitHub, cuando lo apruebes.

## Costo computacional estimado (CPU)

| Etapa | Entrenamientos nuevos | Tiempo aproximado |
|---|---|---|
| 0 Verificación | 2 | 3 min |
| 1 a 3 | Ninguno (predicciones guardadas) | minutos |

## Estado

| Etapa | Estado |
|---|---|
| 0 | Completada: 96 corridas de v02 y 20 de v04 copiadas; el reentrenamiento de A6 (semilla 0, MSE) y de A6-P2 (semilla 0, α = 2) coincide exactamente (diferencia 0 en historial y predicciones); `compute_metrics` agrega, en todo el conjunto y por zona, sesgo, desviación del error, sobreestimación y subestimación medias, y los aportes de cada una al score NASA; las métricas de v04 se reproducen (diferencia máxima 7e-15) y las identidades se cumplen en las 35 corridas analizadas; tabla en `results/etapa0_metricas_val.csv` |
| 1 | Completada: con la configuración final, en la zona final, se sobreestima en el 48 % de las ventanas y se subestima en el resto, con partes casi iguales (0,77 y 0,78 ciclos; 1,36 y 1,43 cuando ocurren): sesgo final −0,04. El sesgo global (−0,93) viene casi todo de la zona temprana (aporte −1,11), subestimada por el recorte en 62 (las ventanas con etiqueta 62 se subestiman 6,24 ciclos). El sesgo final cercano a 0 es el promedio del motor 18 (−1,04) y el 20 (+1,07). En la zona final el sesgo explica el 0,2 % del RMSE² de la configuración final y el 35 % del de la base. Figuras `etapa1_descomposicion.png`, `etapa1_sesgo_vs_rul.png` y `etapa1_distribucion_error.png`; tabla `etapa1_descomposicion.csv` |
| 2 | Completada: el score es pequeño porque con errores de 1 a 2 ciclos cada ventana aporta 0,08 a 0,22 y aquí se promedia; el 0,14 de la configuración final equivale a sobreestimar 1,3 ciclos o subestimar 1,7 en todas las ventanas. Verificado en las fuentes que la bibliografía **suma** el score (Sateesh Babu et al., 2016, ec. 1; Li, Ding y Sun, 2018, ec. 5; Li, Wang y Li, 2022, en DS02, 5,78 × 10⁵); Saxena et al. (2008) no se consultó directamente. Con errores pequeños el score ≈ Sobreest. media/10 + Subest. media/13 (8 % a 17 % por debajo del real). Sobre 60 corridas el score sigue al RMSE (Spearman 0,95); los desacuerdos se deben a la asimetría, no a la linealidad (Spearman con el MAE simétrico 0,91 y con el MAE asimétrico 0,99): A6-P2 con α = 2 cambia sobreestimación por subestimación, el RMSE empeora (1,83 → 2,03) y el score casi no (0,14 → 0,15). Tabla de rangos de las métricas; el "67 sobre 100" no corresponde a ninguna. Figuras `etapa2_funcion_score.png` y `etapa2_score_vs_rmse.png` |
| 3 | Completada: el sesgo global es métrica desde v01 (en `MAIN_METRICS` desde v02, para mostrar la dirección del error, criterio C4); en v04 la IA lo usó para explicar el pooling porque estaba en la tabla, sin calcular el sesgo de la zona final. Comprobación: A6-P2 sí baja las predicciones en la zona final (+0,70 → −0,03 con MSE; −0,04 → −0,62 con α = 2), menos que lo que sugería el global (−1,32 y −1,83); A6-P1 no lo hace con MSE (+0,74). Al separar el cambio de RMSE² final en sesgo² y varianza, la varianza pesa igual o más (MSE, A6-P2: −0,43 frente a −0,12 en mediana; α = 2: la varianza es el término mayor en 9 de 10 pares), así que el mecanismo de v04 se sostiene solo en parte. Discusión de v04 corregida y registrada en `seleccion.json`; la decisión de v04 no cambia. Aclaraciones de la reunión verificadas con *hooks*: dropout después del max pooling, el pooling cambia pasos de tiempo (30 → 15 → 7) y no canales, un solo dropout por arquitectura heredado de v01. Figura `etapa3_sesgo_vs_varianza.png` |
| 4 | Completada (documentación): `Guia_Estudio.md`, `Preparacion_Defensa.md`, `Metodologia_IA.md` y `Diagramas.md` (copias de v04 ampliadas), `README.md` de la raíz apuntando a v05 y sección final del notebook. Hallazgos al revisar: el "67 sobre 100" de la reunión es, muy probablemente, el % de ventanas sobreestimadas en la zona final en la prueba oficial de v03 (67,4 %), y la columna "Sesgo" de la tabla de prueba era el sesgo global (se rotula así en v05). Pendiente: commit y subida a GitHub, cuando el estudiante lo apruebe |
