# Plan de implementación — ProjectV04

Plan derivado de la **Reunión 05** con el profesor guía (Mauricio Araya L.), en la que se
presentó ProjectV03. Todas las etapas se implementan en un único notebook, `RULv04.ipynb`,
con **todo el código comentado en español**. ProjectV03 no se modifica.

## Pedidos de la Reunión 05 y etapa en que se resuelven

| Pedido | Etapa |
|---|---|
| Investigar por qué la red convolucional no tiene una capa de *pooling* (max pooling) | 1 y 2 |
| Correr el clustering con más grupos para intentar mejorar la silueta de Spectral Clustering | 3 |
| Verificar si el clustering usa todas las dimensiones o solo dos (velocidad y altitud) | 3 |
| Subir el proyecto a GitHub y enviar el enlace | Hecho: `github.com/matii-gr/Memoria_RUL` |

### Aclaraciones sobre lo conversado en la reunión

- **Silueta de 0,45 y "18 grupos".** Son dos resultados distintos de RULv03. Spectral
  Clustering obtuvo su mejor silueta (0,445) con K = 2; los **18 grupos** son de DBSCAN,
  cuya silueta fue negativa (−0,48). La etapa 3 los presenta por separado.
- **"16 dimensiones".** En los datos no existe un conjunto de 16 variables: W tiene 4, X_s
  tiene 14 y el modelo recibe 18 (W + X_s). En RULv03 el clustering usa las **4 variables de
  W** (altitud, Mach, TRA, T2), y la figura muestra solo el plano Mach–altitud, que es una
  proyección de esas 4. Por eso parecía hecho sobre 2 dimensiones.

## Decisiones confirmadas

| Punto | Decisión |
|---|---|
| Arquitecturas | Las mismas A1–A6 de v02 y v03, sin cambios. El pooling se agrega como variantes nuevas de A6 |
| Pooling | A6 con `MaxPool1d(2)` en dos ubicaciones, con α = 1 y α = 2, 5 semillas |
| Adopción del pooling | Regla pre-registrada sobre validación. Si se adopta, validación cruzada y una evaluación en prueba declarada como segunda evaluación del conjunto de prueba |
| Dimensiones del clustering | Se verifica con código que se usan las 4 de W y además se compara con un clustering sobre las 18 que ve el modelo (W + X_s) |
| Número de grupos | K de 2 a 20 en k-means y Spectral Clustering |
| Entrenamiento | Igual que en v02 y v03: early stopping con paciencia 5 y máximo 30 épocas; curvas en un eje común de 1 a 30 |
| Organización | Un solo notebook con todas las etapas |

---

## Etapa 0 — Infraestructura

- Infraestructura de v03 (datos, ventanas, normalización global, `run`, métricas, figuras),
  con el mismo código comentado.
- `RULNet` recibe una opción nueva, `pool`, que agrega `MaxPool1d(2)`. Como el pooling no
  tiene parámetros, no cambia el orden en que se inicializan las demás capas: A1–A6 deben
  seguir dando exactamente los mismos pesos con la misma semilla.
- Se copian las corridas de v02 (mismo `HP_HASH = 405955a4`) y se **verifica** que el código
  de v04 las reproduce: se reentrenan desde cero A3 y A6 (semilla 0, MSE) y se comprueba
  que el historial y las predicciones coinciden exactamente.
- Resumen de la configuración vigente al iniciar v04 (A6 + global + α = 2), cargado desde
  las corridas copiadas, como punto de comparación.

## Etapa 1 — Pooling: bibliografía y experimento

### 1.1 Por qué no hay pooling (investigación)

- Qué hace el max pooling en una CNN 1D: reduce la longitud temporal (30 → 15 con
  `kernel_size = 2`) y da tolerancia a pequeños desplazamientos en el tiempo.
- Revisión de las referencias de la ablación (Sateesh Babu et al., 2016; Li, Ding y Sun,
  2018; Li et al., 2022, entre otras) para ver si usan pooling y cómo lo justifican. **Cada
  afirmación se verifica en la fuente antes de escribirla**; lo que no se pueda verificar
  se declara como no verificado.
- Argumentos a contrastar con el experimento: con ventanas cortas (30 pasos) el pooling
  descarta resolución temporal, y la atención pondera pasos que, después del pooling,
  resumen 2 o 4 pasos originales.

### 1.2 Variantes

| Id | Arquitectura | Forma del tensor antes de la atención |
|---|---|---|
| A6 | CNN + atención (sin pooling, v03) | (B, 30, 64) |
| A6-P1 | Pooling solo tras la 1.ª convolución | (B, 15, 64) |
| A6-P2 | Pooling tras cada convolución | (B, 7, 64) |

### 1.3 Regla de adopción (se guarda en `seleccion.json` antes de entrenar)

Con α = 2, la configuración final vigente:

1. Se elige la variante con pooling (P1 o P2) de menor mediana de **RMSE final** (RUL ≤ 20)
   en validación sobre 5 semillas.
2. Esa variante reemplaza a A6 sin pooling solo si (a) su mediana de RMSE final es menor y
   (b) es mejor que A6 sin pooling en al menos 4 de las 5 semillas, comparando semilla con
   semilla.

La regla es exigente a propósito: adoptar el pooling implica volver a usar el conjunto de
prueba. Los resultados con α = 1 (MSE) se informan para ver si el efecto del pooling
depende de la pérdida, pero no deciden.

### 1.4 Análisis

- Tabla y gráfico de puntos por semilla (RMSE final, sesgo, score NASA final).
- Curvas de aprendizaje en el eje común de 1 a 30 épocas.
- Pesos de atención con 15 y 7 pasos, comparados con los 30 de A6.
- Número de parámetros (debe ser igual en las tres variantes) y tiempo por época.

## Etapa 2 — Validación cruzada y prueba (condicional)

Solo si la regla de la etapa 1 adopta una variante con pooling:

- Validación cruzada dejando un motor fuera con la variante adoptada (6 motores × 5
  semillas), comparada con la configuración final de v03 y con la base A3.
- **Una única evaluación** de la variante adoptada en el conjunto de prueba oficial,
  declarada en el notebook y en `seleccion.json` como segunda evaluación del conjunto de
  prueba en el trabajo.

Si la regla no adopta el pooling, la etapa se reduce a dejarlo registrado y la
configuración final sigue siendo la de v03.

## Etapa 3 — Regímenes de operación, revisión "doble clic"

Es análisis: no modifica la configuración final.

### 3.1 Verificación de las dimensiones

- Imprimir la forma de la matriz que recibe cada método, `(8000, 4)`, y el nombre de cada
  columna.
- Las 6 proyecciones por pares de las 4 variables de W, coloreadas por fase de vuelo y por
  grupo, en lugar de solo Mach–altitud.
- Análisis de componentes principales (PCA) de W: varianza explicada por cada componente,
  para saber si los datos son efectivamente de baja dimensión.

### 3.2 Más grupos

- k-means y Spectral Clustering con K de 2 a 20 sobre W, en la misma submuestra de 8.000
  registros de entrenamiento: silueta, Davies-Bouldin y Rand ajustado contra las fases de
  vuelo, en función de K.
- La silueta no necesariamente mejora al aumentar K. Se informa lo que resulte, sin elegir
  K por la coincidencia con las fases.

### 3.3 Espacio de 18 dimensiones (W + X_s)

- k-means, Spectral Clustering (K de 2 a 20) y DBSCAN (`eps` desde el codo,
  `min_samples = 2 × 18`) sobre W + X_s, estandarizados con los motores de entrenamiento.
- Precaución: los sensores X_s contienen la degradación del motor, así que los grupos
  podrían separar estados de salud en vez de regímenes. Se mide cuánto se asocian los
  grupos a cada motor y al RUL.
- Comparación final de los dos espacios (4D y 18D) en una sola tabla.

### 3.4 Error por régimen

- Si el agrupamiento de mayor silueta cambia respecto de v03, se repite el error por
  régimen de la configuración final (validación y prueba, solo descriptivo).

## Etapa 4 — Documentación

- `Diagramas.md`: A6-P1 y A6-P2 con la forma de cada tensor.
- `Guia_Estudio.md`: qué es el pooling y qué hace sobre el eje temporal; qué significa
  agrupar en 4 o en 18 dimensiones; silueta y su dependencia de K.
- `Preparacion_Defensa.md`: preguntas probables sobre pooling y clustering.
- `Metodologia_IA.md`: registro actualizado de los errores de la herramienta detectados.
- `README.md` de la raíz: apuntar a ProjectV04 como versión vigente.

## Costo computacional estimado (CPU)

| Etapa | Entrenamientos nuevos | Tiempo aproximado |
|---|---|---|
| 0 Verificación | 2 | 3 min |
| 1 Pooling | 20 (2 variantes × 2 valores de α × 5 semillas) | 25 min |
| 2 Validación cruzada (condicional) | 30 (6 motores × 5 semillas) | 40 min |
| 3 Regímenes | sin entrenamiento | 15–30 min (Spectral con K hasta 20 en 4D y 18D) |

## Estado

| Etapa | Estado |
|---|---|
| 0 | Completada: opción `pool` en `RULNet` (A6, A6-P1 y A6-P2 tienen 14.497 parámetros y los mismos pesos iniciales con la misma semilla); 96 corridas de v02 copiadas; el reentrenamiento de A3 y A6 coincide exactamente (diferencia 0); regla de adopción del pooling registrada en `seleccion.json` antes de entrenar |
| 1 | Completada: el pooling no se adopta. Con α = 2, A6 obtiene 1,83 [1,60 – 2,17] de RMSE final en validación; A6-P1 y A6-P2, 2,03, y cada una mejora a A6 en solo 2 de 5 semillas. Con MSE, A6-P2 sí mejora a A6 (1,71 frente a 2,00, 4 de 5 semillas), porque baja las predicciones, un efecto que con α = 2 ya aporta la pérdida asimétrica. Bibliografía verificada: Li, Ding y Sun (2018) desaconsejan el pooling de forma explícita con ventanas de 30 pasos |
| 2 | No aplica: la etapa 1 no adoptó el pooling; la configuración final sigue siendo la de v03 y el conjunto de prueba no se vuelve a usar |
| 3 | Completada: el clustering de v03 sí usaba las 4 variables de W, pero altitud, Mach y T2 son casi redundantes (CP1 + CP2 = 96 % de la varianza), así que el resultado se comporta como un corte por altitud. Más grupos no mejoran la silueta (los datos forman un continuo). Las fases se distinguen por TRA; en 18D, k-means con K = 3 se acerca más a ellas (Rand ajustado 0,32 frente a 0,11) sin separar motores ni estados de salud (η² del RUL 0,002). El error por régimen se calculó solo en validación, para no consultar el conjunto de prueba |
| 4 | Completada: `Diagramas.md` (9 diagramas: pipeline, A6 con y sin pooling con la forma de cada tensor, decisión del pooling y regímenes), `Guia_Estudio.md` (guía de v03 más pooling y agrupamiento), `Preparacion_Defensa.md` (preguntas nuevas sobre pooling y clustering, y corrección de la explicación de v03 sobre las fases), `Metodologia_IA.md` (decisiones, salvaguardas y errores de v04), `README.md` de la raíz apuntando a v04 y sección final del notebook con el resumen de v04 |
