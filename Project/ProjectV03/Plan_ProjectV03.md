# Plan de implementación — ProjectV03

Plan derivado de la **Reunión 04** con el profesor guía (Mauricio Araya L.), en la que
se presentó ProjectV02. Todas las etapas se implementan en un único notebook,
`RULv03.ipynb`, con **todo el código comentado en español**. ProjectV02 no se modifica.

## Decisiones confirmadas

| Punto | Decisión |
|---|---|
| Arquitecturas | Las mismas seis de la ablación de v02 (A1–A6), sin cambios |
| Normalización | Global, fija. Los regímenes de operación solo se estudian en la etapa 4 |
| Atención por producto punto (*Attention Is All You Need*) | Se explica en la guía de estudio; no se agrega al experimento |
| Entrenamiento | Igual que en v02: early stopping con paciencia de 5 épocas y máximo de 30 |
| Curvas de aprendizaje | Todas graficadas en un eje común de 1 a 30 épocas, para compararlas visualmente |
| Organización | Un solo notebook con todas las etapas |

### Reutilización de las corridas de v02

Como el entrenamiento es idéntico al de v02 y el pipeline es determinista, volver a
entrenar daría exactamente los mismos resultados. v03 copia las 96 corridas de v02 y lo
justifica con una verificación: reentrena desde cero tres de ellas con el código de v03
y comprueba que el historial y las predicciones coinciden exactamente. Lo nuevo de v03
está en la presentación de los resultados, el análisis de regímenes y la documentación.

### Aclaración sobre la convolución "32 → 64"

En A1 y A6, la segunda convolución pasa de 32 a 64 **canales (filtros)**. La dimensión
temporal se mantiene en 30 pasos (`kernel_size=3`, `padding=1`, paso 1), así que no
crece ninguna dimensión espacial ni temporal. Los diagramas de v03 rotularán cada bloque
con la forma completa del tensor, por ejemplo `(B, 30, 32) → (B, 30, 64)`.

Al mantener las arquitecturas de v02 se mantiene también su limitación conocida: la
ablación no es estricta entre extractores. Los pares con y sin atención (A1–A6, A2–A4,
A3–A5) sí difieren en un solo factor. La limitación queda declarada en el notebook.

## Pedidos de la Reunión 04 y etapa en que se resuelven

| Pedido | Etapa |
|---|---|
| Revisar la convolución "32 → 64" | Aclarado arriba; diagramas en la etapa 5 |
| Curvas de aprendizaje con el mismo máximo de épocas (30) en todos los gráficos | 1 |
| Referencia bibliográfica y nombre de la pérdida asimétrica | 2 y 5 |
| Mejorar los regímenes de operación (DBSCAN o Spectral Clustering) y moverlos a una sección de "doble clic" | 4 |
| Analizar los resultados por régimen de operación | 4 |
| Explicar la sobreestimación de la unidad 20 | 4 |
| Documentar la metodología de uso de IA | 5 |
| Estudiar el código y la teoría, incluido *Attention Is All You Need* | 5 y código comentado en todas las etapas |

---

## Etapa 0 — Infraestructura

- Infraestructura de v02 reescrita y comentada en español.
- Copia de las corridas de v02 y verificación por reentrenamiento independiente.

## Etapa 1 — Ablación de arquitecturas con curvas comparables

- A1–A6, normalización global, MSE, 5 semillas (corridas de v02).
- Curvas de aprendizaje de todas las arquitecturas en un eje común de 1 a 30 épocas, con
  la época conservada de cada semilla marcada.
- Mismo criterio pre-registrado que en v02.

## Etapa 2 — Pérdida asimétrica

- Arquitectura elegida en la etapa 1, α ∈ {1, 2, 4}, 5 semillas (corridas de v02).
- Referencias verificadas: regresión por expectiles (Newey y Powell, 1987) y trabajos de
  pronóstico de RUL con pérdidas asimétricas.

## Etapa 3 — Validación cruzada y prueba

- Validación cruzada dejando un motor fuera y resultados en el conjunto de prueba oficial
  (corridas de v02; la evaluación en prueba ya se hizo una vez en v02 y aquí solo se
  vuelve a presentar).

## Etapa 4 — Análisis "doble clic": regímenes de operación

Posterior a los resultados principales. Es análisis: no modifica la configuración final.

1. Regímenes definidos con k-means (v02), DBSCAN y Spectral Clustering, comparados con
   métricas de calidad (silueta, Davies-Bouldin) y por su sentido físico como fases de
   vuelo. Spectral Clustering se ajusta sobre una submuestra y el resto se asigna por
   vecinos más cercanos; DBSCAN se calibra con el gráfico de distancia al k-ésimo vecino.
2. Error del modelo final por régimen de operación.
3. Explicación de la sobreestimación de la unidad 20.

## Etapa 5 — Documentación

- `Metodologia_IA.md`: metodología de uso de IA en el desarrollo y la experimentación.
- `Guia_Estudio.md`: recorrido por el código y su teoría, incluido qué es el promedio
  temporal, la forma de los tensores, la atención aditiva (Bahdanau) frente a la
  atención por producto punto escalado (*Attention Is All You Need*).
- `Diagramas.md` y `Preparacion_Defensa.md` actualizados para v03.

## Costo computacional estimado (CPU)

| Etapa | Entrenamientos nuevos | Tiempo aproximado |
|---|---|---|
| 0 Verificación | 3 | 5 min |
| 1 a 3 | 0 (corridas de v02) | minutos |
| 4 Regímenes | sin entrenamiento | 10–20 min |

## Estado

| Etapa | Estado |
|---|---|
| 0 | Completada: 96 corridas de v02 copiadas; el reentrenamiento de A3, A5 y A6 coincide exactamente (historial y predicciones) |
| 1 | Completada: curvas en eje común de 30 épocas y tabla de convergencia; se elige A6, como en v02 |
| 2 | Completada: α = 2 (como en v02); referencias verificadas; la condición de equilibrio de los expectiles se cumple aproximadamente en validación |
| 3 | Completada: resultados idénticos a v02; diagnóstico de predicciones fuera de rango incorporado al notebook; el motor 20 es el único que la configuración final sigue sobreestimando |
| 4 | Completada: ningún método de agrupamiento sobre W recupera las fases de vuelo (ascenso y descenso se superponen en W); el descenso es la fase de mayor error; el sesgo de cada motor sigue a su vida total (Spearman −0,83 base, −0,89 final), lo que explica la sobreestimación del motor 20 |
| 5 | Completada: `Metodologia_IA.md`, `Guia_Estudio.md`, `Diagramas.md` (10 diagramas con la forma de cada tensor) y `Preparacion_Defensa.md` |
