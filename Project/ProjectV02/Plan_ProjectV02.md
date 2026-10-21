# Plan de implementación — ProjectV02

Plan derivado de la **Reunión 03** con el profesor guía (Mauricio Araya L.), en la
que se presentó ProjectV01. Todas las etapas se implementan en un único notebook,
`RULv02.ipynb`.

## Decisiones confirmadas

| Punto | Decisión |
|---|---|
| Normalización | Por condición de vuelo `Fc` en lugar de la estandarización global. **Requiere revisión, ver la etapa 2** |
| CNN + atención (A6) | Se incluye en la ablación |
| Umbral de zona final | **RUL ≤ 20 ciclos** |
| v02 anterior (pérdida asimétrica) | Borrado. Se conserva solo la caché de datos, `results/cache_step10.*` |
| Organización | Un solo notebook con todas las etapas |

**Justificación del umbral de 20 ciclos.** Corresponde al criterio operacional que
planteó el profesor, que la estimación debe ser confiable cuando quedan menos de 20
ciclos. Además deja suficientes muestras para métricas estables: el 27,7 % de los
registros de `dev` y el 34,2 % de los de prueba. Las zonas de análisis quedan en
tramos de 20 ciclos hasta el umbral PwL de 62: **final** (≤ 20), **intermedia**
(20–40) y **temprana** (> 40).

## Pedidos de la Reunión 03 y etapa en que se resuelven

| Pedido | Etapa |
|---|---|
| No usar media ± σ con pocas semillas; mostrar valores individuales o mín.–máx. | 0 |
| Enfocarse en la precisión al final de la vida útil; la sobreestimación temprana no es problema | 0 |
| Línea base y ablación: CNN, LSTM, CNN+LSTM, LSTM+Atención, CNN+Atención | 1 |
| Citar a los autores originales de cada arquitectura implementada | 1 |
| Elegir un mecanismo de atención | 1 |
| Normalización por condición de operación vs. global | 2 |
| Probar la pérdida asimétrica sobre el mecanismo elegido | 3 |
| Validación cruzada dejando una unidad fuera, con la mejor combinación | 4 |
| Diagrama más didáctico: grupos, porcentajes, dibujos | 5 |
| Preparar la defensa con dominio de los detalles técnicos | 6 |
| **No** comparar con Transformers ni con técnicas clásicas | Excluido |

**Principio del diseño:** un factor por etapa (arquitectura → normalización →
pérdida), fijando la elección de cada etapa para la siguiente. Limitación declarada:
no se exploran interacciones entre factores.

---

## Etapa 0 — Infraestructura común (completada)

Secciones de `RULv02.ipynb` que comparten todas las etapas:

1. **Configuración única** de hiperparámetros, zonas y semillas.
2. **Datos:** carga con caché, partición por unidad (55,3 % / 25,5 % / 19,2 % de los
   registros), PwL derivado de `hs` y ventaneo restringido al ciclo de vuelo.
3. **Normalización extensible:** función `fit_normalizer(método, …)` que solo aprende
   de las unidades de entrenamiento. En la etapa 0 se implementa `global`; los métodos
   de la etapa 2 se agregan ahí mismo.
4. **Catálogo de arquitecturas A1–A6**, con referencia bibliográfica y número de
   parámetros. A3 y A5 son, parámetro a parámetro, las arquitecturas de v01.
5. **Pérdida** $\mathcal{L}_\alpha$ (MSE ponderado asimétrico; con α = 1 es exactamente
   el MSE).
6. **Entrenamiento reanudable:** cada corrida se identifica por (arquitectura,
   normalización, α, semilla, partición) más un hash de los hiperparámetros, y guarda
   en `results/runs/` su checkpoint, su historial y sus predicciones. Al re-ejecutar
   el notebook, las corridas ya completas se cargan en vez de reentrenarse. Esto hace
   viable un notebook único de unas 2–2,5 h de cómputo total.
7. **Métricas:** globales, por zona (con la **zona final como métrica principal**) y
   por unidad.
8. **Reporte según el profesor:** valores por semilla, **mediana y rango [mín. – máx.]**.
   Sin media ± σ ni pruebas de significancia. Curvas por semilla con transparencia.
9. **Criterios de selección pre-registrados** en `results/seleccion.json`, escritos
   antes de ver resultados.
10. **Verificación:** A3 y A5 con normalización global y MSE, semillas 0–2,
    reproducen el RMSE de validación de v01 con diferencia 0. Estas corridas quedan en
    caché y la etapa 1 las reutiliza.

## Etapa 1 — Línea base y ablación de arquitecturas (completada)

> **Resultado:** según el criterio pre-registrado, pasa **A6 (CNN + atención)**, con una
> mediana de RMSE final en validación de 2,00 [1,77 – 3,02], frente a 2,60 [2,26 – 2,74]
> de A3. Cinco de las seis arquitecturas son indistinguibles en la zona final; solo A3
> se separa, hacia peor. Salvedades declaradas en la sección 1.5 del notebook: empate con
> A1 (0,01 ciclos), atención casi uniforme y límite de épocas alcanzado en una semilla.

| Id | Arquitectura | Inspirada en | Rol |
|---|---|---|---|
| A1 | 1D-CNN + promedio temporal | Sateesh Babu et al. 2016 [3]; Li, Ding y Sun 2018 [4] | Línea base simple |
| A2 | LSTM | Zheng et al. 2017 [5]; Wu et al. 2018 [6] | Línea base recurrente |
| A3 | CNN + LSTM | *falta referencia* | Base híbrida (= v01 `last`) |
| A4 | LSTM + atención | de Oliveira da Costa et al. 2019 [7] | Atención sin CNN |
| A5 | CNN + LSTM + atención | Li et al. 2022 [8] | Propuesta (= v01 `attention`) |
| A6 | CNN + atención | — | Atención sin LSTM |

- Son versiones **inspiradas** en esos trabajos, no réplicas exactas; así se nombran en
  la memoria. Comparten los tamaños de bloque (32 y 64 unidades) y el cabezal, pero
  **no** el número de parámetros: A1 tiene 10.273, A2 23.617, A3 29.025, A4 27.841,
  A5 33.249 y A6 14.497. Ese número se reporta junto a cada resultado.
- Configuración: normalización global, MSE y 5 semillas, es decir 30 corridas
  (6 ya existen desde la etapa 0).
- **Criterio de selección:** entre las arquitecturas con atención (A4, A5, A6), la de
  menor **mediana de RMSE en la zona final** de validación pasa a las etapas 2 y 3. A3
  sigue como referencia sin atención.

## Etapa 2 — Normalización por condición de operación (completada)

> **Decisión:** se implementaron **N1a y N1b**. Resultado con el criterio pre-registrado:
> **se mantiene la normalización global**. RMSE final de A6 en validación: global 2,00
> [1,77 – 3,02], N1a 2,02 [1,84 – 2,43] (empate; rango más estrecho), N1b 3,29
> [2,90 – 4,06] (empeora claramente). N1a mejora a A3 (2,60 → 2,35). La silueta eligió
> K = 2 regímenes. Limitación: como todo `dev` es `Fc = 3`, la validación no puede
> medir el cambio de dominio de las unidades de prueba 14 y 15.

**Hallazgo de la etapa 0:** en DS02, `Fc` es constante por motor y **todas las
unidades de `dev` son `Fc = 3`**. Las unidades de prueba 14 (`Fc = 1`) y 15 (`Fc = 2`)
pertenecen a clases de vuelo que no aparecen en entrenamiento. Por lo tanto:

- En `dev`, normalizar por `Fc` es idéntico a normalizar globalmente.
- Para las unidades 14 y 15 no existen estadísticas de entrenamiento con que
  normalizar.

Alternativas para decidir:

| Opción | Método | Ventaja | Costo |
|---|---|---|---|
| N1a | **Regímenes de operación:** k-means sobre W (alt, Mach, TRA, T2) ajustado con entrenamiento; z-score de cada sensor según su régimen | Es "por condición de operación" en sentido estricto y cubre todas las clases de vuelo | K es un hiperparámetro |
| N1b | **Línea base por motor:** cada sensor se normaliza con las estadísticas de los primeros 10 ciclos (sanos) del propio motor | Coincide con "de cada motor" y ataca directamente el cambio de dominio que causa el sesgo optimista en prueba | Usa datos del motor evaluado (sus primeros vuelos), lo que es realista en operación |
| N1c | `Fc` literal | Es la propuesta original | Requiere entrenar con otros subconjuntos (DS01–DS08) que contengan `Fc` 1 y 2; cambia el alcance |

- Configuración: arquitectura elegida + A3 × {global, método elegido} × 5 semillas.
- **Criterio:** se adopta el método nuevo si reduce la mediana del RMSE en la zona
  final de validación de la arquitectura elegida.

## Etapa 3 — Pérdida asimétrica (completada)

> **Resultado:** se elige **α = 2**, que supera a la MSE en las 5 de 5 semillas tanto en
> score NASA final (0,18 → 0,14) como en RMSE final (2,00 → 1,83). La sobreestimación
> media final baja un 26 %. Costo: leve aumento del error global y en la zona temprana.
> α = 4 resulta demasiado conservador. **Configuración final: A6 + normalización global
> + α = 2.**

- Arquitectura y normalización elegidas, α ∈ {1, 2, 4} y 5 semillas.
- **Criterio:** α que minimiza la mediana del **score NASA en la zona final** de
  validación. Combina precisión y asimetría solo donde importa.

## Etapa 4 — Validación cruzada dejando una unidad fuera (completada)

> **Prueba oficial (evaluada una sola vez):** RMSE final 1,78 [1,50 – 1,97] de la
> configuración final frente a 2,97 [2,47 – 3,30] de la base A3 + MSE, sin solapamiento
> entre semillas. Mejoran las tres unidades de prueba.
> **Validación cruzada:** la final es mejor solo en 2 de 6 unidades (16 y 20) y en 7 de 18
> pares unidad-semilla, pero reduce la sobreestimación final en 16 de 18. La ventaja
> depende del sesgo del modelo base en cada motor: ayuda donde es optimista y perjudica
> donde ya es pesimista. A6 produce predicciones extremas (hasta 189 ciclos) en algunos
> folds; acotar la salida queda como trabajo futuro.

- 6 folds sobre las unidades de `dev`: en cada fold, 1 unidad de evaluación, 1 de
  validación (rotando) y 4 de entrenamiento.
- La normalización se **reajusta en cada fold** con solo sus unidades de entrenamiento.
- Configuración final vs. A3 (global, MSE), con 3 semillas por fold: 36 corridas.
- Resultados por unidad, sin promediar. Cierre con una única evaluación sobre el test
  oficial (11, 14, 15).

## Etapa 5 — Diagramas didácticos (completada)

> **Decisión:** se usan diagramas **Mermaid en Markdown** (`Diagramas.md`), igual que en
> ProjectV01, en lugar de draw.io.

- Un color fijo por tipo de bloque en todos los diagramas y porcentajes de la partición.
  Sin emojis ni íconos.
- Ocho diagramas: visión general por etapas, pipeline de datos, catálogo A1–A6 en filas
  paralelas, módulo de atención, normalizaciones, pérdida asimétrica, validación cruzada
  y resultado en prueba.

## Etapa 6 — Consolidación y defensa (completada)

> Sección 6 del notebook: tabla de decisiones generada desde `seleccion.json`, resumen
> de resultados, conclusiones, limitaciones y trabajo futuro. Las preguntas probables y
> sus respuestas están en `Preparacion_Defensa.md`, que termina con una lista de
> pendientes por verificar antes de la defensa.

- Discusión final y tabla de decisiones (qué se eligió, con qué evidencia y con qué
  referencia).
- Documento de preguntas probables y sus respuestas. Incluye aclarar que en v01 solo
  `attention` era un mecanismo de atención; `last` y `mean` son formas de agregación.

## Costo computacional estimado (CPU)

| Etapa | Corridas nuevas | Tiempo aprox. |
|---|---|---|
| 0 Verificación | 6 | 7 min |
| 1 Ablación | 24 | 30 min |
| 2 Normalización | 10–20 | 15–25 min |
| 3 Pérdida asimétrica | 10 | 12 min |
| 4 Validación cruzada | 36 | 45 min |
| **Total** | **≈ 90–100** | **≈ 2 h** |
