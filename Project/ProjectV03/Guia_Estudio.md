# Guía de estudio — ProjectV03

Recorrido por el código de `RULv03.ipynb` y por la teoría que lo sustenta, pensado para
poder explicar cada parte en la defensa. Las secciones siguen el camino de un dato: desde
el archivo HDF5 hasta el número que predice la red y cómo se evalúa.

---

## 1. Del archivo a la ventana

1. **Lectura (sección 0.3).** El archivo DS02 trae registros a 1 Hz. Se lee 1 de cada 10
   (`[::10]`), es decir, un registro cada 10 segundos.
2. **Entradas.** Cada registro tiene 18 valores: 4 condiciones de vuelo `W` (altitud, Mach,
   ángulo del acelerador TRA, temperatura de entrada T2) y 14 sensores físicos `X_s`.
3. **Etiqueta.** El RUL de cada registro, en ciclos de vuelo, recortado a 62 (sección 0.4).
4. **Normalización (sección 0.6).** A cada uno de los 18 canales se le resta su media y se
   divide por su desviación estándar, calculadas **solo con los motores de entrenamiento**.
5. **Ventanas (sección 0.7).** Se toman 30 registros consecutivos de un mismo vuelo: 300 s,
   es decir, 5 minutos. La ventana avanza de 10 en 10 registros y nunca cruza de un vuelo al
   siguiente. Su etiqueta es el RUL del último registro.

**Resultado:** un tensor de forma `(N, 30, 18)`, donde N es el número de ventanas. Durante el
entrenamiento se procesan en lotes de 256, así que la red recibe tensores `(B, 30, 18)` con
B = 256.

### Notación de formas

| Letra | Significado | Valor |
|---|---|---|
| B | Ventanas en el lote (*batch*) | 256 |
| T | Pasos de tiempo de la ventana | 30 |
| F | Canales de entrada | 18 |
| C | Canales (filtros) de una convolución | 32 o 64 |
| H | Dimensión de las características que llegan a la agregación | 64 |

---

## 2. Recorrido por la red: A6 (CNN + atención)

| Paso | Operación | Forma de salida | Parámetros |
|---|---|---|---|
| Entrada | — | `(B, 30, 18)` | — |
| `permute(0, 2, 1)` | Reordena: Conv1d exige canales antes que tiempo | `(B, 18, 30)` | — |
| `conv` | Convolución 1D, 18 → 32 canales, kernel 3, padding 1 | `(B, 32, 30)` | 18 × 32 × 3 + 32 = 1.760 |
| `bn` | Normalización por lote de cada canal | `(B, 32, 30)` | 2 × 32 = 64 |
| `relu`, `dropout` | No linealidad y apagado aleatorio (20 %) | `(B, 32, 30)` | — |
| `conv2` | Convolución 1D, 32 → 64 canales | `(B, 64, 30)` | 32 × 64 × 3 + 64 = 6.208 |
| `bn2`, `relu` | | `(B, 64, 30)` | 2 × 64 = 128 |
| `permute(0, 2, 1)` | De vuelta a tiempo antes que canales | `(B, 30, 64)` | — |
| `attention` | Promedio ponderado de los 30 pasos | `(B, 64)` | 64 × 64 + 64 + 64 = 4.224 |
| `head` | Linear 64 → 32, ReLU, Linear 32 → 1 | `(B, 1)` | 2.080 + 33 = 2.113 |
| **Total** | | | **14.497** |

### 2.1 Convolución 1D

Una convolución 1D con 18 canales de entrada, 32 filtros y kernel 3 calcula, para cada
filtro `o` y cada paso `t`:

$$y_{o,t} = b_o + \sum_{c=1}^{18} \sum_{k=-1}^{1} W_{o,c,k}\; x_{c,\,t+k}$$

Es decir, cada filtro mira **3 pasos consecutivos de los 18 canales a la vez** (54 valores)
y produce un número. Se aplican 32 filtros distintos, así que en cada paso salen 32
características.

**"32 → 64" son canales, no tiempo.** La segunda convolución recibe 32 canales y produce 64.
La longitud temporal no cambia. Con relleno (*padding*) `p`, kernel `k` y paso `s`:

$$L_{salida} = \frac{L_{entrada} + 2p - k}{s} + 1 = \frac{30 + 2 - 3}{1} + 1 = 30$$

El relleno agrega un cero al inicio y al final de la ventana para que la salida mantenga 30
pasos. Por eso, en los pasos 1 y 30 la convolución "ve" un valor inventado (cero), lo que
explica en parte los picos de atención en los bordes de la ventana (sección 4.4).

### 2.2 Normalización por lote (BatchNorm)

Durante el entrenamiento, cada canal se normaliza con la media y la varianza de ese canal en
el lote actual, y luego se reescala con dos parámetros aprendidos (γ y β). Estabiliza y
acelera el entrenamiento. En evaluación (`model.eval()`) usa promedios acumulados durante el
entrenamiento, para que la predicción de una ventana no dependa de las demás del lote.

### 2.3 ReLU y dropout

- **ReLU:** $\max(0, x)$. Introduce la no linealidad sin la que la red sería una sola
  transformación lineal.
- **Dropout (0,2):** durante el entrenamiento apaga al azar el 20 % de las activaciones y
  escala el resto por $1/0{,}8$. Obliga a la red a no depender de unas pocas neuronas. En
  evaluación se desactiva.

---

## 3. El bloque recurrente: LSTM

Usado en A2, A3, A4 y A5. Recorre la ventana paso a paso y mantiene dos estados: la
**celda** $c_t$ (memoria de largo plazo) y la **salida** $h_t$. En cada paso, tres
compuertas deciden qué olvidar, qué guardar y qué entregar:

$$
\begin{aligned}
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) &&\text{compuerta de olvido}\\
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) &&\text{compuerta de entrada}\\
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g) &&\text{contenido candidato}\\
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) &&\text{compuerta de salida}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t\\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

donde $\sigma$ es la sigmoide (valores entre 0 y 1) y $\odot$ el producto elemento a
elemento.

- **Forma:** con 32 canales de entrada y 64 unidades, entrega `(B, 30, 64)`: un $h_t$ por paso.
- **Parámetros:** $4 \times 64 \times (32 + 64) + 2 \times 4 \times 64 = 25.088$ (las
  matrices W y U de las cuatro compuertas y dos vectores de sesgo por compuerta, como los
  implementa PyTorch).
- **Por qué su salida está acotada:** $h_t = o_t \odot \tanh(c_t)$ siempre está entre −1 y 1.
  Esto explica por qué las arquitecturas con LSTM nunca produjeron predicciones absurdas,
  mientras que A6, que no tiene ninguna capa acotada, sí (etapa 3).
- **Por qué sus salidas son parecidas entre pasos:** $h_t$ depende de $h_{t-1}$, así que
  cada salida resume todo lo visto hasta ese momento. Dos salidas consecutivas comparten
  casi toda su historia, y por eso son muy similares (sección 4.4).

---

## 4. La agregación temporal

Después del extractor hay 30 vectores de 64 características, uno por paso. La agregación
los resume en **un solo vector** de 64, que es lo que recibe el cabezal.

### 4.1 Último estado (A2, A3)

$$\mathbf{z} = \mathbf{h}_{30}$$

Se queda con el último paso y descarta los demás. Con un LSTM tiene sentido, porque
$h_{30}$ ya resume toda la ventana. En el código: `h[:, -1, :]`.

### 4.2 Promedio temporal (A1)

$$\mathbf{z} = \frac{1}{30}\sum_{t=1}^{30} \mathbf{h}_t$$

Es el promedio de los 30 vectores, **con el mismo peso (1/30) para cada paso**. Cada una de
las 64 características del resultado es la media de esa característica a lo largo de la
ventana. En el código: `h.mean(dim=1)`, que promedia sobre la dimensión del tiempo y pasa de
`(B, 30, 64)` a `(B, 64)`.

Ejemplo: si la característica 5 vale 0,2 en los primeros 15 pasos y 0,4 en los últimos 15,
su valor en el resultado es 0,3.

### 4.3 Atención aditiva (A4, A5, A6)

Es un promedio ponderado en el que **los pesos se aprenden** y cambian según el contenido
de cada ventana. Para cada paso `t`:

$$e_t = \mathbf{v}^{\top} \tanh(\mathbf{W}\mathbf{h}_t + \mathbf{b}) \qquad
\alpha_t = \frac{\exp(e_t)}{\sum_{s=1}^{30} \exp(e_s)} \qquad
\mathbf{z} = \sum_{t=1}^{30} \alpha_t\, \mathbf{h}_t$$

1. **Puntaje** $e_t$: cada $h_t$ pasa por una capa lineal (W, b), una tangente hiperbólica y
   un producto con el vector aprendido $\mathbf{v}$. Resultado: un número por paso.
2. **Pesos** $\alpha_t$: la función softmax convierte los 30 puntajes en 30 números positivos
   que suman 1.
3. **Contexto** $\mathbf{z}$: el promedio de los $h_t$ ponderado por los $\alpha_t$.

**Relación con el promedio:** si todos los puntajes fueran iguales, todos los pesos serían
1/30 y la atención sería exactamente el promedio temporal. La atención generaliza el
promedio: puede dar más peso a los pasos que considere más informativos.

**Origen y simplificación.** La forma aditiva viene de Bahdanau, Cho y Bengio (2015). En el
original, propuesto para traducción automática, el puntaje compara cada $h_j$ con el estado
del decodificador, que actúa como **consulta**:
$e_{ij} = \mathbf{v}^{\top}\tanh(\mathbf{W}\mathbf{s}_{i-1} + \mathbf{U}\mathbf{h}_j)$. En
este trabajo no hay decodificador: la consulta se reemplaza por los parámetros aprendidos
$\mathbf{W}$, $\mathbf{b}$ y $\mathbf{v}$, que son los mismos para todas las ventanas. Es una
**atención de agrupamiento** (*attention pooling*): su única función es resumir la
secuencia en un vector. Conviene decirlo con estas palabras si preguntan.

### 4.4 Qué mostraron los pesos y por qué

- **En A6 los pesos son casi uniformes**, cerca de 1/30, con picos en los pasos 2 y 29.
  Los picos coinciden con los bordes donde las dos convoluciones procesan el relleno de
  ceros: probablemente son un efecto de borde, no una elección con significado físico. En
  la práctica, A6 se comporta como un promedio temporal, lo que explica que empate con A1.
- **Una explicación de por qué no hay mucho que elegir:** una ventana cubre 5 minutos de un
  vuelo, y la degradación avanza de un vuelo a otro, no dentro de 5 minutos. Dentro de una
  ventana, el estado de salud del motor es prácticamente constante. Las diferencias entre
  pasos se deben sobre todo a cambios en la condición de operación, no a la salud. Si
  ningún paso es más informativo que otro, el promedio uniforme es casi óptimo, y eso es lo
  que la atención aprende. Es una hipótesis coherente con lo observado, no algo demostrado
  en el trabajo.
- **El argumento del profesor sobre el LSTM (Reunión 04):** como cada $h_t$ resume toda la
  historia anterior, salidas consecutivas son muy parecidas, y la atención no tiene vectores
  distintos entre los que elegir. Aplicar la atención directamente sobre las
  características de la convolución, que son más locales, evita ese problema. Es el diseño
  de A6.

---

## 5. Atención por producto punto escalado (*Attention Is All You Need*)

Vaswani et al. (2017) propusieron el Transformer, basado solo en atención. Su mecanismo
central es la **atención por producto punto escalado**. No se usa en este trabajo, pero el
profesor pidió estudiarla.

### 5.1 Definición

Para cada paso se calculan tres vectores a partir de su entrada $\mathbf{x}_t$, con matrices
aprendidas:

- **Consulta** (*query*) $\mathbf{q}_t = \mathbf{W}_Q \mathbf{x}_t$: qué busca este paso.
- **Clave** (*key*) $\mathbf{k}_t = \mathbf{W}_K \mathbf{x}_t$: qué ofrece este paso para ser
  encontrado.
- **Valor** (*value*) $\mathbf{v}_t = \mathbf{W}_V \mathbf{x}_t$: qué información entrega si
  es elegido.

Con todas las consultas, claves y valores apilados en las matrices Q, K y V:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right) V$$

1. $QK^{\top}$ es una matriz T × T: el producto punto entre la consulta de cada paso y la
   clave de cada paso. Mide cuánto le "interesa" cada paso a cada otro.
2. Se divide por $\sqrt{d_k}$ (la dimensión de las claves).
3. Softmax por filas: cada paso obtiene pesos que suman 1 sobre todos los pasos.
4. Se multiplica por V: **cada paso obtiene su propio promedio ponderado** de los valores de
   todos los pasos.

### 5.2 Por qué se divide por $\sqrt{d_k}$

Si los componentes de q y k son independientes, con media 0 y varianza 1, su producto punto
tiene varianza $d_k$. Con $d_k$ grande, los productos punto toman valores grandes y la
softmax se satura: un peso queda cerca de 1 y los demás cerca de 0, y sus gradientes se
vuelven muy pequeños. Dividir por $\sqrt{d_k}$ devuelve la varianza a 1. Es la justificación
que da el propio artículo.

### 5.3 Autoatención, cabezas múltiples y posición

- **Autoatención:** Q, K y V salen de la misma secuencia, así que cada paso atiende a todos
  los demás. La salida tiene la misma forma que la entrada, `(B, T, d)`: no resume la
  secuencia, la **transforma**.
- **Cabezas múltiples:** se ejecutan varias atenciones en paralelo, cada una con sus propias
  matrices y una dimensión menor, y sus resultados se concatenan. Cada cabeza puede
  especializarse en un tipo de relación.
- **Codificación de posición:** la autoatención no distingue el orden de los pasos (si se
  permutan las entradas, las salidas se permutan igual). Por eso el Transformer suma a cada
  entrada una codificación de su posición.

### 5.4 Comparación con la atención de este trabajo

| | Atención aditiva de agrupamiento (este trabajo) | Producto punto escalado (Transformer) |
|---|---|---|
| Puntaje | $\mathbf{v}^{\top}\tanh(\mathbf{W}\mathbf{h}_t + \mathbf{b})$ | $\mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}$ |
| Consulta | Una sola, fija y aprendida, para todas las ventanas | Una por paso, calculada desde la entrada |
| Salida | Un vector por ventana `(B, 64)` | Una secuencia `(B, T, d)` |
| Pesos | 30 por ventana | T × T por ventana |
| Costo | Proporcional a T | Proporcional a T² |
| Función en la red | Resumir la ventana antes del cabezal | Transformar la secuencia; se apilan capas |

Para resumir una ventana con producto punto, como hace este trabajo, se usaría una sola
consulta aprendida: $\mathbf{z} = \mathrm{softmax}(K\mathbf{q}/\sqrt{d_k})\,V$. Se
diferencia de la versión aditiva solo en cómo se calcula el puntaje.

**Por qué no se usó:** en la Reunión 03 se acordó con el profesor no comparar con
Transformers, y la decisión de v03 fue explicarla aquí sin agregarla al experimento.

---

## 6. La función de pérdida

### 6.1 Definición

$$\mathcal{L}_\alpha = \frac{1}{N}\sum_i w(d_i)\, d_i^2, \qquad d = \hat{y} - y, \qquad
w(d) = \begin{cases} \alpha & d > 0 \\ 1 & d \le 0 \end{cases}$$

Se llama **pérdida cuadrática asimétrica** (Efron, 1991) y es la que minimiza la **regresión
por expectiles** (Newey y Powell, 1987). En el código, `AsymmetricMSELoss` construye el peso
como `1 + (alpha - 1) * (d > 0)`, que vale α donde se sobreestima y 1 donde no.

### 6.2 Qué estima el modelo: la condición de equilibrio

Para una predicción $\hat{y}$ fija, el valor esperado de la pérdida es
$\mathbb{E}[w(\hat{y}-Y)(\hat{y}-Y)^2]$. Derivando respecto de $\hat{y}$ e igualando a
cero:

$$\alpha\, \mathbb{E}\big[\max(\hat{y} - Y, 0)\big] = \mathbb{E}\big[\max(Y - \hat{y}, 0)\big]$$

La sobreestimación media, multiplicada por α, debe igualar la subestimación media.

- Con α = 1, ambas son iguales: es la condición que cumple la **media**.
- Con α > 1, el modelo debe subestimar en promedio α veces más de lo que sobreestima: la
  predicción queda **por debajo de la media**.

Ese valor es el **expectil de nivel** $\tau = 1/(1+\alpha)$. La sección 2.4 del notebook
comprueba esta condición sobre los datos de validación: el cociente medido fue 1,05; 1,75
y 3,53 para α = 1, 2 y 4.

### 6.3 Por qué el early stopping usa la misma pérdida

Si se detuviera por RMSE, se elegiría la época con menos sesgo conservador, que es justo lo
contrario de lo que busca la pérdida. Por eso el criterio de detención es
$\sqrt{\mathcal{L}_\alpha}$ en validación.

---

## 7. Entrenamiento

- **Optimizador Adam** con tasa de aprendizaje 0,001 y regularización L2 (*weight decay*)
  de 0,0001.
- **Una época** es una pasada por todos los lotes de entrenamiento. En cada lote: se calcula
  la predicción, la pérdida, los gradientes (`loss.backward()`) y se actualizan los pesos
  (`optimizer.step()`).
- **Early stopping:** después de cada época se mide el criterio en validación. Se guarda una
  copia de los pesos cuando mejora, y el entrenamiento se detiene tras 5 épocas sin mejorar,
  con un máximo de 30. El modelo final usa los pesos de la **mejor** época, no de la última.
- **Semillas:** fijan la inicialización de los pesos, el orden de los lotes y el dropout. Con
  la misma semilla, el resultado es idéntico; con 5 semillas se mide cuánto varía el
  resultado por azar.

---

## 8. Métricas

| Métrica | Fórmula | Lectura |
|---|---|---|
| RMSE | $\sqrt{\frac{1}{N}\sum (\hat{y} - y)^2}$ | Error típico, en ciclos; castiga más los errores grandes |
| MAE | $\frac{1}{N}\sum \lvert \hat{y} - y \rvert$ | Error absoluto medio, en ciclos |
| Sesgo | $\frac{1}{N}\sum (\hat{y} - y)$ | Positivo: el modelo sobreestima en promedio |
| Score NASA | $\frac{1}{N}\sum s_i$, con $s = e^{-d/13} - 1$ si $d < 0$ y $e^{d/10} - 1$ si $d \ge 0$ | Penaliza más la sobreestimación (10 < 13) y crece exponencialmente |
| RMSE final | RMSE con las ventanas de RUL ≤ 20 | **Métrica principal:** precisión cerca del fallo |

---

## 9. Protocolo experimental en cinco frases

1. **Partición por motor:** entrenamiento con 2, 5, 10 y 16; validación con 18 y 20; prueba
   con 11, 14 y 15, la partición oficial.
2. **Sin fuga de información:** la normalización se aprende solo con entrenamiento.
3. **Criterios fijados antes de ver resultados**, en `results/seleccion.json`.
4. **La prueba se evalúa una sola vez**, con todas las decisiones ya tomadas.
5. **Resultados con mediana y rango** sobre semillas, sin media ± desviación estándar.

---

## 10. Preguntas para comprobar la comprensión

1. ¿Qué forma tiene el tensor que sale de la primera convolución de A6, y por qué el tiempo
   sigue siendo 30? (Sección 2.1)
2. ¿Cuántos parámetros tiene la segunda convolución de A6 y de dónde sale el número?
   (Sección 2)
3. ¿Qué es exactamente el promedio temporal y en qué se diferencia de la atención? (Secciones
   4.2 y 4.3)
4. ¿Por qué la atención de A6 se comporta casi como un promedio? (Sección 4.4)
5. ¿Qué cambia entre la atención de este trabajo y la de Bahdanau original? (Sección 4.3)
6. ¿Para qué se divide por $\sqrt{d_k}$ en el Transformer? (Sección 5.2)
7. ¿Qué estima un modelo entrenado con α = 2 y cómo se comprobó en los datos? (Sección 6.2)
8. ¿Por qué las arquitecturas con LSTM nunca predijeron valores absurdos? (Sección 3)
9. ¿Por qué el early stopping no usa el RMSE cuando α > 1? (Sección 6.3)
10. ¿Qué parte de los datos se usó para decidir y cuál solo para informar? (Sección 9)

---

## Referencias

- Bahdanau, D., Cho, K. y Bengio, Y. (2015). Neural Machine Translation by Jointly Learning
  to Align and Translate. *ICLR 2015*. arXiv:1409.0473.
- Efron, B. (1991). Regression percentiles using asymmetric squared error loss. *Statistica
  Sinica*, 1(1), 93–125.
- Newey, W. K. y Powell, J. L. (1987). Asymmetric Least Squares Estimation and Testing.
  *Econometrica*, 55(4), 819–847. https://doi.org/10.2307/1911031
- Saxena, A., Goebel, K., Simon, D. y Eklund, N. (2008). Damage propagation modeling for
  aircraft engine run-to-failure simulation. *PHM 2008*.
  https://doi.org/10.1109/PHM.2008.4711414
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. y
  Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. arXiv:1706.03762.
