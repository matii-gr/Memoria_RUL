# Guía de estudio — ProjectV05

Recorrido por el código de `RULv03.ipynb` a `RULv05.ipynb` y por la teoría que lo sustenta,
pensado para poder explicar cada parte en la defensa. Las secciones siguen el camino de un
dato: desde el archivo HDF5 hasta el número que predice la red y cómo se evalúa.

**Novedades de v05:** la sección 2.3 (orden real de las capas y un solo dropout por
arquitectura), la explicación corregida del efecto del pooling (sección 2.4) y la sección 8
ampliada (el sesgo y su descomposición, la relación entre RMSE, sesgo y varianza, y el score
NASA). Las novedades de v04 respecto de v03 fueron la sección 2.4 (pooling), la
actualización de la sección 4.4 y la sección 9 (agrupamiento de regímenes de operación).
Las referencias a secciones del notebook sin otra indicación son de `RULv03.ipynb`; las que
dicen "v04" o "v05" son de `RULv04.ipynb` o `RULv05.ipynb`.

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

**Orden real de las capas (v05, sección 3.4, comprobado recorriendo la red).** En el bloque
convolucional el orden es convolución → BatchNorm → ReLU → (MaxPool, solo en A6-P1 y A6-P2) →
Dropout. El dropout va **después** del pooling.

**Un solo dropout por arquitectura.** Todas las arquitecturas tienen exactamente un dropout:
tras la primera convolución en las que tienen CNN y tras la LSTM en A2 y A4. Viene del diseño
de v01, cuya red era convolución → dropout → LSTM. Al crear A1 y A6 en v02, la LSTM se
reemplazó por una segunda convolución sin dropout. Que no haya dropout tras la segunda
convolución no fue una decisión ajustada con experimentos.

### 2.4 Pooling (v04, etapa 1)

Una capa de **max pooling** de tamaño 2 recorre el eje temporal en tramos de 2 pasos, sin
solaparse, y deja el mayor de los dos valores de cada canal:

$$y_{c,t} = \max\left(x_{c,\,2t-1},\; x_{c,\,2t}\right), \qquad t = 1, \dots, \lfloor L/2 \rfloor$$

- **Forma:** la longitud temporal se divide por 2, redondeando hacia abajo; los canales no
  cambian. `(B, 32, 30)` pasa a `(B, 32, 15)`, y `(B, 64, 15)` pasa a `(B, 64, 7)`: el
  paso 15 queda fuera, porque no completa un tramo de 2.
- **Parámetros:** ninguno. Por eso A6, A6-P1 y A6-P2 tienen los mismos 14.497 parámetros, y
  con la misma semilla parten de los mismos pesos (v04, sección 0.8).
- **Para qué se usa en imágenes:** reduce el tamaño de los mapas de características, y con
  ello el costo de las capas siguientes, y da tolerancia a desplazamientos pequeños de la
  entrada (Goodfellow, Bengio y Courville, 2016, sección 9.3).
- **Dónde se puso en v04:** después de la ReLU y antes del dropout. A6-P1 lo aplica solo
  tras la primera convolución, `(B, 15, 64)` antes de la atención; A6-P2, tras cada
  convolución, `(B, 7, 64)`.

**Por qué A6 no lo tenía.** A1 y A6 siguen a Li, Ding y Sun (2018), que usan ventanas de 30
pasos, como aquí, y desaconsejan el pooling de forma explícita: reduce la dimensión a costa
de filtrar información útil, lo que conviene en imágenes pero no cuando la entrada es de
baja dimensión (sección 2.1 de su artículo).

**Qué mostró el experimento (v04, etapa 1).** Con la pérdida de la configuración final
(α = 2), el pooling empeora el RMSE final en validación: de 1,83 a 2,03 en ambas variantes,
que mejoran a A6 en solo 2 de 5 semillas. Con MSE ocurre lo contrario: A6-P2 mejora (2,00 a
1,71). El tiempo por época no cambia (unos 3 s).

**Por qué depende de la pérdida (corregido en v05, etapa 3).** v04 lo explicó con el sesgo
global: A6-P2 "baja las predicciones" (sesgo global −1,32 con MSE y −1,83 con α = 2). Esa
explicación exageraba el efecto y se sostiene solo en parte:

- En la **zona final**, A6-P2 sí baja las predicciones, pero menos: el sesgo final pasa de
  +0,70 a −0,03 con MSE y de −0,04 a −0,62 con α = 2. A6-P1 no las baja con MSE (+0,74).
- Al separar el cambio de RMSE² final en sesgo² y varianza (sección 8.3), la **varianza pesa
  igual o más**: con MSE, A6-P2 mejora −0,43 ciclos² por varianza y −0,12 por sesgo²
  (medianas); con α = 2, la varianza es el término que más cambia en 9 de 10 pares de
  semillas.

Con 5 semillas no se puede atribuir el efecto del pooling a un solo mecanismo: desplaza las
predicciones de la zona final hacia abajo y cambia la dispersión del error de forma irregular
entre semillas. La decisión de no adoptarlo no cambia, porque se tomó con el RMSE final.

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
- **Con pooling (v04) la conclusión no cambia.** En A6-P1 y A6-P2 los pesos también son
  planos en el interior de la ventana, y el pico se mueve junto con el borde: en A6-P1
  aparece en el segundo y el penúltimo de sus 15 pasos, y en A6-P2 en el primero de sus 7.
  Es coherente con un efecto de borde del relleno, aunque no se comprobó directamente.
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

## 8. Métricas (ampliada en v05)

Todas parten del **error con signo** de cada ventana, $d = \hat{y} - y$ en ciclos: positivo
si el modelo **sobreestima** (anuncia más vida de la real, el caso peligroso) y negativo si
**subestima**.

### 8.1 Tabla de métricas y sus rangos

| Métrica | Fórmula | Rango | Lectura | Final, zona final (v05) |
|---|---|---|---|---|
| RMSE | $\sqrt{\frac{1}{N}\sum d_i^2}$ | 0 a ∞ ciclos | Error típico; castiga más los errores grandes | 1,83 |
| MAE | $\frac{1}{N}\sum \lvert d_i \rvert$ | 0 a ∞ ciclos | Error absoluto medio | 1,45 |
| Sesgo | $\frac{1}{N}\sum d_i$ | −∞ a ∞ ciclos | Dirección del error; los errores de signo opuesto se cancelan | −0,04 |
| Desv. error | $\sqrt{\frac{1}{N}\sum (d_i - \text{Sesgo})^2}$ | 0 a ∞ ciclos | Cuánto varía el error de una ventana a otra | 1,82 |
| Sobreest. media | $\frac{1}{N}\sum \max(d_i, 0)$ | 0 a ∞ ciclos | Parte del sesgo que aporta la sobreestimación | 0,77 |
| Subest. media | $\frac{1}{N}\sum \max(-d_i, 0)$ | 0 a ∞ ciclos | Parte del sesgo que aporta la subestimación | 0,78 |
| % sobreest. | $100 \cdot \frac{1}{N}\#\{d_i > 0\}$ | **0 a 100 %** | Fracción de ventanas sobreestimadas | 48 % |
| Score NASA | $\frac{1}{N}\sum s(d_i)$ | 0 a ∞, sin unidad | Castigo exponencial, mayor al sobreestimar | 0,14 |

Valores de la configuración final (A6, α = 2) en validación, mediana sobre 5 semillas (v05,
sección 2.5). **Sin sufijo**, una métrica se calcula sobre todas las ventanas; **"final"**
indica la zona final (RUL ≤ 20), que es la métrica principal. Solo el porcentaje de
sobreestimación va de 0 a 100; el score NASA no tiene máximo. El "67 sobre 100" de la
Reunión 06 es, muy probablemente, el porcentaje de ventanas sobreestimadas en la zona final
de la configuración final en la prueba oficial de v03 (67,4 %).

### 8.2 El sesgo y su descomposición (v05, etapa 1)

Cada error se escribe como una parte positiva menos una negativa,
$d = \max(d, 0) - \max(-d, 0)$. Promediando:

$$\text{Sesgo} = \text{Sobreest. media} - \text{Subest. media}, \qquad
\text{MAE} = \text{Sobreest. media} + \text{Subest. media}$$

Las dos medias se promedian sobre **todas** las ventanas; las que no aportan cuentan como 0.
El error *cuando ocurre* es la media dividida por la fracción de ventanas en que ocurre.

**Ejemplo.** Con errores +4, −4, +1 y −1: sesgo 0, MAE 2,5 y RMSE 2,92. Un sesgo de 0 no
significa acertar: significa sobreestimar tanto como se subestima.

**Configuración final, zona final:** sobreestima en el 48 % de las ventanas y subestima en el
resto. Las dos partes son casi iguales (0,77 y 0,78 ciclos; 1,36 y 1,43 cuando ocurren), así
que el sesgo final es −0,04.

**El sesgo global no representa la zona final.** Es el promedio de los sesgos de las tres
zonas, ponderado por su número de ventanas:

$$\text{Sesgo global} = f_\text{final}\,\text{Sesgo}_\text{final} + f_\text{interm.}\,\text{Sesgo}_\text{interm.} + f_\text{temprana}\,\text{Sesgo}_\text{temprana}$$

con $f$ = 0,31, 0,29 y 0,39 en validación. En la configuración final, el sesgo global
(−0,93) viene casi todo de la zona temprana (aporte −1,11).

**Por qué la zona temprana se subestima: el recorte en 62.** Ninguna etiqueta supera 62
ciclos. Una ventana con etiqueta 62 se parece a otras con 55 a 61 ciclos, porque la
degradación aún es casi invisible, y el promedio que aprende el modelo para ellas solo puede
quedar por debajo de 62. Las ventanas con etiqueta 62 se subestiman 6,2 ciclos con la
configuración final. Cerca de RUL 0 ocurre lo contrario: no hay etiquetas negativas, y el
error tiende a ser positivo.

**Por motor.** El sesgo final de −0,04 es el promedio del motor 18 (−1,04) y el 20 (+1,07):
un sesgo cercano a 0 con dos motores puede esconder errores sistemáticos de cada uno.

**Relación con la pérdida (sección 6.2).** La condición de equilibrio de la pérdida
asimétrica es $\alpha \cdot \text{Sobreest. media} = \text{Subest. media}$, que es la misma
descomposición. Se cumple sobre el conjunto completo: con α = 2, el cociente
Subest./Sobreest. de A6 es 1,75 en todas las ventanas de validación (mediana sobre semillas). No se cumple zona por
zona: en la zona final el cociente es 1,07. La pérdida impone la condición en promedio sobre
todos los datos, no en cada zona.

### 8.3 RMSE, sesgo y varianza

$$\text{RMSE}^2 = \text{Sesgo}^2 + \text{Desv. error}^2$$

El sesgo² es la parte **sistemática** del error (lo que se corregiría restando una
constante); la varianza, la que cambia de ventana a ventana. En la zona final, el sesgo
explica el 0,2 % del RMSE² de la configuración final y el 35 % del de la base A3. La mejora de
la base a la final (RMSE 2,60 → 1,83) viene de eliminar la sobreestimación sistemática (sesgo
+1,61 → −0,04) y de reducir la variación del error (2,21 → 1,82).

Esta identidad permite separar el cambio de RMSE² entre dos modelos en un cambio de sesgo² y
uno de varianza. Así se revisó en v05 la explicación del pooling (sección 2.4).

### 8.4 El score NASA (v05, etapa 2)

$$s(d) = \begin{cases} e^{-d/13} - 1 & d < 0 \\ e^{d/10} - 1 & d \ge 0 \end{cases}$$

**Por qué es pequeño (0,14 con un RMSE de 1,83).**

1. Con errores pequeños la exponencial es casi una recta: $s(d) \approx d/10$ al
   sobreestimar y $|d|/13$ al subestimar. Un error de 1 ciclo aporta 0,105 o 0,080; uno de 2
   ciclos, 0,221 o 0,166. El 90 % de los errores de la zona final está entre −2,9 y +3,2
   ciclos.
2. Aquí se **promedia** sobre las ventanas. Un score de 0,14 equivale a sobreestimar 1,3
   ciclos en todas las ventanas ($10\ln 1{,}14$) o a subestimar 1,7 ($13\ln 1{,}14$).

**Promedio frente a suma.** En la bibliografía el score se **suma**: Sateesh Babu et al.
(2016, ec. 1) y Li, Ding y Sun (2018, ec. 5) suman sobre los motores de prueba de C-MAPSS
(273,7 en FD001), y Li, Wang y Li (2022), en DS02, suman sobre todas las predicciones de
prueba (5,78 × 10⁵). Con la suma, las predicciones de este trabajo darían unos 700 en la zona
final. Los valores no son comparables con los publicados.

**MAE asimétrico.** En este rango,
$\text{Score} \approx \text{Sobreest. media}/10 + \text{Subest. media}/13$ (entre un 8 % y un
17 % por debajo del valor real).

**Por qué no siempre sigue al RMSE.** Sobre 60 corridas, el score ordena casi igual que el
RMSE (Spearman 0,95). Los desacuerdos se deben a la **asimetría**: el score ordena casi igual
que el MAE asimétrico (0,99) y menos parecido al MAE simétrico (0,91). Una configuración que
cambia sobreestimación por subestimación, como A6-P2 con α = 2, empeora en RMSE (1,83 → 2,03)
y casi nada en score (0,14 → 0,15). Sateesh Babu et al. (2016, p. 5) advierten que el score
favorece a los modelos que subestiman y recomiendan reportarlo junto con el RMSE.

---

## 9. Agrupamiento de regímenes de operación (v03, etapa 4; v04, etapa 3)

### 9.1 Qué se agrupa

Cada registro se describe por sus condiciones de vuelo W: altitud, Mach, TRA (posición del
acelerador) y T2 (temperatura a la entrada del ventilador). Cada variable se estandariza
con los motores de entrenamiento, y se agrupa una submuestra de 8.000 registros. En v04
se agrega un segundo espacio: las 18 variables que recibe el modelo (W + X_s).

### 9.2 Los tres métodos

- **k-means:** elige K centros y asigna cada punto al más cercano; mueve los centros hasta
  minimizar la suma de distancias al cuadrado. Forma grupos compactos y redondeados.
- **Spectral Clustering:** conecta cada punto con sus 15 vecinos más cercanos y corta el
  grafo donde las conexiones son débiles. Admite grupos alargados.
- **DBSCAN:** un grupo es una zona densa: puntos con al menos `min_samples` vecinos a una
  distancia menor que `eps`. No se fija K, y los puntos aislados quedan como ruido.

### 9.3 Cómo se evalúa un agrupamiento

- **Silueta** (Rousseeuw, 1987). Para cada punto $i$, $a(i)$ es su distancia media a los
  puntos de su grupo y $b(i)$ su distancia media a los del grupo vecino más cercano:

  $$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}$$

  Vale cerca de 1 si el punto está mucho más cerca de su grupo que del vecino, cerca de 0 si
  está en la frontera y negativo si estaría mejor en otro grupo. La silueta de un
  agrupamiento es el promedio sobre los puntos. Escala orientativa de Kaufman y Rousseeuw
  (1990): sobre 0,70, estructura fuerte; de 0,51 a 0,70, razonable; de 0,26 a 0,50, débil;
  bajo 0,25, sin estructura sustancial.
- **Davies-Bouldin** (Davies y Bouldin, 1979): compara el tamaño de cada grupo con la
  distancia a su grupo más parecido. Menor es mejor.
- **Índice de Rand ajustado** (Hubert y Arabie, 1985): mide cuánto coinciden dos
  particiones de los mismos puntos, corregido para que el azar dé 0. Vale 1 si coinciden
  exactamente. Se usa para comparar los grupos con las fases de vuelo y con los motores.
- **η² (eta cuadrado):** fracción de la varianza de una variable, aquí el RUL, que explican
  los grupos: la varianza entre las medias de los grupos dividida por la varianza total.

### 9.4 Por qué el agrupamiento en W queda dominado por la altitud

Altitud, Mach y T2 están muy correlacionadas (|r| entre 0,84 y 0,98): T2 depende de la
temperatura del aire, que baja con la altitud, y del Mach. El **análisis de componentes
principales** (PCA) lo confirma. El PCA busca las direcciones en que los datos varían más:
la primera explica el 79 % de la varianza, la segunda (casi solo TRA) el 17 %, y la cuarta,
0,0 %. Como la información de la altitud entra tres veces en la distancia euclidiana, k-means
con K = 2 corta los datos por altitud. Esa es la razón de que la figura de v03 pareciera
hecha con dos variables.

Las fases de vuelo, en cambio, se distinguen por **TRA**: ascenso con el acelerador casi al
máximo (80 % en promedio), descenso con el acelerador reducido (48 %) y nivelado entre ambos
(71 %). Por eso un agrupamiento dominado por la altitud no recupera las fases.

### 9.5 Por qué más grupos no mejoran la silueta

Los datos de operación forman un **continuo**: cada vuelo pasa de forma gradual por todas
las altitudes y velocidades. Al cortar un continuo, los puntos a cada lado del corte quedan
cerca del grupo vecino ($a(i) \approx b(i)$, silueta cerca de 0). Con más grupos hay más
cortes, y la silueta baja. En v04, con K de 2 a 20, ningún método supera su valor con K = 2
o 3. Spectral baja más que k-means porque sus grupos pueden ser alargados, y la silueta, que
mide compacidad, los evalúa peor.

### 9.6 Agrupar con los sensores (18 dimensiones)

Muchos sensores (flujo de combustible, velocidades de los ejes, temperaturas internas)
responden a la posición del acelerador, de modo que en 18 dimensiones TRA pesa más. k-means
con K = 3 se acerca más a las fases (Rand ajustado 0,32 frente a 0,11 en W): uno de sus
grupos es en un 90 % descenso. Los grupos no separan motores ni estados de salud (η² del RUL
0,002), porque en los sensores la variación por condición de vuelo es mucho mayor que la
variación por degradación.

---

## 10. Protocolo experimental en cinco frases

1. **Partición por motor:** entrenamiento con 2, 5, 10 y 16; validación con 18 y 20; prueba
   con 11, 14 y 15, la partición oficial.
2. **Sin fuga de información:** la normalización se aprende solo con entrenamiento.
3. **Criterios fijados antes de ver resultados**, en `results/seleccion.json`.
4. **La prueba se evalúa una sola vez**, con todas las decisiones ya tomadas.
5. **Resultados con mediana y rango** sobre semillas, sin media ± desviación estándar.

---

## 11. Preguntas para comprobar la comprensión

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
10. ¿Qué parte de los datos se usó para decidir y cuál solo para informar? (Sección 10)
11. ¿Qué forma tiene el tensor que llega a la atención en A6-P2, y por qué es 7 y no 7,5?
    (Sección 2.4)
12. ¿Por qué el pooling mejora con MSE pero empeora con α = 2, y qué parte de la explicación
    de v04 se corrigió en v05? (Sección 2.4)
13. ¿Qué mide la silueta y por qué baja al aumentar el número de grupos? (Secciones 9.3 y
    9.5)
14. ¿Por qué el clustering de v03 parecía hecho con solo dos variables? (Sección 9.4)
15. ¿En qué orden van la convolución, el max pooling y el dropout, y por qué A6 tiene un solo
    dropout? (Sección 2.3)
16. ¿Por qué un sesgo de 0 no significa que el modelo acierte, y cómo se calcula cuánto se
    sobreestima y cuánto se subestima? (Sección 8.2)
17. ¿Por qué el sesgo global de la configuración final es negativo si su sesgo en la zona
    final es casi 0? (Sección 8.2)
18. ¿Qué parte del RMSE² final de la configuración final es sesgo? (Sección 8.3)
19. ¿Por qué el score NASA vale 0,14 si el RMSE final es 1,83, y por qué los valores
    publicados son de cientos o miles? (Sección 8.4)
20. ¿Por qué el score trata mejor que el RMSE a A6-P2 con α = 2? (Sección 8.4)

---

## Referencias

- Bahdanau, D., Cho, K. y Bengio, Y. (2015). Neural Machine Translation by Jointly Learning
  to Align and Translate. *ICLR 2015*. arXiv:1409.0473.
- Davies, D. L. y Bouldin, D. W. (1979). A Cluster Separation Measure. *IEEE Transactions
  on Pattern Analysis and Machine Intelligence*, 1(2), 224–227.
- Efron, B. (1991). Regression percentiles using asymmetric squared error loss. *Statistica
  Sinica*, 1(1), 93–125.
- Goodfellow, I., Bengio, Y. y Courville, A. (2016). *Deep Learning*. MIT Press, cap. 9.
- Hubert, L. y Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2,
  193–218.
- Kaufman, L. y Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to
  Cluster Analysis*. Wiley.
- Li, H., Wang, Z. y Li, Z. (2022). An enhanced CNN-LSTM remaining useful life prediction
  model for aircraft engine with attention mechanism. *PeerJ Computer Science*, 8, e1084.
  https://doi.org/10.7717/peerj-cs.1084
- Li, X., Ding, Q. y Sun, J.-Q. (2018). Remaining useful life estimation in prognostics
  using deep convolution neural networks. *Reliability Engineering & System Safety*, 172,
  1–11.
- Newey, W. K. y Powell, J. L. (1987). Asymmetric Least Squares Estimation and Testing.
  *Econometrica*, 55(4), 819–847. https://doi.org/10.2307/1911031
- Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and
  validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20,
  53–65.
- Sateesh Babu, G., Zhao, P. y Li, X.-L. (2016). Deep convolutional neural network based
  regression approach for estimation of remaining useful life. *DASFAA 2016*, LNCS 9642,
  214–228.
- Saxena, A., Goebel, K., Simon, D. y Eklund, N. (2008). Damage propagation modeling for
  aircraft engine run-to-failure simulation. *PHM 2008*.
  https://doi.org/10.1109/PHM.2008.4711414
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. y
  Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. arXiv:1706.03762.
