# Preparación de la defensa — ProjectV05

Preguntas probables y sus respuestas, con las cifras de `RULv03.ipynb`, `RULv04.ipynb` y
`RULv05.ipynb`. Lo nuevo de v05 está en la sección 2 (orden de las capas, dropout y la
explicación corregida del pooling) y en la sección 5 (sesgo y score NASA). Lo de v04 está
en la sección 2 (pooling) y en la sección 6 (regímenes de operación). Están redactadas
para decirse en voz alta: primero la respuesta directa, después el respaldo. Para la teoría
de fondo, ver `Guia_Estudio.md`; para el uso de IA, `Metodologia_IA.md`.

**Resultado en una frase:** una CNN con atención, entrenada con una pérdida cuadrática
asimétrica que estima el expectil 0,33 del RUL, se equivoca en promedio 1,78 ciclos en los
últimos 20 ciclos de vida de motores nunca vistos, frente a 2,97 de la arquitectura base; la
ventaja se confirma en la prueba oficial, pero depende de qué tan rápido se degrada cada
motor.

---

## 1. Datos y protocolo

### ¿Qué datos usa el trabajo?

El subconjunto DS02 de N-CMAPSS: 9 motores turbofán simulados hasta la falla, con datos a
1 Hz. Seis motores forman la partición de desarrollo (2, 5, 10, 16, 18, 20) y tres la de
prueba oficial (11, 14, 15). La entrada son 18 señales: 4 condiciones de vuelo (altitud,
Mach, ángulo del acelerador y temperatura de entrada) y 14 sensores físicos.

### ¿Por qué no usa los sensores virtuales ni los parámetros de salud?

Porque no se pueden medir en un motor real. Son salidas internas del simulador; usarlos
haría el problema artificialmente fácil y el resultado no sería transferible.

### ¿Por qué la partición es por motor y no aleatoria?

Porque en operación el modelo se enfrenta a motores que nunca ha visto. Además, las ventanas
consecutivas se solapan: con una partición aleatoria, ventanas casi idénticas quedan en
entrenamiento y en evaluación. La primera versión del pipeline partía por índice y
reportaba una mejora del 26 % al agregar atención; con partición por motor esa mejora bajó a
cerca del 3 %, dentro del ruido entre semillas.

### ¿Para qué sirve cada conjunto?

- **Entrenamiento (55,3 %, motores 2, 5, 10, 16):** ajusta los pesos.
- **Validación (25,5 %, motores 18 y 20):** decide la época de detención y todas las
  elecciones del estudio.
- **Prueba (19,2 %, motores 11, 14 y 15):** se evaluó una sola vez, con todas las decisiones
  ya tomadas.

### ¿Qué es el etiquetado lineal por tramos y por qué el umbral es 62?

Mientras el motor está sano, los sensores no indican cuánto falta para la falla, así que el
RUL se recorta a un máximo. Ese máximo es la mediana del RUL en el instante en que los
motores de entrenamiento pasan de sanos a degradados, según el indicador de salud del
dataset (58, 72, 65 y 47 ciclos). El valor 125 habitual viene del C-MAPSS de 2008 y en DS02
no tendría efecto, porque el RUL máximo es 88.

### ¿Por qué la métrica principal es el error en los últimos 20 ciclos?

Porque es donde la estimación tiene consecuencias: sobreestimar cuando quedan 10 ciclos
puede significar una falla en vuelo. Se acordó en la Reunión 03.

### ¿Por qué reporta mediana y rango, y no media con desviación estándar?

Con 5 semillas, la media y la desviación dependen demasiado de un valor atípico. Se muestran
los valores individuales, la mediana y el mínimo y máximo, sin pruebas de significancia.

### ¿Los resultados son reproducibles?

Sí, con diferencia exactamente cero. El código de v03 se reescribió y comentó; para
comprobar que no cambió nada, se reentrenaron desde cero tres corridas y coincidieron en
cada época y en cada predicción con las de v02. Por eso v03 reutiliza las corridas de v02.

---

## 2. Arquitecturas y atención

### ¿Qué significa "32 → 64" en la convolución? ¿Aumenta una dimensión espacial?

No. Son **canales (filtros)**: la segunda convolución recibe 32 características por paso y
produce 64. La dimensión temporal se mantiene en 30: con kernel 3, padding 1 y paso 1, la
longitud de salida es (30 + 2 − 3) / 1 + 1 = 30. En ninguna capa de la red cambia la
longitud de la ventana. Aumentar el número de filtros en capas sucesivas es habitual en las
CNN.

### ¿Qué es exactamente el promedio temporal?

Después del extractor hay 30 vectores de 64 características, uno por paso de tiempo. El
promedio temporal los resume en un solo vector de 64, promediando cada característica sobre
los 30 pasos, con el mismo peso (1/30) para cada uno. En el código es `h.mean(dim=1)`.

### ¿Y en qué se diferencia de la atención?

La atención también es un promedio de los 30 vectores, pero con **pesos aprendidos** que
dependen del contenido de cada ventana: un puntaje por paso,
$e_t = v^\top\tanh(W h_t + b)$, convertido en pesos que suman 1 con softmax. Si todos los
puntajes fueran iguales, la atención sería exactamente el promedio temporal.

### ¿Es la atención de Bahdanau?

Es su forma aditiva, simplificada. En el original (Bahdanau, Cho y Bengio, 2015), el puntaje
compara cada paso con el estado de un decodificador, que hace de consulta. Aquí no hay
decodificador: la consulta la reemplazan parámetros aprendidos, iguales para todas las
ventanas. Es una atención de agrupamiento, cuya función es resumir la secuencia.

### ¿En qué se diferencia de la atención de *Attention Is All You Need*?

En tres cosas:

- **El puntaje:** allá es el producto punto entre una consulta y una clave, dividido por
  $\sqrt{d_k}$; aquí, una red pequeña con tangente hiperbólica.
- **La consulta:** allá cada paso tiene la suya, calculada desde la entrada, y atiende a
  todos los demás (autoatención); aquí hay una sola, fija.
- **El resultado:** allá la salida es una secuencia del mismo largo, transformada; aquí, un
  solo vector por ventana.

### ¿Por qué se divide por $\sqrt{d_k}$ en el Transformer?

Porque el producto punto entre vectores de dimensión $d_k$ tiene una varianza que crece con
$d_k$. Con valores grandes, la softmax se satura (un peso cercano a 1 y el resto cercanos a
0) y sus gradientes se vuelven muy pequeños. Dividir por $\sqrt{d_k}$ devuelve la varianza a
1.

### ¿Por qué quedó elegida A6 y no A5, que era la alternativa seleccionada?

Por el criterio fijado antes de ver los resultados: entre las arquitecturas con atención, la
de menor mediana de error en la zona final. A6 obtuvo 2,00 ciclos y A5, 2,18. Hay que decir
también lo que el resultado no demuestra: cinco de las seis arquitecturas caen dentro de
0,18 ciclos, con rangos solapados, y A6 supera a una CNN simple por solo 0,01.

### ¿La atención sirve?

Depende de sobre qué se aplica. Sobre CNN + LSTM mejora (2,60 → 2,18); sobre la CNN no
cambia la zona final (2,02 → 2,00), aunque mejora el error global y elimina el sesgo; sobre
la LSTM empeora levemente (2,07 → 2,17). No hay un beneficio uniforme.

### ¿Por qué los pesos de atención de A6 son casi uniformes?

Una ventana cubre 5 minutos de un vuelo, y la degradación avanza de un vuelo a otro, no
dentro de 5 minutos. Dentro de una ventana, el estado de salud es prácticamente constante,
así que ningún paso es más informativo que otro y el promedio uniforme es casi óptimo. Los
picos que aparecen en los pasos 2 y 29 coinciden con los bordes donde las convoluciones
procesan el relleno de ceros. Es una explicación coherente con lo observado, no demostrada.

### ¿Por qué la correlación temporal dificulta la atención, como dijo el profesor?

Porque la salida del LSTM en cada paso resume toda la historia anterior, así que salidas
consecutivas son muy parecidas y la atención no tiene vectores distintos entre los que
elegir. Aplicar la atención directamente sobre las características de la convolución, que
son más locales, evita ese problema; es el diseño de A6.

### ¿Por qué A6 converge más lento?

Lo muestran las curvas en el eje común de 30 épocas: A6 conserva su mejor época en la 16
(mediana), frente a la 7 a 9 del resto, y una de sus semillas seguía mejorando cuando se
alcanzó el límite. Lo que las curvas no permiten afirmar es la causa: eso requeriría medir
los gradientes durante el entrenamiento, y no se midieron.

### ¿La ablación es estricta?

No del todo, y conviene decirlo antes de que lo pregunten. Los pares con y sin atención
(A1–A6, A2–A4, A3–A5) sí difieren en un solo factor. Las comparaciones entre extractores no:
A1 y A6 tienen una convolución más que A3 y A5, y la agregación sin atención es el promedio en
A1 y el último estado en A2 y A3. Las conclusiones sobre la atención se sostienen; las
comparaciones entre extractores, no del todo.

### ¿Por qué no se comparó con Transformers?

Se acordó con el profesor guía en la Reunión 03, para concentrar el trabajo en la ablación
de bloques.

### ¿En qué orden van la convolución, el max pooling y el dropout? (v05)

Convolución → BatchNorm → ReLU → max pooling → dropout. El dropout va **después** del
pooling, que es la secuencia tradicional. Se comprobó recorriendo la red capa por capa
(v05, sección 3.4); en la Reunión 06 se dijo lo contrario por error.

### ¿El max pooling deja 15 canales? (v05)

No: deja 15 **pasos de tiempo**. El pooling divide el tiempo por 2 (30 → 15, y 15 → 7 en
A6-P2) y no cambia los canales, que siguen siendo 32 o 64. La convolución hace lo contrario:
cambia los canales (18 → 32 → 64) y mantiene los pasos. Por eso A6, A6-P1 y A6-P2 tienen los
mismos 14.497 parámetros.

### ¿Por qué hay dropout tras la primera convolución y no tras la segunda? (v05)

Porque cada arquitectura tiene un solo dropout, heredado de v01. La red de v01 era
convolución → dropout → LSTM. Al crear A1 y A6 en v02, la LSTM se reemplazó por una segunda
convolución de 32 a 64 canales, y no se agregó otro dropout. No fue una decisión ajustada con
experimentos, y así lo declaro. Probar un segundo dropout queda como trabajo futuro.

### ¿Por qué la red convolucional no tiene una capa de pooling? (v04)

Porque A1 y A6 siguen a Li, Ding y Sun (2018), que trabajan con ventanas de 30 pasos, como
este trabajo, y desaconsejan el pooling de forma explícita: reduce la dimensión a costa de
filtrar información, lo que conviene en imágenes pero no cuando la entrada es pequeña. No
todos los trabajos lo omiten: Sateesh Babu et al. (2016) usan *average pooling* y Li, Wang y
Li (2022) usan max pooling, pero sobre entradas mucho más grandes (400 × 42).

### ¿Se comprobó con datos? (v04)

Sí. Se probaron dos variantes de A6 con max pooling de tamaño 2: tras la primera
convolución (la atención ve 15 pasos) y tras cada convolución (ve 7). Con la pérdida de la
configuración final, α = 2, ninguna mejora a A6: el RMSE final en validación sube de 1,83
a 2,03, y cada variante es mejor en solo 2 de 5 semillas. La regla fijada antes del
experimento no adoptó el pooling. El pooling tampoco ahorra tiempo: unos 3 segundos por
época con y sin él.

### ¿Por qué con MSE el pooling sí mejora? (v04, respuesta corregida en v05)

Con MSE, A6-P2 obtiene 1,71 frente a 2,00 de A6, mejor en 4 de 5 semillas. Hay dos efectos,
y no se puede atribuir la mejora a uno solo:

- **A6-P2 baja las predicciones en la zona final:** el sesgo final pasa de +0,70 a −0,03.
  Con MSE, A6 sobreestimaba, así que eso ayuda.
- **También cambia cuánto varía el error.** Al separar el RMSE² final en sesgo² y varianza,
  la mejora viene más de la varianza (−0,43 ciclos² en mediana) que del sesgo² (−0,12). Con
  α = 2, la varianza es el término que más cambia en 9 de 10 pares de semillas.

En v04 lo expliqué solo con el sesgo global (−1,32 y −1,83), y el profesor cuestionó esa
métrica con razón: el sesgo global lo domina la zona temprana. La explicación corregida
está en v05, etapa 3. La decisión de no adoptar el pooling no cambia, porque se tomó con el
RMSE final.

### Entonces, ¿por qué no usar A6-P2 con MSE, que tiene la mediana más baja? (v04)

Porque esa comparación no estaba en la regla fijada antes del experimento, y elegir después
de mirar todas las combinaciones es lo que el pre-registro busca evitar. Además, la ventaja
no es consistente: frente a la configuración final, A6-P2 con MSE es mejor en solo 3 de 5
semillas, una de ellas por 0,0007 ciclos. Quedó registrado como observación en
`seleccion.json`.

---

## 3. Función de pérdida

### ¿Cómo se llama el método y de dónde viene?

Es la **pérdida cuadrática asimétrica** (*asymmetric squared error loss*, Efron 1991), la
función que minimiza la **regresión por expectiles** (Newey y Powell, 1987, *Econometrica*).
Es un error cuadrático en que sobreestimar pesa α veces más que subestimar.

### ¿Por qué penalizar más la sobreestimación?

Porque sobreestimar el RUL es declarar más vida útil de la que queda: la intervención llega
tarde. Subestimar solo adelanta una inspección. Esa asimetría ya está en el score NASA
(Saxena et al., 2008).

### ¿Qué estima el modelo?

El expectil de nivel 1/(1+α) del RUL, en vez de su media. Con α = 2 es el expectil 0,33: una
estimación conservadora. Matemáticamente, en el óptimo la subestimación media es α veces la
sobreestimación media.

### ¿Se comprobó eso en los datos?

Sí, en validación. El cociente entre subestimación y sobreestimación medias fue 1,05; 1,75 y
3,53 para α = 1, 2 y 4, frente a los valores teóricos 1, 2 y 4. Queda algo por debajo
porque la condición rige exactamente sobre los datos de entrenamiento, pero α controla de
forma predecible cuánto más conservador es el modelo. La condición se cumple en promedio
sobre todas las ventanas, no zona por zona: en la zona final, con α = 2, el cociente es 1,07
(v05, etapa 1).

### ¿Por qué α = 2?

Es el que minimiza el score NASA en la zona final en validación, el criterio fijado de
antemano. Frente al error cuadrático medio mejora en las 5 semillas: el RMSE final baja de
2,00 a 1,83 y la sobreestimación media un 26 %. Con α = 4 el modelo es demasiado
conservador.

### ¿Es un aporte original?

En la búsqueda realizada no se encontraron trabajos que apliquen la regresión por expectiles
al pronóstico de RUL. Sí hay antecedentes de pérdidas que penalizan la sobreestimación:
Peringal et al. (2025, AIAA SciTech) con una LSTM y Abdullah (2026, preprint en arXiv), que
usa el score NASA como pérdida. La formulación correcta es "no se encontraron antecedentes",
no "es la primera vez", porque la búsqueda no fue exhaustiva.

### ¿Por qué no usar el score NASA como pérdida?

La pérdida elegida tiene una interpretación estadística precisa y un solo parámetro, y
reservar el score NASA como métrica permite usarlo para elegir α sin que la selección dependa
de la misma función que se optimiza. Entrenar con el score NASA no se evaluó.

---

## 4. Validación cruzada y resultado final

### ¿Cuál es el resultado final?

En la prueba oficial, sobre 5 semillas:

| Configuración | RMSE final | % sobreest. final | Sesgo global |
|---|---|---|---|
| Base: CNN + LSTM, error cuadrático medio | 2,97 [2,47 – 3,30] | 86 % | +3,00 |
| A6, error cuadrático medio | 2,16 [1,92 – 2,45] | 82 % | +1,24 |
| Final: A6, pérdida asimétrica α = 2 | 1,78 [1,50 – 1,97] | 67 % | +0,19 |

La peor semilla de la configuración final es mejor que la mejor de la base, y mejoran los tres
motores de prueba, incluidos los dos de clases de vuelo nunca vistas.

**Cuidado con la columna de sesgo:** es el sesgo **global**, sobre todas las ventanas de
prueba, no el de la zona final (v05, etapa 1). Para la zona final, la tabla informa el
porcentaje de ventanas sobreestimadas.

### ¿Qué mostró la validación cruzada?

Que la ventaja no es uniforme: la configuración final es más precisa en solo 2 de los 6
motores (16 y 20), aunque reduce la sobreestimación en los 6.

### ¿Por qué la validación cruzada y la prueba no coinciden?

Por el sesgo del modelo base en cada motor. La pérdida asimétrica desplaza las predicciones
hacia abajo: corrige donde la base era optimista (motores 16 y 20) y perjudica donde ya era
pesimista (5 y 10). En los motores de prueba la base sobreestima (+3,0), y por eso ahí la
ganancia es clara.

### ¿Qué son las predicciones extremas de A6?

En la validación cruzada, A6 produjo 19 ventanas con predicciones fuera del rango válido, con
un máximo de 190 ciclos cuando el RUL no puede superar 62. La causa probable es que no tiene
ninguna capa que acote su salida, a diferencia del LSTM. Recortarlas solo cambiaría un motor.
No se aplicó porque habría sido una decisión tomada después de ver los resultados.

---

## 5. Sesgo y score NASA (v05)

### ¿Qué es el sesgo global, matemáticamente?

El error medio con signo, $\frac{1}{N}\sum_i(\hat{y}_i - y_i)$, sobre todas las ventanas,
en ciclos. Positivo significa que en promedio se sobreestima. Indica la dirección del error,
no su tamaño, porque los errores de signo opuesto se cancelan: con errores +4 y −4 el sesgo
es 0 aunque el modelo se equivoque 4 ciclos en cada ventana.

### ¿Cuánto se sobreestima y cuánto se subestima?

El sesgo es exactamente la sobreestimación media menos la subestimación media. En la zona
final, la configuración final sobreestima en el 48 % de las ventanas y subestima en el
resto: 0,77 y 0,78 ciclos promediados sobre todas las ventanas, o 1,36 y 1,43 ciclos cuando
ocurre cada uno. Por eso su sesgo final es −0,04. La base A3 sobreestima en el 75 % de las
ventanas de la zona final, con sesgo +1,61.

### ¿Por qué el sesgo global es negativo (−0,93) si en la zona final es casi cero?

Porque el sesgo global es un promedio ponderado de las tres zonas, y la zona temprana tiene
el 39 % de las ventanas y aporta −1,11 ciclos. Esa zona se subestima por el recorte en 62:
ninguna etiqueta lo supera, así que el promedio que aprende el modelo en el tope de la
escala solo puede quedar por debajo. Las ventanas con etiqueta 62 se subestiman 6,2 ciclos.

### ¿Un sesgo cercano a cero significa que no hay error sistemático?

No. El −0,04 de la zona final es el promedio del motor 18 (−1,04) y el 20 (+1,07): uno se
subestima y el otro se sobreestima. Con solo dos motores de validación, un sesgo cercano a
cero puede esconder errores sistemáticos de cada motor. Es coherente con lo que mostró la
validación cruzada de v03: los motores de vida corta se sobreestiman.

### ¿Qué parte del error es sesgo?

$\text{RMSE}^2 = \text{Sesgo}^2 + \text{varianza del error}$. En la zona final, el sesgo
explica el 0,2 % del RMSE² de la configuración final y el 35 % del de la base. La mejora de
la base a la final (2,60 → 1,83) viene de quitar la sobreestimación sistemática y de una
menor variación del error.

### ¿Por qué el score NASA es tan bajo (0,14) si el RMSE final es 1,83?

Por dos razones. Primero, con errores pequeños la función es casi una recta: un error de 2
ciclos aporta 0,22 si sobreestima y 0,17 si subestima. Segundo, aquí se **promedia** por
ventana. Un score de 0,14 equivale a sobreestimar 1,3 ciclos o subestimar 1,7 en todas las
ventanas, coherente con el RMSE. El score va de 0 a infinito; no tiene un máximo de 100.

### ¿Por qué los scores publicados son de cientos o miles?

Porque en la bibliografía se **suma**, no se promedia. Lo verifiqué en Sateesh Babu et al.
(2016, ec. 1), Li, Ding y Sun (2018, ec. 5; 273,7 en FD001) y Li, Wang y Li (2022), que en
DS02 reportan 5,78 × 10⁵. Con la suma, mis predicciones darían unos 700 en la zona final.
Los valores no son comparables con los publicados.

### ¿Por qué A6-P2 empeora mucho en RMSE y poco en score?

Por la asimetría del score: subestimar cuesta 10/13 ≈ 0,77 veces lo que cuesta sobreestimar.
Con α = 2, A6-P2 cambia errores por sobreestimación por errores por subestimación, más
grandes. El RMSE empeora (1,83 → 2,03), pero el score casi no (0,14 → 0,15), porque descuenta
la sobreestimación evitada. Sobre 60 corridas, el score ordena casi igual que el RMSE
(Spearman 0,95), y sus desacuerdos se explican por la asimetría: ordena casi igual que un MAE
asimétrico (0,99). Sateesh Babu et al. (2016) advierten que el score favorece a los modelos
que subestiman; por eso se reporta junto con el RMSE.

### ¿Qué era el "67 sobre 100" de la Reunión 06?

Muy probablemente el porcentaje de ventanas sobreestimadas en la zona final de la
configuración final en la prueba oficial: 67,4 %. Es una métrica de 0 a 100. No era el
score, que en esa misma tabla vale 0,14.

### ¿Por qué usaste el sesgo global para explicar el pooling?

Porque era una de las métricas principales desde v02, y la IA lo usó en la discusión sin
calcular el sesgo de la zona final, que era el que correspondía a la afirmación. El argumento
era débil y lo corregí en v05: en la zona final el efecto es menor, y el cambio de RMSE se
explica tanto o más por la varianza que por el sesgo. Quedó registrado como error en
`Metodologia_IA.md`, con una salvaguarda nueva: argumentar con métricas de la misma zona en
que se decide.

---

## 6. Regímenes de operación y motor 20

### ¿Por qué no usaste DBSCAN o Spectral Clustering para los regímenes?

Se probaron en v03, junto con k-means. Ninguno recupera las fases de vuelo (índice de Rand
ajustado ≤ 0,11 en sus configuraciones elegidas). DBSCAN además resulta inadecuado: los datos
forman trayectorias continuas sin zonas densas separadas, y el número de grupos depende mucho
de su parámetro (34, 18 o 9).

### ¿El clustering usaba todas las dimensiones o solo velocidad y altitud? (v04)

Usaba las 4 variables de W: la matriz de entrada es de 8.000 × 4. Si se agrupa con solo
altitud y Mach, el resultado no es igual (5 % de los registros cambia de grupo). Pero la
impresión tiene una causa: altitud, Mach y T2 están muy correlacionadas (|r| de 0,84 a
0,98), porque T2 depende de la altitud y del Mach. En el PCA, la primera componente explica
el 79 % de la varianza y la cuarta, 0,0 %. La información de la altitud entra casi tres
veces en la distancia, y k-means termina cortando por altitud. La figura de v03, que solo
mostraba el plano Mach–altitud, lo hacía ver como un agrupamiento en dos dimensiones.

### ¿Por qué ningún método recupera las fases? (corregido en v04)

Porque lo que separa las fases es **TRA**, la posición del acelerador: alrededor de 80 % en
el ascenso, 71 % en vuelo nivelado y 48 % en el descenso, mientras que las tres fases
recorren las mismas altitudes. En W, TRA es una sola de cuatro variables y la altitud pesa
casi el triple, así que el agrupamiento sigue la altitud y no las fases. En v03 se dijo que
el ascenso y el descenso "se superponen en W"; no es exacto, porque en TRA se distinguen.
Lo correcto es que la distancia está dominada por la altitud.

### ¿Más grupos mejoran la silueta? (v04)

No. Con K de 2 a 20, ningún método supera su valor con K = 2 o 3; Spectral Clustering baja
de 0,45 a entre 0,09 y 0,28. Los datos de operación no forman nubes separadas sino un
continuo: cada vuelo pasa gradualmente por todas las altitudes. Al cortar un continuo, los
puntos de cada lado del corte quedan cerca del grupo vecino, y la silueta castiga eso. Más
grupos significan más cortes.

### ¿Una silueta de 0,45 es buena? (v04)

Es débil. En la escala de Kaufman y Rousseeuw (1990), de 0,26 a 0,50 indica una estructura
débil y posiblemente artificial; solo k-means con K = 2 en W llega a 0,51, el límite de
"razonable". Y los 18 grupos que se mencionaron en la Reunión 05 eran de DBSCAN, con silueta
negativa (−0,48), no de Spectral Clustering.

### ¿Y si se agrupa con todos los sensores? (v04)

Con las 18 variables (W + X_s), k-means con K = 3 se acerca más a las fases (Rand ajustado
0,32 frente a 0,11): uno de sus grupos es en un 90 % descenso. Muchos sensores responden al
acelerador, así que TRA pesa más. Se temía que los grupos separaran motores o estados de
salud, pero no: el η² del RUL es 0,002. En los sensores, la variación por condición de vuelo
es mucho mayor que la debida a la degradación.

### ¿Cómo se definieron las fases de vuelo?

Con la tasa de cambio de la altitud dentro de cada vuelo: ascenso si sube más de 60 pies por
minuto, descenso si baja más que eso, nivelado en otro caso. El umbral se eligió a partir de
los datos: en N-CMAPSS los ascensos son lentos, y el resultado casi no cambia entre 30 y 120
pies por minuto. Con 600, un valor típico de aviones reales, el 98 % del tiempo salía
nivelado, lo que no corresponde a los perfiles.

### ¿En qué fase se equivoca más el modelo?

En el descenso, tanto en validación como en prueba (RMSE final de 2,31 frente a 1,55–1,66 en
validación). Es la fase en que las condiciones de operación cambian más rápido. La
configuración final mejora a la base en las tres fases.

### ¿Por qué se sobreestima el motor 20?

No por su régimen de operación: su perfil es casi idéntico al de los demás motores, y su
sesgo es positivo en las tres fases. Lo explica su **vida corta**. En la validación cruzada,
el sesgo de cada motor disminuye a medida que su vida aumenta (correlación de Spearman −0,83
para la base y −0,89 para la final). El modelo aprende una degradación promedio, y un motor
que se degrada más rápido que ese promedio, como el 20 (66 ciclos), recibe predicciones
optimistas. Con 6 motores es un indicio fuerte, no una prueba.

---

## 7. Uso de inteligencia artificial

### ¿Cómo usaste la IA en el trabajo?

Como herramienta de programación y análisis: Claude Code propuso planes por etapas, escribió
y ejecutó el código, generó figuras y redactó documentos de apoyo. Las decisiones
metodológicas las tomé yo a partir de las reuniones contigo. El detalle está en
`Metodologia_IA.md`.

### ¿Cómo sabes que el código es correcto si lo escribió una IA?

Por las salvaguardas: los criterios de selección se fijaron antes de ver resultados; cada
versión reproduce exactamente a la anterior (diferencia cero); cada etapa se probó primero con
pocos datos; el conjunto de prueba se evaluó una sola vez; cada figura se revisó; las
referencias se verificaron con su DOI. Además, todo el código está comentado y lo puedo
explicar línea a línea.

### ¿La IA cometió errores?

Sí, y están registrados en `Metodologia_IA.md`. Por ejemplo: una justificación falsa sobre
los gradientes del score NASA, una ablación que no era estricta, la interpretación equivocada
de tu pedido sobre las curvas a 30 épocas, un umbral inadecuado para detectar ascensos y, en
v03, una explicación imprecisa de por qué el clustering no recupera las fases, que se corrigió
en v04. En v04, la explicación del efecto del pooling se apoyó en el sesgo global; lo detectó
el profesor en la Reunión 06 y se corrigió en v05. Todos se detectaron y corrigieron.

---

## 8. Limitaciones que conviene declarar sin esperar la pregunta

1. La ablación entre extractores no es estricta.
2. Un solo subconjunto, con 6 motores de desarrollo de una misma clase de vuelo.
3. Cinco semillas por configuración; tres en la validación cruzada.
4. Hiperparámetros fijados de antemano, no ajustados por arquitectura.
5. A6 alcanzó el límite de 30 épocas en una semilla.
6. Se cambió un factor por etapa; no se exploraron interacciones.
7. La salida de A6 no está acotada.
8. La relación entre vida total y sesgo se basa en 6 motores.
9. El pooling solo se probó como max pooling de tamaño 2, en A6 y con el α elegido para A6
   sin pooling.
10. Los agrupamientos se ajustan sobre una submuestra de 8.000 registros de entrenamiento.
11. Con dos motores de validación, el sesgo de la zona final cercano a cero de la
    configuración final es el promedio de dos sesgos opuestos (−1,04 y +1,07).
12. El sesgo de la tabla de prueba es el global. El sesgo de la zona final en prueba no se
    ha calculado, para no volver a consultar ese conjunto.
13. El efecto del pooling no se puede atribuir a un solo mecanismo con 5 semillas.

## 9. Pendientes antes de la defensa

- **No afirmar que los resultados superan al estado del arte.** En la Reunión 04 se mencionó
  esa idea, pero el trabajo no tiene una comparación verificada con resultados publicados
  sobre DS02, y los protocolos (submuestreo, ventana, umbral PwL, partición) suelen diferir.
  Antes de afirmarlo hay que encontrar esos trabajos y declarar las diferencias.
- **Referencia para A3** (CNN + LSTM sin atención): falta en la bibliografía.
- **Lineamientos de la USM sobre IA:** no se encontraron; confirmarlos con el profesor guía.
- **Otras herramientas de IA:** si se usó ChatGPT u otra herramienta, agregarla a
  `Metodologia_IA.md`.
- **Fórmula de Peringal et al. (2025):** no se pudo consultar; leer el artículo antes de
  compararlo con este trabajo.
- **Definición original del score:** Saxena et al. (2008) no está en `Documentos/`; la
  convención de la suma se verificó en tres trabajos que la citan. Leer el original antes de
  citarlo directamente.
- **Sesgo de la zona final en prueba:** decidir con el profesor si se informa. Calcularlo
  sería una consulta descriptiva nueva del conjunto de prueba.
- **Fidelidad a los trabajos citados:** A1, A2, A4 y A5 están inspiradas en esos trabajos,
  no los replican.
