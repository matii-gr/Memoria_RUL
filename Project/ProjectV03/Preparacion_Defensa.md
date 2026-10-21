# Preparación de la defensa — ProjectV03

Preguntas probables y sus respuestas, con las cifras de `RULv03.ipynb`. Están redactadas
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
forma predecible cuánto más conservador es el modelo.

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

| Configuración | RMSE final | Sesgo |
|---|---|---|
| Base: CNN + LSTM, error cuadrático medio | 2,97 [2,47 – 3,30] | +3,00 |
| A6, error cuadrático medio | 2,16 [1,92 – 2,45] | +1,24 |
| Final: A6, pérdida asimétrica α = 2 | 1,78 [1,50 – 1,97] | +0,19 |

La peor semilla de la configuración final es mejor que la mejor de la base, y mejoran los tres
motores de prueba, incluidos los dos de clases de vuelo nunca vistas.

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

## 5. Regímenes de operación y motor 20

### ¿Por qué no usaste DBSCAN o Spectral Clustering para los regímenes?

Se probaron en v03, junto con k-means. Ninguno recupera las fases de vuelo (índice de Rand
ajustado ≤ 0,11 en sus configuraciones elegidas). DBSCAN además resulta inadecuado: los datos
forman trayectorias continuas sin zonas densas separadas, y el número de grupos depende mucho
de su parámetro (34, 18 o 9).

### ¿Por qué ningún método funciona?

Porque el problema no es el algoritmo, sino lo que se agrupa. Un avión pasa por las mismas
condiciones de vuelo al subir y al bajar, así que en el espacio de W el ascenso y el descenso
se superponen. Las fases se distinguen por cómo cambian las condiciones, no por su valor en
un instante. Para definir regímenes con sentido físico hay que incluir variables dinámicas,
como la tasa de ascenso.

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

## 6. Uso de inteligencia artificial

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
de tu pedido sobre las curvas a 30 épocas y un umbral inadecuado para detectar ascensos.
Todos se detectaron y corrigieron.

---

## 7. Limitaciones que conviene declarar sin esperar la pregunta

1. La ablación entre extractores no es estricta.
2. Un solo subconjunto, con 6 motores de desarrollo de una misma clase de vuelo.
3. Cinco semillas por configuración; tres en la validación cruzada.
4. Hiperparámetros fijados de antemano, no ajustados por arquitectura.
5. A6 alcanzó el límite de 30 épocas en una semilla.
6. Se cambió un factor por etapa; no se exploraron interacciones.
7. La salida de A6 no está acotada.
8. La relación entre vida total y sesgo se basa en 6 motores.

## 8. Pendientes antes de la defensa

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
- **Fidelidad a los trabajos citados:** A1, A2, A4 y A5 están inspiradas en esos trabajos,
  no los replican.
