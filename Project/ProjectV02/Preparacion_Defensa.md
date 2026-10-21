# Preparación de la defensa — ProjectV02

Preguntas probables y sus respuestas, con las cifras de `RULv02.ipynb`. Las respuestas
están redactadas para decirse en voz alta: primero la respuesta directa, después el
respaldo.

**Resultado en una frase:** una CNN con atención, entrenada con una pérdida que penaliza
el doble la sobreestimación, estima el RUL en los últimos 20 ciclos con un error de 1,78
ciclos en motores nunca vistos, frente a 2,97 de la arquitectura base; esa ventaja se
confirma en la prueba oficial pero no es uniforme entre motores.

---

## 1. Datos y protocolo

### ¿Qué datos usa el trabajo?

El subconjunto DS02 de N-CMAPSS: 9 motores turbofán simulados hasta la falla, con datos a
1 Hz. Seis motores forman la partición de desarrollo (2, 5, 10, 16, 18, 20) y tres la de
prueba oficial (11, 14, 15). La entrada son 18 señales: 4 condiciones de vuelo (altitud,
Mach, ángulo de acelerador y temperatura de entrada) y 14 sensores físicos.

### ¿Por qué no usa los sensores virtuales ni los parámetros de salud?

Porque no se pueden medir en un motor real. Los sensores virtuales y los parámetros de
salud son salidas internas del simulador. Usarlos haría el problema artificialmente fácil
y el resultado no sería transferible.

### ¿Por qué la partición es por motor y no aleatoria?

Porque en operación el modelo se enfrenta a motores que nunca ha visto. Además, las
ventanas consecutivas se solapan: con una partición aleatoria, ventanas casi idénticas
quedan en entrenamiento y en evaluación, y el error medido es en parte memorización.

Esto no es teórico. La primera versión del pipeline partía por índice y reportaba una
mejora del 26 % al agregar atención. Con partición por motor y selección de la mejor
época, esa mejora bajó a cerca del 3 %, dentro del ruido entre semillas.

### ¿Para qué sirve cada conjunto?

- **Entrenamiento (55,3 %, motores 2, 5, 10, 16):** ajusta los pesos.
- **Validación (25,5 %, motores 18 y 20):** decide la época de detención y todas las
  elecciones del estudio (arquitectura, normalización, α).
- **Prueba (19,2 %, motores 11, 14 y 15):** se evaluó una sola vez, al final. El notebook
  verifica que ninguna decisión dependió de ella.

### ¿Qué es el etiquetado lineal por tramos y por qué el umbral es 62?

Mientras el motor está sano, los sensores no contienen información sobre cuánto falta
para la falla, así que pedirle al modelo que distinga 80 de 70 ciclos no tiene sentido.
El RUL se recorta a un máximo. Ese máximo, 62 ciclos, es la mediana del RUL en el momento
en que los motores de entrenamiento pasan de sanos a degradados, según el indicador de
salud del propio dataset (58, 72, 65 y 47 ciclos).

El valor 125 que se usa habitualmente proviene del C-MAPSS de 2008. En DS02 no tiene
efecto, porque el RUL máximo es 88.

### ¿Por qué ventanas de 30 pasos y un dato cada 10 segundos?

Cada ventana cubre 5 minutos de vuelo. El submuestreo se eligió para que el experimento
fuera viable en CPU. Ninguno de los dos valores se optimizó: es una limitación declarada.
Las ventanas nunca cruzan de un vuelo al siguiente, para no mezclar un aterrizaje con un
despegue.

### ¿Por qué la métrica principal es el error en los últimos 20 ciclos?

Porque es donde la estimación tiene consecuencias. Sobreestimar cuando quedan 50 ciclos
no cambia ninguna decisión de mantenimiento; hacerlo cuando quedan 10 puede significar
una falla en vuelo. Este criterio se acordó en la Reunión 03. La zona final contiene el
27,7 % de las muestras de desarrollo, suficiente para medir con estabilidad.

### ¿Por qué reporta mediana y rango, y no media con desviación estándar?

Con 5 semillas, una media o una desviación estándar dependen demasiado de un solo valor
atípico. Se muestran todos los valores individuales, la mediana y el mínimo y máximo. Por
la misma razón no se usan pruebas de significancia.

### ¿Los resultados son reproducibles?

Sí. El pipeline es determinista: con las mismas semillas, la infraestructura de v02
reproduce los resultados de v01 con diferencia exactamente cero. Cada entrenamiento se
guarda en disco con su configuración, y los criterios de selección se escribieron en un
archivo antes de ver los resultados.

---

## 2. Arquitecturas

### ¿Qué hace cada bloque?

- **Convolución 1D:** extrae patrones locales en el tiempo (tres pasos consecutivos)
  combinando las 18 señales.
- **LSTM:** recorre la ventana en orden y acumula memoria de lo que vio.
- **Atención temporal:** asigna un peso a cada uno de los 30 pasos y resume la ventana
  como su promedio ponderado. Los pesos se aprenden.
- **Cabezal:** dos capas densas que convierten ese resumen en un número, el RUL.

### En v01 se presentaron "tres mecanismos de atención". ¿Lo eran?

No. Eran tres formas de resumir la ventana, y solo una es un mecanismo de atención:

- **Último estado:** toma la salida del LSTM en el paso 30. No hay atención.
- **Promedio temporal:** promedia los 30 pasos con peso igual. No hay atención.
- **Atención:** promedia los 30 pasos con pesos aprendidos.

El promedio es el control que permite saber si una mejora viene de los pesos aprendidos o
simplemente de usar toda la ventana.

### ¿Por qué quedó elegida A6 y no A5, que era la alternativa seleccionada?

Porque el criterio se fijó antes de ver los resultados: entre las arquitecturas con
atención, la de menor mediana de error en la zona final. A6 obtuvo 2,00 ciclos; A5, 2,18.

Hay que decir también lo que el resultado no demuestra. Cinco de las seis arquitecturas
caen dentro de 0,18 ciclos, con rangos solapados. A6 supera a una CNN simple por 0,01
ciclos. No se puede afirmar que A6 sea mejor que A5: son indistinguibles con 5 semillas.

### ¿Entonces la arquitectura híbrida propuesta no era necesaria?

En DS02 y en la zona final, una CNN de 10 mil parámetros rinde como la híbrida con
atención de 33 mil. Es un resultado del trabajo, no un fracaso: el aporte es el análisis
de qué bloque aporta qué, y eso solo se puede saber comparando contra líneas base simples.

### ¿La atención sirve o no?

Depende de sobre qué se aplica:

| Sobre | Zona final | Efecto |
|---|---|---|
| CNN + LSTM | 2,60 → 2,18 | Mejora, en 4 de 5 semillas |
| CNN | 2,02 → 2,00 | Sin cambio; mejora el error global y elimina el sesgo |
| LSTM | 2,07 → 2,17 | Empeora levemente |

No hay un beneficio uniforme. Además, en A6 los pesos aprendidos son casi iguales entre
sí (cerca de 1/30), así que en la práctica A6 promedia la ventana.

### ¿La ablación es estricta?

No del todo, y conviene decirlo antes de que lo pregunten.

- **Los pares con y sin atención sí son limpios:** A1 con A6, A2 con A4 y A3 con A5
  difieren solo en la forma de resumir la ventana.
- **Las comparaciones entre extractores no lo son.** A1 y A6 tienen una segunda
  convolución que A3 y A5 no tienen; se agregó para que todas entregaran un vector de 64
  dimensiones al cabezal. La agregación sin atención es el promedio en A1 y el último
  estado en A2 y A3. Y el dropout cambia de posición.

En consecuencia, las conclusiones sobre la atención se sostienen, pero no se puede
atribuir con certeza la diferencia entre, por ejemplo, la CNN y la CNN + LSTM a la
recurrencia. Una ablación estricta (tres extractores por dos agregaciones, con bloques
idénticos) es el primer punto del trabajo futuro.

### ¿Más parámetros dan mejor resultado?

No en este estudio. La arquitectura más pequeña empata con las mejores y la más grande
queda quinta.

### ¿Por qué no se comparó con Transformers?

Se acordó con el profesor guía en la Reunión 03 dejar fuera los Transformers y las
técnicas clásicas, y concentrar el trabajo en la ablación de bloques.

---

## 3. Normalización

### ¿Por qué no se normalizó por clase de vuelo, como estaba planeado?

Porque en DS02 no es posible. La clase de vuelo es constante por motor, y los seis
motores de desarrollo son de la clase 3. Normalizar por clase sería idéntico a normalizar
globalmente, y para los motores de prueba 14 y 15, de clases 1 y 2, no habría estadísticas
con que hacerlo.

### ¿Qué se probó en su lugar?

- **Por régimen de operación:** se agrupan las condiciones de vuelo con k-means y cada
  sensor se normaliza según su régimen. El coeficiente de silueta eligió 2 regímenes:
  crucero alto, y ascenso y descenso.
- **Línea base por motor:** cada motor se normaliza con sus propios primeros 10 ciclos,
  que son sanos en todas las unidades.

### ¿Cuál fue el resultado?

Se mantuvo la normalización global.

| Normalización | Error en zona final (A6) |
|---|---|
| Global | 2,00 |
| Por régimen | 2,02 |
| Línea base por motor | 3,29 |

### ¿Por qué la normalización por régimen no ayudó?

Con solo 2 regímenes elimina poco: la parte de la variación de los sensores explicada por
las condiciones de vuelo baja apenas del 99 % al 84 %. Además, asignar cada registro a un
régimen de forma tajante introduce un escalón artificial al cruzar de uno a otro.

### ¿Por qué la línea base por motor empeoró?

La hipótesis, no verificada, es que al restar el nivel propio de cada motor se pierde
información útil sobre su estado o su vida total. Además, este método no reduce la
dependencia de las condiciones de vuelo (98 %), que era el objetivo.

### ¿La validación podía detectar un beneficio de la línea base por motor?

No del todo. Su ventaja esperable aparece con motores de otra clase de vuelo, y todos los
de desarrollo son de la misma. Aun así, su costo dentro del mismo dominio fue tan grande
que no se justificaba adoptarla.

---

## 4. Función de pérdida

### ¿Qué es la pérdida asimétrica?

Es un error cuadrático en el que sobreestimar pesa α veces más que subestimar. Con α = 1
es el error cuadrático medio habitual.

### ¿Por qué penalizar más la sobreestimación?

Porque los dos errores no cuestan lo mismo. Sobreestimar el RUL es declarar más vida útil
de la que queda: la intervención llega tarde. Subestimar solo adelanta una inspección.

### ¿Qué estima un modelo entrenado así?

Deja de estimar la media del RUL y pasa a estimar un expectil, de nivel 1/(1+α). Con
α = 2 es el expectil 0,33: una estimación deliberadamente conservadora. La pérdida es la
de la regresión por expectiles de Newey y Powell (1987).

### ¿Por qué α = 2?

Es el valor que minimiza el score NASA en la zona final en validación, que era el
criterio fijado de antemano. Frente al error cuadrático medio, mejora en las 5 semillas:
el error en la zona final baja de 2,00 a 1,83 y la sobreestimación media, un 26 %.

Con α = 4 el modelo se vuelve demasiado conservador: reduce más la sobreestimación, pero
empieza a subestimar en la zona final y pierde precisión.

### ¿Cuál es el costo?

El error aumenta al inicio de la vida útil, donde el modelo pasa a subestimar. En la
prueba oficial, el error en la zona temprana sube de 6,67 a 7,72 ciclos. Es el compromiso
que se consideró aceptable en la Reunión 03.

### ¿Por qué no usar directamente el score NASA como pérdida?

Por dos razones. La pérdida elegida tiene una interpretación estadística cerrada y un
solo parámetro. Y reservar el score NASA como métrica permite usarlo para elegir α sin
que la selección dependa de la misma función que se optimiza. Entrenar con el score NASA
no se evaluó en este trabajo.

### ¿Por qué el early stopping no usa el RMSE?

Porque detener por RMSE elegiría la época con menos sesgo conservador, y anularía
justamente el efecto que la pérdida busca. Se detiene con la misma pérdida asimétrica,
medida en validación.

---

## 5. Validación cruzada y resultado final

### ¿Cuál es el resultado final?

En la prueba oficial, sobre 5 semillas:

| Configuración | Error en zona final | Sesgo |
|---|---|---|
| Base: CNN + LSTM, error cuadrático medio | 2,97 [2,47 – 3,30] | +3,00 |
| A6, error cuadrático medio | 2,16 [1,92 – 2,45] | +1,24 |
| Final: A6, pérdida asimétrica α = 2 | 1,78 [1,50 – 1,97] | +0,19 |

La peor semilla de la configuración final es mejor que la mejor de la base. Mejoran los
tres motores de prueba, incluidos los dos de clases de vuelo nunca vistas.

### ¿Cuánto aporta cada cambio?

La arquitectura reduce el error en 0,82 ciclos y la pérdida asimétrica en 0,38 más. El
sesgo optimista de la base prácticamente desaparece.

### ¿Qué mostró la validación cruzada?

Que la ventaja no es uniforme. Dejando un motor fuera cada vez, la configuración final es
mejor en solo 2 de los 6 motores (16 y 20). En cambio, reduce la sobreestimación en 16 de
18 casos, que es lo que la pérdida busca.

### ¿Por qué la validación cruzada y la prueba no coinciden?

Por la dirección del sesgo del modelo base en cada motor. La pérdida asimétrica desplaza
las predicciones hacia abajo:

- Donde la base era optimista (motores 16 y 20, sesgo de +3,6 y +4,8), eso corrige.
- Donde ya era pesimista (motores 5 y 10, sesgo de −3,7 y −4,1), eso empeora.

En los motores de prueba la base sobreestima (+3,0), por eso ahí la ganancia es clara.

### ¿Entonces el resultado de la prueba es confiable?

Es un resultado real, obtenido sin usar la prueba para decidir nada. Pero la validación
cruzada muestra que depende del motor. La afirmación que se sostiene es esta: la pérdida
asimétrica siempre reduce la sobreestimación, y mejora la precisión cuando el modelo
tiende a ser optimista. Con 6 motores de desarrollo no se puede prometer más.

### ¿Qué son las predicciones extremas de A6?

En algunos folds, A6 predijo hasta 189 ciclos para unas pocas ventanas, cuando el máximo
posible es 62. La causa probable es que no tiene ninguna capa que acote su salida; el LSTM
sí acota sus estados internos. Recortar las predicciones al rango válido solo cambia un
fold, así que no explica el patrón general. Acotar la salida queda como trabajo futuro: no
se aplicó porque habría sido una decisión tomada después de ver la prueba.

---

## 6. Limitaciones que conviene declarar sin esperar la pregunta

1. La ablación entre extractores no es estricta (sección 2).
2. Un solo subconjunto, con 6 motores de desarrollo de una misma clase de vuelo.
3. Cinco semillas por configuración; tres en la validación cruzada.
4. Hiperparámetros fijados de antemano, no ajustados por arquitectura.
5. A6 alcanzó el límite de 30 épocas en una semilla.
6. Se cambió un factor por etapa; no se exploraron interacciones.

## 7. Pendientes antes de la defensa

Afirmaciones que hoy no tienen respaldo verificado y que no deben decirse sin él:

- **Comparación con la literatura.** No hay en el trabajo una comparación verificada con
  errores publicados sobre DS02. Los protocolos difieren (submuestreo, ventana, umbral
  PwL), así que las cifras no son directamente comparables. Hay que buscar las
  referencias y declarar las diferencias de protocolo, o no hacer la comparación.
- **Referencia para la arquitectura CNN + LSTM sin atención (A3).** Falta en la
  bibliografía.
- **Trabajos de RUL con pérdidas asimétricas o por expectiles.** Hay que buscarlos para
  situar la propuesta en el estado del arte.
- **Newey y Powell (1987)** y **Saxena et al. (2008)**: confirmar los datos bibliográficos
  completos antes de citarlos.
- **Fidelidad a los trabajos citados.** Las arquitecturas A1, A2, A4 y A5 están inspiradas
  en esos trabajos, no los replican. Hay que presentarlas así.
