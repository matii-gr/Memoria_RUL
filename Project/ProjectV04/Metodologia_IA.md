# Metodología de uso de inteligencia artificial

Documento pedido en la **Reunión 04**: describir cómo se usó la IA en el desarrollo del
código y en la experimentación, para incorporarlo a la metodología de la memoria. Esta
versión, de ProjectV04, agrega lo ocurrido al implementar los pedidos de la **Reunión 05**.

---

## 1. Marco de referencia

La declaración de posición de COPE (*Committee on Publication Ethics*) sobre autoría y
herramientas de IA establece tres principios que este trabajo adopta:

1. **La IA no es autora.** No puede asumir responsabilidad por el trabajo, declarar
   conflictos de interés ni gestionar derechos.
2. **Su uso se declara en la metodología.** Cuando una herramienta de IA participa en la
   redacción, en la producción de figuras o en la recolección y el análisis de datos, debe
   describirse en la sección de materiales y métodos, indicando qué herramienta se usó y
   cómo.
3. **El autor es responsable de todo el contenido**, incluidas las partes producidas con
   ayuda de la IA.

Referencia: COPE (2023). *Authorship and AI tools* (declaración de posición, 13 de febrero
de 2023). https://publicationethics.org/guidance/cope-position/authorship-and-ai-tools

Las recomendaciones del ICMJE (*International Committee of Medical Journal Editors*)
siguen la misma línea. Su texto vigente debe consultarse en
https://www.icmje.org/icmje-recommendations.pdf antes de citarlo, porque se ha actualizado
varias veces desde 2023.

**Pendiente:** no se encontraron lineamientos oficiales de la Universidad Técnica Federico
Santa María sobre el uso de IA generativa en trabajos de titulación. Conviene confirmarlo
con el profesor guía o con el Departamento de Electrónica.

---

## 2. Herramientas

| Herramienta | Uso |
|---|---|
| **Claude Code** (Anthropic), extensión de VS Code, con los modelos Claude Opus 5 y 5.5 | Revisión de código, planificación por etapas, escritura y ejecución de los notebooks, figuras, búsqueda y verificación de referencias, redacción de documentos de apoyo |
| *Otras herramientas* | En la Reunión 04 se mencionó el uso de ChatGPT. Si se usó en el trabajo (por ejemplo, en ProjectV00), debe agregarse aquí con su uso concreto |

Ejecución: Python 3.12, PyTorch 2.7 (CPU), scikit-learn, pandas y matplotlib, en el
computador del estudiante. La IA ejecutó el código en ese mismo equipo, y todos los
resultados provienen de esas ejecuciones.

---

## 3. División de roles

| Quién | Qué hizo |
|---|---|
| **Profesor guía** | Definió la dirección del trabajo en cada reunión: qué experimentos hacer, qué comparar, cómo reportar (por ejemplo, mediana y rango en lugar de media ± desviación) y qué dejar fuera (Transformers y técnicas clásicas) |
| **Estudiante** | Fijó los objetivos de cada versión a partir de las reuniones, tomó las decisiones en cada punto abierto, revisó y aprobó los planes, interpretó los pedidos del profesor y presentó los resultados |
| **IA** | Propuso planes por etapas, escribió el código, lo ejecutó, verificó los resultados, generó figuras y tablas, redactó las discusiones y documentos, y buscó referencias |

Ejemplos concretos de decisiones tomadas por el estudiante:

- Borrar la primera versión de ProjectV02 y reorganizar todo en un único notebook.
- Normalizar por condición de operación con dos métodos (N1a y N1b) cuando la propuesta
  original, por clase de vuelo, resultó inviable en DS02.
- Incluir la arquitectura A6 (CNN + atención) en la ablación.
- Mantener en ProjectV03 las mismas arquitecturas de ProjectV02.
- Corregir la interpretación de la IA sobre el pedido de curvas a 30 épocas (ver 6.2).
- Usar diagramas Mermaid en lugar de draw.io, y no usar emojis en ningún documento.
- En ProjectV04: responder al pedido sobre las "16 dimensiones" verificando las 4 de W y
  comparando además con las 18 variables del modelo; probar el pooling en A6 con dos
  ubicaciones; y fijar que, si el pooling se adoptaba, se evaluaría una segunda vez en el
  conjunto de prueba, declarándolo.
- Qué archivos subir al repositorio de GitHub y cuáles excluir.

---

## 4. Flujo de trabajo

Cada versión del proyecto siguió el mismo ciclo:

1. **Reunión con el profesor** y resumen escrito de los acuerdos (`Reuniones/`).
2. **Plan por etapas** redactado por la IA a partir del resumen (`Plan_ProjectVxx.md`), con
   los puntos que requerían una decisión explícitamente separados.
3. **Decisiones del estudiante** sobre esos puntos.
4. **Implementación etapa por etapa**, con este orden en cada una:
   - Prueba rápida (`SMOKE_TEST = True`): el código completo con pocos datos, para detectar
     errores antes de gastar horas de cómputo.
   - Ejecución completa.
   - Revisión de las salidas y de cada figura.
   - Redacción de la discusión con las cifras obtenidas.
5. **Revisión del estudiante**, que preguntó, corrigió o pidió cambios antes de pasar a la
   etapa siguiente.

---

## 5. Salvaguardas metodológicas

Una IA puede producir código que corre pero que está mal, o afirmaciones convincentes pero
falsas. Estas salvaguardas buscan que esos errores se detecten:

| Salvaguarda | Qué evita | Dónde |
|---|---|---|
| **Criterios de selección escritos antes de ver resultados** | Elegir la configuración que más conviene después de mirar los números | `results/seleccion.json`, sección 0.12 |
| **Verificación de reproducibilidad** | Que una reescritura del código cambie los resultados sin que nadie lo note | v02 reproduce v01 con diferencia 0; el código de v03 reentrena 3 corridas y el de v04, 2, y coinciden exactamente con v02 (sección 0.14) |
| **Pruebas rápidas antes de cada corrida** | Errores de código descubiertos tras horas de cómputo | Cada etapa |
| **Caché con huella de configuración** | Reutilizar resultados obtenidos con otra configuración | `HP_HASH`, sección 0.2 |
| **Conjunto de prueba evaluado una sola vez** | Que la prueba influya en las decisiones | Un `assert` verifica que ninguna corrida de prueba se entrene durante la evaluación |
| **Revisión visual de cada figura** | Gráficos engañosos o ilegibles | Cada etapa |
| **Referencias verificadas en la web, con DOI** | Citas inventadas o con datos erróneos | Sección 2.1 de v03 |
| **Citas textuales tomadas del PDF original, con página** | Atribuir a un autor algo que no dijo | v04, sección 1.1: la cita de Li, Ding y Sun (2018) y las configuraciones de los otros trabajos se extrajeron de sus PDF |
| **Confirmar los pedidos ambiguos antes de implementar** | Repetir el error de la Reunión 04 (ver 6.2) | En v04, la IA preguntó qué significaban las "16 dimensiones" en vez de suponerlo, porque en los datos no existe un conjunto de 16 variables |
| **Observaciones posteriores separadas de las decisiones** | Cambiar la configuración por un resultado que la regla no contemplaba | v04, sección 1.6: A6-P2 con MSE se informa como observación, sin cambiar la decisión |
| **Comprobaciones teóricas sobre los datos** | Afirmaciones teóricas sin respaldo empírico | Condición de equilibrio de los expectiles (sección 2.4) |

---

## 6. Errores de la IA detectados y corregidos

Este registro forma parte de la metodología: muestra que el trabajo no aceptó sin revisión
lo que produjo la IA.

### 6.1 Errores de contenido

| Error | Cómo se detectó | Corrección |
|---|---|---|
| Se afirmó que usar el score NASA como pérdida hacía "explotar" los gradientes con errores de 30 a 60 ciclos. Era falso: a esos errores, sus gradientes son menores que los del MSE | El estudiante preguntó dónde se observaba | Se corrigió el notebook; la justificación pasó a ser la interpretación estadística de la pérdida |
| La segunda convolución de A1 y A6 (32 → 64 canales) rompió la ablación entre extractores, y no se declaró como decisión de diseño | El estudiante preguntó por qué las arquitecturas no compartían los bloques | Se declaró como limitación en v02 y v03 |
| La frase "capacidad igualada" describía mal las arquitecturas, cuyos parámetros van de 10 mil a 33 mil | Revisión de la IA al ver la tabla de parámetros | Se cambió por "mismos tamaños de bloque" y se reportan los parámetros |
| En v01 se afirmó que el RMSE estaba "en el rango de lo publicado" sin verificarlo | Revisión de la IA al preparar la defensa | Se marcó como pendiente; no debe afirmarse sin referencias |
| En v03 se explicó que el clustering no recupera las fases porque el ascenso y el descenso "se superponen en W". No es exacto: TRA los separa; el problema es que altitud, Mach y T2 repiten la misma información y dominan la distancia | El análisis pedido por el profesor en la Reunión 05 (v04, etapa 3) | Se corrigió en la discusión de v04 y en `Preparacion_Defensa.md` |

### 6.2 Errores de interpretación

| Error | Cómo se detectó | Corrección |
|---|---|---|
| Se interpretó el pedido de "curvas con el mismo número de épocas" como entrenar 30 épocas fijas | El estudiante aclaró que se pedía un eje común de 30 épocas manteniendo el early stopping | Se restauró el early stopping y se reutilizaron las corridas de v02, verificadas |

### 6.3 Errores de implementación

| Error | Cómo se detectó | Corrección |
|---|---|---|
| El umbral para detectar ascensos (600 pies por minuto, típico de aviones reales) clasificaba el 98 % del tiempo como vuelo nivelado | Comprobación de plausibilidad de la IA frente a una figura de un vuelo | Umbral elegido desde los datos (60 pies por minuto), con tabla de sensibilidad |
| Una columna contaba como "detenidas por el límite de 30 épocas" corridas que se detuvieron por early stopping justo en la época 30 | Revisión de la IA de la tabla | Se redefinió la condición |
| Una fila de los diagramas describía la comparación A1 → A3 como "agregar recurrencia a la CNN", aunque cambian dos cosas | Pregunta del estudiante | Señalado; el estudiante decidió no modificar el diagrama de v02 |
| El plan de v04 indicaba calcular el error por régimen también en el conjunto de prueba, lo que contradecía la regla de la etapa 0 de no consultarlo | Revisión de la IA al implementar la etapa 3 | Se calculó solo en validación y se registró en el plan |
| En la figura de la silueta según K, la leyenda tapaba la escala de interpretación | Revisión visual de la figura | Se movió la leyenda debajo de los paneles |

---

## 7. Comprensión y responsabilidad

El profesor guía pidió explícitamente que el estudiante entienda y pueda explicar toda la
metodología. Para eso:

- Todo el código de ProjectV03 y ProjectV04 está comentado en español, incluida la forma de
  los tensores en cada paso de la red.
- `Guia_Estudio.md` recorre el código y su teoría, incluida la atención de *Attention Is All
  You Need*.
- `Preparacion_Defensa.md` reúne las preguntas probables y sus respuestas.
- El profesor guía realizará una sesión de preguntas tipo examen de grado.

---

## 8. Texto sugerido para la memoria

Párrafo para adaptar en la sección de metodología:

> Para el desarrollo del código y la experimentación se utilizó Claude Code (Anthropic), un
> asistente de programación basado en modelos de lenguaje, como herramienta de apoyo. La
> herramienta propuso planes de trabajo por etapas, escribió y ejecutó el código, generó
> figuras y tablas y asistió en la búsqueda de referencias. Las decisiones metodológicas
> fueron tomadas por el autor a partir de las reuniones con el profesor guía. Para
> controlar los errores propios de estas herramientas se adoptaron las siguientes
> salvaguardas: criterios de selección fijados antes de observar los resultados,
> verificaciones de reproducibilidad exacta entre versiones del código, pruebas rápidas
> previas a cada experimento, evaluación única del conjunto de prueba, revisión de cada
> figura y verificación de todas las referencias bibliográficas. Los errores de la
> herramienta detectados durante el trabajo se registraron y corrigieron. Siguiendo la
> declaración de posición de COPE (2023), la herramienta no se considera autora y el autor
> asume la responsabilidad por la totalidad del contenido.
