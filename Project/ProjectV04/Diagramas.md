# Diagramas del pipeline y las arquitecturas — RULv04 (N-CMAPSS DS02)

Diagramas de `RULv04.ipynb`. Todas las cifras provienen de las salidas del notebook. Los
diagramas del catálogo de arquitecturas A1–A6, de la atención frente al Transformer, de la
pérdida asimétrica y de la validación cruzada no cambian y están en `ProjectV03/Diagramas.md`.

**Notación de formas:** `(B, T, C)` = (ventanas del lote, pasos de tiempo, canales). Con
B = 256 ventanas, T = 30 pasos (5 minutos de vuelo) y C el número de canales. Cada bloque de
las arquitecturas indica la forma del tensor a su salida, para que se vea qué dimensión
cambia: sin pooling, la temporal se mantiene en 30 y solo cambian los canales; con pooling
(diagrama 4), el tiempo se reduce a 15 o 7.

## Lenguaje visual

Cada tipo de bloque tiene un color fijo en todos los diagramas.

```mermaid
flowchart LR
    L1["Datos"]:::datos
    L2["Preprocesamiento"]:::prep
    L3["Convolución 1D"]:::conv
    L4["LSTM"]:::lstm
    L5["Atención"]:::att
    L6["Agregación simple"]:::agg
    L7["Cabezal de regresión"]:::head
    L8["Evaluación"]:::evalc
    L9{{"Decisión"}}:::dec

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef lstm fill:#fbe7b5,stroke:#c98500,stroke-width:2px,color:#0b0b0b
    classDef att fill:#f8d7e4,stroke:#d55181,stroke-width:2px,color:#0b0b0b
    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

---

## 1. Visión general de ProjectV04

```mermaid
flowchart LR
    E0["Etapa 0<br/>Infraestructura con opción de pooling<br/><i>reproduce v02 con diferencia 0</i>"]:::prep
    E1["Etapa 1<br/>Pooling: bibliografía<br/>y A6-P1, A6-P2 · α = 1 y 2"]:::conv
    D1{{"Regla pre-registrada<br/>no adopta el pooling"}}:::dec
    E2["Etapa 2<br/>Validación cruzada y prueba<br/><i>no aplica</i>"]:::agg
    E3["Etapa 3<br/>Regímenes: dimensiones,<br/>K de 2 a 20, espacio 18D"]:::evalc
    E4["Etapa 4<br/>Documentación"]:::prep

    E0 --> E1 --> D1
    D1 -.->|"solo si se adoptaba"| E2
    D1 --> E3 --> E4

    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

**Configuración final (sin cambios respecto de v03):** A6 (CNN + atención) · normalización
global · pérdida asimétrica con α = 2. El conjunto de prueba no se volvió a usar en v04.

---

## 2. Pipeline de datos

```mermaid
flowchart TD
    H5[("N-CMAPSS DS02<br/>9 motores turbofán · sensores a 1 Hz<br/>5,26 M registros dev + 1,25 M test")]:::datos

    subgraph CARGA["Carga · 1 de cada 10 s"]
        direction LR
        W["Condiciones de vuelo W · 4<br/>alt · Mach · TRA · T2"]:::datos
        XS["Sensores físicos X_s · 14<br/>T24 T30 T48 T50 P15 P2 P21<br/>P24 Ps30 P40 P50 Nf Nc Wf"]:::datos
        AUX["Auxiliares<br/>motor · ciclo · Fc · hs"]:::datos
        NO["Descartados<br/>X_v virtuales · T salud<br/><i>no medibles en un motor real</i>"]:::agg
    end

    subgraph SPLIT["Partición por motor · ningún motor en dos conjuntos"]
        direction LR
        TR["Entrenamiento · 55,3 %<br/>motores 2 · 5 · 10 · 16<br/>360.457 registros"]:::datos
        VA["Validación · 25,5 %<br/>motores 18 · 20<br/>165.888 registros"]:::datos
        TE["Prueba oficial · 19,2 %<br/>motores 11 · 14 · 15<br/>125.375 registros"]:::datos
    end

    subgraph PRE["Preprocesamiento sin fuga"]
        direction TB
        SC["Normalización global<br/>aprendida solo con entrenamiento"]:::prep
        PWL["Etiqueta PwL · RUL ≤ 62 ciclos<br/>mediana de la transición sano → degradado"]:::prep
        WIN["Ventanas de 30 pasos = 5 min de vuelo<br/>paso 10 · sin cruzar entre vuelos<br/>etiqueta = RUL del último instante"]:::prep
    end

    subgraph TENS["Tensores de entrada · (N, 30, 18)"]
        direction LR
        XT["Entrenamiento<br/>N = 35.287"]:::datos
        XV["Validación<br/>N = 16.251"]:::datos
        XE["Prueba<br/>N = 12.047"]:::datos
    end

    H5 --> CARGA --> SPLIT --> PRE --> TENS
    SC --> WIN
    PWL --> WIN

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
```

---

## 3. Arquitectura elegida (A6, sin pooling) con la forma de cada tensor

```mermaid
flowchart TD
    IN["Entrada<br/>(B, 30, 18)"]:::datos
    P1["permute<br/>(B, 18, 30)<br/><i>Conv1d exige canales antes que tiempo</i>"]:::agg
    C1["Conv1d · 18 → 32 canales · kernel 3 · padding 1<br/>(B, 32, 30)<br/>1.760 parámetros"]:::conv
    B1["BatchNorm · ReLU · Dropout 0,2<br/>(B, 32, 30)"]:::conv
    C2["Conv1d · 32 → 64 canales · kernel 3 · padding 1<br/>(B, 64, 30)<br/>6.208 parámetros"]:::conv
    B2["BatchNorm · ReLU<br/>(B, 64, 30)"]:::conv
    P2["permute<br/>(B, 30, 64)"]:::agg
    AT["Atención temporal<br/>pesos α₁ … α₃₀ que suman 1<br/>(B, 64) · 4.224 parámetros"]:::att
    HD["Cabezal · Linear 64 → 32 · ReLU · Linear 32 → 1<br/>(B, 1) · 2.113 parámetros"]:::head
    OUT(["RUL estimado ŷ"]):::head

    IN --> P1 --> C1 --> B1 --> C2 --> B2 --> P2 --> AT --> HD --> OUT

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef att fill:#f8d7e4,stroke:#d55181,stroke-width:2px,color:#0b0b0b
    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
```

Total: 14.497 parámetros. En ninguna capa cambia la dimensión temporal (30): con kernel 3,
padding 1 y paso 1, la longitud de salida es (30 + 2 − 3) / 1 + 1 = 30. Las convoluciones
solo cambian el número de canales.

---

## 4. A6 con y sin pooling, con la forma de cada tensor (etapa 1)

```mermaid
flowchart TB
    subgraph S0["A6 · sin pooling · 14.497 parámetros · ELEGIDA"]
        direction LR
        a0i["(B, 30, 18)"]:::datos --> a0c["Conv 18 → 32<br/>BN · ReLU · Dropout<br/>(B, 30, 32)"]:::conv --> a0c2["Conv 32 → 64<br/>BN · ReLU<br/>(B, 30, 64)"]:::conv --> a0a["Atención<br/>30 pesos<br/>(B, 64)"]:::att --> a0h["Cabezal<br/>(B, 1)"]:::head
    end

    subgraph S1["A6-P1 · pooling tras la 1.ª convolución · 14.497 parámetros"]
        direction LR
        a1i["(B, 30, 18)"]:::datos --> a1c["Conv 18 → 32<br/>BN · ReLU<br/>(B, 30, 32)"]:::conv --> a1p["MaxPool 2 · Dropout<br/>(B, 15, 32)"]:::agg --> a1c2["Conv 32 → 64<br/>BN · ReLU<br/>(B, 15, 64)"]:::conv --> a1a["Atención<br/>15 pesos<br/>(B, 64)"]:::att --> a1h["Cabezal<br/>(B, 1)"]:::head
    end

    subgraph S2["A6-P2 · pooling tras cada convolución · 14.497 parámetros"]
        direction LR
        a2i["(B, 30, 18)"]:::datos --> a2c["Conv 18 → 32<br/>BN · ReLU<br/>(B, 30, 32)"]:::conv --> a2p["MaxPool 2 · Dropout<br/>(B, 15, 32)"]:::agg --> a2c2["Conv 32 → 64<br/>BN · ReLU<br/>(B, 15, 64)"]:::conv --> a2p2["MaxPool 2<br/>(B, 7, 64)"]:::agg --> a2a["Atención<br/>7 pesos<br/>(B, 64)"]:::att --> a2h["Cabezal<br/>(B, 1)"]:::head
    end

    S0 ~~~ S1 ~~~ S2

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef att fill:#f8d7e4,stroke:#d55181,stroke-width:2px,color:#0b0b0b
    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    style S0 stroke:#0b0b0b,stroke-width:3px
```

El max pooling de tamaño 2 deja el mayor de cada par de pasos: divide el tiempo por 2,
redondeando hacia abajo (15 // 2 = 7), y no cambia los canales ni agrega parámetros. Con la
misma semilla, las tres variantes parten de los mismos pesos.

---

## 5. Resultado del pooling y decisión (etapa 1)

```mermaid
flowchart LR
    BIB["Bibliografía<br/>Li, Ding y Sun 2018: no usar pooling<br/>con entradas de baja dimensión<br/><i>ventanas de 30 pasos, como aquí</i>"]:::datos
    EXP["Experimento · validación · 5 semillas<br/>α = 2: A6 1,83 · A6-P1 2,03 · A6-P2 2,03<br/>α = 1: A6 2,00 · A6-P1 1,93 · A6-P2 1,71"]:::evalc
    R{{"Regla con α = 2<br/>(a) mediana menor: no<br/>(b) 4 de 5 semillas: 2 de 5"}}:::dec
    NO["No se adopta<br/>sigue A6 sin pooling"]:::conv
    WHY["Por qué depende de α<br/>A6-P2 baja las predicciones<br/>MSE: corrige la sobreestimación<br/>α = 2: se suma a la pérdida · sesgo −1,83"]:::head

    BIB --> R
    EXP --> R --> NO
    EXP --> WHY

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

RMSE en la zona final (RUL ≤ 20), mediana sobre 5 semillas.

---

## 6. ¿El clustering usaba 4 dimensiones o 2? (etapa 3)

```mermaid
flowchart TD
    W["W estandarizado · matriz (8000, 4)<br/>altitud · Mach · TRA · T2"]:::datos

    subgraph RED["Altitud, Mach y T2 repiten la misma información"]
        direction LR
        COR["Correlaciones en valor absoluto<br/>0,84 a 0,98"]:::prep
        PCA["PCA · varianza explicada<br/>CP1 79 % · CP2 17 % (TRA) · CP4 0,0 %"]:::prep
    end

    KM["k-means K = 2<br/>la altitud pesa casi 3 veces<br/>corta en ~26.000 y ~15.000 pies"]:::conv
    T2D["Prueba directa<br/>4 variables vs. solo altitud y Mach<br/>Rand ajustado 0,80 · 5 % cambia de grupo"]:::evalc
    TRA["Las fases se separan por TRA<br/>ascenso 80 % · nivelado 71 % · descenso 48 %"]:::evalc
    C{{"Usaba las 4 variables,<br/>pero el resultado es un corte por altitud<br/>y no recupera las fases (Rand 0,11)"}}:::dec

    W --> RED --> KM --> C
    W --> T2D --> C
    TRA --> C

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

---

## 7. Más grupos y dos espacios de agrupamiento (etapa 3)

```mermaid
flowchart TB
    subgraph E4D["W · 4 dimensiones"]
        direction LR
        k4["k-means<br/>mejor K = 2 · silueta 0,51<br/>K de 4 a 20: 0,32 a 0,40"]:::prep
        s4["Spectral<br/>mejor K = 2 · silueta 0,45<br/>K de 4 a 20: 0,09 a 0,28"]:::prep
        d4["DBSCAN<br/>18 grupos · silueta −0,48"]:::agg
    end

    subgraph E18D["W + X_s · 18 dimensiones"]
        direction LR
        k18["k-means<br/>mejor K = 3 · silueta 0,48<br/>Rand vs. fases 0,32"]:::prep
        s18["Spectral<br/>mejor K = 2 · silueta 0,44"]:::prep
        d18["DBSCAN<br/>un solo grupo en el codo"]:::agg
    end

    CONT{{"Más grupos no mejoran la silueta:<br/>los datos forman un continuo,<br/>no nubes separadas"}}:::dec
    SAL{{"18D se acerca más a las fases<br/>(un grupo es 90 % descenso)<br/>sin separar motores ni salud: η² del RUL 0,002"}}:::dec

    E4D --> CONT
    E18D --> CONT
    E18D --> SAL

    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

Escala orientativa de la silueta (Kaufman y Rousseeuw, 1990): sobre 0,70, estructura
fuerte; de 0,51 a 0,70, razonable; de 0,26 a 0,50, débil; bajo 0,25, sin estructura
sustancial.

---

## 8. Resultado en el conjunto de prueba oficial (v03, sin cambios en v04)

```mermaid
flowchart LR
    B["Base · v01<br/>A3 + global + MSE<br/>RMSE final 2,97 · sesgo +3,00"]:::agg
    M["Cambio de arquitectura<br/>A6 + global + MSE<br/>RMSE final 2,16 · sesgo +1,24"]:::conv
    F["Configuración final<br/>A6 + global + α = 2<br/>RMSE final 1,78 · sesgo +0,19"]:::evalc

    B -->|"arquitectura<br/>−0,82"| M -->|"pérdida asimétrica<br/>−0,38"| F

    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:3px,color:#0b0b0b
```

Mediana sobre 5 semillas. Evaluado una sola vez en v02 y presentado de nuevo en v03; v04
no modificó la configuración final ni volvió a usar el conjunto de prueba. En la validación cruzada la ventaja no es uniforme: depende de si el
modelo base es optimista o pesimista con cada motor, lo que a su vez se relaciona con la vida
total del motor.
