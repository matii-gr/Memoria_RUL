# Diagramas del pipeline y las arquitecturas — RULv03 (N-CMAPSS DS02)

Diagramas de `RULv03.ipynb`. Todas las cifras provienen de las salidas del notebook.

**Notación de formas:** `(B, T, C)` = (ventanas del lote, pasos de tiempo, canales). Con
B = 256 ventanas, T = 30 pasos (5 minutos de vuelo) y C el número de canales. Cada bloque de
las arquitecturas indica la forma del tensor a su salida, para que se vea qué dimensión
cambia: la temporal se mantiene siempre en 30 y solo cambian los canales.

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

## 1. Visión general de ProjectV03

```mermaid
flowchart LR
    E0["Etapa 0<br/>Infraestructura comentada<br/><i>reproduce v02 con diferencia 0</i>"]:::prep
    E1["Etapa 1<br/>Ablación A1 … A6<br/>curvas en eje común de 30 épocas"]:::conv
    D1{{"A6 · CNN + atención<br/>converge más lento"}}:::dec
    E2["Etapa 2<br/>Pérdida asimétrica<br/>α = 1, 2, 4 · referencias"]:::head
    D2{{"α = 2<br/>expectil τ = 0,33"}}:::dec
    E3["Etapa 3<br/>Validación cruzada<br/>y prueba oficial"]:::evalc
    E4["Etapa 4<br/>Doble clic: regímenes,<br/>error por fase, motor 20"]:::evalc
    E5["Etapa 5<br/>Documentación"]:::prep

    E0 --> E1 --> D1 --> E2 --> D2 --> E3 --> E4 --> E5

    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

**Configuración final:** A6 (CNN + atención) · normalización global · pérdida asimétrica con
α = 2. El entrenamiento es el de v02 (early stopping, paciencia 5, máximo 30 épocas), cuyas
corridas se reutilizan tras verificar que el código de v03 las reproduce exactamente.

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

## 3. Arquitectura elegida (A6) con la forma de cada tensor

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

## 4. Catálogo de arquitecturas (ablación de la etapa 1)

```mermaid
flowchart TB
    subgraph A1["A1 · 1D-CNN · 10.273 parámetros · Sateesh Babu et al. 2016 · Li, Ding y Sun 2018"]
        direction LR
        a1i["(B, 30, 18)"]:::datos --> a1c["Conv 18 → 32 canales<br/>(B, 30, 32)"]:::conv --> a1c2["Conv 32 → 64 canales<br/>(B, 30, 64)"]:::conv --> a1g["Promedio temporal<br/>(B, 64)"]:::agg --> a1h["Cabezal<br/>(B, 1)"]:::head
    end

    subgraph A2["A2 · LSTM · 23.617 parámetros · Zheng et al. 2017 · Wu et al. 2018"]
        direction LR
        a2i["(B, 30, 18)"]:::datos --> a2l["LSTM 18 → 64<br/>(B, 30, 64)"]:::lstm --> a2g["Último estado<br/>(B, 64)"]:::agg --> a2h["Cabezal<br/>(B, 1)"]:::head
    end

    subgraph A3["A3 · CNN + LSTM · 29.025 parámetros · base de v01"]
        direction LR
        a3i["(B, 30, 18)"]:::datos --> a3c["Conv 18 → 32 canales<br/>(B, 30, 32)"]:::conv --> a3l["LSTM 32 → 64<br/>(B, 30, 64)"]:::lstm --> a3g["Último estado<br/>(B, 64)"]:::agg --> a3h["Cabezal<br/>(B, 1)"]:::head
    end

    subgraph A4["A4 · LSTM + atención · 27.841 parámetros · de Oliveira da Costa et al. 2019"]
        direction LR
        a4i["(B, 30, 18)"]:::datos --> a4l["LSTM 18 → 64<br/>(B, 30, 64)"]:::lstm --> a4a["Atención<br/>(B, 64)"]:::att --> a4h["Cabezal<br/>(B, 1)"]:::head
    end

    subgraph A5["A5 · CNN + LSTM + atención · 33.249 parámetros · Li et al. 2022"]
        direction LR
        a5i["(B, 30, 18)"]:::datos --> a5c["Conv 18 → 32 canales<br/>(B, 30, 32)"]:::conv --> a5l["LSTM 32 → 64<br/>(B, 30, 64)"]:::lstm --> a5a["Atención<br/>(B, 64)"]:::att --> a5h["Cabezal<br/>(B, 1)"]:::head
    end

    subgraph A6["A6 · CNN + atención · 14.497 parámetros · ELEGIDA"]
        direction LR
        a6i["(B, 30, 18)"]:::datos --> a6c["Conv 18 → 32 canales<br/>(B, 30, 32)"]:::conv --> a6c2["Conv 32 → 64 canales<br/>(B, 30, 64)"]:::conv --> a6a["Atención<br/>(B, 64)"]:::att --> a6h["Cabezal<br/>(B, 1)"]:::head
    end

    A1 ~~~ A2 ~~~ A3 ~~~ A4 ~~~ A5 ~~~ A6

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef lstm fill:#fbe7b5,stroke:#c98500,stroke-width:2px,color:#0b0b0b
    classDef att fill:#f8d7e4,stroke:#d55181,stroke-width:2px,color:#0b0b0b
    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    style A6 stroke:#0b0b0b,stroke-width:3px
```

Las formas se muestran como `(B, tiempo, canales)` para comparar las arquitecturas entre sí;
dentro de las convoluciones el tensor se reordena a `(B, canales, tiempo)`, como detalla el
diagrama 3.

**Pares que aíslan la atención** (difieren solo en la agregación):

| Par | RMSE final en validación (mediana) |
|---|---|
| A1 → A6 (CNN) | 2,02 → 2,00 |
| A2 → A4 (LSTM) | 2,07 → 2,17 |
| A3 → A5 (CNN + LSTM) | 2,60 → 2,18 |

Las comparaciones entre extractores (por ejemplo, A1 frente a A3) no son estrictas: cambian
el bloque, el número de convoluciones y la agregación a la vez.

---

## 5. Atención de este trabajo frente a la atención del Transformer

```mermaid
flowchart TB
    subgraph ADD["Atención aditiva de agrupamiento · este trabajo · Bahdanau et al. 2015"]
        direction LR
        h1["h₁ … h₃₀<br/>(B, 30, 64)"]:::conv --> s1["Puntaje por paso<br/>eₜ = v · tanh(W hₜ + b)<br/>(B, 30)"]:::att --> w1["softmax<br/>pesos αₜ<br/>(B, 30)"]:::att --> z1["Suma ponderada<br/>Σ αₜ hₜ<br/>(B, 64)"]:::att
    end

    subgraph DOT["Atención por producto punto escalado · Vaswani et al. 2017"]
        direction LR
        x2["x₁ … x_T<br/>(B, T, d)"]:::datos --> q2["Q, K, V<br/>tres proyecciones<br/>(B, T, d)"]:::att --> s2["Puntajes<br/>Q Kᵀ / √d_k<br/>(B, T, T)"]:::att --> w2["softmax por fila<br/>(B, T, T)"]:::att --> z2["Pesos × V<br/>(B, T, d)"]:::att
    end

    ADD ~~~ DOT

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef att fill:#f8d7e4,stroke:#d55181,stroke-width:2px,color:#0b0b0b
```

| | Este trabajo | Transformer |
|---|---|---|
| Consulta | Una sola, aprendida y fija | Una por paso, calculada desde la entrada |
| Resultado | Un vector que resume la ventana | Una secuencia transformada, del mismo largo |
| Pesos por ventana | 30 | T × T |

---

## 6. Función de pérdida asimétrica (etapa 2)

```mermaid
flowchart LR
    P["Predicción ŷ<br/>y etiqueta y"]:::head
    D["Error<br/>d = ŷ − y"]:::head
    Q{"¿d > 0?<br/>sobreestima"}:::dec
    WA["peso α<br/>error peligroso"]:::att
    WB["peso 1<br/>error conservador"]:::prep
    L["Pérdida cuadrática asimétrica<br/>L_α = mean(peso · d²)<br/>estima el expectil τ = 1/(1+α)"]:::head
    R["α = 2 · τ = 0,33<br/>RMSE final 2,00 → 1,83<br/>mejor en 5 de 5 semillas"]:::evalc

    P --> D --> Q
    Q -->|"sí"| WA --> L
    Q -->|"no"| WB --> L
    L --> R

    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    classDef att fill:#f8d7e4,stroke:#d55181,stroke-width:2px,color:#0b0b0b
    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

| α | Expectil τ | Cociente subestimación / sobreestimación: teórico · medido en validación |
|---|---|---|
| 1 (= MSE) | 0,50 | 1 · 1,05 |
| 2 · elegido | 0,33 | 2 · 1,75 |
| 4 | 0,20 | 4 · 3,53 |

Referencias: Newey y Powell (1987) para la regresión por expectiles; Efron (1991) para el
nombre de la pérdida.

---

## 7. Validación cruzada dejando un motor fuera (etapa 3)

```mermaid
flowchart LR
    subgraph F["Un fold · ejemplo: evaluar el motor 16"]
        direction LR
        T["Entrenar<br/>motores 2 · 5 · 10 · 20"]:::datos --> V["Early stopping<br/>motor 18"]:::datos --> E["Evaluar<br/>motor 16"]:::evalc
    end
    F --> REP["Se repite para los 6 motores<br/>× 2 configuraciones × 3 semillas<br/>= 36 corridas"]:::prep

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
```

---

## 8. Análisis "doble clic": regímenes de operación (etapa 4)

```mermaid
flowchart TD
    WZ["Condiciones de vuelo W<br/>estandarizadas con entrenamiento"]:::datos
    SUB["Submuestra común<br/>8.000 registros"]:::datos

    subgraph MET["Tres métodos de agrupamiento"]
        direction LR
        KM["k-means<br/>K por silueta → 2"]:::prep
        DB["DBSCAN<br/>eps en el codo<br/>18 grupos · 3,5 % ruido"]:::prep
        SP["Spectral Clustering<br/>K por silueta → 2"]:::prep
    end

    PH["Referencia física<br/>fases de vuelo desde la tasa de ascenso<br/>ascenso 30 % · nivelado 47 % · descenso 23 %"]:::evalc
    CMP{{"Ningún método recupera las fases<br/>Rand ajustado ≤ 0,11<br/>ascenso y descenso se superponen en W"}}:::dec
    ERR["Error por fase<br/>el descenso es la fase más difícil"]:::evalc
    U20["Motor 20<br/>sesgo explicado por su vida corta<br/>Spearman vida-sesgo −0,83 a −0,89"]:::evalc

    WZ --> SUB --> MET --> CMP
    PH --> CMP
    PH --> ERR
    ERR --> U20

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

---

## 9. Resultado en el conjunto de prueba oficial

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

Mediana sobre 5 semillas. En la validación cruzada la ventaja no es uniforme: depende de si el
modelo base es optimista o pesimista con cada motor, lo que a su vez se relaciona con la vida
total del motor.
