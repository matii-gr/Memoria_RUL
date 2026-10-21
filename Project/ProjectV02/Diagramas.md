# Diagramas del pipeline y las arquitecturas — RULv02 (N-CMAPSS DS02)

Diagramas de `RULv02.ipynb`. Todas las cifras provienen de las salidas del notebook.

## Lenguaje visual

Cada tipo de bloque tiene un color fijo en todos los diagramas, para que la diferencia
entre arquitecturas se vea de un vistazo.

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

## 1. Visión general: un factor por etapa

Cada etapa cambia un solo factor y fija su elección para la siguiente. Los criterios de
selección se registraron antes de ver resultados (`results/seleccion.json`).

```mermaid
flowchart LR
    E0["Etapa 0<br/>Infraestructura común<br/><i>reproduce v01 con Δ = 0</i>"]:::prep
    E1["Etapa 1<br/>Ablación de arquitecturas<br/>A1 … A6 · 5 semillas"]:::conv
    D1{{"A6 · CNN + atención<br/>RMSE final 2,00"}}:::dec
    E2["Etapa 2<br/>Normalización<br/>global · N1a · N1b"]:::prep
    D2{{"Se mantiene la global<br/>N1a empata · N1b empeora"}}:::dec
    E3["Etapa 3<br/>Pérdida asimétrica<br/>α = 1, 2, 4"]:::head
    D3{{"α = 2<br/>mejor en 5 de 5 semillas"}}:::dec
    E4["Etapa 4<br/>Validación cruzada<br/>6 folds · 3 semillas"]:::evalc
    R["Prueba oficial · única evaluación<br/>RMSE final 1,78 frente a 2,97 de la base"]:::evalc

    E0 --> E1 --> D1 --> E2 --> D2 --> E3 --> D3 --> E4 --> R

    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

**Configuración final:** A6 (CNN + atención) · normalización global · pérdida asimétrica
con α = 2.

---

## 2. Pipeline de datos

```mermaid
flowchart TD
    H5[("N-CMAPSS DS02<br/>9 motores turbofán · sensores a 1 Hz<br/>5,26 M registros dev + 1,25 M test")]:::datos

    subgraph CARGA["Carga · 1 de cada 10 s"]
        direction LR
        W["Condiciones de vuelo W · 4<br/>alt · Mach · TRA · T2"]:::datos
        XS["Sensores físicos X_s · 14<br/>T24 T30 T48 T50 P15 P2 P21<br/>P24 Ps30 P40 P50 Nf Nc Wf"]:::datos
        AUX["Auxiliares<br/>unidad · ciclo · Fc · hs"]:::datos
        NO["Descartados<br/>X_v virtuales · T salud<br/><i>no medibles en un motor real</i>"]:::agg
    end

    subgraph SPLIT["Partición por motor · ningún motor en dos conjuntos"]
        direction LR
        TR["Entrenamiento · 55,3 %<br/>motores 2 · 5 · 10 · 16<br/>360.457 registros"]:::datos
        VA["Validación · 25,5 %<br/>motores 18 · 20<br/>165.888 registros"]:::datos
        TE["Prueba oficial · 19,2 %<br/>motores 11 · 14 · 15<br/>125.375 registros<br/><i>se evalúa una sola vez</i>"]:::datos
    end

    subgraph PRE["Preprocesamiento sin fuga"]
        direction TB
        SC["Normalización<br/>aprendida solo con entrenamiento"]:::prep
        PWL["Etiqueta PwL · RUL ≤ 62 ciclos<br/>mediana de la transición sano → degradado"]:::prep
        WIN["Ventanas de 30 pasos = 5 min de vuelo<br/>paso 10 · sin cruzar entre vuelos<br/>etiqueta = RUL del último instante"]:::prep
    end

    subgraph TENS["Tensores de entrada · ventanas × 30 × 18"]
        direction LR
        XT["Entrenamiento<br/>35.287"]:::datos
        XV["Validación<br/>16.251"]:::datos
        XE["Prueba<br/>12.047"]:::datos
    end

    subgraph ZONAS["Zonas de evaluación · RUL restante"]
        direction LR
        Z1["Final · ≤ 20 ciclos<br/>27,7 % de dev<br/><b>métrica principal</b>"]:::evalc
        Z2["Intermedia · 20 – 40<br/>27,2 % de dev"]:::evalc
        Z3["Temprana · > 40<br/>45,1 % de dev"]:::evalc
    end

    H5 --> CARGA --> SPLIT --> PRE --> TENS --> ZONAS
    SC --> WIN
    PWL --> WIN

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
```

---

## 3. Catálogo de arquitecturas (ablación de la etapa 1)

Todas reciben ventanas de 30 pasos × 18 canales y terminan en el mismo cabezal
(Linear 64 → 32 + ReLU → Linear 32 → 1). Comparten tamaños de bloque, pero no el
número de parámetros.

```mermaid
flowchart TB
    subgraph A1["A1 · 1D-CNN · 10.273 parámetros · Sateesh Babu et al. 2016 · Li, Ding y Sun 2018"]
        direction LR
        a1i["Ventana<br/>30 × 18"]:::datos --> a1c["Conv 18 → 32<br/>BN · ReLU · Dropout"]:::conv --> a1c2["Conv 32 → 64<br/>BN · ReLU"]:::conv --> a1g["Promedio<br/>temporal"]:::agg --> a1h["Cabezal"]:::head --> a1o(["ŷ"]):::head
    end

    subgraph A2["A2 · LSTM · 23.617 parámetros · Zheng et al. 2017 · Wu et al. 2018"]
        direction LR
        a2i["Ventana<br/>30 × 18"]:::datos --> a2l["LSTM 18 → 64<br/>Dropout"]:::lstm --> a2g["Último<br/>estado h₃₀"]:::agg --> a2h["Cabezal"]:::head --> a2o(["ŷ"]):::head
    end

    subgraph A3["A3 · CNN + LSTM · 29.025 parámetros · base de v01"]
        direction LR
        a3i["Ventana<br/>30 × 18"]:::datos --> a3c["Conv 18 → 32<br/>BN · ReLU · Dropout"]:::conv --> a3l["LSTM 32 → 64"]:::lstm --> a3g["Último<br/>estado h₃₀"]:::agg --> a3h["Cabezal"]:::head --> a3o(["ŷ"]):::head
    end

    subgraph A4["A4 · LSTM + atención · 27.841 parámetros · de Oliveira da Costa et al. 2019"]
        direction LR
        a4i["Ventana<br/>30 × 18"]:::datos --> a4l["LSTM 18 → 64<br/>Dropout"]:::lstm --> a4a["Atención<br/>temporal"]:::att --> a4h["Cabezal"]:::head --> a4o(["ŷ"]):::head
    end

    subgraph A5["A5 · CNN + LSTM + atención · 33.249 parámetros · Li et al. 2022"]
        direction LR
        a5i["Ventana<br/>30 × 18"]:::datos --> a5c["Conv 18 → 32<br/>BN · ReLU · Dropout"]:::conv --> a5l["LSTM 32 → 64"]:::lstm --> a5a["Atención<br/>temporal"]:::att --> a5h["Cabezal"]:::head --> a5o(["ŷ"]):::head
    end

    subgraph A6["A6 · CNN + atención · 14.497 parámetros · ELEGIDA"]
        direction LR
        a6i["Ventana<br/>30 × 18"]:::datos --> a6c["Conv 18 → 32<br/>BN · ReLU · Dropout"]:::conv --> a6c2["Conv 32 → 64<br/>BN · ReLU"]:::conv --> a6a["Atención<br/>temporal"]:::att --> a6h["Cabezal"]:::head --> a6o(["ŷ"]):::head
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

**Cómo leer la ablación:** cada par aísla un bloque.

| Par | Qué aísla | RMSE final en validación (mediana) |
|---|---|---|
| A1 → A6 | Atención sobre una CNN | 2,02 → 2,00 |
| A2 → A4 | Atención sobre una LSTM | 2,07 → 2,17 |
| A3 → A5 | Atención sobre CNN + LSTM | 2,60 → 2,18 |
| A1 → A3 | Agregar recurrencia a la CNN | 2,02 → 2,60 |

---

## 4. Módulo de atención temporal

```mermaid
flowchart LR
    H["Secuencia de características<br/>h₁ … h₃₀ · 30 × 64"]:::conv
    WA["Linear 64 → 64"]:::att
    TH["tanh"]:::att
    V["Linear 64 → 1<br/>sin sesgo"]:::att
    SM["softmax<br/>sobre los 30 pasos"]:::att
    A["Pesos αₜ<br/>suman 1"]:::att
    MUL["Suma ponderada<br/>c = Σ αₜ hₜ"]:::att
    C["Contexto c · 64"]:::head
    I["Interpretabilidad<br/>qué instantes pesan más"]:::evalc

    H --> WA --> TH --> V --> SM --> A --> MUL --> C
    H --> MUL
    A -.-> I

    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef att fill:#f8d7e4,stroke:#d55181,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
```

$e_t = v^\top \tanh(W h_t + b)$, $\quad \alpha_t = \mathrm{softmax}(e_t)$, $\quad c = \sum_t \alpha_t h_t$

En A6 los pesos resultaron casi uniformes (≈ 1/30), con picos solo en los bordes de la
ventana: en la práctica, A6 se comporta como un promedio temporal.

---

## 5. Normalizaciones comparadas (etapa 2)

```mermaid
flowchart TD
    S["Sensores X_s de cada registro"]:::datos

    subgraph G["Global · elegida"]
        direction TB
        G1["Una media y desviación<br/>por sensor"]:::prep
        G2["Aprendidas con<br/>los 4 motores de entrenamiento"]:::prep
        G1 --- G2
    end

    subgraph N1A["N1a · por régimen de operación"]
        direction TB
        R1["k-means sobre W<br/>K = 2 por silueta"]:::prep
        R2["R1 crucero alto · 60 %<br/>R2 ascenso y descenso · 40 %"]:::prep
        R3["Media y desviación<br/>por régimen"]:::prep
        R1 --> R2 --> R3
    end

    subgraph N1B["N1b · línea base por motor"]
        direction TB
        B1["Primeros 10 ciclos<br/>del propio motor · sanos"]:::prep
        B2["Media y desviación<br/>por motor"]:::prep
        B1 --> B2
    end

    S --> G
    S --> N1A
    S --> N1B

    G --> RG["RMSE final 2,00<br/>varianza explicada por W: 99 %"]:::evalc
    N1A --> RA["RMSE final 2,02 · empate<br/>varianza explicada por W: 84 %<br/>actúa dentro de cada vuelo"]:::evalc
    N1B --> RB["RMSE final 3,29 · empeora<br/>varianza explicada por W: 98 %<br/>actúa entre motores"]:::evalc

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef prep fill:#c8efe0,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    style G stroke:#0b0b0b,stroke-width:3px
```

Las condiciones de vuelo W se estandarizan globalmente en los tres casos y siguen siendo
entrada del modelo. No se normaliza literalmente por la clase de vuelo `Fc` porque todo
`dev` es `Fc = 3`; las unidades de prueba 14 y 15 (`Fc` 1 y 2) quedarían sin
estadísticas.

---

## 6. Función de pérdida asimétrica (etapa 3)

```mermaid
flowchart LR
    P["Predicción ŷ<br/>y etiqueta y"]:::head
    D["Error<br/>d = ŷ − y"]:::head
    Q{"¿d > 0?<br/>sobreestima"}:::dec
    WA["peso α<br/>error peligroso"]:::att
    WB["peso 1<br/>error conservador"]:::prep
    L["L_α = mean(peso · d²)<br/>estima el expectil τ = 1/(1+α)"]:::head
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

| α | Expectil τ | Sobreestimar 5 ciclos cuesta lo mismo que subestimar |
|---|---|---|
| 1 (= MSE) | 0,50 | 5,0 ciclos |
| 2 · elegido | 0,33 | 7,1 ciclos |
| 4 | 0,20 | 10,0 ciclos |

---

## 7. Validación cruzada dejando un motor fuera (etapa 4)

Cada motor de `dev` se evalúa una vez. El siguiente de la lista se usa para el early
stopping y los 4 restantes para entrenar. La normalización se reaprende en cada fold.

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

| Fold | Evaluación | Validación | Entrenamiento |
|---|---|---|---|
| 1 | 2 | 5 | 10, 16, 18, 20 |
| 2 | 5 | 10 | 2, 16, 18, 20 |
| 3 | 10 | 16 | 2, 5, 18, 20 |
| 4 | 16 | 18 | 2, 5, 10, 20 |
| 5 | 18 | 20 | 2, 5, 10, 16 |
| 6 | 20 | 2 | 5, 10, 16, 18 |

---

## 8. Resultado final en el conjunto de prueba oficial

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

Mediana sobre 5 semillas; la peor semilla de la configuración final (1,97) es mejor que la
mejor de la base (2,47). En la validación cruzada la ventaja no es uniforme: depende de si
el modelo base es optimista o pesimista con cada motor.
