# Diagramas del pipeline y las arquitecturas — RULv05 (N-CMAPSS DS02)

Diagramas de `RULv05.ipynb`. Todas las cifras provienen de las salidas del notebook y son de
validación (motores 18 y 20), salvo el diagrama 9, que es la prueba oficial de v03. Los
diagramas del catálogo de arquitecturas A1–A6, de la atención frente al Transformer, de la
pérdida asimétrica y de la validación cruzada están en `ProjectV03/Diagramas.md`; los del
agrupamiento de regímenes de operación, en `ProjectV04/Diagramas.md`.

**Notación de formas:** `(B, T, C)` = (ventanas del lote, pasos de tiempo, canales). Con
B = 256 ventanas, T = 30 pasos (5 minutos de vuelo) y C el número de canales. Cada bloque de
las arquitecturas indica la forma del tensor a su salida, para que se vea qué dimensión
cambia: sin pooling, la temporal se mantiene en 30 y solo cambian los canales; con pooling
(diagramas 4 y 5), el tiempo se reduce a 15 o 7.

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

## 1. Visión general de ProjectV05

```mermaid
flowchart LR
    R6["Reunión 06<br/>score, sesgo global,<br/>dropout y pooling"]:::datos
    E0["Etapa 0<br/>Infraestructura y métricas nuevas<br/><i>reproduce v02 y v04 con diferencia 0</i>"]:::prep
    E1["Etapa 1<br/>El sesgo: qué es y cuánto<br/>se sobreestima o subestima"]:::evalc
    E2["Etapa 2<br/>El score NASA:<br/>por qué es tan bajo"]:::evalc
    E3["Etapa 3<br/>Por qué el sesgo global<br/>y correcciones de v04"]:::head
    E4["Etapa 4<br/>Documentación y GitHub"]:::prep

    R6 --> E0 --> E1 --> E2 --> E3 --> E4

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

**Versión de análisis:** no hay experimentos nuevos ni cambia la configuración final, que
sigue siendo A6 (CNN + atención) · normalización global · pérdida asimétrica con α = 2. Se
reanalizan las predicciones guardadas de v02 y v04, y el conjunto de prueba no se consulta.

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

## 4. A6 con y sin pooling, con la forma de cada tensor (v04, etapa 1)

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

## 5. Orden real de las capas y origen del único dropout (etapa 3)

Orden comprobado en v05 recorriendo la red con *hooks* de PyTorch (sección 3.4). El dropout
va **después** del max pooling. Formas en la notación de Conv1d, `(B, canales, tiempo)`.

```mermaid
flowchart LR
    subgraph P2["A6-P2 · orden de ejecución"]
        direction LR
        c1["1 · Conv1d<br/>(B, 32, 30)"]:::conv --> b1["2 · BatchNorm<br/>(B, 32, 30)"]:::conv --> r1["3 · ReLU<br/>(B, 32, 30)"]:::conv --> m1["4 · MaxPool<br/>(B, 32, 15)<br/><i>cambia el tiempo</i>"]:::agg --> d1["5 · Dropout<br/>(B, 32, 15)"]:::agg --> c2["6 · Conv1d<br/>(B, 64, 15)<br/><i>cambia los canales</i>"]:::conv --> b2["7 · BatchNorm · 8 · ReLU<br/>(B, 64, 15)"]:::conv --> m2["9 · MaxPool<br/>(B, 64, 7)"]:::agg
    end

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

```mermaid
flowchart LR
    V1["v01 · red base<br/>Conv → BN → ReLU → Dropout → LSTM"]:::lstm
    V2["v02 · A1 y A6<br/>la LSTM se reemplaza por Conv 32 → 64<br/>sin agregar un segundo dropout"]:::conv
    R{"Un solo dropout por arquitectura<br/>no fue una decisión ajustada<br/>con experimentos"}:::dec

    V1 --> V2 --> R

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

En A2 y A4, que no tienen convolución, el único dropout va después de la LSTM.

---

## 6. Resultado del pooling y decisión (v04, etapa 1; explicación corregida en v05)

```mermaid
flowchart LR
    BIB["Bibliografía<br/>Li, Ding y Sun 2018: no usar pooling<br/>con entradas de baja dimensión<br/><i>ventanas de 30 pasos, como aquí</i>"]:::datos
    EXP["Experimento · validación · 5 semillas<br/>α = 2: A6 1,83 · A6-P1 2,03 · A6-P2 2,03<br/>α = 1: A6 2,00 · A6-P1 1,93 · A6-P2 1,71"]:::evalc
    R{{"Regla con α = 2<br/>(a) mediana menor: no<br/>(b) 4 de 5 semillas: 2 de 5"}}:::dec
    NO["No se adopta<br/>sigue A6 sin pooling"]:::conv
    WHY["Por qué depende de α · corregido en v05<br/>zona final: A6-P2 baja las predicciones<br/>sesgo +0,70 → −0,03 (MSE) · −0,04 → −0,62 (α = 2)<br/>pero el cambio de RMSE² viene tanto o más<br/>de la varianza del error que del sesgo"]:::head

    BIB --> R
    EXP --> R --> NO
    EXP --> WHY

    classDef datos fill:#cde2fb,stroke:#2a78d6,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef head fill:#dcd8f2,stroke:#4a3aa7,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:2px,color:#0b0b0b
    classDef dec fill:#fcfcfb,stroke:#0b0b0b,stroke-width:2px,color:#0b0b0b
```

RMSE en la zona final (RUL ≤ 20), mediana sobre 5 semillas. v04 explicaba la dependencia de α con el
sesgo global (−1,32 y −1,83), que exageraba el efecto; la explicación corregida está en
`RULv05.ipynb`, etapa 3.

---

## 7. Descomposición del sesgo (etapa 1)

```mermaid
flowchart TB
    D["Error de cada ventana<br/>d = predicho − real (ciclos)"]:::datos
    OV["Sobreest. media<br/>promedio de max(d, 0)<br/>final: 0,77 ciclos · 48 % de ventanas"]:::conv
    UN["Subest. media<br/>promedio de max(−d, 0)<br/>final: 0,78 ciclos · 52 % de ventanas"]:::datos
    S["Sesgo = Sobreest. − Subest.<br/>zona final: −0,04"]:::evalc
    M["MAE = Sobreest. + Subest.<br/>zona final: 1,45"]:::evalc
    R["RMSE² = Sesgo² + Desv²<br/>zona final: el sesgo explica el 0,2 %"]:::evalc

    D --> OV
    D --> UN
    OV --> S
    UN --> S
    OV --> M
    UN --> M
    S --> R

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

```mermaid
flowchart LR
    ZF["Zona final · RUL ≤ 20<br/>31 % de las ventanas<br/>sesgo −0,04 · aporte −0,01"]:::evalc
    ZI["Zona intermedia · 20 a 40<br/>29 % de las ventanas<br/>sesgo +0,55 · aporte +0,16"]:::prep
    ZT["Zona temprana · RUL > 40<br/>39 % de las ventanas<br/>sesgo −2,82 · aporte −1,11<br/><i>recorte en 62: se subestima</i>"]:::conv
    G{"Sesgo global −0,93<br/>= suma de los aportes<br/>(fracción × sesgo de la zona)"}:::dec

    ZF --> G
    ZI --> G
    ZT --> G

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

Configuración final (A6, α = 2), mediana sobre 5 semillas. Las medianas de los aportes no
suman exactamente la mediana del global; la suma es exacta en cada semilla. Por motor, el
sesgo final de −0,04 es el promedio del motor 18 (−1,04) y el 20 (+1,07).

---

## 8. Por qué el score NASA es tan bajo (etapa 2)

```mermaid
flowchart TB
    SD["s(d) por ventana<br/>e^(d/10) − 1 si sobreestima<br/>e^(−d/13) − 1 si subestima"]:::datos
    LIN["Errores pequeños: casi una recta<br/>s ≈ d/10 o −d/13<br/>2 ciclos aportan 0,22 o 0,17"]:::prep
    MEAN["Este trabajo: PROMEDIO por ventana<br/>final: 0,14<br/>= sobreestimar 1,3 ciclos o subestimar 1,7"]:::evalc
    SUM["Bibliografía: SUMA<br/>Li, Ding y Sun 2018: 273,7 en FD001<br/>Li, Wang y Li 2022, DS02: 5,78 × 10⁵"]:::agg
    ASY{"Asimetría: subestimar cuesta 0,77 veces<br/>lo que cuesta sobreestimar<br/>score ≈ MAE asimétrico (Spearman 0,99)"}:::dec
    P2["A6-P2 con α = 2<br/>RMSE 1,83 → 2,03 · score 0,14 → 0,15<br/><i>cambia sobreestimación por subestimación</i>"]:::conv

    SD --> LIN --> MEAN
    SD -.->|"otra convención"| SUM
    SD --> ASY --> P2

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

El score va de 0 a ∞; no tiene un máximo de 100. Con la suma, las predicciones de este
trabajo darían unos 700 en la zona final: los valores no son comparables con los publicados.

---

## 9. Resultado en el conjunto de prueba oficial (v03, sin cambios en v04 ni v05)

```mermaid
flowchart LR
    B["Base · v01<br/>A3 + global + MSE<br/>RMSE final 2,97<br/>sobreest. final 86 % · sesgo global +3,00"]:::agg
    M["Cambio de arquitectura<br/>A6 + global + MSE<br/>RMSE final 2,16<br/>sobreest. final 82 % · sesgo global +1,24"]:::conv
    F["Configuración final<br/>A6 + global + α = 2<br/>RMSE final 1,78<br/>sobreest. final 67 % · sesgo global +0,19"]:::evalc

    B -->|"arquitectura<br/>−0,82"| M -->|"pérdida asimétrica<br/>−0,38"| F

    classDef agg fill:#ecebe6,stroke:#898781,stroke-width:2px,color:#0b0b0b
    classDef conv fill:#fbd9c9,stroke:#eb6834,stroke-width:2px,color:#0b0b0b
    classDef evalc fill:#cce6cc,stroke:#008300,stroke-width:3px,color:#0b0b0b
```

Mediana sobre 5 semillas. Evaluado una sola vez en v02 y presentado de nuevo en v03; v04 y
v05 no modificaron la configuración final ni volvieron a usar el conjunto de prueba. El
sesgo es el **global** (todas las ventanas de prueba), no el de la zona final; el porcentaje
de ventanas sobreestimadas sí es de la zona final, y su 67 % es, muy probablemente, el "67
sobre 100" mencionado en la Reunión 06. En la validación cruzada la ventaja no es uniforme: depende de si el
modelo base es optimista o pesimista con cada motor, lo que a su vez se relaciona con la vida
total del motor.
