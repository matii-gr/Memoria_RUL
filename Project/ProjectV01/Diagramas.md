# Diagramas del pipeline y la arquitectura — RULv01 (N-CMAPSS DS02)

Ablación del mecanismo de atención temporal sobre una arquitectura híbrida
1D-CNN + LSTM. Las cifras provienen de las salidas de `RULv01.ipynb`.

---

## 1. Pipeline de datos

```mermaid
flowchart TD
    H5[("N-CMAPSS_DS02-006.h5<br/>5,26 M registros dev + 1,25 M test, 1 Hz")]

    subgraph CARGA["Carga"]
        direction TB
        SUB["Submuestreo 1 de cada 10 s<br/>dev: 526.345 · test: 125.375 registros"]
        VAR["Variables de entrada: 18 canales<br/>W (4): alt, Mach, TRA, T2<br/>X_s (14): T24, T30, T48, T50, P15, P2, P21,<br/>P24, Ps30, P40, P50, Nf, Nc, Wf"]
        DESC["Descartadas: X_v (sensores virtuales)<br/>y T (parámetros de salud), no medibles"]
        AUX["Auxiliares A: unit, cycle, Fc, hs"]
        CACHE[("Caché results/cache_step10.npz")]
        SUB --> VAR
        SUB --> AUX
        VAR -.- DESC
        VAR --> CACHE
        AUX --> CACHE
    end

    subgraph SPLIT["Partición por unidad: ningún motor en dos conjuntos"]
        direction LR
        TR["Entrenamiento<br/>unidades 2, 5, 10, 16<br/>360.457 registros"]
        VA["Validación<br/>unidades 18, 20<br/>165.888 registros"]
        TE["Prueba (partición test oficial)<br/>unidades 11, 14, 15<br/>125.375 registros"]
    end

    subgraph PRE["Preprocesamiento sin fuga"]
        direction TB
        SC["StandardScaler<br/>fit SOLO con entrenamiento<br/>transform en val y test"]
        PWL["Etiqueta PwL: RUL recortado a 62 ciclos<br/>mediana del RUL en la transición sano→degradado<br/>(indicador hs) de las unidades de entrenamiento"]
        WIN["Ventanas deslizantes L = 30 (300 s), stride = 10<br/>sin cruzar fronteras de (unidad, ciclo de vuelo)<br/>etiqueta = RUL del último instante"]
        SC --> WIN
        PWL --> WIN
    end

    subgraph TENS["Tensores"]
        direction LR
        XTR["Train X: 35.287 × 30 × 18"]
        XVA["Val X: 16.251 × 30 × 18"]
        XTE["Test X: 12.047 × 30 × 18"]
    end

    DL["DataLoaders · batch 256 · num_workers 0<br/>train con shuffle y generador sembrado"]

    H5 --> SUB
    CACHE --> SPLIT
    TR --> SC
    VA --> SC
    TE --> SC
    WIN --> TENS
    TENS --> DL
```

---

## 2. Arquitectura híbrida 1D-CNN + LSTM con tres modos de agregación

El parámetro `aggregation` es la **única** diferencia entre los tres brazos de la ablación.

```mermaid
flowchart TD
    IN["Entrada<br/>(B, 30, 18)"]
    P1["permute → (B, 18, 30)"]

    subgraph CNN["Extracción local: 1D-CNN"]
        direction TB
        C1["Conv1D 18 → 32 canales<br/>kernel 3, padding 1"]
        BN["BatchNorm1d(32)"]
        RE["ReLU"]
        DO["Dropout 0,2"]
        C1 --> BN --> RE --> DO
    end

    P2["permute → (B, 30, 32)"]

    subgraph REC["Modelado temporal"]
        LSTM["LSTM 32 → 64<br/>secuencia completa h₁ … h₃₀<br/>(B, 30, 64)"]
    end

    subgraph AGG["Agregación temporal: interruptor de ablación"]
        direction LR
        LAST["last<br/>último estado h₃₀<br/>línea base"]
        MEAN["mean<br/>promedio (1/30) Σ hₜ<br/>control"]
        ATT["attention<br/>atención aditiva de Bahdanau<br/>alternativa A4"]
    end

    FEAT["Vector de características<br/>(B, 64)"]

    subgraph HEAD["Cabezal de regresión"]
        direction TB
        L1["Linear 64 → 32 + ReLU"]
        L2["Linear 32 → 1"]
        L1 --> L2
    end

    OUT["RUL estimado ŷ<br/>(B, 1)"]
    W["Pesos αₜ (B, 30)<br/>interpretabilidad, criterio C8"]

    IN --> P1 --> CNN --> P2 --> LSTM
    LSTM --> LAST
    LSTM --> MEAN
    LSTM --> ATT
    LAST --> FEAT
    MEAN --> FEAT
    ATT --> FEAT
    ATT -.-> W
    FEAT --> HEAD --> OUT
```

| Brazo | Agregación | Parámetros | Qué aísla |
|---|---|---|---|
| `last` | último estado $h_T$ | 29.025 | Línea base |
| `mean` | promedio temporal | 29.025 | Efecto de agregar toda la ventana |
| `attention` | atención de Bahdanau | 33.249 (+4.224) | Efecto de la ponderación aprendida |

---

## 3. Detalle del módulo de atención temporal

```mermaid
flowchart LR
    H["Salidas LSTM<br/>h₁ … h₃₀<br/>(B, 30, 64)"]
    WA["Linear 64 → 64"]
    TH["tanh"]
    V["Linear 64 → 1<br/>sin sesgo"]
    E["Energías eₜ<br/>(B, 30, 1)"]
    SM["softmax sobre t"]
    A["Pesos αₜ<br/>Σ αₜ = 1"]
    MUL["Suma ponderada<br/>c = Σ αₜ hₜ"]
    C["Contexto c<br/>(B, 64)"]

    H --> WA --> TH --> V --> E --> SM --> A --> MUL
    H --> MUL
    MUL --> C
```

$e_t = v^\top \tanh(W h_t + b)$, $\quad \alpha_t = \mathrm{softmax}(e_t)$, $\quad c = \sum_t \alpha_t h_t$

---

## 4. Protocolo experimental

```mermaid
flowchart TB
    DATA["Tensores train / val / test<br/>(diagrama 1)"]
    G["3 brazos: last · mean · attention<br/>× 3 semillas (0, 1, 2)<br/>= 9 entrenamientos"]
    T["Adam lr 1e-3, weight decay 1e-4<br/>pérdida MSE: L = mean((ŷ − y)²)<br/>máx. 30 épocas"]
    E["Early stopping: RMSE de validación<br/>paciencia 5, restaura la mejor época"]
    CK[("Checkpoints results/ckpt_*.pt")]
    R["Métricas en validación<br/>media ± σ entre semillas"]
    X["Evaluación única en test<br/>unidades 11, 14, 15"]
    FIG["Figuras: curvas de validación, RMSE en test,<br/>residuos, pesos de atención por régimen,<br/>trayectorias de RUL por unidad"]
    O["Resultado: atención ≈ último estado ≈ promedio<br/>(diferencia ≈ 3 %, dentro del ruido entre semillas)<br/>pesos αₜ no uniformes y dependientes del régimen<br/>~80 % de predicciones sobreestiman en test"]

    DATA --> G --> T --> E --> R --> X --> FIG --> O
    E --> CK
```

### Métricas reportadas

- **RMSE** y **MAE**, en ciclos.
- **Score NASA** asimétrico normalizado por N: `exp(−d/13) − 1` si d < 0, `exp(d/10) − 1` si d ≥ 0, con d = ŷ − y.
- **Sesgo** y **% de sobreestimación**.
- **RMSE por tramo:** crítico (RUL ≤ 26), medio (26–52) y sano (> 52), a partir de los terciles de entrenamiento.
- **RMSE por unidad de prueba:** 11, 14 y 15.
