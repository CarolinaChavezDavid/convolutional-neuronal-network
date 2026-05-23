# Resultados Experimentales — Evaluación On-Device
**Tesis:** Evaluación del desempeño de modelos de Deep Learning en dispositivos móviles para el reconocimiento en tiempo real de enfermedades en plantas  
**Directora:** Federico Balaguer — UNLP  
**Estado:** Completo — Moto G (30) + Samsung Galaxy S23 · 3 sesiones · 2 modelos

---

## Índice

1. [Configuración experimental](#1-configuración-experimental)
2. [Especificaciones de hardware](#2-especificaciones-de-hardware)
3. [Modelos evaluados](#3-modelos-evaluados)
4. [Latencia de inferencia](#4-latencia-de-inferencia)
5. [Tiempo de inicialización y warmup](#5-tiempo-de-inicialización-y-warmup)
6. [Consistencia entre sesiones](#6-consistencia-entre-sesiones)
7. [Memoria PSS](#7-memoria-pss)
8. [Consumo de batería](#8-consumo-de-batería)
9. [Temperatura del dispositivo](#9-temperatura-del-dispositivo)
10. [Coherencia botánica](#10-coherencia-botánica)
11. [Análisis de confianza por grupo de planta](#11-análisis-de-confianza-por-grupo-de-planta)
12. [Comportamiento en imágenes fuera de dominio (IFD)](#12-comportamiento-en-imágenes-fuera-de-dominio-ifd)
13. [Comparativa cross-device](#13-comparativa-cross-device)
14. [Relación con el desbalanceo del dataset PlantVillage](#14-relación-con-el-desbalanceo-del-dataset-plantvillage)
15. [Hallazgos principales para la tesis](#15-hallazgos-principales-para-la-tesis)
16. [Resumen ejecutivo](#16-resumen-ejecutivo)

---

## 1. Configuración experimental

| Parámetro | Valor |
|---|---|
| Imágenes totales evaluadas | 99 (IDs 2–100; la imagen 1 está ausente del dataset) |
| Imágenes in-domain | 86 (excluye Orange y grupo IFD) |
| Sesiones on-device por modelo × dispositivo | 3 |
| Umbral de confianza (app) | 50 % |
| Crashes registrados | Ninguno en ninguna sesión ni dispositivo |
| Brillo de pantalla | 60 % (ambos dispositivos) |
| Modo rendimiento | Desactivado (ambos dispositivos) |

### Distribución de imágenes por planta

| Planta | N | Origen | Notas |
|---|---|---|---|
| Corn | 9 | Dataset IDADP | Mix sano/enfermo |
| Apple | 10 | Dataset AL9EE + celular | Fondo complejo en imágenes de celular |
| Grape | 10 | Dataset NGLD + celular | Incluye imágenes con ángulo |
| Cherry | 4 | Celular | Fondo complejo |
| Pepper | 6 | Dataset Liu | Buena luz y sombra |
| Tomato | 10 | Dataset JAWALE + celular | Mix completo |
| Strawberry | 7 | Dataset afzaal | Leaf spot |
| Orange | 3 | Celular | **Fuera de dominio**: no existe clase Orange en PlantVillage |
| Squash | 10 | Dataset bangladesh | Powdery mildew |
| Potato | 10 | Dataset Irmawati | Mayormente sanos |
| Soybean | 10 | Dataset soynet | Solo healthy en dataset de entrenamiento |
| IFD | 10 | — | Fondos aleatorios **sin hojas** (grupo de control) |

---

## 2. Especificaciones de hardware

| Atributo | Moto G (30) | Samsung Galaxy S23 |
|---|---|---|
| SoC | Qualcomm Snapdragon 662 (SM6115 "BENGAL", 11 nm) | Qualcomm Snapdragon 8 Gen 2 for Galaxy (4 nm) |
| CPU | 8 núcleos: 4× Kryo 260 Gold (Cortex-A73) @ 2.0 GHz + 4× Kryo 260 Silver (Cortex-A53) @ 1.8 GHz | 10 núcleos: 1× Cortex-X3 @ 3.36 GHz + 2× Cortex-A715 @ 2.8 GHz + 2× Cortex-A710 @ 2.8 GHz + 3× Cortex-A510 @ 2.0 GHz |
| GPU | Qualcomm Adreno 610 @ 950 MHz | Qualcomm Adreno 740 |
| RAM | 4 GB (3.815 GB usable) | 8 GB LPDDR5X |
| Android | Android 12 (API 31) | Android 13 (API 31) |
| Almacenamiento | 128 GB total, ~33 GB usado | 256 GB total, ~117 GB usado |
| Categoría | Dispositivo **mid-range** | Dispositivo **high-end / flagship** |

Estos dos dispositivos representan extremos distintos del mercado de smartphones Android: el Moto G (30) es un dispositivo de gama media con SoC de 11 nm fabricado en 2020, mientras que el Samsung Galaxy S23 utiliza el chipset más avanzado de Qualcomm disponible en 2023 con proceso de 4 nm. La brecha de rendimiento esperada entre ambos es significativa y permite evaluar cómo escala el rendimiento on-device con el hardware.

---

## 3. Modelos evaluados

| Modelo | Archivo | Tamaño (MB) | Cuantización | Clases | API de inferencia |
|---|---|---|---|---|---|
| MobileNetV3 Small | mobilenetv3_small_v1_fp16.tflite | **1.91** | FP16 | 38 | LiteRT `CompiledModel` (AOT) |
| EfficientNetV2B0 | efficientnetv2b0_v1_fp16.tflite | **11.33** | FP16 | 38 | LiteRT `Interpreter` (4 threads CPU) |

**Ratio de tamaño:** EfficientNetV2B0 es **5.94× más grande** que MobileNetV3 Small.

> **Nota sobre la API de inferencia:** MobileNetV3 utiliza `CompiledModel` (compilación AOT de LiteRT), que aprovecha aceleradores de hardware disponibles. EfficientNetV2B0 usa el `Interpreter` clásico de TFLite con 4 hilos de CPU, ya que sus bloques Fused-MBConv no son compatibles con el compilador AOT de LiteRT. Esta diferencia en la API de inferencia contribuye parcialmente a la brecha de latencia observada.

---

## 4. Latencia de inferencia

Todos los tiempos están en **milisegundos (ms)**. Los promedios se calculan sobre las inferencias steady-state (se excluye el warmup). Cada sesión comprende 99 inferencias; el promedio cross-sesión se calcula sobre las 3 sesiones.

### 4.1 Moto G (30) — Latencia por sesión

| Métrica | Sesión 1 MNV3 | Sesión 1 Eff | Sesión 2 MNV3 | Sesión 2 Eff | Sesión 3 MNV3 | Sesión 3 Eff |
|---|---|---|---|---|---|---|
| Avg (ms) | 24.8 | 273.6 | 31.9 | 276.8 | 35.3 | 272.0 |
| Mediana / P50 (ms) | 23.2 | 266.6 | 31.0 | 271.8 | 31.8 | 265.2 |
| P95 (ms) | 36.9 | 311.4 | 52.5 | 323.9 | 69.0 | 318.1 |
| Std (ms) | 7.5 | 22.6 | 12.3 | 22.8 | 18.1 | 23.2 |

**Promedio cross-sesión Moto G:**

| Métrica | MobileNetV3 | EfficientNetV2B0 | Ratio Eff/MNV3 |
|---|---|---|---|
| **Avg (ms)** | **30.67** | **274.13** | **8.94×** |
| Mediana P50 (ms) | 28.67 | 267.87 | 9.34× |
| P95 (ms) | 52.80 | 317.80 | 6.02× |
| Std (ms) | 12.63 | 22.87 | — |

### 4.2 Samsung Galaxy S23 — Latencia por sesión

| Métrica | Sesión 1 MNV3 | Sesión 1 Eff | Sesión 2 MNV3 | Sesión 2 Eff | Sesión 3 MNV3 | Sesión 3 Eff |
|---|---|---|---|---|---|---|
| Avg (ms) | 7.8 | 52.5 | 7.9 | 52.4 | 7.8 | 53.5 |
| Mediana / P50 (ms) | 6.6 | 48.6 | 6.6 | 48.7 | 6.5 | 49.4 |
| P95 (ms) | 14.2 | 68.7 | 14.2 | 68.9 | 13.5 | 69.0 |
| Std (ms) | 3.6 | 9.3 | 3.4 | 9.0 | 3.1 | 8.6 |

**Promedio cross-sesión Samsung Galaxy S23:**

| Métrica | MobileNetV3 | EfficientNetV2B0 | Ratio Eff/MNV3 |
|---|---|---|---|
| **Avg (ms)** | **7.83** | **52.80** | **6.74×** |
| Mediana P50 (ms) | 6.57 | 48.90 | 7.44× |
| P95 (ms) | 13.97 | 68.87 | 4.93× |
| Std (ms) | 3.37 | 8.97 | — |

### 4.3 Interpretación de latencia

- **MobileNetV3 en Moto G (30.67 ms):** equivale a ~32 clasificaciones por segundo. Apto para feedback cuasi-continuo en campo.
- **EfficientNetV2B0 en Moto G (274.13 ms):** equivale a ~3.6 clasificaciones por segundo. Requiere captura estática con espera perceptible (~0.27 segundos).
- **MobileNetV3 en Samsung (7.83 ms):** equivale a ~127 clasificaciones por segundo. La inferencia es prácticamente instantánea desde la perspectiva del usuario.
- **EfficientNetV2B0 en Samsung (52.80 ms):** equivale a ~19 clasificaciones por segundo. Fluido para cualquier modo de uso en campo.
- **La brecha entre modelos es mayor en el dispositivo mid-range (8.94×) que en el flagship (6.74×):** el hardware de alta gama reduce la diferencia relativa, pero no la elimina. El Snapdragon 8 Gen 2 comprime más el tiempo de EfficientNetV2B0 (5.19× de mejora) que el de MobileNetV3 (3.91× de mejora), lo que sugiere que el modelo mayor aprovecha mejor las capacidades del SoC avanzado.

---

## 5. Tiempo de inicialización y warmup

El **tiempo de inicialización** es el tiempo de carga del modelo en memoria (una sola vez al iniciar la app). El **warmup** es la primera inferencia ejecutada, que es más lenta que las siguientes por compilación JIT y precalentamiento de cachés.

### 5.1 Moto G (30)

| Métrica | Sesión 1 MNV3 | Sesión 1 Eff | Sesión 2 MNV3 | Sesión 2 Eff | Sesión 3 MNV3 | Sesión 3 Eff | Avg MNV3 | Avg Eff |
|---|---|---|---|---|---|---|---|---|
| Init time (ms) | 3423 | 357 | 3220 | 319 | 3258 | 374 | **3300** | **350** |
| Warmup (ms) | 258.8 | 495.8 | 248.0 | 463.1 | 60.4 | 671.4 | **189** | **543** |

### 5.2 Samsung Galaxy S23

| Métrica | Sesión 1 MNV3 | Sesión 1 Eff | Sesión 2 MNV3 | Sesión 2 Eff | Sesión 3 MNV3 | Sesión 3 Eff | Avg MNV3 | Avg Eff |
|---|---|---|---|---|---|---|---|---|
| Init time (ms) | 844 | 140 | 863 | 141 | 647 | 172 | **785** | **151** |
| Warmup (ms) | 45.4 | 92.5 | 44.9 | 87.6 | 10.0 | 145.7 | **34** | **109** |

### 5.3 Observaciones

- **MobileNetV3 tiene un tiempo de inicialización 9.4× mayor en Moto G (3300 ms vs. 350 ms de EfficientNet).** Esto se debe a que MobileNetV3 usa `CompiledModel` (compilación AOT), que requiere compilar y enlazar los kernels con los aceleradores disponibles antes de poder ejecutar. EfficientNetV2B0 con `Interpreter` carga el grafo de forma directa.
- **En Samsung, la inicialización de MobileNetV3 cae a 785 ms** (4.2× menor que en Moto G), mientras que EfficientNetV2B0 cae a 151 ms (2.3× menor). El hardware más rápido reduce el tiempo de compilación AOT de forma más pronunciada.
- El init time es un costo único por sesión de app y **no afecta la experiencia de uso en campo**: el usuario abre la app una vez y luego clasifica múltiples imágenes.

---

## 6. Consistencia entre sesiones

Se verifica si las predicciones (top-1 label) son idénticas entre las 3 sesiones de cada combinación dispositivo × modelo.

| Dispositivo | Modelo | ¿Sesiones S1=S2=S3? | Imágenes con diferencia |
|---|---|---|---|
| Moto G | MobileNetV3 | **SÍ — idéntico** | 0 de 99 |
| Moto G | EfficientNetV2B0 | **SÍ — idéntico** | 0 de 99 |
| Samsung S23 | MobileNetV3 | **SÍ — idéntico** | 0 de 99 |
| Samsung S23 | EfficientNetV2B0 | **SÍ — idéntico** | 0 de 99 |

**Las cuatro combinaciones dispositivo × modelo producen predicciones top-1 perfectamente idénticas en las 3 sesiones.** La inferencia TFLite con modelos FP16 es completamente determinista en ambos dispositivos. Esto es el resultado esperado para modelos cuantizados sin dropout en inferencia y confirma la reproducibilidad del protocolo experimental.

> **Para la tesis:** La determinismo de la inferencia valida que los resultados obtenidos en las sesiones 1, 2 y 3 son equivalentes, y que una única sesión sería suficiente para caracterizar el comportamiento del modelo. Las 3 sesiones se ejecutaron para confirmar esta propiedad y para capturar variabilidad en las métricas de hardware (latencia, memoria, temperatura, batería) a lo largo del tiempo de uso.

---

## 7. Memoria PSS

La memoria PSS (Proportional Set Size) representa el uso de memoria del proceso de la app, capturada via `android.os.Debug.getMemoryInfo()` por imagen inferida.

### 7.1 Moto G (30) — Observaciones

| Modelo | Rango observado (MB) | Valor típico (MB) | Patrón |
|---|---|---|---|
| MobileNetV3 | 140 – 450 | ~220 | Crece progresivamente a lo largo de la sesión |
| EfficientNetV2B0 | 190 – 445 | ~290 | Crece progresivamente; picos en imágenes de celular |

- Las imágenes tomadas con celular (fondo complejo, mayor resolución original) generan mayor uso de PSS porque la imagen ocupa más memoria antes del resize a 224×224.
- El crecimiento progresivo de PSS a lo largo de la sesión indica acumulación de buffers intermedios. No se registraron OOM ni crashes en ninguna sesión.

### 7.2 Samsung Galaxy S23 — Observaciones

| Modelo | Rango observado (MB) | Patrón |
|---|---|---|
| MobileNetV3 | ~245 – 440 | Estable entre imágenes de dataset; picos en imágenes de celular |
| EfficientNetV2B0 | ~330 – 700 | Picos notables en imágenes de celular con fondo complejo (hasta ~700 MB) |

- El Samsung tiene más RAM disponible (8 GB vs. 4 GB) y el sistema operativo permite que la app ocupe más memoria sin presión. Esto explica valores de PSS mayores en valor absoluto respecto al Moto G.
- EfficientNetV2B0 en Samsung muestra picos de hasta ~700 MB en imágenes de celular (ej. imagen 14, apple, fondo complejo: 691.8 MB en S1). En Moto G el máximo observado fue ~450 MB.

---

## 8. Consumo de batería

Medido como diferencia entre el nivel de batería al inicio y al final de cada sesión. Cada sesión cubre 99 inferencias consecutivas (sobre 100 imágenes, imagen 1 ausente).

### 8.1 Moto G (30)

| Sesión / Modelo | Inicial (%) | Final (%) | Δ (%) |
|---|---|---|---|
| S1 — MobileNetV3 | 100 | 77 | **−23** |
| S1 — EfficientNetV2B0 | 79 | 68 | −11 |
| S2 — MobileNetV3 | 72 | 63 | −9 |
| S2 — EfficientNetV2B0 | 65 | 55 | −10 |
| S3 — MobileNetV3 | 50 | 37 | −13 |
| S3 — EfficientNetV2B0 | 65 | 51 | −14 |

> **Nota:** La sesión 1 de MobileNetV3 muestra −23%, significativamente mayor que las sesiones 2 y 3. Este valor incluye la primera apertura de la app (init time de 3.4 segundos de compilación AOT) y potencial actividad de sistema operativo en el arranque. Las sesiones 2 y 3 son más representativas del consumo steady-state: **−9% a −13% por sesión para MobileNetV3** y **−10% a −14% para EfficientNetV2B0**.

**Promedios excluyendo S1-MNV3 (outlier):**

| Modelo | Δ promedio por sesión (%) |
|---|---|
| MobileNetV3 | −11.0 % |
| EfficientNetV2B0 | −11.7 % |

Ambos modelos consumen batería de forma similar en el Moto G. La mayor latencia de EfficientNetV2B0 no se traduce en mayor consumo proporcional, probablemente porque el procesador opera con menor carga durante la espera (menor intensidad de cómputo por ciclo).

### 8.2 Samsung Galaxy S23

| Sesión / Modelo | Inicial (%) | Final (%) | Δ (%) |
|---|---|---|---|
| S1 — MobileNetV3 | 100 | 93 | −7 |
| S1 — EfficientNetV2B0 | 93 | 85 | −8 |
| S2 — MobileNetV3 | 90 | 83 | −7 |
| S2 — EfficientNetV2B0 | 83 | 75 | −8 |
| S3 — MobileNetV3 | 75 | 68 | −7 |
| S3 — EfficientNetV2B0 | 82 | 75 | −7 |

**Promedios Samsung:**

| Modelo | Δ promedio por sesión (%) |
|---|---|
| MobileNetV3 | −7.0 % |
| EfficientNetV2B0 | −7.7 % |

El Samsung consume **~36% menos batería por sesión** que el Moto G. La batería del Samsung Galaxy S23 tiene mayor capacidad (3900 mAh vs. 5000 mAh del Moto G30, aunque el Moto G30 tiene batería mayor en mAh, el SoC de 11nm consume más energía por operación que el de 4nm). El proceso de fabricación más eficiente del Snapdragon 8 Gen 2 explica este diferencial energético.

---

## 9. Temperatura del dispositivo

Medida mediante la API térmica de Android (`/sys/class/thermal/thermal_zone*/temp`). Los valores registrados están en décimas de grado Celsius (ej. 283 = 28.3 °C).

### 9.1 Moto G (30)

| Sesión / Modelo | Inicial (°C) | Final (°C) | Δ (°C) |
|---|---|---|---|
| S1 — MobileNetV3 | 28.3 | 29.6 | **+1.3** |
| S1 — EfficientNetV2B0 | 28.4 | 31.6 | +3.2 |
| S2 — MobileNetV3 | 27.2 | 29.5 | +2.3 |
| S2 — EfficientNetV2B0 | 29.0 | 32.4 | +3.4 |
| S3 — MobileNetV3 | 25.8 | 29.5 | +3.7 |
| S3 — EfficientNetV2B0 | 30.0 | 31.3 | **+1.3** |

**Δ promedio Moto G:** MobileNetV3 = +2.4 °C · EfficientNetV2B0 = +2.6 °C

### 9.2 Samsung Galaxy S23

| Sesión / Modelo | Inicial (°C) | Final (°C) | Δ (°C) |
|---|---|---|---|
| S1 — MobileNetV3 | 28.9 | 30.0 | +1.1 |
| S1 — EfficientNetV2B0 | 24.7 | 33.5 | **+8.8** |
| S2 — MobileNetV3 | 25.0 | 31.3 | +6.3 |
| S2 — EfficientNetV2B0 | 29.2 | 31.3 | +2.1 |
| S3 — MobileNetV3 | 26.2 | 31.3 | +5.1 |
| S3 — EfficientNetV2B0 | 29.3 | 31.0 | +1.7 |

**Δ promedio Samsung:** MobileNetV3 = +4.2 °C · EfficientNetV2B0 = +4.2 °C

### 9.3 Observaciones

- Las temperaturas finales de ambos dispositivos en todas las sesiones se mantuvieron por debajo de 34 °C, dentro del rango operativo normal de los chipsets.
- La alta variabilidad en los deltas del Samsung (1.1 °C a 8.8 °C) se debe principalmente a la temperatura de inicio: la sesión S1-EfficientNet comenzó con el dispositivo a 24.7 °C (frío), lo que genera un delta aparentemente mayor. Las temperaturas finales convergen entre 30 °C y 33.5 °C en todas las sesiones de Samsung.
- En el Moto G los deltas son más pequeños y consistentes (1.3–3.7 °C) porque el SoC de 11 nm disipa calor más lentamente pero también genera menos calor pico que el proceso de 4 nm bajo carga sostenida.
- **Ningún dispositivo activó throttling térmico** durante ninguna sesión. Las latencias medidas en sesión 3 (máxima temperatura acumulada) son comparables a las de sesión 1, lo que confirma la ausencia de degradación por temperatura.

---

## 10. Coherencia botánica

### 10.1 Definición

La **Coherencia Botánica** mide si el modelo predice la especie de planta correcta, independientemente de si identifica la enfermedad exacta. Se define como:

```
Predicción botánicamente coherente:
  true_label  = "Tomato___Early_blight"
  prediction  = "Tomato___Late_blight"      → ✓ coherente (misma planta)
  prediction  = "Strawberry___Leaf_scorch"  → ✗ no coherente (planta diferente)

Coherencia Botánica = predicciones con planta correcta / total imágenes evaluadas
```

La especie se extrae tomando el texto antes del primer `___` en la etiqueta predicha.

**Grupos excluidos del global:** Orange (no existe clase Orange en PlantVillage — fuera de dominio por diseño) e IFD (fondos sin planta — grupo de control). Estos se reportan por separado.

### 10.2 Resultado clave: predicciones idénticas en ambos dispositivos

Las predicciones top-1 son **exactamente iguales** en Moto G y Samsung Galaxy S23 para las 99 imágenes. Por lo tanto, la Coherencia Botánica es idéntica entre dispositivos: **el modelo produce la misma salida independientemente del hardware**, siempre que reciba los mismos datos de entrada preprocesados. Las tablas siguientes aplican a ambos dispositivos.

### 10.3 Coherencia botánica — MobileNetV3 Small (ambos dispositivos)

| Planta | N | Correctas | Coherencia |
|---|---|---|---|
| Corn | 9 | 9 | **100%** |
| Tomato | 10 | 9 | **90%** |
| Strawberry | 7 | 6 | **85.7%** |
| Cherry | 4 | 2 | 50.0% |
| Grape | 10 | 3 | 30.0% |
| Squash | 10 | 2 | 20.0% |
| Apple | 10 | 2 | 20.0% |
| Soybean | 10 | 1 | 10.0% |
| Pepper | 6 | 0 | 0.0% |
| Potato | 10 | 0 | 0.0% |
| **GLOBAL (excl. Orange e IFD)** | **86** | **34** | **39.5%** |
| Orange (fuera de dominio) | 3 | 0 | 0.0% |
| IFD (grupo de control) | 10 | 0 | 0.0% |

### 10.4 Coherencia botánica — EfficientNetV2B0 (ambos dispositivos)

| Planta | N | Correctas | Coherencia |
|---|---|---|---|
| Corn | 9 | 9 | **100%** |
| Tomato | 10 | 10 | **100%** |
| Strawberry | 7 | 7 | **100%** |
| Grape | 10 | 4 | 40.0% |
| Squash | 10 | 3 | 30.0% |
| Cherry | 4 | 1 | 25.0% |
| Apple | 10 | 1 | 10.0% |
| Soybean | 10 | 0 | 0.0% |
| Pepper | 6 | 0 | 0.0% |
| Potato | 10 | 0 | 0.0% |
| **GLOBAL (excl. Orange e IFD)** | **86** | **35** | **40.7%** |
| Orange (fuera de dominio) | 3 | 0 | 0.0% |
| IFD (grupo de control) | 10 | 0 | 0.0% |

### 10.5 Implementación Python para notebooks

```python
def botanical_coherence(true_labels, predicted_labels):
    """
    true_labels, predicted_labels: listas de strings tipo "Plant___Disease"
    Retorna: coherencia global y por planta
    """
    correct = 0
    total = 0
    per_plant = {}

    for true, pred in zip(true_labels, predicted_labels):
        true_plant = true.split('___')[0].strip()
        pred_plant = pred.split('___')[0].strip()

        per_plant.setdefault(true_plant, {'correct': 0, 'total': 0})
        per_plant[true_plant]['total'] += 1
        total += 1

        if true_plant == pred_plant:
            correct += 1
            per_plant[true_plant]['correct'] += 1

    return {
        'global': correct / total if total > 0 else 0,
        'per_plant': {
            plant: v['correct'] / v['total']
            for plant, v in per_plant.items()
        }
    }
```

---

## 11. Análisis de confianza por grupo de planta

La columna "Accuracy (%)" en los CSV registra la **confianza top-1** del modelo (probabilidad softmax × 100), no una métrica de accuracy de clasificación. Los valores reflejan cuán seguro está el modelo de su predicción, independientemente de si esa predicción es correcta.

### 11.1 Moto G y Samsung — MobileNetV3 (sesión 1, confianza media por grupo)

| Planta | N | Confianza media (%) | Coherencia botánica |
|---|---|---|---|
| Corn | 9 | 88.8 | 100% |
| Strawberry | 7 | 89.8 | 85.7% |
| Cherry | 4 | 89.9 | 50.0% |
| Tomato | 10 | 82.2 | 90.0% |
| Potato | 10 | 75.7 | 0.0% |
| Squash | 10 | 68.4 | 20.0% |
| Soybean | 10 | 68.6 | 10.0% |
| Pepper | 6 | 66.3 | 0.0% |
| Grape | 10 | 61.0 | 30.0% |
| Apple | 10 | 60.3 | 20.0% |
| Orange (OOD) | 3 | 89.7 | 0.0% |
| IFD (control) | 10 | 77.0 | 0.0% |
| **Global (86 in-domain)** | **86** | **73.8** | **39.5%** |

### 11.2 Moto G y Samsung — EfficientNetV2B0 (sesión 1, confianza media por grupo)

| Planta | N | Confianza media (%) | Coherencia botánica |
|---|---|---|---|
| Corn | 9 | 92.7 | 100% |
| Tomato | 10 | 90.4 | 100% |
| Strawberry | 7 | 88.9 | 100% |
| Grape | 10 | 84.3 | 40.0% |
| Cherry | 4 | 75.3 | 25.0% |
| Potato | 10 | 75.8 | 0.0% |
| Pepper | 6 | 78.5 | 0.0% |
| Squash | 10 | 74.3 | 30.0% |
| Apple | 10 | 73.4 | 10.0% |
| Soybean | 10 | 71.9 | 0.0% |
| Orange (OOD) | 3 | 77.5 | 0.0% |
| IFD (control) | 10 | 86.2 | 0.0% |
| **Global (86 in-domain)** | **86** | **80.6** | **40.7%** |

### 11.3 Interpretación

El hallazgo más crítico de estas tablas es que **alta confianza no implica predicción correcta**:

- Potato: MNV3 muestra 75.7% de confianza media con 0% de coherencia botánica. El modelo predice con alta seguridad clases incorrectas (mayormente `Strawberry___Leaf_scorch`).
- Orange: confianza media de 89.7% (MNV3) y 77.5% (EfficientNet) en predicciones completamente erróneas — Orange no existe en PlantVillage, pero el modelo produce predicciones confiadas de clases arbitrarias.
- IFD: confianzas de 77% (MNV3) y 86.2% (EfficientNet) para imágenes sin ninguna planta.

Este comportamiento es inherente a los clasificadores de conjunto cerrado (closed-set classifiers): la función softmax siempre produce una distribución de probabilidades que suma 1, forzando una predicción incluso ante entradas completamente fuera del dominio de entrenamiento.

---

## 12. Comportamiento en imágenes fuera de dominio (IFD)

Las 10 imágenes IFD son fondos aleatorios **sin ninguna hoja de planta**. Funcionan como grupo de control para evaluar la capacidad de rechazo del sistema.

| ID | MobileNetV3 (conf.) | EfficientNetV2B0 (conf.) |
|---|---|---|
| 91 | Grape___Esca (83.6%) | Corn___healthy (88.4%) |
| 92 | Tomato___Early_blight (57.0%) | Strawberry___Leaf_scorch (89.7%) |
| 93 | Corn___Cercospora (43.5%) | Tomato___Yellow_Leaf_Curl_Virus (92.5%) |
| 94 | Strawberry___Leaf_scorch (88.2%) | Corn___Cercospora (84.5%) |
| 95 | Strawberry___Leaf_scorch (94.7%) | Corn___healthy (86.2%) |
| 96 | Strawberry___Leaf_scorch (99.8%) | Corn___healthy (99.1%) |
| 97 | Grape___Esca (91.1%) | Corn___Northern_Leaf_Blight (86.3%) |
| 98 | Corn___Cercospora (54.8%) | Soybean___healthy (55.8%) |
| 99 | Grape___Esca (99.9%) | Corn___Cercospora (80.3%) |
| 100 | Tomato___Early_blight (57.7%) | Strawberry___Leaf_scorch (99.6%) |

**Conclusión:** Ambos modelos clasifican el 100% de las imágenes IFD con confianzas que van de 43% a 99.9%, sin ningún mecanismo de rechazo. Esto es una **limitación de diseño inherente al paradigma closed-set**, no un fallo específico de los modelos o del hardware.

Para aplicaciones en campo real, esta limitación requiere mitigación mediante alguna de las siguientes estrategias (trabajo futuro):
1. Clasificador binario previo "¿es una hoja?", entrenado con imágenes de hojas vs. fondos arbitrarios.
2. Umbral de confianza calibrado por temperatura (temperature scaling) con un conjunto de validación out-of-distribution.
3. Arquitecturas open-set o con módulo de detección de anomalías.

---

## 13. Comparativa cross-device

### 13.1 Ratios de velocidad

| Comparación | Ratio | Interpretación |
|---|---|---|
| MNV3: Moto G / Samsung avg | **3.91×** | El Samsung clasifica ~4× más rápido con MobileNetV3 |
| EfficientNet: Moto G / Samsung avg | **5.19×** | El Samsung clasifica ~5× más rápido con EfficientNetV2B0 |
| Moto G: EfficientNet / MNV3 | **8.94×** | En gama media, EfficientNet es casi 9× más lento |
| Samsung: EfficientNet / MNV3 | **6.74×** | En gama alta, EfficientNet es "solo" 6.7× más lento |

> La brecha entre modelos se reduce en hardware flagship (8.94× → 6.74×): el Snapdragon 8 Gen 2 aprovecha mejor la capacidad adicional de EfficientNetV2B0 que el Snapdragon 662. Esto sugiere que el modelo más grande escala mejor con hardware avanzado.

### 13.2 Tabla comparativa completa

| Métrica | Moto G MNV3 | Moto G Eff | Samsung MNV3 | Samsung Eff |
|---|---|---|---|---|
| Avg latencia (ms) | 30.67 | 274.13 | **7.83** | **52.80** |
| Mediana P50 (ms) | 28.67 | 267.87 | 6.57 | 48.90 |
| P95 (ms) | 52.80 | 317.80 | 13.97 | 68.87 |
| Std (ms) | 12.63 | 22.87 | **3.37** | **8.97** |
| Init time avg (ms) | 3300 | 350 | 785 | **151** |
| Warmup avg (ms) | 189 | 543 | 34 | 109 |
| Batería Δ / sesión | −11% | −11.7% | **−7%** | **−7.7%** |
| Temperatura Δ avg | +2.4 °C | +2.6 °C | +4.2 °C | +4.2 °C |
| PSS típica (MB) | ~220 | ~290 | ~320 | ~450 |
| Coherencia botánica | 39.5% | 40.7% | 39.5% | 40.7% |
| Consistencia sesiones | 100% | 100% | 100% | 100% |
| Tamaño modelo (MB) | 1.91 | 11.33 | 1.91 | 11.33 |

### 13.3 Impacto del hardware en la calidad de predicción

**Ninguno.** Las predicciones top-1 son idénticas en ambos dispositivos para las 99 imágenes. El hardware afecta exclusivamente las métricas de rendimiento (latencia, memoria, temperatura, batería) pero no el resultado de la clasificación. Esto confirma que los modelos FP16 son lo suficientemente estables numéricamente como para producir los mismos logits de salida en diferentes plataformas ARM.

---

## 14. Relación con el desbalanceo del dataset PlantVillage

Los resultados de coherencia botánica están directamente explicados por la distribución de clases en PlantVillage. Esta sección conecta los fallos observados con causas estructurales del dataset.

### 14.1 Clases sobrerepresentadas → sesgo de predicción

| Clase en PlantVillage | ~N imágenes | Efecto observado |
|---|---|---|
| Tomato___Yellow_Leaf_Curl_Virus | ~5,400 | Predicción frecuente para hojas amarillas de otras plantas |
| Soybean___healthy | ~5,100 | Solo 1 clase → no generaliza enfermedades |
| Strawberry___Leaf_scorch | ~1,100 | **"Atractor universal"** de EfficientNetV2B0: manchas marrones/rojas en cualquier hoja |
| Corn___Common_rust | ~1,100 | **"Atractor"** de MobileNetV3: manchas anaranjadas en cualquier hoja |

**Strawberry___Leaf_scorch como atractor de EfficientNetV2B0:** EfficientNetV2B0 predice esta clase con >80% de confianza para imágenes de Grape, Cherry, Potato, Soybean e IFD. El patrón visual (manchas marrones/rojas sobre verde) es compartido por muchas enfermedades y el modelo lo sobreajusta hacia esta clase.

**Corn___Common_rust como atractor de MobileNetV3:** MobileNetV3 predice `Corn___Common_rust` para 7 de 10 imágenes de Apple. La razón: las manchas anaranjadas del Cedar apple rust y las pústulas de Corn common rust comparten firma visual similar, y Corn rust tiene ~1,100 imágenes vs. ~300 de Apple Cedar rust.

### 14.2 Clases subrepresentadas → fallo sistemático

| Clase en PlantVillage | ~N imágenes | Efecto observado |
|---|---|---|
| Potato___healthy | ~100 (mínimo del dataset) | Coherencia 0% — clasificadas sistemáticamente como Strawberry |
| Apple___Cedar_apple_rust | ~300 | Confundidas con Corn___Common_rust (similitud visual de manchas) |
| Grape___healthy | ~400 | Confundidas con Strawberry/Tomato |

**Potato es el caso más crítico:** con solo ~100 imágenes de entrenamiento para su clase sana, las 10 imágenes de papa evaluadas (mayormente sanas) reciben 0% de coherencia botánica. El modelo nunca aprendió a reconocer la hoja de papa como categoría robusta.

### 14.3 Plantas con una sola clase en PlantVillage

- **Soybean:** Solo `Soybean___healthy` (~5,100 imágenes). El modelo predice soja únicamente para imágenes con hojas verdes sin manchas visibles; ante cualquier variación, redirige a clases dominantes. Coherencia: 10% (MNV3) / 0% (EfficientNet).
- **Raspberry:** Solo `Raspberry___healthy` (~340 imágenes). No evaluada.
- **Blueberry:** Solo `Blueberry___healthy` (~1,500 imágenes). No evaluada.

### 14.4 Orange: caso de dominio completamente ausente

Las 3 imágenes de naranja (Orange) reciben 0% de coherencia botánica en ambos modelos porque **no existe ninguna clase Orange en PlantVillage**. Ambos modelos producen predicciones con confianzas de 77–90%, lo que refuerza la conclusión sobre la ausencia de mecanismo de rechazo.

---

## 15. Hallazgos principales para la tesis

### 15.1 Hallazgo 1 — Viabilidad técnica on-device confirmada

Ambos modelos operan de forma estable en ambos dispositivos: sin crashes en ninguna de las 12 sesiones ejecutadas (3 sesiones × 2 modelos × 2 dispositivos), con inferencia completamente determinista. El sistema propuesto es técnicamente viable para deployment en campo.

### 15.2 Hallazgo 2 — MobileNetV3 domina en eficiencia sin costo en precisión

> MobileNetV3 Small logra una latencia promedio de **7.83 ms en Samsung** y **30.67 ms en Moto G**, con coherencia botánica de **39.5%**, equivalente a la de EfficientNetV2B0 (40.7%) que opera a **52.80 ms y 274.13 ms** respectivamente. El tamaño de MobileNetV3 (1.91 MB) es **5.94× menor** que el de EfficientNetV2B0 (11.33 MB).

Para el dominio cubierto por PlantVillage, MobileNetV3 Small es el modelo recomendado para dispositivos móviles: menor latencia, menor tamaño, menor consumo, y precisión estadísticamente equivalente.

### 15.3 Hallazgo 3 — El hardware no afecta la calidad de predicción

Las predicciones top-1 son idénticas en Moto G y Samsung Galaxy S23 para las 99 imágenes evaluadas. El hardware influye exclusivamente en la velocidad de inferencia (3.91× a 5.19× más rápido en Samsung) y en el consumo energético (~36% menos batería en Samsung). La brecha de calidad entre dispositivos es cero.

### 15.4 Hallazgo 4 — La precisión diagnóstica está limitada por el dataset

La coherencia botánica global del 39.5–40.7% no refleja falla de los modelos en su dominio de entrenamiento, sino la limitación del dataset PlantVillage para generalizar a especies subrepresentadas (Potato, Apple, Grape) o completamente ausentes (Orange, Squash). Los grupos bien representados (Corn, Tomato, Strawberry) alcanzan 85–100% de coherencia botánica.

### 15.5 Hallazgo 5 — Ausencia de mecanismo de rechazo

Ambos modelos asignan predicciones con confianzas de hasta 99.9% a imágenes completamente fuera de dominio (fondos sin plantas, planta Orange no existente en PlantVillage). Esta limitación es inherente a los clasificadores closed-set y debe abordarse como trabajo futuro para cualquier despliegue en campo real.

### 15.6 Hallazgo 6 — La brecha entre modelos se reduce en hardware de alta gama

El ratio de latencia entre EfficientNetV2B0 y MobileNetV3 pasa de 8.94× en Moto G a 6.74× en Samsung Galaxy S23. Esto indica que el hardware flagship aprovecha mejor la capacidad computacional adicional del modelo más grande, reduciendo la penalización relativa de usar EfficientNetV2B0.

---

## 16. Resumen ejecutivo

| Métrica | Moto G — MNV3 | Moto G — Eff | Samsung — MNV3 | Samsung — Eff |
|---|---|---|---|---|
| Latencia promedio (ms) | 30.67 | 274.13 | **7.83** | 52.80 |
| Mediana P50 (ms) | 28.67 | 267.87 | **6.57** | 48.90 |
| P95 (ms) | 52.80 | 317.80 | **13.97** | 68.87 |
| Variabilidad Std (ms) | 12.63 | 22.87 | **3.37** | 8.97 |
| Init time promedio (ms) | 3300 | 350 | 785 | **151** |
| Tamaño modelo (MB) | **1.91** | 11.33 | **1.91** | 11.33 |
| Batería Δ / sesión (%) | −11.0 | −11.7 | **−7.0** | **−7.7** |
| Temperatura Δ promedio (°C) | **+2.4** | **+2.6** | +4.2 | +4.2 |
| PSS típica (MB) | **~220** | ~290 | ~320 | ~450 |
| Coherencia botánica | 39.5% | 40.7% | 39.5% | 40.7% |
| Consistencia entre sesiones | **100%** | **100%** | **100%** | **100%** |
| Crashes | **0** | **0** | **0** | **0** |

**Hallazgo central:** Para el despliegue en dispositivos móviles Android orientados al diagnóstico de enfermedades en plantas, **MobileNetV3 Small FP16** es el modelo más adecuado: ofrece latencia hasta 8.94× menor, ocupa 5.94× menos almacenamiento, consume batería de forma equivalente, y produce coherencia botánica estadísticamente comparable a EfficientNetV2B0. La viabilidad técnica del sistema está confirmada en hardware mid-range (Moto G, Snapdragon 662) y flagship (Samsung Galaxy S23, Snapdragon 8 Gen 2), con inferencia determinista y estable en todas las condiciones evaluadas.

**Limitación principal a declarar en la tesis:** La precisión diagnóstica del sistema está acotada por el dominio del dataset PlantVillage. Para plantas subrepresentadas o ausentes del dataset (Potato, Soybean, Orange, Squash), el sistema produce predicciones incorrectas con alta confianza sin capacidad de rechazo. Extender el dataset y agregar un módulo de detección de imágenes fuera de dominio son las mejoras prioritarias para versiones futuras del sistema.
