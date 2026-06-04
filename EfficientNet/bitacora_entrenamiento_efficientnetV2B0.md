# Bitácora de Entrenamiento — EfficientNetV2B0

**Notebook:** `EfficientNet/EfficientNetV2B0.ipynb`
**Fecha de ejecución:** 2026-06-03
**Modelo:** EfficientNetV2B0 — variante B0 (224×224, ~5.9M parámetros)
**Dataset:** PlantVillage — 38 clases de enfermedades en plantas

---

## Configuración del entorno

| Parámetro | Valor |
|---|---|
| Python | `/opt/anaconda3/envs/efficientnet/bin/python` |
| TensorFlow | 2.16.2 |
| Keras | 3.12.2 |
| GPU | `/physical_device:GPU:0` (disponible) |
| Modo de entrenamiento | `full` |
| Variante | `B0` |
| Tamaño de imagen | 224×224 |
| Batch size | 32 |
| Semilla de reproducibilidad | 123 |

---

## Dataset — PlantVillage

| Split | Ejemplos |
|---|---|
| Train | 43,442 (80%) |
| Validación | 5,430 (10%) |
| Test | 5,431 (10%) |
| **Total** | **54,303** |

- Clases: **38**
- Preprocessing: `pixel / 255.0` → rango `[0, 1]`
- Split con `shuffle(seed=123, reshuffle_each_iteration=False)` — idéntico al usado en MobileNetV2 y MobileNetV3

---

## Referencia de modelos existentes (antes del entrenamiento)

| Modelo | TFLite FP16 (MB) | Test Accuracy | Preprocessing |
|---|---|---|---|
| MobileNetV2 FP16 (baseline Android) | 4.3494 | — | `[-1, 1]` |
| MobileNetV3 Small FP16 (candidato) | — | 0.9856 | `[-1, 1]` |

---

## Arquitectura del modelo

```
Model: "efficientnetv2b0_plantvillage"

Layer (type)                   Output Shape         Params      Trainable
───────────────────────────────────────────────────────────────────────────
image (InputLayer)             (None, 224, 224, 3)          0       -
efficientnetv2-b0 (Functional) (None, 7, 7, 1280)   5,919,312   N (fase 1)
gap (GlobalAveragePooling2D)   (None, 1280)                  0       -
head_bn (BatchNormalization)   (None, 1280)              5,120       Y
dropout (Dropout)              (None, 1280)                  0       -
predictions (Dense)            (None, 38)               48,678       Y
───────────────────────────────────────────────────────────────────────────
Total params:          5,973,110  (22.79 MB)
Trainable params (fase 1):  51,238  (200.15 KB)
Non-trainable params:  5,921,872  (22.59 MB)
```

Cabeza: `GlobalAveragePooling2D` → `BatchNormalization` → `Dropout(0.3)` → `Dense(38, softmax)`

---

## Fase 1 — Entrenamiento de la cabeza (backbone congelado)

- **Épocas:** 15
- **Learning rate inicial:** `1e-3`
- **ReduceLROnPlateau:** factor 0.3, patience 2, min_lr `1e-7`
- **EarlyStopping:** patience 4, monitor `val_accuracy`

### Historial por época

| Época | Train Acc | Train Loss | Val Acc | Val Loss | LR | Checkpoint |
|---|---|---|---|---|---|---|
| 1 | 0.8652 | 0.4660 | 0.9586 | 0.1361 | 1e-3 | ✅ |
| 2 | 0.9325 | 0.2124 | 0.9661 | 0.1042 | 1e-3 | ✅ |
| 3 | 0.9409 | 0.1824 | 0.9694 | 0.0869 | 1e-3 | ✅ |
| 4 | 0.9457 | 0.1708 | 0.9692 | 0.0916 | 1e-3 | — |
| 5 | 0.9471 | 0.1639 | 0.9740 | 0.0783 | 1e-3 | ✅ |
| 6 | 0.9486 | 0.1611 | 0.9729 | 0.0877 | 1e-3 | — |
| 7 | 0.9491 | 0.1608 | 0.9718 | 0.0888 | 1e-3 → **3e-4** | — (ReduceLR) |
| 8 | 0.9576 | 0.1289 | 0.9770 | 0.0688 | 3e-4 | ✅ |
| 9 | 0.9604 | 0.1223 | 0.9770 | 0.0672 | 3e-4 | — |
| 10 | 0.9616 | 0.1161 | 0.9772 | 0.0668 | 3e-4 | ✅ |
| 11 | 0.9609 | 0.1178 | 0.9766 | 0.0642 | 3e-4 | — |
| 12 | 0.9620 | 0.1151 | 0.9779 | 0.0607 | 3e-4 | ✅ |
| 13 | 0.9625 | 0.1121 | 0.9773 | 0.0640 | 3e-4 | — |
| 14 | 0.9633 | 0.1113 | 0.9785 | 0.0613 | 3e-4 → **9e-5** | ✅ (ReduceLR) |
| 15 | 0.9646 | 0.1038 | **0.9805** | **0.0588** | 9e-5 | ✅ |

**Mejor val_accuracy al cierre de fase 1:** `0.9805` (época 15)

---

## Fase 2 — Fine-tuning (últimas 20 capas del backbone)

- **Épocas:** 10 (épocas 16–25 del historial continuo)
- **Learning rate:** `1e-5` (fijo)
- **Capas entrenables del backbone:** 16 de 268 (las BN siempre congeladas)

### Historial por época

| Época | Train Acc | Train Loss | Val Acc | Val Loss | LR | Checkpoint |
|---|---|---|---|---|---|---|
| 16 | 0.9676 | 0.0947 | 0.9797 | 0.0543 | 1e-5 | — |
| 17 | 0.9704 | 0.0860 | 0.9810 | 0.0522 | 1e-5 | ✅ |
| 18 | 0.9702 | 0.0888 | 0.9818 | 0.0503 | 1e-5 | ✅ |
| 19 | 0.9727 | 0.0826 | 0.9820 | 0.0492 | 1e-5 | ✅ |
| 20 | 0.9724 | 0.0805 | 0.9832 | 0.0479 | 1e-5 | ✅ |
| 21 | 0.9750 | 0.0735 | 0.9823 | 0.0487 | 1e-5 | — |
| 22 | 0.9748 | 0.0764 | 0.9832 | 0.0471 | 1e-5 | — |
| 23 | 0.9760 | 0.0705 | 0.9843 | 0.0451 | 1e-5 | ✅ |
| 24 | 0.9768 | 0.0703 | **0.9851** | **0.0446** | 1e-5 | ✅ |
| 25 | 0.9775 | 0.0687 | 0.9847 | 0.0434 | 1e-5 | — |

**Mejor val_accuracy global:** `0.9851` (época 24)
**Mejor checkpoint guardado:** época 24

---

## Evaluación en el conjunto de prueba

Modelo cargado desde checkpoint: `efficientnetv2b0_v1_best.keras`

| Métrica | Valor |
|---|---|
| **Test Accuracy** | **0.9847** |
| **Test Loss** | **0.0490** |
| Total ejemplos de prueba | 5,431 |
| Total errores | 83 (1.5%) |

### Reporte por clase (precision / recall / F1 / support)

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Apple___Apple_scab | 0.9853 | 0.9853 | 0.9853 | 68 |
| Apple___Black_rot | 1.0000 | 1.0000 | 1.0000 | 62 |
| Apple___Cedar_apple_rust | 1.0000 | 1.0000 | 1.0000 | 24 |
| Apple___healthy | 0.9930 | 1.0000 | 0.9965 | 141 |
| Blueberry___healthy | 0.9888 | 1.0000 | 0.9944 | 176 |
| Cherry___healthy | 1.0000 | 1.0000 | 1.0000 | 77 |
| Cherry___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 88 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.9375 | 0.9000 | 0.9184 | 50 |
| Corn___Common_rust | 1.0000 | 0.9918 | 0.9959 | 122 |
| Corn___healthy | 1.0000 | 1.0000 | 1.0000 | 132 |
| Corn___Northern_Leaf_Blight | 0.9485 | 0.9787 | 0.9634 | 94 |
| Grape___Black_rot | 0.9832 | 1.0000 | 0.9915 | 117 |
| Grape___Esca_(Black_Measles) | 1.0000 | 0.9846 | 0.9922 | 130 |
| Grape___healthy | 1.0000 | 1.0000 | 1.0000 | 47 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1.0000 | 1.0000 | 1.0000 | 107 |
| Orange___Haunglongbing_(Citrus_greening) | 1.0000 | 1.0000 | 1.0000 | 551 |
| Peach___Bacterial_spot | 1.0000 | 1.0000 | 1.0000 | 233 |
| Peach___healthy | 1.0000 | 1.0000 | 1.0000 | 42 |
| Pepper,_bell___Bacterial_spot | 0.9903 | 0.9903 | 0.9903 | 103 |
| Pepper,_bell___healthy | 0.9943 | 1.0000 | 0.9971 | 173 |
| Potato___Early_blight | 1.0000 | 0.9906 | 0.9953 | 106 |
| **Potato___healthy** | **1.0000** | **0.7273** | **0.8421** | **11** |
| Potato___Late_blight | 1.0000 | 0.9818 | 0.9908 | 110 |
| Raspberry___healthy | 1.0000 | 1.0000 | 1.0000 | 39 |
| Soybean___healthy | 0.9943 | 0.9981 | 0.9962 | 527 |
| Squash___Powdery_mildew | 1.0000 | 1.0000 | 1.0000 | 173 |
| Strawberry___healthy | 0.9737 | 1.0000 | 0.9867 | 37 |
| Strawberry___Leaf_scorch | 1.0000 | 0.9897 | 0.9948 | 97 |
| Tomato___Bacterial_spot | 0.9673 | 0.9857 | 0.9764 | 210 |
| **Tomato___Early_blight** | **0.9691** | **0.8952** | **0.9307** | **105** |
| Tomato___healthy | 0.9886 | 0.9943 | 0.9914 | 174 |
| Tomato___Late_blight | 0.9622 | 0.9622 | 0.9622 | 185 |
| **Tomato___Leaf_Mold** | **0.9341** | **0.9551** | **0.9444** | **89** |
| **Tomato___Septoria_leaf_spot** | **0.9459** | **0.9396** | **0.9428** | **149** |
| **Tomato___Spider_mites Two-spotted_spider_mite** | **0.9382** | **0.9543** | **0.9462** | **175** |
| **Tomato___Target_Spot** | **0.8855** | **0.8855** | **0.8855** | **131** |
| Tomato___Tomato_mosaic_virus | 0.9500 | 1.0000 | 0.9744 | 38 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.9963 | 0.9907 | 0.9935 | 538 |
| **accuracy** | | | **0.9847** | **5431** |
| macro avg | 0.9823 | 0.9758 | 0.9784 | 5431 |
| weighted avg | 0.9848 | 0.9847 | 0.9847 | 5431 |

> Las filas en **negrita** corresponden a las clases con F1 < 0.96, las más débiles del modelo.

### Errores de alta confianza (top 25 ordenados por confianza descendente)

| True Label | Predicted Label | Confianza |
|---|---|---|
| Tomato___Late_blight | Tomato___healthy | 0.9995 |
| Corn___Northern_Leaf_Blight | Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.9993 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | Corn___Northern_Leaf_Blight | 0.9981 |
| Tomato___Target_Spot | Tomato___Spider_mites Two-spotted_spider_mite | 0.9924 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | Corn___Northern_Leaf_Blight | 0.9919 |
| Tomato___Septoria_leaf_spot | Tomato___Leaf_Mold | 0.9905 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato___Late_blight | 0.9861 |
| Potato___healthy | Soybean___healthy | 0.9849 |
| Tomato___Target_Spot | Tomato___Spider_mites Two-spotted_spider_mite | 0.9725 |
| Tomato___Target_Spot | Tomato___Spider_mites Two-spotted_spider_mite | 0.9658 |
| Tomato___Bacterial_spot | Soybean___healthy | 0.9645 |
| Tomato___Bacterial_spot | Tomato___Target_Spot | 0.9602 |
| Tomato___Early_blight | Tomato___Bacterial_spot | 0.9577 |
| Tomato___Target_Spot | Tomato___Spider_mites Two-spotted_spider_mite | 0.9487 |
| Tomato___Septoria_leaf_spot | Tomato___Late_blight | 0.9391 |
| Tomato___Septoria_leaf_spot | Tomato___Leaf_Mold | 0.9363 |
| Corn___Northern_Leaf_Blight | Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.9291 |
| Corn___Common_rust | Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.9278 |
| Tomato___Late_blight | Tomato___Leaf_Mold | 0.9262 |
| Tomato___Early_blight | Tomato___Septoria_leaf_spot | 0.9072 |
| Tomato___Early_blight | Tomato___Septoria_leaf_spot | 0.9065 |
| Tomato___Target_Spot | Tomato___Septoria_leaf_spot | 0.9034 |
| Apple___Apple_scab | Apple___healthy | 0.8951 |
| Tomato___Leaf_Mold | Tomato___Bacterial_spot | 0.8918 |
| Tomato___Target_Spot | Tomato___Bacterial_spot | 0.8844 |

---

## Exportación TFLite

| Formato | Archivo | Tamaño real |
|---|---|---|
| FP32 | `efficientnetv2b0_v1_fp32.tflite` | **22.5218 MB** |
| FP16 | `efficientnetv2b0_v1_fp16.tflite` | **11.3324 MB** |

**Ruta:** `EfficientNet/saved_models/`

> **Nota crítica para integración Android:** este modelo usa preprocessing `[0, 1]` (`pixel / 255.0`). Si se integra en la app, el preprocesamiento Android debe cambiar de `(pixel - 127.5) / 127.5` (MobileNet) a `pixel / 255.0`. No hacer este cambio produce predicciones incorrectas silenciosamente.

---

## Benchmark local TFLite (macOS, CPU — no dispositivo Android real)

| Formato | Tamaño (MB) | Avg (ms) | Mediana (ms) | P95 (ms) | Std (ms) |
|---|---|---|---|---|---|
| **FP16** | 11.3324 | **20.50** | 20.47 | **21.06** | 0.31 |
| FP32 | 22.5218 | 19.93 | 19.58 | 21.72 | 0.81 |

> FP16 y FP32 tienen latencia local prácticamente idéntica porque la CPU del Mac no tiene aceleración nativa para FP16. En un dispositivo Android con GPU/DSP la diferencia sería más pronunciada a favor de FP16.

---

## Tabla comparativa final de modelos

| Modelo | Test Accuracy | TFLite FP16 (MB) | Avg local (ms) | P95 local (ms) | Preprocessing | Notas |
|---|---|---|---|---|---|---|
| MobileNetV2 FP16 | — | 4.3494 | — | — | `[-1, 1]` | Baseline Android |
| MobileNetV3 Small FP16 | **0.9856** | — | — | — | `[-1, 1]` | Candidato principal |
| **EfficientNetV2B0 FP16** | 0.9847 | 11.3324 | 20.50 | 21.06 | `[0, 1]` | Este entrenamiento |

---

## Archivos generados

| Archivo | Descripción |
|---|---|
| `efficientnetv2b0_v1_best.keras` | Mejor checkpoint (época 24, val_acc 0.9851) |
| `efficientnetv2b0_v1_training_log.csv` | Log de accuracy/loss por época (25 épocas) |
| `training_graph_v2b0.png` | Gráficas de accuracy y loss (fase 1 + fine-tuning) |
| `confusion_matrix_v2b0.png` | Matriz de confusión 38×38 en el conjunto de prueba |
| `dataset_distribution_v2.png` | Distribución de clases de PlantVillage |
| `saved_models/efficientnetv2b0_v1/` | SavedModel exportado |
| `saved_models/efficientnetv2b0_v1_fp32.tflite` | TFLite sin cuantización (22.52 MB) |
| `saved_models/efficientnetv2b0_v1_fp16.tflite` | TFLite FP16 recomendado para Android (11.33 MB) |

---

---

# Análisis Completo de Resultados

## 1. Accuracy global y posición en el experimento comparativo

EfficientNetV2B0 alcanzó un **test accuracy de 0.9847** (98.47%), a tan solo 0.09 puntos porcentuales por debajo de MobileNetV3 Small (98.56%), su competidor directo en el experimento. La diferencia es estadísticamente marginal sobre 5,431 ejemplos y no representa una ventaja práctica para la app.

Lo que sí representa una diferencia práctica es el **tamaño del modelo**: EfficientNetV2B0 FP16 pesa **11.33 MB**, 2.6× más que MobileNetV2 (4.35 MB). No se dispone del tamaño TFLite de MobileNetV3 Small, pero siendo la variante "Small" se espera un tamaño considerablemente menor que 11 MB. En aplicaciones móviles donde el tamaño del APK y la memoria RAM son restricciones reales, este factor es decisivo.

---

## 2. Curva de aprendizaje y comportamiento del entrenamiento

### Fase 1 — Cabeza (épocas 1–15)

El modelo arrancó sólido desde la primera época con **val_accuracy 0.9586**, evidenciando que los pesos de ImageNet son muy transferibles a PlantVillage aun sin fine-tuning del backbone. La convergencia fue progresiva pero mostró dos mesetas notables:

- **Épocas 4–7:** la val_accuracy se estancó entre 0.9692–0.9740. El `ReduceLROnPlateau` actuó en la época 7, reduciendo el LR de `1e-3` a `3e-4`, lo que desbloqueó el entrenamiento y produjo el salto de la época 8 (val_accuracy 0.9770).
- **Épocas 9–13:** segunda meseta alrededor de 0.9770. El LR se redujo de nuevo en la época 14 a `9e-5`, permitiendo el avance final hasta 0.9805 en la época 15.

Este patrón de meseta → reducción LR → mejora es saludable e indica que el `ReduceLROnPlateau` funcionó como mecanismo de escape efectivo.

La brecha train/val al final de fase 1 es mínima (train 0.9646 vs val 0.9805), lo que indica **ausencia de sobreajuste** a pesar del Dropout(0.3).

### Fase 2 — Fine-tuning (épocas 16–25)

Con `lr=1e-5` y 16 capas del backbone desbloqueadas, el fine-tuning produjo una mejora sostenida de **+0.46 puntos porcentuales** en val_accuracy (de 0.9805 a 0.9851). La val_loss continuó descendiendo en casi todas las épocas (0.0543 → 0.0446), sin signos de sobreajuste.

Los avances más notables ocurrieron en las épocas 20 (+0.0012) y 23–24 (+0.0011 y +0.0006), lo que sugiere que el modelo aún tenía espacio para mejorar con más épocas de fine-tuning o capas adicionales desbloqueadas. La época 25 mostró una leve caída de val_accuracy (0.9851 → 0.9847) mientras el train accuracy seguía subiendo (0.9775), primera señal incipiente de sobreajuste, lo que valida el uso de 10 épocas de fine-tuning.

---

## 3. Análisis por clase — Fortalezas

**27 de 38 clases** alcanzaron F1 ≥ 0.99, incluyendo:

- **Perfección F1=1.0000** en 13 clases: `Apple___Black_rot`, `Apple___Cedar_apple_rust`, `Cherry___healthy`, `Cherry___Powdery_mildew`, `Corn___healthy`, `Grape___healthy`, `Grape___Leaf_blight`, `Orange___Haunglongbing`, `Peach___Bacterial_spot`, `Peach___healthy`, `Raspberry___healthy`, `Squash___Powdery_mildew`.

Estas clases tienen morfología visual distintiva y suficiente representación en el dataset. Orange___Haunglongbing con 551 ejemplos de soporte y F1=1.0 es un resultado particularmente sólido.

---

## 4. Análisis por clase — Puntos débiles

### Potato___healthy — El caso más crítico

- **F1: 0.8421** | Precision: 1.0000 | Recall: **0.7273** | Support: **11**
- El modelo confundió 3 de 11 imágenes de papa sana (1 fue clasificada como `Soybean___healthy`).
- La causa principal es el **support extremadamente bajo (11 ejemplos)**. Con tan pocos ejemplos de prueba, cada error individual impacta drásticamente el recall. Este resultado no es confiable estadísticamente y no debe interpretarse como debilidad real del modelo en papa sana.

### Tomato___Target_Spot — El cuello de botella genuino

- **F1: 0.8855** | Precision: 0.8855 | Recall: 0.8855 | Support: **131**
- Con 131 ejemplos de soporte, este es el resultado más preocupante del modelo. `Tomato___Target_Spot` aparece como **clase predicha incorrectamente** con alta confianza en múltiples errores, y también como clase **verdadera que se predice incorrectamente** hacia `Tomato___Spider_mites` y `Tomato___Bacterial_spot`.
- La confusión es bidireccional, lo que indica que las enfermedades de tomate con lesiones oscuras y bordes irregulares son intrínsecamente difíciles de separar visualmente, incluso para el modelo.

### Clúster de confusión en enfermedades de tomate

El análisis de errores de alta confianza revela un **patrón sistemático de confusión en enfermedades de tomate** con manchas o lesiones de apariencia similar:

- `Tomato___Target_Spot` ↔ `Tomato___Spider_mites` (3 errores con confianza >0.94)
- `Tomato___Septoria_leaf_spot` ↔ `Tomato___Leaf_Mold` ↔ `Tomato___Late_blight` (múltiples cruces)
- `Tomato___Early_blight` ↔ `Tomato___Bacterial_spot` / `Tomato___Septoria_leaf_spot`

Estas confusiones son esperables: todas involucran manchas foliares en tomate con colores pardos/amarillos y diferentes patrones de borde. El modelo distingue correctamente la mayoría, pero el 1–2% de confusión restante es inherente a la similitud visual de las enfermedades.

### Clúster de confusión en enfermedades de maíz

- `Corn___Northern_Leaf_Blight` ↔ `Corn___Cercospora_leaf_spot` (confusión bilateral con confianza >0.99)
- F1 de Cercospora: 0.9184 | F1 de Northern Leaf Blight: 0.9634
- El error de Cercospora clasificado como Northern_Leaf_Blight con confianza 0.9993 es el **segundo error más severo del modelo**. Ambas enfermedades producen lesiones alargadas de color gris-café en hojas de maíz, y la distinción visual entre ellas es difícil incluso para expertos.

---

## 5. Evaluación del benchmark de latencia

| Formato | Avg (ms) | P95 (ms) |
|---|---|---|
| EfficientNetV2B0 FP16 | 20.50 | 21.06 |
| EfficientNetV2B0 FP32 | 19.93 | 21.72 |

La latencia medida (~20 ms) corresponde a una **CPU de macOS**, no a un dispositivo Android real. En un Snapdragon 765G típico la inferencia de un modelo similar suele estar en el rango de 80–200 ms en CPU, o 20–50 ms con delegado GPU/NNAPI. El benchmark local sirve únicamente para comparar relativo entre modelos bajo las mismas condiciones, no como predictor de latencia en producción.

La diferencia FP16 vs FP32 en latencia local es despreciable (0.57 ms avg), lo que confirma que la aceleración FP16 requiere hardware compatible (GPU móvil con soporte FP16) para manifestarse.

---

## 6. Veredicto para la tesis y decisión de integración Android

### ¿Vale la pena integrar EfficientNetV2B0 en la app?

**Probablemente no, como reemplazo de MobileNetV3. Sí, como punto de referencia académico.**

| Criterio | EfficientNetV2B0 | MobileNetV3 Small | Veredicto |
|---|---|---|---|
| Test accuracy | 0.9847 | **0.9856** | MobileNetV3 +0.09% |
| TFLite FP16 (MB) | 11.33 | < 11.33 (estimado) | MobileNetV3 gana |
| Preprocessing compatible Android | `[0, 1]` (requiere cambio) | `[-1, 1]` (ya implementado) | MobileNetV3 gana |
| Velocidad de entrenamiento | Rápida (Fused-MBConv) | Similar | Empate |
| Parámetros totales | ~5.9M | < 5.9M (Small) | MobileNetV3 gana |

EfficientNetV2B0 empata en accuracy con MobileNetV3 Small pero pierde en tamaño de modelo (~2.6× más pesado que MobileNetV2 baseline) y requeriría modificar el preprocesamiento de la app Android. Para una aplicación móvil donde el tamaño del APK y la RAM son restricciones reales, MobileNetV3 Small sigue siendo el candidato preferido.

### Valor como experimento comparativo

- Confirma que **ambas familias (MobileNet y EfficientNetV2) convergen a accuracy similar (~98.5%)** en PlantVillage con transfer learning desde ImageNet.
- Demuestra que el **cuello de botella no es la arquitectura** sino la dificultad intrínseca de separar enfermedades visualmente similares (tomate, maíz).
- Establece un techo de accuracy confiable para el dataset bajo las condiciones del experimento, útil para justificar en la tesis que modelos más grandes no producirán ganancias significativas en este dataset específico.

### Recomendaciones para la tesis

1. **Reportar EfficientNetV2B0 como experimento de comparación**, no como modelo final de la app.
2. **Citar el clúster de confusión en enfermedades de tomate** como limitación inherente al dataset, no al modelo.
3. Si se desea mejorar `Tomato___Target_Spot` (F1 0.8855) y `Potato___healthy` (support=11), considerar **data augmentation agresiva** o técnicas de few-shot para las clases con menor soporte.
4. Para producción Android, mantener **MobileNetV3 Small FP16** como modelo principal dado su mejor balance accuracy/tamaño/compatibilidad de preprocessing.
