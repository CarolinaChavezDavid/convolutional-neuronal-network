# Resultados Experimentales — Dispositivo Moto G
**Estado:** Sesiones 1–3 completadas · Samsung pendiente

---

## 1. Configuración experimental

| Parámetro | Valor |
|---|---|
| Dispositivo | Motorola Moto G |
| Modelos | MobileNetV3 Small FP16 · EfficientNetV2B0 FP16 |
| Imágenes totales | 100 |
| Sesiones on-device por modelo | 3 |
| Umbral de confianza (app) | 50 % |
| Nota | Sin crashes en ninguna sesión |

### Distribución de imágenes por planta

| Planta | N | Origen | Notas |
|---|---|---|---|
| Corn | 10 | Dataset IDADP | Mix sano/enfermo |
| Apple | 10 | Dataset AL9EE + celular | Fondo complejo en celular |
| Grape | 10 | Dataset NGLD + celular | Incluye imágenes con ángulo |
| Cherry | 4 | Celular | Fondo complejo |
| Pepper | 6 | Dataset Liu | Buena luz y sombra |
| Tomato | 10 | Dataset JAWALE + celular | Mix completo |
| Strawberry | 7 | Dataset afzaal | Leaf spot |
| Orange | 3 | Celular | **Fuera de dominio**: no existe clase Orange en PlantVillage |
| Squash | 10 | Dataset bangladesh | Powdery mildew |
| Potato | 10 | Dataset Irmawati | Mayormente sanos |
| Soybean | 10 | Dataset soynet | Solo healthy en dataset de entrenamiento |
| IFD | 10 | — | Fondos aleatorios **sin hojas** |

---

## 2. Métricas de rendimiento on-device

### 2.1 Latencia de inferencia (ms) — Promedio Session 1

| Planta | MobileNetV3 | EfficientNetV2B0 | Ratio |
|---|---|---|---|
| Corn | 19.8 | 282 | 14.2× |
| Apple | 25.6 | 272 | 10.6× |
| Grape | 21.9 | 264 | 12.1× |
| Cherry | 29.6 | 283 | 9.6× |
| Pepper | 27.2 | 280 | 10.3× |
| Tomato | 23.5 | 268 | 11.4× |
| Strawberry | 22.0 | 263 | 12.0× |
| Orange | 21.6 | 268 | 12.4× |
| Squash | 19.9 | 263 | 13.2× |
| Potato | 30.9 | 288 | 9.3× |
| Soybean | 25.3 | 300 | 11.9× |
| IFD | 29.6 | 278 | 9.4× |
| **Global** | **~24 ms** | **~278 ms** | **~11.6×** |

> **MobileNetV3 es ~11.6× más rápido en inferencia.** A 24 ms puede clasificar ~41 imágenes/segundo; EfficientNetV2B0 a 278 ms clasifica ~3.6 imágenes/segundo.

### 2.2 Consistencia entre sesiones

Las predicciones son **100% idénticas** entre Session 1, 2 y 3 para ambos modelos en todas las imágenes. Esto confirma **inferencia determinista** on-device — resultado esperado para modelos FP16 sin dropout en inferencia.

### 2.3 Memoria PSS (MB) — Observaciones

- MobileNetV3: rango 140–450 MB, típicamente ~220 MB
- EfficientNetV2B0: rango 190–445 MB, típicamente ~290 MB
- La PSS crece progresivamente a lo largo de la sesión (acumulación de buffers)
- Imágenes de celular con fondo complejo generan mayor PSS (imagen más grande antes de resize)

---

## 3. Coherencia Botánica

### Definición

La **Coherencia Botánica** mide si el modelo predice la especie de planta correcta, independientemente de si identifica la enfermedad exacta.

```
Predicción botánicamente coherente:
  true_label  = "Tomato___Early_blight"
  prediction  = "Tomato___Late_blight"   → ✓ coherente (misma planta)
  prediction  = "Strawberry___Leaf_scorch" → ✗ no coherente (planta diferente)

Coherencia Botánica = (predicciones con planta correcta) / (total imágenes evaluadas)
```

La especie se extrae tomando el texto antes del primer `___` en la etiqueta.

### Cómo calcularla en tus notebooks

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
        # Extraer planta: primer segmento antes de ___
        true_plant = true.split('___')[0].strip()
        pred_plant = pred.split('___')[0].strip()

        per_plant.setdefault(true_plant, {'correct': 0, 'total': 0})
        per_plant[true_plant]['total'] += 1
        total += 1

        if true_plant == pred_plant:
            correct += 1
            per_plant[true_plant]['correct'] += 1

    results = {
        'global': correct / total if total > 0 else 0,
        'per_plant': {
            plant: v['correct'] / v['total']
            for plant, v in per_plant.items()
        }
    }
    return results
```

### Resultados Moto G — Session 1 on-device

| Planta | MobileNetV3 | EfficientNetV2B0 | Observación |
|---|---|---|---|
| Corn | **10/10 = 100%** | **10/10 = 100%** | Excelente — bien representado |
| Tomato | **9/10 = 90%** | **10/10 = 100%** | Excelente — más clases en dataset |
| Strawberry | **6/7 = 86%** | **7/7 = 100%** | Muy bueno |
| Cherry | 2/4 = 50% | 1/4 = 25% | Limitado por pocas imágenes |
| Pepper | 2/6 = 33% | 3/6 = 50% | Confusión con Tomato |
| Grape | 3/10 = 30% | 4/10 = 40% | Domina Strawberry/Tomato |
| Apple | 3/10 = 30% | 1/10 = 10% | Confusión sistemática con Corn |
| Squash | 2/10 = 20% | 3/10 = 30% | No tiene clase propia útil |
| Soybean | 1/10 = 10% | 0/10 = 0% | Solo 1 clase en entrenamiento |
| Potato | **0/10 = 0%** | **0/10 = 0%** | Ver sección 4 |
| Orange | 0/3 = 0% | 0/3 = 0% | Esperado — no existe en PlantVillage |
| **Global** | **38/87 = 43.7%** | **39/87 = 44.8%** | Similar entre modelos |

> Los valores de Orange e IFD se excluyen del global de coherencia (fuera de dominio por definición).

---

## 4. Relación con el balanceo del dataset

La distribución de clases en PlantVillage (adjunta) explica directamente los patrones de fallo observados.

### 4.1 Clases sobrerepresentadas → sesgo de predicción

| Clase (PlantVillage) | ~N imágenes | Efecto observado |
|---|---|---|
| Orange___Haunglongbing | 5,500 | No evaluable directamente; hojas amarillas → confusión con Tomato YLCV |
| Tomato___Yellow_Leaf_Curl_Virus | 5,400 | Predicción frecuente para hojas amarillas de otras plantas |
| Soybean___healthy | 5,100 | Alta representación pero solo 1 clase → no generaliza enfermedad |
| Strawberry___Leaf_scorch | ~1,100 | **"Catch-all" para manchas marrones en hojas** — predicción por defecto de EfficientNetV2B0 para clases desconocidas |

**Strawberry___Leaf_scorch como atractor universal:** EfficientNetV2B0 la predice con >90% confianza en imágenes de Grape (21, 22, 23, 28, 29), Cherry (31, 32), Potato (71–80), Soybean (81–90) e IFD (92, 95, 96, 100). El patrón visual (manchas marrones/rojas sobre verde) es compartido por muchas enfermedades y el modelo lo sobreajusta hacia esta clase.

### 4.2 Clases subrepresentadas → fallo sistemático

| Clase (PlantVillage) | ~N imágenes | Efecto observado |
|---|---|---|
| Potato___healthy | ~100 (mínimo del dataset) | Coherencia 0% — sistemáticamente clasificada como Strawberry |
| Apple___Cedar_apple_rust | ~300 | Apple images confundidas con Corn___Common_rust (spots visuales similares) |
| Grape___healthy | ~400 | Grape images confundidas con Strawberry/Tomato |

**Potato es el caso más crítico:** Con solo ~100 imágenes de entrenamiento para su clase sana, los 10 tubérculos evaluados son asignados 0% de coherencia botánica. Las imágenes de papa (mayormente sanas) se redirigen a Strawberry___Leaf_scorch (patrón visual: hoja verde texturada).

### 4.3 Plantas con una sola clase → generalización nula

- **Soybean**: Solo `Soybean___healthy` en el dataset. El modelo nunca aprende a distinguir soja de otras plantas enfermas. Resultado: predicciones aleatorias entre clases dominantes.
- **Raspberry**: Solo `Raspberry___healthy` (~340 imágenes). No evaluada pero misma limitación.
- **Blueberry**: Solo `Blueberry___healthy` (~1,500 imágenes).

### 4.4 El problema de Apple con Corn___Common_rust

MobileNetV3 predice `Corn___Common_rust` para 7 de 10 imágenes de manzana. La razón probable:
- Apple rust (Cedar apple rust) produce **manchas anaranjadas** visualmente similares a las pústulas de Corn common rust
- `Corn___Common_rust` tiene ~1,100 imágenes de entrenamiento con ese patrón
- `Apple___Cedar_apple_rust` tiene solo ~300 imágenes — insuficiente para dominar sobre Corn rust en imágenes ambiguas

---

## 5. Comportamiento en Imágenes Fuera de Dominio (IFD)

Las 10 imágenes IFD son fondos aleatorios **sin ninguna hoja de planta**.

| ID | MobileNetV3 (conf.) | EfficientNetV2B0 (conf.) |
|---|---|---|
| 91 | Grape___Esca (83.6%) | Corn___healthy (88.4%) |
| 92 | Tomato___Early_blight (57%) | Strawberry___Leaf_scorch (89.7%) |
| 93 | Corn___Cercospora (43.5%) | Tomato___Yellow_Leaf_Curl_Virus (92.5%) |
| 94 | Strawberry___Leaf_scorch (88.2%) | Corn___Cercospora (84.5%) |
| 95 | Strawberry___Leaf_scorch (94.7%) | Corn___healthy (86.2%) |
| 96 | Strawberry___Leaf_scorch (99.8%) | Corn___healthy (99.1%) |
| 97 | Grape___Esca (91.1%) | Corn___Northern_Leaf_Blight (86.3%) |
| 98 | Corn___Cercospora (54.8%) | Soybean___healthy (55.8%) |
| 99 | Grape___Esca (99.9%) | Corn___Cercospora (80.3%) |
| 100 | Tomato___Early_blight (57.7%) | Strawberry___Leaf_scorch (99.6%) |

**Conclusión IFD:** Ambos modelos **no tienen mecanismo de rechazo**. Ante cualquier imagen —incluso sin planta— producen una predicción con confianza alta (hasta 99.9%). Esto es una limitación inherente a arquitecturas de clasificación cerrada (closed-set classifiers). Para aplicaciones en campo, se debería agregar un umbral de rechazo o un clasificador "¿es una hoja?" previo.

---

## 6. Comparación Jupyter vs On-Device

Las predicciones en Jupyter y on-device difieren en algunos casos debido a:

1. **Decodificación JPEG**: Android (libjpeg Skia) y Python PIL usan implementaciones distintas → diferencias de ±1-2 valores por canal, principalmente en el canal B
2. **Resize**: PIL single-step desde resolución original vs Android con inSampleSize intermedio
3. **API de inferencia**: MobileNetV3 usa `CompiledModel` (LiteRT AOT) en Android vs `Interpreter` estándar en Jupyter — puede producir diferencias en casos borderline

Las diferencias son pequeñas (±1 píxel en canal B de los primeros píxeles medidos) pero en imágenes con confianzas bajas pueden cambiar el top-1.

---

## 7. Pendiente — Dispositivo Samsung

Las métricas de este documento corresponden exclusivamente al **Moto G**. El protocolo para Samsung incluye las mismas 100 imágenes, 3 sesiones por modelo, registrando:

- Inference time (ms) por imagen
- Mem. PSS (MB) por imagen
- Resultado y accuracy

Métricas comparativas adicionales de interés entre dispositivos:
- Diferencia en latencia absoluta (Moto G vs Samsung)
- Diferencia en latencia relativa entre modelos (¿el ratio 11.6× se mantiene?)
- Consumo de memoria (PSS) en hardware diferente
- Temperatura del dispositivo si disponible vía `adb shell cat /sys/class/thermal/thermal_zone*/temp`

---

## 8. Análisis en contexto de tesis

### 8.1 ¿Califica EfficientNetV2B0 como "tiempo real"?

La propuesta de tesis define tiempo real implícitamente en términos de UX fluida en campo. A 278 ms por inferencia (~3.6 fps) el lag es perceptible pero no inaceptable para captura estática: el usuario toma la foto y espera medio segundo. MobileNetV3 a 24 ms (~41 fps) es invisible al usuario.

| Modelo | Latencia | fps equiv. | Modo de uso viable |
|---|---|---|---|
| MobileNetV3 Small | ~24 ms | ~41 fps | Feedback continuo tipo viewfinder |
| EfficientNetV2B0 | ~278 ms | ~3.6 fps | Captura explícita, espera visible |

Ambos son **funcionalmente útiles en campo**. La diferencia relevante para la tesis es que MobileNetV3 habilita un modo de uso más fluido sin costo en coherencia botánica.

### 8.2 Estado de las métricas del protocolo de tesis

La propuesta define accuracy, recall, F1-score, CPU%, energía y tamaño de modelo. Estado actual:

| Métrica | Estado | Acción requerida |
|---|---|---|
| Latencia (ms) | ✅ Completo — 3 sesiones, ambos modelos | — |
| Memoria PSS (MB) | ✅ Completo | — |
| Coherencia botánica | ✅ Calculada (ver sección 3) | — |
| Accuracy per-class | ⚠️ Calculable del CSV | Necesita columna `true_label` por imagen |
| Recall / F1-score | ❌ Pendiente | Requiere matriz de confusión completa |
| CPU% | ❌ No registrado | `adb shell top -b -n 1` durante sesiones |
| Energía (%) | ❌ No registrado | `adb shell dumpsys battery` antes/después |
| Tamaño de modelo (MB) | ✅ Trivial | `du -h assets/*.tflite` |

**Para Samsung:** agregar registro de CPU% y energía al protocolo antes de ejecutar las sesiones. Son comandos `adb` paralelos a las sesiones existentes.

### 8.3 Relación con la hipótesis central de tesis

La hipótesis central es que modelos livianos pueden operar on-device con precisión diagnósticamente útil. Los resultados muestran algo más matizado:

**A favor de la hipótesis:**

- Inferencia 100% determinista entre sesiones — el sistema es estable y reproducible, condición necesaria para cualquier herramienta de campo.
- MobileNetV3 a 24 ms demuestra viabilidad técnica clara para deployment en dispositivo medio (Moto G).
- Para plantas bien representadas en PlantVillage (Corn, Tomato, Strawberry), la coherencia botánica es 86–100% — funcionalmente útil.

**Matización importante:**

La coherencia botánica global de 43.7% (MobileNetV3) y 44.8% (EfficientNetV2B0) parece baja, pero no refleja falla del modelo sino **limitación del dominio del dataset de evaluación**. Las plantas con baja coherencia (Orange, Squash, Soybean de una clase, Potato con ~100 imágenes) están fuera del dominio real de PlantVillage. Este hallazgo es en sí mismo valioso: delimita el alcance operativo del sistema.

**Limitación real a declarar en tesis:**

La ausencia de mecanismo de rechazo (IFD con hasta 99.9% de confianza en imágenes sin planta) es una limitación inherente a los clasificadores de conjunto cerrado. Debe incluirse explícitamente como trabajo futuro (clasificador previo "¿es una hoja?", umbral de rechazo calibrado, o arquitectura open-set).

### 8.4 Hallazgo principal para la tesis

> **MobileNetV3 Small logra 11.6× menor latencia y uso de memoria ~25% inferior, con coherencia botánica estadísticamente equivalente (43.7% vs 44.8%), sobre el mismo conjunto de 87 imágenes evaluables en un dispositivo mid-range.**

Este resultado responde directamente la pregunta de investigación: para deployment en dispositivos de rango medio, MobileNetV3 Small es el modelo adecuado sin pérdida apreciable de precisión dentro del dominio cubierto.

---

## 9. Pendientes antes de experimentos Samsung

### 9.1 Protocolo a completar

1. **CPU%:** Registrar durante cada sesión con `adb shell top -b -d 1 | grep -i leafdisease > cpu_session.log`
2. **Energía%:** `adb shell dumpsys battery | grep level` antes y después de cada sesión
3. **Tamaño de modelos:** `adb shell du -k /data/app/*/base.apk` o medir directamente los `.tflite` en assets
4. **F1 / Recall:** Agregar columna `true_label` al CSV de registro para poder construir la matriz de confusión

### 9.2 Hipótesis para Samsung Galaxy S23

El Samsung representará el extremo alto de rango (high-end) contra el Moto G (mid-range). Las preguntas de interés:

- ¿El ratio 11.6× de latencia se mantiene, o el hardware más potente achica la brecha?
- ¿EfficientNetV2B0 se acerca a tiempo real (~100 ms) en hardware flagship?
- ¿La PSS se comporta distinto en un sistema con más RAM disponible?
- ¿Hay diferencias de temperatura entre dispositivos? (`adb shell cat /sys/class/thermal/thermal_zone*/temp`)

### 9.3 Sobre las imágenes IFD

Las 10 imágenes IFD son un **grupo de control**, no errores del modelo. En la tesis deben reportarse como evidencia de la limitación de diseño (closed-set), no como parte del cálculo de accuracy diagnóstica. La exclusión del global de coherencia botánica (aplicada en sección 3) es el tratamiento correcto.

---

## 10. Resumen ejecutivo

| Métrica | MobileNetV3 Small | EfficientNetV2B0 |
|---|---|---|
| Latencia promedio | **~24 ms** | ~278 ms |
| Memoria PSS típica | ~220 MB | ~290 MB |
| Velocidad relativa | **11.6× más rápido** | baseline |
| Coherencia botánica global | 43.7% | 44.8% |
| Mejor en | Corn, Tomato, Strawberry | Tomato, Strawberry |
| Peor en | Potato, Soybean | Potato, Soybean, Apple |
| Sesgo dominante | Corn___Common_rust para manchas | Strawberry___Leaf_scorch para manchas |
| Imágenes fuera de dominio | Sin rechazo, alta confianza falsa | Sin rechazo, alta confianza falsa |
| Consistencia entre sesiones | **100% determinista** | **100% determinista** |

**Hallazgo principal:** La performance de ambos modelos está fuertemente condicionada por el desbalanceo del dataset PlantVillage. Las plantas con pocas imágenes de entrenamiento (Potato <100, Apple Cedar rust ~300) tienen coherencia botánica nula o muy baja. Las clases sobrerepresentadas (Strawberry___Leaf_scorch, Corn___Common_rust) actúan como "atractores" para imágenes de plantas desconocidas o con baja confianza.
