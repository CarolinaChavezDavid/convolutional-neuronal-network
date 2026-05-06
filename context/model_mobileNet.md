# Comparativa de Versiones MobileNet

## Resumen comparativo

| Version | Diferencia principal | Ventaja | Limitacion |
|---|---|---|---|
| MobileNetV1 | Introduce convoluciones depthwise separable. | Muy simple y liviano. | Menor precision que versiones posteriores. |
| MobileNetV2 | Usa inverted residuals y linear bottlenecks. | Muy buen balance precision/tamano; facil de exportar a TFLite. | Puede quedarse corto en datasets mas complejos o fotos reales. |
| MobileNetV3 | Agrega busqueda NAS, optimizacion para CPU movil, activaciones como hard-swish y bloques SE. | Mejor que V2 en eficiencia/precision movil. | Mas complejo, pero todavia practico. |
| MobileNetV4 | Introduce bloques UIB, Mobile MQA y busqueda optimizada para distintos aceleradores moviles. | Mas moderno y pensado para hardware movil actual. | Menos estandar en workflows simples de Keras/TFLite que V2/V3. |
| MobileNetV5 | En KerasHub aparece como encoder grande para Gemma 3n, con preset de aproximadamente 294M parametros. | Puede tener mucha capacidad representacional. | Demasiado pesado para una app movil comun; el entrenamiento inicial ya mostro tiempos altos. |

## Recomendacion para la tesis

Para esta tesis, el modelo candidato principal recomendado para reemplazar o mejorar el baseline actual es MobileNetV3.

Razon principal: el objetivo no es solamente obtener mejor accuracy en notebook, sino integrar el modelo en una aplicacion Android. MobileNetV5 es interesante como experimento exploratorio, pero el preset disponible actualmente en KerasHub es demasiado grande para una app movil comun. MobileNetV3 ofrece una evolucion natural sobre MobileNetV2 manteniendo una arquitectura disenada para dispositivos moviles, exportable a TFLite y defendible metodologicamente.

## Estrategia experimental recomendada

- Baseline: MobileNetV2 actual.
- Candidato principal: MobileNetV3.
- Experimento exploratorio: MobileNetV5, documentado como modelo de alta capacidad pero probablemente no viable para movil por tamano y latencia.

## Criterios de decision

- Accuracy y perdida en test.
- Tamano final del archivo `.tflite`.
- Tiempo promedio de inferencia.
- Compatibilidad con TensorFlow Lite y Android.
- Balance entre mejora de precision y costo computacional.

## Preparacion y entrenamiento de MobileNetV3

El notebook `MobileNetV3.ipynb` implementa MobileNetV3 como candidato principal para la tesis, manteniendo la comparacion contra el baseline MobileNetV2.

Preparacion del dataset:

- Se carga `tensorflow_datasets/plant_village` con `as_supervised=True`.
- Se conservan las 38 clases para que el modelo sea comparable con la app Android actual.
- Se usa split reproducible 80/10/10: entrenamiento, validacion y test.
- Las imagenes se redimensionan a `224x224`.
- Las imagenes se convierten a `float32` y se normalizan a `[-1, 1]` con `(pixel - 127.5) / 127.5`.
- Se usa `batch(32)` y `prefetch(AUTOTUNE)` para mejorar el rendimiento de lectura.
- El notebook incluye grafica de distribucion de clases para revisar desbalance antes de entrenar.

Construccion del modelo:

- Se usa `tf.keras.applications.MobileNetV3Small` como variante recomendada para movil.
- Se carga con pesos `imagenet`, `include_top=False` e `include_preprocessing=False`.
- Como `include_preprocessing=False`, el preprocessing queda controlado manualmente en el notebook y alineado con la app Android.
- El backbone se congela inicialmente para entrenar solo la cabeza.
- La cabeza agregada contiene `GlobalAveragePooling2D`, `Dropout(0.2)` y una capa `Dense(38, activation="softmax")`.

Entrenamiento:

- Fase 1: entrenar solo la cabeza de clasificacion con el backbone congelado.
- Fase 2 opcional: en modo `full`, descongelar solo las ultimas capas del backbone y hacer fine-tuning con learning rate bajo (`1e-5`).
- Las capas BatchNorm se mantienen congeladas durante fine-tuning para estabilidad.
- Se usan callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau` y `CSVLogger`.
- El notebook inicia en `TRAINING_MODE = 'smoke_test'` para validar rapido el pipeline antes de una corrida completa.

Graficas y evaluacion:

- Grafica de distribucion de clases.
- Curvas de accuracy y loss para train/validation.
- Matriz de confusion.
- Tabla de errores con mayor confianza.
- Comparacion contra MobileNetV2 por tamano TFLite y tiempo de inferencia local cuando exista exportacion.

Decision metodologica:

MobileNetV3Small es el primer candidato recomendado porque mejora la familia MobileNetV2 sin saltar a un modelo excesivamente pesado como MobileNetV5. Esto mantiene el foco de la tesis en precision practica y viabilidad para Android.


## Corrida MobileNetV3 - 2026-05-04 10:58:25

- Notebook: `/home/nicolaurenti/Documents/convolutional-neuronal-network/MobileNetV2/MobileNetV3.ipynb`
- Modelo: `MobileNetV3Small` con pesos ImageNet, `include_preprocessing=False`
- Training mode: `full`
- Dataset: PlantVillage, 38 clases, split 80/10/10, seed `123`
- Effective split sizes: train `43442`, validation `5430`, test `5431`
- Test accuracy: `0.9856`
- Test loss: `0.0485`
- TFLite FP16 size MB: `1.90887451171875`
- Decision provisional: comparar contra MobileNetV2 por accuracy, tamano y latencia antes de integrar a Android.

## Experimentacion

Antes de subir el modelo a la aplicacion Android, se realiza una fase de experimentacion local sobre el modelo TFLite exportado para validar su comportamiento con imagenes reales y registrar las metricas definitivas.

### Modelo bajo evaluacion

- Archivo: `saved_models/mobilenetv3_small_v1_fp16.tflite`
- Cuantizacion: FP16
- Clases: 38 (PlantVillage completo)
- Tamano: ~1.91 MB

### Notebook de experimentacion

- Ruta: `MobileNet/experimentation/mobilenet_experimentation.ipynb`
- Permite cargar imagenes desde el computador usando un widget interactivo o especificando la ruta directamente.
- Realiza inferencia con el interprete TFLite y muestra:
  - Clasificacion top-1 y top-5 con confianza
  - Latencia promedio de inferencia (ms) sobre N ejecuciones
  - Tamano del modelo en MB
- El preprocesamiento es identico al del entrenamiento: normalizacion a `[-1, 1]` con `(pixel - 127.5) / 127.5`.

### Metricas a registrar en el Excel de seguimiento

Hoja de registro: https://docs.google.com/spreadsheets/d/13_oqj5JWWKHnTzcsgkbcmKLMdpNZmSif45y6sf3gGeo/edit?gid=1383304375

Columnas relevantes a completar:
- Nombre del modelo
- Tipo de cuantizacion
- Tamano TFLite (MB)
- Latencia promedio local (ms)
- Clases
- Observaciones sobre imagenes de prueba reales
