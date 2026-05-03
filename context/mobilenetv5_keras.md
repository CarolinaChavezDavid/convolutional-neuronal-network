# MobileNetV5 con KerasHub - Bitácora técnica

## Contexto y problema actual con MobileNetV2

La aplicación Android actualmente usa `mobilenetv2_v3_fp16.tflite` con 38 clases de PlantVillage. El archivo pesa aproximadamente 4.35 MB y está integrado en `app/src/main/assets` junto con `labels.txt`. La percepción actual es que los resultados no son suficientemente buenos, por lo que se probará MobileNetV5 con Keras/KerasHub antes de decidir cualquier reemplazo en móvil.

## Hipótesis del experimento MobileNetV5

MobileNetV5 podría mejorar la calidad de clasificación frente a MobileNetV2, pero su preset oficial disponible en KerasHub (`mobilenetv5_300m_enc_gemma3n`) es mucho más grande que MobileNetV2. La hipótesis se evaluará con tres criterios al mismo tiempo: precisión, tamaño del `.tflite` y tiempo promedio de inferencia local.

## Ambiente usado

- Ambiente Conda: `learning_ai`
- Python: `/opt/anaconda3/envs/learning_ai/bin/python`
- TensorFlow inicial observado: `2.18.0`
- Keras inicial observado: `3.8.0`
- Acción realizada el 2026-05-03: instalación de `keras-hub` e `ipykernel`
- Versiones después de instalar KerasHub:
  - TensorFlow: `2.20.0`
  - Keras: `3.14.0`
  - KerasHub: `0.28.0`
- Kernel Jupyter registrado: `Python (learning_ai)` / `learning_ai`

## Decisiones

- Mantener `tensorflow_datasets/plant_village` con 38 clases para comparar contra el modelo actual de la app.
- Mantener preprocessing a `[-1, 1]`, compatible con `PIXEL_MEAN = 127.5` y `PIXEL_STD = 127.5` en Android.
- Entrenar en dos fases: cabeza congelada y luego fine-tuning con learning rate bajo.
- Exportar a SavedModel, TFLite FP32 y TFLite FP16.
- No reemplazar todavía el modelo Android; primero se comparan métricas y viabilidad móvil.
- Si el preset preentrenado de MobileNetV5 no descarga o no exporta bien, registrar fallback explícito en el notebook.

## Dataset PlantVillage

Datos confirmados desde el notebook `MobileNetKerasV5.ipynb`:

- Total examples: 54303
- Split sizes: `{'train': 43442, 'validation': 5430, 'test': 5431}`
- Number of classes: 38
- First labels: `['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy']`

## Arquitectura cargada desde preset

Resultado confirmado al construir el modelo en el notebook:

- Model source: `mobilenetv5_300m_enc_gemma3n: pretrained weights`
- Estado de carga: `Loaded preset with pretrained weights.`
- Modelo Keras: `mobile_net_v5_image_classifier`
- Entrada: `(None, 224, 224, 3)`
- Backbone: `MobileNetV5Backbone`
- Salida del backbone: `(None, 16, 16, 2048)`
- Parametros del backbone: `294,284,096`
- Pooling: `SelectAdaptivePool2d`, salida `(None, 2048)`
- Dropout: salida `(None, 2048)`
- Clasificador final: `Dense`, salida `(None, 38)`, parametros `77,862`
- Total params: `294,361,958` aprox. `1.10 GB`
- Trainable params: `77,862` aprox. `304.15 KB`
- Non-trainable params: `294,284,096` aprox. `1.10 GB`

Decision registrada: en la primera fase solo queda entrenable la cabeza de clasificacion. El backbone preentrenado queda congelado, por eso los parametros entrenables son solamente `77,862`.

## Referencia de arquitectura para fallback ligero

El fallback `build_custom_lightweight_backbone()` no es el preset oficial de 294M parametros. Se diseno como una variante pequena inspirada en el ejemplo oficial de KerasHub para crear un `MobileNetV5Backbone` con configuracion custom y pesos aleatorios.

Referencia oficial: [KerasHub MobileNetV5Backbone](https://keras.io/keras_hub/api/models/mobilenetv5/mobilenetv5_backbone/).

La documentacion oficial muestra una configuracion custom con bloques `er` y `uir`, listas `stackwise_*`, kernels depthwise, expansion ratios, parametros de atencion en `0`, y `use_msfa=False`. En este notebook se tomo esa misma estructura como base y se extendio a 3 stacks (`24 -> 48 -> 96` filtros) para tener un fallback mas expresivo que el ejemplo minimo, pero todavia mucho mas pequeno y simple que el preset `mobilenetv5_300m_enc_gemma3n`.

## Decision de entrenamiento rapido

Durante la primera ejecucion, el entrenamiento quedo mas de 30 minutos en `Epoch 1/8` con el kernel ocupado. Causa probable: el preset `mobilenetv5_300m_enc_gemma3n` tiene aproximadamente 294M parametros en el backbone; aunque este congelado, cada batch hace forward pass por todo el backbone. Con 43,442 imagenes de entrenamiento y batch size 32, una epoca completa son aproximadamente 1,358 pasos.

Decision tomada: agregar `TRAINING_MODE = 'smoke_test'` al notebook para validar el pipeline antes del entrenamiento completo.

Configuracion smoke test:

- Train examples: 1,024
- Validation examples: 256
- Test examples: 256
- Head epochs: 2
- Fine-tuning: desactivado
- Exportacion TFLite: desactivada hasta pasar a `TRAINING_MODE = 'full'`

Para una corrida completa, cambiar `TRAINING_MODE` a `'full'`, pero solo despues de confirmar que el smoke test corre correctamente y preferiblemente con aceleracion adecuada.

## Resultados por corrida

Aún no hay resultados de entrenamiento completos. El notebook quedó preparado para agregar automáticamente una entrada nueva en esta sección al terminar una corrida.

## Conclusión provisional y siguientes acciones

- Ejecutar `/Users/carolinachavez/convolutional-neuronal-network/MobileNetV2/MobileNetKerasV5.ipynb` desde Jupyter con el kernel `Python (learning_ai)`.
- Revisar si MobileNetV5 mejora accuracy sin producir un `.tflite` demasiado grande o lento.
- Si MobileNetV5 es demasiado pesado para móvil, documentar la decisión y considerar alternativas más compactas antes de tocar la app Android.
