# EfficientNetV2 — Documentacion del notebook

Notebook: `EfficientNet/EfficientNetV2B0.ipynb`
Proposito: entrenamiento, evaluacion y exportacion de EfficientNetV2 (variantes B0/B1/B2/B3/S) sobre PlantVillage para comparar contra MobileNetV2 (baseline Android) y MobileNetV3 (candidato principal).

---

## Por que EfficientNetV2 sobre EfficientNetV1

EfficientNetV2 es una evolucion directa de EfficientNetV1 que introduce tres mejoras relevantes para este proyecto:

1. **Bloques Fused-MBConv en las capas iniciales:** las primeras etapas del backbone reemplazan los bloques depthwise separables de V1 por convoluciones estandar fusionadas, lo que reduce la latencia en hardware moderno hasta 4x mas rapido que V1 con accuracy equivalente o superior.

2. **`include_preprocessing` soportado:** EfficientNetV2 acepta el parametro `include_preprocessing=False`, igual que MobileNetV3. Con `include_preprocessing=False`, el backbone no aplica ninguna normalizacion interna y delega esa responsabilidad al pipeline. Esto es una diferencia critica respecto a EfficientNetV1, que aplica su propio `Rescaling` interno y produce un `TypeError` si se le pasa `include_preprocessing`.

3. **Preprocessing explicito `[0, 1]`:** al usar `include_preprocessing=False`, el pipeline entrega `pixel / 255.0`, produciendo valores en `[0, 1]`. Esto contrasta con MobileNet (`[-1, 1]`) y con EfficientNetV1 (`[0, 255]` sin normalizar). La diferencia es relevante para la integracion Android: si se despliega EfficientNetV2, el preprocesamiento de la app debe cambiarse.

**Variante por defecto:** `B0` (224x224, ~5.9M parametros). Cambiar `MODEL_VARIANT` a `'B1'`, `'B2'`, `'B3'` o `'S'` sin modificar ninguna otra celda.

---

## Descripcion celda por celda

---

### Celda 1 — Titulo y contexto (markdown)

Introduce el proposito del notebook y las dos decisiones de diseno mas importantes:

**Por que V2 sobre V1:** EfficientNetV2 entrena hasta 4x mas rapido gracias a los bloques Fused-MBConv. Ademas resuelve el `TypeError` que ocurria en V1 al intentar usar `include_preprocessing`, ya que V2 si acepta este parametro.

**Preprocessing:** se usa `include_preprocessing=False` con normalizacion manual `pixel / 255.0`, produciendo entrada en `[0, 1]`. Es distinto a MobileNetV3 (`[-1, 1]`) y a EfficientNetV1 (`[0, 255]` sin normalizar el pipeline). Esta decision se documenta explicitamente porque tiene impacto directo en el codigo Android de la app si el modelo llegara a integrarse.

---

### Celda 2 — Encabezado de seccion: Configuracion (markdown)

Separador visual que indica el inicio del bloque de configuracion.

---

### Celda 3 — Configuracion (codigo)

Centraliza todos los hiperparametros, rutas y flags en un unico lugar. Ejecutar esta celda es el unico paso necesario para adaptar el notebook a otro entorno.

**Imports:**
```python
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from sklearn.metrics import classification_report, confusion_matrix
```
Librerias estandar del proyecto. `tensorflow_datasets` provee PlantVillage directamente. `sklearn` se usa para el reporte de clasificacion y la matriz de confusion.

**Semillas de reproducibilidad:**
```python
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
Fijar la semilla en `random`, `numpy` y `tensorflow` garantiza que el shuffle del dataset, la inicializacion de pesos y los dropouts sean identicos en cada ejecucion. El mismo `SEED = 123` se usa en MobileNetV2 y MobileNetV3, haciendo los splits directamente comparables entre notebooks.

**Rutas:**
- `PROJECT_DIR`: directorio de salida de este modelo (checkpoints, logs, TFLite).
- `MOBILENET_DIR`: apunta a los modelos MobileNet ya entrenados para mostrarlos en la tabla comparativa.
- `ANDROID_ASSETS_DIR`: directorio de assets de la app Android para verificar que las labels coincidan.
- `CONTEXT_DOC`: este mismo archivo (`model_efficientNetV2.md`), donde se registran automaticamente los resultados de cada corrida completa.

**Variante del modelo:**
```python
MODEL_VARIANT = 'B0'   # opciones: 'B0', 'B1', 'B2', 'B3', 'S'
MODEL_NAME    = f'efficientnetv2{MODEL_VARIANT.lower()}_v1'
```
El diccionario `backbone_map` en `build_efficientnetv2` mapea esta cadena a la clase de Keras correspondiente. Cambiar esta variable es suficiente para cambiar de variante; no hay que modificar ninguna otra celda.

Las variantes disponibles y sus tamanos de entrada nativos:

| Variante | Parametros aprox. | Entrada nativa | Recomendacion |
|---|---|---|---|
| `B0` | ~5.9M | 224x224 | defecto, comparable con MobileNetV3 Small |
| `B1` | ~6.9M | 240x240 | leve mejora de accuracy |
| `B2` | ~8.8M | 260x260 | balance accuracy/tamano |
| `B3` | ~14M | 300x300 | mayor accuracy, mas pesado |
| `S` | ~21M | 384x384 | no recomendado para movil |

**Modo de entrenamiento:**
```python
TRAINING_MODE = 'smoke_test'
```
El modo `smoke_test` usa 2048 imagenes de entrenamiento, 512 de validacion y 512 de prueba, con 3 epocas de cabeza. Su unico proposito es verificar que el pipeline completo funciona antes de lanzar la corrida completa. Solo cambiar a `'full'` cuando el smoke test pasa sin errores.

**Flags de control:**
```python
RUN_TRAINING        = True
RUN_FINE_TUNING     = TRAINING_MODE == 'full'
RUN_EXPORT          = TRAINING_MODE == 'full'
RUN_DATASET_PROFILE = True
```
El fine-tuning y la exportacion TFLite se desactivan automaticamente en `smoke_test`, reduciendo el tiempo de validacion del pipeline a menos de 2 minutos.

**Epocas por modo:**
```python
HEAD_EPOCHS      = 3  if TRAINING_MODE == 'smoke_test' else 15
FINE_TUNE_EPOCHS = 0  if TRAINING_MODE == 'smoke_test' else 10
```
EfficientNetV2 converge mas rapido que V1, por lo que 15 epocas de cabeza son suficientes para alcanzar accuracy alta. En V1 se usaban 10.

**Hiperparametros:**
- `HEAD_LR = 1e-3`: learning rate alto para la primera fase (solo se entrena la cabeza).
- `FINE_TUNE_LR = 1e-5`: learning rate bajo para la segunda fase. Un LR alto en fine-tuning destruye los pesos preentrenados de ImageNet.
- `FINE_TUNE_LAST_N_LAYERS = 20`: capas finales del backbone a descongelar en fine-tuning.

---

### Celda 4 — Encabezado de seccion: Carga del dataset (markdown)

Documenta la decision de preprocessing antes de mostrar el codigo:

**Preprocessing EfficientNetV2:** con `include_preprocessing=False` el backbone no aplica ninguna normalizacion. El pipeline entrega `pixel / 255.0`, produciendo `[0, 1]`. Esto es diferente a MobileNetV3 (`[-1, 1]`) y a EfficientNetV1 (`[0, 255]` como float32 sin normalizar).

---

### Celda 5 — Carga del dataset y preparacion (codigo)

**Funcion `preprocess`:**
```python
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # [0, 1] — requerido por EfficientNetV2 con include_preprocessing=False
    return image, label
```
Tres operaciones en orden:
1. **Redimensionar:** PlantVillage tiene imagenes de tamano variable; el modelo espera entrada fija de `224x224`.
2. **Convertir a float32:** TensorFlow requiere punto flotante para las operaciones de la red.
3. **Dividir por 255.0:** normaliza a `[0, 1]`. Con `include_preprocessing=False`, el backbone de EfficientNetV2 espera esta distribucion de entrada exacta.

A diferencia de MobileNetV3, no se usa `(pixel - 127.5) / 127.5`. A diferencia de EfficientNetV1, si se normaliza (V1 espera `[0, 255]` sin normalizar). Esta distincion es la causa mas comun de degradacion silenciosa de accuracy si se mezclan notebooks sin prestar atencion.

**Carga y split del dataset:**
```python
raw_ds, info = tfds.load('plant_village', split='train', as_supervised=True, with_info=True)
```
`tensorflow_datasets` descarga y cachea PlantVillage automaticamente. `as_supervised=True` devuelve tuplas `(imagen, etiqueta)`. `with_info=True` provee los nombres de clases y el conteo de ejemplos.

```python
shuffled_ds = raw_ds.shuffle(total_examples, seed=SEED, reshuffle_each_iteration=False)
train_raw   = shuffled_ds.take(train_size)
val_raw     = shuffled_ds.skip(train_size).take(val_size)
test_raw    = shuffled_ds.skip(train_size + val_size)
```
Shuffle global con seed fija antes de dividir. `reshuffle_each_iteration=False` garantiza que el orden sea el mismo en cada epoch, produciendo splits identicos a los de MobileNetV2 y MobileNetV3. Division: 80% train / 10% val / 10% test.

**Pipeline de entrada:**
```python
train_ds = train_raw.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
```
- `.map(..., num_parallel_calls=AUTOTUNE)`: aplica `preprocess` en paralelo usando todos los cores disponibles.
- `.batch(32)`: agrupa en batches de 32 imagenes.
- `.prefetch(AUTOTUNE)`: mientras el modelo procesa el batch actual, la CPU prepara el siguiente, eliminando tiempo muerto de I/O.

---

### Celda 6 — Guardado de labels y verificacion Android (codigo)

```python
labels_path = PROJECT_DIR / f'labels_efficientnetv2{MODEL_VARIANT.lower()}.txt'
labels_path.write_text('\n'.join(class_names) + '\n')
```
Guarda las 38 clases en el orden exacto que devuelve `tensorflow_datasets`. La clase en posicion `i` corresponde al indice de salida `i` del modelo. Este orden debe coincidir con el que el modelo usa internamente.

```python
android_labels = [l.strip() for l in android_labels_path.read_text().splitlines() if l.strip()]
labels_match_android = android_labels == class_names
```
Compara las labels generadas contra el archivo `labels.txt` que ya esta en la app Android. Si `labels_match_android = False`, el modelo predecira indices que no corresponden a las etiquetas que la app muestra al usuario. Este chequeo debe pasar antes de integrar cualquier modelo nuevo en la app.

---

### Celda 7 — Encabezado de seccion: Grafica de distribucion (markdown)

Separador visual de seccion.

---

### Celda 8 — Grafica de distribucion de clases (codigo)

```python
counts = np.zeros(num_classes, dtype=np.int64)
for _, labels in raw_ds.batch(2048):
    counts += np.bincount(labels.numpy(), minlength=num_classes)
```
Cuenta cuantas imagenes hay por clase recorriendo el dataset completo en batches de 2048. PlantVillage tiene desbalance moderado: algunas clases tienen mas de 5000 imagenes mientras otras tienen menos de 500. Visualizar esto antes de entrenar ayuda a anticipar que clases tendran F1 bajo y donde puede necesitarse analisis adicional.

La grafica se guarda en `dataset_distribution_v2.png` para incluirla en la tesis sin necesidad de volver a ejecutar el notebook.

---

### Celda 9 — Encabezado de seccion: Referencia de modelos existentes (markdown)

Separador visual de seccion.

---

### Celda 10 — Referencia de modelos existentes (codigo)

```python
def file_size_mb(path):
    return round(path.stat().st_size / (1024 * 1024), 4) if path.exists() else None
```
Utilitario reutilizado en todo el notebook para reportar tamanos de archivos TFLite en MB.

Antes de entrenar, el notebook muestra el tamano y la accuracy de los modelos MobileNetV2 y MobileNetV3 ya exportados. Esto establece el punto de referencia: si EfficientNetV2B0 es significativamente mas grande sin una mejora proporcional de accuracy, la decision de no integrarlo a Android queda justificada cuantitativamente desde el inicio.

La columna `preprocessing` de la tabla referencia resume las diferencias de pipeline entre modelos, visible en un solo lugar para comparacion rapida.

---

### Celda 11 — Encabezado de seccion: Construccion del modelo (markdown)

Documenta las decisiones de arquitectura antes del codigo:

- EfficientNetV2 SI acepta `include_preprocessing=False`, a diferencia de V1 que lanza `TypeError`.
- Con `include_preprocessing=False` el backbone no aplica rescaling y el pipeline es responsable de entregar `[0, 1]`.
- La cabeza agrega `GlobalAveragePooling2D`, `BatchNormalization`, `Dropout(0.3)` y `Dense(38, softmax)`.

---

### Celda 12 — Construccion del modelo EfficientNetV2 (codigo)

Esta es la celda central del notebook.

**Diccionario de variantes:**
```python
backbone_map = {
    'B0': tf.keras.applications.EfficientNetV2B0,
    'B1': tf.keras.applications.EfficientNetV2B1,
    'B2': tf.keras.applications.EfficientNetV2B2,
    'B3': tf.keras.applications.EfficientNetV2B3,
    'S' : tf.keras.applications.EfficientNetV2S,
}
```
Mapea la cadena `MODEL_VARIANT` a la clase de Keras correspondiente. Agregar o quitar variantes solo requiere modificar este diccionario.

**Instanciacion del backbone:**
```python
backbone = backbone_map[variant](
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    include_preprocessing=False,  # el pipeline ya entrega [0, 1]
)
```
- `include_top=False`: descarta la capa de clasificacion original de ImageNet (1000 clases). Solo se conservan las capas convolucionales que extraen features.
- `weights='imagenet'`: carga pesos preentrenados en ImageNet. Estos pesos ya reconocen texturas, bordes y patrones visuales generales, lo que acelera el entrenamiento en PlantVillage.
- **`include_preprocessing=False`**: le dice al backbone que no aplique ninguna normalizacion interna. El pipeline ya entrega `[0, 1]`. Este parametro es VALIDO en EfficientNetV2 pero produce `TypeError` en EfficientNetV1 — es la diferencia tecnica clave entre los dos notebooks.

**Congelar el backbone:**
```python
backbone.trainable = False
```
Durante la primera fase de entrenamiento el backbone queda completamente congelado. Solo la cabeza aprende. Esto evita que los pesos de ImageNet se destruyan en las primeras epocas cuando la cabeza aun produce gradientes muy ruidosos.

**Construccion de la cabeza:**
```python
inputs  = keras.Input(shape=(*IMG_SIZE, 3), name='image')
x       = backbone(inputs, training=False)
x       = keras.layers.GlobalAveragePooling2D(name='gap')(x)
x       = keras.layers.BatchNormalization(name='head_bn')(x)
x       = keras.layers.Dropout(0.3, name='dropout')(x)
outputs = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
```

Cada capa tiene un proposito especifico:

- **`backbone(inputs, training=False)`**: mantiene las capas BatchNormalization del backbone en modo inferencia incluso durante el entrenamiento de la cabeza, preservando las estadisticas de normalizacion aprendidas en ImageNet.

- **`GlobalAveragePooling2D`**: convierte el mapa de activaciones de forma `(H, W, C)` en un vector de forma `(C,)` calculando el promedio espacial. Reduce drasticamente el numero de parametros respecto a un `Flatten` y hace al modelo invariante a pequenas traslaciones.

- **`BatchNormalization` en la cabeza**: normaliza las activaciones del backbone antes de la capa densa. EfficientNetV2, al igual que V1, es sensible a cambios de distribucion durante transfer learning; agregar BN en la cabeza estabiliza el entrenamiento.

- **`Dropout(0.3)`**: desactiva aleatoriamente el 30% de las neuronas durante cada paso de entrenamiento. Reduce el sobreajuste. Se usa 0.3 en lugar del 0.2 de MobileNetV3 porque EfficientNetV2 tiene mayor capacidad y mayor riesgo de memorizar el conjunto de entrenamiento.

- **`Dense(38, activation='softmax')`**: capa de clasificacion final. Produce una distribucion de probabilidad sobre las 38 clases.

**Compilacion:**
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=HEAD_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```
- `Adam`: optimizador adaptativo que ajusta el learning rate por parametro.
- `sparse_categorical_crossentropy`: funcion de perdida para clasificacion multiclase cuando las etiquetas son enteros (no one-hot). PlantVillage devuelve etiquetas como indices.

---

### Celda 13 — Encabezado de seccion: Entrenamiento de la cabeza (markdown)

Menciona que EfficientNetV2 converge mas rapido que V1 gracias a los bloques Fused-MBConv en las capas iniciales, lo que justifica usar 15 epocas en lugar de 10.

---

### Celda 14 — Entrenamiento de la cabeza (codigo)

**Callbacks:**

```python
keras.callbacks.ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
```
Guarda el modelo completo en disco solo cuando `val_accuracy` mejora. Al final del entrenamiento, el archivo en disco contiene el mejor modelo visto, no el del ultimo epoch.

```python
keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
```
Detiene el entrenamiento si `val_accuracy` no mejora en 4 epochs consecutivos. `restore_best_weights=True` regresa los pesos al mejor epoch antes de detenerse.

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7)
```
Si `val_loss` no mejora en 2 epochs, reduce el learning rate multiplicandolo por 0.3. Ayuda al modelo a salir de mesetas sin ajustar el LR manualmente. El limite `min_lr=1e-7` evita que el LR se haga tan pequeno que el entrenamiento se paralice efectivamente.

```python
keras.callbacks.CSVLogger(str(training_log_path))
```
Guarda accuracy y loss de cada epoch en un archivo CSV. Permite reconstruir las graficas de entrenamiento sin reejecutar el notebook.

---

### Celda 15 — Encabezado de seccion: Fine-tuning controlado (markdown)

Documenta la estrategia de fine-tuning: descongelar las ultimas `FINE_TUNE_LAST_N_LAYERS` capas del backbone, manteniendo las capas `BatchNormalization` del backbone siempre congeladas para preservar las estadisticas de ImageNet.

---

### Celda 16 — Fine-tuning controlado (codigo)

```python
backbone.trainable = True
for layer in backbone.layers[:-FINE_TUNE_LAST_N_LAYERS]:
    layer.trainable = False
for layer in backbone.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False
```

Tres pasos en orden:
1. Se habilita el entrenamiento del backbone completo.
2. Se vuelven a congelar todas las capas excepto las ultimas 20. Las capas iniciales reconocen features genericas (bordes, texturas) utiles para cualquier dominio y no necesitan ajustarse para PlantVillage.
3. Todas las capas `BatchNormalization` del backbone se congelan independientemente de su posicion. Si las BN se descongelan, sus estadisticas de media y varianza cambian y pueden desestabilizar el entrenamiento.

```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), ...)
history_finetune = model.fit(..., initial_epoch=history_head.epoch[-1] + 1, ...)
```
Se recompila con `FINE_TUNE_LR = 1e-5`, dos ordenes de magnitud menor que `HEAD_LR`. `initial_epoch` indica a Keras que el historial continua desde donde termino la fase 1, produciendo graficas continuas entre fases.

---

### Celda 17 — Encabezado de seccion: Graficas de entrenamiento (markdown)

Separador visual de seccion.

---

### Celda 18 — Graficas de entrenamiento (codigo)

```python
def history_to_frame(*histories):
    ...
    frame['phase'] = name
```
Combina los historiales de la fase 1 (cabeza) y fase 2 (fine-tuning) en un unico DataFrame con una columna `phase`. Esto permite diferenciar visualmente en las graficas donde termina una fase y empieza la otra, util para detectar si el fine-tuning mejoro o empeoro el modelo.

Se generan dos graficas lado a lado:
- **Accuracy**: train y val accuracy por epoch. Una brecha grande entre ambas indica sobreajuste.
- **Loss**: train y val loss por epoch. La val loss debe decrecer; si sube mientras el train loss baja, el modelo esta memorizando el conjunto de entrenamiento.

Las graficas se guardan en `training_graph_v2b0.png` para incluirlas en la tesis sin reejecutar el notebook.

---

### Celda 19 — Encabezado de seccion: Evaluacion (markdown)

Separador visual de seccion.

---

### Celda 20 — Evaluacion en el conjunto de prueba (codigo)

```python
if checkpoint_path.exists():
    eval_model = keras.models.load_model(checkpoint_path)
```
Carga el mejor checkpoint guardado durante el entrenamiento. Si el notebook se ejecuto en partes o el kernel se reinicio, este paso recupera el mejor modelo sin necesidad de reentrenar.

```python
y_true, y_pred, y_conf = [], [], []
for images, labels in test_ds:
    probs = eval_model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy().tolist())
    y_pred.extend(preds.tolist())
    y_conf.extend(probs.max(axis=1).tolist())
```
Recorre el conjunto de prueba en batches, guarda las etiquetas reales, las predicciones y la confianza (probabilidad maxima del softmax) de cada prediccion. La confianza es importante para el analisis de errores: un error con confianza 0.99 es mas preocupante para la app que uno con confianza 0.51.

```python
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
```
Muestra precision, recall y F1 por cada una de las 38 clases con 4 decimales. Las clases con F1 bajo son candidatas a analisis adicional en la matriz de confusion.

---

### Celda 21 — Matriz de confusion (codigo)

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, cmap='Greens', xticklabels=class_names, yticklabels=class_names)
```
La matriz de confusion de 38x38 muestra cuantas imagenes de cada clase real fueron clasificadas en cada clase predicha. La diagonal principal representa predicciones correctas. Celdas fuera de la diagonal son errores. Un bloque de errores entre dos clases especificas (ej. `Tomato___Early_blight` y `Tomato___Late_blight`) indica confusion sistematica entre enfermedades visualmente similares.

Se guarda en `confusion_matrix_v2b0.png` para la tesis.

---

### Celda 22 — Errores de alta confianza (codigo)

```python
errors = [
    {'true_label': class_names[t], 'predicted_label': class_names[p], 'confidence': c}
    for t, p, c in zip(y_true, y_pred, y_conf) if t != p
]
errors_df.sort_values('confidence', ascending=False).head(25)
```
Filtra solo los errores y los ordena por confianza descendente. Los errores con mayor confianza son los mas criticos para una aplicacion de diagnostico: el modelo dice estar muy seguro de algo incorrecto. Esta tabla es util para argumentar en la tesis cuales tipos de confusion son sistematicos y cuales son casos borde.

---

### Celda 23 — Encabezado de seccion: Exportacion (markdown)

Documenta el proceso de exportacion y la nota critica de integracion Android:

**Nota para integracion Android:** este modelo fue entrenado con preprocessing `[0, 1]`. Si se integra en la app, el preprocesamiento Android debe cambiar de `(pixel - 127.5) / 127.5` (MobileNet) a `pixel / 255.0` (EfficientNetV2). No hacer este cambio produce predicciones incorrectas silenciosamente.

---

### Celda 24 — Exportacion a SavedModel y TFLite (codigo)

**Exportacion a SavedModel:**
```python
eval_model.export(str(saved_model_dir))
```
Guarda el modelo en formato SavedModel de TensorFlow. Es el formato intermedio necesario para convertir a TFLite.

**Conversion FP32:**
```python
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
fp32_tflite_path.write_bytes(converter.convert())
```
Sin cuantizacion: los pesos se mantienen en `float32`. Produce el archivo mas grande pero con la precision original del modelo. Util para comparar contra FP16 y verificar que la cuantizacion no degrada la accuracy.

**Conversion FP16:**
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
fp16_tflite_path.write_bytes(converter.convert())
```
Cuantizacion a `float16`: los pesos se comprimen de 32 a 16 bits, reduciendo el tamano aproximadamente a la mitad sin perdida significativa de accuracy en la mayoria de dispositivos Android modernos. Este es el formato recomendado para despliegue movil.

---

### Celda 25 — Encabezado de seccion: Benchmark local (markdown)

Aclara que la medicion local no reemplaza la medicion en dispositivo Android real, ya que la arquitectura CPU/GPU difiere completamente.

---

### Celda 26 — Benchmark local TFLite (codigo)

```python
try:
    from ai_edge_litert.interpreter import Interpreter as LiteInterpreter
except ImportError:
    warnings.filterwarnings('ignore', message='.*tf.lite.Interpreter is deprecated.*')
    LiteInterpreter = tf.lite.Interpreter
```
Usa `ai_edge_litert` si esta disponible (reemplazante oficial de `tf.lite.Interpreter` desde TF 2.18). Si no esta instalado, usa `tf.lite.Interpreter` suprimiendo el warning de deprecacion con `warnings.filterwarnings`.

```python
for i in range(warmup + runs):
    interp.set_tensor(input_details['index'], sample)
    t0 = time.perf_counter()
    interp.invoke()
    t1 = time.perf_counter()
    if i >= warmup:
        timings_ms.append((t1 - t0) * 1000)
```
Las primeras `warmup=3` ejecuciones no se cuentan porque la primera inferencia TFLite es mas lenta por la compilacion JIT del interprete. A partir de la ejecucion 4 se miden los tiempos reales del steady-state.

El resultado reporta: promedio (`avg_ms`), mediana (`median_ms`), P95 (`p95_ms`) y desviacion estandar (`std_ms`) en milisegundos. **El P95 es la metrica mas relevante para la experiencia de usuario**: indica el tiempo de inferencia en el peor caso del 95% de las ejecuciones.

---

### Celda 27 — Encabezado de seccion: Tabla comparativa (markdown)

Separador visual de seccion.

---

### Celda 28 — Tabla comparativa de modelos (codigo)

```python
comparison_rows = [
    {'modelo': 'MobileNetV2 FP16',           'preprocessing': '[-1, 1]', 'notas': 'Baseline Android'},
    {'modelo': 'MobileNetV3 Small FP16',      'preprocessing': '[-1, 1]', 'notas': 'Candidato principal'},
    {'modelo': f'EfficientNetV2{MODEL_VARIANT} FP16', 'preprocessing': '[0, 1]', 'notas': 'Este notebook'},
]
```
Consolida en una sola tabla los tres modelos del experimento comparativo. Para MobileNetV2 y MobileNetV3 se leen los archivos TFLite ya existentes para obtener su tamano real. La columna `preprocessing` destaca la diferencia critica de pipeline entre familias de modelos.

Esta tabla es el punto de partida para la discusion de resultados en la tesis. Los campos `avg_ms_local` y `p95_ms_local` de EfficientNetV2 se llenan en esta celda desde el benchmark de la celda 26.

---

### Celda 29 — Encabezado de seccion: Registro en contexto (markdown)

Separador visual de seccion.

---

### Celda 30 — Registro en contexto (codigo)

```python
def append_run_to_context():
    timestamp  = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = [
        f'## Corrida EfficientNetV2{MODEL_VARIANT} - {timestamp}',
        f'- Preprocessing: `[0, 1]` (pixel / 255.0)',
        f'- Test accuracy: `{test_accuracy:.4f}`',
        ...
    ]
    with open(CONTEXT_DOC, 'a') as f:
        f.write('\n'.join(lines) + '\n')
```
Al finalizar una corrida completa (`TRAINING_MODE = 'full'`), agrega automaticamente un bloque de resultados a este mismo archivo (`model_efficientNetV2.md`). Incluye: timestamp, variante del modelo, preprocessing, modo de entrenamiento, tamanos del split, accuracy, loss, tamano TFLite y latencia local.

En `smoke_test`, esta celda imprime un mensaje informativo pero no escribe nada, evitando contaminar el historial con resultados de prueba.

---

## Flujo de uso recomendado

1. Ejecutar con `TRAINING_MODE = 'smoke_test'` para verificar que el pipeline funciona de principio a fin sin errores.
2. Revisar que `labels_match_android = True` en la celda 6.
3. Cambiar a `TRAINING_MODE = 'full'` y ejecutar todas las celdas para el entrenamiento completo.
4. Revisar las graficas de entrenamiento (celda 18) y la tabla de errores de alta confianza (celda 22).
5. Comparar la tabla de la celda 28 contra MobileNetV2 y MobileNetV3.
6. Si los resultados justifican integracion Android, copiar `saved_models/efficientnetv2b0_v1_fp16.tflite` a `app/src/main/assets/` y **cambiar el preprocesamiento de la app de `(pixel - 127.5) / 127.5` a `pixel / 255.0`**.

---

## Diferencias clave respecto a los otros notebooks

| Aspecto | EfficientNetV2 (este) | EfficientNetV1 | MobileNetV3 |
|---|---|---|---|
| `include_preprocessing` | `False` (valido) | No existe (TypeError) | `False` (valido) |
| Preprocessing pipeline | `pixel / 255.0` → `[0, 1]` | Solo float32 → `[0, 255]` | `(pixel - 127.5) / 127.5` → `[-1, 1]` |
| Preprocesamiento Android | `pixel / 255.0` | No normalizar | `(pixel - 127.5) / 127.5` |
| Bloques principales | Fused-MBConv (capas iniciales) + MBConv | MBConv | Bottleneck inverted residual |
| Velocidad de entrenamiento | Mas rapido que V1 | Referencia | Similar a V2 |
| Variantes disponibles | B0, B1, B2, B3, S | B0, B3, B4 | Small, Large |
| `HEAD_EPOCHS` (full) | 15 | 10 | 15 |

---

## Variantes disponibles

| Variante | Parametros | Entrada | Cambios necesarios |
|---|---|---|---|
| `B0` | ~5.9M | 224x224 | ninguno (defecto) |
| `B1` | ~6.9M | 240x240 | `MODEL_VARIANT='B1'`, `IMG_SIZE=(240,240)` |
| `B2` | ~8.8M | 260x260 | `MODEL_VARIANT='B2'`, `IMG_SIZE=(260,260)` |
| `B3` | ~14M | 300x300 | `MODEL_VARIANT='B3'`, `IMG_SIZE=(300,300)` |
| `S`  | ~21M | 384x384 | `MODEL_VARIANT='S'`, `IMG_SIZE=(384,384)` — no recomendado para movil |
