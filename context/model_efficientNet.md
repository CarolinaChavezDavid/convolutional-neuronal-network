# EfficientNetB0 — Documentacion del notebook

Notebook: `EfficientNet/EfficientNetB0.ipynb`
Proposito: entrenamiento, evaluacion y exportacion de EfficientNetB0 sobre PlantVillage para comparar contra MobileNetV2 (baseline Android) y MobileNetV3 (candidato principal).

---

## Por que EfficientNet como modelo de comparacion

EfficientNet es una familia de redes convolucionales disenada mediante busqueda de arquitectura neural (NAS) que escala ancho, profundidad y resolucion del modelo de manera conjunta y balanceada (Tan & Le, 2019). EfficientNetB0 es el punto de entrada de la familia, con aproximadamente 5.3 millones de parametros y entrada nativa de 224x224, lo que lo hace comparable en tamano a MobileNetV3 Small y viable para exportacion TFLite en Android.

La decision de usar B0 como primer punto de comparacion responde a tres criterios:
- **Tamano comparable**: B0 genera un archivo TFLite de tamano similar a MobileNetV3 Small, haciendo la comparacion justa.
- **Arquitectura distinta**: mientras MobileNet usa inverted residuals lineales (V2) o bloques SE + hard-swish (V3), EfficientNet usa MBConv con squeeze-and-excitation y escalado compuesto, lo que permite evaluar si una familia diferente de arquitecturas aporta mejoras reales sobre PlantVillage.
- **Escalabilidad**: si B0 no alcanza la accuracy de MobileNetV3, se puede escalar a B3 o B4 cambiando una sola variable, manteniendo el mismo notebook.

---

## Descripcion celda por celda

---

### Celda 1 — Titulo y contexto (markdown)

Introduce el proposito del notebook: implementar EfficientNetB0 como modelo de comparacion con el mismo protocolo que MobileNetV2 y MobileNetV3.

Declara dos decisiones de diseno que se mantienen a lo largo de todo el notebook:

**Decision de arquitectura:** se elige B0 por ser la variante mas liviana de la familia EfficientNet, comparable en tamano con MobileNetV3 Small. Si la accuracy no es suficiente, se escala a B3 cambiando `MODEL_VARIANT = 'B3'` sin modificar ninguna otra celda.

**Decision de preprocessing:** EfficientNetV1 (B0-B7) NO acepta el parametro `include_preprocessing` — es exclusivo de MobileNetV3. EfficientNet aplica su propio rescaling interno y espera entrada en `[0, 255]` como `float32`. El pipeline solo redimensiona y convierte el tipo; el backbone normaliza internamente. Esto es una diferencia importante respecto a MobileNet: si EfficientNet se integra en Android, el preprocesamiento de la app debe adaptarse a `[0, 255]` en lugar de `[-1, 1]`.

---

### Celda 2 — Configuracion (codigo)

Centraliza todos los hiperparametros y rutas del experimento en un solo lugar. Ejecutar esta celda es el unico paso necesario para adaptar el notebook a un entorno diferente.

**Imports:** se cargan las librerias estandar del proyecto. `tensorflow_datasets` provee PlantVillage directamente. `sklearn` se usa para el reporte de clasificacion y la matriz de confusion.

**Semillas de reproducibilidad:**
```python
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```
Fijar la semilla en `random`, `numpy` y `tensorflow` garantiza que el shuffle del dataset, la inicializacion de pesos y los dropouts produzcan el mismo resultado en cada ejecucion, haciendo los resultados reproducibles y comparables con MobileNetV3 que usa la misma semilla.

**Rutas:**
- `PROJECT_DIR`: directorio de salida de este modelo (checkpoints, logs, TFLite).
- `MOBILENET_DIR`: apunta a los modelos MobileNet ya entrenados para mostrarlos en la tabla comparativa.
- `ANDROID_ASSETS_DIR`: directorio de assets de la app Android para verificar que las labels coincidan.
- `CONTEXT_DOC`: archivo markdown donde se registran automaticamente los resultados de cada corrida.

**Variante del modelo:**
```python
MODEL_VARIANT = 'B0'   # opciones: 'B0', 'B3', 'B4'
```
El diccionario `backbone_map` en la funcion `build_efficientnet` mapea esta cadena a la clase de Keras correspondiente. Cambiar esta variable es suficiente para cambiar de variante.

**Modo de entrenamiento:**
```python
TRAINING_MODE = 'smoke_test'
```
El modo `smoke_test` usa solo 2048 imagenes de entrenamiento, 512 de validacion y 512 de prueba, y limita las epocas a 3. Su unico proposito es verificar que el pipeline completo (carga de datos, construccion del modelo, entrenamiento, evaluacion, exportacion) funciona correctamente antes de lanzar la corrida completa. Solo cambia a `'full'` cuando el smoke test pasa sin errores.

**Flags de control:**
```python
RUN_TRAINING    = True
RUN_FINE_TUNING = TRAINING_MODE == 'full'
RUN_EXPORT      = TRAINING_MODE == 'full'
```
Estos flags desactivan automaticamente el fine-tuning y la exportacion TFLite en smoke_test, reduciendo el tiempo de validacion del pipeline a menos de 2 minutos.

**Hiperparametros de entrenamiento:**
- `HEAD_LR = 1e-3`: learning rate alto para la primera fase donde solo se entrena la cabeza.
- `FINE_TUNE_LR = 1e-5`: learning rate bajo para la segunda fase donde se descongelan capas del backbone. Un LR alto en fine-tuning destruye los pesos preentrenados.
- `FINE_TUNE_LAST_N_LAYERS = 20`: EfficientNetB0 tiene menos bloques que MobileNetV3 Large, por lo que se descongelan 20 capas en lugar de 30.

---

### Celda 3 — Carga del dataset y preparacion (markdown + codigo)

**Funcion `preprocess`:**
```python
def preprocess(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    # EfficientNetV1 aplica su propio rescaling interno: NO normalizar a [-1,1]
    # El backbone espera pixeles en [0, 255] como float32
    return image, label
```
Hace dos cosas (no tres, a diferencia de MobileNetV3):
1. Redimensiona la imagen a `224x224`. PlantVillage tiene imagenes de tamano variable; el modelo espera entrada fija.
2. Convierte de `uint8` (0-255) a `float32`. TensorFlow requiere punto flotante para las operaciones de la red.

**No se normaliza a `[-1, 1]`** porque EfficientNetV1 incluye un `Rescaling` layer interno que aplica su propia normalizacion. Si se normalizara manualmente antes, el modelo recibiria una distribucion de entrada incorrecta y el rendimiento seria significativamente peor.

**Carga y split del dataset:**
```python
raw_ds, info = tfds.load('plant_village', split='train', as_supervised=True, with_info=True)
```
`tensorflow_datasets` descarga y cachea PlantVillage automaticamente. `as_supervised=True` devuelve tuplas `(imagen, etiqueta)` en lugar de diccionarios. `with_info=True` da acceso a los nombres de clases y conteo de ejemplos.

```python
shuffled_ds = raw_ds.shuffle(total_examples, seed=SEED, reshuffle_each_iteration=False)
train_raw = shuffled_ds.take(train_size)
val_raw   = shuffled_ds.skip(train_size).take(val_size)
test_raw  = shuffled_ds.skip(train_size + val_size)
```
Se aplica un shuffle global con seed fija antes de dividir. `reshuffle_each_iteration=False` garantiza que el orden sea el mismo en cada epoch, produciendo splits identicos a los de MobileNetV3. La division es 80% train / 10% val / 10% test.

**Pipeline de entrada:**
```python
train_ds = train_raw.map(preprocess, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
```
- `.map(..., num_parallel_calls=AUTOTUNE)`: aplica `preprocess` en paralelo usando todos los cores disponibles.
- `.batch(32)`: agrupa en batches de 32 imagenes.
- `.prefetch(AUTOTUNE)`: mientras la GPU entrena el batch actual, la CPU prepara el siguiente, eliminando tiempo muerto de I/O.

---

### Celda 4 — Guardado de labels y verificacion Android (codigo)

```python
labels_path = PROJECT_DIR / f'labels_efficientnet{MODEL_VARIANT.lower()}.txt'
labels_path.write_text('\n'.join(class_names) + '\n')
```
Guarda las 38 clases en el orden exacto que devuelve `tensorflow_datasets`. Este orden debe coincidir con el que el modelo usa internamente: la clase en posicion `i` corresponde al indice de salida `i`.

```python
android_labels = [l.strip() for l in android_labels_path.read_text().splitlines() if l.strip()]
labels_match_android = android_labels == class_names
```
Compara las labels generadas contra el archivo `labels.txt` que ya esta en la app Android. Si `labels_match_android = False`, el modelo predecira indices que no corresponden a las etiquetas que la app muestra al usuario, produciendo resultados incorrectos. Este chequeo debe pasar antes de integrar cualquier modelo nuevo.

---

### Celda 5 — Grafica de distribucion de clases (codigo)

```python
counts = np.zeros(num_classes, dtype=np.int64)
for _, labels in raw_ds.batch(2048):
    counts += np.bincount(labels.numpy(), minlength=num_classes)
```
Cuenta cuantas imagenes hay por clase recorriendo el dataset en batches de 2048. PlantVillage tiene desbalance moderado: algunas clases como `Tomato___healthy` tienen mas de 5000 imagenes mientras otras tienen menos de 500. Visualizar esto antes de entrenar ayuda a anticipar clases con F1 bajo.

La grafica se guarda en `dataset_distribution.png` para incluirla en la tesis sin necesidad de volver a ejecutar el notebook.

---

### Celda 6 — Referencia de modelos existentes (codigo)

```python
def file_size_mb(path):
    return round(path.stat().st_size / (1024 * 1024), 4) if path.exists() else None
```
Utilitario reutilizado en todo el notebook para reportar tamanos de archivos TFLite en MB.

Antes de entrenar, el notebook muestra el tamano de los modelos MobileNetV2 y MobileNetV3 ya exportados. Esto establece el punto de referencia: si EfficientNetB0 es significativamente mas grande sin una mejora proporcional de accuracy, la decision de no integrarlo a Android queda justificada cuantitativamente desde el inicio.

---

### Celda 7 — Construccion del modelo (codigo)

Esta es la celda central del notebook. Define la funcion `build_efficientnet` que construye el modelo completo.

**Seleccion del backbone:**
```python
backbone_map = {
    'B0': tf.keras.applications.EfficientNetB0,
    'B3': tf.keras.applications.EfficientNetB3,
    'B4': tf.keras.applications.EfficientNetB4,
}
backbone = backbone_map[variant](
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    # include_preprocessing NO existe en EfficientNetV1 —
    # el backbone aplica su propio rescaling internamente
)
```
- `include_top=False`: descarta la capa de clasificacion original de ImageNet (1000 clases). Solo se conservan las capas convolucionales que extraen features.
- `weights='imagenet'`: carga pesos preentrenados en ImageNet. Estos pesos ya reconocen texturas, bordes y patrones visuales generales, lo que acelera significativamente el entrenamiento en PlantVillage.
- **Sin `include_preprocessing`**: EfficientNetV1 no acepta este parametro — es exclusivo de MobileNetV3. El backbone de EfficientNet contiene internamente un `Rescaling` layer que normaliza los pixeles `[0, 255]` a la distribucion que espera cada variante. Por eso el pipeline solo entrega `float32` en `[0, 255]` y no aplica ninguna normalizacion adicional.

**Congelar el backbone:**
```python
backbone.trainable = False
```
Durante la primera fase de entrenamiento el backbone queda completamente congelado. Solo la cabeza de clasificacion aprende. Esto evita que los pesos de ImageNet se destruyan en las primeras epocas cuando la cabeza aun produce gradientes muy ruidosos.

**Construccion de la cabeza:**
```python
inputs = keras.Input(shape=(*IMG_SIZE, 3), name='image')
x = backbone(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D(name='gap')(x)
x = keras.layers.BatchNormalization(name='head_bn')(x)
x = keras.layers.Dropout(0.3, name='dropout')(x)
outputs = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
```

Cada capa tiene un proposito especifico:

- **`backbone(inputs, training=False)`**: el argumento `training=False` mantiene las capas BatchNormalization del backbone en modo inferencia incluso durante el entrenamiento de la cabeza, preservando las estadisticas de normalizacion aprendidas en ImageNet.

- **`GlobalAveragePooling2D`**: convierte el mapa de activaciones de forma `(H, W, C)` en un vector de forma `(C,)` calculando el promedio espacial. Reduce drasticamente el numero de parametros respecto a un `Flatten` y hace al modelo invariante a pequenas traslaciones espaciales.

- **`BatchNormalization`** en la cabeza: EfficientNet es mas sensible que MobileNet a los cambios de distribucion durante transfer learning. Agregar una BN despues del pooling normaliza las activaciones antes de la capa densa, estabilizando el entrenamiento de la cabeza y reduciendo la necesidad de ajustar el learning rate manualmente.

- **`Dropout(0.3)`**: desactiva aleatoriamente el 30% de las neuronas durante cada paso de entrenamiento. Fuerza al modelo a no depender de ninguna feature individual, reduciendo el sobreajuste. Se usa 0.3 en lugar del 0.2 de MobileNetV3 porque EfficientNet tiene mas capacidad y mayor riesgo de memorizar el conjunto de entrenamiento.

- **`Dense(38, activation='softmax')`**: capa de clasificacion final. Produce una distribucion de probabilidad sobre las 38 clases. La clase con mayor probabilidad es la prediccion del modelo.

**Compilacion:**
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=HEAD_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
```
- `Adam`: optimizador adaptativo que ajusta el learning rate por parametro. Funciona bien para transfer learning.
- `sparse_categorical_crossentropy`: funcion de perdida para clasificacion multiclase cuando las etiquetas son enteros (no one-hot). PlantVillage devuelve etiquetas como indices enteros.

---

### Celda 8 — Entrenamiento de la cabeza (codigo)

**Callbacks:**

```python
keras.callbacks.ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
```
Guarda el modelo completo en disco solo cuando `val_accuracy` mejora. Al final del entrenamiento, el archivo en disco contiene el mejor modelo visto, no el del ultimo epoch.

```python
keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
```
Detiene el entrenamiento si `val_accuracy` no mejora en 4 epochs consecutivos. `restore_best_weights=True` regresa los pesos al mejor epoch antes de detenerse, evitando que el modelo se quede con pesos de un epoch peor.

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7)
```
Si `val_loss` no mejora en 2 epochs, reduce el learning rate multiplicandolo por 0.3. Esto ayuda al modelo a salir de mesetas de loss sin necesidad de ajustar el LR manualmente. El limite `min_lr=1e-7` evita que el LR se haga tan pequeno que el entrenamiento se detenga efectivamente.

```python
keras.callbacks.CSVLogger(str(training_log_path))
```
Guarda accuracy y loss de cada epoch en un archivo CSV. Permite reconstruir las graficas de entrenamiento sin reejecutar el notebook.

---

### Celda 9 — Fine-tuning controlado (codigo)

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
2. Se vuelven a congelar todas las capas excepto las ultimas 20. Las capas iniciales de EfficientNet reconocen features genericas (bordes, texturas) que son utiles para cualquier dominio visual y no necesitan ajustarse.
3. Todas las capas `BatchNormalization` del backbone se congelan independientemente de su posicion. Si las BN se descongelan, sus estadisticas de media y varianza cambian y pueden desestabilizar las capas que dependen de ellas.

```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), ...)
history_finetune = model.fit(..., initial_epoch=history_head.epoch[-1] + 1, ...)
```
Se recompila con `FINE_TUNE_LR = 1e-5`, dos ordenes de magnitud menor que `HEAD_LR`. `initial_epoch` indica a Keras que el historial de entrenamiento continua desde donde termino la fase 1, produciendo graficas continuas.

---

### Celda 10 — Graficas de entrenamiento (codigo)

```python
def history_to_frame(*histories):
    ...
    frame['phase'] = name
```
Combina los historiales de la fase 1 (cabeza) y fase 2 (fine-tuning) en un unico DataFrame con una columna `phase`. Esto permite diferenciar visualmente en las graficas donde termina una fase y empieza la otra, lo que es importante para detectar si el fine-tuning mejoro o empeoro el modelo.

Se generan dos graficas lado a lado:
- **Accuracy**: train y val accuracy por epoch. Una brecha grande entre ambas indica sobreajuste.
- **Loss**: train y val loss por epoch. La val loss debe decrecer durante el entrenamiento; si sube mientras el train loss baja, el modelo esta memorizando.

Las graficas se guardan en `training_graph.png` para incluirlas en la tesis.

---

### Celda 11 — Evaluacion en el conjunto de prueba (codigo)

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
Muestra precision, recall y F1 por cada una de las 38 clases. Las clases con F1 bajo son candidatas a analisis adicional en la matriz de confusion.

---

### Celda 12 — Matriz de confusion (codigo)

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, cmap='Blues', ...)
```
La matriz de confusion de 38x38 muestra cuantas imagenes de cada clase real fueron clasificadas en cada clase predicha. La diagonal principal representa predicciones correctas. Las celdas fuera de la diagonal son errores. Un bloque de errores entre dos clases especificas (ej. `Tomato___Early_blight` y `Tomato___Late_blight`) indica que el modelo tiene dificultad para distinguir enfermedades visualmente similares dentro del mismo genero.

---

### Celda 13 — Errores de alta confianza (codigo)

```python
errors = [
    {'true_label': ..., 'predicted_label': ..., 'confidence': c}
    for t, p, c in zip(y_true, y_pred, y_conf) if t != p
]
errors_df.sort_values('confidence', ascending=False).head(25)
```
Filtra solo los errores y los ordena por confianza descendente. Los errores con mayor confianza son los mas criticos para una aplicacion de diagnostico: el modelo dice estar muy seguro de algo incorrecto. Esta tabla es util para la tesis porque permite argumentar cuales tipos de confusion son sistematicos y cuales son casos borde.

---

### Celda 14 — Exportacion a SavedModel y TFLite (codigo)

**Exportacion a SavedModel:**
```python
eval_model.export(str(saved_model_dir))
```
Guarda el modelo en formato SavedModel de TensorFlow. Es el formato intermedio necesario para convertir a TFLite.

**Conversion FP32:**
```python
converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
tflite_fp32 = converter.convert()
```
Sin cuantizacion: los pesos se mantienen en `float32`. Produce el archivo mas grande pero con la precision original del modelo.

**Conversion FP16:**
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_fp16 = converter.convert()
```
Cuantizacion a `float16`: los pesos se comprimen de 32 a 16 bits, reduciendo el tamano aproximadamente a la mitad sin perdida significativa de accuracy en la mayoria de dispositivos Android modernos. Este es el formato recomendado para despliegue movil y el que se integra en la app.

---

### Celda 15 — Benchmark local TFLite (codigo)

```python
try:
    from ai_edge_litert.interpreter import Interpreter as LiteInterpreter
except ImportError:
    warnings.filterwarnings('ignore', message='.*tf.lite.Interpreter is deprecated.*')
    LiteInterpreter = tf.lite.Interpreter
```
Usa `ai_edge_litert` si esta disponible (reemplazante oficial de `tf.lite.Interpreter` desde TF 2.18). Si no esta instalado, usa `tf.lite.Interpreter` suprimiendo el warning de deprecacion.

```python
for i in range(warmup + runs):
    ...
    if i >= warmup:
        timings_ms.append((t1 - t0) * 1000)
```
Las primeras `warmup=3` ejecuciones no se cuentan porque la primera inferencia TFLite es mas lenta por la compilacion JIT del interprete. A partir de la ejecucion 4 se miden los tiempos reales del steady-state.

El resultado reporta: promedio, mediana, P95 y desviacion estandar en milisegundos. **El P95 es la metrica mas relevante para la experiencia de usuario**: indica el tiempo de inferencia en el peor caso del 95% de las ejecuciones.

---

### Celda 16 — Tabla comparativa (codigo)

```python
comparison_rows = [
    {'model': 'MobileNetV2 FP16 (baseline Android)', ...},
    {'model': 'MobileNetV3 Small FP16', ...},
    {'model': f'EfficientNet{MODEL_VARIANT} FP16', ...},
]
```
Consolida en una sola tabla los tres modelos del experimento comparativo. Para MobileNetV2 y MobileNetV3 se leen los archivos TFLite ya existentes para obtener su tamano real. Esta tabla es el punto de partida para la discusion de resultados en la tesis.

---

### Celda 17 — Registro en contexto (codigo)

```python
def append_run_to_context():
    ...
    with open(CONTEXT_DOC, 'a') as f:
        f.write('\n'.join(lines) + '\n')
```
Al finalizar una corrida completa (`TRAINING_MODE = 'full'`), agrega automaticamente un bloque de resultados a este mismo archivo (`model_efficientNet.md`). Incluye: timestamp, variante del modelo, modo de entrenamiento, tamanos del split, accuracy, loss, tamano TFLite y latencia local. Esto mantiene un registro historico de todas las corridas sin necesidad de copiar resultados manualmente.

En `smoke_test`, esta celda imprime un mensaje informativo pero no escribe nada, evitando contaminar el historial con resultados de prueba.

---

## Flujo de uso recomendado

1. Ejecutar con `TRAINING_MODE = 'smoke_test'` para verificar que el pipeline funciona de principio a fin sin errores.
2. Cambiar a `TRAINING_MODE = 'full'` y ejecutar todas las celdas para el entrenamiento completo.
3. Revisar las graficas de entrenamiento y la tabla de errores de alta confianza.
4. Comparar la tabla de la celda 16 contra MobileNetV2 y MobileNetV3.
5. Si los resultados justifican integracion, copiar `saved_models/efficientnetb0_v1_fp16.tflite` a `app/src/main/assets/` de la app Android.

## Variantes disponibles

| Variante | Parametros | Entrada | Cambios necesarios |
|---|---|---|---|
| `B0` | ~5.3M | 224x224 | ninguno (defecto) |
| `B3` | ~12M | 300x300 | `MODEL_VARIANT='B3'`, `IMG_SIZE=(300,300)` |
| `B4` | ~19M | 380x380 | `MODEL_VARIANT='B4'`, `IMG_SIZE=(380,380)` |
