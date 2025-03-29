import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

# Configuración de Streamlit
st.title("Entrenador de Modelo CNN Personalizado")

# Ruta a la carpeta de imágenes (ahora es por defecto 'dataset')
directorio_imagenes = 'dataset'  # Se establece la ruta por defecto

# Parámetros del modelo
epocas = st.slider("Selecciona el número de épocas", min_value=1, max_value=100, value=15, step=1)
tamano_batch = st.slider("Selecciona el tamaño del batch", min_value=16, max_value=128, value=64, step=16)
tamano_imagen = st.slider("Selecciona el tamaño de la imagen", min_value=32, max_value=256, value=64, step=32)

# Inicializar session_state para almacenar los datos de entrenamiento
if 'modelo' not in st.session_state:
    st.session_state.modelo = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'nombres_clases' not in st.session_state:
    st.session_state.nombres_clases = None
if 'train_dataset' not in st.session_state:
    st.session_state.train_dataset = None
if 'val_dataset' not in st.session_state:
    st.session_state.val_dataset = None
if 'test_dataset' not in st.session_state:
    st.session_state.test_dataset = None

# Función para cargar y preprocesar las imágenes
def cargar_imagenes_desde_directorio(directorio, tamano_objetivo=(tamano_imagen, tamano_imagen)):
    X = []
    y = []
    nombres_clases = []
    etiquetas = {}
    
    for nombre_carpeta in os.listdir(directorio):
        ruta_carpeta = os.path.join(directorio, nombre_carpeta)
        if os.path.isdir(ruta_carpeta):
            nombres_clases.append(nombre_carpeta)
            etiquetas[nombre_carpeta] = len(nombres_clases) - 1

            for nombre_archivo in os.listdir(ruta_carpeta):
                ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
                try:
                    with Image.open(ruta_archivo) as img:
                        img = img.convert('RGB')
                        img = img.resize(tamano_objetivo)
                        X.append(np.array(img))
                        y.append(etiquetas[nombre_carpeta])
                except (IOError, UnidentifiedImageError) as e:
                    st.warning(f"Advertencia: {ruta_archivo} no se pudo abrir o no es una imagen válida. Error: {e}")
    return np.array(X), np.array(y), nombres_clases

# Verificar si la carpeta 'dataset' existe, si no, mostrar un mensaje
if not os.path.exists(directorio_imagenes):
    st.error(f"La carpeta '{directorio_imagenes}' no existe. Asegúrate de tener la carpeta 'dataset' con imágenes en el directorio adecuado.")
else:
    # Solo cargar los datos si no están cargados previamente
    if st.session_state.X is None or st.session_state.y is None:
        # Cargar el conjunto de datos
        X, y, nombres_clases = cargar_imagenes_desde_directorio(directorio_imagenes)
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.nombres_clases = nombres_clases
        st.write("Clases:", nombres_clases)
        st.write("Tamaño del conjunto de datos:", X.shape)

        # Crear el conjunto de datos de TensorFlow
        dataset = image_dataset_from_directory(
            directorio_imagenes,
            image_size=(tamano_imagen, tamano_imagen),
            batch_size=tamano_batch,
            label_mode='int'
        )

        # Preprocesar los datos
        dataset = dataset.map(lambda imagen, etiqueta: (tf.cast(imagen, tf.float32) / 255.0, etiqueta))
        dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)

        def preprocesar_imagen(imagen, etiqueta):
            imagen = tf.image.resize(imagen, [tamano_imagen, tamano_imagen])
            imagen = tf.image.convert_image_dtype(imagen, tf.float32)
            return imagen, etiqueta

        dataset = dataset.map(preprocesar_imagen)

        # Dividir el conjunto de datos en entrenamiento, validación y prueba
        num_lotes = tf.data.experimental.cardinality(dataset).numpy()
        tamano_entrenamiento = int(0.8 * num_lotes)
        tamano_validacion = int(0.1 * num_lotes)
        tamano_prueba = num_lotes - tamano_entrenamiento - tamano_validacion

        st.session_state.train_dataset = dataset.take(tamano_entrenamiento).unbatch().batch(tamano_batch)
        st.session_state.val_dataset = dataset.skip(tamano_entrenamiento).take(tamano_validacion).unbatch().batch(tamano_batch)
        st.session_state.test_dataset = dataset.skip(tamano_entrenamiento + tamano_validacion).take(tamano_prueba).unbatch().batch(tamano_batch)

    # Crear la arquitectura del modelo basada en la entrada del usuario
    modelo = tf.keras.Sequential()

    # Definir las capas interactivamente según la entrada del usuario
    num_capas = st.slider("Selecciona el número de capas convolucionales", 1, 5, 3)
    filtros_por_capa = st.slider("Selecciona el número de filtros por capa", 16, 128, 32, step=16)

    for i in range(num_capas):
        modelo.add(tf.keras.layers.Conv2D(
            filters=filtros_por_capa*(i+1), kernel_size=(5, 5),
            strides=(1, 1), padding='same', data_format='channels_last',
            activation='relu', name=f'conv_{i}'
        ))
        modelo.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), name=f'pool_{i}'))

    modelo.add(tf.keras.layers.Flatten())
    modelo.add(tf.keras.layers.Dense(units=1024, activation='relu'))
    modelo.add(tf.keras.layers.Dropout(0.5))
    modelo.add(tf.keras.layers.Dense(units=len(st.session_state.nombres_clases), activation='softmax'))

    # Construir y mostrar el resumen del modelo
    modelo.build(input_shape=(None, tamano_imagen, tamano_imagen, 3))
    modelo.summary()

    # Compilar el modelo
    modelo.compile(optimizer=tf.keras.optimizers.Adam(),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                   metrics=['accuracy'])

    # Botón para iniciar el entrenamiento
    if st.button('Iniciar Entrenamiento'):
        st.write("Entrenando el modelo...")
        historia = modelo.fit(st.session_state.train_dataset, epochs=epocas, validation_data=st.session_state.val_dataset)

        # Mostrar los gráficos de la historia del entrenamiento
        hist = historia.history
        x_arr = np.arange(len(hist['loss'])) + 1

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(x_arr, hist['loss'], '-o', label='Pérdida de entrenamiento')
        if 'val_loss' in hist:  # Verifica que val_loss esté presente
            axes[0].plot(x_arr, hist['val_loss'], '--<', label='Pérdida de validación')
        axes[0].set_xlabel('Época')
        axes[0].set_ylabel('Pérdida')
        axes[0].legend()

        axes[1].plot(x_arr, hist['accuracy'], '-o', label='Precisión de entrenamiento')
        if 'val_accuracy' in hist:  # Verifica que val_accuracy esté presente
            axes[1].plot(x_arr, hist['val_accuracy'], '--<', label='Precisión de validación')
        axes[1].set_xlabel('Época')
        axes[1].set_ylabel('Precisión')
        axes[1].legend()

        st.pyplot(fig)

        # Guardar el modelo
        if not os.path.exists('modelos'):
            os.mkdir('modelos')

        modelo_guardado_path = 'modelos/modelo_personalizado.keras'
        modelo.save(modelo_guardado_path)
        st.success(f"Modelo guardado como '{modelo_guardado_path}'.")
        
        # Botón para descargar el modelo guardado
        with open(modelo_guardado_path, "rb") as f:
            st.download_button(
                label="Descargar Modelo",
                data=f,
                file_name="modelo_personalizado.keras",
                mime="application/octet-stream"
            )

        # Evaluar en los datos de prueba
        resultados_prueba = modelo.evaluate(st.session_state.test_dataset)
        st.write(f"Precisión en prueba: {resultados_prueba[1]*100:.2f}%")

        # Predicciones y evaluación
        y_true = []
        y_pred = []

        for images, labels in st.session_state.test_dataset:
            preds = modelo(images)
            preds = tf.argmax(preds, axis=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

        # Visualización de la matriz de confusión
        fig_cm, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=st.session_state.nombres_clases, yticklabels=st.session_state.nombres_clases)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de Confusión')
        st.pyplot(fig_cm)

        # Métricas de clasificación
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        st.write(f"\n*Precisión:* {precision:.4f}")
        st.write(f"*Recall:* {recall:.4f}")
        st.write(f"*F1 Score:* {f1:.4f}")
