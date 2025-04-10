from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import math
import tensorflow as tf
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las conexiones
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los encabezados
)

# Descargar el dataset que necesitamos
datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

# Separar datos de entrenamiento y de prueba
datos_entrenamiento = datos['train']
datos_pruebas = datos['test']

# Obtener el nombre de las clases
nombres_clases = metadatos.features['label'].names
print(nombres_clases)

# Normalizar las imágenes para que los valores de los píxeles sean entre 0 y 1 (actualmente están entre 0 y 255)
def normalizar(imagenes, etiquetas):
    # Convertir enteros a flotantes
    imagenes = tf.cast(imagenes, tf.float32)
    # Dividir entre 255 para normalizar
    imagenes /= 255
    return imagenes, etiquetas

# Normalizar los datos de entrenamiento y los datos de prueba
datos_entrenamiento = datos_entrenamiento.map(normalizar)
datos_pruebas = datos_pruebas.map(normalizar)

# Agregar los datos a caché
datos_entrenamiento = datos_entrenamiento.cache()
datos_pruebas = datos_pruebas.cache()

# Crear modelo
modelo = tf.keras.Sequential([
    # Definir la capa de entrada Flatten
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # Definir la primera capa oculta
    tf.keras.layers.Dense(128, activation='relu'),
    # Capa de salida con 10 neuronas (una por cada dígito)
    tf.keras.layers.Dense(10, activation='softmax')
])

# Definir los hiperparámetros
epochs = 10
learning_rate = 0.0008  # Ajusta este valor según el rendimiento del modelo

# Definir el tamaño del lote (batch size)
batch_size = 32

# Compilar el modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # Entropía cruzada
    metrics=['accuracy']
)

# Obtener el número de imágenes de entrenamiento y de prueba
num_img_entrenamiento = metadatos.splits["train"].num_examples
num_img_pruebas = metadatos.splits["test"].num_examples
print(f"Imágenes de entrenamiento: {num_img_entrenamiento}")
print(f"Imágenes de prueba: {num_img_pruebas}")

# Aplicar una estrategia para mejorar la eficiencia del modelo
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_img_entrenamiento).batch(batch_size)
datos_pruebas = datos_pruebas.batch(batch_size)

# Entrenar el modelo
steps_per_epoch = math.ceil(num_img_entrenamiento / batch_size)
historial = modelo.fit(datos_entrenamiento, epochs=epochs, steps_per_epoch=steps_per_epoch)

# Predecir todas las imágenes
for imagenes_pruebas, etiquetas_pruebas in datos_pruebas.take(1):
    imagenes_pruebas = imagenes_pruebas.numpy()
    etiquetas_pruebas = etiquetas_pruebas.numpy()
    predicciones = modelo.predict(imagenes_pruebas)

# Predecir nuestra imagen
index_imagen = 2
imagen = imagenes_pruebas[index_imagen]
plt.imshow(np.reshape(imagen, (28, 28)), cmap=plt.cm.binary)
plt.show()

# Realizar la predicción
imagen = np.array([imagen])
p = modelo.predict(imagen)
print(f"La imagen es un {nombres_clases[np.argmax(p)]}")

# Cargar el modelo previamente entrenado
# modelo = tf.keras.models.load_model("modelo.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer la imagen enviada
        image = await file.read()
        image = Image.open(BytesIO(image)).convert("L")
        
        # Redimensionar la imagen a 28x28
        image = image.resize((28, 28))
        
        # Convertir a array y normalizar
        image_array = np.array(image).astype(np.float32) / 255.0

        # Asegurarse de que la imagen tiene la forma correcta (28, 28, 1)
        image_array = image_array.reshape(1, 28, 28, 1)
        
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        plt.show()


        # Predecir
        prediction = modelo.predict(image_array)
        print(prediction) 
        predicted_class = int(np.argmax(prediction))
        print(f"Imagen recibida: {predicted_class}") 
        print(tf.__version__)
        return JSONResponse(content={"prediccion": predicted_class})

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": "Error en la predicción. Intenta nuevamente."}, status_code=500)
