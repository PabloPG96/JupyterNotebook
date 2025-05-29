# py -3.10 -m venv venv (Version con tensorflow)
# .\venv\Scripts\activate
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
#import tensorflow_datasets as tfds
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las conexiones
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP
    allow_headers=["*"],  # Permite todos los encabezados
)

# 1. Cargar los modelos previamente entrenado
modelo = tf.keras.models.load_model("modelo_covid.keras")


# Montar archivos estáticos del frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

# Ruta para devolver el HTML principal (por ejemplo, el canvas)
@app.get("/inicio")
async def mostrar_html():
    return FileResponse("frontend/index.html")

# Ruta de prueba
@app.get("/")
async def principal():
    return {"mensaje": "API de predicción de dígitos funcionando"}

# Ruta para predecir número a partir de imagen enviada desde el frontend
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer la imagen enviada desde el frontend
        contents = await file.read()
        imagen = Image.open(BytesIO(contents)).convert("RGB")

        # Redimensionar a lo que espera el modelo (224x224)
        imagen = imagen.resize((224, 224))

        # Convertir a array, normalizar y dar forma (batch de 1 imagen)
        imagen_array = tf.keras.preprocessing.image.img_to_array(imagen)
        imagen_array = imagen_array / 255.0
        imagen_array = imagen_array.reshape((1, 224, 224, 3))

        # Predecir (salida tipo [[0.87]] si es "normal", por ejemplo)
        prediccion = modelo.predict(imagen_array)
        probabilidad = float(prediccion[0][0])  # De 0 a 1
        clase_predicha = int(probabilidad >= 0.5)  # Umbral típico de 0.5

        clases = ["covid", "normal"]
        confianza = probabilidad if clase_predicha == 1 else 1 - probabilidad

        return JSONResponse(content={
            "indice_clase": clase_predicha,
            "clase": clases[clase_predicha],
            "confianza": f"{confianza * 100:.2f}%"  # Ej: 86.34%
        })

    except Exception as e:
        return JSONResponse(
            content={"error": f"Error en la predicción: {str(e)}"},
            status_code=500
        )
#http://127.0.0.1:8000/frontend/index.html