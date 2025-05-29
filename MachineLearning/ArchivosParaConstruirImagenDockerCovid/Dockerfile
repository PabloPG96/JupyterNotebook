FROM python:3.10

# Crear un directorio de trabajo
WORKDIR /code

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el contenido del proyecto (incluye app.py y modelo.keras)
COPY . .

# Exponer el puerto
EXPOSE 8000

# Ejecutar el servidor
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
