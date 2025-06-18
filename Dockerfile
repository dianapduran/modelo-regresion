# Usa Python oficial
FROM python:3.11-slim

# Crea carpeta de trabajo
WORKDIR /app

# Copia todo (código + model)
COPY . .

# Instala dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expone puerto
EXPOSE 8000

# Ejecuta FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]



#abrir docker
#construir la imagen de docker (empaquetando proyecto dentro de la imagen 'ingresos-api')
##Docker descarga Python base (python:3.11-slim) y copia tus archivos.
##Luego instalará dependencias y configurará FastAPI para exponerlo.
# docker build -t ingresos-api . 

#correr contenedor
#Inicia la API FastAPI dentro de Docker.
# acceder al navegador: http://127.0.0.1:8000/docs
# docker run -p 8000:8000 ingresos-api

#Construir imagen	docker build -t ingresos-api . en terminal nueva
#Ejecutar contenedor	docker run -p 8000:8000 ingresos-api
#Ver la API	http://127.0.0.1:8000/docs