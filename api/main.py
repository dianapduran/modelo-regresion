# api/main.py

# --- Librerías estándar ---
import os

# --- Librerías de terceros ---
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd  # para crear el DataFrame

# --- Cargar modelo registrado ---
MODEL_NAME = "ElasticNetIncomePredictor"
MODEL_VERSION = 1

model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

# --- Inicializa FastAPI ---
app = FastAPI()

# --- Esquema de entrada ---
class InputData(BaseModel):
    edad: float
    nivel_educativo: str
    horas_trabajadas: float

# --- Endpoint de prueba ---
@app.get("/")
def home():
    return {"mensaje": "API OK"}

# --- Endpoint de predicción ---
@app.post("/predict")
def predict(data: InputData):
    # usar un DataFrame real
    input_df = pd.DataFrame([{
        "edad": data.edad,
        "nivel_educativo": data.nivel_educativo,
        "horas_trabajadas": data.horas_trabajadas
    }])
    prediction = model.predict(input_df)
    return {"ingreso_mensual_estimado": float(prediction[0])}
