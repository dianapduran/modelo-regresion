# api/main.py

# Librerías estándar
import os
import joblib

# Librerías de terceros
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

# Cargar el modelo final desde /model
MODEL_PATH = os.path.join(os.getcwd(), "model", "model.pkl")
model = joblib.load(MODEL_PATH)

# Inicializa FastAPI
app = FastAPI()

# Esquema de entrada
class InputData(BaseModel):
    edad: float
    nivel_educativo: str
    horas_trabajadas: float

# Endpoint prueba
@app.get("/")
def home():
    return {"mensaje": "API OK"}

# Endpoint predicción
@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([{
        "edad": data.edad,
        "nivel_educativo": data.nivel_educativo,
        "horas_trabajadas": data.horas_trabajadas
    }])
    prediction = model.predict(input_df)
    return {"ingreso_mensual_estimado": float(prediction[0])}
