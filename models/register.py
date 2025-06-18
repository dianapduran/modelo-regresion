import mlflow

'''
Registrar = crear un catálogo versionado:
Cada vez que se tenga una versión mejorada, MLflow la versiona (v1, v2, v3...).
FastAPI usa siempre la última versión aprobada.
'''

#mejor experimento
RUN_ID = "27734e8349474435ba485ed1974548d7"

#registrar el modelo en el catálogo oficial de modelos de MLflow
result = mlflow.register_model(
    f"runs:/{RUN_ID}/model",
    "ElasticNetIncomePredictor"
)

#mostrar confirmación
print(result)
