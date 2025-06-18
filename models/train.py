# models/train.py

# --- Librerías estándar ---
import os                # Para rutas y carpetas
import joblib            # Para guardar el modelo en archivo .pkl (portátil)

# --- Librerías de terceros ---
import pandas as pd                      # Para manipular datos tabulares
from sklearn.model_selection import train_test_split  # Para dividir datos en train/test
from sklearn.metrics import mean_squared_error        # Para evaluar el modelo
import optuna                              # Para optimizar hiperparámetros
import mlflow                              # Para registrar experimentos
import mlflow.sklearn                      # Para registrar modelos sklearn en MLflow

# --- Librerías locales ---
from models.pipeline import build_pipeline  # Tu función personalizada para armar el pipeline

# --- Definir rutas importantes ---
BASE_DIR = os.getcwd()                      # Carpeta raíz del proyecto (carpeta actual)
DATA_DIR = os.path.join(BASE_DIR, "data")   # Carpeta donde están los datos
MODEL_DIR = os.path.join(BASE_DIR, "model") # Carpeta para guardar modelo final
os.makedirs(MODEL_DIR, exist_ok=True)       # Crea la carpeta /model si no existe

INGRESOS_PATH = os.path.join(DATA_DIR, "ingresos.txt")  # Ruta completa del archivo de datos

# --- Función para cargar datos ---
def cargar_datos(path=INGRESOS_PATH, sep=","):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    return pd.read_csv(path, sep=sep)  # Lee CSV y lo devuelve como DataFrame

# --- Bloque principal ---
if __name__ == "__main__":
    os.chdir(BASE_DIR)   # Asegura que la carpeta de trabajo sea la raíz del proyecto

    # Cargar datos
    data = cargar_datos()

    # Separar variables predictoras (X) y variable objetivo (y)
    X = data[['edad', 'nivel_educativo', 'horas_trabajadas']]
    y = data[['ingreso_mensual']]

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Construir pipeline con transformaciones + modelo ElasticNet
    pipeline = build_pipeline()

    # Crear experimento en MLflow (o usarlo si existe)
    mlflow.set_experiment("ElasticNet_optuna")

    # --- Definir función objetivo para Optuna ---
    def objective(trial):
        # Sugerir valores para hiperparámetros dentro de rangos
        alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

        # Aplicar esos valores al pipeline
        pipeline.set_params(
            regressor__alpha=alpha,
            regressor__l1_ratio=l1_ratio,
            regressor__fit_intercept=fit_intercept
        )

        # Entrenar modelo con train set
        pipeline.fit(X_train, y_train)

        # Predecir en test set y calcular error (RMSE)
        y_pred = pipeline.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        # Registrar run en MLflow (anidado)
        with mlflow.start_run(nested=True):
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_param("fit_intercept", fit_intercept)
            mlflow.log_metric("rmse", rmse)

        return rmse  # Retornar el error para que Optuna lo minimice

    # --- Ejecutar búsqueda Optuna (50 pruebas) ---
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    # --- Entrenar modelo final con los mejores hiperparámetros ---
    pipeline.set_params(
        regressor__alpha=study.best_params['alpha'],
        regressor__l1_ratio=study.best_params['l1_ratio'],
        regressor__fit_intercept=study.best_params['fit_intercept']
    )
    pipeline.fit(X_train, y_train)

    # Registrar el modelo final en MLflow
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.sklearn.log_model(pipeline, "model")               # Guarda pipeline en MLflow
        mlflow.log_params(study.best_params)                      # Guarda hiperparámetros óptimos
        final_rmse = mean_squared_error(y_test, pipeline.predict(X_test)) ** 0.5
        mlflow.log_metric("final_rmse", final_rmse)               # Guarda RMSE final

    # Guardar pipeline como archivo .pkl en /model para FastAPI y Docker
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "model.pkl"))
    print(f"Modelo guardado en {MODEL_DIR}/model.pkl")  # Confirmación
