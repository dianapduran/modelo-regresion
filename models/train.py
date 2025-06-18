# Librerías estándar
import os  # Manejo de rutas y archivos

# Librerías de terceros
import pandas as pd  # Para manejar DataFrames
from sklearn.model_selection import train_test_split  # Para dividir datos
from sklearn.metrics import mean_squared_error  # Para evaluar error del modelo
import optuna  # Para optimizar hiperparámetros automáticamente
import mlflow  # Para registrar experimentos
import mlflow.sklearn  # Para registrar modelos sklearn en MLflow

# Librerías locales 
from models.pipeline import build_pipeline  # Función que crea el pipeline de preprocesamiento + modelo

# Rutas de carpetas y archivo
BASE_DIR = r'd:\OneDrive - Landers\Documentos\Curso Python\parte_2\Sln_PublicarModelodeRegresion'
DATA_DIR = os.path.join(BASE_DIR, "data")
INGRESOS_FILE = "ingresos.txt"
INGRESOS_PATH = os.path.join(DATA_DIR, INGRESOS_FILE)

# Función para cargar datos
def cargar_datos(path=INGRESOS_PATH, sep=","):
    # Verifica que el archivo exista
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    # Lee el archivo CSV y devuelve un DataFrame
    return pd.read_csv(path, sep=sep)

# Bloque principal
if __name__ == "__main__":
    # Cambia la carpeta de trabajo para evitar errores de rutas
    os.chdir(BASE_DIR)

    # Carga los datos del archivo CSV
    data = cargar_datos()

    # Separa variables independientes (X) y dependiente (y)
    X = data[['edad', 'nivel_educativo', 'horas_trabajadas']]
    y = data[['ingreso_mensual']]

    # Divide en entrenamiento y prueba (70% train, 30% test por defecto)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Construye el pipeline de ingeniería de características + modelo
    pipeline = build_pipeline()

    # Define o crea un experimento en MLflow para guardar todos los runs
    mlflow.set_experiment("ElasticNet_optuna")

    # Función objetivo de Optuna: prueba diferentes hiperparámetros
 
    def objective(trial):
        # Define rangos para alpha, l1_ratio, fit_intercept
        alpha = trial.suggest_float("alpha", 1e-4, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

        # Aplica esos valores al pipeline
        pipeline.set_params(
            regressor__alpha=alpha,
            regressor__l1_ratio=l1_ratio,
            regressor__fit_intercept=fit_intercept
        )

        # Entrena el modelo con esos hiperparámetros
        pipeline.fit(X_train, y_train)

        # Predice en los datos de prueba
        y_pred = pipeline.predict(X_test)

        # Calcula RMSE (raíz del error cuadrático medio)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5

        # Registra los parámetros y el resultado en MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_param("fit_intercept", fit_intercept)
            mlflow.log_metric("rmse", rmse)

        # Retorna RMSE (el mejor es el menor rmse)
        return rmse

    # Ejecuta Optuna para encontrar los mejores hiperparámetros
    study = optuna.create_study(direction='minimize')  # Minimizar RMSE
    study.optimize(objective, n_trials=50)  # Probar 50 combinaciones

    # Reentrena con los mejores hiperparámetros encontrados
    pipeline.set_params(
        regressor__alpha=study.best_params['alpha'],
        regressor__l1_ratio=study.best_params['l1_ratio'],
        regressor__fit_intercept=study.best_params['fit_intercept']
    )
    pipeline.fit(X_train, y_train)

    # Guarda el modelo final entrenado en MLflow
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.sklearn.log_model(pipeline, "model")  # Guarda el pipeline completo
        mlflow.log_params(study.best_params)  # Guarda los hiperparámetros finales
        final_rmse = mean_squared_error(y_test, pipeline.predict(X_test)) ** 0.5
        mlflow.log_metric("final_rmse", final_rmse)  # Guarda el RMSE final
