import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import mlflow
import mlflow.sklearn

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from mlflow.models import infer_signature


# =========================
# Configuración del dataset
# =========================
# Dataset externo (UCI): Wine Quality - Red
# Nota: el CSV usa separador ';'
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "winequality-red.csv")

# =========================
# Configuración de MLflow
# =========================
EXPERIMENT_NAME = "CI-CD-Workshop4"
MLRUNS_DIR = os.path.abspath("mlruns")  # tracking local file://
TRACKING_URI = "file://" + MLRUNS_DIR


def ensure_data() -> pd.DataFrame:
    """Descarga el CSV externo si no existe y lo carga como DataFrame."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"[DATA] Descargando: {DATA_URL}")
        r = requests.get(DATA_URL, timeout=60)
        r.raise_for_status()
        with open(DATA_PATH, "wb") as f:
            f.write(r.content)
        print(f"[DATA] Guardado en: {DATA_PATH}")

    # Importante: el separador es ';'
    df = pd.read_csv(DATA_PATH, sep=';')
    return df


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) y target (y). Todas las columnas excepto 'quality' son features."""
    target = "quality"
    features = [c for c in df.columns if c != target]
    X = df[features].copy()
    y = df[target].copy()
    return X, y


def build_pipeline(alpha: float = 1.0) -> Pipeline:
    """Pipeline: estandarización + Ridge Regression (rápido y robusto para CI)."""
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=42)),
        ]
    )
    return pipe


if __name__ == "__main__":
    print(f"[CWD] {os.getcwd()}")

    # 1) Tracking local de MLflow (carpeta ./mlruns)
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)

    # 2) Crear/obtener experimento
    try:
        exp_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=TRACKING_URI)
        print(f"[MLflow] Experimento creado: {EXPERIMENT_NAME} ({exp_id})")
    except mlflow.exceptions.MlflowException:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        exp_id = exp.experiment_id if exp else None
        print(f"[MLflow] Experimento existente: {EXPERIMENT_NAME} ({exp_id})")

    if not exp_id:
        print("[FATAL] No se pudo crear/obtener el experimento.")
        sys.exit(1)

    # 3) Datos
    df = ensure_data()
    X, y = build_xy(df)

    # Split reproducible
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Modelo
    alpha = 1.0
    pipe = build_pipeline(alpha=alpha)

    # 5) Entrenamiento + logging de MLflow
    with mlflow.start_run(experiment_id=exp_id) as run:
        run_id = run.info.run_id
        print(f"[MLflow] Run ID: {run_id}")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))

        # Log de parámetros y métricas
        mlflow.log_params({
            "model": "Ridge",
            "alpha": alpha,
            "dataset": "UCI Wine Quality (Red)",
            "test_size": 0.2,
            "random_state": 42,
            "n_features": X.shape[1],
        })
        mlflow.log_metrics({
            "mse": float(mse),
            "rmse": rmse
        })

        # Firma del modelo + ejemplo de entrada
        signature = infer_signature(X_train, pipe.predict(X_train.head(5)))
        input_example = X_train.head(5)

        # Registrar (loguear) el modelo como artefacto del run
        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        # Guardar run_id para validate.py
        with open("last_run_id.txt", "w", encoding="utf-8") as f:
            f.write(run_id + "\n")

        print(f"✅ Entrenamiento OK | MSE={mse:.4f} | RMSE={rmse:.4f}")
        print(f"[MLflow] Modelo logueado con firma e input_example.")
        print(f"[MLflow] last_run_id.txt -> {run_id}")
