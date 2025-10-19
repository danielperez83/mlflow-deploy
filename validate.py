import argparse
import os
import sys

import mlflow
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# ===== Dataset externo (UCI Wine Quality - Red) =====
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "winequality-red.csv")

# ===== MLflow local =====
MLRUNS_DIR = os.path.abspath("mlruns")
TRACKING_URI = "file://" + MLRUNS_DIR

# ===== Umbral del quality gate =====
# Nota: con Ridge y escalado, un RMSE <= 0.85 suele ser razonable para este dataset.
RMSE_THRESHOLD = 0.85


def ensure_data() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_PATH):
        print(f"[DATA] Descargando: {DATA_URL}")
        r = requests.get(DATA_URL, timeout=60)
        r.raise_for_status()
        with open(DATA_PATH, "wb") as f:
            f.write(r.content)
        print(f"[DATA] Guardado en: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=";")
    return df


def split_xy(df: pd.DataFrame):
    target = "quality"
    features = [c for c in df.columns if c != target]
    X = df[features].copy()
    y = df[target].copy()
    return X, y


def main(run_id_arg: str | None) -> int:
    # Configurar tracking local
    mlflow.set_tracking_uri(TRACKING_URI)

    # Obtener run_id
    run_id = run_id_arg
    if not run_id:
        if not os.path.exists("last_run_id.txt"):
            print("[ERROR] No se proporcionó --run-id y no existe last_run_id.txt")
            return 1
        run_id = open("last_run_id.txt", "r", encoding="utf-8").read().strip()

    print(f"[MLflow] Validando Run ID: {run_id}")

    # Cargar modelo desde MLflow (pyfunc)
    model_uri = f"runs:/{run_id}/model"
    print(f"[MLflow] Cargando modelo: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # Preparar datos (mismo split determinístico que en train)
    df = ensure_data()
    X, y = split_xy(df)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predicción + métrica
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"🔎 RMSE del modelo: {rmse:.4f} (umbral: {RMSE_THRESHOLD})")

    # Quality gate
    if rmse <= RMSE_THRESHOLD:
        print("✅ El modelo cumple el umbral de calidad.")
        return 0
    else:
        print("❌ El modelo NO cumple el umbral de calidad.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None, help="Run ID de MLflow (opcional)")
    args = parser.parse_args()
    sys.exit(main(args.run_id))
