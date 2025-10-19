# Taller 4 — MLflow + GitHub Actions

## Objetivo
Crear un pipeline de **ejecución, entrenamiento, registro y validación** de un modelo con **GitHub Actions** y **MLflow**.

## Estructura del proyecto
mlflow-deploy/
├── train.py
├── validate.py
├── requirements.txt
├── .github/workflows/mlflow-ci.yml
├── Makefile
└── mlruns/ # se genera al entrenar; no se versiona


## Dataset externo
- **Wine Quality (Red)** — UCI Machine Learning Repository (CSV separado por `;`).
- **Problema**: regresión (predecir `quality`).
- **Motivo**: público, pequeño y 100% numérico → rápido y estable en CI.

## Entrenamiento y registro (MLflow)
- Modelo: `StandardScaler` + `Ridge(alpha=1.0)`
- División determinística: `test_size=0.2`, `random_state=42`
- **MLflow Tracking** (`file://./mlruns`):
  - `log_params`: modelo, alpha, split, etc.
  - `log_metrics`: `mse`, `rmse`
  - `log_model`: con **signature** e **input_example**
- Se guarda `last_run_id.txt` para la validación.

## Validación (quality gate)
- Carga el modelo desde **`runs:/<run_id>/model`**.
- Métrica: **RMSE** en test.
- **Umbral**: `RMSE ≤ 0.85` → aprobado (exit 0). Si no, falla (exit 1).

## Comandos (Makefile)
```bash
# crear venv y activar (ejemplo Linux/WSL)
python3 -m venv .venv && source .venv/bin/activate

# instalar dependencias
make install

# entrenar y registrar en MLflow
make train

# validar (quality gate)
make validate

# abrir UI de MLflow
make mlflow-ui   # http://127.0.0.1:5001

