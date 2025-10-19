# === Taller 4: MLflow + GitHub Actions ===
# Targets básicos para probar que Make funciona

.PHONY: install train validate mlflow-ui clean

install:
	python -m pip install -U pip
	pip install -r requirements.txt

train:
	python train.py

validate:
	python validate.py --run-id "$$(cat last_run_id.txt 2>/dev/null)"

mlflow-ui:
	mlflow ui --backend-store-uri "file://$$(pwd)/mlruns" -p 5001

clean:
	rm -rf __pycache__ mlruns last_run_id.txt data
