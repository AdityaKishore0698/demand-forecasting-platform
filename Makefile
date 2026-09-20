PY ?= python3
API_PORT ?= 8000

.PHONY: help install install-frontend train demo-data demo-train demo eda api frontend test test-frontend docker

help:
	@echo "make install           install Python dependencies"
	@echo "make install-frontend  install frontend dependencies"
	@echo "make demo              generate SYNTHETIC data and train the demo bundle -> artifacts/demo/"
	@echo "make train             validate YOUR data in data/raw/, back-test, train -> artifacts/"
	@echo "make eda               regenerate EDA figures into docs/figures/"
	@echo "make api               serve the API on :$(API_PORT)"
	@echo "make frontend          run the Vite dev server on :5173"
	@echo "make test              run backend tests"
	@echo "make test-frontend     type-check, build and unit-test the frontend"
	@echo "make docker            build the API image"

install:
	$(PY) -m pip install -r requirements-dev.txt

install-frontend:
	cd frontend && npm ci

train:
	$(PY) -m src.forecasting.pipeline

demo-data:
	$(PY) scripts/make_synthetic_data.py

demo-train:
	$(PY) -m src.forecasting.pipeline --config configs/demo.yaml

demo: demo-data demo-train

eda:
	$(PY) scripts/eda.py

api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port $(API_PORT)

frontend:
	cd frontend && npm run dev

test:
	$(PY) -m pytest -q

test-frontend:
	cd frontend && npm run build && npm test

docker:
	docker build -t demand-forecasting-api .
