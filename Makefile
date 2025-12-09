# client_defolt_project/Makefile

# ПЕРЕМЕННЫЕ
PYTHON = python3
DOCKER_IMAGE_NAME = credit-default-api
DOCKER_TAG ?= latest
REGISTRY ?= # например, registry.vkcs.cloud/<project-id>/
API_PORT ?= 8000
MONITORING_DIR = infrastructure/monitoring

.PHONY: help
.PHONY: install install-dev install-docker
.PHONY: data-download data-process data-validate
.PHONY: model-train model-onnx model-quantize
.PHONY: test test-api test-data test-model
.PHONY: format lint
.PHONY: serve-api serve-api-docker
.PHONY: monitoring-up monitoring-down
.PHONY: airflow-init airflow-start
.PHONY: build-api push-api
.PHONY: clean clean-data clean-models

# === ПОМОЩЬ ===
help:
	@echo "MLOps Makefile для проекта 'Промышленное развертывание кредитной скоринговой системы'"
	@echo ""
	@echo "Основные команды:"
	@echo "  make install-dev      — установить зависимости для разработки"
	@echo "  make data-process     — скачать и обработать данные"
	@echo "  make model-train      — обучить модель и сохранить артефакты"
	@echo "  make model-onnx       — конвертировать в ONNX"
	@echo "  make test             — запустить все тесты"
	@echo "  make serve-api        — запустить FastAPI локально"
	@echo "  make monitoring-up    — запустить Prometheus + Grafana"
	@echo "  make build-api        — собрать Docker-образ API"
	@echo "  make airflow-start    — запустить Airflow для переобучения"

# === УСТАНОВКА ===
install:
	$(PYTHON) -m pip install -r requirements-core.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

install-docker:
	$(PYTHON) -m pip install -r requirements-docker.txt

# === ДАННЫЕ ===
data-download:
	$(PYTHON) scripts/download_data.py

data-process: data-download
	$(PYTHON) data/make_dataset.py

data-validate: data-process
	$(PYTHON) src/data/validation.py

# === МОДЕЛЬ ===
model-train: data-validate
	$(PYTHON) scripts/model_training/train_nn.py

model-onnx: model-train
	$(PYTHON) scripts/model_training/convert_to_onnx.py

model-quantize: model-onnx
	$(PYTHON) scripts/model_training/quantize_model.py

# === ТЕСТИРОВАНИЕ ===
test: test-api test-data test-model

test-api:
	$(PYTHON) -m pytest tests/test_api.py -v

test-:
	$(PYTHON) -m pytest tests/test_data.py -v

test-model:
	$(PYTHON) -m pytest tests/test_model.py -v

ci-test:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=xml --cov-report=html

# === КОДСТАЙЛ ===
format:
	black src/ tests/ scripts/

lint:
	flake8 src/ tests/ scripts/

# === ЗАПУСК API ===
serve-api:
	uvicorn src.api.app:app --host 0.0.0.0 --port $(API_PORT) --reload

serve-api-docker: build-api
	docker run --rm -p $(API_PORT):8000 $(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

# === МОНИТОРИНГ ===
monitoring-up:
	cd $(MONITORING_DIR) && docker-compose -f docker-compose.monitoring.yml up -d

monitoring-down:
	cd $(MONITORING_DIR) && docker-compose -f docker-compose.monitoring.yml down

# === AIRFLOW ===
airflow-init:
	airflow db init

airflow-start: airflow-init
	airflow webserver --port 8080 &
	airflow scheduler &

# === DOCKER ===
build-api:
	docker build -f Dockerfile.api -t $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) .

push-api: build-api
	docker tag $(DOCKER_IMAGE_NAME):$(DOCKER_TAG) $(REGISTRY)$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)
	docker push $(REGISTRY)$(DOCKER_IMAGE_NAME):$(DOCKER_TAG)

# === ОЧИСТКА ===
clean: clean-data clean-models
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov

clean-:
	rm -rf data/processed/* mlruns/

clean-models:
	rm -f models/best_model.pkl models/*.onnx models/*.quant
