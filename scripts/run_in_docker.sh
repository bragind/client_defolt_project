#!/bin/bash

# Сборка образа
docker build -t credit-default-model .

# Запуск API
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/reports:/app/reports \
  --name ml-api \
  credit-default-model

# Запуск MLflow UI
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/mlruns:/app/mlruns \
  --name mlflow-ui \
  credit-default-model \
  mlflow ui --host 0.0.0.0 --port 5000

echo "API доступен на http://localhost:8000"
echo "MLflow UI доступен на http://localhost:5000"
echo "Документация API: http://localhost:8000/docs"