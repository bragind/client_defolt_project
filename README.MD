
# Прогнозирование кредитного дефолта

End-to-end MLops пайплайн для прогнозирования вероятности дефолта по кредитной карте.

##  Обзор проекта

Этот проект реализует полный конвейер машинного обучения для прогнозирования вероятности дефолта по кредитной карте с использованием [набора данных Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) из репозитория UCI Machine Learning.

Решение охватывает весь жизненный цикл ML - от проверки данных и экспериментов до развертывания модели и мониторинга.

##  Возможности

- **Валидация данных** - Great Expectations для проверки качества данных
- **Feature Engineering** - Автоматическое создание и преобразование признаков
- **ML Pipeline** - Scikit-learn пайплайны с предобработкой и моделированием
- **Трекинг экспериментов** - MLflow для управления экспериментами
- **Версионирование** - DVC для версионирования данных и моделей
- **REST API** - FastAPI для обслуживания моделей
- **Мониторинг** - Обнаружение дрейфа данных
- **CI/CD** - GitHub Actions для автоматизированного тестирования
- **Контейнеризация** - Docker для развертывания

## Стуртура проекта
```bash
├── 📊 data/                          # Управление данными
│   ├── 📁 raw/                       # Исходные данные
│   ├── 📁 processed/                 # Обработанные данные
│   └── 📁 external/                  # Внешние данные
│
├── 💻 src/                           # Исходный код
│   ├── 📁 data/                      # Обработка и валидация данных
│   ├── 📁 models/                    # ML модели и пайплайны
│   ├── 📁 api/                       # FastAPI приложение
│   └── 📁 monitoring/                # Мониторинг и обнаружение дрейфа
│
├── 🧪 tests/                         # Тестирование
│   ├── test_data.py                  # Тесты обработки данных
│   ├── test_models.py                # Тесты ML моделей
│   └── test_api.py                   # Тесты API endpoints
│
├── 🤖 models/                        # Обученные модели
│   ├── best_model.pkl               # Лучшая модель
│   └── model_versions/              # Версии моделей
│
├── 📈 reports/                       # Отчеты и аналитика
│   ├── 📁 figures/                   # Графики и визуализации
│   │   ├── roc_curves/              # ROC кривые
│   │   ├── feature_importance/      # Важность признаков
│   │   └── drift_analysis/          # Анализ дрейфа
│   └── 📁 metrics/                   # Метрики моделей
│       ├── training_metrics.json    # Метрики обучения
│       ├── validation_metrics.json  # Метрики валидации
│       └── drift_metrics.json       # Метрики дрейфа
│
├── 📓 notebooks/                     # Исследовательский анализ
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Инженерия признаков
│   └── 03_model_experiments.ipynb   # Эксперименты с моделями
│
├── ⚙️ scripts/                       # Вспомогательные скрипты
│   ├── setup_environment.sh         # Настройка окружения
│   ├── run_pipeline.py              # Запуск полного пайплайна
│   └── deploy_model.py              # Деплой модели
│
├── 🔧 .github/                       # CI/CD автоматизация
│   └── workflows/
│       ├── ci-cd.yml                   # Непрерывная доставка и интеграция│       
│       └── monitoring.yml           # Мониторинг моделей
│
├── 🐳 Dockerfile                     # Контейнеризация приложения
├── 🐳 docker-compose.yml            # Оркестрация контейнеров
├── 📋 requirements.txt              # Зависимости Python
├── ⚡ Makefile                       # Автоматизация команд
├── 🔗 .gitignore                     # Игнорируемые файлы Git
└── 📖 README.md                      # Документация проекта


## Быстрый старт

### Предварительные требования

- Python 3.9+
- Docker
- Git

## Установка

1. **Клонируйте репозиторий**

```bash
git clone <URL-репозитория>
cd credit_default_prediction
Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
Установите зависимости

```bash
pip install -r requirements-core.txt #базовые зависимости
pip instal -r requirements-dev.txt #для разработки
pip install -r requirements-docker.txt #для контейнера
Настройте DVC (опционально)

```bash
dvc init
dvc remote add -d myremote /path/to/remote/storage

### Загрузка данных
```bash
python scripts/download_data.py
Использование тестовых данных
```bash
python scripts/create_sample_data.py
### Предобработка данных
```bash
python src/data/preprocess.py
### Валидация данных
```bash
python src/data/validation.py

### Обучение моделей с трекингом в MLflow
```bash
python src/models/train.py
### Просмотр экспериментов
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
### Откройте в браузере: http://localhost:5000

## Развертывание API
### Сборка Docker образа
```bash
docker build -t credit-default-api .
### Запуск API локально
```bash
docker run -p 8000:8000 credit-default-api
### Использование Docker Compose
``` bash
docker-compose up ml-api
## Использование API
###Проверка здоровья сервиса
```bash
curl http://localhost:8000/health
Ответ:

json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
### Одиночный прогноз
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "LIMIT_BAL": 200000,
       "SEX": 2,
       "EDUCATION": 2,
       "MARRIAGE": 1,
       "AGE": 35,
       "PAY_0": 2,
       "PAY_2": 2,
       "PAY_3": -1,
       "PAY_4": -1,
       "PAY_5": -2,
       "PAY_6": -2,
       "BILL_AMT1": 100000,
       "BILL_AMT2": 120000,
       "BILL_AMT3": 90000,
       "BILL_AMT4": 80000,
       "BILL_AMT5": 70000,
       "BILL_AMT6": 60000,
       "PAY_AMT1": 5000,
       "PAY_AMT2": 4000,
       "PAY_AMT3": 3000,
       "PAY_AMT4": 2000,
       "PAY_AMT5": 1000,
       "PAY_AMT6": 5000
     }'
Ответ:

json
{
  "prediction": 0,
  "probability": 0.15,
  "status": "success"
}
### Пакетный прогноз
```bash
curl -X POST "http://localhost:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '[{...}, {...}]'
##Мониторинг
###Запуск обнаружения дрейфа
```bash
python src/monitoring/drift_detection.py
###Просмотр метрик
```bash
python src/monitoring/metrics_tracking.py
## DVC Пайплайн
### Запуск полного пайплайна
```bash
dvc repro
###Просмотр графа пайплайна
```bash
dvc dag

## Тестирование
Запуск юнит-тестов
bash
pytest tests/ -v
Запуск с покрытием кода
bash
pytest --cov=src tests/
Проверка качества кода
bash

## Форматирование кода
black src/ tests/

## Проверка стиля
flake8 src/ tests/

## Проверка типов
mypy src/

## Оценка модели
Ключевые метрики отслеживаются в MLflow и сохраняются в директории reports/:
ROC-AUC - Площадь под ROC-кривой
Precision, Recall, F1-Score - Метрики классификации
Feature Importance - Важность признаков
Validation Reports - Отчеты валидации данных

## Лицензия
Этот проект создан в образовательных целях.

# 1. Сборка и запуск API
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api

# 2. Запуск мониторинга дрифта
python src/monitoring/drift_detector.py

# 3. Запуск Airflow (опционально)
airflow scheduler
airflow webserver
