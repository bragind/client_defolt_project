# Client Defolt Model

Этот проект реализует систему для автоматизированной разработки и обслуживания PD-модели (Probability of Default) для кредитного отдела банка.

## Структура проекта

- `data/`: исходные и обработанные данные.
- `src/`: основной код проекта.
- `notebooks/`: Jupyter-ноутбуки для EDA и экспериментов.
- `models/`: сохраненные модели.
- `tests/`: unit-тесты.
- `Dockerfile`: для контейнеризации.
- `requirements.txt`: зависимости.
- `.github/workflows/ci-cd.yml`: CI/CD пайплайн.

## Инструкция по запуску

1. Установите зависимости: `pip install -r requirements.txt`
2. Запустите подготовку данных: `python src/data/make_dataset.py`
3. Запустите обучение: `python src/models/train.py`
4. Запустите API: `uvicorn src.api.app:app --reload`

## Требования

- Python 3.9+
- Git
- DVC
- MLflow
