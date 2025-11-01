FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Установка DVC
RUN pip install dvc[s3]

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .
COPY pyproject.toml .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data/raw data/processed models reports plots

# Установка прав доступа
RUN chmod +x src/*.py

# Expose порт для API
EXPOSE 8000

# Команда по умолчанию
CMD ["python", "src/api/app.py"]