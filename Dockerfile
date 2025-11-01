FROM python:3.9-slim

WORKDIR /app

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание необходимых директорий
RUN mkdir -p data/raw data/processed models reports plots

EXPOSE 8000

CMD ["python", "src/api/app.py"]