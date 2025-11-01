#!/bin/bash

echo "Инициализация контейнера..."

# Копирование данных если нужно
if [ ! -f "data/raw/default_of_credit_card_clients.csv" ]; then
    echo "Пожалуйста, разместите файл данных в data/raw/default_of_credit_card_clients.csv"
    exit 1
fi

# Запуск подготовки данных
echo "Запуск подготовки данных..."
python src/data/preprocess.py

# Запуск обучения модели
echo "Запуск обучения модели..."
python src/models/train.py

echo "Инициализация завершена!"