# src/models/train_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

class CreditScoringNN(nn.Module):
    def __init__(self, input_size):
        super(CreditScoringNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_neural_network():
    # Создание директорий
    Path("models").mkdir(exist_ok=True)
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    
    # Загрузка данных
    print("🔍 Loading training data...")
    data = pd.read_csv('data/processed/train.csv')
    
    # Указание правильного имени целевой переменной
    target_col = 'default_payment_next_month'
    
    if target_col not in data.columns:
        available = list(data.columns)
        raise ValueError(f"Target column '{target_col}' not found. Available columns: {available}")
    
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
    
    print(f"📊 Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"🎯 Target distribution: {np.bincount(y)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Конвертация в тензоры
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    
    # Инициализация модели
    model = CreditScoringNN(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    epochs = 100
    batch_size = 32
    
    print("🚀 Starting neural network training...")
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Валидация
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_preds = (val_outputs > 0.5).float()
            val_accuracy = (val_preds == y_val_tensor).float().mean()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, '
                  f'Val Loss: {val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Сохранение модели
    torch.save(model.state_dict(), 'models/credit_scoring_nn.pth')
    
    # Сохранение метрик
    metrics = {
        'val_loss': float(val_loss.item()),
        'val_accuracy': float(val_accuracy.item())
    }
    
    with open('reports/metrics/nn_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✅ Модель успешно обучена и сохранена в models/credit_scoring_nn.pth")
    return model, X_train.shape[1]

if __name__ == "__main__":
    model, input_size = train_neural_network()