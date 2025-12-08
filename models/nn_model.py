# src/models/nn_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class CreditScoringNN(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=[64, 32], dropout_rate=0.3, epochs=100, batch_size=32):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        for dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X = self.scaler.fit_transform(X)
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device).view(-1, 1)

        self.model = self._build_model(X.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['model', 'scaler'])
        X = check_array(X)
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probas = self.model(X).cpu().numpy()
        return np.concatenate([1 - probas, probas], axis=1)

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)