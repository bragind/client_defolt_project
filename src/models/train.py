import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import os

from .pipeline import create_pipeline

def load_training_data():
    """Загрузка тренировочных данных"""
    train_df = pd.read_csv('data/processed/train.csv')
    
    target_col = 'default_payment_next_month'
    X = train_df.drop(target_col, axis=1)
    y = train_df[target_col]
    
    return X, y

def plot_roc_curve(y_true, y_proba, save_path=None):
    """Создание и сохранение ROC-кривой"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

def plot_feature_importance(pipeline, feature_names, save_path=None):
    """Визуализация важности признаков"""
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].feature_importances_
        
        # Получаем выбранные признаки после feature selection
        selected_mask = pipeline.named_steps['feature_selection'].get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        selected_importances = importances[selected_mask]
        
        # Сортируем по важности
        indices = np.argsort(selected_importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(min(20, len(selected_importances))), 
                selected_importances[indices][:20])
        plt.xticks(range(min(20, len(selected_importances))), 
                  [selected_features[i] for i in indices[:20]], rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt

def train_model():
    """Основная функция обучения"""
    
    # Создание директорий
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Настройка MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("credit_default_prediction")
    
    # Загрузка данных
    X, y = load_training_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Эксперименты с разными моделями и параметрами
    experiments = [
        {
            'model_type': 'logistic',
            'params': {'C': 0.1, 'max_iter': 1000, 'class_weight': 'balanced'},
            'name': 'Logistic_Regression_C_0.1'
        },
        {
            'model_type': 'logistic', 
            'params': {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'},
            'name': 'Logistic_Regression_C_1.0'
        },
        {
            'model_type': 'logistic',
            'params': {'C': 10.0, 'max_iter': 1000, 'class_weight': 'balanced'},
            'name': 'Logistic_Regression_C_10.0'
        },
        {
            'model_type': 'random_forest',
            'params': {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'},
            'name': 'Random_Forest_100_10'
        },
        {
            'model_type': 'random_forest',
            'params': {'n_estimators': 200, 'max_depth': 15, 'class_weight': 'balanced'},
            'name': 'Random_Forest_200_15'
        },
        {
            'model_type': 'random_forest',
            'params': {'n_estimators': 50, 'max_depth': 8, 'class_weight': 'balanced'},
            'name': 'Random_Forest_50_8'
        }
    ]
    
    best_score = 0
    best_model = None
    best_run_id = None
    
    training_metrics = {}
    
    for i, exp in enumerate(experiments):
        with mlflow.start_run(run_name=exp['name']):
            print(f"\nTraining {exp['name']}...")
            
            # Логирование параметров
            mlflow.log_params(exp['params'])
            mlflow.log_param('model_type', exp['model_type'])
            
            # Создание и обучение pipeline
            pipeline = create_pipeline(
                model_type=exp['model_type'],
                **exp['params']
            )
            
            pipeline.fit(X_train, y_train)
            
            # Предсказания
            y_pred = pipeline.predict(X_val)
            y_proba = pipeline.predict_proba(X_val)[:, 1]
            
            # Вычисление метрик
            metrics = {
                'roc_auc': roc_auc_score(y_val, y_proba),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1_score': f1_score(y_val, y_pred),
                'accuracy': (y_pred == y_val).mean()
            }
            
            # Логирование метрик
            mlflow.log_metrics(metrics)
            
            # Создание и сохранение ROC-кривой
            roc_plot = plot_roc_curve(y_val, y_proba, f'plots/roc_curve_{i}.png')
            mlflow.log_artifact(f'plots/roc_curve_{i}.png')
            roc_plot.close()
            
            # Визуализация важности признаков (для tree-based моделей)
            if exp['model_type'] == 'random_forest':
                feature_names = X_train.columns.tolist()
                feature_importance_plot = plot_feature_importance(
                    pipeline, feature_names, f'plots/feature_importance_{i}.png'
                )
                if feature_importance_plot:
                    mlflow.log_artifact(f'plots/feature_importance_{i}.png')
                    feature_importance_plot.close()
            
            # Логирование модели
            mlflow.sklearn.log_model(pipeline, "model")
            
            # Сохранение лучшей модели
            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = pipeline
                best_run_id = mlflow.active_run().info.run_id
            
            training_metrics[exp['name']] = metrics
            
            print(f"Completed {exp['name']} with ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Сохранение лучшей модели
    if best_model is not None:
        joblib.dump(best_model, 'models/best_model.pkl')
        mlflow.sklearn.log_model(best_model, "best_model")
        
        # Сохранение метрик обучения
        with open('reports/training_metrics.json', 'w') as f:
            json.dump({
                'best_model': best_run_id,
                'best_score': best_score,
                'all_metrics': training_metrics
            }, f, indent=2)
        
        print(f"\nBest model: {best_run_id} with ROC-AUC: {best_score:.4f}")
        print("Training completed successfully!")
    
    return best_model

if __name__ == "__main__":
    train_model()