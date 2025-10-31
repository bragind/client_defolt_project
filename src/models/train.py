import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import os

def create_pipeline(model_type='random_forest', **model_params):
    """Создание Sklearn Pipeline"""
    
    # Определяем типы признаков
    numeric_features = [
        'LIMIT_BAL', 'AGE', 
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        'PAY_MEAN', 'PAY_STD', 'PAY_MAX', 'PAY_MIN',
        'BILL_AMT_MEAN', 'BILL_AMT_STD', 'BILL_AMT_TOTAL',
        'PAY_AMT_MEAN', 'PAY_AMT_TOTAL',
        'PAYMENT_RATIO', 'CREDIT_UTILIZATION'
    ]
    
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE_BINNED']
    pay_history_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    # Трансформеры для разных типов признаков
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    pay_history_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Комбинированный препроцессор
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('pay', pay_history_transformer, pay_history_features)
        ]
    )
    
    # Выбор модели
    if model_type == 'random_forest':
        model = RandomForestClassifier(**model_params)
    elif model_type == 'logistic':
        model = LogisticRegression(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Упрощенный pipeline БЕЗ feature selection
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

def load_training_data():
    """Загрузка тренировочных данных"""
    try:
        train_df = pd.read_csv('data/processed/train.csv')
        
        target_col = 'default_payment_next_month'
        if target_col not in train_df.columns:
            # Попробуем найти целевую переменную
            possible_targets = [col for col in train_df.columns if 'default' in col.lower()]
            if possible_targets:
                target_col = possible_targets[0]
                print(f"Using target column: {target_col}")
            else:
                raise ValueError("Target column not found in training data")
        
        X = train_df.drop(target_col, axis=1)
        y = train_df[target_col]
        
        print(f"Loaded training data: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    except Exception as e:
        print(f"Error loading training data: {e}")
        print("Please run data preprocessing first: python src/data/preprocess.py")
        raise

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
        print(f"ROC curve saved to: {save_path}")
    
    return plt

def save_train_metrics(best_model_name, best_roc_auc, results):
    """Сохранение метрик обучения в JSON файл"""
    train_metrics = {
        "best_model": best_model_name,
        "best_roc_auc": float(best_roc_auc),
        "all_models": {}
    }
    
    for model_name, metrics in results.items():
        train_metrics["all_models"][model_name] = {
            "roc_auc": float(metrics["roc_auc"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"]),
            "accuracy": float(metrics["accuracy"])
        }
    
    # Сохранение метрик в JSON файл
    os.makedirs('reports', exist_ok=True)
    with open('reports/train_metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print("Training metrics saved to reports/train_metrics.json")
    return train_metrics

def train_model():
    """Основная функция обучения"""
    
    print("Starting model training...")
    
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
    print(f"Train target distribution: {y_train.value_counts().to_dict()}")
    print(f"Validation target distribution: {y_val.value_counts().to_dict()}")
    
    # Эксперименты с разными моделями и параметрами
    experiments = [
        {
            'model_type': 'logistic',
            'params': {'C': 0.1, 'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42},
            'name': 'Logistic_Regression_C_0.1'
        },
        {
            'model_type': 'logistic', 
            'params': {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42},
            'name': 'Logistic_Regression_C_1.0'
        },
        {
            'model_type': 'logistic',
            'params': {'C': 10.0, 'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42},
            'name': 'Logistic_Regression_C_10.0'
        },
        {
            'model_type': 'random_forest',
            'params': {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42},
            'name': 'Random_Forest_100_10'
        },
        {
            'model_type': 'random_forest',
            'params': {'n_estimators': 200, 'max_depth': 15, 'class_weight': 'balanced', 'random_state': 42},
            'name': 'Random_Forest_200_15'
        },
        {
            'model_type': 'random_forest',
            'params': {'n_estimators': 50, 'max_depth': 8, 'class_weight': 'balanced', 'random_state': 42},
            'name': 'Random_Forest_50_8'
        }
    ]
    
    best_score = 0
    best_model = None
    best_run_id = None
    best_model_name = None
    
    training_metrics = {}
    
    for i, exp in enumerate(experiments):
        try:
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
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1_score': f1_score(y_val, y_pred, zero_division=0),
                    'accuracy': (y_pred == y_val).mean()
                }
                
                # Логирование метрик
                mlflow.log_metrics(metrics)
                
                # Создание и сохранение ROC-кривой
                roc_plot = plot_roc_curve(y_val, y_proba, f'plots/roc_curve_{i}.png')
                mlflow.log_artifact(f'plots/roc_curve_{i}.png')
                roc_plot.close()
                
                # Логирование модели
                mlflow.sklearn.log_model(pipeline, "model")
                
                # Сохранение лучшей модели
                if metrics['roc_auc'] > best_score:
                    best_score = metrics['roc_auc']
                    best_model = pipeline
                    best_run_id = mlflow.active_run().info.run_id
                    best_model_name = exp['name']
                
                training_metrics[exp['name']] = metrics
                
                print(f"Completed {exp['name']} with ROC-AUC: {metrics['roc_auc']:.4f}")
                
        except Exception as e:
            print(f"Error training {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Сохранение лучшей модели
    if best_model is not None:
        joblib.dump(best_model, 'models/best_model.pkl')
        print(f"\nBest model saved: {best_model_name} with ROC-AUC: {best_score:.4f}")
        
        # Сохранение метрик обучения в файл train_metrics.json
        save_train_metrics(best_model_name, best_score, training_metrics)
        
        print("Training completed successfully!")
        
        # Вывод сводки
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        for model_name, metrics in training_metrics.items():
            print(f"{model_name}:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print()
        
        print(f"BEST MODEL: {best_model_name}")
        print(f"BEST ROC-AUC: {best_score:.4f}")
        
        # Информация о MLflow
        print(f"\nMLflow UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")
        print(f"Best model run ID: {best_run_id}")
        
    else:
        print("No model was successfully trained!")
        return None
    
    return best_model

if __name__ == "__main__":
    train_model()