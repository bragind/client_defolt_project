from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import pandas as pd

def get_feature_names():
    """Определение типов признаков"""
    
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
    
    return numeric_features, categorical_features, pay_history_features

def create_pipeline(model_type='random_forest', **model_params):
    """Создание Sklearn Pipeline"""
    
    numeric_features, categorical_features, pay_history_features = get_feature_names()
    
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
    
    # Полный pipeline с feature selection
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )),
        ('classifier', model)
    ])
    
    return pipeline

def get_feature_names_after_preprocessing(pipeline, X):
    """Получение имен признаков после препроцессинга"""
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Применяем препроцессор к данным
    X_transformed = preprocessor.fit_transform(X)
    
    # Получаем имена признаков из ColumnTransformer
    feature_names = []
    
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        elif name == 'cat':
            # Для one-hot encoding получаем имена категорий
            if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                feature_names.extend(cat_features)
            else:
                feature_names.extend(features)
        elif name == 'pay':
            feature_names.extend(features)
    
    return feature_names