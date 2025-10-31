from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
from typing import List, Optional
import os
import sys

# Добавляем путь к корню проекта для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

app = FastAPI(
    title="Credit Default Prediction API",
    description="API для предсказания вероятности дефолта клиента",
    version="1.0.0"
)

# Глобальная переменная для модели
model = None

class CreditData(BaseModel):
    LIMIT_BAL: float = Field(..., description="Кредитный лимит")
    SEX: int = Field(..., description="Пол (1=мужской, 2=женский)")
    EDUCATION: int = Field(..., description="Образование (1=аспирант, 2=университет, 3=школа, 4=другое)")
    MARRIAGE: int = Field(..., description="Семейное положение (1=женат/замужем, 2=холост/не замужем, 3=другое)")
    AGE: int = Field(..., description="Возраст")
    PAY_0: int = Field(..., description="Статус погашения в сентябре (-2=нет использования, -1=оплачено полностью, 0=использование revolving credit, 1=задержка 1 месяц, ...)")
    PAY_2: int = Field(..., description="Статус погашения в августе")
    PAY_3: int = Field(..., description="Статус погашения в июле")
    PAY_4: int = Field(..., description="Статус погашения в июне")
    PAY_5: int = Field(..., description="Статус погашения в мае")
    PAY_6: int = Field(..., description="Статус погашения в апреле")
    BILL_AMT1: float = Field(..., description="Сумма счета в сентябре")
    BILL_AMT2: float = Field(..., description="Сумма счета в августе")
    BILL_AMT3: float = Field(..., description="Сумма счета в июле")
    BILL_AMT4: float = Field(..., description="Сумма счета в июне")
    BILL_AMT5: float = Field(..., description="Сумма счета в мае")
    BILL_AMT6: float = Field(..., description="Сумма счета в апреле")
    PAY_AMT1: float = Field(..., description="Сумма предыдущего платежа в сентябре")
    PAY_AMT2: float = Field(..., description="Сумма предыдущего платежа в августе")
    PAY_AMT3: float = Field(..., description="Сумма предыдущего платежа в июле")
    PAY_AMT4: float = Field(..., description="Сумма предыдущего платежа в июне")
    PAY_AMT5: float = Field(..., description="Сумма предыдущего платежа в мае")
    PAY_AMT6: float = Field(..., description="Сумма предыдущего платежа в апреле")

class PredictionResponse(BaseModel):
    default_probability: float = Field(..., description="Вероятность дефолта")
    default_class: int = Field(..., description="Класс предсказания (0=нет дефолта, 1=дефолт)")
    risk_level: str = Field(..., description="Уровень риска")
    model_version: str = Field(..., description="Версия модели")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

def load_model():
    """Загрузка модели при старте приложения"""
    global model
    try:
        model_path = 'models/best_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded successfully")
        else:
            print(f"Model file not found at {model_path}")
            print("Please train the model first: python src/models/train.py")
    except Exception as e:
        print(f"Error loading model: {e}")

def create_features_from_input(df):
    """Feature Engineering для входных данных (такой же как при обучении)"""
    # Создание агрегированных признаков из истории платежей
    pay_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['PAY_MEAN'] = df[pay_columns].mean(axis=1)
    df['PAY_STD'] = df[pay_columns].std(axis=1)
    df['PAY_MAX'] = df[pay_columns].max(axis=1)
    df['PAY_MIN'] = df[pay_columns].min(axis=1)
    
    # Агрегация сумм счетов
    bill_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    df['BILL_AMT_MEAN'] = df[bill_columns].mean(axis=1)
    df['BILL_AMT_STD'] = df[bill_columns].std(axis=1)
    df['BILL_AMT_TOTAL'] = df[bill_columns].sum(axis=1)
    
    # Агрегация сумм платежей
    pay_amt_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    df['PAY_AMT_MEAN'] = df[pay_amt_columns].mean(axis=1)
    df['PAY_AMT_TOTAL'] = df[pay_amt_columns].sum(axis=1)
    
    # Биннинг возраста
    df['AGE_BINNED'] = pd.cut(
        df['AGE'], 
        bins=[20, 30, 40, 50, 60, 80], 
        labels=[0, 1, 2, 3, 4]
    ).astype(int)
    
    # Создание отношения платежей к счетам
    df['PAYMENT_RATIO'] = df['PAY_AMT_TOTAL'] / (df['BILL_AMT_TOTAL'] + 1)
    
    # Создание отношения кредитного лимита к общим счетам
    df['CREDIT_UTILIZATION'] = df['BILL_AMT_TOTAL'] / (df['LIMIT_BAL'] + 1)
    
    # Заполняем возможные NaN значения
    df = df.fillna(0)
    
    return df

@app.on_event("startup")
async def startup_event():
    """Загрузка модели при запуске приложения"""
    load_model()

@app.get("/", response_model=dict)
async def read_root():
    return {
        "message": "Credit Default Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version="1.0.0" if model is not None else "none"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CreditData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check if the model file exists.")
    
    try:
        # Преобразование входных данных в DataFrame
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Добавление фичей (как в обучении)
        input_df = create_features_from_input(input_df)
        
        # Предсказание
        probability = model.predict_proba(input_df)[0, 1]
        prediction = int(probability > 0.5)
        
        # Определение уровня риска
        if probability < 0.2:
            risk_level = "low"
        elif probability < 0.5:
            risk_level = "medium"
        elif probability < 0.8:
            risk_level = "high"
        else:
            risk_level = "very high"
        
        return PredictionResponse(
            default_probability=round(probability, 4),
            default_class=prediction,
            risk_level=risk_level,
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(data: List[CreditData]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Преобразование списка данных в DataFrame
        input_dicts = [item.dict() for item in data]
        input_df = pd.DataFrame(input_dicts)
        
        # Добавление фичей
        input_df = create_features_from_input(input_df)
        
        # Пакетное предсказание
        probabilities = model.predict_proba(input_df)[:, 1]
        predictions = (probabilities > 0.5).astype(int)
        
        # Формирование ответа
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            if prob < 0.2:
                risk_level = "low"
            elif prob < 0.5:
                risk_level = "medium"
            elif prob < 0.8:
                risk_level = "high"
            else:
                risk_level = "very high"
            
            results.append({
                "id": i,
                "default_probability": round(prob, 4),
                "default_class": int(pred),
                "risk_level": risk_level
            })
        
        return {
            "predictions": results,
            "total_count": len(results),
            "model_version": "1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        model_info = {
            "model_type": type(model.named_steps['classifier']).__name__,
            "model_version": "1.0.0",
            "pipeline_steps": list(model.named_steps.keys())
        }
        
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)