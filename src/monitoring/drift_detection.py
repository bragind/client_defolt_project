import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import warnings
import json
import os
from typing import Dict, List, Tuple
import requests

warnings.filterwarnings('ignore')

class DataDriftDetector:
    """Класс для обнаружения дрифта данных"""
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.numerical_columns = self._get_numerical_columns()
        self.categorical_columns = self._get_categorical_columns()
    
    def _get_numerical_columns(self) -> List[str]:
        """Получение числовых колонок"""
        return self.reference_data.select_dtypes(include=[np.number]).columns.tolist()
    
    def _get_categorical_columns(self) -> List[str]:
        """Получение категориальных колонок"""
        return self.reference_data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        # Определение границ бакетов на основе expected данных
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        
        # Гистограммы для expected и actual
        expected_hist, _ = np.histogram(expected, breakpoints)
        actual_hist, _ = np.histogram(actual, breakpoints)
        
        # Нормализация в проценты
        expected_percents = expected_hist / len(expected)
        actual_percents = actual_hist / len(actual)
        
        # Избегаем деления на ноль
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Расчет PSI
        psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
        return float(psi)  # Конвертируем в Python float
    
    def detect_numerical_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Обнаружение дрифта для числовых признаков"""
        drift_scores = {}
        
        for col in self.numerical_columns:
            if col in current_data.columns:
                # KS test
                ks_stat, ks_pvalue = ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                # PSI
                psi = self.calculate_psi(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )
                
                drift_scores[col] = {
                    'ks_statistic': float(ks_stat),  # Конвертируем в Python float
                    'ks_pvalue': float(ks_pvalue),   # Конвертируем в Python float
                    'psi': psi,
                    'drift_detected': bool(ks_pvalue < 0.05 or psi > 0.1)  # Конвертируем в Python bool
                }
        
        return drift_scores
    
    def detect_categorical_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Обнаружение дрифта для категориальных признаков"""
        drift_scores = {}
        
        for col in self.categorical_columns:
            if col in current_data.columns:
                # Создание contingency table
                ref_counts = self.reference_data[col].value_counts().sort_index()
                curr_counts = current_data[col].value_counts().sort_index()
                
                # Объединение всех возможных категорий
                all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
                
                # Заполнение отсутствующих категорий нулями
                ref_vector = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_vector = [curr_counts.get(cat, 0) for cat in all_categories]
                
                # Chi-square test
                try:
                    chi2, p_value, _, _ = chi2_contingency([ref_vector, curr_vector])
                    drift_scores[col] = {
                        'chi2_statistic': float(chi2) if chi2 is not None else None,  # Конвертируем
                        'p_value': float(p_value) if p_value is not None else None,   # Конвертируем
                        'drift_detected': bool(p_value < 0.05)  # Конвертируем в Python bool
                    }
                except:
                    drift_scores[col] = {
                        'chi2_statistic': None,
                        'p_value': None,
                        'drift_detected': False
                    }
        
        return drift_scores
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Полное обнаружение дрифта"""
        numerical_drift = self.detect_numerical_drift(current_data)
        categorical_drift = self.detect_categorical_drift(current_data)
        
        # Агрегация результатов
        total_features = len(numerical_drift) + len(categorical_drift)
        drifted_features = sum(
            [1 for score in numerical_drift.values() if score['drift_detected']] +
            [1 for score in categorical_drift.values() if score['drift_detected']]
        )
        
        drift_ratio = drifted_features / total_features if total_features > 0 else 0
        
        return {
            'numerical_drift': numerical_drift,
            'categorical_drift': categorical_drift,
            'summary': {
                'total_features': int(total_features),  # Конвертируем в Python int
                'drifted_features': int(drifted_features),  # Конвертируем в Python int
                'drift_ratio': float(drift_ratio),  # Конвертируем в Python float
                'overall_drift_detected': bool(drift_ratio > 0.1)  # Конвертируем в Python bool
            }
        }

def convert_numpy_types(obj):
    """Рекурсивно конвертирует NumPy типы в нативные Python типы"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(element) for element in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def monitor_prediction_drift(api_url: str, test_data_path: str, n_samples: int = 1000):
    """Мониторинг дрифта предсказаний"""
    
    # Загрузка тестовых данных
    test_data = pd.read_csv(test_data_path)
    sample_size = min(n_samples, len(test_data))
    sample_data = test_data.sample(sample_size)
    
    # Получение предсказаний от API
    predictions = []
    
    for _, row in sample_data.iterrows():
        try:
            # Подготовка данных для API
            # Проверяем наличие целевой переменной
            if 'default_payment_next_month' in row:
                payload = row.drop('default_payment_next_month').to_dict()
            else:
                payload = row.to_dict()
            
            # Отправка запроса к API
            response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                predictions.append(result.get('default_probability', np.nan))
            else:
                print(f"API request failed: {response.status_code}")
                predictions.append(np.nan)
                
        except Exception as e:
            print(f"Error getting prediction: {e}")
            predictions.append(np.nan)
    
    # Фильтрация успешных предсказаний
    successful_predictions = [p for p in predictions if not np.isnan(p)]
    
    if len(successful_predictions) > 0:
        # Расчет статистик предсказаний
        prediction_stats = {
            'mean': float(np.mean(successful_predictions)),
            'std': float(np.std(successful_predictions)),
            'min': float(np.min(successful_predictions)),
            'max': float(np.max(successful_predictions)),
            'n_predictions': len(successful_predictions),
            'drift_detected': bool(np.mean(successful_predictions) > 0.3)  # Пример порога
        }
        
        return prediction_stats
    else:
        return {'error': 'No successful predictions'}

def main():
    """Основная функция мониторинга дрифта"""
    print("Starting drift monitoring...")
    
    try:
        # Загрузка reference данных (тренировочные)
        reference_data = pd.read_csv('data/processed/train.csv')
        print(f"Reference data loaded: {len(reference_data)} rows")
        
        # Загрузка current данных (имитация новых данных)
        current_data = pd.read_csv('data/processed/test.csv')
        print(f"Current data loaded: {len(current_data)} rows")
        
        # Определение размера выборки (не больше размера данных)
        sample_size = min(1000, len(current_data))
        current_data_sample = current_data.sample(sample_size, random_state=42)
        print(f"Using sample size: {sample_size}")
        
        # Инициализация детектора
        detector = DataDriftDetector(reference_data)
        
        # Обнаружение дрифта
        drift_report = detector.detect_drift(current_data_sample)
        
        # Мониторинг дрифта предсказаний (если API доступно)
        try:
            prediction_drift = monitor_prediction_drift(
                "http://localhost:8000",
                "data/processed/test.csv",
                n_samples=100
            )
            drift_report['prediction_drift'] = prediction_drift
        except Exception as e:
            print(f"Prediction drift monitoring failed: {e}")
            drift_report['prediction_drift'] = {'error': str(e)}
        
        # Конвертируем все NumPy типы в нативные Python типы
        drift_report_serializable = convert_numpy_types(drift_report)
        
        # Сохранение отчета
        os.makedirs('reports', exist_ok=True)
        with open('reports/drift_report.json', 'w') as f:
            json.dump(drift_report_serializable, f, indent=2, ensure_ascii=False)
        
        print("Drift monitoring completed!")
        print(f"Drift ratio: {drift_report['summary']['drift_ratio']:.2%}")
        print(f"Overall drift detected: {drift_report['summary']['overall_drift_detected']}")
        
        return drift_report_serializable
    
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        print("Please check that the data files exist in the specified paths")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    report = main()