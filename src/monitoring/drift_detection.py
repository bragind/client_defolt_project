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
        return psi
    
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
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'psi': psi,
                    'drift_detected': ks_pvalue < 0.05 or psi > 0.1
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
                        'chi2_statistic': chi2,
                        'p_value': p_value,
                        'drift_detected': p_value < 0.05
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
                'total_features': total_features,
                'drifted_features': drifted_features,
                'drift_ratio': drift_ratio,
                'overall_drift_detected': drift_ratio > 0.1  # Порог 10%
            }
        }

def monitor_prediction_drift(api_url: str, test_data_path: str, n_samples: int = 1000):
    """Мониторинг дрифта предсказаний"""
    
    # Загрузка тестовых данных
    test_data = pd.read_csv(test_data_path)
    sample_data = test_data.sample(min(n_samples, len(test_data)))
    
    # Получение предсказаний от API
    predictions = []
    
    for _, row in sample_data.iterrows():
        try:
            # Подготовка данных для API
            payload = row.drop('default_payment_next_month').to_dict()
            
            # Отправка запроса к API
            response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                predictions.append(result['default_probability'])
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
            'mean': np.mean(successful_predictions),
            'std': np.std(successful_predictions),
            'min': np.min(successful_predictions),
            'max': np.max(successful_predictions),
            'drift_detected': np.mean(successful_predictions) > 0.3  # Пример порога
        }
        
        return prediction_stats
    else:
        return {'error': 'No successful predictions'}

def main():
    """Основная функция мониторинга дрифта"""
    print("Starting drift monitoring...")
    
    # Загрузка reference данных (тренировочные)
    reference_data = pd.read_csv('data/processed/train.csv')
    
    # Загрузка current данных (имитация новых данных)
    current_data = pd.read_csv('data/processed/test.csv').sample(1000)
    
    # Инициализация детектора
    detector = DataDriftDetector(reference_data)
    
    # Обнаружение дрифта
    drift_report = detector.detect_drift(current_data)
    
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
    
    # Сохранение отчета
    os.makedirs('reports', exist_ok=True)
    with open('reports/drift_report.json', 'w') as f:
        json.dump(drift_report, f, indent=2)
    
    print("Drift monitoring completed!")
    print(f"Drift ratio: {drift_report['summary']['drift_ratio']:.2%}")
    print(f"Overall drift detected: {drift_report['summary']['overall_drift_detected']}")
    
    return drift_report

if __name__ == "__main__":
    report = main()