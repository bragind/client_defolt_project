import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset import ClassificationPreset
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class DriftMonitor:
    def __init__(self, reference_data_path: str, current_window_days: int = 7):
        """
        Мониторинг дрифта данных и концепта
        
        Args:
            reference_data_path: путь к референсным данным
            current_window_days: размер окна для текущих данных (дни)
        """
        self.reference_data = pd.read_csv(reference_data_path)
        self.current_window_days = current_window_days
        self.client = MlflowClient()
        
    def load_current_data(self) -> pd.DataFrame:
        """Загрузка текущих данных за указанный период"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.current_window_days)
        
        # Загрузка данных из базы или API
        # В реальном проекте здесь будет запрос к БД
        query = f"""
        SELECT * FROM predictions 
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        """
        
        # Для демо используем сгенерированные данные
        n_samples = len(self.reference_data)
        current_data = self.reference_data.copy()
        
        # Добавляем небольшой шум для симуляции дрифта
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, current_data.shape)
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        current_data[numeric_cols] = current_data[numeric_cols] + noise[:len(numeric_cols)]
        
        return current_data
    
    def detect_data_drift(self) -> Dict:
        """Детектирование дрифта данных"""
        
        current_data = self.load_current_data()
        
        # Создание отчета Evidently AI
        data_drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        # Получение метрик дрифта
        report_result = data_drift_report.as_dict()
        
        # Извлечение ключевых метрик
        drift_metrics = {
            'dataset_drift': report_result['metrics'][0]['result']['dataset_drift'],
            'number_of_drifted_columns': report_result['metrics'][0]['result']['number_of_drifted_columns'],
            'share_of_drifted_columns': report_result['metrics'][0]['result']['share_of_drifted_columns'],
            'timestamp': datetime.now().isoformat(),
            'features_drift': {}
        }
        
        # Детали по каждому признаку
        for metric in report_result['metrics']:
            if 'column_name' in metric:
                drift_metrics['features_drift'][metric['column_name']] = {
                    'drift_score': metric['result'].get('drift_score', 0),
                    'statistical_test': metric['result'].get('statistical_test', ''),
                    'drift_detected': metric['result'].get('drift_detected', False)
                }
        
        return drift_metrics
    
    def detect_concept_drift(self, y_true: np.array, y_pred: np.array) -> Dict:
        """Детектирование концепт дрифта"""
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        # Сравнение с референсными метриками
        with open('reports/metrics/training_metrics.json', 'r') as f:
            reference_metrics = json.load(f)
        
        current_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred)
        }
        
        # Расчет дрифта метрик
        concept_drift = {
            'accuracy_drift': reference_metrics['accuracy'] - current_metrics['accuracy'],
            'f1_drift': reference_metrics['f1_score'] - current_metrics['f1_score'],
            'roc_auc_drift': reference_metrics['roc_auc'] - current_metrics['roc_auc'],
            'concept_drift_detected': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Проверка на значимый дрифт
        drift_threshold = 0.05
        concept_drift['concept_drift_detected'] = any(
            abs(drift) > drift_threshold for drift in [
                concept_drift['accuracy_drift'],
                concept_drift['f1_drift'],
                concept_drift['roc_auc_drift']
            ]
        )
        
        return concept_drift
    
    def monitor_shadow_deployment(self, 
                                 production_model_version: str,
                                 shadow_model_version: str,
                                 inference_data: pd.DataFrame) -> Dict:
        """Сравнение моделей в shadow deployment"""
        
        # Загрузка моделей
        prod_model = self.load_model_from_mlflow(production_model_version)
        shadow_model = self.load_model_from_mlflow(shadow_model_version)
        
        # Получение предсказаний
        prod_predictions = prod_model.predict(inference_data)
        shadow_predictions = shadow_model.predict(inference_data)
        
        # Сравнение метрик
        comparison = {
            'production_model': production_model_version,
            'shadow_model': shadow_model_version,
            'predictions_agreement': np.mean(prod_predictions == shadow_predictions),
            'mse_difference': np.mean((prod_predictions - shadow_predictions) ** 2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Логирование в MLflow
        with mlflow.start_run():
            mlflow.log_metrics({
                'predictions_agreement': comparison['predictions_agreement'],
                'mse_difference': comparison['mse_difference']
            })
            
            mlflow.log_dict(comparison, 'model_comparison.json')
        
        return comparison
    
    def load_model_from_mlflow(self, version: str):
        """Загрузка модели из MLflow"""
        model_uri = f"models:/credit_scoring_model/{version}"
        return mlflow.pyfunc.load_model(model_uri)
    
    def run_monitoring_pipeline(self) -> Dict:
        """Запуск полного пайплайна мониторинга"""
        
        results = {
            'data_drift': self.detect_data_drift(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Если обнаружен дрифт данных, проверяем концепт дрифт
        if results['data_drift']['dataset_drift']:
            print("Дрифт данных обнаружен! Проверка концепт дрифта...")
            
            # Загрузка текущих истинных меток (в реальном проекте из БД)
            # Здесь для демо используем сгенерированные данные
            current_data = self.load_current_data()
            y_true = current_data['target'].values
            y_pred = np.random.randint(0, 2, len(y_true))  # Заглушка для демо
            
            results['concept_drift'] = self.detect_concept_drift(y_true, y_pred)
        
        # Сохранение результатов
        self.save_monitoring_results(results)
        
        # Отправка алертов при необходимости
        if self.should_send_alert(results):
            self.send_alert(results)
        
        return results
    
    def save_monitoring_results(self, results: Dict):
        """Сохранение результатов мониторинга"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранение JSON
        with open(f'reports/metrics/drift_metrics_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Обновление latest файла
        with open('reports/metrics/drift_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Логирование в MLflow
        with mlflow.start_run(run_name=f"drift_monitoring_{timestamp}"):
            mlflow.log_metrics({
                'data_drift_detected': int(results['data_drift']['dataset_drift']),
                'drifted_columns': results['data_drift']['number_of_drifted_columns']
            })
            
            if 'concept_drift' in results:
                mlflow.log_metrics({
                    'concept_drift_detected': int(results['concept_drift']['concept_drift_detected']),
                    'accuracy_drift': results['concept_drift']['accuracy_drift']
                })
    
    def should_send_alert(self, results: Dict) -> bool:
        """Определение необходимости отправки алерта"""
        
        # Алгоритм принятия решения об алерте
        conditions = [
            results['data_drift']['dataset_drift'],
            results['data_drift']['share_of_drifted_columns'] > 0.3,
        ]
        
        if 'concept_drift' in results:
            conditions.extend([
                results['concept_drift']['concept_drift_detected'],
                abs(results['concept_drift']['accuracy_drift']) > 0.1
            ])
        
        return any(conditions)
    
    def send_alert(self, results: Dict):
        """Отправка алерта"""
        
        import smtplib
        from email.mime.text import MIMEText
        
        # Формирование сообщения
        subject = "🚨 Обнаружен дрифт в кредитной скоринговой системе"
        
        body = f"""
        Время обнаружения: {results['timestamp']}
        
        Дрифт данных: {'ОБНАРУЖЕН' if results['data_drift']['dataset_drift'] else 'не обнаружен'}
        Количество дрифтующих признаков: {results['data_drift']['number_of_drifted_columns']}
        Доля дрифтующих признаков: {results['data_drift']['share_of_drifted_columns']:.2%}
        
        """
        
        if 'concept_drift' in results:
            body += f"""
            Концепт дрифт: {'ОБНАРУЖЕН' if results['concept_drift']['concept_drift_detected'] else 'не обнаружен'}
            Изменение accuracy: {results['concept_drift']['accuracy_drift']:.4f}
            Изменение F1-score: {results['concept_drift']['f1_drift']:.4f}
            """
        
        body += "\n\nРекомендуемые действия:\n1. Проверить качество входных данных\n2. Запустить переобучение модели\n3. Проанализировать изменения в бизнес-процессах"
        
        # Отправка email (пример)
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'mlops@credit-scoring.com'
        msg['To'] = 'data-science-team@credit-scoring.com'
        
        # В реальном проекте здесь будет код отправки email
        print(f"ALERT: {subject}")
        print(body)
        
        # Также можно отправлять в Slack/Telegram/etc.
        self.send_slack_alert(results)

def main():
    """Основная функция запуска мониторинга"""
    
    # Инициализация монитора
    monitor = DriftMonitor(
        reference_data_path='data/processed/train.csv',
        current_window_days=7
    )
    
    # Запуск мониторинга
    results = monitor.run_monitoring_pipeline()
    
    print(f"Мониторинг завершен. Результаты сохранены.")
    print(f"Дрифт данных: {'ОБНАРУЖЕН' if results['data_drift']['dataset_drift'] else 'не обнаружен'}")
    
    if 'concept_drift' in results:
        print(f"Концепт дрифт: {'ОБНАРУЖЕН' if results['concept_drift']['concept_drift_detected'] else 'не обнаружен'}")

if __name__ == "__main__":
    main()