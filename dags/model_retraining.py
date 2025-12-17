from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from kubernetes.client import models as k8s
import json
import mlflow

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=24),
}

dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Пайплайн автоматического переобучения кредитной скоринговой модели',
    schedule_interval='0 2 * * 0',  # Каждое воскресенье в 2:00
    catchup=False,
    tags=['mlops', 'retraining', 'credit-scoring'],
)

def check_data_drift(**context):
    """Проверка дрифта данных и принятие решения о переобучении"""
    import pandas as pd
    from scripts.monitoring.drift_detection import DriftMonitor
    
    monitor = DriftMonitor('data/processed/train.csv')
    results = monitor.run_monitoring_pipeline()
    
    # Принятие решения о переобучении
    retrain_required = False
    retrain_reason = ""
    
    if results['data_drift']['dataset_drift']:
        retrain_required = True
        retrain_reason = f"Дрифт данных обнаружен ({results['data_drift']['share_of_drifted_columns']:.2%} признаков)"
    
    if 'concept_drift' in results and results['concept_drift']['concept_drift_detected']:
        retrain_required = True
        retrain_reason = f"Концепт дрифт обнаружен (Δaccuracy={results['concept_drift']['accuracy_drift']:.4f})"
    
    # Проверка временного триггера (переобучение раз в месяц)
    last_retraining = datetime.now() - timedelta(days=30)
    if not retrain_required:
        retrain_required = True
        retrain_reason = "Плановое переобучение (раз в 30 дней)"
    
    context['ti'].xcom_push(key='retrain_required', value=retrain_required)
    context['ti'].xcom_push(key='retrain_reason', value=retrain_reason)
    context['ti'].xcom_push(key='drift_results', value=json.dumps(results))
    
    return retrain_required

def collect_new_data(**context):
    """Сбор новых данных для переобучения"""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # В реальном проекте здесь будет запрос к БД
    # Для демо используем генерацию данных
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 90 дней истории
    
    print(f"Сбор данных с {start_date} по {end_date}")
    
    # Здесь должен быть код сбора реальных данных
    # new_data = pd.read_sql_query(query, db_connection)
    
    # Для демо создаем тестовые данные
    n_samples = 10000
    n_features = 30
    
    new_data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Добавление целевой переменной
    new_data['target'] = np.random.randint(0, 2, n_samples)
    
    # Сохранение данных
    new_data.to_csv('data/raw/new_training_data.csv', index=False)
    
    context['ti'].xcom_push(key='new_data_samples', value=len(new_data))
    
    return f"Собрано {len(new_data)} новых образцов"

def retrain_model(**context):
    """Переобучение модели"""
    import subprocess
    import mlflow
    
    # Запуск скрипта переобучения
    result = subprocess.run(
        ['python', 'scripts/model_training/train_nn.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise Exception(f"Ошибка при переобучении: {result.stderr}")
    
    # Регистрация модели в MLflow
    with mlflow.start_run(run_name=f"retraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Загрузка метрик
        with open('reports/metrics/training_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        mlflow.log_metrics(metrics)
        
        # Логирование параметров
        mlflow.log_params({
            'model_type': 'neural_network',
            'input_size': 30,
            'retraining_reason': context['ti'].xcom_pull(key='retrain_reason')
        })
        
        # Логирование артефактов
        mlflow.log_artifact('models/credit_scoring_nn.pth')
        mlflow.log_artifact('reports/figures/feature_importance/')
        
        # Регистрация модели
        mlflow.pytorch.log_model(
            pytorch_model='models/credit_scoring_nn.pth',
            artifact_path="model",
            registered_model_name="credit_scoring_model"
        )
    
    return "Модель успешно переобучена и зарегистрирована"

def validate_model(**context):
    """Валидация переобученной модели"""
    import subprocess
    import json
    
    # Запуск валидационных тестов
    validation_results = {}
    
    # Тест производительности
    perf_result = subprocess.run(
        ['python', 'scripts/model_training/benchmark.py'],
        capture_output=True,
        text=True
    )
    
    if perf_result.returncode == 0:
        with open('reports/metrics/performance_benchmark.json', 'r') as f:
            perf_metrics = json.load(f)
        validation_results['performance'] = perf_metrics
    
    # Тест точности
    accuracy_result = subprocess.run(
        ['python', 'scripts/model_training/evaluate_model.py'],
        capture_output=True,
        text=True
    )
    
    if accuracy_result.returncode == 0:
        with open('reports/metrics/validation_metrics.json', 'r') as f:
            accuracy_metrics = json.load(f)
        validation_results['accuracy'] = accuracy_metrics
    
    # Проверка минимальных требований
    requirements_met = True
    if 'accuracy' in validation_results:
        if validation_results['accuracy']['accuracy'] < 0.8:
            requirements_met = False
    
    context['ti'].xcom_push(key='validation_results', value=json.dumps(validation_results))
    context['ti'].xcom_push(key='requirements_met', value=requirements_met)
    
    return requirements_met

def deploy_model(**context):
    """Деплой валидированной модели"""
    import subprocess
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Получение информации о лучшей модели
    latest_versions = client.get_latest_versions(
        "credit_scoring_model",
        stages=["None", "Staging", "Production"]
    )
    
    # Находим последнюю production модель
    production_model = None
    for version in latest_versions:
        if version.current_stage == "Production":
            production_model = version
            break
    
    # Получаем новую модель
    new_model = latest_versions[0]  # Последняя зарегистрированная
    
    # Сравнение метрик
    if production_model:
        # В реальном проекте здесь должно быть сравнение метрик
        # Если новая модель лучше, деплоим её
        deploy_new = True
    else:
        deploy_new = True
    
    if deploy_new:
        # Переход модели в стадию Staging
        client.transition_model_version_stage(
            name="credit_scoring_model",
            version=new_model.version,
            stage="Staging",
            archive_existing_versions=False
        )
        
        # Триггер CI/CD пайплайна для деплоя
        # В реальном проекте здесь будет вызов API или запуск GitHub Actions
        
        print(f"Модель версии {new_model.version} переведена в Staging")
        
        return f"Модель {new_model.version} готова к деплою"
    else:
        return "Новая модель не превосходит текущую production модель"

# Определение задач
check_drift_task = PythonOperator(
    task_id='check_data_drift',
    python_callable=check_data_drift,
    dag=dag,
)

collect_data_task = PythonOperator(
    task_id='collect_new_data',
    python_callable=collect_new_data,
    dag=dag,
)

retrain_model_task = KubernetesPodOperator(
    task_id='retrain_model',
    namespace='airflow',
    image='registry.yandex.net/credit-scoring/training:latest',
    cmds=['python', 'scripts/model_training/train_nn.py'],
    name='retrain-model-pod',
    is_delete_pod_volume=True,
    get_logs=True,
    dag=dag,
)

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

send_notification_task = SlackWebhookOperator(
    task_id='send_slack_notification',
    slack_webhook_conn_id='slack_webhook',
    message='''🚀 Переобучение модели кредитного скоринга завершено!
    Причина: {{ ti.xcom_pull(task_ids="check_data_drift", key="retrain_reason") }}
    Результаты: {{ ti.xcom_pull(task_ids="validate_model", key="validation_results") }}''',
    dag=dag,
)

# Определение зависимостей
check_drift_task >> collect_data_task
collect_data_task >> retrain_model_task
retrain_model_task >> validate_model_task
validate_model_task >> deploy_model_task
deploy_model_task >> send_notification_task

# Условное выполнение
from airflow.operators.python import BranchPythonOperator

def decide_to_retrain(**context):
    """Принятие решения о необходимости переобучения"""
    retrain_required = context['ti'].xcom_pull(
        task_ids='check_data_drift',
        key='retrain_required'
    )
    
    if retrain_required:
        return 'collect_new_data'
    else:
        return 'send_no_retrain_notification'

branch_task = BranchPythonOperator(
    task_id='decide_to_retrain',
    python_callable=decide_to_retrain,
    dag=dag,
)

send_no_retrain_task = SlackWebhookOperator(
    task_id='send_no_retrain_notification',
    slack_webhook_conn_id='slack_webhook',
    message='ℹ️ Переобучение не требуется на данный момент',
    dag=dag,
)

# Обновление зависимостей с учетом ветвления
check_drift_task >> branch_task
branch_task >> collect_data_task
branch_task >> send_no_retrain_task