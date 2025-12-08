# dags/credit_default_retrain.py
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import subprocess

def retrain_model():
    # Запуск скрипта переобучения
    result = subprocess.run([
        "python", "scripts/retrain_model.py"
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Переобучение не удалось: {result.stderr}")

def check_drift_and_trigger():
    # Здесь можно добавить логику проверки дрифта
    # Если дрифт > threshold → запуск retrain_model
    pass

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'credit_default_retrain',
    default_args=default_args,
    description='Еженедельное переобучение модели при обнаружении дрифта',
    schedule_interval=timedelta(weeks=1),
    catchup=False,
)

check_drift_task = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift_and_trigger,
    dag=dag,
)

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
)

check_drift_task >> retrain_task