from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time
from functools import wraps

# Метрики приложения
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Active HTTP requests'
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction latency',
    ['model_version']
)

MODEL_METRICS = Gauge(
    'model_metrics',
    'Model performance metrics',
    ['metric_name', 'model_version']
)

def monitor_request(func):
    """Декоратор для мониторинга запросов"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        
        try:
            response = await func(*args, **kwargs)
            status = response.status_code if hasattr(response, 'status_code') else 200
            REQUEST_COUNT.labels(
                method=kwargs.get('method', 'GET'),
                endpoint=func.__name__,
                status=status
            ).inc()
            return response
        except Exception as e:
            REQUEST_COUNT.labels(
                method=kwargs.get('method', 'GET'),
                endpoint=func.__name__,
                status=500
            ).inc()
            raise e
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(
                method=kwargs.get('method', 'GET'),
                endpoint=func.__name__
            ).observe(latency)
            ACTIVE_REQUESTS.dec()
    
    return wrapper

def update_model_metrics(metrics: dict, model_version: str = "latest"):
    """Обновление метрик модели"""
    for metric_name, value in metrics.items():
        MODEL_METRICS.labels(
            metric_name=metric_name,
            model_version=model_version
        ).set(value)

def get_metrics():
    """Эндпоинт для получения метрик Prometheus"""
    return Response(generate_latest(), media_type="text/plain")