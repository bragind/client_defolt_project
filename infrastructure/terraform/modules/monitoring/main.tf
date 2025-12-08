# Prometheus Stack with Thanos
resource "helm_release" "prometheus_stack" {
  name       = "prometheus-stack"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  create_namespace = true
  
  set {
    name  = "prometheus.prometheusSpec.retentionSize"
    value = "50GB"
  }
  
  set {
    name  = "prometheus.prometheusSpec.retentionTime"
    value = "30d"
  }
  
  set {
    name  = "grafana.adminPassword"
    value = var.grafana_password
  }
  
  set {
    name  = "grafana.persistence.enabled"
    value = "true"
  }
  
  set {
    name  = "grafana.persistence.size"
    value = "10Gi"
  }
}

# Loki for logs
resource "helm_release" "loki" {
  name       = "loki"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "loki-stack"
  namespace  = "monitoring"
  
  set {
    name  = "loki.persistence.enabled"
    value = "true"
  }
  
  set {
    name  = "loki.persistence.size"
    value = "20Gi"
  }
}

# Alertmanager configuration
resource "kubernetes_config_map" "alertmanager_config" {
  metadata {
    name      = "alertmanager-config"
    namespace = "monitoring"
  }
  
  data = {
    "alertmanager.yml" = <<-EOT
    global:
      slack_api_url: '${var.slack_webhook_url}'
    
    route:
      group_by: ['alertname']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 12h
      receiver: 'slack-notifications'
    
    receivers:
    - name: 'slack-notifications'
      slack_configs:
      - channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.summary }}'
    
    inhibit_rules:
      - source_match:
          severity: 'critical'
        target_match:
          severity: 'warning'
        equal: ['alertname', 'cluster', 'service']
    EOT
  }
}

# Custom Prometheus rules for ML monitoring
resource "kubernetes_config_map" "ml_monitoring_rules" {
  metadata {
    name      = "ml-monitoring-rules"
    namespace = "monitoring"
  }
  
  data = {
    "ml-rules.yml" = <<-EOT
    groups:
    - name: ml_monitoring
      rules:
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High prediction latency detected"
          description: "95th percentile prediction latency is above 1 second"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5%"
      
      - alert: DataDriftDetected
        expr: model_metrics{metric_name="data_drift_score"} > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected"
          description: "Data drift score is above 0.3"
      
      - alert: ModelPerformanceDegradation
        expr: model_metrics{metric_name="accuracy"} < 0.8
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Model performance degradation"
          description: "Model accuracy dropped below 80%"
    EOT
  }
}