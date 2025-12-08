# src/monitoring/drift_detector.py
from evidently.report import Report
from evidently.metrics import DataDriftTable
import pandas as pd

def generate_drift_report(reference_path, current_path, output_path):
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)
    
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(output_path)
    print(f"📊 Отчет о дрейфе сохранен: {output_path}")

if __name__ == "__main__":
    generate_drift_report(
        reference_path="data/processed/train.csv",
        current_path="data/processed/current.csv",
        output_path="monitoring/reports/drift_report.html"
    )