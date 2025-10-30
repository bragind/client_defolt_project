import pandas as pd
import numpy as np
import json
import great_expectations as ge
from great_expectations.core.expectation_configuration import ExpectationConfiguration

def create_expectation_suite():
    """Создание набора ожиданий для данных"""
    suite_name = "credit_default_suite"
    
    # Создание контекста
    context = ge.get_context()
    
    # Создание suite
    suite = context.create_expectation_suite(
        suite_name, 
        overwrite_existing=True
    )
    
    # Добавление ожиданий
    expectations = [
        # Проверка структуры данных
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": [
                    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                    'BILL_AMT5', 'BILL_AMT6',
                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
                    'PAY_AMT5', 'PAY_AMT6',
                    'default_payment_next_month',
                    # Новые фичи
                    'PAY_MEAN', 'PAY_STD', 'PAY_MAX', 'PAY_MIN',
                    'BILL_AMT_MEAN', 'BILL_AMT_STD', 'BILL_AMT_TOTAL',
                    'PAY_AMT_MEAN', 'PAY_AMT_TOTAL',
                    'AGE_BINNED', 'PAYMENT_RATIO', 'CREDIT_UTILIZATION'
                ]
            }
        ),
        
        # Проверка на отсутствие null в ключевых колонках
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "LIMIT_BAL"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "default_payment_next_month"}
        ),
        
        # Проверка диапазонов значений
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "AGE", "min_value": 20, "max_value": 80}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={"column": "LIMIT_BAL", "min_value": 0, "max_value": 1000000}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "SEX", "value_set": [1, 2]}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={"column": "default_payment_next_month", "value_set": [0, 1]}
        ),
        
        # Проверка уникальности (нет полностью дублированных строк)
        ExpectationConfiguration(
            expectation_type="expect_compound_columns_to_be_unique",
            kwargs={"column_list": [
                'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
            ]}
        )
    ]
    
    for expectation in expectations:
        suite.add_expectation(expectation)
    
    context.save_expectation_suite(suite)
    return suite

def validate_dataset(df, dataset_type="train"):
    """Валидация датасета"""
    print(f"Validating {dataset_type} data...")
    
    context = ge.get_context()
    suite = context.get_expectation_suite("credit_default_suite")
    
    # Создание batch
    batch = context.get_batch(
        batch_request={"dataset": df, "datasource": "pandas"},
        expectation_suite=suite
    )
    
    # Запуск валидации
    results = batch.validate()
    
    # Сохранение отчета
    report = {
        "dataset_type": dataset_type,
        "success": results.success,
        "statistics": {
            "evaluated_expectations": results.statistics["evaluated_expectations"],
            "successful_expectations": results.statistics["successful_expectations"],
            "unsuccessful_expectations": results.statistics["unsuccessful_expectations"],
            "success_percent": results.statistics["success_percent"]
        },
        "results": []
    }
    
    for result in results.results:
        report["results"].append({
            "expectation_type": result.expectation_config.expectation_type,
            "success": result.success,
            "kwargs": result.expectation_config.kwargs
        })
    
    return report

def main():
    """Основная функция валидации"""
    print("Starting data validation...")
    
    # Создание набора ожиданий
    create_expectation_suite()
    print("Expectation suite created")
    
    # Загрузка train и test данных
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
    
    # Валидация train данных
    train_report = validate_dataset(train_df, "train")
    
    # Валидация test данных  
    test_report = validate_dataset(test_df, "test")
    
    # Сохранение отчетов
    with open('reports/data_validation_report.json', 'w') as f:
        json.dump({
            "train_validation": train_report,
            "test_validation": test_report
        }, f, indent=2)
    
    print("Data validation completed!")
    print(f"Train validation success: {train_report['success']}")
    print(f"Test validation success: {test_report['success']}")

if __name__ == "__main__":
    main()