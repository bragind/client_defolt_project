import pandas as pd
import numpy as np
import json
import os
import sys

try:
    import great_expectations as ge
    # Для версий >= 0.15.0
    from great_expectations.core import ExpectationConfiguration
    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    print("Great Expectations not available. Using simplified validation.")
    GREAT_EXPECTATIONS_AVAILABLE = False

def create_expectation_suite():
    """Создание набора ожиданий для данных"""
    if not GREAT_EXPECTATIONS_AVAILABLE:
        print("Great Expectations not installed. Skipping expectation suite creation.")
        return None
    
    try:
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
                        'default_payment_next_month'
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
            )
        ]
        
        for expectation in expectations:
            suite.add_expectation(expectation)
        
        context.save_expectation_suite(suite)
        return suite
    except Exception as e:
        print(f"Error creating expectation suite: {e}")
        return None

def simple_data_validation(df, dataset_type="train"):
    """Упрощенная валидация данных без Great Expectations"""
    print(f"Performing simple validation for {dataset_type} data...")
    
    validation_results = {
        "dataset_type": dataset_type,
        "success": True,
        "errors": [],
        "warnings": [],
        "statistics": {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }
    }
    
    # Проверка обязательных колонок
    required_columns = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
        'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 
        'PAY_AMT5', 'PAY_AMT6',
        'default_payment_next_month'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        validation_results["errors"].append(f"Missing columns: {missing_columns}")
        validation_results["success"] = False
    
    # Проверка на пропущенные значения
    missing_values = df[required_columns].isnull().sum()
    columns_with_missing = missing_values[missing_values > 0]
    if not columns_with_missing.empty:
        validation_results["warnings"].append(
            f"Columns with missing values: {columns_with_missing.to_dict()}"
        )
    
    # Проверка диапазонов значений
    if 'AGE' in df.columns:
        invalid_age = df[(df['AGE'] < 20) | (df['AGE'] > 80)]
        if len(invalid_age) > 0:
            validation_results["warnings"].append(
                f"Found {len(invalid_age)} records with AGE outside [20, 80]"
            )
    
    if 'SEX' in df.columns:
        invalid_sex = df[~df['SEX'].isin([1, 2])]
        if len(invalid_sex) > 0:
            validation_results["warnings"].append(
                f"Found {len(invalid_sex)} records with SEX not in [1, 2]"
            )
    
    if 'default_payment_next_month' in df.columns:
        invalid_target = df[~df['default_payment_next_month'].isin([0, 1])]
        if len(invalid_target) > 0:
            validation_results["errors"].append(
                f"Found {len(invalid_target)} records with invalid target values"
            )
            validation_results["success"] = False
    
    return validation_results

def validate_dataset(df, dataset_type="train"):
    """Валидация датасета"""
    print(f"Validating {dataset_type} data...")
    
    if GREAT_EXPECTATIONS_AVAILABLE:
        try:
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
        except Exception as e:
            print(f"Great Expectations validation failed: {e}")
            print("Falling back to simple validation...")
            return simple_data_validation(df, dataset_type)
    else:
        return simple_data_validation(df, dataset_type)

def main():
    """Основная функция валидации"""
    print("Starting data validation...")
    
    # Создание директорий если не существуют
    os.makedirs('reports', exist_ok=True)
    
    # Загрузка train и test данных
    try:
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run data preprocessing first: python src/data/preprocess.py")
        return
    
    # Создание набора ожиданий (если Great Expectations доступен)
    if GREAT_EXPECTATIONS_AVAILABLE:
        create_expectation_suite()
        print("Expectation suite created (if Great Expectations available)")
    else:
        print("Using simplified validation without Great Expectations")
    
    # Валидация train данных
    train_report = validate_dataset(train_df, "train")
    
    # Валидация test данных  
    test_report = validate_dataset(test_df, "test")
    
    # Сохранение отчетов
    with open('reports/data_validation_report.json', 'w') as f:
        json.dump({
            "train_validation": train_report,
            "test_validation": test_report,
            "great_expectations_used": GREAT_EXPECTATIONS_AVAILABLE
        }, f, indent=2)
    
    print("Data validation completed!")
    print(f"Train validation success: {train_report['success']}")
    print(f"Test validation success: {test_report['success']}")
    
    # Вывод предупреждений
    if 'warnings' in train_report and train_report['warnings']:
        print("\nTrain data warnings:")
        for warning in train_report['warnings']:
            print(f"  - {warning}")
    
    if 'warnings' in test_report and test_report['warnings']:
        print("\nTest data warnings:")
        for warning in test_report['warnings']:
            print(f"  - {warning}")

if __name__ == "__main__":
    main()