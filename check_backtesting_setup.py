#!/usr/bin/env python3
"""
Проверка готовности системы бэктестинга
"""

import sys
import os

def check_imports():
    """Проверка импортов"""
    print("1. Проверка импортов...")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        import seaborn
        print("   ✓ Базовые библиотеки: OK")
    except ImportError as e:
        print(f"   ❌ Ошибка импорта базовых библиотек: {e}")
        return False
    
    try:
        import tqdm
        import sklearn
        print("   ✓ tqdm, scikit-learn: OK")
    except ImportError as e:
        print(f"   ❌ Ошибка импорта tqdm/sklearn: {e}")
        print("   Установите: pip install tqdm scikit-learn")
        return False
    
    try:
        from src.backtesting import (
            Portfolio, ExecutionEngine, PerformanceAnalyzer,
            Backtester, MLBacktester, WalkForwardAnalyzer
        )
        print("   ✓ Backtesting модули: OK")
    except ImportError as e:
        print(f"   ❌ Ошибка импорта backtesting: {e}")
        return False
    
    return True

def check_data():
    """Проверка наличия данных"""
    print("\n2. Проверка данных...")
    
    data_path = 'data/processed/aapl_with_features_5y.parquet'
    
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_parquet(data_path)
        print(f"   ✓ Файл найден: {data_path}")
        print(f"   ✓ Строк: {len(df)}, Колонок: {len(df.columns)}")
        
        # Проверяем необходимые фичи
        required = ['close', 'sma_50', 'sma_200', 'rsi_14']
        missing = [f for f in required if f not in df.columns]
        
        if missing:
            print(f"   ⚠️  Отсутствуют фичи: {missing}")
            print("   Запустите ноутбук 02_feature_engineering.ipynb")
            return False
        else:
            print(f"   ✓ Все необходимые фичи присутствуют")
        
        return True
    else:
        print(f"   ❌ Файл не найден: {data_path}")
        print("   Запустите ноутбук 02_feature_engineering.ipynb")
        return False

def check_tests():
    """Проверка тестов"""
    print("\n3. Проверка тестов...")
    
    if os.path.exists('test_backtesting.py'):
        print("   ✓ Файл test_backtesting.py найден")
        print("   Для запуска тестов: python test_backtesting.py")
        return True
    else:
        print("   ⚠️  Файл test_backtesting.py не найден")
        return False

def main():
    print("=" * 70)
    print("  ПРОВЕРКА ГОТОВНОСТИ BACKTESTING СИСТЕМЫ")
    print("=" * 70 + "\n")
    
    results = []
    
    results.append(check_imports())
    results.append(check_data())
    results.append(check_tests())
    
    print("\n" + "=" * 70)
    if all(results):
        print("  ✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
        print("  Можно запускать notebooks/exploratory/03_backtesting_demo.ipynb")
    else:
        print("  ❌ ЕСТЬ ПРОБЛЕМЫ")
        print("  Исправьте ошибки выше перед запуском")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    main()
