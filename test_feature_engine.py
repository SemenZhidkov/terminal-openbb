#!/usr/bin/env python3
"""
Быстрый тест работоспособности Feature Engine.
Запустите перед открытием ноутбука для проверки всех зависимостей.
"""

import sys
import os

# Добавляем корень проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Тест импортов"""
    print("1. Проверка импортов...")
    
    try:
        from src.data.data_manager import DataManager
        from src.features.feature_engine import FeatureEngine
        from src.features.feature_registry import register_all_features, get_feature_count
        print("   ✓ Все модули импортированы")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка импорта: {e}")
        return False


def test_feature_registry():
    """Тест регистрации фич"""
    print("\n2. Проверка Feature Registry...")
    
    try:
        from src.features.feature_engine import FeatureEngine
        from src.features.feature_registry import register_all_features, get_feature_count
        
        engine = FeatureEngine()
        register_all_features(engine)
        
        summary = engine.get_summary()
        print(f"   ✓ Зарегистрировано {summary['total_features']} фич")
        print(f"   ✓ Группы: {list(summary['groups'].keys())}")
        
        if summary['total_features'] < 50:
            print(f"   ⚠ Ожидалось минимум 50 фич, найдено {summary['total_features']}")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Тест загрузки данных"""
    print("\n3. Проверка загрузки данных...")
    
    try:
        from src.data.data_manager import DataManager
        
        dm = DataManager()
        print("   ✓ DataManager инициализирован")
        
        # Пробуем загрузить небольшой набор данных
        print("   → Загрузка тестовых данных AAPL (3 месяца)...")
        df = dm.get_stock_data('AAPL', timeframe='1d', period='3mo', use_cache=True)
        
        print(f"   ✓ Загружено {len(df)} строк")
        print(f"   ✓ Колонки: {list(df.columns)}")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"   ⚠ Отсутствуют колонки: {missing}")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка загрузки: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_computation():
    """Тест расчёта фич"""
    print("\n4. Проверка расчёта фич...")
    
    try:
        from src.data.data_manager import DataManager
        from src.features.feature_engine import FeatureEngine
        from src.features.feature_registry import register_all_features
        import pandas as pd
        import numpy as np
        
        # Создаём тестовые данные
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': 100 + np.random.randn(100),
            'high': 102 + np.random.randn(100),
            'low': 98 + np.random.randn(100),
            'close': 100 + np.random.randn(100),
            'volume': np.random.randint(1000000, 5000000, 100),
            'symbol': ['TEST'] * 100
        }, index=dates)
        
        # Обеспечиваем корректность OHLC
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        engine = FeatureEngine()
        register_all_features(engine)
        
        print("   → Расчёт одной фичи (RSI)...")
        result = engine.compute_feature('rsi_14', df, use_cache=False)
        print(f"   ✓ RSI рассчитан, добавлено колонок: {len(result.columns) - len(df.columns)}")
        
        print("   → Расчёт технических индикаторов...")
        result_tech = engine.compute_all(df, groups=['technical'], use_cache=False)
        tech_features = len(result_tech.columns) - len(df.columns)
        print(f"   ✓ Технические индикаторы рассчитаны: {tech_features} фич")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка расчёта: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ FEATURE ENGINE")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_feature_registry,
        test_data_loading,
        test_feature_computation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Пройдено тестов: {passed}/{total}")
    
    if all(results):
        print("\n✅ Все тесты пройдены! Можно запускать ноутбук.")
        print("\nЗапустите:")
        print("  jupyter lab notebooks/exploratory/02_feature_engineering_demo.ipynb")
        return 0
    else:
        print("\n❌ Некоторые тесты не прошли. Проверьте ошибки выше.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
