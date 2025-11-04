#!/usr/bin/env python3
"""
Тестирование всего пайплайна данных
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_manager import DataManager
from data.data_validator import DataValidator

def test_data_pipeline():
    """Тестирование всего пайплайна"""
    print("🧪 Тестирование пайплайна данных...")
    
    # Инициализация
    print("1. Инициализация DataManager...")
    data_manager = DataManager()
    
    # Тест загрузки отдельных активов
    print("\n2. Тест загрузки отдельных активов...")
    
    # Акции
    try:
        aapl_data = data_manager.get_stock_data('AAPL', '1d', '6mo')
        print(f"✓ AAPL данные загружены: {len(aapl_data)} записей")
    except Exception as e:
        print(f"✗ Ошибка загрузки AAPL: {e}")
    
    # ETF
    try:
        spy_data = data_manager.get_etf_data('SPY', '1d', '6mo')
        print(f"✓ SPY данные загружены: {len(spy_data)} записей")
    except Exception as e:
        print(f"✗ Ошибка загрузки SPY: {e}")
    
    # Индекс
    try:
        spx_data = data_manager.get_index_data('^GSPC', '1d', '6mo')
        print(f"✓ SPX данные загружены: {len(spx_data)} записей")
    except Exception as e:
        print(f"✗ Ошибка загрузки SPX: {e}")
    
    # Тест валидации
    print("\n3. Тест валидации данных...")
    validator = DataValidator(data_manager.config)
    
    test_symbols = ['AAPL', 'SPY', '^GSPC']
    for symbol in test_symbols:
        try:
            # Получаем данные через manager (будет использовать кэш)
            data = data_manager.get_stock_data(symbol, '1d', '6mo') if symbol != '^GSPC' else \
                   data_manager.get_index_data(symbol, '1d', '6mo')
            
            validation_result = validator.validate_dataset(data, symbol)
            
            status = "✓" if validation_result['is_valid'] else "✗"
            print(f"{status} {symbol}: {validation_result['passed_tests']}/{validation_result['total_tests']} тестов пройдено")
            
            if validation_result['warnings']:
                for warning in validation_result['warnings']:
                    print(f"  ⚠ Предупреждение: {warning}")
                    
            if validation_result['errors']:
                for error in validation_result['errors']:
                    print(f"  ❌ Ошибка: {error}")
                    
        except Exception as e:
            print(f"✗ Ошибка валидации {symbol}: {e}")
    
    # Тест информации о кэше
    print("\n4. Информация о кэше...")
    cache_info = data_manager.get_cache_info()
    print(f"Файлов в кэше: {cache_info['total_files']}")
    print(f"Размер кэша: {cache_info['total_size_gb']:.2f} GB")
    print("Последние файлы:")
    for file_info in cache_info['recent_files'][:3]:
        print(f"  - {file_info['name']} ({file_info['size_mb']:.1f} MB)")
    
    print("\n🎉 Тестирование завершено!")

if __name__ == "__main__":
    test_data_pipeline()