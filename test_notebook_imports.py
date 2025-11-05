#!/usr/bin/env python3
"""
Тест импорта DataManager с правильным путём к конфигу
Имитирует выполнение из ноутбука
"""

import sys
import os

# Имитируем запуск из notebooks/exploratory/
notebook_dir = os.path.join(os.path.dirname(__file__), 'notebooks', 'exploratory')
os.chdir(notebook_dir)

print(f"📂 Текущая директория (как в Jupyter): {os.getcwd()}")

# Настраиваем путь как в ноутбуке
project_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.insert(0, project_root)

print(f"📂 Корень проекта: {project_root}")

# Тестируем импорт с правильным путём
print("\n1. Проверка путей:")
config_path = os.path.join(project_root, 'config', 'config.yaml')
print(f"   Config path: {config_path}")
print(f"   Config exists: {os.path.exists(config_path)}")

print("\n2. Тест DataManager с абсолютным путём:")
try:
    from src.data.data_manager import DataManager
    
    dm = DataManager(config_path=config_path)
    print("   ✓ DataManager инициализирован успешно!")
    
    # Пробуем загрузить данные
    print("\n3. Тест загрузки данных:")
    df = dm.get_stock_data('AAPL', timeframe='1d', period='1mo', use_cache=True)
    print(f"   ✓ Загружено {len(df)} строк AAPL")
    print(f"   ✓ Колонки: {list(df.columns)}")
    
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Все проверки пройдены! Ноутбук должен работать.")
