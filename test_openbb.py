from openbb import obb
import pandas as pd
import os
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

def test_openbb():
    """Тестирование базового функционала OpenBB"""
    print("Testing OpenBB Platform...")
    
    # Получение данных акций
    try:
        # Данные Apple
        aapl_data = obb.equity.price.historical("AAPL")
        print(f"AAPL data shape: {aapl_data.to_df().shape}")
        print("OpenBB installation successful!")
        
        # Сохраняем пример данных
        aapl_data.to_df().to_csv("data/raw/aapl_sample.csv")
        print("Sample data saved to data/raw/aapl_sample.csv")
        
    except Exception as e:
        print(f"Error testing OpenBB: {e}")

if __name__ == "__main__":
    test_openbb()