import pandas as pd
from openbb import obb
from typing import Optional
import os

class DataLoader:
    """Класс для загрузки финансовых данных через OpenBB"""
    
    def __init__(self):
        self.obb = obb
    
    def load_stock_data(self, symbol: str, start_date: str = "2020-01-01", 
                       end_date: Optional[str] = None):
        """Загрузка данных по акциям"""
        try:
            data = self.obb.equity.price.historical(
                symbol, 
                start_date=start_date, 
                end_date=end_date
            )
            return data.to_df()
        except Exception as e:
            print(f"Ошибка при загрузке данных для {symbol}: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """Сохранение данных в CSV"""
        if not os.path.exists('data/raw'):
            os.makedirs('data/raw')
        filepath = f"data/raw/{filename}"
        df.to_csv(filepath)
        print(f"Данные сохранены в {filepath}")

# Пример использования
if __name__ == "__main__":
    loader = DataLoader()
    aapl_data = loader.load_stock_data("AAPL")
    if aapl_data is not None:
        loader.save_data(aapl_data, "aapl_data.csv")