import pandas as pd
import numpy as np
import yfinance as yf
from openbb import obb
import os
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Класс для управления загрузкой и хранением финансовых данных"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_directories()
        self.obb = obb
        self._initialize_openbb()
        
    def _load_config(self, config_path: str) -> dict:
        """Загрузка конфигурации"""
        try:
            import yaml
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            raise
    
    def _setup_directories(self):
        """Создание необходимых директорий"""
        dirs = [
            self.config['data_settings']['cache_dir'],
            self.config['data_settings']['raw_dir'], 
            self.config['data_settings']['backup_dir']
        ]
        
        for directory in dirs:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_openbb(self):
        """Инициализация OpenBB с API ключами"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            # Настройка API ключей
            api_keys = {
                'polygon': os.getenv('POLYGON_API_KEY'),
                'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY')
            }
            
            for provider, key in api_keys.items():
                if key:
                    self.obb.account.login(provider=provider, token=key)
                    
            logger.info("OpenBB успешно инициализирован")
            
        except Exception as e:
            logger.warning(f"Не удалось инициализировать все API ключи: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,))
    )
    def _fetch_with_retry(self, symbol: str, timeframe: str, **kwargs):
        """Загрузка данных с повторными попытками"""
        try:
            # Пробуем разные методы загрузки
            data = None
            
            # Метод 1: Через OpenBB Equity
            if not data:
                try:
                    data = self.obb.equity.price.historical(
                        symbol, 
                        interval=timeframe,
                        start_date=kwargs.get('start_date'),
                        end_date=kwargs.get('end_date')
                    )
                    if hasattr(data, 'to_df'):
                        data = data.to_df()
                except Exception as e:
                    logger.debug(f"OpenBB equity failed for {symbol}: {e}")
            
            # Метод 2: Через yfinance как fallback
            if data is None or data.empty:
                try:
                    yf_symbol = symbol.replace('^', '')
                    data = yf.download(
                        yf_symbol, 
                        period=kwargs.get('period', '2y'),
                        interval=timeframe
                    )
                    data.reset_index(inplace=True)
                except Exception as e:
                    logger.debug(f"YFinance failed for {symbol}: {e}")
            
            if data is None or data.empty:
                raise ValueError(f"Не удалось загрузить данные для {symbol}")
                
            return data
            
        except Exception as e:
            logger.error(f"Ошибка загрузки данных для {symbol}: {e}")
            raise
    
    def get_stock_data(self, symbol: str, timeframe: str = "1d", 
                      period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
        """Загрузка данных по акциям"""
        return self._get_asset_data(symbol, 'stocks', timeframe, period, use_cache)
    
    def get_etf_data(self, symbol: str, timeframe: str = "1d",
                    period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
        """Загрузка данных по ETF"""
        return self._get_asset_data(symbol, 'etfs', timeframe, period, use_cache)
    
    def get_index_data(self, symbol: str, timeframe: str = "1d",
                      period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
        """Загрузка данных по индексам"""
        return self._get_asset_data(symbol, 'indices', timeframe, period, use_cache)
    
    def _get_asset_data(self, symbol: str, asset_type: str, timeframe: str,
                       period: str, use_cache: bool) -> pd.DataFrame:
        """Базовый метод загрузки данных по активам"""
        
        cache_file = self._get_cache_file_path(symbol, timeframe, asset_type)
        
        # Проверка кэша
        if use_cache and self._is_cache_valid(cache_file):
            logger.info(f"Загрузка из кэша: {symbol}")
            return self._load_from_cache(cache_file)
        
        # Загрузка новых данных
        logger.info(f"Загрузка новых данных: {symbol}")
        data = self._fetch_with_retry(
            symbol=symbol,
            timeframe=timeframe,
            period=period
        )
        
        # Обработка и сохранение данных
        processed_data = self._process_data(data, symbol, asset_type)
        self._save_to_cache(processed_data, cache_file)
        
        return processed_data
    
    def _get_cache_file_path(self, symbol: str, timeframe: str, asset_type: str) -> Path:
        """Генерация пути к файлу кэша"""
        cache_dir = Path(self.config['data_settings']['cache_dir'])
        file_format = self.config['data_settings']['file_format']
        filename = f"{symbol}_{timeframe}_{asset_type}.{file_format}"
        return cache_dir / filename
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Проверка актуальности кэша"""
        if not cache_file.exists():
            return False
        
        # Проверяем время последнего обновления (24 часа по умолчанию)
        cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        update_interval = timedelta(
            hours=int(os.getenv('DATA_UPDATE_INTERVAL_HOURS', 24))
        )
        
        return datetime.now() - cache_time < update_interval
    
    def _load_from_cache(self, cache_file: Path) -> pd.DataFrame:
        """Загрузка данных из кэша"""
        try:
            if cache_file.suffix == '.parquet':
                return pd.read_parquet(cache_file)
            else:
                return pd.read_csv(cache_file, parse_dates=['date'], index_col='date')
        except Exception as e:
            logger.warning(f"Ошибка загрузки из кэша {cache_file}: {e}")
            raise
    
    def _save_to_cache(self, data: pd.DataFrame, cache_file: Path):
        """Сохранение данных в кэш"""
        try:
            if cache_file.suffix == '.parquet':
                data.to_parquet(cache_file, index=True)
            else:
                data.to_csv(cache_file, index=True)
            logger.info(f"Данные сохранены в кэш: {cache_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения в кэш: {e}")
            raise
    
    def _process_data(self, data: pd.DataFrame, symbol: str, asset_type: str) -> pd.DataFrame:
        """Обработка и очистка данных"""
        # Копируем данные
        processed = data.copy()
        
        # Стандартизация названий колонок
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close',
            'Date': 'date', 'datetime': 'date'
        }
        
        processed.rename(columns=column_mapping, inplace=True)
        
        # Убеждаемся, что есть колонка даты
        if 'date' not in processed.columns and processed.index.name == 'date':
            processed.reset_index(inplace=True)
        
        # Добавляем метаданные
        processed['symbol'] = symbol
        processed['asset_type'] = asset_type
        processed['data_loaded_at'] = datetime.now()
        
        # Сортировка по дате
        if 'date' in processed.columns:
            processed['date'] = pd.to_datetime(processed['date'])
            processed.sort_values('date', inplace=True)
            processed.set_index('date', inplace=True)
        
        # Обработка пропущенных значений
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in processed.columns:
                # Forward fill для цен, backward fill для volume
                if col == 'volume':
                    processed[col].fillna(method='bfill', inplace=True)
                else:
                    processed[col].fillna(method='ffill', inplace=True)
        
        return processed
    
    def batch_download(self, asset_types: List[str] = None, 
                      timeframes: List[str] = None) -> Dict:
        """Пакетная загрузка данных"""
        if asset_types is None:
            asset_types = ['stocks', 'etfs', 'indices']
        
        if timeframes is None:
            timeframes = ['1d']
        
        results = {}
        
        for asset_type in asset_types:
            results[asset_type] = {}
            symbols = self.config['data_settings']['assets'].get(asset_type, [])
            
            for symbol in symbols:
                results[asset_type][symbol] = {}
                
                for timeframe in timeframes:
                    try:
                        data = self._get_asset_data(
                            symbol, asset_type, timeframe, "2y", True
                        )
                        results[asset_type][symbol][timeframe] = data
                        logger.info(f"Успешно: {asset_type} {symbol} {timeframe}")
                        
                        # Задержка для избежания rate limits
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.error(f"Ошибка: {asset_type} {symbol} {timeframe}: {e}")
                        results[asset_type][symbol][timeframe] = None
        
        return results
    
    def get_cache_info(self) -> Dict:
        """Информация о кэшированных данных"""
        cache_dir = Path(self.config['data_settings']['cache_dir'])
        cache_files = list(cache_dir.glob('*.*'))
        
        info = {
            'total_files': len(cache_files),
            'total_size_gb': sum(f.stat().st_size for f in cache_files) / (1024**3),
            'file_types': {},
            'recent_files': []
        }
        
        for file in cache_files[-10:]:  # Последние 10 файлов
            info['recent_files'].append({
                'name': file.name,
                'size_mb': file.stat().st_size / (1024**2),
                'modified': datetime.fromtimestamp(file.stat().st_mtime)
            })
        
        return info