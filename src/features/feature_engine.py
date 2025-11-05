"""
Ядро системы расчёта фичей с версионированием, кешированием и метаданными.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import logging
import json
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Движок для расчёта, кеширования и версионирования фичей.
    
    Возможности:
    - Регистрация фич-функций с метаданными
    - Версионирование (автоматическое хеширование кода + параметров)
    - Кеширование результатов
    - Пакетный расчёт всех фичей для символа
    - Метаданные и отчёты
    """
    
    def __init__(self, cache_dir: str = "data/processed/features", version: str = "1.0.0"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.version = version
        self.registry: Dict[str, Dict[str, Any]] = {}
        self.feature_groups: Dict[str, List[str]] = {}
        
        # Метаданные для аудита
        self.computation_log = []
        
        logger.info(f"FeatureEngine v{version} инициализирован")
    
    def register_feature(
        self,
        name: str,
        func: Callable,
        group: str = "general",
        description: str = "",
        dependencies: List[str] = None,
        params: Dict = None
    ):
        """
        Регистрация фичи в реестре.
        
        Args:
            name: уникальное имя фичи
            func: функция расчёта (принимает df, возвращает Series/DataFrame)
            group: группа (technical, price_based, macro, rolling, etc.)
            description: человеко-читаемое описание
            dependencies: список обязательных колонок в исходном df
            params: параметры для расчёта фичи
        """
        if dependencies is None:
            dependencies = ['close']
        if params is None:
            params = {}
        
        # Генерируем hash версии на основе кода функции и параметров
        feature_hash = self._compute_feature_hash(func, params)
        
        self.registry[name] = {
            'func': func,
            'group': group,
            'description': description,
            'dependencies': dependencies,
            'params': params,
            'hash': feature_hash,
            'registered_at': datetime.now().isoformat()
        }
        
        # Добавляем в группу
        if group not in self.feature_groups:
            self.feature_groups[group] = []
        self.feature_groups[group].append(name)
        
        logger.debug(f"Зарегистрирована фича '{name}' в группе '{group}'")
    
    def _compute_feature_hash(self, func: Callable, params: Dict) -> str:
        """Хеш функции + параметров для версионирования"""
        import inspect
        code_str = inspect.getsource(func)
        params_str = json.dumps(params, sort_keys=True)
        combined = f"{code_str}:{params_str}"
        return hashlib.md5(combined.encode()).hexdigest()[:8]
    
    def compute_feature(
        self,
        name: str,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Вычислить одну фичу для датафрейма.
        
        Args:
            name: имя зарегистрированной фичи
            df: исходные данные (OHLCV)
            use_cache: использовать кеш
            
        Returns:
            DataFrame с добавленной колонкой фичи
        """
        if name not in self.registry:
            raise ValueError(f"Фича '{name}' не зарегистрирована")
        
        feature_meta = self.registry[name]
        
        # Проверка зависимостей
        missing_deps = [dep for dep in feature_meta['dependencies'] if dep not in df.columns]
        if missing_deps:
            raise ValueError(f"Для фичи '{name}' отсутствуют колонки: {missing_deps}")
        
        # Проверка кеша
        if use_cache:
            cached = self._load_from_cache(name, df)
            if cached is not None:
                logger.debug(f"Загружена из кеша: {name}")
                return cached
        
        # Расчёт
        start_time = datetime.now()
        try:
            result = feature_meta['func'](df, **feature_meta['params'])
            
            # Приводим к DataFrame если вернулась Series
            if isinstance(result, pd.Series):
                result_df = df.copy()
                result_df[name] = result
            else:
                result_df = result
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # Логирование
            self.computation_log.append({
                'feature': name,
                'timestamp': datetime.now().isoformat(),
                'elapsed_sec': elapsed,
                'rows': len(df),
                'hash': feature_meta['hash']
            })
            
            logger.info(f"Фича '{name}' вычислена за {elapsed:.3f}s")
            
            # Сохранение в кеш
            if use_cache:
                self._save_to_cache(name, result_df, feature_meta)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Ошибка расчёта фичи '{name}': {e}")
            raise
    
    def compute_all(
        self,
        df: pd.DataFrame,
        groups: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Вычислить все зарегистрированные фичи (или выбранные группы).
        
        Args:
            df: исходные OHLCV данные
            groups: список групп для расчёта (None = все)
            use_cache: использовать кеш
            
        Returns:
            DataFrame со всеми фичами
        """
        result_df = df.copy()
        
        # Определяем список фич для расчёта
        if groups is None:
            features_to_compute = list(self.registry.keys())
        else:
            features_to_compute = []
            for group in groups:
                features_to_compute.extend(self.feature_groups.get(group, []))
        
        logger.info(f"Расчёт {len(features_to_compute)} фичей...")
        
        for feature_name in features_to_compute:
            try:
                # Вычисляем фичу; она вернёт df с добавленной колонкой
                temp_df = self.compute_feature(feature_name, result_df, use_cache)
                
                # Добавляем новые колонки в result_df
                new_cols = [col for col in temp_df.columns if col not in result_df.columns]
                for col in new_cols:
                    result_df[col] = temp_df[col]
                    
            except Exception as e:
                logger.warning(f"Пропуск фичи '{feature_name}': {e}")
        
        logger.info(f"Расчёт завершён. Итого колонок: {len(result_df.columns)}")
        return result_df
    
    def _get_cache_path(self, feature_name: str, df: pd.DataFrame) -> Path:
        """Путь к файлу кеша для фичи"""
        # Хеш на основе символа и диапазона дат
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'unknown'
        date_range = f"{df.index.min().date()}_{df.index.max().date()}"
        feature_meta = self.registry[feature_name]
        cache_filename = f"{symbol}_{feature_name}_{feature_meta['hash']}_{date_range}.parquet"
        return self.cache_dir / cache_filename
    
    def _load_from_cache(self, feature_name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Загрузка фичи из кеша"""
        cache_path = self._get_cache_path(feature_name, df)
        if cache_path.exists():
            try:
                cached_df = pd.read_parquet(cache_path)
                return cached_df
            except Exception as e:
                logger.warning(f"Ошибка чтения кеша {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, feature_name: str, result_df: pd.DataFrame, feature_meta: Dict):
        """Сохранение фичи в кеш"""
        try:
            cache_path = self._get_cache_path(feature_name, result_df)
            result_df.to_parquet(cache_path)
            logger.debug(f"Кеш сохранён: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить кеш для '{feature_name}': {e}")
    
    def get_feature_metadata(self, name: Optional[str] = None) -> Dict:
        """
        Получить метаданные фичи (или всех фичей).
        
        Args:
            name: имя фичи (None = все)
            
        Returns:
            Словарь с метаданными
        """
        if name:
            if name not in self.registry:
                raise ValueError(f"Фича '{name}' не найдена")
            meta = self.registry[name].copy()
            meta.pop('func')  # Не показываем саму функцию
            return meta
        else:
            all_meta = {}
            for feat_name, feat_data in self.registry.items():
                meta_copy = feat_data.copy()
                meta_copy.pop('func')
                all_meta[feat_name] = meta_copy
            return all_meta
    
    def get_summary(self) -> Dict:
        """Сводка по всем зарегистрированным фичам"""
        summary = {
            'engine_version': self.version,
            'total_features': len(self.registry),
            'groups': {group: len(features) for group, features in self.feature_groups.items()},
            'features_by_group': self.feature_groups,
            'last_computation_count': len(self.computation_log)
        }
        return summary
    
    def clear_cache(self, feature_name: Optional[str] = None):
        """
        Очистка кеша (одной фичи или всех).
        
        Args:
            feature_name: имя фичи (None = все)
        """
        if feature_name:
            pattern = f"*_{feature_name}_*.parquet"
        else:
            pattern = "*.parquet"
        
        removed = 0
        for cache_file in self.cache_dir.glob(pattern):
            cache_file.unlink()
            removed += 1
        
        logger.info(f"Удалено {removed} кеш-файлов")
