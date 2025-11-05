"""
Тест полного пайплайна расчёта 50+ фич.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.feature_engine import FeatureEngine
from src.features.feature_registry import register_all_features, get_feature_count


@pytest.fixture
def sample_ohlcv():
    """Синтетические OHLCV данные для полного теста."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    close_prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.015))
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(500) * 0.003),
        'high': close_prices * (1 + np.abs(np.random.randn(500) * 0.01)),
        'low': close_prices * (1 - np.abs(np.random.randn(500) * 0.01)),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 500),
        'symbol': 'TEST'
    }, index=dates)
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def test_feature_engine_initialization():
    """FeatureEngine должен инициализироваться."""
    engine = FeatureEngine()
    assert engine.version == "1.0.0"
    assert len(engine.registry) == 0


def test_feature_registration():
    """Фичи должны регистрироваться в engine."""
    engine = FeatureEngine()
    
    def dummy_feature(df):
        return df['close'].rolling(10).mean()
    
    engine.register_feature(
        'test_sma',
        dummy_feature,
        group='test',
        description='Test SMA'
    )
    
    assert 'test_sma' in engine.registry
    assert engine.registry['test_sma']['group'] == 'test'


def test_register_all_features_count():
    """Должно регистрироваться 50+ фич."""
    summary = get_feature_count()
    
    assert summary['total_features'] >= 50, f"Ожидается минимум 50 фич, найдено {summary['total_features']}"
    
    # Проверяем наличие всех групп
    assert 'technical' in summary['groups']
    assert 'price_based' in summary['groups']
    assert 'rolling' in summary['groups']


def test_compute_single_feature(sample_ohlcv):
    """Расчёт одной фичи должен работать."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    result = engine.compute_feature('rsi_14', sample_ohlcv, use_cache=False)
    
    assert 'rsi_14' in result.columns
    assert len(result) == len(sample_ohlcv)


def test_compute_technical_group(sample_ohlcv):
    """Расчёт группы технических фич."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    result = engine.compute_all(sample_ohlcv, groups=['technical'], use_cache=False)
    
    # Должны появиться новые колонки
    new_cols = set(result.columns) - set(sample_ohlcv.columns)
    assert len(new_cols) > 0
    
    # Проверяем наличие нескольких ключевых индикаторов
    assert 'rsi_14' in result.columns or any('rsi' in col for col in result.columns)


def test_compute_all_features(sample_ohlcv):
    """Расчёт всех 50+ фич."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    result = engine.compute_all(sample_ohlcv, use_cache=False)
    
    # Итоговых колонок должно быть намного больше исходных
    assert len(result.columns) > len(sample_ohlcv.columns) + 40
    
    # Проверяем, что индекс сохранился
    assert result.index.equals(sample_ohlcv.index)


def test_feature_caching(sample_ohlcv, tmp_path):
    """Тест кеширования фич."""
    engine = FeatureEngine(cache_dir=str(tmp_path / 'features_cache'))
    register_all_features(engine)
    
    # Первый расчёт (без кеша)
    result1 = engine.compute_feature('rsi_14', sample_ohlcv, use_cache=True)
    
    # Второй расчёт (из кеша)
    result2 = engine.compute_feature('rsi_14', sample_ohlcv, use_cache=True)
    
    # Результаты должны совпадать
    pd.testing.assert_frame_equal(result1, result2)


def test_feature_metadata():
    """Получение метаданных фич."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    meta = engine.get_feature_metadata('rsi_14')
    
    assert 'group' in meta
    assert 'description' in meta
    assert 'dependencies' in meta
    assert 'params' in meta


def test_feature_versioning():
    """Фичи должны иметь hash для версионирования."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    meta = engine.get_feature_metadata('rsi_14')
    
    assert 'hash' in meta
    assert len(meta['hash']) == 8  # MD5 truncated to 8 chars


def test_computation_log():
    """Engine должен вести лог вычислений."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    sample = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [99, 100, 101],
        'close': [101, 102, 103],
        'volume': [1000, 1000, 1000],
        'symbol': ['TEST'] * 3
    }, index=pd.date_range('2023-01-01', periods=3))
    
    engine.compute_feature('rsi_14', sample, use_cache=False)
    
    assert len(engine.computation_log) > 0
    assert 'feature' in engine.computation_log[0]
    assert 'elapsed_sec' in engine.computation_log[0]


def test_missing_dependencies():
    """Расчёт фичи без необходимых колонок должен вызвать ошибку."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    incomplete_df = pd.DataFrame({
        'close': [100, 101, 102]
    })
    
    with pytest.raises(ValueError, match="отсутствуют колонки"):
        engine.compute_feature('bollinger_bands', incomplete_df, use_cache=False)


def test_feature_groups():
    """Проверка группировки фич."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    assert 'technical' in engine.feature_groups
    assert 'price_based' in engine.feature_groups
    assert 'rolling' in engine.feature_groups
    
    # В каждой группе должны быть фичи
    assert len(engine.feature_groups['technical']) > 0
    assert len(engine.feature_groups['price_based']) > 0


def test_clear_cache(sample_ohlcv, tmp_path):
    """Очистка кеша должна работать."""
    cache_dir = tmp_path / 'features_cache'
    engine = FeatureEngine(cache_dir=str(cache_dir))
    register_all_features(engine)
    
    # Создаём кеш
    engine.compute_feature('rsi_14', sample_ohlcv, use_cache=True)
    
    # Проверяем что кеш создан
    cache_files = list(cache_dir.glob('*.parquet'))
    assert len(cache_files) > 0
    
    # Очищаем
    engine.clear_cache()
    
    # Проверяем что кеш пуст
    cache_files = list(cache_dir.glob('*.parquet'))
    assert len(cache_files) == 0


def test_summary():
    """get_summary должен возвращать статистику."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    summary = engine.get_summary()
    
    assert 'engine_version' in summary
    assert 'total_features' in summary
    assert 'groups' in summary
    assert summary['total_features'] >= 50


def test_multicolumn_features(sample_ohlcv):
    """Фичи возвращающие несколько колонок должны корректно обрабатываться."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    # MACD возвращает DataFrame с 3 колонками
    result = engine.compute_feature('macd', sample_ohlcv, use_cache=False)
    
    assert 'macd' in result.columns
    assert 'macd_signal' in result.columns
    assert 'macd_hist' in result.columns


def test_feature_computation_robustness():
    """Фичи должны корректно обрабатывать граничные случаи."""
    engine = FeatureEngine()
    register_all_features(engine)
    
    # Короткий ряд данных
    short_df = pd.DataFrame({
        'open': [100, 101],
        'high': [102, 103],
        'low': [99, 100],
        'close': [101, 102],
        'volume': [1000, 1000],
        'symbol': ['TEST'] * 2
    }, index=pd.date_range('2023-01-01', periods=2))
    
    # Не должно падать, хотя многие фичи будут NaN
    result = engine.compute_feature('rsi_14', short_df, use_cache=False)
    assert len(result) == len(short_df)
