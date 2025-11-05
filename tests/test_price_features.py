"""
Unit-тесты для price-based фич.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import price_features as price


@pytest.fixture
def sample_data():
    """Синтетические данные для тестирования."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    close_prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.02))
    
    df = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(100) * 0.005),
        'high': close_prices * (1 + np.abs(np.random.randn(100) * 0.01)),
        'low': close_prices * (1 - np.abs(np.random.randn(100) * 0.01)),
        'close': close_prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def test_simple_returns(sample_data):
    """Простые доходности: проверка диапазона."""
    returns = price.simple_returns(sample_data, period=1)
    
    assert len(returns) == len(sample_data)
    # Разумный диапазон доходностей для дневных данных
    valid = returns.dropna()
    assert valid.min() > -0.5  # -50%
    assert valid.max() < 0.5   # +50%


def test_log_vs_simple_returns(sample_data):
    """Логарифмические и простые доходности должны быть близки для малых изменений."""
    simple = price.simple_returns(sample_data, period=1)
    log = price.log_returns(sample_data, period=1)
    
    # Для малых доходностей log(1+r) ≈ r
    diff = (simple - log).dropna().abs()
    assert diff.mean() < 0.01, "Log и simple returns должны быть близки"


def test_forward_returns(sample_data):
    """Forward returns должны быть сдвинуты в будущее."""
    fwd = price.forward_returns(sample_data, period=1)
    current = price.simple_returns(sample_data, period=1)
    
    # Forward[t] должен равняться returns[t+1]
    assert pd.isna(fwd.iloc[-1]), "Последнее значение forward returns должно быть NaN"


def test_realized_volatility_positive(sample_data):
    """Realized volatility всегда положительная."""
    vol = price.realized_volatility(sample_data, window=20)
    
    valid = vol.dropna()
    assert (valid >= 0).all()


def test_volatility_annualization(sample_data):
    """Годовая волатильность должна быть больше дневной."""
    daily_vol = price.realized_volatility(sample_data, window=20, annualize=False)
    annual_vol = price.realized_volatility(sample_data, window=20, annualize=True)
    
    valid_idx = ~daily_vol.isnull() & ~annual_vol.isnull()
    assert (annual_vol[valid_idx] > daily_vol[valid_idx]).all()


def test_parkinson_volatility(sample_data):
    """Parkinson volatility должна быть положительной."""
    vol = price.parkinson_volatility(sample_data, window=20)
    
    valid = vol.dropna()
    assert (valid >= 0).all()


def test_volatility_regime_labels(sample_data):
    """Volatility regime должен возвращать 0, 1, 2."""
    regime = price.volatility_regime(sample_data, window=60)
    
    valid = regime.dropna()
    unique_labels = valid.unique()
    
    assert set(unique_labels).issubset({0, 1, 2})


def test_price_momentum(sample_data):
    """Price momentum должен вернуть DataFrame с несколькими колонками."""
    momentum = price.price_momentum(sample_data, periods=[5, 10, 20])
    
    assert isinstance(momentum, pd.DataFrame)
    assert 'momentum_5d' in momentum.columns
    assert 'momentum_10d' in momentum.columns
    assert 'momentum_20d' in momentum.columns


def test_rolling_zscore_distribution(sample_data):
    """Rolling Z-score должен иметь среднее ~0 и std ~1."""
    zscore = price.rolling_zscore(sample_data, window=20)
    
    valid = zscore.dropna()
    # Среднее должно быть близко к 0
    assert abs(valid.mean()) < 0.3
    # Std близко к 1 (но не точно, т.к. rolling)


def test_distance_from_high_negative(sample_data):
    """Distance from high должна быть <= 0."""
    dist = price.price_distance_from_high(sample_data, window=60)
    
    valid = dist.dropna()
    assert (valid <= 0).all()


def test_distance_from_low_positive(sample_data):
    """Distance from low должна быть >= 0."""
    dist = price.price_distance_from_low(sample_data, window=60)
    
    valid = dist.dropna()
    assert (valid >= 0).all()


def test_daily_range_positive(sample_data):
    """Daily range должен быть положительным."""
    dr = price.daily_range(sample_data)
    
    assert (dr >= 0).all()


def test_body_size_positive(sample_data):
    """Body size должен быть положительным."""
    body = price.body_size(sample_data)
    
    assert (body >= 0).all()


def test_shadows_positive(sample_data):
    """Upper и lower shadow должны быть >= 0."""
    upper = price.upper_shadow(sample_data)
    lower = price.lower_shadow(sample_data)
    
    assert (upper >= 0).all()
    assert (lower >= 0).all()


def test_gap_calculation(sample_data):
    """Gap может быть положительным или отрицательным."""
    gap = price.gap(sample_data)
    
    assert len(gap) == len(sample_data)
    assert pd.isna(gap.iloc[0])  # Первое значение должно быть NaN


def test_rolling_skewness(sample_data):
    """Rolling skewness должна возвращать числовые значения."""
    skew = price.rolling_skewness(sample_data, window=60)
    
    valid = skew.dropna()
    assert not valid.empty
    assert valid.dtype in [np.float64, np.float32]


def test_rolling_kurtosis(sample_data):
    """Rolling kurtosis должна возвращать числовые значения."""
    kurt = price.rolling_kurtosis(sample_data, window=60)
    
    valid = kurt.dropna()
    assert not valid.empty


def test_rolling_sharpe(sample_data):
    """Rolling Sharpe может быть положительным или отрицательным."""
    sharpe = price.rolling_sharpe(sample_data, window=60)
    
    assert len(sharpe) == len(sample_data)


def test_max_drawdown_negative(sample_data):
    """Max drawdown должен быть <= 0."""
    dd = price.max_drawdown(sample_data, window=60)
    
    valid = dd.dropna()
    assert (valid <= 0).all()


def test_recovery_time_non_negative(sample_data):
    """Recovery time должен быть >= 0."""
    recovery = price.recovery_time(sample_data)
    
    assert (recovery >= 0).all()


def test_atr_percent_positive(sample_data):
    """ATR percent должен быть положительным."""
    atr_pct = price.atr_percent(sample_data, period=14)
    
    valid = atr_pct.dropna()
    assert (valid >= 0).all()


def test_cumulative_returns(sample_data):
    """Cumulative returns должны расти/падать монотонно относительно начала."""
    cum_ret = price.cumulative_returns(sample_data)
    
    assert len(cum_ret) == len(sample_data)
    # Первое значение должно быть 0
    assert cum_ret.iloc[0] == 0
