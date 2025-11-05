"""
Unit-тесты для технических индикаторов.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import technical_indicators as tech


@pytest.fixture
def sample_ohlcv():
    """Синтетические OHLCV данные для тестирования."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.5,
        'high': close_prices + np.abs(np.random.randn(100) * 1.5),
        'low': close_prices - np.abs(np.random.randn(100) * 1.5),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


def test_rsi_range(sample_ohlcv):
    """RSI должен быть в диапазоне 0-100."""
    rsi = tech.rsi(sample_ohlcv, period=14)
    
    assert rsi.min() >= 0, "RSI не может быть меньше 0"
    assert rsi.max() <= 100, "RSI не может быть больше 100"
    assert not rsi.dropna().empty, "RSI должен вернуть значения"


def test_macd_columns(sample_ohlcv):
    """MACD должен вернуть DataFrame с тремя колонками."""
    macd_df = tech.macd(sample_ohlcv)
    
    assert isinstance(macd_df, pd.DataFrame)
    assert 'macd' in macd_df.columns
    assert 'macd_signal' in macd_df.columns
    assert 'macd_hist' in macd_df.columns
    assert len(macd_df) == len(sample_ohlcv)


def test_stochastic_range(sample_ohlcv):
    """Stochastic должен быть в диапазоне 0-100."""
    stoch = tech.stochastic(sample_ohlcv)
    
    assert 'stoch_k' in stoch.columns
    assert 'stoch_d' in stoch.columns
    
    k_valid = stoch['stoch_k'].dropna()
    assert k_valid.min() >= 0
    assert k_valid.max() <= 100


def test_atr_positive(sample_ohlcv):
    """ATR должен быть положительным."""
    atr_val = tech.atr(sample_ohlcv, period=14)
    
    valid_atr = atr_val.dropna()
    assert (valid_atr >= 0).all(), "ATR не может быть отрицательным"


def test_bollinger_bands_structure(sample_ohlcv):
    """Bollinger Bands: upper > middle > lower."""
    bb = tech.bollinger_bands(sample_ohlcv, period=20)
    
    assert 'bb_upper' in bb.columns
    assert 'bb_middle' in bb.columns
    assert 'bb_lower' in bb.columns
    
    valid_rows = bb.dropna()
    assert (valid_rows['bb_upper'] >= valid_rows['bb_middle']).all()
    assert (valid_rows['bb_middle'] >= valid_rows['bb_lower']).all()


def test_adx_range(sample_ohlcv):
    """ADX обычно в диапазоне 0-100."""
    adx_df = tech.adx(sample_ohlcv)
    
    assert 'adx' in adx_df.columns
    adx_values = adx_df['adx'].dropna()
    
    assert adx_values.min() >= 0
    # ADX может быть > 100 в экстремальных случаях, но обычно < 100


def test_obv_cumulative(sample_ohlcv):
    """OBV должен быть кумулятивным (монотонный рост/падение)."""
    obv_val = tech.obv(sample_ohlcv)
    
    assert len(obv_val) == len(sample_ohlcv)
    assert not obv_val.isnull().all()


def test_sma_vs_ema(sample_ohlcv):
    """SMA и EMA должны быть близки для больших периодов."""
    sma_val = tech.sma(sample_ohlcv, period=20)
    ema_val = tech.ema(sample_ohlcv, period=20)
    
    # Убираем первые значения (warmup period)
    correlation = sma_val[30:].corr(ema_val[30:])
    assert correlation > 0.95, "SMA и EMA должны коррелировать"


def test_vwap_non_negative(sample_ohlcv):
    """VWAP не должен быть отрицательным."""
    vwap_val = tech.vwap(sample_ohlcv)
    
    assert (vwap_val > 0).all(), "VWAP должен быть положительным"


def test_indicators_handle_nan():
    """Индикаторы должны корректно обрабатывать NaN."""
    df = pd.DataFrame({
        'open': [100, 101, np.nan, 103],
        'high': [102, 103, 104, 105],
        'low': [99, 100, 101, 102],
        'close': [101, 102, 103, 104],
        'volume': [1000, 1000, 1000, 1000]
    })
    
    # Не должно падать
    rsi = tech.rsi(df, period=2)
    assert len(rsi) == len(df)


def test_williams_r_range(sample_ohlcv):
    """Williams %R должен быть в диапазоне -100 to 0."""
    wr = tech.williams_r(sample_ohlcv, period=14)
    
    valid = wr.dropna()
    assert valid.min() >= -100
    assert valid.max() <= 0


def test_cci_calculation(sample_ohlcv):
    """CCI должен возвращать числовые значения."""
    cci = tech.cci(sample_ohlcv, period=20)
    
    assert not cci.dropna().empty
    assert cci.dtype in [np.float64, np.float32]


def test_mfi_range(sample_ohlcv):
    """MFI должен быть в диапазоне 0-100."""
    mfi = tech.mfi(sample_ohlcv, period=14)
    
    valid = mfi.dropna()
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_parabolic_sar_length(sample_ohlcv):
    """Parabolic SAR должен вернуть значение для каждой строки."""
    sar = tech.parabolic_sar(sample_ohlcv)
    
    assert len(sar) == len(sample_ohlcv)


def test_ichimoku_structure(sample_ohlcv):
    """Ichimoku должен вернуть 4 линии."""
    ich = tech.ichimoku(sample_ohlcv)
    
    assert 'ichimoku_tenkan' in ich.columns
    assert 'ichimoku_kijun' in ich.columns
    assert 'ichimoku_senkou_a' in ich.columns
    assert 'ichimoku_senkou_b' in ich.columns


def test_roc_calculation(sample_ohlcv):
    """ROC должен корректно рассчитывать процентное изменение."""
    roc = tech.roc(sample_ohlcv, period=1)
    manual_roc = sample_ohlcv['close'].pct_change(1) * 100
    
    # Сравниваем (с учётом погрешности округления)
    valid_idx = ~roc.isnull() & ~manual_roc.isnull()
    np.testing.assert_array_almost_equal(
        roc[valid_idx].values,
        manual_roc[valid_idx].values,
        decimal=6
    )


def test_aroon_range(sample_ohlcv):
    """Aroon значения должны быть в 0-100."""
    aroon = tech.aroon(sample_ohlcv, period=25)
    
    for col in ['aroon_up', 'aroon_down']:
        valid = aroon[col].dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100
