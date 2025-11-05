"""
Библиотека технических индикаторов (20+).

Каждая функция принимает DataFrame и возвращает Series/DataFrame с индикатором.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================

def rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Relative Strength Index (RSI).
    
    Диапазон: 0-100
    Overbought: >70, Oversold: <30
    """
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))
    return rsi_values


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = 'close'
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).
    
    Возвращает DataFrame с колонками: macd, macd_signal, macd_hist
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    result = pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram
    }, index=df.index)
    
    return result


def stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K, %D).
    
    Диапазон: 0-100
    """
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * (df['close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    
    result = pd.DataFrame({
        'stoch_k': k,
        'stoch_d': d
    }, index=df.index)
    
    return result


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI).
    
    Типичный диапазон: от -100 до +100
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    
    cci_values = (tp - sma_tp) / (0.015 * mad)
    return cci_values


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Williams %R.
    
    Диапазон: -100 to 0
    Overbought: > -20, Oversold: < -80
    """
    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()
    
    wr = -100 * (high_max - df['close']) / (high_max - low_min)
    return wr


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI).
    
    Volume-weighted RSI. Диапазон: 0-100
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    
    delta = tp.diff()
    
    positive_mf = mf.where(delta > 0, 0).rolling(window=period).sum()
    negative_mf = mf.where(delta < 0, 0).rolling(window=period).sum()
    
    mfi_ratio = positive_mf / negative_mf
    mfi_values = 100 - (100 / (1 + mfi_ratio))
    return mfi_values


def roc(df: pd.DataFrame, period: int = 12, column: str = 'close') -> pd.Series:
    """
    Rate of Change (ROC) - процентное изменение цены.
    
    ROC = (Price - Price[n]) / Price[n] * 100
    """
    roc_values = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
    return roc_values


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) - средний истинный диапазон.
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_values = tr.rolling(window=period).mean()
    return atr_values


def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = 'close'
) -> pd.DataFrame:
    """
    Bollinger Bands (средняя, верхняя, нижняя полоса).
    """
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    result = pd.DataFrame({
        'bb_middle': sma,
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_width': upper - lower,
        'bb_pct': (df[column] - lower) / (upper - lower)
    }, index=df.index)
    
    return result


def keltner_channel(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0
) -> pd.DataFrame:
    """
    Keltner Channel - канал на основе EMA и ATR.
    """
    ema = df['close'].ewm(span=ema_period, adjust=False).mean()
    atr_val = atr(df, atr_period)
    
    upper = ema + (atr_val * atr_mult)
    lower = ema - (atr_val * atr_mult)
    
    result = pd.DataFrame({
        'kc_middle': ema,
        'kc_upper': upper,
        'kc_lower': lower
    }, index=df.index)
    
    return result


def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Donchian Channel - канал по максимумам и минимумам.
    """
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2
    
    result = pd.DataFrame({
        'dc_upper': upper,
        'dc_middle': middle,
        'dc_lower': lower
    }, index=df.index)
    
    return result


# ============================================================================
# TREND INDICATORS
# ============================================================================

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (ADX) + DI+ / DI-.
    
    ADX > 25: сильный тренд
    """
    # True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_smooth = tr.rolling(window=period).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=period).mean() / atr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=period).mean() / atr_smooth)
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_values = dx.rolling(window=period).mean()
    
    result = pd.DataFrame({
        'adx': adx_values,
        'di_plus': plus_di,
        'di_minus': minus_di
    }, index=df.index)
    
    return result


def aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """
    Aroon Indicator (Aroon Up, Aroon Down).
    
    Показывает силу тренда и его направление.
    """
    aroon_up = df['high'].rolling(window=period + 1).apply(
        lambda x: (period - (len(x) - 1 - x.values.argmax())) / period * 100,
        raw=False
    )
    aroon_down = df['low'].rolling(window=period + 1).apply(
        lambda x: (period - (len(x) - 1 - x.values.argmin())) / period * 100,
        raw=False
    )
    
    result = pd.DataFrame({
        'aroon_up': aroon_up,
        'aroon_down': aroon_down,
        'aroon_osc': aroon_up - aroon_down
    }, index=df.index)
    
    return result


def ema(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """Exponential Moving Average."""
    return df[column].ewm(span=period, adjust=False).mean()


def sma(df: pd.DataFrame, period: int = 20, column: str = 'close') -> pd.Series:
    """Simple Moving Average."""
    return df[column].rolling(window=period).mean()


def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP).
    
    Обычно рассчитывается внутри дня; здесь - кумулятивный.
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    vwap_values = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap_values


# ============================================================================
# VOLUME INDICATORS
# ============================================================================

def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV).
    
    Кумулятивный индикатор объёма.
    """
    obv_values = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv_values


def ad_line(df: pd.DataFrame) -> pd.Series:
    """
    Accumulation/Distribution Line.
    
    Показывает баланс покупок/продаж.
    """
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    clv = clv.fillna(0)
    ad = (clv * df['volume']).cumsum()
    return ad


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF).
    
    Диапазон: -1 to +1
    """
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfm = mfm.fillna(0)
    mfv = mfm * df['volume']
    
    cmf_values = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return cmf_values


# ============================================================================
# ADDITIONAL INDICATORS
# ============================================================================

def parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    """
    Parabolic SAR (Stop and Reverse).
    
    Упрощённая реализация.
    """
    # Инициализация
    sar = df['low'].iloc[0]
    ep = df['high'].iloc[0]
    af = af_start
    uptrend = True
    
    sar_values = [sar]
    
    for i in range(1, len(df)):
        if uptrend:
            sar = sar + af * (ep - sar)
            sar = min(sar, df['low'].iloc[i - 1])
            if i > 1:
                sar = min(sar, df['low'].iloc[i - 2])
            
            if df['low'].iloc[i] < sar:
                uptrend = False
                sar = ep
                ep = df['low'].iloc[i]
                af = af_start
            else:
                if df['high'].iloc[i] > ep:
                    ep = df['high'].iloc[i]
                    af = min(af + af_increment, af_max)
        else:
            sar = sar - af * (sar - ep)
            sar = max(sar, df['high'].iloc[i - 1])
            if i > 1:
                sar = max(sar, df['high'].iloc[i - 2])
            
            if df['high'].iloc[i] > sar:
                uptrend = True
                sar = ep
                ep = df['high'].iloc[i]
                af = af_start
            else:
                if df['low'].iloc[i] < ep:
                    ep = df['low'].iloc[i]
                    af = min(af + af_increment, af_max)
        
        sar_values.append(sar)
    
    return pd.Series(sar_values, index=df.index)


def ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> pd.DataFrame:
    """
    Ichimoku Cloud (основные линии).
    
    Возвращает: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b
    """
    # Tenkan-sen (Conversion Line)
    high_tenkan = df['high'].rolling(window=tenkan).max()
    low_tenkan = df['low'].rolling(window=tenkan).min()
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line)
    high_kijun = df['high'].rolling(window=kijun).max()
    low_kijun = df['low'].rolling(window=kijun).min()
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B)
    high_senkou = df['high'].rolling(window=senkou_b).max()
    low_senkou = df['low'].rolling(window=senkou_b).min()
    senkou_span_b = ((high_senkou + low_senkou) / 2).shift(kijun)
    
    result = pd.DataFrame({
        'ichimoku_tenkan': tenkan_sen,
        'ichimoku_kijun': kijun_sen,
        'ichimoku_senkou_a': senkou_span_a,
        'ichimoku_senkou_b': senkou_span_b
    }, index=df.index)
    
    return result
