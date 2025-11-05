"""
Rolling window фичи: статистика, кроссоверы, автокорреляция.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ============================================================================
# ROLLING STATISTICS
# ============================================================================

def rolling_mean(df: pd.DataFrame, windows: list = None, column: str = 'close') -> pd.DataFrame:
    """
    Скользящие средние на нескольких окнах.
    
    Args:
        windows: список периодов (по умолчанию [5, 10, 20, 50, 200])
    """
    if windows is None:
        windows = [5, 10, 20, 50, 200]
    
    result = pd.DataFrame(index=df.index)
    for window in windows:
        result[f'sma_{window}'] = df[column].rolling(window=window).mean()
    
    return result


def rolling_std(df: pd.DataFrame, windows: list = None, column: str = 'close') -> pd.DataFrame:
    """
    Скользящее стандартное отклонение.
    """
    if windows is None:
        windows = [10, 20, 60]
    
    result = pd.DataFrame(index=df.index)
    for window in windows:
        result[f'std_{window}'] = df[column].rolling(window=window).std()
    
    return result


def rolling_min_max(df: pd.DataFrame, windows: list = None, column: str = 'close') -> pd.DataFrame:
    """
    Скользящие минимумы и максимумы.
    """
    if windows is None:
        windows = [20, 60, 252]
    
    result = pd.DataFrame(index=df.index)
    for window in windows:
        result[f'rolling_min_{window}'] = df[column].rolling(window=window).min()
        result[f'rolling_max_{window}'] = df[column].rolling(window=window).max()
    
    return result


def rolling_quantile(
    df: pd.DataFrame,
    windows: list = None,
    quantiles: list = None,
    column: str = 'close'
) -> pd.DataFrame:
    """
    Скользящие квантили.
    
    Args:
        windows: список периодов
        quantiles: список квантилей (напр. [0.1, 0.5, 0.9])
    """
    if windows is None:
        windows = [20, 60]
    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75]
    
    result = pd.DataFrame(index=df.index)
    for window in windows:
        for q in quantiles:
            col_name = f'q{int(q*100)}_{window}'
            result[col_name] = df[column].rolling(window=window).quantile(q)
    
    return result


def expanding_stats(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
    """
    Expanding window статистика (от начала ряда).
    """
    result = pd.DataFrame({
        'expanding_mean': df[column].expanding().mean(),
        'expanding_std': df[column].expanding().std(),
        'expanding_min': df[column].expanding().min(),
        'expanding_max': df[column].expanding().max()
    }, index=df.index)
    
    return result


# ============================================================================
# EMA CROSSOVERS
# ============================================================================

def ema_crossover(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    column: str = 'close'
) -> pd.DataFrame:
    """
    EMA кроссовер и расстояние между EMA.
    
    Returns:
        DataFrame с: ema_fast, ema_slow, ema_diff, ema_cross_signal
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    
    diff = ema_fast - ema_slow
    
    # Сигнал пересечения: 1 = fast выше slow, -1 = наоборот
    cross_signal = np.sign(diff)
    
    result = pd.DataFrame({
        f'ema_{fast}': ema_fast,
        f'ema_{slow}': ema_slow,
        f'ema_diff_{fast}_{slow}': diff,
        f'ema_cross_{fast}_{slow}': cross_signal
    }, index=df.index)
    
    return result


def golden_death_cross(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
    """
    Golden Cross (50/200) и Death Cross.
    
    Golden Cross: SMA50 пересекает SMA200 снизу вверх
    Death Cross: SMA50 пересекает SMA200 сверху вниз
    """
    sma_50 = df[column].rolling(window=50).mean()
    sma_200 = df[column].rolling(window=200).mean()
    
    diff = sma_50 - sma_200
    cross_signal = np.sign(diff)
    
    # Детект момента пересечения
    cross_change = cross_signal.diff()
    golden_cross = (cross_change == 2).astype(int)  # -1 -> +1
    death_cross = (cross_change == -2).astype(int)  # +1 -> -1
    
    result = pd.DataFrame({
        'sma_50': sma_50,
        'sma_200': sma_200,
        'sma_50_200_diff': diff,
        'golden_cross': golden_cross,
        'death_cross': death_cross
    }, index=df.index)
    
    return result


# ============================================================================
# PRICE POSITION RELATIVE TO MOVING AVERAGES
# ============================================================================

def price_vs_ma(df: pd.DataFrame, windows: list = None, column: str = 'close') -> pd.DataFrame:
    """
    Отклонение цены от скользящих средних (в %).
    
    Показывает, насколько цена выше/ниже MA.
    """
    if windows is None:
        windows = [20, 50, 200]
    
    result = pd.DataFrame(index=df.index)
    for window in windows:
        ma = df[column].rolling(window=window).mean()
        result[f'price_vs_sma_{window}'] = ((df[column] - ma) / ma) * 100
    
    return result


# ============================================================================
# AUTOCORRELATION
# ============================================================================

def autocorrelation(df: pd.DataFrame, lags: list = None, column: str = 'close') -> pd.DataFrame:
    """
    Автокорреляция доходностей на разных лагах.
    
    Помогает детектить mean reversion или momentum persistence.
    """
    if lags is None:
        lags = [1, 5, 10, 20]
    
    returns = df[column].pct_change()
    
    result = pd.DataFrame(index=df.index)
    for lag in lags:
        # Rolling autocorrelation
        result[f'autocorr_lag_{lag}'] = returns.rolling(window=60).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan,
            raw=False
        )
    
    return result


# ============================================================================
# TREND STRENGTH
# ============================================================================

def trend_strength(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    Сила тренда через R² линейной регрессии.
    
    Значения близкие к 1 = сильный тренд, близкие к 0 = боковик.
    """
    def rolling_r2(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        
        # Линейная регрессия
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            y_pred = m * x + c
            
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            return r2
        except:
            return np.nan
    
    r2 = df[column].rolling(window=window).apply(rolling_r2, raw=False)
    return r2


def linear_regression_slope(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    Наклон линейной регрессии в скользящем окне.
    
    Положительный = восходящий тренд, отрицательный = нисходящий.
    """
    def rolling_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            m, c = np.linalg.lstsq(A, y, rcond=None)[0]
            return m
        except:
            return np.nan
    
    slope = df[column].rolling(window=window).apply(rolling_slope, raw=False)
    return slope


def linear_regression_angle(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    Угол наклона линейной регрессии в градусах.
    
    Удобнее интерпретировать чем просто slope.
    """
    slope = linear_regression_slope(df, window, column)
    angle = np.arctan(slope) * (180 / np.pi)
    return angle


# ============================================================================
# ROLLING CORRELATIONS (для нескольких инструментов)
# ============================================================================

def rolling_correlation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    window: int = 60,
    column: str = 'close'
) -> pd.Series:
    """
    Скользящая корреляция между двумя инструментами.
    
    Args:
        df1, df2: датафреймы с совпадающими индексами
        window: окно расчёта
        column: колонка для расчёта
    """
    returns1 = df1[column].pct_change()
    returns2 = df2[column].pct_change()
    
    corr = returns1.rolling(window=window).corr(returns2)
    return corr


# ============================================================================
# REGIME DETECTION (через rolling std)
# ============================================================================

def volatility_ratio(df: pd.DataFrame, short_window: int = 10, long_window: int = 60) -> pd.Series:
    """
    Отношение краткосрочной волатильности к долгосрочной.
    
    >1 = волатильность растёт
    <1 = волатильность падает
    """
    returns = df['close'].pct_change()
    short_vol = returns.rolling(window=short_window).std()
    long_vol = returns.rolling(window=long_window).std()
    
    ratio = short_vol / long_vol
    return ratio


def hurst_exponent(df: pd.DataFrame, window: int = 100, column: str = 'close') -> pd.Series:
    """
    Hurst Exponent в скользящем окне (упрощённая версия).
    
    H < 0.5: mean-reverting
    H = 0.5: random walk
    H > 0.5: trending/persistent
    """
    def calc_hurst(series):
        if len(series) < 20:
            return np.nan
        
        lags = range(2, min(20, len(series) // 2))
        tau = []
        
        for lag in lags:
            # Rescaled range
            lagged_series = series.iloc[:len(series) - lag]
            std = lagged_series.std()
            
            if std == 0:
                continue
            
            mean_centered = lagged_series - lagged_series.mean()
            cumsum = mean_centered.cumsum()
            
            R = cumsum.max() - cumsum.min()
            S = std
            
            if S != 0:
                tau.append(R / S)
        
        if len(tau) < 2:
            return np.nan
        
        # Линейная регрессия log(R/S) vs log(lag)
        try:
            lags_arr = np.array(range(2, 2 + len(tau)))
            tau_arr = np.array(tau)
            
            log_lags = np.log(lags_arr)
            log_tau = np.log(tau_arr)
            
            hurst = np.polyfit(log_lags, log_tau, 1)[0]
            return hurst
        except:
            return np.nan
    
    hurst = df[column].rolling(window=window).apply(calc_hurst, raw=False)
    return hurst
