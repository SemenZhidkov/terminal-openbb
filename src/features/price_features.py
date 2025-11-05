"""
Price-based фичи: доходности, волатильность, моментум, режимы волатильности.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ============================================================================
# RETURNS
# ============================================================================

def simple_returns(df: pd.DataFrame, period: int = 1, column: str = 'close') -> pd.Series:
    """
    Простые доходности: (P[t] - P[t-n]) / P[t-n]
    """
    return df[column].pct_change(periods=period)


def log_returns(df: pd.DataFrame, period: int = 1, column: str = 'close') -> pd.Series:
    """
    Логарифмические доходности: ln(P[t] / P[t-n])
    """
    return np.log(df[column] / df[column].shift(period))


def forward_returns(df: pd.DataFrame, period: int = 1, column: str = 'close') -> pd.Series:
    """
    Будущие доходности (для таргетов): (P[t+n] - P[t]) / P[t]
    """
    return df[column].pct_change(periods=period).shift(-period)


def cumulative_returns(df: pd.DataFrame, column: str = 'close') -> pd.Series:
    """
    Кумулятивные доходности с начала ряда.
    """
    return (df[column] / df[column].iloc[0]) - 1


# ============================================================================
# VOLATILITY
# ============================================================================

def realized_volatility(
    df: pd.DataFrame,
    window: int = 20,
    column: str = 'close',
    annualize: bool = True
) -> pd.Series:
    """
    Realized Volatility - стандартное отклонение доходностей.
    
    Args:
        window: окно расчёта
        annualize: умножать на sqrt(252) для годовой волатильности
    """
    returns = df[column].pct_change()
    vol = returns.rolling(window=window).std()
    
    if annualize:
        vol = vol * np.sqrt(252)
    
    return vol


def parkinson_volatility(df: pd.DataFrame, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Parkinson Volatility - оценка волатильности по high/low.
    
    Более эффективна, чем realized vol при отсутствии гэпов.
    """
    hl_ratio = np.log(df['high'] / df['low'])
    parkinson_vol = hl_ratio.rolling(window=window).apply(
        lambda x: np.sqrt((1 / (4 * np.log(2))) * (x ** 2).mean()),
        raw=True
    )
    
    if annualize:
        parkinson_vol = parkinson_vol * np.sqrt(252)
    
    return parkinson_vol


def garman_klass_volatility(df: pd.DataFrame, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Garman-Klass Volatility - учитывает OHLC.
    
    Ещё точнее чем Parkinson.
    """
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    
    gk_vol = log_hl.rolling(window=window).apply(
        lambda x: np.sqrt(0.5 * (x ** 2).mean() - (2 * np.log(2) - 1) * (log_co ** 2).mean()),
        raw=False
    )
    
    if annualize:
        gk_vol = gk_vol * np.sqrt(252)
    
    return gk_vol


def yang_zhang_volatility(df: pd.DataFrame, window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Yang-Zhang Volatility - комбинация overnight и intraday волатильности.
    
    Устойчива к гэпам и дрифту.
    """
    log_ho = np.log(df['high'] / df['open'])
    log_lo = np.log(df['low'] / df['open'])
    log_co = np.log(df['close'] / df['open'])
    log_oc = np.log(df['open'] / df['close'].shift(1))
    log_cc = np.log(df['close'] / df['close'].shift(1))
    
    # Overnight variance
    overnight_var = log_oc.rolling(window=window).var()
    
    # Open-close variance
    open_close_var = log_cc.rolling(window=window).var()
    
    # Rogers-Satchell component
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    rs_var = rs.rolling(window=window).mean()
    
    # Yang-Zhang estimator
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_vol = np.sqrt(overnight_var + k * open_close_var + (1 - k) * rs_var)
    
    if annualize:
        yz_vol = yz_vol * np.sqrt(252)
    
    return yz_vol


def atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR в процентах от цены (нормализованная волатильность).
    """
    from .technical_indicators import atr
    atr_val = atr(df, period)
    return (atr_val / df['close']) * 100


# ============================================================================
# VOLATILITY REGIMES
# ============================================================================

def volatility_regime(df: pd.DataFrame, window: int = 60, threshold: float = 1.0) -> pd.Series:
    """
    Классификация режима волатильности (Low/Normal/High).
    
    Args:
        window: окно для расчёта
        threshold: порог в стандартных отклонениях
        
    Returns:
        Series с метками: 0 = Low, 1 = Normal, 2 = High
    """
    vol = realized_volatility(df, window=20, annualize=False)
    vol_mean = vol.rolling(window=window).mean()
    vol_std = vol.rolling(window=window).std()
    
    z_score = (vol - vol_mean) / vol_std
    
    regime = pd.Series(1, index=df.index)  # Normal по умолчанию
    regime[z_score < -threshold] = 0  # Low vol
    regime[z_score > threshold] = 2   # High vol
    
    return regime


# ============================================================================
# MOMENTUM & PRICE PATTERNS
# ============================================================================

def price_momentum(df: pd.DataFrame, periods: list = None, column: str = 'close') -> pd.DataFrame:
    """
    Моментум цены на нескольких горизонтах.
    
    Args:
        periods: список периодов (по умолчанию [5, 10, 20, 60])
    """
    if periods is None:
        periods = [5, 10, 20, 60]
    
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f'momentum_{period}d'] = df[column].pct_change(period)
    
    return result


def price_acceleration(df: pd.DataFrame, period: int = 10, column: str = 'close') -> pd.Series:
    """
    Ускорение цены (вторая производная).
    
    acceleration = momentum[t] - momentum[t-1]
    """
    momentum = df[column].pct_change(period)
    acceleration = momentum.diff()
    return acceleration


def rolling_zscore(df: pd.DataFrame, window: int = 20, column: str = 'close') -> pd.Series:
    """
    Rolling Z-score цены.
    
    Показывает отклонение от скользящего среднего в стандартных отклонениях.
    """
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    
    z_score = (df[column] - rolling_mean) / rolling_std
    return z_score


def price_distance_from_high(df: pd.DataFrame, window: int = 252, column: str = 'close') -> pd.Series:
    """
    Расстояние текущей цены от максимума за период (в %).
    
    Negative values = below high
    """
    rolling_high = df[column].rolling(window=window).max()
    distance = ((df[column] - rolling_high) / rolling_high) * 100
    return distance


def price_distance_from_low(df: pd.DataFrame, window: int = 252, column: str = 'close') -> pd.Series:
    """
    Расстояние текущей цены от минимума за период (в %).
    
    Positive values = above low
    """
    rolling_low = df[column].rolling(window=window).min()
    distance = ((df[column] - rolling_low) / rolling_low) * 100
    return distance


# ============================================================================
# INTRADAY PATTERNS
# ============================================================================

def daily_range(df: pd.DataFrame) -> pd.Series:
    """
    Дневной диапазон: (high - low) / close
    """
    return (df['high'] - df['low']) / df['close']


def body_size(df: pd.DataFrame) -> pd.Series:
    """
    Размер тела свечи: |close - open| / close
    """
    return np.abs(df['close'] - df['open']) / df['close']


def upper_shadow(df: pd.DataFrame) -> pd.Series:
    """
    Верхняя тень свечи: (high - max(open, close)) / close
    """
    body_high = df[['open', 'close']].max(axis=1)
    return (df['high'] - body_high) / df['close']


def lower_shadow(df: pd.DataFrame) -> pd.Series:
    """
    Нижняя тень свечи: (min(open, close) - low) / close
    """
    body_low = df[['open', 'close']].min(axis=1)
    return (body_low - df['low']) / df['close']


def gap(df: pd.DataFrame) -> pd.Series:
    """
    Gap между свечами: (open[t] - close[t-1]) / close[t-1]
    """
    return (df['open'] - df['close'].shift(1)) / df['close'].shift(1)


# ============================================================================
# STATISTICAL FEATURES
# ============================================================================

def rolling_skewness(df: pd.DataFrame, window: int = 60, column: str = 'close') -> pd.Series:
    """
    Скользящий коэффициент асимметрии (skewness) доходностей.
    """
    returns = df[column].pct_change()
    skew = returns.rolling(window=window).skew()
    return skew


def rolling_kurtosis(df: pd.DataFrame, window: int = 60, column: str = 'close') -> pd.Series:
    """
    Скользящий эксцесс (kurtosis) доходностей.
    """
    returns = df[column].pct_change()
    kurt = returns.rolling(window=window).kurt()
    return kurt


def rolling_sharpe(df: pd.DataFrame, window: int = 60, column: str = 'close', rf_rate: float = 0.0) -> pd.Series:
    """
    Rolling Sharpe Ratio.
    
    Args:
        rf_rate: безрисковая ставка (годовая)
    """
    returns = df[column].pct_change()
    excess_returns = returns - (rf_rate / 252)
    
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    
    sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    return sharpe


def rolling_sortino(df: pd.DataFrame, window: int = 60, column: str = 'close', rf_rate: float = 0.0) -> pd.Series:
    """
    Rolling Sortino Ratio (учитывает только downside volatility).
    """
    returns = df[column].pct_change()
    excess_returns = returns - (rf_rate / 252)
    
    rolling_mean = excess_returns.rolling(window=window).mean()
    
    # Downside deviation
    downside_returns = returns.where(returns < 0, 0)
    downside_std = downside_returns.rolling(window=window).std()
    
    sortino = (rolling_mean / downside_std) * np.sqrt(252)
    return sortino


def max_drawdown(df: pd.DataFrame, window: int = 252, column: str = 'close') -> pd.Series:
    """
    Максимальная просадка в скользящем окне.
    """
    rolling_max = df[column].rolling(window=window, min_periods=1).max()
    drawdown = (df[column] - rolling_max) / rolling_max
    return drawdown


def recovery_time(df: pd.DataFrame, column: str = 'close') -> pd.Series:
    """
    Количество дней с момента последнего ATH.
    """
    cummax = df[column].cummax()
    is_new_high = (df[column] == cummax).astype(int)
    
    # Инкремент счётчика если не новый максимум
    recovery = pd.Series(0, index=df.index)
    counter = 0
    
    for i in range(len(df)):
        if is_new_high.iloc[i] == 1:
            counter = 0
        else:
            counter += 1
        recovery.iloc[i] = counter
    
    return recovery
