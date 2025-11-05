"""
Convenience imports для фич-модулей.
"""

from .feature_engine import FeatureEngine

# Technical indicators
from .technical_indicators import (
    rsi, macd, stochastic, cci, williams_r, mfi, roc,
    atr, bollinger_bands, keltner_channel, donchian_channel,
    adx, aroon, ema, sma, vwap,
    obv, ad_line, cmf,
    parabolic_sar, ichimoku
)

# Price-based features
from .price_features import (
    simple_returns, log_returns, forward_returns, cumulative_returns,
    realized_volatility, parkinson_volatility, garman_klass_volatility,
    yang_zhang_volatility, atr_percent, volatility_regime,
    price_momentum, price_acceleration, rolling_zscore,
    price_distance_from_high, price_distance_from_low,
    daily_range, body_size, upper_shadow, lower_shadow, gap,
    rolling_skewness, rolling_kurtosis, rolling_sharpe, rolling_sortino,
    max_drawdown, recovery_time
)

# Rolling window features
from .rolling_features import (
    rolling_mean, rolling_std, rolling_min_max, rolling_quantile,
    expanding_stats, ema_crossover, golden_death_cross, price_vs_ma,
    autocorrelation, trend_strength, linear_regression_slope,
    linear_regression_angle, volatility_ratio, hurst_exponent
)

# Macro features (placeholders)
from .macro_features import (
    get_vix, get_treasury_yield, get_yield_curve_slope,
    get_dxy, get_oil_price, get_gold_price,
    merge_macro_features
)

__all__ = [
    'FeatureEngine',
    # Technical
    'rsi', 'macd', 'stochastic', 'cci', 'williams_r', 'mfi', 'roc',
    'atr', 'bollinger_bands', 'keltner_channel', 'donchian_channel',
    'adx', 'aroon', 'ema', 'sma', 'vwap',
    'obv', 'ad_line', 'cmf',
    'parabolic_sar', 'ichimoku',
    # Price-based
    'simple_returns', 'log_returns', 'forward_returns', 'cumulative_returns',
    'realized_volatility', 'parkinson_volatility', 'garman_klass_volatility',
    'yang_zhang_volatility', 'atr_percent', 'volatility_regime',
    'price_momentum', 'price_acceleration', 'rolling_zscore',
    'price_distance_from_high', 'price_distance_from_low',
    'daily_range', 'body_size', 'upper_shadow', 'lower_shadow', 'gap',
    'rolling_skewness', 'rolling_kurtosis', 'rolling_sharpe', 'rolling_sortino',
    'max_drawdown', 'recovery_time',
    # Rolling
    'rolling_mean', 'rolling_std', 'rolling_min_max', 'rolling_quantile',
    'expanding_stats', 'ema_crossover', 'golden_death_cross', 'price_vs_ma',
    'autocorrelation', 'trend_strength', 'linear_regression_slope',
    'linear_regression_angle', 'volatility_ratio', 'hurst_exponent',
    # Macro
    'get_vix', 'get_treasury_yield', 'get_yield_curve_slope',
    'get_dxy', 'get_oil_price', 'get_gold_price',
    'merge_macro_features'
]
