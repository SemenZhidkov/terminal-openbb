"""
Регистратор фич: автоматическая регистрация всех 50+ фич в FeatureEngine.
"""

from .feature_engine import FeatureEngine
from . import technical_indicators as tech
from . import price_features as price
from . import rolling_features as rolling


def register_all_features(engine: FeatureEngine):
    """
    Регистрация всех доступных фич в движке.
    
    Args:
        engine: экземпляр FeatureEngine
    """
    
    # ========================================================================
    # TECHNICAL INDICATORS
    # ========================================================================
    
    # Momentum
    engine.register_feature(
        'rsi_14', tech.rsi, 'technical',
        'Relative Strength Index (14)',
        ['close'], {'period': 14}
    )
    
    engine.register_feature(
        'macd', tech.macd, 'technical',
        'MACD (12, 26, 9)',
        ['close'], {'fast': 12, 'slow': 26, 'signal': 9}
    )
    
    engine.register_feature(
        'stochastic', tech.stochastic, 'technical',
        'Stochastic Oscillator (14, 3)',
        ['high', 'low', 'close'], {'k_period': 14, 'd_period': 3}
    )
    
    engine.register_feature(
        'cci_20', tech.cci, 'technical',
        'Commodity Channel Index (20)',
        ['high', 'low', 'close'], {'period': 20}
    )
    
    engine.register_feature(
        'williams_r_14', tech.williams_r, 'technical',
        'Williams %R (14)',
        ['high', 'low', 'close'], {'period': 14}
    )
    
    engine.register_feature(
        'mfi_14', tech.mfi, 'technical',
        'Money Flow Index (14)',
        ['high', 'low', 'close', 'volume'], {'period': 14}
    )
    
    engine.register_feature(
        'roc_12', tech.roc, 'technical',
        'Rate of Change (12)',
        ['close'], {'period': 12}
    )
    
    # Volatility
    engine.register_feature(
        'atr_14', tech.atr, 'technical',
        'Average True Range (14)',
        ['high', 'low', 'close'], {'period': 14}
    )
    
    engine.register_feature(
        'bollinger_bands', tech.bollinger_bands, 'technical',
        'Bollinger Bands (20, 2)',
        ['close'], {'period': 20, 'std_dev': 2.0}
    )
    
    engine.register_feature(
        'keltner_channel', tech.keltner_channel, 'technical',
        'Keltner Channel (20, 10, 2)',
        ['high', 'low', 'close'], {'ema_period': 20, 'atr_period': 10, 'atr_mult': 2.0}
    )
    
    engine.register_feature(
        'donchian_channel', tech.donchian_channel, 'technical',
        'Donchian Channel (20)',
        ['high', 'low'], {'period': 20}
    )
    
    # Trend
    engine.register_feature(
        'adx_14', tech.adx, 'technical',
        'ADX with DI+/DI- (14)',
        ['high', 'low', 'close'], {'period': 14}
    )
    
    engine.register_feature(
        'aroon_25', tech.aroon, 'technical',
        'Aroon Indicator (25)',
        ['high', 'low'], {'period': 25}
    )
    
    engine.register_feature(
        'ema_20', tech.ema, 'technical',
        'Exponential Moving Average (20)',
        ['close'], {'period': 20}
    )
    
    engine.register_feature(
        'sma_50', tech.sma, 'technical',
        'Simple Moving Average (50)',
        ['close'], {'period': 50}
    )
    
    engine.register_feature(
        'sma_200', tech.sma, 'technical',
        'Simple Moving Average (200)',
        ['close'], {'period': 200}
    )
    
    engine.register_feature(
        'vwap', tech.vwap, 'technical',
        'Volume Weighted Average Price',
        ['high', 'low', 'close', 'volume'], {}
    )
    
    # Volume
    engine.register_feature(
        'obv', tech.obv, 'technical',
        'On-Balance Volume',
        ['close', 'volume'], {}
    )
    
    engine.register_feature(
        'ad_line', tech.ad_line, 'technical',
        'Accumulation/Distribution Line',
        ['high', 'low', 'close', 'volume'], {}
    )
    
    engine.register_feature(
        'cmf_20', tech.cmf, 'technical',
        'Chaikin Money Flow (20)',
        ['high', 'low', 'close', 'volume'], {'period': 20}
    )
    
    # Advanced
    engine.register_feature(
        'parabolic_sar', tech.parabolic_sar, 'technical',
        'Parabolic SAR',
        ['high', 'low'], {'af_start': 0.02, 'af_increment': 0.02, 'af_max': 0.2}
    )
    
    engine.register_feature(
        'ichimoku', tech.ichimoku, 'technical',
        'Ichimoku Cloud',
        ['high', 'low'], {'tenkan': 9, 'kijun': 26, 'senkou_b': 52}
    )
    
    # ========================================================================
    # PRICE-BASED FEATURES
    # ========================================================================
    
    # Returns
    engine.register_feature(
        'returns_1d', price.simple_returns, 'price_based',
        'Simple Returns (1 day)',
        ['close'], {'period': 1}
    )
    
    engine.register_feature(
        'returns_5d', price.simple_returns, 'price_based',
        'Simple Returns (5 days)',
        ['close'], {'period': 5}
    )
    
    engine.register_feature(
        'log_returns', price.log_returns, 'price_based',
        'Log Returns (1 day)',
        ['close'], {'period': 1}
    )
    
    engine.register_feature(
        'forward_returns_1d', price.forward_returns, 'price_based',
        'Forward Returns (1 day) - target',
        ['close'], {'period': 1}
    )
    
    # Volatility
    engine.register_feature(
        'realized_vol_20', price.realized_volatility, 'price_based',
        'Realized Volatility (20 days, annualized)',
        ['close'], {'window': 20, 'annualize': True}
    )
    
    engine.register_feature(
        'parkinson_vol_20', price.parkinson_volatility, 'price_based',
        'Parkinson Volatility (20)',
        ['high', 'low'], {'window': 20, 'annualize': True}
    )
    
    engine.register_feature(
        'garman_klass_vol_20', price.garman_klass_volatility, 'price_based',
        'Garman-Klass Volatility (20)',
        ['high', 'low', 'open', 'close'], {'window': 20, 'annualize': True}
    )
    
    engine.register_feature(
        'atr_percent', price.atr_percent, 'price_based',
        'ATR as % of price',
        ['high', 'low', 'close'], {'period': 14}
    )
    
    engine.register_feature(
        'volatility_regime', price.volatility_regime, 'price_based',
        'Volatility Regime (Low/Normal/High)',
        ['close'], {'window': 60, 'threshold': 1.0}
    )
    
    # Momentum
    engine.register_feature(
        'price_momentum', price.price_momentum, 'price_based',
        'Price Momentum (5, 10, 20, 60 days)',
        ['close'], {'periods': [5, 10, 20, 60]}
    )
    
    engine.register_feature(
        'price_acceleration', price.price_acceleration, 'price_based',
        'Price Acceleration (10)',
        ['close'], {'period': 10}
    )
    
    engine.register_feature(
        'rolling_zscore_20', price.rolling_zscore, 'price_based',
        'Rolling Z-score (20)',
        ['close'], {'window': 20}
    )
    
    engine.register_feature(
        'distance_from_high_252', price.price_distance_from_high, 'price_based',
        'Distance from 1Y High (%)',
        ['close'], {'window': 252}
    )
    
    engine.register_feature(
        'distance_from_low_252', price.price_distance_from_low, 'price_based',
        'Distance from 1Y Low (%)',
        ['close'], {'window': 252}
    )
    
    # Intraday patterns
    engine.register_feature(
        'daily_range', price.daily_range, 'price_based',
        'Daily Range (H-L)/Close',
        ['high', 'low', 'close'], {}
    )
    
    engine.register_feature(
        'body_size', price.body_size, 'price_based',
        'Candle Body Size',
        ['open', 'close'], {}
    )
    
    engine.register_feature(
        'upper_shadow', price.upper_shadow, 'price_based',
        'Upper Shadow',
        ['high', 'open', 'close'], {}
    )
    
    engine.register_feature(
        'lower_shadow', price.lower_shadow, 'price_based',
        'Lower Shadow',
        ['low', 'open', 'close'], {}
    )
    
    engine.register_feature(
        'gap', price.gap, 'price_based',
        'Overnight Gap',
        ['open', 'close'], {}
    )
    
    # Statistical
    engine.register_feature(
        'rolling_skewness', price.rolling_skewness, 'price_based',
        'Rolling Skewness (60)',
        ['close'], {'window': 60}
    )
    
    engine.register_feature(
        'rolling_kurtosis', price.rolling_kurtosis, 'price_based',
        'Rolling Kurtosis (60)',
        ['close'], {'window': 60}
    )
    
    engine.register_feature(
        'rolling_sharpe', price.rolling_sharpe, 'price_based',
        'Rolling Sharpe Ratio (60)',
        ['close'], {'window': 60, 'rf_rate': 0.03}
    )
    
    engine.register_feature(
        'max_drawdown', price.max_drawdown, 'price_based',
        'Max Drawdown in 1Y window',
        ['close'], {'window': 252}
    )
    
    engine.register_feature(
        'recovery_time', price.recovery_time, 'price_based',
        'Days since ATH',
        ['close'], {}
    )
    
    # ========================================================================
    # ROLLING WINDOW FEATURES
    # ========================================================================
    
    engine.register_feature(
        'sma_multi', rolling.rolling_mean, 'rolling',
        'Multiple SMAs (5,10,20,50,200)',
        ['close'], {'windows': [5, 10, 20, 50, 200]}
    )
    
    engine.register_feature(
        'rolling_std', rolling.rolling_std, 'rolling',
        'Rolling Std Dev (10,20,60)',
        ['close'], {'windows': [10, 20, 60]}
    )
    
    engine.register_feature(
        'ema_crossover_12_26', rolling.ema_crossover, 'rolling',
        'EMA Crossover (12/26)',
        ['close'], {'fast': 12, 'slow': 26}
    )
    
    engine.register_feature(
        'golden_death_cross', rolling.golden_death_cross, 'rolling',
        'Golden/Death Cross (50/200)',
        ['close'], {}
    )
    
    engine.register_feature(
        'price_vs_ma', rolling.price_vs_ma, 'rolling',
        'Price vs MA % (20,50,200)',
        ['close'], {'windows': [20, 50, 200]}
    )
    
    engine.register_feature(
        'autocorrelation', rolling.autocorrelation, 'rolling',
        'Returns Autocorrelation',
        ['close'], {'lags': [1, 5, 10, 20]}
    )
    
    engine.register_feature(
        'trend_strength', rolling.trend_strength, 'rolling',
        'Trend Strength (R²) (20)',
        ['close'], {'window': 20}
    )
    
    engine.register_feature(
        'linear_regression_slope', rolling.linear_regression_slope, 'rolling',
        'Linear Regression Slope (20)',
        ['close'], {'window': 20}
    )
    
    engine.register_feature(
        'volatility_ratio', rolling.volatility_ratio, 'rolling',
        'Volatility Ratio (10/60)',
        ['close'], {'short_window': 10, 'long_window': 60}
    )
    
    engine.register_feature(
        'hurst_exponent', rolling.hurst_exponent, 'rolling',
        'Hurst Exponent (100)',
        ['close'], {'window': 100}
    )


def get_feature_count() -> dict:
    """
    Подсчёт зарегистрированных фич.
    
    Returns:
        Словарь с числом фич по группам
    """
    # Создаём временный engine и регистрируем
    temp_engine = FeatureEngine()
    register_all_features(temp_engine)
    
    return temp_engine.get_summary()
