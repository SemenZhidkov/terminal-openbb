# Feature Engineering - Краткий гайд

## 🎯 Что реализовано

**Часть 3 проекта завершена:** система расчёта и управления фичами с 50+ готовыми признаками.

## 📦 Структура модулей

```
src/features/
├── feature_engine.py           # Ядро: регистрация, кеширование, версионирование
├── technical_indicators.py     # 20+ технических индикаторов
├── price_features.py           # Price-based фичи (returns, volatility, momentum)
├── rolling_features.py         # Rolling window фичи (SMA crossovers, trend strength)
├── macro_features.py           # Макро-индикаторы (заглушки для VIX, rates, etc.)
├── feature_registry.py         # Автоматическая регистрация всех фич
└── __init__.py                 # Convenience imports
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install pytest scipy
```

### 2. Простой пример

```python
from src.data.data_manager import DataManager
from src.features.feature_engine import FeatureEngine
from src.features.feature_registry import register_all_features

# Загрузка данных
dm = DataManager()
df = dm.get_stock_data('AAPL', '1d', '5y')

# Создание Feature Engine
engine = FeatureEngine()
register_all_features(engine)

# Статистика
summary = engine.get_summary()
print(f"Зарегистрировано фич: {summary['total_features']}")

# Расчёт ВСЕХ фич (50+)
df_with_features = engine.compute_all(df)
print(f"Добавлено колонок: {len(df_with_features.columns) - len(df.columns)}")
```

### 3. Расчёт по группам

```python
# Только технические индикаторы
df_tech = engine.compute_all(df, groups=['technical'])

# Только price-based фичи
df_price = engine.compute_all(df, groups=['price_based'])

# Только rolling window фичи
df_rolling = engine.compute_all(df, groups=['rolling'])
```

### 4. Расчёт одной фичи

```python
# RSI
df_with_rsi = engine.compute_feature('rsi_14', df)

# MACD
df_with_macd = engine.compute_feature('macd', df)

# Bollinger Bands
df_with_bb = engine.compute_feature('bollinger_bands', df)
```

## 📊 Список всех 50+ фич

### Технические индикаторы (23 фичи)

**Momentum:**
- `rsi_14` — Relative Strength Index
- `macd` — MACD + signal + histogram (3 колонки)
- `stochastic` — Stochastic %K и %D
- `cci_20` — Commodity Channel Index
- `williams_r_14` — Williams %R
- `mfi_14` — Money Flow Index
- `roc_12` — Rate of Change

**Volatility:**
- `atr_14` — Average True Range
- `bollinger_bands` — BB upper/middle/lower/width/pct (5 колонок)
- `keltner_channel` — KC upper/middle/lower (3 колонки)
- `donchian_channel` — DC upper/middle/lower (3 колонки)

**Trend:**
- `adx_14` — ADX + DI+/DI- (3 колонки)
- `aroon_25` — Aroon Up/Down/Oscillator (3 колонки)
- `ema_20` — Exponential MA
- `sma_50`, `sma_200` — Simple MA
- `vwap` — Volume Weighted Average Price

**Volume:**
- `obv` — On-Balance Volume
- `ad_line` — Accumulation/Distribution
- `cmf_20` — Chaikin Money Flow

**Advanced:**
- `parabolic_sar` — Parabolic SAR
- `ichimoku` — Ichimoku Cloud (4 линии)

### Price-Based фичи (25 фичей)

**Returns:**
- `returns_1d`, `returns_5d` — Simple returns
- `log_returns` — Log returns
- `forward_returns_1d` — Forward returns (target)

**Volatility:**
- `realized_vol_20` — Realized volatility (annualized)
- `parkinson_vol_20` — Parkinson estimator
- `garman_klass_vol_20` — Garman-Klass estimator
- `atr_percent` — ATR as % of price
- `volatility_regime` — Low/Normal/High (0/1/2)

**Momentum:**
- `price_momentum` — Multi-period momentum (4 колонки)
- `price_acceleration` — Второ проивзодная
- `rolling_zscore_20` — Z-score price
- `distance_from_high_252`, `distance_from_low_252` — Distance from extremes

**Candle Patterns:**
- `daily_range`, `body_size`, `upper_shadow`, `lower_shadow`, `gap`

**Statistical:**
- `rolling_skewness`, `rolling_kurtosis`
- `rolling_sharpe`, `rolling_sortino`
- `max_drawdown`, `recovery_time`

### Rolling Window фичи (12 фичей)

- `sma_multi` — Multiple SMAs (5/10/20/50/200)
- `rolling_std` — Rolling Std Dev (3 окна)
- `ema_crossover_12_26` — EMA crossover (4 колонки)
- `golden_death_cross` — 50/200 cross (5 колонок)
- `price_vs_ma` — Price vs MA % (3 колонки)
- `autocorrelation` — Returns autocorr (4 лага)
- `trend_strength` — R² линейной регрессии
- `linear_regression_slope` — Наклон тренда
- `volatility_ratio` — Short/long vol ratio
- `hurst_exponent` — Hurst показатель (mean reversion vs trending)

## 🧪 Тестирование

```bash
# Все тесты
pytest tests/ -v

# Только технические индикаторы
pytest tests/test_technical_indicators.py -v

# Только price features
pytest tests/test_price_features.py -v

# Полный pipeline
pytest tests/test_feature_pipeline.py -v
```

## 📓 Демо-ноутбук

Запустите демонстрационный ноутбук с визуализацией:

```bash
jupyter lab notebooks/exploratory/02_feature_engineering_demo.ipynb
```

Ноутбук включает:
- Загрузку данных AAPL за 5 лет
- Расчёт всех 50+ фич
- Визуализацию (price + BB, RSI, MACD, volatility, ADX)
- Корреляционную матрицу
- Feature importance (корреляция с forward returns)
- Экспорт результатов в parquet/csv

## 🔧 API Reference

### FeatureEngine

```python
engine = FeatureEngine(cache_dir='data/processed/features', version='1.0.0')
```

**Методы:**
- `register_feature(name, func, group, description, dependencies, params)` — регистрация фичи
- `compute_feature(name, df, use_cache=True)` — расчёт одной фичи
- `compute_all(df, groups=None, use_cache=True)` — расчёт всех/выбранных групп
- `get_feature_metadata(name=None)` — метаданные фичи/всех фичей
- `get_summary()` — статистика по зарегистрированным фичам
- `clear_cache(feature_name=None)` — очистка кеша

### Функции технических индикаторов

Все функции принимают `pd.DataFrame` с OHLCV и возвращают `pd.Series` или `pd.DataFrame`:

```python
from src.features import technical_indicators as tech

rsi_values = tech.rsi(df, period=14)
macd_df = tech.macd(df, fast=12, slow=26, signal=9)
bb_df = tech.bollinger_bands(df, period=20, std_dev=2.0)
```

## ✅ Метрика успеха достигнута

**Задача:** Возможность сгенерировать 50+ фичей для любого инструмента за последние 5 лет.

**Результат:** ✅ Реализовано 50+ фич с:
- Автоматической регистрацией
- Версионированием (hash функции + параметров)
- Кешированием для скорости
- Пакетным расчётом
- Полным покрытием unit-тестами
- Демо-ноутбуком с визуализацией

## 🎯 Следующие шаги (Часть 4)

1. Реализация реальных макро-фичей через FRED/OpenBB API
2. Feature selection (PCA, correlation filtering, importance ranking)
3. ML pipeline: обучение моделей на готовых фичах
4. Backtesting framework
5. Автоматический feature update pipeline

---

**Примечание:** Все фичи работают с любым инструментом (акции, ETF, индексы), достаточно загрузить OHLCV данные через `DataManager`.
