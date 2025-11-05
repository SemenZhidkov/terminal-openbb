## OpenBB Market Data Pipeline

> Пайплайн для загрузки, кеширования и валидации рыночных данных на базе OpenBB с резервным источником yfinance. Подходит для ежедневных обновлений, исследовательских ноутбуков и последующей обработки признаков/моделей.

---

### Возможности

- Загрузка ценовых рядов акций, ETF и индексов через OpenBB; fallback на yfinance при сбоях
- Кеширование данных в `parquet`/`csv` с контролем актуальности
- Нормализация колонок (open/high/low/close/volume, date)
- Автоматическая валидация качества данных (размер, пропуски, консистентность цен, даты, волатильность)
- Пакетная загрузка по спискам активов и таймфреймам из конфигурации
- Ежедневный апдейтер с отчётом и логированием

---

## Архитектура и ключевые модули

### Модуль загрузки данных
- `src/data/data_manager.py` — ядро пайплайна: конфиг, ретраи, загрузка, кеш, нормализация, пакетные задачи, метрики кеша
- `src/data/data_validator.py` — проверка качества таймсерий (обязательные колонки, пропуски, high/low/open/close, объём, даты, дисперсия)
- `src/data/data_loader.py` — простой пример загрузчика через OpenBB и сохранения в `data/raw`
- `scripts/update_data.py` — ежедневный апдейтер: пакетная загрузка, валидация, отчёт в `data/backup`, логирование в `logs/data_update.log`
- `config/config.yaml` — активы, таймфреймы, параметры кеша/валидации/ретраев, формат файлов

### Модуль расчёта фич (Feature Engineering)
- `src/features/feature_engine.py` — ядро системы фич: регистрация, версионирование, кеширование, пакетный расчёт
- `src/features/technical_indicators.py` — 20+ технических индикаторов (RSI, MACD, Bollinger Bands, ADX, Stochastic, etc.)
- `src/features/price_features.py` — price-based фичи: доходности, волатильность (realized/Parkinson/Garman-Klass), momentum, Z-score, drawdown
- `src/features/rolling_features.py` — rolling window фичи: SMA/EMA кроссоверы, автокорреляция, trend strength, Hurst exponent
- `src/features/macro_features.py` — заглушки для макро-индикаторов (VIX, rates, spreads, commodities)
- `src/features/feature_registry.py` — автоматическая регистрация всех 50+ фич

**Возможности Feature Engine:**
- 50+ готовых фич (технические индикаторы, price patterns, rolling stats)
- Версионирование фич (hash функции + параметров)
- Автоматическое кеширование результатов
- Пакетный расчёт по группам (technical/price_based/rolling)
- Метаданные и audit log

Диаграмма потока:

1) Конфиг → 2) Загрузка (OpenBB → yfinance) → 3) Нормализация → 4) Кеширование → 5) Валидация → **6) Feature Engineering** → 7) ML-Ready Dataset

---

## Структура репозитория

```
.
├── config/
│   └── config.yaml          # настройки пайплайна (активы, таймфреймы, формат файлов и пр.)
├── data/
│   ├── raw/                 # сырые выгрузки/образцы
│   ├── processed/           # кеш нормализованных рядов (parquet/csv)
│   ├── external/            # внешние данные (если нужны)
│   └── backup/              # отчёты апдейта
├── notebooks/
│   ├── exploratory/         # исследовательские ноутбуки
│   └── reports/             # отчёты
├── scripts/
│   └── update_data.py       # ежедневное обновление и валидация
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── data_manager.py
│   │   └── data_validator.py
│   ├── features/
│   │   ├── feature_engine.py       # ядро Feature Engine
│   │   ├── technical_indicators.py # 20+ технических индикаторов
│   │   ├── price_features.py       # price-based фичи
│   │   ├── rolling_features.py     # rolling window фичи
│   │   ├── macro_features.py       # макро-индикаторы (заглушки)
│   │   └── feature_registry.py     # регистратор всех фич
│   ├── models/
│   └── visualization/
├── tests/
│   ├── test_technical_indicators.py
│   ├── test_price_features.py
│   └── test_feature_pipeline.py
├── requirements.txt
├── setup.py
├── test_data_pipeline.py    # интеграционный скрипт-тест (использует сеть)
└── test_openbb.py           # проверка OpenBB (использует сеть)
```

---

## Требования

- Python 3.10+
- Пакеты из `requirements.txt` (OpenBB Platform 4+, pandas, numpy, yfinance, pyarrow и др.)
- Для провайдеров OpenBB рекомендуется задать API-ключи (см. ниже)

---

## Установка

```zsh
# 1) создать и активировать виртуальное окружение
python -m venv .venv
source .venv/bin/activate

# 2) установить зависимости
pip install -r requirements.txt
```

Если используете Jupyter, установите кернел:
```zsh
python -m ipykernel install --user --name market-pricing-prj
```

---

## Конфигурация

### 1) Переменные окружения (.env)

Модуль `DataManager` читает ключи провайдеров из окружения и логинится в OpenBB:
- `POLYGON_API_KEY`
- `ALPHA_VANTAGE_API_KEY`

Создайте файл `.env` в корне и добавьте туда ваши ключи:
```dotenv
POLYGON_API_KEY=...
ALPHA_VANTAGE_API_KEY=...
```

Без ключей загрузка может работать через yfinance (fallback), но лучше настроить провайдеры для качества/лимитов.

### 2) Настройки пайплайна (`config/config.yaml`)

- Директории кеша/сырых данных/бэкапов
- Формат файлов (`parquet` или `csv`)
- Списки активов: акции, ETF, индексы
- Таймфреймы: daily/hourly/weekly
- Ретраи и бэкофф
- Пороговые значения для валидации (минимальное число точек, допустимый процент пропусков и т.п.)

---

## Быстрый старт

### Загрузка данных для одного тикера
```zsh
python -c "from src.data.data_manager import DataManager; dm=DataManager(); df=dm.get_stock_data('AAPL','1d','6mo'); print(df.tail())"
```

### Расчёт фич для инструмента
```zsh
python -c "
from src.data.data_manager import DataManager
from src.features.feature_engine import FeatureEngine
from src.features.feature_registry import register_all_features

# Загрузка данных
dm = DataManager()
df = dm.get_stock_data('AAPL', '1d', '5y')

# Инициализация Feature Engine
engine = FeatureEngine()
register_all_features(engine)

# Расчёт всех 50+ фич
df_with_features = engine.compute_all(df)
print(f'Добавлено {len(df_with_features.columns) - len(df.columns)} фич')
print(df_with_features.tail())
"
```

### Пакетная загрузка по спискам из конфига
```zsh
python scripts/update_data.py
```
Результаты:
- Кешированные файлы: `data/processed/*.parquet|csv`
- Отчёт: `data/backup/update_report_YYYYMMDD_HHMMSS.json`
- Логи: `logs/data_update.log`

### Демо-ноутбук с визуализацией фич
```zsh
jupyter lab notebooks/exploratory/02_feature_engineering_demo.ipynb
```

### Пример информации о кешe
```zsh
python -c "from src.data.data_manager import DataManager; dm=DataManager(); print(dm.get_cache_info())"
```

---

## Валидация данных
Встроенный `DataValidator` проверяет:
- Минимальный объём данных, наличие обязательных колонок
- Уровень пропусков по колонкам
- Консистентность цен (high ≥ low, open/close ∈ [low, high])
- Аномалии по объёму (нулевые/отрицательные)
- Целостность дат (дубликаты, большие разрывы)
- Дисперсию доходностей (аномально низкая волатильность)

Результаты валидации доступны в отчёте апдейтера и могут логироваться.

---

## Тестирование

В репозитории есть два интеграционных скрипта, которые используют сеть:
- `test_openbb.py` — базовая проверка OpenBB
- `test_data_pipeline.py` — прогон пайплайна (загрузка → кеш → валидация)

Для запуска unit-тестов Feature Engine:
```zsh
pip install pytest
pytest tests/ -v
```

Unit-тесты покрывают:
- **Технические индикаторы** (`tests/test_technical_indicators.py`): корректность диапазонов, структура выходных данных, edge cases
- **Price-based фичи** (`tests/test_price_features.py`): returns, volatility, momentum, drawdown
- **Feature Pipeline** (`tests/test_feature_pipeline.py`): регистрация, расчёт, кеширование, версионирование

Для быстрых "smoke"-проверок импорта без сети:
```zsh
python -m pytest -q tests/test_imports.py
```

Рекомендации:
- Разделять unit-тесты (без сети, с моками) и интеграционные (с сетью)
- Добавить dev-зависимости: `pytest`, `ruff/black`, `mypy` (по желанию)

---

## Логи и отчёты

- Логи: `logs/data_update.log` (создаётся автоматически)
- Отчёты обновления: `data/backup/update_report_*.json`
	- В отчёте: число успешных/неуспешных задач, доля успеха, краткая статистика валидации, список недавних кеш-файлов

Скрипт также удаляет старые отчёты (по умолчанию хранит последние ~30 дней).

---

## Советы по эксплуатации

- Формат данных: используйте `parquet` для скорости и экономии места
- Актуальность кеша: по умолчанию 24 часа (переменная окружения `DATA_UPDATE_INTERVAL_HOURS`)
- Таймфреймы Intraday могут иметь особенности по торговым сессиям и TZ — учитывайте при анализе
- Провайдеры OpenBB: задайте ключи, чтобы улучшить стабильность и лимиты
- Fallback yfinance: работает без ключей, но покрытия/качество может отличаться

---

## Типичные проблемы и решения

- Нет пакета `openbb` или провайдеров → установите зависимости, задайте ключи в `.env`
- Ошибки парсинга `parquet` → убедитесь, что установлен `pyarrow`
- Мало данных/пустой датафрейм → проверьте тикер, таймфрейм, доступность данных у провайдера
- Rate limits → увеличьте backoff/ретраи в конфиге, добавьте ключи провайдеров
- Ошибки импорта модулей → запускайте из корня репозитория или добавьте корень в `PYTHONPATH`

---

## Roadmap (идеи на будущее)

- ~~Unit-тесты без сети с моками источников~~ ✅ Реализовано
- ~~Система расчёта и версионирования фич~~ ✅ 50+ фич готово
- Линтинг/форматирование (ruff/black), типизация (mypy)
- Консольные entry points для апдейтера
- Реализация макро-фичей через FRED/OpenBB API
- Feature selection и dimensionality reduction
- Расширенные отчёты по качеству данных и визуализация
- Интеграция с CI (GitHub Actions)
- ML pipeline: обучение и бэктестинг моделей

---

## Feature Engineering Capabilities

### 50+ готовых фич включают:

**Технические индикаторы (20+):**
- Momentum: RSI, MACD, Stochastic, CCI, Williams %R, MFI, ROC
- Volatility: ATR, Bollinger Bands, Keltner Channel, Donchian Channel
- Trend: ADX, Aroon, EMA, SMA, VWAP
- Volume: OBV, A/D Line, CMF
- Advanced: Parabolic SAR, Ichimoku Cloud

**Price-Based Features (20+):**
- Returns: simple, log, forward, cumulative
- Volatility: realized, Parkinson, Garman-Klass, Yang-Zhang
- Volatility regimes (Low/Normal/High)
- Momentum: multi-period, acceleration, Z-score
- Distance from high/low
- Candle patterns: body size, shadows, gaps
- Statistical: rolling Sharpe/Sortino, skewness, kurtosis, drawdown

**Rolling Window Features (10+):**
- Multiple timeframe SMAs/EMAs
- EMA crossovers (Golden/Death Cross)
- Price vs MA deviations
- Autocorrelation (mean reversion detection)
- Trend strength (R²), linear regression slope/angle
- Volatility ratios
- Hurst exponent (trending vs mean-reverting)

**Макро-индикаторы (заглушки):**
- VIX, Treasury yields, yield curve slope
- DXY (Dollar Index), commodity prices
- Sentiment indicators

### Использование Feature Engine:

```python
from src.features.feature_engine import FeatureEngine
from src.features.feature_registry import register_all_features

# Инициализация
engine = FeatureEngine(cache_dir='data/processed/features')
register_all_features(engine)

# Получение статистики
summary = engine.get_summary()
print(f"Зарегистрировано фич: {summary['total_features']}")

# Расчёт всех фич
df_with_features = engine.compute_all(df_ohlcv)

# Расчёт только технических индикаторов
df_technical = engine.compute_all(df_ohlcv, groups=['technical'])

# Расчёт одной фичи
df_with_rsi = engine.compute_feature('rsi_14', df_ohlcv)

# Метаданные фичи
meta = engine.get_feature_metadata('rsi_14')
```

Подробный пример с визуализацией: `notebooks/exploratory/02_feature_engineering_demo.ipynb`

---

## Backtesting Engine

### Профессиональный движок бэктестинга стратегий

**Ключевые компоненты:**

**Portfolio Management:**
- Управление капиталом и позициями
- Учет комиссий и slippage
- Margin requirements
- Реализованный и нереализованный P&L
- История equity и drawdown

**Execution Engine:**
- Market, Limit, Stop, Stop-Limit orders
- Автоматическая проверка капитала
- Realistic order filling
- Slippage modeling

**Performance Metrics:**
- Risk-adjusted: Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Drawdown metrics: Max DD, DD Duration, Recovery Factor, Ulcer Index
- Trade statistics: Win Rate, Profit Factor, Avg Trade, Consecutive wins/losses
- Annual returns, Monthly returns heatmap

**Advanced Features:**
- ML Backtester: автоматический predict на каждом баре
- Walk-Forward Analysis: проверка устойчивости на rolling windows
- Multi-timeframe support
- Подробное логирование и визуализация

### Использование Backtester:

```python
from src.backtesting import Backtester, MLBacktester, WalkForwardAnalyzer

# Простая стратегия
def sma_crossover_strategy(bar, portfolio, context):
    if bar['sma_50'] > bar['sma_200'] and not portfolio.has_position('AAPL'):
        return 1  # BUY
    elif bar['sma_50'] < bar['sma_200'] and portfolio.has_position('AAPL'):
        return -1  # SELL
    return 0  # HOLD

# Запуск бэктеста
backtester = Backtester(
    data=df_with_features,
    strategy=sma_crossover_strategy,
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.0005,
    position_size=0.95
)

results = backtester.run()
backtester.print_summary()
backtester.plot_results()

# ML Backtesting
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

ml_bt = MLBacktester(
    data=test_data,
    model=model,
    feature_columns=['rsi_14', 'macd', 'atr_14', 'adx'],
    prediction_threshold=0.55
)

ml_results = ml_bt.run()
ml_bt.print_ml_summary()

# Walk-Forward Analysis
wf_analyzer = WalkForwardAnalyzer(
    data=df,
    train_period=252,  # 1 год
    test_period=63,    # 3 месяца
    step_size=63
)

wf_results = wf_analyzer.run_ml(
    model_class=RandomForestClassifier,
    feature_columns=feature_cols,
    target_column='target'
)

wf_analyzer.print_summary()
wf_analyzer.plot_results()
```

**Примеры:**
- Простой backtest: `notebooks/exploratory/03_backtesting_demo.ipynb`
- Подробная документация: `docs/BACKTESTING_GUIDE.md`

---

