# Backtesting Guide: Полное руководство

## Содержание
1. [Введение](#введение)
2. [Архитектура](#архитектура)
3. [Portfolio Management](#portfolio-management)
4. [Execution Engine](#execution-engine)
5. [Performance Metrics](#performance-metrics)
6. [Simple Strategy Backtesting](#simple-strategy-backtesting)
7. [ML Strategy Backtesting](#ml-strategy-backtesting)
8. [Walk-Forward Analysis](#walk-forward-analysis)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)

---

## Введение

Система бэктестинга предназначена для:
- Тестирования торговых стратегий на исторических данных
- Оценки производительности с учетом реальных торговых издержек
- Валидации ML-моделей в торговом контексте
- Проверки устойчивости стратегий через walk-forward analysis

### Ключевые особенности:
- ✅ Event-driven архитектура (реалистичная симуляция)
- ✅ Учет комиссий, slippage, margin
- ✅ Market, Limit, Stop orders
- ✅ 15+ метрик производительности
- ✅ ML integration
- ✅ Walk-forward testing
- ✅ Подробная визуализация

---

## Архитектура

```
┌─────────────────┐
│   Data Feed     │  OHLCV + Features
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Strategy      │  Signal Generation
│   Function      │  (User-defined)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execution      │  Order Processing
│  Engine         │  Market/Limit/Stop
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Portfolio     │  Position Management
│                 │  P&L Tracking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Performance    │  Metrics Calculation
│  Analyzer       │  Sharpe, DD, etc.
└─────────────────┘
```

---

## Portfolio Management

### Основной класс `Portfolio`

Управляет капиталом, позициями и P&L.

#### Инициализация:

```python
from src.backtesting import Portfolio

portfolio = Portfolio(
    initial_capital=100000.0,
    commission_rate=0.001,      # 0.1% per trade
    slippage_rate=0.0005,       # 0.05% slippage
    margin_requirement=1.0,     # 1.0 = no leverage
    name="MyPortfolio"
)
```

#### Основные методы:

```python
# Открыть/увеличить позицию
trade = portfolio.open_position(
    symbol='AAPL',
    side=PositionSide.LONG,
    quantity=100,
    price=150.50,
    timestamp=datetime.now()
)

# Закрыть позицию
trade = portfolio.close_position(
    symbol='AAPL',
    price=155.25,
    timestamp=datetime.now()
)

# Обновить цены
portfolio.update_prices({'AAPL': 152.00}, timestamp)

# Получить equity
current_equity = portfolio.equity

# Сводка
summary = portfolio.get_summary()
print(f"Total Return: {summary['total_return_pct']:.2f}%")
print(f"Win Rate: {summary['win_rate']*100:.1f}%")
print(f"Max Drawdown: {summary['max_drawdown']*100:.2f}%")
```

#### Ключевые атрибуты:

- `cash`: Доступные средства
- `equity`: Общая стоимость портфеля (cash + positions)
- `positions`: Dict открытых позиций
- `closed_trades`: List закрытых сделок
- `equity_history`: История equity по времени

---

## Execution Engine

### Класс `ExecutionEngine`

Обрабатывает ордера и исполнение сделок.

#### Типы ордеров:

**Market Order:**
```python
order = execution.submit_order(
    symbol='AAPL',
    side=PositionSide.LONG,
    quantity=100,
    order_type=OrderType.MARKET
)
```
Исполняется по текущей цене (open/close бара в зависимости от настройки).

**Limit Order:**
```python
order = execution.submit_order(
    symbol='AAPL',
    side=PositionSide.LONG,
    quantity=100,
    order_type=OrderType.LIMIT,
    price=150.00  # Купить не дороже $150
)
```
Исполняется, если цена достигает лимита.

**Stop Order:**
```python
order = execution.submit_order(
    symbol='AAPL',
    side=PositionSide.SHORT,
    quantity=100,
    order_type=OrderType.STOP,
    stop_price=145.00  # Продать при падении до $145
)
```
Триггерится при достижении стоп-цены.

**Stop-Limit Order:**
```python
order = execution.submit_order(
    symbol='AAPL',
    side=PositionSide.LONG,
    quantity=100,
    order_type=OrderType.STOP_LIMIT,
    stop_price=152.00,  # Триггер
    price=152.50        # Лимит после триггера
)
```

#### Обработка ордеров:

```python
# На каждом баре
execution.process_orders(bar, timestamp)

# Отмена ордеров
execution.cancel_order(order)
execution.cancel_all_orders(symbol='AAPL')

# Статистика
summary = execution.get_order_summary()
```

---

## Performance Metrics

### Класс `PerformanceAnalyzer`

Рассчитывает метрики эффективности.

#### Доступные метрики:

**Risk-Adjusted Returns:**
- **Sharpe Ratio**: (Return - RiskFreeRate) / StdDev
- **Sortino Ratio**: Учитывает только downside volatility
- **Calmar Ratio**: Annual Return / Max Drawdown

**Drawdown Metrics:**
- **Max Drawdown**: Максимальное падение от пика
- **Max DD Duration**: Длительность максимального DD
- **Recovery Factor**: Net Profit / Max DD
- **Ulcer Index**: Мера downside risk

**Trade Statistics:**
- **Win Rate**: % прибыльных сделок
- **Profit Factor**: Gross Profit / Gross Loss
- **Average Trade**: Средний P&L на сделку
- **Avg Win/Loss**: Средняя прибыль/убыток
- **Max Consecutive Wins/Losses**

#### Использование:

```python
from src.backtesting import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(portfolio, risk_free_rate=0.02)

# Отдельные метрики
sharpe = analyzer.calculate_sharpe_ratio()
sortino = analyzer.calculate_sortino_ratio()
dd_metrics = analyzer.calculate_max_drawdown()

# Полный отчет
report = analyzer.generate_report()

# Вывод в консоль
analyzer.print_report()

# Monthly returns
monthly_returns = analyzer.calculate_monthly_returns()
```

---

## Simple Strategy Backtesting

### Создание стратегии

Стратегия = функция с сигнатурой:
```python
def strategy(bar: pd.Series, portfolio: Portfolio, context: dict) -> int:
    """
    Args:
        bar: Текущий бар с OHLCV и фичами
        portfolio: Портфель (для проверки позиций)
        context: Dict для хранения состояния между барами
    
    Returns:
        1: BUY signal
        -1: SELL signal
        0: HOLD
    """
    pass
```

### Пример 1: SMA Crossover

```python
def sma_crossover_strategy(bar, portfolio, context):
    """
    Golden Cross: SMA50 пересекает SMA200 снизу вверх -> BUY
    Death Cross: SMA50 пересекает SMA200 сверху вниз -> SELL
    """
    symbol = bar.get('symbol', 'UNKNOWN')
    
    # Проверяем наличие индикаторов
    if 'sma_50' not in bar or 'sma_200' not in bar:
        return 0
    
    if pd.isna(bar['sma_50']) or pd.isna(bar['sma_200']):
        return 0
    
    # Используем context для отслеживания предыдущего состояния
    prev_diff = context.get('prev_diff', 0)
    current_diff = bar['sma_50'] - bar['sma_200']
    context['prev_diff'] = current_diff
    
    # Golden Cross
    if prev_diff <= 0 and current_diff > 0:
        if not portfolio.has_position(symbol):
            return 1  # BUY
    
    # Death Cross
    if prev_diff >= 0 and current_diff < 0:
        if portfolio.has_position(symbol):
            return -1  # SELL
    
    return 0
```

### Запуск бэктеста:

```python
from src.backtesting import Backtester

# Инициализация
backtester = Backtester(
    data=df_with_features,           # DataFrame с OHLCV и фичами
    strategy=sma_crossover_strategy,  # Ваша стратегия
    initial_capital=100000,
    commission_rate=0.001,            # 0.1%
    slippage_rate=0.0005,             # 0.05%
    position_size=0.95,               # 95% капитала на сделку
    name="SMA_Crossover"
)

# Запуск
results = backtester.run(verbose=True)

# Результаты
backtester.print_summary()
backtester.plot_results()

# Доступ к данным
equity_curve = results['equity_curve']
trades = results['trades']
sharpe_ratio = results['sharpe_ratio']
max_dd = results['max_drawdown']
```

### Пример 2: RSI Mean Reversion

```python
def rsi_strategy(bar, portfolio, context):
    """
    Oversold (RSI < 30): BUY
    Overbought (RSI > 70): SELL
    """
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if 'rsi_14' not in bar or pd.isna(bar['rsi_14']):
        return 0
    
    rsi = bar['rsi_14']
    
    if rsi < 30 and not portfolio.has_position(symbol):
        return 1  # Oversold - BUY
    
    if rsi > 70 and portfolio.has_position(symbol):
        return -1  # Overbought - SELL
    
    return 0
```

### Пример 3: Multi-Indicator Strategy

```python
def multi_indicator_strategy(bar, portfolio, context):
    """
    Комбинация нескольких сигналов
    """
    symbol = bar.get('symbol', 'UNKNOWN')
    
    # Требуемые фичи
    required = ['rsi_14', 'macd', 'macd_signal', 'adx']
    if not all(f in bar and not pd.isna(bar[f]) for f in required):
        return 0
    
    # Bullish conditions
    bullish = (
        bar['rsi_14'] < 40 and                    # RSI oversold
        bar['macd'] > bar['macd_signal'] and      # MACD bullish
        bar['adx'] > 25                           # Strong trend
    )
    
    # Bearish conditions
    bearish = (
        bar['rsi_14'] > 60 or
        bar['macd'] < bar['macd_signal']
    )
    
    if bullish and not portfolio.has_position(symbol):
        return 1
    
    if bearish and portfolio.has_position(symbol):
        return -1
    
    return 0
```

---

## ML Strategy Backtesting

### Класс `MLBacktester`

Специализированный бэктестер для ML-моделей.

### Подготовка данных:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Выбираем фичи
feature_cols = [
    'rsi_14', 'macd', 'atr_14', 'adx',
    'returns_1d', 'realized_vol_20', 'rolling_zscore_20'
]

# Создаем таргет (например, будущая доходность > 0)
df['target'] = (df['forward_returns_1d'] > 0).astype(int)

# Train/Test split
train_df = df.iloc[:int(len(df)*0.7)]
test_df = df.iloc[int(len(df)*0.7):]

# Обучение модели
X_train = train_df[feature_cols].dropna()
y_train = train_df.loc[X_train.index, 'target']

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    random_state=42
)
model.fit(X_train, y_train)
```

### Запуск ML бэктеста:

```python
from src.backtesting import MLBacktester

ml_bt = MLBacktester(
    data=test_df,
    model=model,
    feature_columns=feature_cols,
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.0005,
    position_size=0.95,
    prediction_threshold=0.55,  # Требуем 55% уверенности
    name="ML_RandomForest"
)

# Запуск
ml_results = ml_bt.run(verbose=True)

# ML-специфичные метрики
ml_bt.print_summary()
ml_bt.print_ml_summary()

# Визуализация
ml_bt.plot_results()
ml_bt.plot_predictions()

# Доступ к предсказаниям
predictions = ml_results['predictions']
ml_metrics = ml_results['ml_metrics']
```

### ML Metrics:

- `total_predictions`: Количество предсказаний
- `avg_prediction`: Средняя вероятность
- `prediction_std`: Стандартное отклонение
- `prediction_pnl_correlation`: Корреляция предсказаний с P&L

---

## Walk-Forward Analysis

### Класс `WalkForwardAnalyzer`

Тестирование устойчивости на rolling windows.

### Концепция:

```
Window 1: [====Train====][Test]
Window 2:      [====Train====][Test]
Window 3:           [====Train====][Test]
```

- **Train**: Обучение/оптимизация параметров
- **Test**: Out-of-sample тестирование
- **Step**: Сдвиг окна

### Использование с простой стратегией:

```python
from src.backtesting import WalkForwardAnalyzer

wf_analyzer = WalkForwardAnalyzer(
    data=df,
    train_period=252,   # 1 год обучения
    test_period=63,     # 3 месяца теста
    step_size=63,       # Сдвиг на 3 месяца
    anchored=False,     # False = rolling, True = anchored
    name="WF_SMA"
)

wf_results = wf_analyzer.run(
    strategy_func=sma_crossover_strategy,
    initial_capital=100000,
    commission_rate=0.001,
    slippage_rate=0.0005,
    position_size=0.95,
    verbose=True
)

# Результаты
wf_analyzer.print_summary()
wf_analyzer.plot_results()
```

### ML Walk-Forward:

```python
wf_ml_analyzer = WalkForwardAnalyzer(
    data=df_ml,
    train_period=504,   # 2 года
    test_period=126,    # 6 месяцев
    step_size=126,
    anchored=False,
    name="WF_ML"
)

wf_ml_results = wf_ml_analyzer.run_ml(
    model_class=RandomForestClassifier,
    feature_columns=feature_cols,
    target_column='target',
    model_params={
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 50,
        'random_state': 42
    },
    initial_capital=100000,
    prediction_threshold=0.55,
    verbose=True
)

wf_ml_analyzer.print_summary()
wf_ml_analyzer.plot_results()
```

### Walk-Forward Metrics:

- `avg_return`: Средняя доходность по окнам
- `avg_sharpe`: Средний Sharpe Ratio
- `return_std`: Стабильность доходности
- `positive_windows`: Количество прибыльных окон
- `stability_score`: Общий скор устойчивости (0-100)

---

## Best Practices

### 1. Реалистичные издержки:

```python
# Консервативные значения
backtester = Backtester(
    commission_rate=0.001,   # 0.1% (или больше для акций)
    slippage_rate=0.001,     # 0.1% (зависит от ликвидности)
    ...
)
```

### 2. Position Sizing:

```python
# Не используйте весь капитал
position_size=0.90  # 90% максимум, оставляйте резерв
```

### 3. Data Leakage Prevention:

```python
# Не используйте будущую информацию!
# ❌ Плохо: df['signal'] = df['close'].shift(-1)
# ✅ Хорошо: df['signal'] = df['close'].shift(1)

# Убедитесь что forward_returns не доступны в момент T
```

### 4. Walk-Forward обязателен:

```python
# Не доверяйте single backtest
# Всегда проверяйте устойчивость через WF
```

### 5. Оптимизация параметров:

```python
# Избегайте overfitting
# Используйте простые стратегии
# Валидируйте на out-of-sample данных
```

### 6. Метрики интерпретации:

- **Sharpe > 1.0**: Хорошо
- **Sharpe > 2.0**: Отлично (подозрительно высоко)
- **Max DD < 20%**: Приемлемо
- **Win Rate ≈ 50-60%**: Нормально для trend-following
- **Profit Factor > 1.5**: Хорошо

---

## API Reference

### Core Classes:

#### `Portfolio`
```python
Portfolio(
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    margin_requirement: float = 1.0,
    name: str = "Portfolio"
)
```

**Methods:**
- `open_position(symbol, side, quantity, price, timestamp)` → Trade
- `close_position(symbol, price, timestamp, quantity=None)` → Trade
- `update_prices(prices, timestamp)` → None
- `get_summary()` → Dict
- `get_equity_curve()` → DataFrame
- `get_trades_df()` → DataFrame

#### `ExecutionEngine`
```python
ExecutionEngine(
    portfolio: Portfolio,
    use_bid_ask: bool = False,
    fill_at: str = 'close'
)
```

**Methods:**
- `submit_order(symbol, side, quantity, order_type, price, stop_price, timestamp)` → Order
- `process_orders(bar, timestamp)` → None
- `cancel_order(order)` → None
- `cancel_all_orders(symbol=None)` → None

#### `PerformanceAnalyzer`
```python
PerformanceAnalyzer(
    portfolio: Portfolio,
    risk_free_rate: float = 0.02
)
```

**Methods:**
- `calculate_sharpe_ratio()` → float
- `calculate_sortino_ratio()` → float
- `calculate_calmar_ratio()` → float
- `calculate_max_drawdown()` → Dict
- `calculate_win_rate()` → float
- `calculate_profit_factor()` → float
- `generate_report()` → Dict
- `print_report()` → None

#### `Backtester`
```python
Backtester(
    data: DataFrame,
    strategy: Callable,
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
    name: str = "Backtest"
)
```

**Methods:**
- `run(verbose=True)` → Dict
- `get_results()` → Dict
- `plot_results(figsize)` → None
- `print_summary()` → None

#### `MLBacktester`
```python
MLBacktester(
    data: DataFrame,
    model: Any,
    feature_columns: List[str],
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
    prediction_threshold: float = 0.5,
    name: str = "ML_Backtest"
)
```

**Methods:**
- `run(verbose=True)` → Dict
- `print_ml_summary()` → None
- `plot_predictions(figsize)` → None

#### `WalkForwardAnalyzer`
```python
WalkForwardAnalyzer(
    data: DataFrame,
    train_period: int,
    test_period: int,
    optimization_func: Callable = None,
    step_size: int = None,
    anchored: bool = False,
    name: str = "WalkForward"
)
```

**Methods:**
- `run(strategy_func, ...)` → Dict
- `run_ml(model_class, feature_columns, target_column, ...)` → Dict
- `print_summary()` → None
- `plot_results(figsize)` → None

---

## Примеры

Полные рабочие примеры:
- **Simple strategies**: `notebooks/exploratory/03_backtesting_demo.ipynb`
- **Feature engineering**: `notebooks/exploratory/02_feature_engineering_demo.ipynb`
- **Quick test**: `test_backtesting.py`

---

## Дополнительные ресурсы

- [README.md](../readme.md) - Общее описание проекта
- [FEATURE_ENGINEERING_GUIDE.md](FEATURE_ENGINEERING_GUIDE.md) - Руководство по фичам
- [API Documentation](../src/backtesting/) - Исходный код с docstrings

---

**Версия:** 1.0.0  
**Дата обновления:** 5 ноября 2025
