# 🚀 Quick Start: Backtesting Engine

## Шаг 1: Проверка готовности

```bash
python check_backtesting_setup.py
```

Должны быть все галочки ✅. Если нет - следуйте инструкциям.

## Шаг 2: Установка зависимостей (если требуется)

```bash
pip install -r requirements.txt
```

Или для Jupyter Notebook:
```python
!pip install tqdm scikit-learn
```

## Шаг 3: Подготовка данных (если требуется)

Если файл `data/processed/aapl_with_features_5y.parquet` отсутствует, запустите:

```bash
jupyter lab notebooks/exploratory/02_feature_engineering.ipynb
```

И выполните все ячейки для создания фичей.

## Шаг 4: Запуск демо-ноутбука

```bash
jupyter lab notebooks/exploratory/03_backtesting_demo.ipynb
```

Или через VS Code:
1. Откройте файл `03_backtesting_demo.ipynb`
2. Выберите kernel `venv (Python 3.11.6)`
3. Нажмите `Run All` или выполните ячейки последовательно

## Шаг 5: Запуск тестов (опционально)

```bash
python test_backtesting.py
```

Все 6 тестов должны пройти успешно ✅

---

## 📚 Что дальше?

### Простое использование

```python
from src.backtesting import Backtester
import pandas as pd

# Ваши данные
df = pd.read_parquet('data/processed/aapl_with_features_5y.parquet')

# Простая стратегия
def my_strategy(bar, portfolio, context):
    # Покупаем когда RSI < 30
    if bar.get('rsi_14', 100) < 30:
        return 1  # BUY
    # Продаем когда RSI > 70
    if bar.get('rsi_14', 0) > 70:
        return -1  # SELL
    return 0  # HOLD

# Бэктест
bt = Backtester(
    data=df,
    strategy=my_strategy,
    initial_capital=100000,
    position_size=0.95
)

results = bt.run()
bt.print_summary()
bt.plot_results()
```

### ML Стратегия

```python
from src.backtesting import MLBacktester
from sklearn.ensemble import RandomForestClassifier

# Обучите модель
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ML бэктест
ml_bt = MLBacktester(
    data=test_df,
    model=model,
    feature_columns=['rsi_14', 'macd', 'atr_14'],
    prediction_threshold=0.55
)

results = ml_bt.run()
ml_bt.print_ml_summary()
```

### Walk-Forward Analysis

```python
from src.backtesting import WalkForwardAnalyzer

wf = WalkForwardAnalyzer(
    data=df,
    train_period=252,  # 1 год
    test_period=63,    # 3 месяца
    step_size=63
)

results = wf.run(
    strategy_func=my_strategy,
    initial_capital=100000
)

wf.print_summary()
wf.plot_results()
```

---

## 📖 Полная документация

- **Руководство**: `docs/BACKTESTING_GUIDE.md`
- **Примеры**: `notebooks/exploratory/03_backtesting_demo.ipynb`
- **API Reference**: `docs/BACKTESTING_GUIDE.md` (раздел API)

---

## ❓ Проблемы?

### ModuleNotFoundError: No module named 'tqdm'

```bash
pip install tqdm scikit-learn
```

### Нет файла с данными

Запустите:
```bash
jupyter lab notebooks/exploratory/02_feature_engineering.ipynb
```

### Ошибка импорта backtesting модулей

Проверьте что вы в корне проекта:
```bash
cd /path/to/market-pricing-prj
python check_backtesting_setup.py
```

---

## 📞 Контакты

Если проблемы не решаются - создайте issue в репозитории.
