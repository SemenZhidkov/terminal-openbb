# 🚀 Как запустить Feature Engineering Notebook

## Вариант 1: Через скрипт (рекомендуется)

Просто запустите из корня проекта:

```bash
./start_jupyter.sh
```

Этот скрипт автоматически:
- Перейдёт в правильную директорию
- Запустит Jupyter Lab
- Откроет нужный ноутбук

## Вариант 2: Вручную

1. **Откройте терминал в корне проекта:**
   ```bash
   cd /Users/semenzidkov/market-pricing-prj
   ```

2. **Запустите Jupyter Lab:**
   ```bash
   jupyter lab notebooks/exploratory/02_feature_engineering_demo.ipynb
   ```

3. **В открывшемся ноутбуке:**
   - Выберите `Kernel → Restart Kernel`
   - Запустите `Cell → Run All` или выполняйте ячейки по порядку (Shift+Enter)

## ⚠️ Важно!

- **НЕ запускайте** Jupyter из директории `notebooks/exploratory/` - относительные пути не будут работать
- **Всегда запускайте** из корня проекта: `/Users/semenzidkov/market-pricing-prj`
- Если видите `FileNotFoundError: config/config.yaml` - вы в неправильной директории

## 🐛 Решение проблем

### Ошибка: `FileNotFoundError: config/config.yaml`

**Причина:** Jupyter запущен не из корня проекта.

**Решение:**
1. Закройте Jupyter Lab (Ctrl+C в терминале)
2. Убедитесь что вы в корне: `pwd` должен показать `/Users/semenzidkov/market-pricing-prj`
3. Запустите снова: `jupyter lab notebooks/exploratory/02_feature_engineering_demo.ipynb`

### Ошибка: `NameError: name 'df' is not defined`

**Причина:** Ячейки запущены не по порядку или с ошибками.

**Решение:**
1. `Kernel → Restart Kernel`
2. `Cell → Run All`

### Ошибка: `ModuleNotFoundError: No module named 'src'`

**Причина:** Ячейка с импортами (ячейка 2) не выполнена успешно.

**Решение:**
1. Проверьте что ячейка 2 выполнилась и показывает: `✓ Добавлен путь: /Users/semenzidkov/market-pricing-prj`
2. Если нет - перезапустите kernel и выполните ячейку 2 снова

## 📊 Ожидаемый результат

После успешного запуска всех ячеек вы увидите:

- ✅ **56 зарегистрированных фич** (22 technical + 24 price_based + 10 rolling)
- ✅ **~1260 строк данных AAPL** за 5 лет
- ✅ **Расчёт всех фич** занимает 20-30 секунд в первый раз
- ✅ **Красивые графики**: Bollinger Bands, RSI, MACD, Volatility, ADX
- ✅ **Корреляционная матрица** фич
- ✅ **Feature Importance** анализ
- ✅ **Экспорт** в `data/processed/aapl_with_features_5y.parquet` и `.csv`

## 🎯 Что дальше?

После успешного выполнения ноутбука у вас будет:

1. **Готовый датасет** с 50+ фичами для ML-моделей
2. **Понимание** какие фичи коррелируют с будущей доходностью
3. **Кеш фич** для быстрого повторного использования
4. **Визуализации** для анализа и презентаций

Можно приступать к:
- Feature selection и dimensionality reduction
- Обучению ML-моделей
- Бэктестингу торговых стратегий
