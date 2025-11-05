#!/usr/bin/env python3
"""
Quick test для Backtesting Engine
Проверка всех компонентов системы бэктестинга
"""

import sys
import os

# Добавляем корень проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_imports():
    """Тест импортов"""
    print("1. Проверка импортов backtesting модулей...")
    
    try:
        from src.backtesting import (
            Portfolio, ExecutionEngine, PerformanceAnalyzer,
            Backtester, MLBacktester, WalkForwardAnalyzer,
            Order, Trade, Position, OrderType, OrderStatus, PositionSide
        )
        print("   ✓ Все backtesting модули импортированы")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка импорта: {e}")
        return False


def test_portfolio():
    """Тест Portfolio класса"""
    print("\n2. Проверка Portfolio...")
    
    try:
        from src.backtesting import Portfolio, PositionSide
        
        portfolio = Portfolio(initial_capital=100000)
        
        # Открываем позицию
        trade = portfolio.open_position(
            symbol='TEST',
            side=PositionSide.LONG,
            quantity=100,
            price=100.0,
            timestamp=datetime.now()
        )
        
        assert portfolio.cash < 100000, "Cash должен уменьшиться"
        assert portfolio.has_position('TEST'), "Позиция должна быть открыта"
        
        # Обновляем цену
        portfolio.update_prices({'TEST': 105.0}, datetime.now())
        assert portfolio.equity > 100000, "Equity должна вырасти"
        
        # Закрываем позицию
        trade = portfolio.close_position('TEST', 105.0, datetime.now())
        assert not portfolio.has_position('TEST'), "Позиция должна быть закрыта"
        assert len(portfolio.closed_trades) == 1, "Должна быть 1 закрытая сделка"
        
        summary = portfolio.get_summary()
        print(f"   ✓ Portfolio работает")
        print(f"     Final Equity: ${summary['current_equity']:,.2f}")
        print(f"     Total Return: {summary['total_return_pct']:.2f}%")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка Portfolio: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_execution_engine():
    """Тест Execution Engine"""
    print("\n3. Проверка Execution Engine...")
    
    try:
        from src.backtesting import Portfolio, ExecutionEngine, PositionSide, OrderType
        
        portfolio = Portfolio(initial_capital=100000)
        execution = ExecutionEngine(portfolio, fill_at='close')
        
        # Создаем тестовый бар
        bar = pd.Series({
            'symbol': 'TEST',
            'open': 100.0,
            'high': 102.0,
            'low': 99.0,
            'close': 101.0,
            'volume': 1000000
        })
        
        # Подаем market order
        order = execution.submit_order(
            symbol='TEST',
            side=PositionSide.LONG,
            quantity=100,
            order_type=OrderType.MARKET,
            timestamp=datetime.now()
        )
        
        # Обрабатываем
        execution.process_orders(bar, datetime.now())
        
        assert len(execution.filled_orders) == 1, "Ордер должен быть исполнен"
        assert portfolio.has_position('TEST'), "Позиция должна быть открыта"
        
        print("   ✓ Execution Engine работает")
        print(f"     Filled orders: {len(execution.filled_orders)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка Execution Engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_analyzer():
    """Тест Performance Analyzer"""
    print("\n4. Проверка Performance Analyzer...")
    
    try:
        from src.backtesting import Portfolio, PerformanceAnalyzer, PositionSide
        
        portfolio = Portfolio(initial_capital=100000)
        
        # Симулируем несколько сделок
        timestamps = [datetime.now() + timedelta(days=i) for i in range(10)]
        prices = [100 + i*2 for i in range(10)]
        
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            if i % 2 == 0:  # Открываем
                portfolio.open_position('TEST', PositionSide.LONG, 100, price, ts)
            else:  # Закрываем
                portfolio.close_position('TEST', price, ts)
            
            portfolio.update_prices({'TEST': price}, ts)
        
        # Анализ
        analyzer = PerformanceAnalyzer(portfolio)
        
        sharpe = analyzer.calculate_sharpe_ratio()
        sortino = analyzer.calculate_sortino_ratio()
        max_dd = analyzer.calculate_max_drawdown()
        win_rate = analyzer.calculate_win_rate()
        
        print("   ✓ Performance Analyzer работает")
        print(f"     Sharpe Ratio: {sharpe:.3f}")
        print(f"     Sortino Ratio: {sortino:.3f}")
        print(f"     Max Drawdown: {max_dd['max_drawdown']*100:.2f}%")
        print(f"     Win Rate: {win_rate*100:.1f}%")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка Performance Analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtester():
    """Тест основного Backtester"""
    print("\n5. Проверка Backtester...")
    
    try:
        from src.backtesting import Backtester, PositionSide
        
        # Создаем синтетические данные с трендом вверх
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(100) * 0.5 + 0.1)  # Добавляем тренд
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.2,
            'high': prices + np.abs(np.random.randn(100)) * 0.5,
            'low': prices - np.abs(np.random.randn(100)) * 0.5,
            'close': prices + np.random.randn(100) * 0.2,
            'volume': np.random.randint(1000000, 5000000, 100),
            'symbol': ['TEST'] * 100
        }, index=dates)
        
        # Корректируем OHLC для валидности
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Простая стратегия: buy on bar 10, sell on bar 50
        def simple_strategy(bar, portfolio, context):
            if 'bar_count' not in context:
                context['bar_count'] = 0
            context['bar_count'] += 1
            
            symbol = bar.get('symbol', 'TEST')
            has_position = portfolio.has_position(symbol)
            
            # Покупаем на 10-м баре
            if context['bar_count'] == 10 and not has_position:
                return 1  # BUY
            
            # Продаем на 50-м баре
            if context['bar_count'] == 50 and has_position:
                return -1  # SELL
            
            return 0  # HOLD
        
        # Бэктест
        bt = Backtester(
            data=df,
            strategy=simple_strategy,
            initial_capital=100000,
            commission_rate=0.001,
            slippage_rate=0.0005,
            position_size=0.50,  # Используем 50% капитала вместо 95%
            name="SimpleTest"
        )
        
        results = bt.run(verbose=False)
        
        # Проверяем что есть сигналы (покупка и продажа)
        assert len(results['signals']) > 0, "Должны быть сигналы"
        buy_signals = results['signals'][results['signals']['signal'] > 0]
        assert len(buy_signals) > 0, "Должен быть хотя бы 1 сигнал на покупку"
        
        print("   ✓ Backtester работает")
        print(f"     Total Signals: {len(results['signals'])}")
        print(f"     Buy Signals: {len(buy_signals)}")
        print(f"     Final Equity: ${results['final_equity']:.2f}")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка Backtester: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_backtester():
    """Тест ML Backtester"""
    print("\n6. Проверка ML Backtester...")
    
    try:
        from src.backtesting import MLBacktester
        from src.backtesting.ml_backtester import create_dummy_ml_model
        
        # Создаем данные с фичами и трендом
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(100) * 0.5 + 0.1)
        
        df = pd.DataFrame({
            'open': prices + np.random.randn(100) * 0.2,
            'high': prices + np.abs(np.random.randn(100)) * 0.5,
            'low': prices - np.abs(np.random.randn(100)) * 0.5,
            'close': prices + np.random.randn(100) * 0.2,
            'volume': np.random.randint(1000000, 5000000, 100),
            'symbol': ['TEST'] * 100,
            # Фичи
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100)
        }, index=dates)
        
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Dummy модель
        model = create_dummy_ml_model(feature_count=3)
        
        # ML бэктест
        ml_bt = MLBacktester(
            data=df,
            model=model,
            feature_columns=['feature_1', 'feature_2', 'feature_3'],
            initial_capital=100000,
            prediction_threshold=0.5,
            name="MLTest"
        )
        
        results = ml_bt.run(verbose=False)
        
        assert 'predictions' in results, "Должны быть предсказания"
        assert 'ml_metrics' in results, "Должны быть ML метрики"
        
        print("   ✓ ML Backtester работает")
        print(f"     Total Predictions: {results['ml_metrics']['total_predictions']}")
        print(f"     Avg Prediction: {results['ml_metrics']['avg_prediction']:.3f}")
        
        return True
    except Exception as e:
        print(f"   ❌ Ошибка ML Backtester: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция"""
    print("=" * 70)
    print("BACKTESTING ENGINE - QUICK TEST")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_portfolio,
        test_execution_engine,
        test_performance_analyzer,
        test_backtester,
        test_ml_backtester
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Пройдено тестов: {passed}/{total}")
    
    if all(results):
        print("\n✅ Все тесты пройдены! Backtesting Engine готов к использованию.")
        print("\nЗапустите демо-ноутбук:")
        print("  jupyter lab notebooks/exploratory/03_backtesting_demo.ipynb")
        return 0
    else:
        print("\n❌ Некоторые тесты не прошли. Проверьте ошибки выше.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
