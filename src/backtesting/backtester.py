"""
Backtester
Основной движок бэктестинга торговых стратегий
"""

import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from .base import PositionSide, OrderType
from .portfolio import Portfolio
from .execution import ExecutionEngine
from .performance import PerformanceAnalyzer

logger = logging.getLogger(__name__)


class Backtester:
    """
    Основной движок бэктестинга
    
    Features:
    - Event-driven архитектура
    - Поддержка пользовательских стратегий
    - Multi-timeframe support
    - Подробное логирование
    - Метрики производительности
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: Callable,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        position_size: float = 1.0,  # Доля капитала на сделку (0-1)
        name: str = "Backtest"
    ):
        """
        Инициализация бэктестера
        
        Args:
            data: DataFrame с OHLCV данными (должен иметь индекс datetime)
            strategy: Функция стратегии strategy(row, portfolio, context) -> signal
            initial_capital: Начальный капитал
            commission_rate: Комиссия
            slippage_rate: Проскальзывание
            position_size: Размер позиции (доля от капитала)
            name: Название бэктеста
        """
        self.name = name
        self.data = data.copy()
        self.strategy = strategy
        self.position_size = position_size
        
        # Компоненты
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            name=f"{name}_Portfolio"
        )
        self.execution = ExecutionEngine(self.portfolio, fill_at='close')
        self.performance = PerformanceAnalyzer(self.portfolio)
        
        # Context для стратегии (можно хранить состояние)
        self.context = {}
        
        # Результаты
        self.signals: List[dict] = []
        self.completed = False
        
        logger.info(f"Backtester '{name}' инициализирован")
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Запуск бэктеста
        
        Args:
            verbose: Показывать progress bar
        
        Returns:
            dict: Результаты бэктеста
        """
        logger.info(f"Запуск бэктеста '{self.name}'...")
        
        # Проверка данных
        if self.data.empty:
            logger.error("Данные пустые!")
            return {}
        
        if not isinstance(self.data.index, pd.DatetimeIndex):
            logger.error("Индекс данных должен быть DatetimeIndex!")
            return {}
        
        # Сброс состояния
        self.portfolio.reset()
        self.execution.reset()
        self.signals.clear()
        self.context.clear()
        
        # Итерация по барам
        iterator = tqdm(self.data.iterrows(), total=len(self.data), desc="Backtesting") if verbose else self.data.iterrows()
        
        for timestamp, bar in iterator:
            # 1. Обработать pending ордера
            self.execution.process_orders(bar, timestamp)
            
            # 2. Обновить цены в портфеле
            symbol = bar.get('symbol', 'UNKNOWN')
            current_price = bar.get('close', 0)
            self.portfolio.update_prices({symbol: current_price}, timestamp)
            
            # 3. Генерация сигнала от стратегии
            try:
                signal = self.strategy(bar, self.portfolio, self.context)
            except Exception as e:
                logger.error(f"Ошибка в стратегии на {timestamp}: {e}")
                signal = 0
            
            # 4. Исполнение сигнала
            if signal != 0:
                self._process_signal(signal, bar, timestamp)
            
            # Логирование сигнала
            self.signals.append({
                'timestamp': timestamp,
                'signal': signal,
                'price': current_price,
                'equity': self.portfolio.equity,
                'position': self.portfolio.get_position(symbol).quantity if self.portfolio.has_position(symbol) else 0
            })
        
        self.completed = True
        logger.info(f"Бэктест '{self.name}' завершен")
        
        # Генерация отчета
        return self.get_results()
    
    def _process_signal(self, signal: float, bar: pd.Series, timestamp: datetime):
        """
        Обработать сигнал стратегии
        
        Args:
            signal: Сигнал (-1 = sell, 0 = hold, 1 = buy)
            bar: Текущий бар
            timestamp: Время
        """
        symbol = bar.get('symbol', 'UNKNOWN')
        current_price = bar.get('close', 0)
        
        # Определяем размер позиции
        available_capital = self.portfolio.equity * self.position_size
        quantity = available_capital / current_price if current_price > 0 else 0
        
        if signal > 0:  # BUY signal
            # Открываем/увеличиваем long позицию
            if not self.portfolio.has_position(symbol):
                self.execution.submit_order(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                )
                logger.debug(f"{timestamp}: BUY signal - открытие позиции {quantity:.2f}")
            else:
                logger.debug(f"{timestamp}: BUY signal - позиция уже открыта")
        
        elif signal < 0:  # SELL signal
            # Закрываем long позицию
            if self.portfolio.has_position(symbol):
                pos = self.portfolio.get_position(symbol)
                self.execution.submit_order(
                    symbol=symbol,
                    side=PositionSide.SHORT,  # SHORT order закрывает LONG
                    quantity=pos.quantity,
                    order_type=OrderType.MARKET,
                    timestamp=timestamp
                )
                logger.debug(f"{timestamp}: SELL signal - закрытие позиции {pos.quantity:.2f}")
            else:
                logger.debug(f"{timestamp}: SELL signal - нет открытой позиции")
    
    def get_results(self) -> Dict:
        """
        Получить результаты бэктеста
        
        Returns:
            dict: Результаты с метриками и данными
        """
        if not self.completed:
            logger.warning("Бэктест еще не завершен")
        
        # Генерируем отчет
        report = self.performance.generate_report()
        
        # Добавляем дополнительную информацию
        report['name'] = self.name
        report['data_points'] = len(self.data)
        report['start_date'] = self.data.index[0] if len(self.data) > 0 else None
        report['end_date'] = self.data.index[-1] if len(self.data) > 0 else None
        
        # DataFrames
        report['equity_curve'] = self.portfolio.get_equity_curve()
        report['trades'] = self.portfolio.get_trades_df()
        report['signals'] = pd.DataFrame(self.signals).set_index('timestamp') if self.signals else pd.DataFrame()
        
        return report
    
    def plot_results(self, figsize=(15, 10)):
        """
        Визуализация результатов
        
        Args:
            figsize: Размер фигуры
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            results = self.get_results()
            
            if results['equity_curve'].empty:
                logger.warning("Нет данных для визуализации")
                return
            
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
            
            # 1. Equity Curve
            ax1 = axes[0]
            equity_df = results['equity_curve']
            ax1.plot(equity_df.index, equity_df['equity'], label='Equity', linewidth=2)
            ax1.axhline(self.portfolio.initial_capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
            ax1.set_ylabel('Equity ($)')
            ax1.set_title(f'{self.name}: Equity Curve', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdown
            ax2 = axes[1]
            ax2.fill_between(equity_df.index, 0, -equity_df['drawdown'] * 100, 
                            color='red', alpha=0.3, label='Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_title('Drawdown', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            # 3. Position Value
            ax3 = axes[2]
            ax3.fill_between(equity_df.index, 0, equity_df['position_value'], 
                            alpha=0.5, label='Position Value')
            ax3.plot(equity_df.index, equity_df['cash'], label='Cash', linewidth=2)
            ax3.set_ylabel('Value ($)')
            ax3.set_xlabel('Date')
            ax3.set_title('Cash vs Position Value', fontsize=12)
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Trade distribution
            if not results['trades'].empty:
                fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
                
                # Histogram of P&L
                ax1 = axes2[0]
                trades_df = results['trades']
                ax1.hist(trades_df['pnl'], bins=30, edgecolor='black', alpha=0.7)
                ax1.axvline(0, color='red', linestyle='--', linewidth=2)
                ax1.set_xlabel('P&L ($)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Distribution of Trade P&L')
                ax1.grid(True, alpha=0.3)
                
                # Win/Loss distribution
                ax2 = axes2[1]
                win_loss = trades_df['pnl'].apply(lambda x: 'Win' if x > 0 else 'Loss')
                win_loss.value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%', 
                                            colors=['green', 'red'])
                ax2.set_ylabel('')
                ax2.set_title('Win/Loss Distribution')
                
                plt.tight_layout()
                plt.show()
        
        except ImportError:
            logger.warning("Matplotlib не установлен. Визуализация недоступна.")
    
    def print_summary(self):
        """Вывести краткую сводку"""
        if not self.completed:
            logger.warning("Бэктест еще не завершен")
            return
        
        self.performance.print_report()


# Примеры простых стратегий для демонстрации

def simple_sma_crossover_strategy(bar: pd.Series, portfolio: Portfolio, context: dict) -> int:
    """
    Простая стратегия на основе пересечения SMA
    
    Требует наличия колонок 'sma_50' и 'sma_200' в данных
    
    Returns:
        1: BUY signal
        -1: SELL signal
        0: HOLD
    """
    symbol = bar.get('symbol', 'UNKNOWN')
    
    # Проверяем наличие индикаторов
    if 'sma_50' not in bar or 'sma_200' not in bar:
        return 0
    
    if pd.isna(bar['sma_50']) or pd.isna(bar['sma_200']):
        return 0
    
    # Golden Cross: SMA50 пересекает SMA200 снизу вверх
    if bar['sma_50'] > bar['sma_200']:
        if not portfolio.has_position(symbol):
            return 1  # BUY
    
    # Death Cross: SMA50 пересекает SMA200 сверху вниз
    elif bar['sma_50'] < bar['sma_200']:
        if portfolio.has_position(symbol):
            return -1  # SELL
    
    return 0  # HOLD


def rsi_strategy(bar: pd.Series, portfolio: Portfolio, context: dict) -> int:
    """
    Стратегия на основе RSI
    
    Требует наличия колонки 'rsi_14' в данных
    
    Returns:
        1: BUY when oversold (RSI < 30)
        -1: SELL when overbought (RSI > 70)
        0: HOLD
    """
    symbol = bar.get('symbol', 'UNKNOWN')
    
    if 'rsi_14' not in bar or pd.isna(bar['rsi_14']):
        return 0
    
    rsi = bar['rsi_14']
    
    # Oversold - покупаем
    if rsi < 30 and not portfolio.has_position(symbol):
        return 1
    
    # Overbought - продаем
    if rsi > 70 and portfolio.has_position(symbol):
        return -1
    
    return 0
