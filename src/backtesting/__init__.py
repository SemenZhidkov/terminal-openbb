"""
Backtesting Module
Профессиональный движок бэктестинга с поддержкой:
- Portfolio management с комиссиями и slippage
- Execution engine (market, limit, stop orders)
- Performance metrics (Sharpe, Sortino, Calmar, MaxDD, etc.)
- Multi-timeframe testing
- ML signal backtesting
- Walk-forward analysis
"""

from .base import Order, Trade, Position, OrderType, OrderStatus, PositionSide
from .portfolio import Portfolio
from .execution import ExecutionEngine
from .performance import PerformanceAnalyzer
from .backtester import Backtester
from .ml_backtester import MLBacktester
from .walk_forward import WalkForwardAnalyzer

__all__ = [
    'Order',
    'Trade',
    'Position',
    'OrderType',
    'OrderStatus',
    'PositionSide',
    'Portfolio',
    'ExecutionEngine',
    'PerformanceAnalyzer',
    'Backtester',
    'MLBacktester',
    'WalkForwardAnalyzer'
]

__version__ = '1.0.0'
