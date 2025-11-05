"""
Portfolio Management
Управление капиталом, позициями, учет комиссий и slippage
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .base import Position, Trade, PositionSide, Order

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Портфель для управления капиталом и позициями
    
    Features:
    - Управление балансом и позициями
    - Учет комиссий и slippage
    - Расчет margin requirements
    - История equity и drawdown
    - Реализованный и нереализованный P&L
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,   # 0.05%
        margin_requirement: float = 1.0,  # 1.0 = no leverage, 0.5 = 2x leverage
        name: str = "Portfolio"
    ):
        """
        Инициализация портфеля
        
        Args:
            initial_capital: Начальный капитал
            commission_rate: Комиссия (доля от объема сделки)
            slippage_rate: Проскальзывание (доля от цены)
            margin_requirement: Требование по марже (1.0 = 100%, 0.5 = 50%)
            name: Название портфеля
        """
        self.name = name
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.margin_requirement = margin_requirement
        
        # Балансы
        self.cash = initial_capital
        self.equity = initial_capital
        
        # Позиции и сделки
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.open_trades: Dict[str, Trade] = {}
        
        # История
        self.equity_history: List[dict] = []
        self.balance_history: List[dict] = []
        
        # Метрики
        self.total_commission_paid = 0.0
        self.total_slippage_paid = 0.0
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Portfolio '{name}' создан: начальный капитал ${initial_capital:,.2f}")
    
    def calculate_commission(self, price: float, quantity: float) -> float:
        """Расчет комиссии"""
        return price * quantity * self.commission_rate
    
    def calculate_slippage(self, price: float, quantity: float) -> float:
        """Расчет проскальзывания"""
        return price * quantity * self.slippage_rate
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Получить позицию по символу"""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Проверить наличие открытой позиции"""
        pos = self.positions.get(symbol)
        return pos is not None and pos.quantity > 0
    
    def get_position_value(self, symbol: str) -> float:
        """Стоимость позиции"""
        pos = self.get_position(symbol)
        return pos.get_market_value() if pos else 0.0
    
    def get_total_position_value(self) -> float:
        """Общая стоимость всех позиций"""
        return sum(pos.get_market_value() for pos in self.positions.values())
    
    def get_available_capital(self) -> float:
        """Доступный капитал для торговли"""
        used_margin = self.get_total_position_value() * self.margin_requirement
        return self.cash - used_margin
    
    def can_open_position(self, symbol: str, price: float, quantity: float) -> tuple[bool, str]:
        """
        Проверить возможность открытия позиции
        
        Returns:
            (bool, str): (возможно ли открыть, причина отказа)
        """
        required_capital = price * quantity
        commission = self.calculate_commission(price, quantity)
        slippage = self.calculate_slippage(price, quantity)
        total_cost = required_capital + commission + slippage
        
        required_margin = required_capital * self.margin_requirement
        
        if self.cash < total_cost:
            return False, f"Недостаточно средств: требуется ${total_cost:,.2f}, доступно ${self.cash:,.2f}"
        
        available = self.get_available_capital()
        if available < required_margin:
            return False, f"Недостаточная маржа: требуется ${required_margin:,.2f}, доступно ${available:,.2f}"
        
        return True, "OK"
    
    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        price: float,
        timestamp: datetime,
        order: Optional[Order] = None
    ) -> Optional[Trade]:
        """
        Открыть или добавить к позиции
        
        Returns:
            Trade: Объект сделки, если успешно
        """
        # Расчет комиссии и slippage
        commission = self.calculate_commission(price, quantity)
        slippage = self.calculate_slippage(price, quantity)
        
        # Проверка капитала
        can_open, reason = self.can_open_position(symbol, price, quantity)
        if not can_open:
            logger.warning(f"Невозможно открыть позицию {symbol}: {reason}")
            return None
        
        # Списываем средства
        total_cost = (price * quantity) + commission + slippage
        self.cash -= total_cost
        self.total_commission_paid += commission
        self.total_slippage_paid += slippage
        
        # Создаем или обновляем позицию
        if symbol not in self.positions or self.positions[symbol].quantity == 0:
            # Новая позиция
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
                current_price=price,
                total_commission=commission,
                total_slippage=slippage
            )
            
            # Создаем Trade
            trade = Trade(
                trade_id=f"{symbol}_{timestamp.timestamp()}",
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
                commission=commission,
                slippage=slippage
            )
            self.open_trades[trade.trade_id] = trade
            
            logger.info(f"Открыта {side.value} позиция {symbol}: {quantity} @ ${price:.2f}")
            return trade
        else:
            # Добавление к существующей позиции
            pos = self.positions[symbol]
            if pos.side == side:
                pos.add_quantity(quantity, price, commission, slippage)
                logger.info(f"Увеличена {side.value} позиция {symbol}: +{quantity} @ ${price:.2f}")
            else:
                logger.warning(f"Попытка открыть {side.value} при наличии {pos.side.value} позиции {symbol}")
                return None
        
        return None
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        quantity: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Закрыть позицию полностью или частично
        
        Args:
            symbol: Тикер
            price: Цена закрытия
            timestamp: Время закрытия
            quantity: Количество для закрытия (None = закрыть все)
        
        Returns:
            Trade: Закрытая сделка
        """
        if symbol not in self.positions or self.positions[symbol].quantity == 0:
            logger.warning(f"Нет открытой позиции {symbol} для закрытия")
            return None
        
        pos = self.positions[symbol]
        close_qty = quantity if quantity else pos.quantity
        
        if close_qty > pos.quantity:
            logger.warning(f"Попытка закрыть {close_qty}, но открыто только {pos.quantity}")
            close_qty = pos.quantity
        
        # Расчет комиссии и slippage
        commission = self.calculate_commission(price, close_qty)
        slippage = self.calculate_slippage(price, close_qty)
        
        # Обновляем позицию и получаем реализованный P&L
        realized_pnl = pos.reduce_quantity(close_qty, price, commission, slippage)
        
        # Зачисляем средства
        proceeds = price * close_qty
        self.cash += proceeds - commission - slippage
        self.total_commission_paid += commission
        self.total_slippage_paid += slippage
        
        # Закрываем соответствующую сделку
        trade = None
        for trade_id, open_trade in list(self.open_trades.items()):
            if open_trade.symbol == symbol:
                open_trade.close(price, timestamp, commission, slippage)
                trade = open_trade
                self.closed_trades.append(trade)
                del self.open_trades[trade_id]
                break
        
        logger.info(
            f"Закрыта позиция {symbol}: {close_qty} @ ${price:.2f}, "
            f"P&L: ${realized_pnl:+,.2f}"
        )
        
        return trade
    
    def update_prices(self, prices: Dict[str, float], timestamp: datetime):
        """
        Обновить цены всех позиций и пересчитать equity
        
        Args:
            prices: Словарь {symbol: price}
            timestamp: Текущее время
        """
        # Обновляем цены позиций
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol])
        
        # Пересчитываем equity
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.equity = self.cash + self.get_total_position_value()
        
        # Обновляем peak и drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        current_dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        # Записываем в историю
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': self.equity,
            'cash': self.cash,
            'position_value': self.get_total_position_value(),
            'unrealized_pnl': total_unrealized_pnl,
            'drawdown': current_dd
        })
    
    def get_summary(self) -> dict:
        """Получить сводку по портфелю"""
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t.pnl > 0]
        losing_trades = [t for t in self.closed_trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        
        total_pnl = sum(t.pnl for t in self.closed_trades)
        total_return = (self.equity - self.initial_capital) / self.initial_capital * 100
        
        return {
            'name': self.name,
            'initial_capital': self.initial_capital,
            'current_equity': self.equity,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
            'max_drawdown': self.max_drawdown,
            'total_commission': self.total_commission_paid,
            'total_slippage': self.total_slippage_paid,
            'open_positions': len([p for p in self.positions.values() if p.quantity > 0])
        }
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Получить equity curve как DataFrame"""
        if not self.equity_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_history)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Получить все закрытые сделки как DataFrame"""
        if not self.closed_trades:
            return pd.DataFrame()
        
        trades_data = [trade.to_dict() for trade in self.closed_trades]
        return pd.DataFrame(trades_data)
    
    def reset(self):
        """Сбросить портфель к начальному состоянию"""
        self.cash = self.initial_capital
        self.equity = self.initial_capital
        self.positions.clear()
        self.closed_trades.clear()
        self.open_trades.clear()
        self.equity_history.clear()
        self.balance_history.clear()
        self.total_commission_paid = 0.0
        self.total_slippage_paid = 0.0
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Portfolio '{self.name}' сброшен")
