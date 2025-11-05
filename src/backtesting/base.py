"""
Базовые классы для бэктестинга:
- Order: представляет заявку
- Trade: представляет исполненную сделку
- Position: представляет открытую позицию
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import pandas as pd


class OrderType(Enum):
    """Тип ордера"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Статус ордера"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionSide(Enum):
    """Направление позиции"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """
    Ордер на покупку/продажу
    
    Attributes:
        symbol: Тикер инструмента
        order_type: Тип ордера (market, limit, stop)
        side: Направление (long/short)
        quantity: Количество
        price: Цена (для limit/stop)
        timestamp: Время создания
        stop_price: Стоп-цена (для stop orders)
        order_id: Уникальный ID
        status: Статус ордера
        filled_quantity: Исполненное количество
        filled_price: Цена исполнения
        commission: Комиссия
        slippage: Проскальзывание
    """
    symbol: str
    order_type: OrderType
    side: PositionSide
    quantity: float
    timestamp: datetime
    price: Optional[float] = None
    stop_price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.symbol}_{self.timestamp.timestamp()}_{id(self)}"
    
    def to_dict(self) -> dict:
        """Конвертация в словарь"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'slippage': self.slippage
        }


@dataclass
class Trade:
    """
    Исполненная сделка
    
    Attributes:
        trade_id: Уникальный ID сделки
        symbol: Тикер
        side: Направление
        quantity: Количество
        entry_price: Цена входа
        exit_price: Цена выхода (None если позиция открыта)
        entry_time: Время входа
        exit_time: Время выхода
        commission: Комиссия
        slippage: Проскальзывание
        pnl: Реализованная прибыль/убыток
        pnl_pct: Прибыль/убыток в процентах
        holding_period: Время удержания позиции (в барах)
    """
    trade_id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    commission: float = 0.0
    slippage: float = 0.0
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_period: int = 0
    
    def close(self, exit_price: float, exit_time: datetime, commission: float = 0.0, slippage: float = 0.0):
        """Закрыть сделку"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.commission += commission
        self.slippage += slippage
        
        # Расчет P&L
        if self.side == PositionSide.LONG:
            price_diff = exit_price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - exit_price
        
        gross_pnl = price_diff * self.quantity
        self.pnl = gross_pnl - self.commission - self.slippage
        self.pnl_pct = (price_diff / self.entry_price) * 100 if self.entry_price != 0 else 0.0
        
        # Holding period (будет обновлен в бэктестере)
        if exit_time and self.entry_time:
            self.holding_period = (exit_time - self.entry_time).days
    
    def is_open(self) -> bool:
        """Проверка, открыта ли сделка"""
        return self.exit_price is None
    
    def to_dict(self) -> dict:
        """Конвертация в словарь"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time,
            'commission': self.commission,
            'slippage': self.slippage,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'holding_period': self.holding_period
        }


@dataclass
class Position:
    """
    Открытая позиция
    
    Attributes:
        symbol: Тикер
        side: Направление (long/short)
        quantity: Количество
        entry_price: Средняя цена входа
        current_price: Текущая цена
        entry_time: Время открытия
        unrealized_pnl: Нереализованная прибыль/убыток
        realized_pnl: Реализованная прибыль/убыток
        total_commission: Общая комиссия
        total_slippage: Общее проскальзывание
    """
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0
    
    def update_price(self, price: float):
        """Обновить текущую цену и нереализованный P&L"""
        self.current_price = price
        
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
            return
        
        if self.side == PositionSide.LONG:
            price_diff = price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - price
        
        self.unrealized_pnl = (price_diff * self.quantity) - self.total_commission - self.total_slippage
    
    def add_quantity(self, quantity: float, price: float, commission: float = 0.0, slippage: float = 0.0):
        """Добавить к позиции (усреднение)"""
        if self.quantity == 0:
            self.entry_price = price
            self.quantity = quantity
        else:
            # Weighted average entry price
            total_cost = (self.entry_price * self.quantity) + (price * quantity)
            self.quantity += quantity
            self.entry_price = total_cost / self.quantity if self.quantity != 0 else 0.0
        
        self.total_commission += commission
        self.total_slippage += slippage
        self.update_price(self.current_price if self.current_price > 0 else price)
    
    def reduce_quantity(self, quantity: float, price: float, commission: float = 0.0, slippage: float = 0.0) -> float:
        """
        Уменьшить позицию (частичное закрытие)
        Возвращает реализованный P&L
        """
        if quantity > self.quantity:
            quantity = self.quantity
        
        if self.side == PositionSide.LONG:
            price_diff = price - self.entry_price
        else:  # SHORT
            price_diff = self.entry_price - price
        
        pnl = (price_diff * quantity) - commission - slippage
        
        self.quantity -= quantity
        self.realized_pnl += pnl
        self.total_commission += commission
        self.total_slippage += slippage
        
        if self.quantity == 0:
            self.side = PositionSide.FLAT
        
        self.update_price(price)
        return pnl
    
    def get_market_value(self) -> float:
        """Рыночная стоимость позиции"""
        return self.quantity * self.current_price
    
    def get_total_pnl(self) -> float:
        """Общий P&L (реализованный + нереализованный)"""
        return self.realized_pnl + self.unrealized_pnl
    
    def to_dict(self) -> dict:
        """Конвертация в словарь"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'entry_time': self.entry_time,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.get_total_pnl(),
            'market_value': self.get_market_value(),
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage
        }
