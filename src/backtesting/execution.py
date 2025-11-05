"""
Execution Engine
Система исполнения ордеров с поддержкой market, limit, stop orders
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from .base import Order, OrderType, OrderStatus, PositionSide
from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Движок исполнения ордеров
    
    Features:
    - Market orders (исполнение по текущей цене)
    - Limit orders (исполнение по заданной цене или лучше)
    - Stop orders (исполнение при достижении стоп-цены)
    - Stop-Limit orders (комбинация)
    - Учет slippage и комиссий
    - Проверка доступности капитала
    """
    
    def __init__(
        self,
        portfolio: Portfolio,
        use_bid_ask: bool = False,
        fill_at: str = 'close'  # 'open', 'close', 'average'
    ):
        """
        Инициализация
        
        Args:
            portfolio: Портфель для исполнения ордеров
            use_bid_ask: Использовать bid/ask (если доступны)
            fill_at: Точка исполнения ('open', 'close', 'average')
        """
        self.portfolio = portfolio
        self.use_bid_ask = use_bid_ask
        self.fill_at = fill_at
        
        # Очередь ордеров
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.cancelled_orders: List[Order] = []
        
        logger.info(f"ExecutionEngine инициализирован (fill_at={fill_at})")
    
    def submit_order(
        self,
        symbol: str,
        side: PositionSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> Order:
        """
        Подать ордер
        
        Args:
            symbol: Тикер
            side: Направление (LONG/SHORT)
            quantity: Количество
            order_type: Тип ордера
            price: Цена (для limit orders)
            stop_price: Стоп-цена (для stop orders)
            timestamp: Время подачи
        
        Returns:
            Order: Созданный ордер
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        order = Order(
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            timestamp=timestamp,
            price=price,
            stop_price=stop_price
        )
        
        # Валидация
        if order_type == OrderType.LIMIT and price is None:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Limit order отклонен: не указана цена")
            return order
        
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and stop_price is None:
            order.status = OrderStatus.REJECTED
            logger.warning(f"Stop order отклонен: не указана стоп-цена")
            return order
        
        # Добавляем в очередь
        self.pending_orders.append(order)
        logger.debug(f"Ордер подан: {order.order_type.value} {side.value} {quantity} {symbol}")
        
        return order
    
    def process_orders(self, bar: pd.Series, timestamp: datetime):
        """
        Обработать все pending ордера на текущем баре
        
        Args:
            bar: Текущий бар с OHLC данными
            timestamp: Текущее время
        """
        symbol = bar.get('symbol', 'UNKNOWN')
        
        # Обрабатываем каждый pending ордер
        for order in list(self.pending_orders):
            if order.symbol != symbol:
                continue
            
            # Пытаемся исполнить
            filled = self._try_fill_order(order, bar, timestamp)
            
            if filled:
                self.pending_orders.remove(order)
                self.filled_orders.append(order)
    
    def _try_fill_order(self, order: Order, bar: pd.Series, timestamp: datetime) -> bool:
        """
        Попытаться исполнить ордер
        
        Returns:
            bool: True если ордер исполнен
        """
        symbol = order.symbol
        open_price = bar.get('open', 0)
        high_price = bar.get('high', 0)
        low_price = bar.get('low', 0)
        close_price = bar.get('close', 0)
        
        fill_price = None
        
        # Определяем цену исполнения в зависимости от типа ордера
        if order.order_type == OrderType.MARKET:
            # Market order исполняется по следующему open или close
            if self.fill_at == 'open':
                fill_price = open_price
            elif self.fill_at == 'close':
                fill_price = close_price
            else:  # average
                fill_price = (open_price + close_price) / 2
        
        elif order.order_type == OrderType.LIMIT:
            # Limit order исполняется если цена достигла лимита
            if order.side == PositionSide.LONG:
                # Buy limit: исполняем если low <= limit_price
                if low_price <= order.price <= high_price:
                    fill_price = min(order.price, close_price)
            else:  # SHORT
                # Sell limit: исполняем если high >= limit_price
                if low_price <= order.price <= high_price:
                    fill_price = max(order.price, close_price)
        
        elif order.order_type == OrderType.STOP:
            # Stop order становится market при достижении стоп-цены
            if order.side == PositionSide.LONG:
                # Buy stop: триггерится если high >= stop_price
                if high_price >= order.stop_price:
                    fill_price = max(order.stop_price, open_price)
            else:  # SHORT
                # Sell stop: триггерится если low <= stop_price
                if low_price <= order.stop_price:
                    fill_price = min(order.stop_price, open_price)
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # Stop-limit: триггерится как stop, но исполняется как limit
            triggered = False
            if order.side == PositionSide.LONG and high_price >= order.stop_price:
                triggered = True
            elif order.side == PositionSide.SHORT and low_price <= order.stop_price:
                triggered = True
            
            if triggered and order.price:
                # После триггера проверяем limit
                if low_price <= order.price <= high_price:
                    fill_price = order.price
        
        # Если цена определена - исполняем
        if fill_price is not None and fill_price > 0:
            return self._fill_order(order, fill_price, timestamp)
        
        return False
    
    def _fill_order(self, order: Order, price: float, timestamp: datetime) -> bool:
        """
        Исполнить ордер
        
        Args:
            order: Ордер для исполнения
            price: Цена исполнения
            timestamp: Время исполнения
        
        Returns:
            bool: True если успешно исполнен
        """
        # Применяем slippage
        if order.side == PositionSide.LONG:
            # При покупке slippage увеличивает цену
            slippage_adjusted_price = price * (1 + self.portfolio.slippage_rate)
        else:
            # При продаже slippage уменьшает цену
            slippage_adjusted_price = price * (1 - self.portfolio.slippage_rate)
        
        # Рассчитываем комиссию и slippage
        commission = self.portfolio.calculate_commission(slippage_adjusted_price, order.quantity)
        slippage_cost = abs(slippage_adjusted_price - price) * order.quantity
        
        # Проверяем возможность исполнения (для открытия позиций)
        if not self.portfolio.has_position(order.symbol) or \
           (self.portfolio.get_position(order.symbol).side != order.side):
            can_open, reason = self.portfolio.can_open_position(
                order.symbol,
                slippage_adjusted_price,
                order.quantity
            )
            if not can_open:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Ордер отклонен: {reason}")
                return False
        
        # Исполняем через портфель
        if order.side == PositionSide.LONG:
            # Открываем/увеличиваем long позицию
            trade = self.portfolio.open_position(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=slippage_adjusted_price,
                timestamp=timestamp,
                order=order
            )
        else:
            # Закрываем long или открываем short
            if self.portfolio.has_position(order.symbol):
                pos = self.portfolio.get_position(order.symbol)
                if pos.side == PositionSide.LONG:
                    # Закрываем long
                    trade = self.portfolio.close_position(
                        symbol=order.symbol,
                        price=slippage_adjusted_price,
                        timestamp=timestamp,
                        quantity=order.quantity
                    )
                else:
                    # Увеличиваем short (если поддерживается)
                    logger.warning(f"Short позиции пока не поддерживаются полностью")
                    return False
            else:
                # Открываем short (если поддерживается)
                logger.warning(f"Short позиции пока не поддерживаются полностью")
                return False
        
        # Обновляем статус ордера
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = slippage_adjusted_price
        order.commission = commission
        order.slippage = slippage_cost
        
        logger.info(
            f"Ордер исполнен: {order.side.value} {order.quantity} {order.symbol} "
            f"@ ${slippage_adjusted_price:.2f} (slippage: ${slippage_cost:.2f})"
        )
        
        return True
    
    def cancel_order(self, order: Order):
        """Отменить ордер"""
        if order in self.pending_orders:
            order.status = OrderStatus.CANCELLED
            self.pending_orders.remove(order)
            self.cancelled_orders.append(order)
            logger.info(f"Ордер отменен: {order.order_id}")
    
    def cancel_all_orders(self, symbol: Optional[str] = None):
        """Отменить все pending ордера (опционально по символу)"""
        for order in list(self.pending_orders):
            if symbol is None or order.symbol == symbol:
                self.cancel_order(order)
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Получить список pending ордеров"""
        if symbol:
            return [o for o in self.pending_orders if o.symbol == symbol]
        return self.pending_orders.copy()
    
    def get_order_summary(self) -> dict:
        """Получить статистику по ордерам"""
        return {
            'pending': len(self.pending_orders),
            'filled': len(self.filled_orders),
            'cancelled': len(self.cancelled_orders),
            'total_submitted': len(self.pending_orders) + len(self.filled_orders) + len(self.cancelled_orders)
        }
    
    def reset(self):
        """Сбросить все ордера"""
        self.pending_orders.clear()
        self.filled_orders.clear()
        self.cancelled_orders.clear()
        logger.info("ExecutionEngine сброшен")
