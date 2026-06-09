"""Real trading execution layer for Glaubenskrieg CTM.

Connects ML signal generation → broker API order execution.
Supports fractional shares, multiple brokers, paper/live modes.
"""

from .base_broker import BaseBroker, Position, AccountInfo, Order, OrderSide, OrderType, OrderStatus
from .account import TradingAccount
from .order_manager import OrderManager
from .alpaca_broker import AlpacaBroker
from .config import TradingConfig, load_trading_config

__all__ = [
    "BaseBroker", "Position", "AccountInfo", "Order",
    "OrderSide", "OrderType", "OrderStatus",
    "TradingAccount", "OrderManager",
    "AlpacaBroker",
    "TradingConfig", "load_trading_config",
]
