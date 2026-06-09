from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytest

from src.execution.base_broker import (
    AccountInfo,
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


class MockBroker(BaseBroker):
    """In-memory broker for unit tests — no network calls."""

    def __init__(
        self,
        supports_fractional: bool = True,
        account_info: Optional[AccountInfo] = None,
        positions: Optional[List[Position]] = None,
    ) -> None:
        self._fractional = supports_fractional
        self._account_info = account_info or AccountInfo(
            cash=10000.0,
            portfolio_value=10000.0,
            buying_power=10000.0,
            equity=10000.0,
        )
        self._positions: Dict[str, Position] = {}
        if positions:
            for p in positions:
                self._positions[p.symbol] = p
        self._orders: Dict[str, Order] = {}
        self._counter = 0
        self._canceled: set = set()

    # --- Abstract method implementations ---

    def connect(self) -> bool:
        return True

    def get_account(self) -> AccountInfo:
        return self._account_info

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        self._counter += 1
        now = datetime.now()
        order = Order(
            id=str(self._counter),
            symbol=symbol,
            side=side,
            qty=qty,
            type=type,
            status=OrderStatus.ACCEPTED,
            filled_qty=0.0,
            filled_avg_price=0.0,
            limit_price=limit_price,
            stop_price=stop_price,
            created_at=now,
            updated_at=now,
            client_order_id=None,
        )
        self._orders[order.id] = order
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order is None:
            return False
        order.status = OrderStatus.CANCELED
        self._canceled.add(order_id)
        return True

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def close_position(
        self, symbol: str, qty: Optional[float] = None
    ) -> Optional[Order]:
        return None

    def supports_fractional_shares(self) -> bool:
        return self._fractional

    # --- Test helpers ---

    def simulate_fill(self, order_id: str, fill_price: float, fill_qty: Optional[float] = None) -> None:
        """Mark an order as filled — used to simulate broker-side fills in tests."""
        order = self._orders.get(order_id)
        if order is None:
            return
        order.status = OrderStatus.FILLED
        order.filled_avg_price = fill_price
        order.filled_qty = fill_qty if fill_qty is not None else order.qty
        order.updated_at = datetime.now()

        qty = order.filled_qty
        if order.side == OrderSide.BUY:
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                pos.qty += qty
                pos.market_value = pos.qty * fill_price
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol, qty=qty, market_value=qty * fill_price,
                    avg_entry_price=fill_price, unrealized_pl=0.0,
                    current_price=fill_price, realized_pl=0.0,
                )
        elif order.side == OrderSide.SELL:
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                pos.qty -= qty
                if abs(pos.qty) < 1e-10:
                    del self._positions[order.symbol]
                else:
                    pos.market_value = pos.qty * fill_price

    def set_account_info(self, account_info: AccountInfo) -> None:
        self._account_info = account_info


@pytest.fixture
def mock_broker() -> MockBroker:
    return MockBroker(supports_fractional=True)


@pytest.fixture
def mock_broker_no_fractional() -> MockBroker:
    return MockBroker(supports_fractional=False)
