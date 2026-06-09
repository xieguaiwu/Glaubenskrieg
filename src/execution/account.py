from __future__ import annotations

import logging
from typing import Dict, Optional

from .base_broker import BaseBroker, Order, OrderSide, Position

logger = logging.getLogger(__name__)


class TradingAccount:
    """Tracks capital allocation, positions, portfolio value, and PnL.

    All quantities are float to support fractional share precision.
    Integrates with any BaseBroker implementation for state synchronization.
    """

    def __init__(self, initial_capital: float = 10000.0) -> None:
        self._initial_capital = initial_capital
        self._cash: float = initial_capital
        self._positions: Dict[str, Position] = {}
        self._realized_pl: float = 0.0

    # ------------------------------------------------------------------
    # Broker synchronization
    # ------------------------------------------------------------------

    def update_from_broker(self, broker: BaseBroker) -> None:
        """Sync internal state with live broker account and positions."""
        account_info = broker.get_account()
        self._cash = account_info.cash

        broker_positions = broker.get_positions()
        self._positions.clear()
        for pos in broker_positions:
            self._positions[pos.symbol] = pos

        logger.info(
            "Synced from broker: cash=%.2f, positions=%d, portfolio_value=%.2f",
            self._cash,
            len(self._positions),
            account_info.portfolio_value,
        )

    # ------------------------------------------------------------------
    # Fill application
    # ------------------------------------------------------------------

    def apply_fill(self, order: Order) -> None:
        """Update cash and positions after an order fill executes.

        Handles BUY (increase position) and SELL (reduce/close position).
        Fractional qty is supported throughout.
        """
        filled_qty = order.filled_qty if order.filled_qty > 0 else order.qty
        price = order.filled_avg_price

        if filled_qty <= 0 or price <= 0:
            logger.warning(
                "Skipping fill for order %s: filled_qty=%s, filled_avg_price=%s",
                order.id,
                filled_qty,
                price,
            )
            return

        if order.side == OrderSide.BUY:
            self._cash -= filled_qty * price
            self._add_to_position(order.symbol, filled_qty, price)
        elif order.side == OrderSide.SELL:
            self._cash += filled_qty * price
            self._remove_from_position(order.symbol, filled_qty, price)

        logger.info(
            "Applied fill: %s %s %s @ %.2f, cash=%.2f, positions=%d",
            order.side.value,
            order.symbol,
            filled_qty,
            price,
            self._cash,
            len(self._positions),
        )

    def _add_to_position(self, symbol: str, qty: float, price: float) -> None:
        """Add shares to an existing or new long position."""
        if symbol in self._positions:
            pos = self._positions[symbol]
            total_cost = pos.qty * pos.avg_entry_price + qty * price
            pos.qty += qty
            if pos.qty != 0:
                pos.avg_entry_price = total_cost / pos.qty
            pos.market_value = pos.qty * pos.current_price
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                market_value=qty * price,
                avg_entry_price=price,
                unrealized_pl=0.0,
                current_price=price,
                realized_pl=0.0,
            )

    def _remove_from_position(self, symbol: str, qty: float, price: float) -> None:
        """Reduce or close a position, realizing P&L on the sold portion."""
        if symbol not in self._positions:
            # Short sale: create a negative position
            self._positions[symbol] = Position(
                symbol=symbol,
                qty=-qty,
                market_value=-qty * price,
                avg_entry_price=price,
                unrealized_pl=0.0,
                current_price=price,
                realized_pl=0.0,
            )
            return

        pos = self._positions[symbol]

        if pos.qty > 0:
            # Closing or reducing a long position
            close_qty = min(qty, pos.qty)
            self._realized_pl += close_qty * (price - pos.avg_entry_price)
            pos.realized_pl += close_qty * (price - pos.avg_entry_price)

        pos.qty -= qty

        if abs(pos.qty) < 1e-10:
            del self._positions[symbol]
        else:
            pos.market_value = pos.qty * pos.current_price

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return the current position for *symbol*, or None."""
        return self._positions.get(symbol)

    # ------------------------------------------------------------------
    # Value & P&L
    # ------------------------------------------------------------------

    def get_total_value(self) -> float:
        """Portfolio total value: cash + sum of position market values."""
        positions_value = sum(p.market_value for p in self._positions.values())
        return self._cash + positions_value

    def get_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all positions.
        
        Computed live from current data: sum(market_value - qty * avg_entry_price).
        Does not rely on the stale ``unrealized_pl`` field on Position.
        """
        total = 0.0
        for p in self._positions.values():
            cost_basis = p.qty * p.avg_entry_price
            total += p.market_value - cost_basis
        return total

    def get_realized_pnl(self) -> float:
        """Cumulative realized P&L from closed trades."""
        return self._realized_pl

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    def get_buying_power(self) -> float:
        """Available cash for new positions."""
        return max(self._cash, 0.0)

    def get_exposure(self) -> float:
        """Gross exposure as a ratio of total portfolio value.

        Gross exposure = sum(|position market value|) / total value.
        Returns 0.0 when total value is zero.
        """
        total_value = self.get_total_value()
        if total_value <= 0:
            return 0.0
        gross = sum(abs(p.market_value) for p in self._positions.values())
        return gross / total_value

    def get_position_concentration(self, symbol: str) -> float:
        """Single position market value as a fraction of total portfolio."""
        total_value = self.get_total_value()
        if total_value <= 0:
            return 0.0
        pos = self._positions.get(symbol)
        if pos is None:
            return 0.0
        return abs(pos.market_value) / total_value

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    @property
    def initial_capital(self) -> float:
        return self._initial_capital
