from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base_broker import BaseBroker, Order, OrderSide, OrderStatus, OrderType
from .account import TradingAccount

# Default risk limits (used when no TradingConfig is provided)
_DEFAULT_MAX_POSITION_CONCENTRATION = 0.10  # 10% of NAV per position
_DEFAULT_MAX_GROSS_EXPOSURE = 0.90  # 90% of NAV max gross

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages the full order lifecycle: signal → order → fill → risk checks.

    Integrates a BaseBroker for execution and a TradingAccount for
    capital/position tracking.  All quantities are float — no integer
    rounding, supporting fractional shares throughout.
    """

    def __init__(
        self,
        broker: BaseBroker,
        account: TradingAccount,
        max_position_concentration: float = _DEFAULT_MAX_POSITION_CONCENTRATION,
        max_gross_exposure: float = _DEFAULT_MAX_GROSS_EXPOSURE,
    ) -> None:
        self._broker = broker
        self._account = account
        self._max_position_concentration = max_position_concentration
        self._max_gross_exposure = max_gross_exposure
        self._orders: Dict[str, Order] = {}  # open orders by id
        self._order_history: List[Order] = []
        # Sync account state from broker on startup so positions & cash are current
        try:
            self._account.update_from_broker(self._broker)
        except Exception as exc:
            logger.warning("Broker sync failed on init (%s), starting with default state", exc)

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------

    def execute_signal(
        self,
        symbol: str,
        target_qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        notional: Optional[float] = None,
    ) -> Optional[Order]:
        """Convert a trading signal into a broker order.

        If *target_qty* is fractional (< 1.0) the broker must support
        fractional shares, otherwise the signal is skipped.

        When *notional* is provided, it is logged but the order is placed
        by *target_qty* (the BaseBroker interface uses qty, not notional).
        """
        if target_qty <= 0:
            logger.warning(
                "execute_signal: non-positive target_qty=%s for %s, skipping",
                target_qty,
                symbol,
            )
            return None

        # Duplicate-order guard — skip if there is already a pending order
        # for the same symbol in the same direction
        self._sync_order_states()
        for existing in self._orders.values():
            if existing.symbol == symbol and existing.side == side:
                logger.warning(
                    "Duplicate signal: %s %s already has open order %s (id=%s), skipping",
                    side.value.upper(),
                    symbol,
                    existing.id,
                )
                return None

        # Risk check: position concentration limit
        # (will use estimated price from current position if available)
        est_price = 1.0
        pos = self._account.get_position(symbol)
        if pos is not None and pos.current_price > 0:
            est_price = pos.current_price
        if side == OrderSide.BUY and self._max_position_concentration < 1.0:
            # Approximate what the new concentration would be
            current_mv = abs(pos.market_value) if pos is not None else 0.0
            new_mv = current_mv + est_price * target_qty
            total_value = self._account.get_total_value()
            if total_value > 0 and (new_mv / total_value) > self._max_position_concentration:
                max_qty = (self._max_position_concentration * total_value - current_mv) / est_price
                if max_qty <= 0:
                    logger.warning(
                        "Position concentration limit hit for %s (%.1f%% of NAV > %.0f%% limit), skipping",
                        symbol, (new_mv / total_value) * 100,
                        self._max_position_concentration * 100,
                    )
                    return None
                target_qty = max_qty
                logger.info(
                    "Capped %s buy to qty=%.4f (concentration limit %.0f%% of NAV)",
                    symbol, target_qty, self._max_position_concentration * 100,
                )

        if target_qty < 1.0 and not self._broker.supports_fractional_shares():
            logger.warning(
                "Broker does not support fractional shares; "
                "skipping signal for %s (qty=%s)",
                symbol,
                target_qty,
            )
            return None

        order = self._broker.place_order(
            symbol=symbol,
            qty=target_qty,
            side=side,
            type=order_type,
        )
        self._orders[order.id] = order
        self._order_history.append(order)

        # Apply immediate fills — broker may fill MARKET orders synchronously
        if order.status == OrderStatus.FILLED:
            self._account.apply_fill(order)

        logger.info(
            "Signal executed: %s %s %s qty=%s type=%s id=%s%s",
            side.value.upper(),
            symbol,
            target_qty,
            order_type.value,
            order.id,
            " (immediate fill)" if order.status == OrderStatus.FILLED else "",
        )
        if notional is not None:
            logger.debug("notional=%.2f provided but order placed by qty", notional)

        return order

    def execute_signals(self, signals: List[Dict]) -> List[Order]:
        """Batch-execute multiple signals.

        Each signal dict must contain ``symbol``, ``target_qty``, ``side``.
        Optional keys: ``order_type``, ``notional``.

        Returns the list of successfully placed orders (may be shorter
        than the input if some signals are rejected).
        """
        orders: List[Order] = []
        for sig in signals:
            symbol = sig["symbol"]
            target_qty = float(sig["target_qty"])
            side = sig["side"]
            if not isinstance(side, OrderSide):
                raise TypeError(
                    f"signal side must be OrderSide, got {type(side).__name__}"
                )
            order_type = sig.get("order_type", OrderType.MARKET)
            notional = sig.get("notional", None)

            result = self.execute_signal(
                symbol=symbol,
                target_qty=target_qty,
                side=side,
                order_type=order_type,  # type: ignore[arg-type]
                notional=notional,
            )
            if result is not None:
                orders.append(result)

        logger.info("Batch executed %d/%d signals", len(orders), len(signals))
        return orders

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def rebalance_to_targets(
        self,
        target_positions: Dict[str, float],
        max_notional_per_order: Optional[float] = None,
    ) -> List[Order]:
        """Generate orders to bring the portfolio to *target_positions*.

        For each symbol, computes ``delta = target_qty - current_qty``.
        Positive delta → BUY, negative delta → SELL.
        Zero delta is skipped.

        *max_notional_per_order* caps the notional size per order
        (logged but not enforced at the broker-interface level since
        place_order uses qty).
        """
        orders: List[Order] = []

        # Sync account state from broker before computing deltas
        try:
            self._account.update_from_broker(self._broker)
        except Exception as exc:
            logger.warning("Broker sync failed during rebalance (%s), using current state", exc)
        current_positions = self._account.positions

        for symbol, target_qty in target_positions.items():
            current_qty = 0.0
            if symbol in current_positions:
                current_qty = current_positions[symbol].qty

            delta = target_qty - current_qty
            if abs(delta) < 1e-10:
                continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            abs_qty = abs(delta)

            if max_notional_per_order is not None:
                pos = self._account.get_position(symbol)
                est_price = pos.current_price if pos and pos.current_price > 0 else 0.0
                if est_price > 0:
                    est_notional = abs_qty * est_price
                    if est_notional > max_notional_per_order:
                        abs_qty = max_notional_per_order / est_price
                        logger.debug(
                            "Capped %s rebalance to qty=%s (notional limit=%.2f)",
                            symbol,
                            abs_qty,
                            max_notional_per_order,
                        )

            result = self.execute_signal(
                symbol=symbol,
                target_qty=abs_qty,
                side=side,
            )
            if result is not None:
                orders.append(result)

        logger.info(
            "Rebalance: %d orders generated for %d targets",
            len(orders),
            len(target_positions),
        )
        return orders

    # ------------------------------------------------------------------
    # Order tracking
    # ------------------------------------------------------------------

    def get_open_orders(self) -> List[Order]:
        """Return all currently open (non-terminal) orders."""
        self._sync_order_states()
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        return [o for o in self._orders.values() if o.status not in terminal]

    def cancel_all_open(self) -> int:
        """Cancel every open order.  Returns the count of successful cancels."""
        self._sync_order_states()
        cancelled = 0
        for order_id in list(self._orders.keys()):
            if self._broker.cancel_order(order_id):
                cancelled += 1
        self._sync_order_states()
        logger.info("Cancelled %d open orders", cancelled)
        return cancelled

    def get_order_history(self, limit: int = 100) -> List[Order]:
        """Return the most recent *limit* orders from history."""
        return self._order_history[-limit:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_order_states(self) -> None:
        """Poll the broker for each open order's current status.

        Orders that reach a terminal state are removed from the open set
        and their fills are applied to the account.
        """
        terminal = {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }
        for order_id in list(self._orders.keys()):
            updated = self._broker.get_order(order_id)
            if updated is None:
                # Order vanished from broker — remove from open set
                del self._orders[order_id]
                continue
            self._orders[order_id] = updated
            if updated.status in terminal:
                if updated.status == OrderStatus.FILLED:
                    self._account.apply_fill(updated)
                del self._orders[order_id]
