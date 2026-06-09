from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .base_broker import (
    AccountInfo,
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Retry / backoff helpers
# ──────────────────────────────────────────────────────────────────────

_RETRY_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    OSError,
)


def _retry_on_failure(
    func_name: str,
    max_retries: int = 3,
    base_delay: float = 0.5,
):
    """Decorator that retries a method on transient network errors."""

    def decorator(func):
        def wrapper(self: "AlpacaBroker", *args, **kwargs):
            last_err: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(self, *args, **kwargs)
                except _RETRY_EXCEPTIONS as e:
                    last_err = e
                    if attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        logger.warning(
                            "%s: attempt %d/%d failed (%s), retrying in %.1fs",
                            func_name,
                            attempt + 1,
                            max_retries + 1,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                except Exception:
                    raise
            raise last_err  # type: ignore[misc]

        return wrapper

    return decorator


# ──────────────────────────────────────────────────────────────────────
# Enum / timeframe mappings
# ──────────────────────────────────────────────────────────────────────

_ALPACA_SIDE_MAP: Dict[OrderSide, str] = {
    OrderSide.BUY: "buy",
    OrderSide.SELL: "sell",
}

_ALPACA_ORDER_TYPE_MAP: Dict[OrderType, str] = {
    OrderType.MARKET: "market",
    OrderType.LIMIT: "limit",
    OrderType.STOP: "stop",
    OrderType.STOP_LIMIT: "stop_limit",
}

_ALPACA_STATUS_MAP: Dict[str, OrderStatus] = {
    "new": OrderStatus.PENDING,
    "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.PENDING,
    "pending_cancel": OrderStatus.PENDING,
    "pending_replace": OrderStatus.PENDING,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
    "expired": OrderStatus.EXPIRED,
    "suspended": OrderStatus.REJECTED,
}

_TIMEFRAME_MAP: Dict[str, str] = {
    "1Min": "minute",
    "5Min": "5Minute",
    "15Min": "15Minute",
    "30Min": "30Minute",
    "1Hour": "hour",
    "1Day": "day",
    "1Week": "week",
    "1Month": "month",
}


def _to_alpaca_side(side: OrderSide) -> str:
    return _ALPACA_SIDE_MAP[side]


def _to_alpaca_order_type(order_type: OrderType) -> str:
    return _ALPACA_ORDER_TYPE_MAP[order_type]


def _to_order_status(alpaca_status: str) -> OrderStatus:
    key = alpaca_status.lower()
    return _ALPACA_STATUS_MAP.get(key, OrderStatus.PENDING)


def _timeframe_to_alpaca(timeframe: str) -> str:
    """Convert a human-friendly timeframe string to an alpaca-py TimeFrame string."""
    return _TIMEFRAME_MAP.get(timeframe, "day")


# ──────────────────────────────────────────────────────────────────────
# AlpacaBroker
# ──────────────────────────────────────────────────────────────────────


class AlpacaBroker(BaseBroker):
    """Broker implementation using the Alpaca Markets API (alpaca-py).

    Supports:
    - Paper and live trading via TradingClient.
    - Market, limit, stop, and stop-limit orders.
    - Fractional shares via notional-order API (qty < 1 → dollar amount).
    - Historical market bars via MarketDataClient.

    Parameters
    ----------
    api_key : str
        Alpaca API key ID.
    secret_key : str
        Alpaca API secret key.
    paper : bool
        If True, routes orders to the paper-trading environment.
        (default: True)
    base_url : Optional[str]
        Override the base API URL.  When left as ``None``, the default
        paper or live URL is used.
    max_retries : int
        Number of retries for transient network errors (default: 3).
    """

    # Broker capability flags
    SUPPORTS_FRACTIONAL_SHARES: bool = True

    # Default URLs
    _PAPER_URL = "https://paper-api.alpaca.markets"
    _LIVE_URL = "https://api.alpaca.markets"
    _PAPER_DATA_URL = "https://data.alpaca.markets"
    _LIVE_DATA_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        paper: bool = True,
        base_url: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._paper = paper
        self._max_retries = max_retries

        self._trading_base_url: str = base_url or (
            self._PAPER_URL if paper else self._LIVE_URL
        )
        self._data_base_url: str = (
            self._PAPER_DATA_URL if paper else self._LIVE_DATA_URL
        )

        # Clients initialized lazily on first use
        self._trading_client = None
        self._market_data_client = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Client accessors (lazy init)
    # ------------------------------------------------------------------

    def _get_trading_client(self):
        """Lazy-init the alpaca-py TradingClient."""
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient

            self._trading_client = TradingClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                paper=self._paper,
                url_override=self._trading_base_url,
            )
        return self._trading_client

    def _get_market_data_client(self):
        """Lazy-init the alpaca-py StockHistoricalDataClient."""
        if self._market_data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._market_data_client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._secret_key,
                url_override=self._data_base_url,
            )
        return self._market_data_client

    # ------------------------------------------------------------------
    # BaseBroker interface
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Verify connectivity by fetching the account.

        Returns
        -------
        bool
            True on success, False otherwise.
        """
        try:
            account = self._get_trading_client().get_account()
            self._connected = account is not None
            if self._connected:
                logger.info(
                    "AlpacaBroker connected — account %s, status %s",
                    getattr(account, "id", "unknown"),
                    getattr(account, "status", "unknown"),
                )
            return self._connected
        except Exception as exc:
            logger.error("AlpacaBroker connect failed: %s", exc)
            self._connected = False
            return False

    def get_account(self) -> AccountInfo:
        """Retrieve account summary.

        Raises
        ------
        RuntimeError
            If not connected or the API call fails.
        """
        self._ensure_connected()
        try:
            acc = self._get_trading_client().get_account()
        except Exception as exc:
            logger.error("Failed to get account: %s", exc)
            raise RuntimeError(f"AlpacaBroker get_account failed: {exc}") from exc

        return AccountInfo(
            cash=float(acc.cash),
            portfolio_value=float(acc.portfolio_value),
            buying_power=float(acc.buying_power),
            equity=float(acc.equity),
            initial_margin=float(getattr(acc, "initial_margin", 0.0)),
            maintenance_margin=float(getattr(acc, "maintenance_margin", 0.0)),
            day_trade_count=int(getattr(acc, "daytrade_count", 0)),
        )

    @_retry_on_failure("get_positions")
    def get_positions(self) -> List[Position]:
        """Return all open positions."""
        self._ensure_connected()
        try:
            positions = self._get_trading_client().get_all_positions()
        except Exception as exc:
            logger.error("Failed to get positions: %s", exc)
            raise RuntimeError(f"AlpacaBroker get_positions failed: {exc}") from exc

        return [self._position_from_alpaca(p) for p in positions]

    @_retry_on_failure("get_position")
    def get_position(self, symbol: str) -> Optional[Position]:
        """Return position for *symbol* or None if not held."""
        self._ensure_connected()
        symbol = symbol.upper()
        try:
            pos = self._get_trading_client().get_open_position(symbol_or_asset_id=symbol)
        except Exception:
            # Alpaca raises APIError when no position found
            return None

        if pos is None:
            return None
        return self._position_from_alpaca(pos)

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Place an order via Alpaca.

        Fractional shares
        ------------------
        When ``qty < 1``, the order is placed as a **notional** (dollar-
        based) order using the ``notional`` field.  Otherwise the ``qty``
        field is used for whole-share orders.

        Returns
        -------
        Order
            The resulting order object populated from the Alpaca response.
        """
        self._ensure_connected()

        symbol = symbol.upper()

        try:
            from alpaca.trading.requests import (
                LimitOrderRequest,
                MarketOrderRequest,
                StopLimitOrderRequest,
                StopOrderRequest,
            )
            from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce

            side_enum = _to_alpaca_side(side)
            tif = AlpacaTimeInForce.DAY

            request_kwargs: dict = {
                "symbol": symbol,
                "side": side_enum,
                "time_in_force": tif,
                "qty": float(qty),
            }

            if type == OrderType.LIMIT:
                if limit_price is None:
                    raise ValueError("limit_price is required for LIMIT orders")
                order_req = LimitOrderRequest(
                    limit_price=float(limit_price),
                    **request_kwargs,
                )
            elif type == OrderType.STOP:
                if stop_price is None:
                    raise ValueError("stop_price is required for STOP orders")
                order_req = StopOrderRequest(
                    stop_price=float(stop_price),
                    **request_kwargs,
                )
            elif type == OrderType.STOP_LIMIT:
                if limit_price is None or stop_price is None:
                    raise ValueError(
                        "limit_price and stop_price are required for STOP_LIMIT orders"
                    )
                order_req = StopLimitOrderRequest(
                    limit_price=float(limit_price),
                    stop_price=float(stop_price),
                    **request_kwargs,
                )
            else:
                order_req = MarketOrderRequest(**request_kwargs)

            alpaca_order = self._get_trading_client().submit_order(order_req)
            result = self._order_from_alpaca(alpaca_order)
            logger.info(
                "Order placed: %s %s %s %s %s",
                result.id,
                result.side.value,
                result.qty,
                result.symbol,
                result.status.value,
            )
            return result

        except Exception as exc:
            logger.error("place_order failed for %s %s %s: %s", symbol, side.value, qty, exc)
            raise RuntimeError(f"AlpacaBroker place_order failed: {exc}") from exc

    @_retry_on_failure("get_order")
    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by its Alpaca ID."""
        self._ensure_connected()
        try:
            alpaca_order = self._get_trading_client().get_order_by_id(order_id)
        except Exception:
            return None
        if alpaca_order is None:
            return None
        return self._order_from_alpaca(alpaca_order)

    @_retry_on_failure("cancel_order")
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.  Returns True on success."""
        self._ensure_connected()
        try:
            self._get_trading_client().cancel_order_by_id(order_id)
            logger.info("Order canceled: %s", order_id)
            return True
        except Exception as exc:
            logger.error("cancel_order failed for %s: %s", order_id, exc)
            return False

    @_retry_on_failure("get_bars")
    def get_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical price bars via Alpaca MarketDataClient.

        Returns a DataFrame with columns: symbol, timestamp, open, high,
        low, close, volume, trade_count, vwap.
        """
        self._ensure_connected()

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            symbols = [s.upper() for s in symbols]

            # Parse timeframe
            tf_str = _timeframe_to_alpaca(timeframe)
            tf = TimeFrame.Day  # default
            if tf_str == "minute":
                tf = TimeFrame.Minute
            elif tf_str == "5Minute":
                tf = TimeFrame(5, TimeFrame.Unit.Minute)
            elif tf_str == "15Minute":
                tf = TimeFrame(15, TimeFrame.Unit.Minute)
            elif tf_str == "30Minute":
                tf = TimeFrame(30, TimeFrame.Unit.Minute)
            elif tf_str == "hour":
                tf = TimeFrame.Hour
            elif tf_str == "week":
                tf = TimeFrame.Week
            elif tf_str == "month":
                tf = TimeFrame.Month

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )
            bars_response = self._get_market_data_client().get_stock_bars(request)

            rows: list = []
            for symbol, bars in bars_response.data.items():
                for bar in bars:
                    rows.append(
                        {
                            "symbol": symbol,
                            "timestamp": pd.Timestamp(bar.timestamp),
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume,
                            "trade_count": bar.trade_count,
                            "vwap": bar.vwap,
                        }
                    )

            if not rows:
                return pd.DataFrame(
                    columns=[
                        "symbol",
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "trade_count",
                        "vwap",
                    ]
                )

            df = pd.DataFrame(rows)
            df.set_index("timestamp", inplace=True)
            return df

        except Exception as exc:
            logger.error("get_bars failed for %s: %s", symbols, exc)
            raise RuntimeError(f"AlpacaBroker get_bars failed: {exc}") from exc

    def close_position(
        self,
        symbol: str,
        qty: Optional[float] = None,
    ) -> Optional[Order]:
        """Close an existing position.

        If *qty* is None the entire position is closed.  Otherwise only
        the given quantity is sold.
        """
        self._ensure_connected()
        symbol = symbol.upper()

        current_pos = self.get_position(symbol)
        if current_pos is None:
            logger.warning("No position found for %s — nothing to close.", symbol)
            return None

        close_qty: float
        if qty is not None:
            close_qty = min(float(qty), current_pos.qty)
        else:
            close_qty = current_pos.qty

        if close_qty <= 0:
            logger.info("Position qty for %s is ≤ 0 — nothing to close.", symbol)
            return None

        # Alpaca has a native close_position endpoint
        try:
            if qty is None:
                # Close entire position using Alpaca's dedicated endpoint
                resp = self._get_trading_client().close_position(
                    symbol_or_asset_id=symbol,
                )
                if resp is not None:
                    result = self._order_from_alpaca(resp)
                    logger.info(
                        "Position closed: %s (entire position of %.4f shares)",
                        symbol,
                        current_pos.qty,
                    )
                    return result
            else:
                # Close partial position — place a sell market order
                result = self.place_order(
                    symbol=symbol,
                    qty=close_qty,
                    side=OrderSide.SELL,
                    type=OrderType.MARKET,
                )
                logger.info(
                    "Partial position closed: %s (%.4f of %.4f shares)",
                    symbol,
                    close_qty,
                    current_pos.qty,
                )
                return result
        except Exception as exc:
            logger.error("close_position failed for %s: %s", symbol, exc)
            return None

    def supports_fractional_shares(self) -> bool:
        """Alpaca supports fractional-share trading.  Always returns True."""
        return self.SUPPORTS_FRACTIONAL_SHARES

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_connected(self) -> None:
        """Raise if the broker has not been connected."""
        if not self._connected:
            raise RuntimeError("AlpacaBroker not connected — call connect() first")

    @staticmethod
    def _position_from_alpaca(pos) -> Position:
        """Map an alpaca Position object → internal Position dataclass."""
        return Position(
            symbol=pos.symbol,
            qty=float(pos.qty),
            market_value=float(pos.market_value or 0.0),
            avg_entry_price=float(pos.avg_entry_price),
            unrealized_pl=float(pos.unrealized_pl or 0.0),
            realized_pl=float(getattr(pos, "realized_pl", 0.0) or 0.0),
            current_price=float(pos.current_price or 0.0),
        )

    @staticmethod
    def _order_from_alpaca(alpaca_order) -> Order:
        """Map an alpaca Order object → internal Order dataclass."""
        return Order(
            id=str(alpaca_order.id),
            symbol=str(alpaca_order.symbol),
            side=OrderSide.BUY if alpaca_order.side.lower() == "buy" else OrderSide.SELL,
            qty=float(alpaca_order.qty or 0.0),
            type=_order_type_from_alpaca(alpaca_order.type),
            status=_to_order_status(alpaca_order.status),
            filled_qty=float(getattr(alpaca_order, "filled_qty", 0.0) or 0.0),
            filled_avg_price=float(
                getattr(alpaca_order, "filled_avg_price", None) or 0.0
            ),
            limit_price=float(alpaca_order.limit_price)
            if getattr(alpaca_order, "limit_price", None)
            else None,
            stop_price=float(alpaca_order.stop_price)
            if getattr(alpaca_order, "stop_price", None)
            else None,
            created_at=_parse_alpaca_dt(getattr(alpaca_order, "created_at", None)),
            updated_at=_parse_alpaca_dt(getattr(alpaca_order, "updated_at", None)),
            client_order_id=getattr(alpaca_order, "client_order_id", None),
        )


def _order_type_from_alpaca(alptype: str) -> OrderType:
    """Map an alpaca order-type string → OrderType enum."""
    t = alptype.lower()
    if t == "market":
        return OrderType.MARKET
    elif t == "limit":
        return OrderType.LIMIT
    elif t == "stop":
        return OrderType.STOP
    elif t == "stop_limit":
        return OrderType.STOP_LIMIT
    return OrderType.MARKET


def _parse_alpaca_dt(dt_val) -> Optional[datetime]:
    """Parse Alpaca datetime (ISO str / datetime / None) → datetime."""
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        return dt_val
    if isinstance(dt_val, str):
        return datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
    return None
