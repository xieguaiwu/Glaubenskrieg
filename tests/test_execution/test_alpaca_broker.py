"""Mock tests for AlpacaBroker — no real API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.execution.alpaca_broker import AlpacaBroker, _parse_alpaca_dt
from src.execution.base_broker import (
    AccountInfo,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)


@pytest.fixture
def broker() -> AlpacaBroker:
    """Return an AlpacaBroker with fake keys, NOT connected."""
    return AlpacaBroker(api_key="test_key", secret_key="test_secret", paper=True)


@pytest.fixture
def connected_broker(broker: AlpacaBroker) -> AlpacaBroker:
    """Return an AlpacaBroker whose connect() already succeeded."""
    broker._connected = True
    return broker


# ═══════════════════════════════════════════════════════════════════════
# Initialisation
# ═══════════════════════════════════════════════════════════════════════


def test_default_paper_mode():
    b = AlpacaBroker(api_key="k", secret_key="s")
    assert b._paper is True
    assert b._trading_base_url == AlpacaBroker._PAPER_URL


def test_live_mode():
    b = AlpacaBroker(api_key="k", secret_key="s", paper=False)
    assert b._paper is False
    assert b._trading_base_url == AlpacaBroker._LIVE_URL


def test_custom_base_url():
    b = AlpacaBroker(api_key="k", secret_key="s", paper=True, base_url="https://custom.api")
    assert b._trading_base_url == "https://custom.api"


def test_not_connected_initially(broker: AlpacaBroker):
    assert broker._connected is False
    assert broker._trading_client is None
    assert broker._market_data_client is None


# ═══════════════════════════════════════════════════════════════════════
# connect()
# ═══════════════════════════════════════════════════════════════════════


def test_connect_success(broker: AlpacaBroker):
    mock_account = MagicMock()
    mock_account.id = "acc-123"
    mock_account.status = "ACTIVE"

    with patch.object(broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_account.return_value = mock_account
        result = broker.connect()

    assert result is True
    assert broker._connected is True


def test_connect_failure(broker: AlpacaBroker):
    with patch.object(broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_account.side_effect = ConnectionError("no network")
        result = broker.connect()

    assert result is False
    assert broker._connected is False


def test_connect_raises_on_other_error(broker: AlpacaBroker):
    with patch.object(broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_account.side_effect = ValueError("wtf")
        result = broker.connect()

    assert result is False


# ═══════════════════════════════════════════════════════════════════════
# get_account()
# ═══════════════════════════════════════════════════════════════════════


def test_get_account(connected_broker: AlpacaBroker):
    mock_acc = MagicMock()
    mock_acc.cash = "100000.50"
    mock_acc.portfolio_value = "250000.00"
    mock_acc.buying_power = "200000.00"
    mock_acc.equity = "250000.00"
    mock_acc.initial_margin = "50000.00"
    mock_acc.maintenance_margin = "25000.00"
    mock_acc.daytrade_count = 3

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_account.return_value = mock_acc
        acc = connected_broker.get_account()

    assert isinstance(acc, AccountInfo)
    assert acc.cash == 100000.50
    assert acc.portfolio_value == 250000.00
    assert acc.buying_power == 200000.00
    assert acc.day_trade_count == 3


def test_get_account_not_connected(broker: AlpacaBroker):
    with pytest.raises(RuntimeError, match="not connected"):
        broker.get_account()


# ═══════════════════════════════════════════════════════════════════════
# get_positions()
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_position(symbol: str, qty: str = "10") -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.qty = qty
    p.market_value = "1500.00"
    p.avg_entry_price = "140.00"
    p.unrealized_pl = "100.00"
    p.realized_pl = "0.00"
    p.current_price = "150.00"
    return p


def test_get_positions(connected_broker: AlpacaBroker):
    mock_positions = [_make_mock_position("AAPL"), _make_mock_position("MSFT", "5")]

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_all_positions.return_value = mock_positions
        positions = connected_broker.get_positions()

    assert len(positions) == 2
    assert all(isinstance(p, Position) for p in positions)
    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 10.0
    assert positions[1].symbol == "MSFT"
    assert positions[1].qty == 5.0


def test_get_positions_empty(connected_broker: AlpacaBroker):
    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_all_positions.return_value = []
        positions = connected_broker.get_positions()

    assert positions == []


# ═══════════════════════════════════════════════════════════════════════
# get_position()
# ═══════════════════════════════════════════════════════════════════════


def test_get_position_found(connected_broker: AlpacaBroker):
    mock_pos = _make_mock_position("AAPL")

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_open_position.return_value = mock_pos
        pos = connected_broker.get_position("AAPL")

    assert pos is not None
    assert pos.symbol == "AAPL"
    assert pos.qty == 10.0


def test_get_position_not_found(connected_broker: AlpacaBroker):
    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_open_position.side_effect = Exception("not found")
        pos = connected_broker.get_position("NOPE")

    assert pos is None


# ═══════════════════════════════════════════════════════════════════════
# place_order()
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_alpaca_order(
    oid: str = "ord-1",
    symbol: str = "AAPL",
    side: str = "buy",
    qty: str = "10",
    order_type: str = "market",
    status: str = "filled",
    filled_qty: str = "10",
    filled_avg_price: str = "150.00",
    limit_price=None,
    stop_price=None,
) -> MagicMock:
    o = MagicMock()
    o.id = oid
    o.symbol = symbol
    o.side = side
    o.qty = qty
    o.type = order_type
    o.status = status
    o.filled_qty = filled_qty
    o.filled_avg_price = filled_avg_price
    o.limit_price = limit_price
    o.stop_price = stop_price
    o.created_at = "2025-01-01T12:00:00Z"
    o.updated_at = "2025-01-01T12:00:01Z"
    o.client_order_id = None
    return o


def test_place_market_order(connected_broker: AlpacaBroker):
    mock_order = _make_mock_alpaca_order()

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.submit_order.return_value = mock_order
        order = connected_broker.place_order(
            symbol="AAPL", qty=10.0, side=OrderSide.BUY, type=OrderType.MARKET
        )

    assert isinstance(order, Order)
    assert order.id == "ord-1"
    assert order.symbol == "AAPL"
    assert order.side == OrderSide.BUY
    assert order.qty == 10.0
    assert order.type == OrderType.MARKET
    assert order.status == OrderStatus.FILLED
    assert order.filled_qty == 10.0
    assert order.filled_avg_price == 150.00


def test_place_limit_order(connected_broker: AlpacaBroker):
    mock_order = _make_mock_alpaca_order(
        order_type="limit", limit_price="155.00", status="accepted", side="sell"
    )

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.submit_order.return_value = mock_order
        order = connected_broker.place_order(
            symbol="AAPL",
            qty=5.0,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            limit_price=155.00,
        )

    assert order.type == OrderType.LIMIT
    assert order.limit_price == 155.00
    assert order.side == OrderSide.SELL


def test_place_limit_order_missing_price(connected_broker: AlpacaBroker):
    with pytest.raises(RuntimeError, match="limit_price is required"):
        connected_broker.place_order(
            symbol="AAPL", qty=5.0, side=OrderSide.BUY, type=OrderType.LIMIT
        )


def test_place_stop_order(connected_broker: AlpacaBroker):
    mock_order = _make_mock_alpaca_order(order_type="stop", stop_price="148.00", status="accepted")

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.submit_order.return_value = mock_order
        order = connected_broker.place_order(
            symbol="AAPL",
            qty=10.0,
            side=OrderSide.SELL,
            type=OrderType.STOP,
            stop_price=148.00,
        )

    assert order.type == OrderType.STOP
    assert order.stop_price == 148.00


def test_place_stop_order_missing_price(connected_broker: AlpacaBroker):
    with pytest.raises(RuntimeError, match="stop_price is required"):
        connected_broker.place_order(
            symbol="AAPL", qty=10.0, side=OrderSide.SELL, type=OrderType.STOP
        )


def test_place_stop_limit_order(connected_broker: AlpacaBroker):
    mock_order = _make_mock_alpaca_order(
        order_type="stop_limit", limit_price="147.00", stop_price="148.00", status="accepted"
    )

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.submit_order.return_value = mock_order
        order = connected_broker.place_order(
            symbol="AAPL",
            qty=10.0,
            side=OrderSide.SELL,
            type=OrderType.STOP_LIMIT,
            limit_price=147.00,
            stop_price=148.00,
        )

    assert order.type == OrderType.STOP_LIMIT
    assert order.limit_price == 147.00
    assert order.stop_price == 148.00


def test_place_stop_limit_missing_prices(connected_broker: AlpacaBroker):
    with pytest.raises(RuntimeError, match="limit_price and stop_price are required"):
        connected_broker.place_order(
            symbol="AAPL", qty=10.0, side=OrderSide.SELL, type=OrderType.STOP_LIMIT
        )


def test_place_fractional_order_uses_qty(connected_broker: AlpacaBroker):
    """Fractional qty (<1) uses the qty field — Alpaca supports fractional shares natively."""
    mock_order = _make_mock_alpaca_order(qty="0.5", filled_qty="0.5")

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.submit_order.return_value = mock_order
        order = connected_broker.place_order(
            symbol="AAPL",
            qty=0.5,
            side=OrderSide.BUY,
            type=OrderType.MARKET,
        )

    assert order.qty == 0.5
    # Verify that qty (not notional) was used in the request
    call_args = mock_tc.return_value.submit_order.call_args
    order_req = call_args[0][0]
    assert hasattr(order_req, "qty")
    assert order_req.qty == 0.5


# ═══════════════════════════════════════════════════════════════════════
# get_order()
# ═══════════════════════════════════════════════════════════════════════


def test_get_order_found(connected_broker: AlpacaBroker):
    mock_order = _make_mock_alpaca_order()

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_order_by_id.return_value = mock_order
        order = connected_broker.get_order("ord-1")

    assert order is not None
    assert order.id == "ord-1"


def test_get_order_not_found(connected_broker: AlpacaBroker):
    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.get_order_by_id.side_effect = Exception("not found")
        order = connected_broker.get_order("ord-999")

    assert order is None


# ═══════════════════════════════════════════════════════════════════════
# cancel_order()
# ═══════════════════════════════════════════════════════════════════════


def test_cancel_order_success(connected_broker: AlpacaBroker):
    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.cancel_order_by_id.return_value = None
        result = connected_broker.cancel_order("ord-1")

    assert result is True


def test_cancel_order_failure(connected_broker: AlpacaBroker):
    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.cancel_order_by_id.side_effect = Exception("order not found")
        result = connected_broker.cancel_order("ord-999")

    assert result is False


# ═══════════════════════════════════════════════════════════════════════
# get_bars()
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_bar(symbol: str = "AAPL", ts: str = "2025-01-01") -> MagicMock:
    b = MagicMock()
    b.symbol = symbol
    b.timestamp = pd.Timestamp(ts)
    b.open = 150.0
    b.high = 152.0
    b.low = 149.0
    b.close = 151.0
    b.volume = 1000000.0
    b.trade_count = 5000.0
    b.vwap = 150.5
    return b


def test_get_bars(connected_broker: AlpacaBroker):
    mock_bar = _make_mock_bar()
    mock_response = MagicMock()
    mock_response.data = {"AAPL": [mock_bar]}

    with patch.object(connected_broker, "_get_market_data_client") as mock_md:
        mock_md.return_value.get_stock_bars.return_value = mock_response
        df = connected_broker.get_bars(
            symbols=["AAPL"],
            timeframe="1Day",
            start="2025-01-01",
            end="2025-01-02",
            limit=100,
        )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "AAPL"
    assert df.iloc[0]["close"] == 151.0


def test_get_bars_empty(connected_broker: AlpacaBroker):
    mock_response = MagicMock()
    mock_response.data = {"AAPL": []}

    with patch.object(connected_broker, "_get_market_data_client") as mock_md:
        mock_md.return_value.get_stock_bars.return_value = mock_response
        df = connected_broker.get_bars(symbols=["AAPL"])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_get_bars_multiple_symbols(connected_broker: AlpacaBroker):
    mock_response = MagicMock()
    mock_response.data = {
        "AAPL": [_make_mock_bar("AAPL")],
        "MSFT": [_make_mock_bar("MSFT")],
    }

    with patch.object(connected_broker, "_get_market_data_client") as mock_md:
        mock_md.return_value.get_stock_bars.return_value = mock_response
        df = connected_broker.get_bars(symbols=["AAPL", "MSFT"])

    assert len(df) == 2
    assert set(df["symbol"].unique()) == {"AAPL", "MSFT"}


# ═══════════════════════════════════════════════════════════════════════
# close_position()
# ═══════════════════════════════════════════════════════════════════════


def test_close_position_no_position(connected_broker: AlpacaBroker):
    with patch.object(connected_broker, "get_position", return_value=None):
        result = connected_broker.close_position("NOPE")

    assert result is None


def test_close_position_full(connected_broker: AlpacaBroker):
    pos = Position(
        symbol="AAPL",
        qty=10.0,
        market_value=1500.0,
        avg_entry_price=140.0,
        unrealized_pl=100.0,
        current_price=150.0,
    )
    mock_order = _make_mock_alpaca_order(side="sell", qty="10")

    with patch.object(connected_broker, "get_position", return_value=pos):
        with patch.object(connected_broker, "_get_trading_client") as mock_tc:
            mock_tc.return_value.close_position.return_value = mock_order
            result = connected_broker.close_position("AAPL")

    assert result is not None
    assert result.symbol == "AAPL"
    assert result.side == OrderSide.SELL


def test_close_position_partial(connected_broker: AlpacaBroker):
    pos = Position(
        symbol="AAPL",
        qty=10.0,
        market_value=1500.0,
        avg_entry_price=140.0,
        unrealized_pl=100.0,
        current_price=150.0,
    )
    mock_order = _make_mock_alpaca_order(side="sell", qty="3")

    with patch.object(connected_broker, "get_position", return_value=pos):
        with patch.object(connected_broker, "_get_trading_client") as mock_tc:
            mock_tc.return_value.submit_order.return_value = mock_order
            result = connected_broker.close_position("AAPL", qty=3.0)

    assert result is not None
    assert result.qty == 3.0


def test_close_position_zero_qty(connected_broker: AlpacaBroker):
    pos = Position(
        symbol="AAPL",
        qty=0.0,
        market_value=0.0,
        avg_entry_price=0.0,
        unrealized_pl=0.0,
        current_price=0.0,
    )

    with patch.object(connected_broker, "get_position", return_value=pos):
        result = connected_broker.close_position("AAPL")

    assert result is None


# ═══════════════════════════════════════════════════════════════════════
# supports_fractional_shares()
# ═══════════════════════════════════════════════════════════════════════


def test_supports_fractional_shares(broker: AlpacaBroker):
    assert broker.supports_fractional_shares() is True


# ═══════════════════════════════════════════════════════════════════════
# Order/Trade status mapping
# ═══════════════════════════════════════════════════════════════════════


def test_sell_order_side(connected_broker: AlpacaBroker):
    mock_order = _make_mock_alpaca_order(side="sell")

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.submit_order.return_value = mock_order
        order = connected_broker.place_order(symbol="AAPL", qty=1.0, side=OrderSide.SELL)

    assert order.side == OrderSide.SELL


def test_status_pending(connected_broker: AlpacaBroker):
    mock_order = _make_mock_alpaca_order(status="pending_new")

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.submit_order.return_value = mock_order
        order = connected_broker.place_order(symbol="AAPL", qty=1.0, side=OrderSide.BUY)

    assert order.status == OrderStatus.PENDING


def test_status_partially_filled(connected_broker: AlpacaBroker):
    mock_order = _make_mock_alpaca_order(status="partially_filled", filled_qty="3")

    with patch.object(connected_broker, "_get_trading_client") as mock_tc:
        mock_tc.return_value.submit_order.return_value = mock_order
        order = connected_broker.place_order(symbol="AAPL", qty=10.0, side=OrderSide.BUY)

    assert order.status == OrderStatus.PARTIALLY_FILLED
    assert order.filled_qty == 3.0


# ═══════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════


def test_parse_alpaca_dt_iso_string():
    result = _parse_alpaca_dt("2025-01-15T10:30:00Z")
    assert result is not None
    assert result.year == 2025
    assert result.month == 1
    assert result.day == 15


def test_parse_alpaca_dt_none():
    assert _parse_alpaca_dt(None) is None


def test_parse_alpaca_dt_datetime():
    from datetime import datetime

    dt = datetime(2025, 6, 1, 12, 0, 0)
    result = _parse_alpaca_dt(dt)
    assert result == dt
