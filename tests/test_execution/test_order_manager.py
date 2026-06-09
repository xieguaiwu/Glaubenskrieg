from __future__ import annotations

import pytest

from src.execution.account import TradingAccount
from src.execution.base_broker import OrderSide, OrderStatus, OrderType
from src.execution.order_manager import OrderManager

from .conftest import MockBroker


class TestExecuteSignal:
    @pytest.fixture
    def om(self, mock_broker: MockBroker) -> OrderManager:
        return OrderManager(broker=mock_broker, account=TradingAccount())

    def test_market_order(self, om: OrderManager, mock_broker: MockBroker) -> None:
        order = om.execute_signal(symbol="AAPL", target_qty=100.0, side=OrderSide.BUY)
        assert order is not None
        assert order.symbol == "AAPL"
        assert order.qty == 100.0
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.MARKET
        assert len(om.get_open_orders()) == 1

    def test_limit_order(self, om: OrderManager) -> None:
        order = om.execute_signal(
            symbol="MSFT", target_qty=50.0, side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
        )
        assert order is not None
        assert order.type == OrderType.LIMIT
        assert order.side == OrderSide.SELL

    def test_fractional_qty_with_support(self, om: OrderManager) -> None:
        order = om.execute_signal(symbol="AAPL", target_qty=0.25, side=OrderSide.BUY)
        assert order is not None
        assert order.qty == 0.25

    def test_fractional_qty_without_support(
        self, mock_broker_no_fractional: MockBroker,
    ) -> None:
        om = OrderManager(
            broker=mock_broker_no_fractional, account=TradingAccount(),
        )
        order = om.execute_signal(symbol="AAPL", target_qty=0.5, side=OrderSide.BUY)
        assert order is None

    def test_zero_qty_rejected(self, om: OrderManager) -> None:
        order = om.execute_signal(symbol="AAPL", target_qty=0.0, side=OrderSide.BUY)
        assert order is None

    def test_negative_qty_rejected(self, om: OrderManager) -> None:
        order = om.execute_signal(symbol="AAPL", target_qty=-1.0, side=OrderSide.BUY)
        assert order is None


class TestExecuteSignals:
    @pytest.fixture
    def om(self, mock_broker: MockBroker) -> OrderManager:
        return OrderManager(broker=mock_broker, account=TradingAccount())

    def test_batch(self, om: OrderManager) -> None:
        signals = [
            {"symbol": "AAPL", "target_qty": 10.0, "side": OrderSide.BUY},
            {"symbol": "MSFT", "target_qty": 5.0, "side": OrderSide.SELL},
            {"symbol": "GOOG", "target_qty": 2.0, "side": OrderSide.BUY},
        ]
        orders = om.execute_signals(signals)
        assert len(orders) == 3
        symbols = {o.symbol for o in orders}
        assert symbols == {"AAPL", "MSFT", "GOOG"}

    def test_batch_skips_rejected(self, om: OrderManager) -> None:
        signals = [
            {"symbol": "AAPL", "target_qty": 10.0, "side": OrderSide.BUY},
            {"symbol": "BAD", "target_qty": 0.0, "side": OrderSide.BUY},
            {"symbol": "MSFT", "target_qty": 5.0, "side": OrderSide.SELL},
        ]
        orders = om.execute_signals(signals)
        assert len(orders) == 2

    def test_batch_with_order_type(self, om: OrderManager) -> None:
        signals = [
            {
                "symbol": "AAPL", "target_qty": 10.0, "side": OrderSide.BUY,
                "order_type": OrderType.LIMIT,
            },
        ]
        orders = om.execute_signals(signals)
        assert len(orders) == 1
        assert orders[0].type == OrderType.LIMIT

    def test_batch_requires_orderside(self, om: OrderManager) -> None:
        signals = [{"symbol": "AAPL", "target_qty": 10.0, "side": "buy"}]
        with pytest.raises(TypeError, match="OrderSide"):
            om.execute_signals(signals)


class TestRebalanceToTargets:
    @pytest.fixture
    def om(self, mock_broker: MockBroker) -> OrderManager:
        return OrderManager(broker=mock_broker, account=TradingAccount(), max_position_concentration=1.0)

    def test_generates_deltas(self, om: OrderManager) -> None:
        targets = {"AAPL": 100.0, "MSFT": 50.0}
        orders = om.rebalance_to_targets(targets)
        assert len(orders) == 2
        symbols_ordered = {o.symbol for o in orders}
        assert symbols_ordered == {"AAPL", "MSFT"}

    def test_partial_rebalance_with_existing_position(
        self, om: OrderManager, mock_broker: MockBroker,
    ) -> None:
        # Build an existing position via a fill
        buy = mock_broker.place_order(symbol="AAPL", qty=30.0, side=OrderSide.BUY)
        mock_broker.simulate_fill(buy.id, fill_price=100.0)
        om._account.apply_fill(mock_broker.get_order(buy.id))  # type: ignore[arg-type]

        targets = {"AAPL": 100.0}
        orders = om.rebalance_to_targets(targets)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].qty == 70.0

    def test_sell_when_over_target(
        self, om: OrderManager, mock_broker: MockBroker,
    ) -> None:
        buy = mock_broker.place_order(symbol="MSFT", qty=80.0, side=OrderSide.BUY)
        mock_broker.simulate_fill(buy.id, fill_price=100.0)
        om._account.apply_fill(mock_broker.get_order(buy.id))  # type: ignore[arg-type]

        targets = {"MSFT": 50.0}
        orders = om.rebalance_to_targets(targets)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].qty == 30.0

    def test_skip_zero_delta(
        self, om: OrderManager, mock_broker: MockBroker,
    ) -> None:
        buy = mock_broker.place_order(symbol="AAPL", qty=10.0, side=OrderSide.BUY)
        mock_broker.simulate_fill(buy.id, fill_price=100.0)
        om._account.apply_fill(mock_broker.get_order(buy.id))  # type: ignore[arg-type]

        targets = {"AAPL": 10.0}
        orders = om.rebalance_to_targets(targets)
        assert len(orders) == 0

    def test_rebalance_sell_new_symbol(self, om: OrderManager) -> None:
        # Target a negative (short) position — not held, should generate sell
        targets = {"SHORT": -10.0}
        orders = om.rebalance_to_targets(targets)
        assert len(orders) == 1
        assert orders[0].symbol == "SHORT"
        assert orders[0].side == OrderSide.SELL
        assert orders[0].qty == 10.0

    def test_max_notional_cap(
        self, om: OrderManager, mock_broker: MockBroker,
    ) -> None:
        # Give account an AAPL position so we have a current_price reference
        buy = mock_broker.place_order(symbol="AAPL", qty=1.0, side=OrderSide.BUY)
        mock_broker.simulate_fill(buy.id, fill_price=200.0)
        om._account.apply_fill(mock_broker.get_order(buy.id))  # type: ignore[arg-type]

        targets = {"AAPL": 101.0}  # delta = 100 shares, $200 each = $20K notional
        orders = om.rebalance_to_targets(targets, max_notional_per_order=5000.0)
        assert len(orders) == 1
        # capped: 5000 / 200 = 25 shares
        assert orders[0].qty == 25.0


class TestOrderTracking:
    @pytest.fixture
    def om(self, mock_broker: MockBroker) -> OrderManager:
        return OrderManager(broker=mock_broker, account=TradingAccount())

    def test_cancel_all_open(self, om: OrderManager, mock_broker: MockBroker) -> None:
        om.execute_signal(symbol="AAPL", target_qty=10.0, side=OrderSide.BUY)
        om.execute_signal(symbol="MSFT", target_qty=5.0, side=OrderSide.BUY)
        om.execute_signal(symbol="GOOG", target_qty=1.0, side=OrderSide.SELL)

        assert len(om.get_open_orders()) == 3
        cancelled = om.cancel_all_open()
        assert cancelled == 3
        assert len(om.get_open_orders()) == 0

    def test_open_orders_excludes_filled(
        self, om: OrderManager, mock_broker: MockBroker,
    ) -> None:
        order1 = om.execute_signal(symbol="AAPL", target_qty=10.0, side=OrderSide.BUY)
        order2 = om.execute_signal(symbol="MSFT", target_qty=5.0, side=OrderSide.BUY)
        assert order1 is not None and order2 is not None

        mock_broker.simulate_fill(order1.id, fill_price=100.0)
        open_orders = om.get_open_orders()
        assert len(open_orders) == 1
        assert open_orders[0].id == order2.id

    def test_order_history(self, om: OrderManager) -> None:
        om.execute_signal(symbol="A", target_qty=1.0, side=OrderSide.BUY)
        om.execute_signal(symbol="B", target_qty=2.0, side=OrderSide.BUY)
        om.execute_signal(symbol="C", target_qty=3.0, side=OrderSide.BUY)

        history = om.get_order_history(limit=2)
        assert len(history) == 2
        assert history[0].symbol == "B"
        assert history[1].symbol == "C"

    def test_cancel_all_empty(self, om: OrderManager) -> None:
        assert om.cancel_all_open() == 0


class TestFillPropagation:
    def test_fill_applied_to_account(
        self, mock_broker: MockBroker,
    ) -> None:
        account = TradingAccount(initial_capital=10000.0)
        om = OrderManager(broker=mock_broker, account=account)

        order = om.execute_signal(symbol="AAPL", target_qty=10.0, side=OrderSide.BUY)
        assert order is not None

        mock_broker.simulate_fill(order.id, fill_price=150.0)
        # get_open_orders triggers _sync_order_states which applies fills
        om.get_open_orders()

        assert account.cash == 10000.0 - 1500.0
        assert account.get_position("AAPL") is not None
        assert account.get_position("AAPL").qty == 10.0  # type: ignore[union-attr]
