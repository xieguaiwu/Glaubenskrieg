from __future__ import annotations

import pytest

from src.execution.account import TradingAccount
from src.execution.base_broker import (
    AccountInfo,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

from .conftest import MockBroker


class TestTradingAccountInit:
    def test_initial_capital(self) -> None:
        acc = TradingAccount(initial_capital=50000.0)
        assert acc.cash == 50000.0
        assert acc.initial_capital == 50000.0
        assert acc.get_total_value() == 50000.0
        assert acc.get_realized_pnl() == 0.0
        assert acc.get_unrealized_pnl() == 0.0

    def test_default_capital(self) -> None:
        acc = TradingAccount()
        assert acc.cash == 10000.0

    def test_no_positions_initially(self) -> None:
        acc = TradingAccount()
        assert len(acc.positions) == 0
        assert acc.get_position("AAPL") is None


class TestApplyFill:
    @pytest.fixture
    def account(self) -> TradingAccount:
        return TradingAccount(initial_capital=20000.0)

    def _make_order(
        self,
        symbol: str = "AAPL",
        qty: float = 10.0,
        side: OrderSide = OrderSide.BUY,
        filled_qty: float = 0.0,
        filled_avg_price: float = 0.0,
    ) -> Order:
        return Order(
            id="test-1",
            symbol=symbol,
            side=side,
            qty=qty,
            type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_qty=filled_qty,
            filled_avg_price=filled_avg_price,
        )

    def test_buy_decreases_cash_and_adds_position(self, account: TradingAccount) -> None:
        order = self._make_order(
            symbol="AAPL", qty=10.0, side=OrderSide.BUY,
            filled_qty=10.0, filled_avg_price=150.0,
        )
        account.apply_fill(order)

        assert account.cash == 20000.0 - 1500.0  # 18500.0
        pos = account.get_position("AAPL")
        assert pos is not None
        assert pos.qty == 10.0
        assert pos.avg_entry_price == 150.0

    def test_sell_increases_cash_and_removes_position(self, account: TradingAccount) -> None:
        buy = self._make_order(
            symbol="AAPL", qty=10.0, side=OrderSide.BUY,
            filled_qty=10.0, filled_avg_price=100.0,
        )
        account.apply_fill(buy)

        sell = self._make_order(
            symbol="AAPL", qty=10.0, side=OrderSide.SELL,
            filled_qty=10.0, filled_avg_price=110.0,
        )
        account.apply_fill(sell)

        assert account.cash == 20000.0 - 1000.0 + 1100.0  # 20100.0
        assert account.get_position("AAPL") is None
        assert account.get_realized_pnl() == pytest.approx(100.0)

    def test_sell_partial_position(self, account: TradingAccount) -> None:
        buy = self._make_order(
            symbol="AAPL", qty=10.0, side=OrderSide.BUY,
            filled_qty=10.0, filled_avg_price=100.0,
        )
        account.apply_fill(buy)

        sell = self._make_order(
            symbol="AAPL", qty=5.0, side=OrderSide.SELL,
            filled_qty=5.0, filled_avg_price=120.0,
        )
        account.apply_fill(sell)

        assert account.cash == 20000.0 - 1000.0 + 600.0  # 19600.0
        pos = account.get_position("AAPL")
        assert pos is not None
        assert pos.qty == 5.0
        assert pos.avg_entry_price == 100.0
        assert account.get_realized_pnl() == pytest.approx(100.0)

    def test_fractional_share_qty(self, account: TradingAccount) -> None:
        order = self._make_order(
            symbol="AAPL", qty=0.5, side=OrderSide.BUY,
            filled_qty=0.5, filled_avg_price=200.0,
        )
        account.apply_fill(order)

        assert account.cash == 20000.0 - 100.0  # 19900.0
        pos = account.get_position("AAPL")
        assert pos is not None
        assert pos.qty == 0.5
        assert pos.avg_entry_price == 200.0

    def test_fractional_sell(self, account: TradingAccount) -> None:
        buy = self._make_order(
            symbol="TSLA", qty=1.5, side=OrderSide.BUY,
            filled_qty=1.5, filled_avg_price=250.0,
        )
        account.apply_fill(buy)

        sell = self._make_order(
            symbol="TSLA", qty=0.7, side=OrderSide.SELL,
            filled_qty=0.7, filled_avg_price=260.0,
        )
        account.apply_fill(sell)

        pos = account.get_position("TSLA")
        assert pos is not None
        assert pos.qty == pytest.approx(0.8)
        assert account.get_realized_pnl() == pytest.approx(7.0)

    def test_uses_filled_qty_when_positive(self, account: TradingAccount) -> None:
        order = self._make_order(
            symbol="AAPL", qty=100.0, side=OrderSide.BUY,
            filled_qty=10.0, filled_avg_price=150.0,
        )
        account.apply_fill(order)
        assert account.cash == 20000.0 - 1500.0
        assert account.get_position("AAPL").qty == 10.0  # type: ignore[union-attr]

    def test_skip_zero_fill(self, account: TradingAccount) -> None:
        order = self._make_order(
            symbol="AAPL", qty=10.0, side=OrderSide.BUY,
            filled_qty=0.0, filled_avg_price=0.0,
        )
        account.apply_fill(order)
        assert account.cash == 20000.0
        assert account.get_position("AAPL") is None


class TestPositionConcentration:
    def test_single_position(self) -> None:
        acc = TradingAccount(initial_capital=10000.0)
        order = Order(
            id="1", symbol="AAPL", side=OrderSide.BUY, qty=10.0,
            type=OrderType.MARKET, status=OrderStatus.FILLED,
            filled_qty=10.0, filled_avg_price=150.0,
        )
        acc.apply_fill(order)
        # position market_value = 10 * 150 = 1500, total = 8500 cash + 1500 position = 10000
        concentration = acc.get_position_concentration("AAPL")
        assert concentration == pytest.approx(1500.0 / 10000.0)

    def test_no_position_returns_zero(self) -> None:
        acc = TradingAccount()
        assert acc.get_position_concentration("AAPL") == 0.0

    def test_zero_total_value_returns_zero(self) -> None:
        acc = TradingAccount(initial_capital=0.0)
        assert acc.get_position_concentration("AAPL") == 0.0


class TestExposure:
    def test_no_positions(self) -> None:
        acc = TradingAccount()
        assert acc.get_exposure() == 0.0

    def test_with_positions(self) -> None:
        acc = TradingAccount(initial_capital=10000.0)
        for sym, qty, price in [("AAPL", 10.0, 150.0), ("MSFT", 5.0, 400.0)]:
            order = Order(
                id=sym, symbol=sym, side=OrderSide.BUY, qty=qty,
                type=OrderType.MARKET, status=OrderStatus.FILLED,
                filled_qty=qty, filled_avg_price=price,
            )
            acc.apply_fill(order)
        # gross = 10*150 + 5*400 = 1500 + 2000 = 3500
        # cash = 10000 - 1500 - 2000 = 6500
        # total = 6500 + 3500 = 10000
        # exposure = 3500 / 10000 = 0.35
        assert acc.get_exposure() == pytest.approx(3500.0 / 10000.0)


class TestUpdateFromBroker:
    def test_syncs_cash_and_positions(self) -> None:
        acc = TradingAccount(initial_capital=5000.0)
        broker = MockBroker(
            account_info=AccountInfo(
                cash=8000.0,
                portfolio_value=12000.0,
                buying_power=16000.0,
                equity=12000.0,
            ),
            positions=[
                Position(
                    symbol="AAPL", qty=5.0, market_value=750.0,
                    avg_entry_price=140.0, unrealized_pl=50.0,
                    current_price=150.0,
                ),
            ],
        )
        acc.update_from_broker(broker)

        assert acc.cash == 8000.0
        assert len(acc.positions) == 1
        pos = acc.get_position("AAPL")
        assert pos is not None
        assert pos.qty == 5.0
        assert pos.market_value == 750.0

    def test_clears_previous_positions(self) -> None:
        acc = TradingAccount(initial_capital=10000.0)
        # First add a position manually
        order = Order(
            id="1", symbol="OLD", side=OrderSide.BUY, qty=10.0,
            type=OrderType.MARKET, status=OrderStatus.FILLED,
            filled_qty=10.0, filled_avg_price=100.0,
        )
        acc.apply_fill(order)
        assert acc.get_position("OLD") is not None

        # Sync from broker with empty positions
        broker = MockBroker(positions=[])
        acc.update_from_broker(broker)

        assert len(acc.positions) == 0
        assert acc.get_position("OLD") is None


class TestBuyingPower:
    def test_returns_cash(self) -> None:
        acc = TradingAccount(initial_capital=5000.0)
        assert acc.get_buying_power() == 5000.0

    def test_after_buy_decreases(self) -> None:
        acc = TradingAccount(initial_capital=5000.0)
        order = Order(
            id="1", symbol="AAPL", side=OrderSide.BUY, qty=10.0,
            type=OrderType.MARKET, status=OrderStatus.FILLED,
            filled_qty=10.0, filled_avg_price=100.0,
        )
        acc.apply_fill(order)
        assert acc.get_buying_power() == 4000.0
