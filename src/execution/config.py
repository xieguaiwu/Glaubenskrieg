"""Trading execution configuration loaded from YAML.

Provides a TradingConfig dataclass and a load_trading_config() factory
that reads the nested YAML structure from configs/execution_default.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    """Broker connection settings."""

    name: str = "alpaca"
    paper: bool = True
    api_key: str = ""
    secret_key: str = ""


@dataclass
class AccountConfig:
    """Account constraints and capital settings."""

    initial_capital: float = 10000.0
    max_leverage: float = 1.0
    max_position_concentration: float = 0.10
    max_gross_exposure: float = 0.90
    fractional_shares: bool = True


@dataclass
class RiskConfig:
    """Risk-management thresholds."""

    max_daily_loss: float = 0.03
    max_drawdown: float = 0.15
    min_signal_confidence: float = 0.01
    max_correlation: float = 0.90


@dataclass
class RebalanceConfig:
    """Rebalancing strategy parameters."""

    method: str = "threshold"
    threshold: float = 0.05
    frequency: str = "daily"
    order_type: str = "market"


@dataclass
class TradingConfig:
    """Top-level execution configuration loaded from YAML.

    Attributes
    ----------
    broker : BrokerConfig
        Broker connection settings (name, paper mode, credentials).
    account : AccountConfig
        Account capital & leverage constraints.
    risk : RiskConfig
        Risk-management thresholds.
    rebalance : RebalanceConfig
        Rebalancing strategy parameters.
    """

    broker: BrokerConfig = field(default_factory=BrokerConfig)
    account: AccountConfig = field(default_factory=AccountConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    rebalance: RebalanceConfig = field(default_factory=RebalanceConfig)


def load_trading_config(path: str = "configs/execution_default.yaml") -> TradingConfig:
    """Load execution configuration from a YAML file.

    If the file does not exist a warning is logged and default values are
    returned.  The YAML is expected to have the following nested structure::

        execution:
          broker:
            name: ...
            paper: ...
            ...
          account:
            ...
          ...
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load trading configuration. "
            "Install it with: pip install pyyaml"
        )

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("Config file '%s' not found — using defaults.", path)
        return TradingConfig()

    # Navigate the nested structure:  execution.broker / execution.account / …
    exec_cfg = (data or {}).get("execution", {})

    broker_data = exec_cfg.get("broker", {})
    account_data = exec_cfg.get("account", {})
    risk_data = exec_cfg.get("risk", {})
    rebalance_data = exec_cfg.get("rebalance", {})

    return TradingConfig(
        broker=BrokerConfig(
            name=broker_data.get("name", "alpaca"),
            paper=broker_data.get("paper", True),
            api_key=broker_data.get("api_key", ""),
            secret_key=broker_data.get("secret_key", ""),
        ),
        account=AccountConfig(
            initial_capital=account_data.get("initial_capital", 10000.0),
            max_leverage=account_data.get("max_leverage", 1.0),
            max_position_concentration=account_data.get(
                "max_position_concentration", 0.10
            ),
            max_gross_exposure=account_data.get("max_gross_exposure", 0.90),
            fractional_shares=account_data.get("fractional_shares", True),
        ),
        risk=RiskConfig(
            max_daily_loss=risk_data.get("max_daily_loss", 0.03),
            max_drawdown=risk_data.get("max_drawdown", 0.15),
            min_signal_confidence=risk_data.get("min_signal_confidence", 0.01),
            max_correlation=risk_data.get("max_correlation", 0.90),
        ),
        rebalance=RebalanceConfig(
            method=rebalance_data.get("method", "threshold"),
            threshold=rebalance_data.get("threshold", 0.05),
            frequency=rebalance_data.get("frequency", "daily"),
            order_type=rebalance_data.get("order_type", "market"),
        ),
    )
