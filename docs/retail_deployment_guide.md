# CTM Live Deployment Guide for Individual Investors

> 撰写日期：2026-05-24 | 目标市场：美股、日股、欧股
> 硬件基线：RTX 4090 24GB GPU (训练) / RTX 3060 12GB (推理) / CPU-only (备选)
> CTM推理延迟（GPU, B=1, d=128, n_loop=5）：~3ms per forward pass

---

## 一、部署概述

Glaubenskrieg CTM经过大规模训练后，个人投资者可以通过以下管道接入实时市场：

```
每日收盘数据 (OHLCV CSV)
    → Feature Engineering (features.py)
        → CTM Inference (scripts/infer.py, ~3ms/stock GPU)
            → [可选: GBDT Inference, <1ms/stock]
                → IC-Weighted Signal Fusion
                    → Ranked Buy/Sell Signals
                        → Broker API Order Execution
                            → Portfolio Management + Risk Control
```

### 为什么选择美股、日股、欧股（而非A股）

| 因素 | 美股 | 日股 | 欧股 | A股 (排除理由) |
|------|------|------|------|---------------|
| **T+0 / 做空** | ✅ T+0 + 无融券限制 | ✅ T+0 | ✅ T+0 | ❌ T+1 + 融券稀缺 |
| **Broker API成熟度** | ✅ Alpaca/IBKR | ✅ Moomoo/kabu | ✅ IBKR/Saxo | ⚠️ 仅限机构 |
| **市场历史** | ✅ 数十年 | ✅ 数十年 | ✅ 数十年 | ✅ 但政策断裂多 |
| **最小资金要求** | $0 (Alpaca) | ¥0 (kabu) | €0 (IBKR) | — |
| **PDT规则** | ⚠️ <$25k 受限 | 无 | 无 | — |
| **API文档质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | — |
| **个人投资者合法性** | ✅ 完全合法 | ✅ 完全合法 | ✅ 完全合法 | ⚠️ 跨境限制 |

**A股排除的核心原因**：
1. 个人投资者通过API直接接入A股交易受限于中国监管（非QFII的个人无法通过API交易A股）
2. 做空/融券在A股对个人投资者几乎不可用
3. T+1制度使日内策略不可行
4. CTM的Long-Short信号在A股无法完全执行

---

## 二、Broker API对比

### 按市场选择

#### 美股：Alpaca（首选）/ Interactive Brokers（大宗/全球）

| 维度 | Alpaca | Interactive Brokers (IBKR) |
|------|--------|---------------------------|
| **佣金** | $0 股票/ETF | $0.0035/股 (最低$0.35) |
| **API风格** | REST + WebSocket，Python SDK最佳 | TWS API (TCP Socket) / Web API (REST)，Python `ib_insync` |
| **设置复杂度** | ⭐ 极简 (API Key) | ⭐⭐⭐⭐ 复杂 (TWS Gateway必须持续运行) |
| **市场覆盖** | 仅美股 | 150+市场，33个国家 |
| **做空** | ✅ Easy-to-Borrow零费用 | ✅ 全面的融券库存 |
| **24/5交易** | ✅ (2025年新功能) | ✅ |
| **Paper Trading** | ✅ 全功能，无限制，免费 | ✅ 良好 |
| **最小资金** | $0 | $0 (现金账户) |
| **PDT规则** | ⚠️ <$25k 现金账户受限 | ⚠️ <$25k 现金账户受限 |
| **分股交易** | ✅ 低至$1 | ✅ |
| **数据订阅** | Algo Trader Plus $99/月 | 市场数据订阅另计 |
| **文档质量** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Python生态** | `alpaca-py` (官方SDK) | `ib_insync` (社区首选) |

**推荐**：**Alpaca** 用于美股CTM部署 — 最低摩擦、零佣金、全功能Paper Trading。

Alpaca连接示例：

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# 初始化（Paper Trading用于测试）
trading_client = TradingClient(
    api_key='PK...', 
    secret_key='SK...',
    paper=True  # False for live
)

# CTM推理后下单
for signal in ranked_signals[:10]:  # Top 10 Long
    order = MarketOrderRequest(
        symbol=signal['symbol'],
        qty=calculate_position_size(signal),
        side=OrderSide.BUY if signal['fused_signal'] > 0 else OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )
    trading_client.submit_order(order)
```

#### 日股：kabu STATION API（首选）/ Moomoo API（美股+日股）

| 维度 | kabu STATION API (三菱UFJ eSmart) | Moomoo API | 松井証券 |
|------|----------------------------------|-----------|---------|
| **佣金** | 按注文（99円~385円）或定額（550円~） | 無料（米国株） | 1日50万円以下無料 |
| **API风格** | REST + WebSocket | REST + WebSocket，Python/Java/C++/JS SDK | REST (ポーリングのみ) |
| **Python SDK** | ✅ 公式SDK | ✅ 公式SDK（5言語） | ❌ なし |
| **リアルタイム板情報** | ✅ WebSocket | ✅ WebSocket | ❌ ポーリング15秒 |
| **API利用料** | 完全無料（口座保有のみ） | 無料 | 無料 |
| **デモ環境** | ✅ (デモ口座) | ✅ (ペーパー取引) | ❌ |
| **約定通知** | ✅ WebSocketプッシュ | ✅ WebSocket | ❌ |
| **Swagger UI** | ✅ 充実 | ✅ 充実 | ❌ |
| **空売り** | ✅ | ✅ | ✅ |

**推荐**：**kabu STATION API** — 唯一提供WebSocket实时推送、完全免费、文档最完善的日股API。

kabu STATION API连接示例：

```python
import requests
import json

# 認証トークン取得
response = requests.post(
    'http://localhost:18080/kabusapi/token',
    json={'APIPassword': 'your_api_password'}
)
token = response.json()['Token']

# 注文発注
headers = {'X-API-KEY': token, 'Content-Type': 'application/json'}
order = {
    'Symbol': '7203',      # トヨタ自動車
    'Exchange': 1,          # 東証
    'Side': '2',            # 買
    'CashMargin': 1,        # 現物
    'DelivType': 2,         # 指定なし
    'FrontOrderType': 10,   # 成行
    'Price': 0,
    'Volume': 100,
}
requests.post('http://localhost:18080/kabusapi/sendorder', 
              headers=headers, json=order)
```

**数据获取**（回测用）：
- J-Quants API (JPX公式) — 無料プラン: 12週遅れデータ → 回测用に十分
- Standardプラン: ¥4,950/月 → 本番リアルタイム

#### 欧股：Interactive Brokers（唯一选择）

IBKR是覆盖欧洲所有主要交易所的唯一零售级Broker API：

| 交易所 | 市场 | 佣金 (IBKR Tiered) |
|--------|------|-------------------|
| Xetra (德国) | DAX 40, MDAX, SDAX | 0.05% (最低€1.25) |
| Euronext (法国/荷兰/比利时) | CAC 40, AEX, BEL 20 | 0.05% (最低€1.50) |
| LSE (英国) | FTSE 100, FTSE 250 | 0.05% (最低£1) |
| SIX Swiss (瑞士) | SMI | 0.05% (最低CHF 1.50) |
| BME (西班牙) | IBEX 35 | 0.05% (最低€1.25) |
| Borsa Italiana | FTSE MIB | 0.05% (最低€1.50) |

**替代选项**：Saxo Bank — OpenAPI可用，但佣金更高（0.08% vs IBKR 0.05%），外汇转换费0.25% vs IBKR 0.03%，不建议高频策略。

IBKR Python示例（欧洲股票）：

```python
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)  # TWS Gateway必须运行

# 创建合约
contract = Stock('SAP', 'XETR', 'EUR')  # SAP在Xetra
ib.qualifyContracts(contract)

# 市价单
order = MarketOrder('BUY', 100)
trade = ib.placeOrder(contract, order)

# 获取持仓
positions = ib.positions()
```

---

## 三、最小可行架构

### 硬件配置

| 配置 | 训练 | 推理 | 月成本（估算） |
|------|------|------|-------------|
| **入门级** (Hetzner AX102) | ❌ 仅推理 | ✅ CPU, batch=1, ~50 stocks | ~€70/月 |
| **标准级** (自建 RTX 4090) | ✅ 本地训练 | ✅ GPU, batch=32, ~500 stocks | ~$200/月(电费+折旧) |
| **云GPU** (Vast.ai RTX 4090) | ✅ 按需 | ✅ GPU | ~$150-290/月 连续 |
| **混合方案** ⭐ | ✅ 本地训练 (自建) | ✅ 云端推理 (Hetzner CPU) | ~$100/月 |

**推荐混合方案**（个人投资者最优）：
- 训练：本地RTX 4090（大规模回测+超参搜索）→ 训练完后保存checkpoint
- 推理：Hetzner AX102 (CPU, 64GB RAM, ~€70/月) → 每日收盘后批量推理
- CTM在CPU上的推理延迟为685ms/forward (d=128, loop=5)，每日推理500只股票约6分钟，完全可行

### 软件栈

```
Ubuntu 22.04 LTS
├── Python 3.12+ (conda/mamba环境)
├── PyTorch 2.4+ (训练用GPU, 推理用CPU)
├── Glaubenskrieg CTM (pip install -e .)
├── 数据获取：
│   ├── yfinance (免费, Yahoo Finance — 美股/欧股)
│   ├── J-Quants API (免费, JPX公式 — 日股)
│   └── Alpaca Data API v2 ($99/月 — 美股实时)
├── Broker连接：
│   ├── alpaca-py (美股)
│   ├── kabu STATION API (日股)
│   └── ib_insync (欧股，通过IBKR)
├── 调度：
│   └── cron / systemd timer (每日美东16:30 / 东京15:30 / 欧洲17:30)
├── 监控：
│   └── Grafana + Prometheus (或简单的日志+邮件通知)
└── 风险管理：
    └── 自建仓位限制 + stop-loss逻辑
```

### 每日运行时间线（以美股为例）

```
美东时间 16:00  市场收盘
     16:05  拉取当日OHLCV数据 (yfinance / Alpaca API)
     16:10  Feature Engineering (features.py)
     16:15  CTM Inference (所有股票, ~6min CPU)
     16:21  GBDT Inference (<1s)
     16:22  IC-weighted Signal Fusion
     16:23  生成 Ranked Buy/Sell Signals
     16:25  风险检查 (position limits, stop-loss, cash balance)
     16:28  提交订单至Alpaca (market-on-close for next day)
     16:30  日志记录 + 邮件通知
     ———
次日 09:30  开盘，订单执行
             (或使用24/5 trading，20:00即可执行)
```

---

## 四、资金与风险管理

### 最小启动资金

| 市场 | 最低资金 | 推荐资金 | 理由 |
|------|---------|---------|------|
| **美股** | $2,000 (现金账户) / $25,000 (保证金+日内) | **$10,000-25,000** | PDT规则：<$25k不能做日内交易(4+笔/5日)；但CTM每日换仓频率可在现金账户内规避PDT |
| **日股** | ¥100,000 (~$640) | **¥1,000,000 (~$6,400)** | 単元株制度：100株単位；丰田(7203)约¥30万/単元；小型股约¥5-10万/単元 |
| **欧股** | €1,000 | **€10,000-25,000** | IBKR无最低，但€1.25-3/笔佣金对小资金侵蚀严重 |

### 仓位管理

```python
def calculate_position_size(signal, account_equity, volatility):
    """
    CTM信号→仓位大小
    基于：波动率调整 + 等风险贡献 + 最大仓位上限
    """
    # 基础仓位 (等权重, 持仓10-30只)
    base_weight = 1.0 / max(active_positions, 10)
    
    # 波动率调整 (高波动 → 减仓)
    vol_scalar = target_vol / max(volatility, min_vol)
    
    # 信号强度调整
    signal_scalar = abs(signal['fused_signal']) / avg_signal_strength
    
    # 最终仓位 (上限10%)
    weight = min(base_weight * vol_scalar * signal_scalar, 0.10)
    
    return account_equity * weight
```

### 风险控制清单

| 检查项 | 阈值 | 行动 |
|--------|------|------|
| 最大单仓位 | 10% of NAV | 拒绝超限订单 |
| 最大总仓位 | 90% of NAV | 保留10%现金缓冲 |
| 最大行业集中度 | 30% | 拒绝同行业超3只 |
| CTM信号置信度最低值 | \|fused_signal\| < 0.01 | 跳过（信号太弱） |
| 滚动IC低于历史10分位 | 连续2周 | **暂停交易**，重新训练模型 |
| VIX > 40 (美股) / Nikkei Vol > 35 (日股) | 单日 | **减仓至50%** |
| 日损失 > 3% NAV | 单日 | **暂停交易**，次日review |
| 周损失 > 8% NAV | 单周 | **暂停交易**，手动审核 |

---

## 五、延迟与执行考量

### CTM推理延迟对交易的影响

CTM是**每日批量推理**，非高频tick级。推理延迟仅影响收盘后生成信号的速度：

| 股票数量 | CPU推理(685ms/stock, batch=1) | GPU推理(3ms/stock, batch=32) |
|---------|-------------------------------|------------------------------|
| 100 | ~69秒 | ~0.3秒 |
| 500 | ~343秒 (5.7分) | ~1.5秒 |
| 2000 | ~23分 | ~6秒 |

**对日频换仓策略而言，CPU推理完全足够**（每日收盘后有数小时窗口）。只有需要盘中实时调整时才需要GPU。

### 执行方式

| 市场 | 推荐执行时间 | 订单类型 |
|------|------------|---------|
| **美股** | 次日开盘 (MOC/LOC) 或 盘后24/5交易 | Market-On-Close / Limit-On-Close |
| **日股** | 次日寄付 (寄成行) | 成行 (寄付) |
| **欧股** | 次日开盘 | Market Order |

**注意**：美股24/5交易（Alpaca，2025年新功能）允许在20:00 ET（收盘后4小时）即可提交订单，在Blue Ocean ATS上执行——这为CTM的日频策略提供了同日执行的选项。

---

## 六、成本估算

### 月运行成本（标准配置）

| 项目 | 美股 | 日股 | 欧股 |
|------|------|------|------|
| **VPS/Cloud** | Hetzner AX102 €70 | Hetzner AX102 €70 | Hetzner AX102 €70 |
| **Broker佣金** | $0 (Alpaca) | 定額¥550 (kabu) | ~€30 (IBKR, 20笔/月) |
| **市场数据** | $99 (Alpaca Algo Trader Plus) | ¥0 (J-Quants免费) | $0 (yfinance免费) |
| **GPU训练** | $0 (本地) 或 $150 (Vast.ai) | 同左 | 同左 |
| **外汇转换** | N/A (USD计价) | N/A (JPY计价) | N/A (EUR计价，仅需IBKR多币种账户) |
| **总计** | **~$170-320/月** | **~¥620-13,000/月** | **~€100-250/月** |

### 启动成本（一次性）

| 项目 | 成本 |
|------|------|
| **GPU工作站** (自建) | $2,500-4,000 (RTX 4090 build) |
| **或云GPU** (Vast.ai 按需) | $0 (仅使用时付费) |
| **数据历史下载** | $0-100 (取决于数据源) |
| **总计** | **$0 ~ $4,000** (取决于是否自建GPU) |

---

## 七、部署检查清单

### Phase 1: Paper Trading验证 (2-4周)

- [ ] 在所选市场开设Paper Trading账户
- [ ] 部署每日数据拉取+推理管道
- [ ] 部署模拟订单（不真实下单，仅记录）
- [ ] 追踪每日信号vs实际收益
- [ ] 计算滚动IC, Sharpe, Max Drawdown
- [ ] 验证所有风险控制规则触发正确

### Phase 2: 最小资金实盘 (2-4周)

- [ ] 开设真实Broker账户（最小启动资金）
- [ ] 每日10%仓位运行（而非100%）
- [ ] 严格跟踪：信号vs执行偏差（滑点）
- [ ] 每日日志：意外事件、API超时、订单拒绝
- [ ] 2周无重大事故 → 仓位升至25%

### Phase 3: 全仓位运行

- [ ] 4周稳定运行，无重大回撤 → 仓位升至100%
- [ ] 设置自动邮件/SMS通知
- [ ] 每周人工review一次模型表现
- [ ] 设置模型再训练触发条件（IC连续2周<10分位→重新训练）

---

## 八、常见陷阱

1. **PDT规则（美股）**：< $25k的保证金账户不能在5个交易日内做4+笔日内交易。CTM的日频换仓策略建议使用**现金账户**（不受PDT限制，但资金T+2结算）

2. **IBKR TWS Gateway稳定性**：TWS Gateway必须24/7持续运行。建议使用Docker容器+自动重启脚本

3. **公司行动（拆股/配股/退市）**：需要在推理前过滤掉即将退市或有公司行动的股票。Alpaca的Assets API提供`status`和`tradable`字段，IBKR的Contract对象提供相应信息

4. **滑点**：Market Order在低流动性股票上可能产生显著滑点。建议对低流动性标的使用Limit Order（限价单）

5. **数据泄露**：确保训练时使用的数据时间严格在测试数据之前（walk-forward已处理），推理时不使用未来数据

6. **API限制**：Alpaca免费层200次/分钟；IBKR TWS API约50次/秒但会节流。批量推理时需处理rate limit

7. **税收**：自动交易产生的大量短线交易的税务影响因国家而异。美国：短线资本利得税率=普通所得税率；日本：譲渡益課税20.315%；欧洲：各国不同
