# 可靠数据获取方式调查报告

> **日期**: 2026-06-06 | **调查范围**: 美股、A 股、宏观、基本面、另类数据
> **网络环境**: 代理 `http://127.0.0.1:7897`（可用，但对中国大陆 API 有限制）

---

## 一、环境实测结论

| 数据源 | 状态 | 通过代理 | 说明 |
|--------|:----:|:--------:|------|
| **yfinance** (Yahoo Finance) | ✅ 可用 | ✅ | 美股 + A 股（.SZ/.SS 后缀）|
| **Wikipedia** (成分股列表) | ✅ 可用 | ✅ | S&P 500 (503), NASDAQ-100 (101) |
| **Alpaca Markets** | 🔑 需注册 | ✅ | API 返回 401（缺 Key），免费层可用 |
| **SEC EDGAR** | ✅ 可用 | ✅ | 10-K/10-Q 季报年报 |
| **FRED** (美联储经济数据) | ⚠️ 待测 | ✅ | 需 `pandas_datareader`，已有部分本地数据 |
| **akshare** (A 股基本面) | ❌ 被墙 | ❌ | eastmoney.com 被代理阻断 |
| **tushare** | ❌ 未安装 | — | 需注册 + Token，接口变动频繁 |
| **baostock** | ❌ 未安装 | — | 免费但数据质量一般 |

---

## 二、各数据源详细评估

### 2.1 yfinance — 美股日频 (★★★★★ 推荐)

**已安装**: yfinance 1.3.0

**能力**:
- 覆盖: S&P 500 (503 只) + NASDAQ-100 (101 只) + 任意美股代码
- 历史: 完整 (2010 年起，视股票而定)
- 频率: 1d / 1h / 1m / 5m / 15m / 30m
- 字段: Open, High, Low, Close, Adj Close, Volume

**实测下载速度**:
```
50 只股票 × 1 年: ~11 秒
503 只股票 × 10 年: ~18 分钟（推荐分批下载，每批 50 只）
503 只股票 × 3 年: ~5 分钟
```

**限制**:
- 速率限制: ~2000 请求/小时（免费层）
- 分钟级数据最多 30 天（60 天需 premium）
- 推荐 `threads=True` 加速

**S&P 500 成分股获取**:
```python
import pandas as pd
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
sp500 = tables[0]['Symbol'].tolist()
sp500 = [t.replace('.', '-') for t in sp500]  # 修复 BRK.B → BRK-B
# → 503 只股票
```

**批量下载示例**:
```python
import yfinance as yf
data = yf.download(
    sp500[:50],  # 分批 50 只
    start='2015-01-01', end='2026-06-01',
    progress=False, auto_adjust=True,
    threads=True
)
```

### 2.2 yfinance — A 股日频 (★★★☆☆ 可用)

**测试通过**: ✅ `000001.SZ` (平安银行), `600000.SS` (浦发银行)

**格式**: `{code}.SZ` (深交所) / `{code}.SS` (上交所)

**限制**:
- 历史数据通常只到 2018 年左右
- 数据质量低于腾讯/东方财富源
- 无前复权/后复权选项（仅股息调整）
- 部分小盘股可能缺失

**与 tencent_clean 的对比**:
| 维度 | yfinance A 股 | tencent_clean (已有) |
|------|:-----------:|:-------------------:|
| 历史长度 | ~2018+ | 2015-2026 (11年) |
| 复权 | 仅股息调整 | 前复权 |
| 数据质量 | 中等 | 高 |
| 股票覆盖 | ~4000 | 已选 200 只 |

**建议**: A 股数据继续使用已有的 `../new_data/data/tencent_clean/` (200 只 × 11 年)。yfinance 仅作补充或验证用途。

### 2.3 Alpaca Markets (★★★★☆ 需注册)

**已安装**: alpaca-py 0.43.4 (已有 `src/execution/alpaca_broker.py`)

**免费层**:
- 所有美股历史日频数据 (无限制)
- 实时数据 (IEX 延迟 15 分钟)
- 纸交易 (paper trading) 账户
- **不支持分钟级历史数据**（需付费 $9/月 起）

**注册步骤**:
1. 访问 https://alpaca.markets
2. 注册免费账户
3. 获取 Paper API Key + Secret Key
4. 设置环境变量: `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`

**Python 用法**:
```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

client = StockHistoricalDataClient(api_key, secret_key)
request = StockBarsRequest(
    symbol_or_symbols=['AAPL', 'MSFT'],
    timeframe=TimeFrame.Day,
    start='2020-01-01',
    end='2025-01-01'
)
bars = client.get_stock_bars(request)
```

**优势**: 
- 数据质量高 (官方交易所数据)
- 无速率限制 (免费层)
- 已有 `alpaca_broker.py` 代码基础

**劣势**:
- 需要注册（免费但需实名）
- 免费层无分钟级历史数据

### 2.4 SEC EDGAR — 美股基本面 (★★★★☆ 免费)

**用途**: 下载上市公司季报/年报 (10-K, 10-Q) 的财务数据

**工具选项**:
| 工具 | 特点 |
|------|------|
| `sec-edgar-api` (pip) | 直接下载 XBRL 数据 |
| `edgartools` (pip) | 更现代的 API |
| 直接 HTTP 请求 | `sec.gov/cgi-bin/browse-edgar` |

**可提取的财务指标**:
- Revenue, Net Income, EPS
- Total Assets, Total Liabilities, Equity
- Operating Cash Flow, Free Cash Flow
- Gross Margin, Operating Margin
- ROE, ROA, Debt-to-Equity

**python-edgar 示例**:
```bash
pip install python-edgar  # 或 edgartools
```
```python
from edgar import Company
company = Company('AAPL')
filings = company.get_filings(form='10-K').latest(5)
```

**限制**: SEC EDGAR 仅覆盖美股，不覆盖 A 股。

### 2.5 FRED — 宏观经济数据 (★★★☆☆)

**已安装**: `pandas_datareader` (刚安装)

**可获取的关键宏观序列**:

| 系列代码 | 名称 | 频率 | 用途 |
|----------|------|:----:|------|
| SP500 | S&P 500 指数 | 日 | 市场基准 |
| VIXCLS | CBOE 波动率指数 | 日 | 恐慌指数 |
| DCOILWTICO | WTI 原油价格 | 日 | 商品因子 |
| DFF | 联邦基金利率 | 日 | 利率环境 |
| GDP | 国内生产总值 | 季 | 经济周期 |
| UNRATE | 失业率 | 月 | 劳动力市场 |
| CPIAUCSL | CPI (消费者价格指数) | 月 | 通胀 |
| T10Y2Y | 10Y-2Y 利差 | 日 | 衰退预警 |
| BAMLH0A0HYM2 | 高收益债 OAS 利差 | 日 | 信用风险 |

**Python 用法**:
```python
import pandas_datareader.data as web
import datetime

start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2026, 6, 1)
sp500 = web.DataReader('SP500', 'fred', start, end)
vix = web.DataReader('VIXCLS', 'fred', start, end)
```

**已有数据**: `../new_data/data/fred/` 已有 DCOILWTICO, VIXCLS (2025-2026，仅 1 年), SP500 (2016-2026，但需解析格式)

### 2.6 akshare — A 股基本面 (❌ 当前不可用)

**问题**: 代理 (`127.0.0.1:7897`) 阻断了对 `eastmoney.com` 的请求。所有依赖东方财富 API 的 akshare 函数均失败。

**替代方案 (A 股基本面)**:
1. **直接使用已有数据**: `../new_data/data/tencent_clean/` + 手动下载财务报表
2. **换代理节点**: 使用支持中国大陆的代理
3. **新浪财经 API**: `stock_financial_report_sina` 等（走不同 API 端点）
4. **Tushare Pro**: 注册获取 Token (需积分)
5. **Wind / Choice 终端**: 付费方案

### 2.7 新闻/情绪数据 (中长期)

| 来源 | 覆盖 | 免费层 | API |
|------|:----:|:------:|-----|
| **NewsAPI** | 全球新闻 | 100 请求/天 | newsapi.org |
| **Finnhub** | 美股新闻 + 情绪 | 60 请求/分钟 | finnhub.io |
| **Alpha Vantage** | 新闻情绪 | 25 请求/天 | alphavantage.co |
| **Reddit API** | r/wallstreetbets | 免费 | praw |
| **Twitter/X API** | Cashtags | 免费层受限 | developer.x.com |

---

## 三、推荐执行计划

### 立即执行 (今天)

| # | 任务 | 耗时 | 产出 |
|---|------|:----:|------|
| **1** | yfinance 下载 S&P 500 × 10 年日频 | ~18 min | `../new_data/data/sp500/` (503 CSVs) |
| **2** | yfinance 下载 NASDAQ-100 × 10 年日频 | ~4 min | `../new_data/data/nasdaq100/` (101 CSVs) |
| **3** | 合并 S&P500 + NASDAQ-100 → ~600 美股 | — | 不重复股票去重 |

### 短期 (本周)

| # | 任务 | 耗时 | 依赖 |
|---|------|:----:|------|
| **4** | 注册 Alpaca 免费账户 | 30 min | 邮箱注册 |
| **5** | 下载 FRED 宏观序列 (8-10 个关键指标) | ~5 min | pandas_datareader |
| **6** | 美股波动率交易回测 (600 只股票) | ~30 min | 任务 1-2 完成 |

### 中期 (2-4 周)

| # | 任务 | 优先级 |
|---|------|:------:|
| **7** | SEC EDGAR 下载 S&P500 基本面数据 | ★★★★☆ |
| **8** | 安装 python-edgar / edgartools | 任务 7 依赖 |
| **9** | Finnhub/NewsAPI 新闻情绪数据 | ★★★☆☆ |
| **10** | A 股基本面数据 (换代理或人工下载) | ★★☆☆☆ |

---

## 四、硬性前置条件检查

对照 `progress.md` §9.4 的硬性前置条件：

| 条件 | 旧 HK 项目 | 完成 Phase A 后 | 完成 Phase B 后 |
|------|:--------:|:-------------:|:-------------:|
| ≥1 非价格数据源 | ❌ | ❌ | ✅ (FRED 宏观 + SEC 基本面) |
| 线性基线 test IC > 0.02 | ❌ (0.006) | 待测试 (美股) | 待测试 |
| ≥20 独立特征 (含外部) | ❌ (9) | ❌ | ✅ (OHLCV + 宏观 + 基本面) |
| ≥500 高流动性标的 | ❌ (200) | ✅ (503 S&P500) | ✅ |

**结论**: 
- Phase A（S&P500 下载）满足 ≥500 标的
- Phase B（FRED + SEC EDGAR）满足独立信息源 + ≥20 特征
- **是否可以突破 test IC > 0.02 取决于美股是否比 A 股有更高的可预测性**。当前 19 只美股 IC=0.025 是微弱正信号，600 只美股的结果将是决定性证据。

---

## 五、yfinance 批量下载脚本

```bash
# 创建下载目录
mkdir -p ../new_data/data/sp500 ../new_data/data/nasdaq100

# 运行下载脚本（见 scripts/download_sp500.py）
python scripts/download_sp500.py --output ../new_data/data/sp500 --years 10
```

---

## 六、不推荐的数据源

| 数据源 | 原因 |
|--------|------|
| **付费数据** (Bloomberg, Reuters, Wind) | 个人无法负担 ($20K+/年) |
| **Tushare Pro** | 需要积分 (需贡献或付费)，接口不稳定 |
| **Quandl / Nasdaq Data Link** | 大部分数据集已转为付费 |
| **Polygon.io** | 免费层仅 5 个 API 调用/分钟 |
| **Tiingo** | 免费层限制 500 个唯一 ticker |
| **IBKR API** | 需要 IBKR 账户 + 市场数据订阅 ($4.5-15/月) |
