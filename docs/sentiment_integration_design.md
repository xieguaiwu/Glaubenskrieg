# Sentiment Analysis Integration Design for Glaubenskrieg CTM

> 撰写日期：2026-05-24 | 范围：舆情分析模型到CTM架构的集成
> 当前状态：设计阶段 | 依赖：CTM P0-P3融合管线已完成

---

## 一、目标

在CTM的纯价格/成交量信号之上，新增**舆情/政策分析信号流**，形成一个能够捕捉基本面之外的"市场情绪"和"政策冲击"的混合预测系统。

**核心假设**（来自先前的A.5分析）：

- 在美股中，舆情信号对纯价格信号提供**边际增量**（DeePM纯价格Sharpe ~3.0已足够高）
- 在A股中，舆情/政策信号对纯价格信号提供**必要增量**（政策断裂不可从价格历史中学习，Brunnermeier et al. 2022; Dang, Li & Wang 2024 JFQA）
- 在日股和欧股中，舆情信号提供**有意义的分散化收益**

---

## 二、架构概览

```
                    ┌──────────────────────────┐
                    │   NLP Sentiment Pipeline  │
                    │                          │
                    │  ├─ 数据采集（微博/雪球/  │
                    │  │   Twitter/Reddit/新闻） │
                    │  ├─ 文本预处理            │
                    │  ├─ 情绪分类（FinBERT）    │
                    │  └─ 时序聚合（日频情绪分数）│
                    └──────────┬───────────────┘
                               │
                               ▼
           ┌───────────────────────────────────┐
           │       Glaubenskrieg CTM           │
           │                                   │
           │  OHLCV ─→ CTM (Mamba SSM) ─→ pred │
           │     │                             │
           │     └─→ GBDT Features ─→ pred     │
           │                     ↑             │
           │                     │             │
           │           ┌─────────┘             │
           │           │ Sentiment Features     │
           │           │ (Path A: 最快验证)     │
           │           │                       │
           │  ┌────────┴──────────┐            │
           │  │ IC-Weighted Fusion │◄── NLP pred│
           │  │ w1·CTM + w2·GBDT  │    (Path C)│
           │  │    + w3·NLP       │            │
           │  └───────────────────┘            │
           └───────────────────────────────────┘
```

---

## 三、三条集成路径

### Path A: GBDT特征注入 ⭐ 推荐起步

**原理**：NLP情绪分数作为GBDT的附加特征列，通过现有的P1特征融合管线喂入树模型。

```
NLP Pipeline → daily_sentiment_score (per stock)
    → 写入OHLCV CSV 或 独立 sentiment.csv
        → GBDT 特征聚合 (last/mean/std/min/max/slope, 6种统计量)
            → 与CTM hidden states 拼接
                → 喂入 GBDT 训练
```

**代码变更**：

| 文件 | 变更 | 工作量 |
|------|------|--------|
| `src/data/features.py` | 新增 `compute_sentiment_features(df, sentiment_df)` | 0.5天 |
| `src/data/dataset.py` | `StockDataset` 支持加载独立 sentiment CSV | 0.5天 |
| `src/data/gbdt_features.py` | 无变更（6种聚合统计自动处理新列） | 0天 |
| `src/train/ensemble_trainer.py` | 无变更（GBDT自动使用新特征） | 0天 |
| **总计** | | **1天** |

**优点**：
- 零架构变更，利用现有P1特征融合管线
- 最快验证情绪信号的增量信息含量
- 可以独立于CTM测试情绪信号质量

**缺点**：
- CTM本身不获益于情绪信号
- 情绪信号仅用于树的截面预测，未用于时序建模

### Path B: CTM输入特征扩展

**原理**：NLP情绪向量直接作为CTM的输入特征，通过 `input_proj` 注入Mamba骨干网络。

```
NLP Pipeline → daily_sentiment_vector (per stock, dim=K)
    → expand input_dim from 9 to 9+K
        → CTM.input_proj: Linear(9+K, model_dim)
            → 整个Mamba骨干网络处理情绪+价格信号
```

**代码变更**：

| 文件 | 变更 | 工作量 |
|------|------|--------|
| `src/data/features.py` | `compute_all_features()` 拼接情绪向量 | 0.5天 |
| `configs/default.yaml` | 新增 `sentiment_dim: K` | 0天 |
| `src/model/ctm_model.py` | `input_proj` 维度动态适配 | 0.5天 |
| `scripts/train.py` | 新增 `--sentiment-csv` CLI参数 | 0.5天 |
| **总计** | | **1.5-2天** |

**优点**：
- CTM和GBDT都从情绪信号中受益
- Mamba的选择性扫描机制可以学习情绪信号的时序动态

**缺点**：
- 需要重新训练CTM（破坏已有checkpoint）
- 需要严格的因果性验证（情绪收集时间不能包含未来信息）
- 情绪信号相比OHLCV特征噪声更高，可能降低CTM信噪比

**因果性约束（关键）**：

```
A股 T+1 制度:
    T-1日 15:00 收盘后的社交媒体帖子
        → T日 09:30 开盘前可用的情绪信号
            → 预测 T日收盘收益率
                ✅ 无前瞻偏差

美股 T+0 制度:
    T-1日 16:00 收盘后到 T日 09:30 开盘前的社交媒体帖子
        → T日开盘可用的情绪信号
            → 预测 T日收盘收益率
                ✅ 无前瞻偏差
```

### Path C: 第三条预测流 + IC加权融合

**原理**：NLP模型独立预测收益率，作为第三条融合流参与IC加权融合。

```
w_CTM · ctm_pred + w_GBDT · gbdt_pred + w_NLP · nlp_pred
    where w_i = IC_i / (IC_CTM + IC_GBDT + IC_NLP + 1e-8)
```

**代码变更**：

| 文件 | 变更 | 工作量 |
|------|------|--------|
| `src/model/ensemble.py` | `EnsembleFusion` 从2路扩展到K路 | 1天 |
| `src/model/nlp_predictor.py` | 新建：独立的NLP预测模型 | 2-3天 |
| `src/train/ensemble_trainer.py` | 支持三路训练+融合 | 1天 |
| `scripts/infer.py` | 加载三个模型+三路融合 | 0.5天 |
| **总计** | | **4-5天** |

**优点**：
- 完全模块化：每条信号流独立训练、独立优化
- IC权重自动处理信号质量退化（一条赛道失效不影响其他）
- 可以A/B测试每条信号流的独立贡献

**缺点**：
- 维护三条管线的复杂性
- 需要NLP模型达到一定的独立IC才能产生增量价值

---

## 四、NLP模型选项

### 按市场选择

| 市场 | 模型 | 数据源 | 成熟度 |
|------|------|--------|--------|
| **A股** | Erlangshen-MegatronBert (1.3B) / FinBERT-Chinese | 微博财经博主、雪球帖子、东方财富股吧 | 高 |
| **美股** | FinBERT (ProsusAI) / DistilBERT | Twitter/X, Reddit r/wallstreetbets, StockTwits | 高 |
| **日股** | 日本語FinBERT / 株予報BERT | 株予報、Yahoo Finance JP 留言板 | 中 |
| **欧股** | FinBERT (multilingual) / XLM-RoBERTa | Twitter, SeekingAlpha, 本地语言新闻 | 中 |

### 推荐MVP方案：词法级情绪分析（快速验证）

| 维度 | 详情 |
|------|------|
| **方法** | 金融情感词典 + 否定词/程度副词处理 |
| **中文词典** | 知网HowNet情感词典 + 清华大学金融情感词典 |
| **英文词典** | Loughran-McDonald金融情感词典（JFE 2011） |
| **日文词典** | 日本語評価極性辞書 |
| **输出** | 每日每股票的正/负/中性情绪分数 [0,1]³ |
| **优点** | 零GPU，零训练，1天内可搭建 |
| **缺点** | 准确率约70-75%，低于BERT-based模型85-90% |
| **用于** | 快速验证情绪信号是否提供增量IC |

### 升级路径：BERT-based模型

```python
# MVP: 词法级
sentiment_score = lexicon_score(text)  # 1天搭建

# v2: FinBERT微调
model = AutoModelForSequenceClassification.from_pretrained(
    "ProsusAI/finbert"  # 英文
    # "IDEA-CCNL/Erlangshen-MegatronBert-1.3B"  # 中文
)
# 在金融标注数据上微调 → 2-3天

# v3: 端到端（Path C专用）
# 直接预测收益率，而非仅情感分类
model = FinBERTForReturnPrediction()
# → 1-2周
```

---

## 五、数据采集管道

### A股数据源

```
┌─ 微博财经博主 ──→ weibo API / 爬虫 ──→ 每日帖子
├─ 雪球帖子 ──→ xueqiu API ──→ 每日讨论
├─ 东方财富股吧 ──→ 爬虫 ──→ 个股讨论
└─ 证监会公告 ──→ 爬虫 ──→ 政策变化文本
```

**因果性约束**：
- 采集时间：每日 15:00-次日08:00（A股收盘后到开盘前）
- 时间戳过滤：仅保留发布时间明确的帖子
- 去重：同一用户重复帖子去重

### 美股数据源

```
┌─ Twitter/X API ──→ cashtag搜索 ($AAPL, $TSLA)
├─ Reddit API ──→ r/wallstreetbets, r/stocks
├─ StockTwits API ──→ 个股消息流
└─ NewsAPI ──→ 财经头条
```

**因果性约束**：
- 采集时间：每日 16:00-次日09:00（美股收盘后到开盘前）
- Reddit/StockTwits帖子时间戳GMT→EST转换

### 日股数据源

```
┌─ 株予報 ──→ 个股预测帖子
├─ Yahoo Finance JP ──→ 留言板
└─ Twitter JP ──→ 日文cashtag
```

### 欧股数据源

```
┌─ SeekingAlpha ──→ 个股分析
├─ Twitter/X ──→ $VOW.DE, $SAP等
└─ 本地新闻API ──→ 财经头条（多语言）
```

---

## 六、验证方案

### Phase 1: 情绪信号独立验证（Path A，1-2天）

```
1. 搭建词法级情绪管道 → 生成每日情绪分数
2. 计算情绪分数的滚动 IC (Spearman ρ)：
   IC(t) = corr(sentiment_score(t-1), future_return(t), cross-section)
3. 对比纯价格因子的IC：
   - 如果 IC_sentiment 独立且 >0.02 → 继续
   - 如果 IC_sentiment 与 IC_CTM 相关性 <0.3 → Path C值得
   - 如果 IC_sentiment <0.01 或与 IC_CTM 相关性 >0.7 → 情绪不提供增量
```

### Phase 2: GBDT集成验证（Path A，1天）

```
1. 将情绪分数加入GBDT特征矩阵
2. 对比 CTM+GBDT vs CTM+GBDT+NLP 的融合IC
3. 量化情绪信号的增量IC贡献
```

### Phase 3: 独立预测流验证（Path C，如果Phase 1/2通过）

```
1. 训练独立NLP预测模型
2. 三路IC加权融合
3. 对比两路 vs 三路的Sharpe差异
```

---

## 七、预期收益与风险

### 各市场的预期情绪增量

| 市场 | 预期 IC_increment | 理由 |
|------|------------------|------|
| **A股** | **+0.03 ~ +0.08** | 政策断裂不可从价格学习中获取；散户情绪（微博/雪球）对短期价格有预测力（Xu et al. 2017 PLOS ONE; Atlantis Press 2025） |
| **美股** | **+0.01 ~ +0.03** | 机构主导，价格已高信息含量；情绪提供边际增量（FinBERT-class models on StockTwits/Twitter） |
| **日股** | **+0.02 ~ +0.05** | 外资主导+散户反周期操作，情绪提供补充视角；日本散户的株予報帖子有独立信息量 |
| **欧股** | **+0.01 ~ +0.03** | 多语言挑战，但本地语言新闻可能提供国际投资者忽略的信号 |

### 核心风险

1. **前瞻偏差**：情绪收集时间戳必须严格在收盘后→开盘前；测试时必须模拟真实可用时间
2. **NLP质量**：中文金融NLP（特别是微博黑话、雪球术语）的准确率远低于英文 → 可能产生噪声而非信号
3. **情绪信号的alpha衰减**：随着更多人使用相同数据源，情绪信号的超额收益会快速衰减
4. **分布偏移**：COVID-19期间的情绪分布与正常期间显著不同（AlphaMLDigger, arXiv 2022）→ 需要定期重训练/微调NLP模型

---

## 八、实施优先级

| 优先级 | 任务 | 工作量 | 阻塞 |
|--------|------|--------|------|
| **P0** | 词法级情绪管道 + Path A集成 | 2天 | 无 |
| **P0** | Phase 1验证：情绪信号独立IC | 0.5天 | P0完成 |
| **P1** | FinBERT微调（如果词法级IC>0.02） | 2-3天 | P0验证通过 |
| **P1** | 因果性验证：时间戳审计 | 0.5天 | P0完成 |
| **P2** | Path C三路融合（如果IC独立且相关<0.3） | 4-5天 | P1验证通过 |
| **P3** | 多市场情绪管道（日股/欧股） | 1-2周 | Path A在A股/美股验证成功 |
