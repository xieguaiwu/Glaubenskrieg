# 架构图生成提示词（给 ChatGPT/DALL·E）

## 目标
生成一张学术论文级别的 CTM 架构彩色矢量图，风格为干净的 2D 等距框线图，配色采用 IBM Carbon 色系（蓝/橙/绿/红/紫）。最终输出应为可缩放矢量格式，以便在 LaTeX 中直接使用。

## 整体布局

**布局类型**：双栏流程图（左栏 = CTM 核心，右栏 = GBDT 集成路径），底部为融合输出。

**画布比例**：宽高比约 4:3（或 16:9），横向构图。

**背景**：纯白色或极浅灰色 #FAFAFA，无网格、无纹理、无渐变背景。

## 详细组件描述

### 左栏：CTM（Mamba SSM）核心路径（从上到下）

每个组件用带圆角的矩形框表示（rounded corners, 3-4px radius），框内分两行——第一行是组件名称（粗体），第二行是关键参数（较小字体）。

**1. 输入层**
- 位置：左上角第一行
- 标签：`OHLCV Features`（第一行）, `T × 9`（第二行，灰色）
- 颜色：浅蓝色背景 #E8F0FE，深蓝色边框 #0062FF
- 出箭头：从底部中心向下

**2. 输入投影**
- 标签：`Input Projection` / `Linear: 9 → 64`
- 颜色：浅蓝色背景 #D2E3FC，深蓝色边框 #0062FF
- 可选标注：`d_model = 64` 在框右侧

**3. 因果卷积层**
- 标签：`CausalConv1D` / `kernel=3, depthwise`
- 颜色：浅蓝色背景 #B8D4FB，深蓝色边框 #0062FF
- 需有明显标识表示这是因果卷积（例如向左偏移的小时钟图标⏱，或标注 "causal"）

**4. 季节-趋势分解（可选）**
- 标签：`SeasonalTrend Decomp` / `period=5`
- 颜色：淡紫色背景 #EDE7F6，紫色边框 #8A3FFC
- 虚线边框表示可选（optional）
- 框内分为两个小方框：S（seasonal）和 T（trend），之间用双向箭头连接

**5. Mamba Block × 2**
- 位置：左列中央，最核心的组件
- 标签：`MambaBlock × 2` / `state_dim=8, d_model=64`
- 颜色：深蓝色背景 #7BB3F0，深蓝色边框 #0043CE
- 框内需绘两个堆叠的小方框（代表两层 Mamba），之间有垂直箭头
- 每个小方框可包含简短文字：S6 Selective Scan
- 左侧标注 `residual connection` 的跳跃连接线（绕过两个方框，从第一个输入直达第二个输出）

**6. 双向 Mamba（可选）**
- 标签：`Bi-Mamba` / `forward + backward`
- 颜色：淡紫色背景 #F3E5F5，紫色边框 #8A3FFC
- 虚线边框表示可选
- 内部用两个水平箭头（→ 左向右，← 右向左）表示双向处理

**7. 多头任务输出头**
- 标签：`Multi-Task Heads` / `Regression + 3-class`
- 颜色：浅绿色背景 #E8F5E9，绿色边框 #24A148
- 框内分为两个子框：
  - 左子框：`Regression` / `r̂_t`（1 个输出）
  - 右子框：`3-class` / `UP/NEUTRAL/DOWN`
  - 之间用平行分隔线

### 右栏：GBDT 集成路径（从上到下）

**8. GBDT（Hoffnung C++）**
- 位置：右上角，与 CausalConv1D 平齐
- 标签：`GBDT` / `Hoffnung C++ 300 trees`
- 颜色：浅橙色背景 #FFF3E0，橙色边框 #FF832B
- 框内绘一排小树图标（3-4 棵简化决策树），或标注 `histogram-building`

**9. 跨资产注意力（可选）**
- 位置：右栏中部，与 Mamba Block 平齐
- 标签：`CrossAsset Attention` / `low-rank U·Vᵀ`
- 颜色：浅红色背景 #FFEBEE，红色边框 #FA4D56
- 虚线边框表示可选
- 框内用三个并排的小圆（代表股票节点）之间用虚线箭头相连

### 底部：融合层

**10. IC 加权融合**
- 位置：底部居中，横跨两栏
- 标签：`IC-Weighted Fusion` / `w = IC_ctm / (IC_ctm + IC_gbdt)`
- 颜色：深青色背景 #E0F2F1，青色边框 #009D9A
- 框内绘一个天平符号⚖，左侧为 CTM 预测值，右侧为 GBDT 预测值

**11. 融合输出**
- 位置：IC 加权融合正下方
- 标签：`Fused Signal` / `w·r̂_ctm + (1-w)·r̂_gbdt`
- 颜色：深青色背景 #B2DFDB，青色边框 #00796B
- 以一个粗箭头指向「Fused Signal」文字

## 连接和数据流（箭头）

所有箭头使用实线带箭头头的线条（solid line with arrowhead），线宽 2-3px，颜色深灰色 #616161。

**左栏垂直流**：从上到下的箭头链，按顺序连接：
`Input → Linear → CausalConv → Decomp → Mamba × 2 → Bi-Mamba → Multi-Task Heads`

**右栏水平流**：
- 从 CausalConv1D 的右侧伸出一个箭头，指向右侧的 `6 aggregated stats` 标注，再向下指向 GBDT 框
- 从 MambaBlock × 2 的右侧伸出箭头到 CrossAssetAttention

**集成流**（底部）：
- 从 Multi-Task Heads 右侧伸出一个箭头向右下方，指向 IC-Weighted Fusion
- 从 GBDT 底部伸出一个箭头向下，再向左指向 IC-Weighted Fusion
- 从 CrossAssetAttention 底部伸出箭头向下指向 IC-Weighted Fusion

**损失桥接**（虚线）：
- 一条虚线箭头从 Multi-Task Heads 右侧向上绕过到 GBDT 上方，标注 `Loss Bridge (P2)`

## 标注和排版

- **参数标注**：在 CTM 核心框左侧，用小号灰色字体标注 `97K parameters`；在 GBDT 框右侧标注 `~5K parameters`
- **维度标注**：在关键连接处标注张量维度，如 `B, T, 64`、`B, N, T, D`
- **分组框**：CTM 核心周围用一个浅蓝色虚线大框包裹，左上角标注 `CTM (Mamba SSM)`；GBDT 用浅橙色虚线框包裹，标注 `GBDT`；融合部分用青色虚线框包裹，标注 `Ensemble`
- **字体**：中文论文用思源黑体（Noto Sans SC），英文论文用 Helvetica 或 Arial。正文 10pt，参数标注 7pt

## 技术细节要求

- 所有矩阵变量使用数学符号（$B$ 批大小，$T$ 时间步长，$N$ 资产数，$D$ 隐藏维度）
- Mamba 的选择性扫描机制缩写为 S6，不要写错
- 损失桥接（P2）、特征融合（P1）、IC 加权融合（P0）三种集成模式都要被标注或提及

## 应避免的元素

- ❌ 不要 3D 效果、透视、阴影、发光、渐变色
- ❌ 不要照片、纹理、复杂背景
- ❌ 不要超过 6 种颜色
- ❌ 不要手写体或衬线字体作为标题
- ❌ 不要与论文结论矛盾的声称（如"优于 LSTM"、"超越 SOTA"）
