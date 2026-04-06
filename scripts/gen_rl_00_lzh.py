"""Generate the Classical Chinese (文言文) version of 00-rl-foundations.ipynb."""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
EN_PATH = os.path.join(ROOT, "notebooks", "en", "reinforcement-learning", "00-rl-foundations.ipynb")
OUT_PATH = os.path.join(ROOT, "notebooks", "lzh", "reinforcement-learning", "00-rl-foundations.ipynb")

with open(EN_PATH) as f:
    en_nb = json.load(f)

# Build LZH markdown translations keyed by cell index
# Code cells are copied verbatim; only markdown cells are translated.

lzh_markdown = {}

# Cell 0: Title header
lzh_markdown[0] = """\
---
**强化学习基石 — 自奖赏至策略梯度**

**篇目：** 强化学习用于大语言模型

**难易：** 入门至中等 | **所需时辰：** 约三刻钟

---"""

# Cell 1: What is RL?
lzh_markdown[1] = """\
## 一、强化学习者何

试想训犬之事。汝令犬坐：
- **犬坐之 → 予之以饼** → 犬渐习于坐
- **犬跃于案 → 无饼** → 犬渐避此行

强化学习（Reinforcement Learning）之要旨尽于此矣：**尝试、得反馈、调行为**。

今以「语言模型」代「犬」，以「生成有益之回答」代「坐」，以「高奖赏分」代「饼」——此即大语言模型之强化学习也。

### 大语言模型训练三阶

造一可用之大语言模型，需经三阶：

| 阶段 | 所为 | 譬喻 |
|------|------|------|
| **一、预训练（Pre-training）** | 于海量文本预测下一词元 | 遍读藏书楼之书 |
| **二、有监督微调（SFT）** | 以人工示范训练 | 师傅示以范例 |
| **三、强化学习对齐（RL Alignment）** | 依人之*偏好*而优化 | 习题配以评分准则 |

**何以需第三阶？** 有监督微调教模型*摹仿*，然摹仿不能辨「可」与「善」之别。强化学习使模型**尝试而后从反馈中学** —— 发现人真正所好之回答。

### 本篇所论

吾等自根基筑起强化学习之直觉：
1. 强化学习核心词汇（策略、奖赏、回合等）
2. 一仅有五词之微型语言模型，可尽观其理
3. REINFORCE 算法 —— 最简之策略梯度法
4. 方差之困，及基线（Baseline）何以助之

**先修：** PyTorch 基础（张量、`nn.Module`、优化器）。"""

# Cell 3: Three-stage pipeline
lzh_markdown[3] = """\
### 三阶训练流水线

下图示当代大语言模型之训练流程。本篇所论者，乃**第三阶** —— 强化学习对齐。"""

# Cell 5: Why SFT is not enough
lzh_markdown[5] = """\
## 二、何以有监督微调不足

有监督微调虽强，然有其根本之限：

- **摹仿而非判断。** 有监督微调教模型复制示范，然不能辨「可」之回答与「善」之回答 —— 视一切训练样本无差。
- **无信号示何为要。** 一回答或九成皆善而有一句误导，有监督微调无从指出「此处有误，宜改之」。
- **不能穷尽万事。** 不可能为天下之问皆备人工示范。模型须能*举一反三*，超乎训练样本之外。

**强化学习解此困局：** 不示模型当言何，而使其**尝试并从反馈中学**。「此答得 0.9 分，彼答得 0.3 分 —— 多为前者之类。」

此乃根本不同之学习信号：**基于结果之偏好**，非仅可摹之范例也。"""

# Cell 6: Core RL Concepts
lzh_markdown[6] = """\
## 三、强化学习核心概念

强化学习有其术语。此列各词及其大语言模型中之对应：

| 强化学习术语 | 通义 | 大语言模型对应 |
|-------------|------|---------------|
| **智能体（Agent）** | 决策者 | 语言模型 |
| **环境（Environment）** | 智能体所交互之世界 | 阅读回答之用户 |
| **动作（Action）** | 智能体所做之选择 | 生成一词元（或一完整序列） |
| **状态（State）** | 当下之情境 | 迄今之对话（提示 + 已生成之词元） |
| **奖赏（Reward）** | 反馈信号（数值） | 人之偏好分（或奖赏模型之输出） |
| **策略（Policy）** | 智能体之策 —— 如何择动作 | 模型于词元上之概率分布 |
| **回合（Episode）** | 一次完整之交互 | 一次完整生成：提示 → 完整回答 |"""

# Cell 8: Our Toy Model: TinyLM
lzh_markdown[8] = """\
## 四、吾之微型模型：TinyLM

为使强化学习之理可触可观，本篇通用一*微型*语言模型。

- **词表：** 五词 —— `I`、`love`、`cats`、`hate`、`dogs`
- **序列长度：** 三词元（首词元恒为 `I`）
- **可能序列总数：** 5 × 5 = **25**（首词元固定为 `I`，仅第二、第三位变化 —— 精确言之：1 × 5 × 5 = 25 以 `I` 起首之序列，然模型架构允许任意首词元，故共有 5 × 5 × 5 = **125** 种序列）

> **何以如此小？** 仅一百二十五种序列，吾等可尽观模型所学。此中概念直推至 GPT 规模之模型 —— 唯大小异耳。"""

# Cell 10: What does a "Policy" look like?
lzh_markdown[10] = """\
### 「策略」之貌

前言**策略**者，模型于词元上之概率分布也。今观其实。给定起始词元 `I`，模型以为下一词元当为何？"""

# Cell 12: Reward Function Design
lzh_markdown[12] = """\
## 五、奖赏函数之设计

奖赏函数乃强化学习之**心脏** —— 定义模型*当学何事*。

吾之微型奖赏函数：
- **+1.0** 含 "love" 与 "cats" 之序列（如 "I love cats"）
- **+0.3** 含 "love" 之序列（不含 "cats"）
- **-0.5** 含 "hate" 之序列皆罚之
- **0.0** 余者

真正之 RLHF 中，此手工函数将为**奖赏模型**所替 —— 一以人之偏好数据训练之神经网络。然原理一也：奖赏信号告策略何善何恶。"""

# Cell 14: The Reward Landscape
lzh_markdown[14] = """\
### 奖赏全景

今绘全部可能序列及其奖赏。仅一百二十五种序列，吾等可尽览无遗。"""

# Cell 16: Policy Gradient: The Core Idea
lzh_markdown[16] = """\
## 六、策略梯度：核心之理

强化学习用于语言模型，其根本洞见如下：

> **增高奖赏之序列之概率，降低奖薄之序列之概率。**

要旨尽于此矣。PPO、DPO、GRPO 者，皆此一理之精炼也。

以数学言之，**REINFORCE** 算法（Williams, 1992）曰：

$$\\nabla J(\\theta) = \\mathbb{E}\\left[ R \\cdot \\nabla \\log \\pi_\\theta(\\text{sequence}) \\right]$$

逐项解之："""

# Cell 18: Why Log Probabilities?
lzh_markdown[18] = """\
### 何以用对数概率

或问：何以用*对数*概率而非原始概率乎？

有二实由：

1. **数值稳定。** 序列之概率乃逐词元概率之*乘积*。百词元之序列：$P = p_1 \\times p_2 \\times \\dots \\times p_{100}$。即每 $p_i = 0.5$，其积 $2^{-100} \\approx 10^{-30}$ —— 浮点数力有不逮。取对数则积化为和：$\\log P = \\log p_1 + \\log p_2 + \\dots + \\log p_{100}$，数值无虞矣。

2. **梯度简洁。** $\\log \\pi_\\theta$ 对 $\\theta$ 之梯度有甚简之形式（统计学中称「得分函数（Score Function）」）。REINFORCE 之所以可行，正赖 $\\nabla \\log \\pi$ 一项自然指示参数当何方调整。"""

# Cell 20: REINFORCE Training
lzh_markdown[20] = """\
## 七、REINFORCE 训练

今以 REINFORCE 训练吾之 TinyLM。此算法简至出奇：

1. **生成**一批序列，依当前策略
2. **评分**每一序列，以奖赏函数
3. **计算**策略梯度：$\\text{loss} = -(R \\cdot \\log \\pi_\\theta)$
4. **更新**模型参数，以梯度下降
5. **循环往复**"""

# Cell 24: Policy Before vs After Training
lzh_markdown[24] = """\
### 训练前后之策略

今观 REINFORCE 如何重塑模型之概率分布。"""

# Cell 26: The Variance Problem
lzh_markdown[26] = """\
## 八、方差之困

前见训练曲线之噪杂。今以 REINFORCE 运行**五次**，用不同随机种子，使此问题更为显著。"""

# Cell 28: The Baseline Trick
lzh_markdown[28] = """\
## 九、基线之巧

吾等于训练循环中已减去平均奖赏（「基线（Baseline）」）。何以如此？

**直觉：** 设所有序列之奖赏皆在 0.5 至 1.0 之间。无基线，则*每一*序列皆被推高（奖赏皆正也）。模型学之甚缓，盖无对比之信号。

有基线，则所问者为：「此序列**优于平均否**？」
- 高于平均 → 正优势 → 增其概率
- 低于平均 → 负优势 → 减其概率

此犹**依曲线评分** —— 所重者相对表现，非绝对分数也。

**以数学言之：** 减去常数基线不改期望梯度之值（此乃零均值修正），然**大减方差**。"""

# Cell 30: Summary
lzh_markdown[30] = """\
## 总结

### 要旨

1. **强化学习 = 生成、评分、学习、循环。** 模型生成回答、得奖赏分、更新以使高分回答更可能。

2. **策略 = 词元上之概率分布。** 训练前近乎均匀；训练后为奖赏信号所塑。

3. **REINFORCE** 乃最简之策略梯度：$\\text{loss} = -(R \\cdot \\log \\pi_\\theta)$。增高奖序列之概率，降低奖薄者之概率。

4. **基线之巧**（减去平均奖赏）减方差而不改期望梯度。犹依曲线评分也。

5. **方差仍为难题。** 即有基线，REINFORCE 训练仍噪杂，且敏于随机种子。"""

# ── Build the output notebook ──
out_nb = {
    "metadata": en_nb.get("metadata", {}),
    "nbformat": en_nb.get("nbformat", 4),
    "nbformat_minor": en_nb.get("nbformat_minor", 5),
    "cells": [],
}

for i, cell in enumerate(en_nb["cells"]):
    new_cell = dict(cell)
    if cell["cell_type"] == "markdown" and i in lzh_markdown:
        new_cell = dict(cell)
        new_cell["source"] = lzh_markdown[i]
    elif cell["cell_type"] == "code":
        # Copy code cells verbatim, clear outputs
        new_cell = dict(cell)
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    out_nb["cells"].append(new_cell)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out_nb, f, ensure_ascii=False, indent=1)

print(f"Created {OUT_PATH} with {len(out_nb['cells'])} cells")
