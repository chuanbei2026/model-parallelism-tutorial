"""Generate the Classical Chinese (文言文) version of 01-ppo.ipynb."""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
EN_PATH = os.path.join(ROOT, "notebooks", "en", "reinforcement-learning", "01-ppo.ipynb")
OUT_PATH = os.path.join(ROOT, "notebooks", "lzh", "reinforcement-learning", "01-ppo.ipynb")

with open(EN_PATH) as f:
    en_nb = json.load(f)

lzh_markdown = {}

# Cell 0: Title header
lzh_markdown[0] = """\
---
**PPO — 近端策略优化**

**篇目：** 强化学习用于大语言模型

**难易：** 中等 | **所需时辰：** 约四刻钟

---"""

# Cell 1: Overview
lzh_markdown[1] = """\
## 概论

### 自 REINFORCE 至 PPO

前篇构建 **REINFORCE** —— 最简之策略梯度算法。虽可行，然有二弊：

1. **方差甚大** —— 训练曲线噪杂而不可预
2. **无安全之网** —— 一次劣梯度更新足以毁前功

**PPO（近端策略优化，Proximal Policy Optimization）** 兼解此二弊，增：
- **价值函数（Value Function）** 以减方差
- **参考模型（Reference Model）** 以防奖赏作弊
- **裁剪机制（Clipping）** 以约束每步策略变化之幅

PPO 乃 ChatGPT 与 GPT-4 之 RLHF 训练所用之算法。本篇逐步构建之，逐件添之。

### 本篇所构

| 章节 | 所为 | 所以 |
|------|------|------|
| REINFORCE 回顾 | 速复前篇之基线 | 观吾等所修之弊 |
| 四模型逐构 | 逐一添加模型 | 明 PPO 何以需四模型 |
| PPO 裁剪 | 核心机制 | 有界而稳之更新 |
| 完整 PPO 训练 | 端到端于 TinyLM | 观其实效 |
| 超参扫描 | 变化 ε | 培养调参之直觉 |"""

# Cell 3: TinyLM Setup
lzh_markdown[3] = """\
## TinyLM 设置

沿用前篇（笔记本 00）之微型语言模型。词表五词，序列长三 —— 小至可尽观模型所学，然概念直推至 GPT 规模之模型。"""

# Cell 5: REINFORCE's Problems
lzh_markdown[5] = """\
## REINFORCE 之弊

先速训 REINFORCE 以忆其何弊。观训练曲线 —— 噪杂而不可预也。"""

# Cell 7: Building Up: From 1 Model to 4
lzh_markdown[7] = """\
## 渐构：自一模型至四

真正之 RLHF 需**四模型同驻显存**。闻之骇然，故吾等逐一添之 —— 各解一具体之难。"""

# Cell 9: Stage 1: Policy Only
lzh_markdown[9] = """\
### 第一阶：策略独用

此即 REINFORCE：一模型生成序列而依奖赏自行更新。

**弊端：** 方差甚大，训练不稳。模型无参照之框 —— 不知 0.3 之奖赏于此情境下为「善」为「恶」。"""

# Cell 10: Stage 2: Adding a Value Model
lzh_markdown[10] = """\
### 第二阶：增价值模型

**价值函数（Value Function）** $V(s)$ 估「此状态平均能得几何」。充基线之用，答曰：「以目前之势，当得何赏？」

**优势（Advantage）** 乃所得与所期之差：

$$A = R - V(s)$$

- 若 $A > 0$：结果*优于*预期 —— 当强化此行为
- 若 $A < 0$：结果*劣于*预期 —— 当抑制此行为

此减方差，盖以*相对*表现代替绝对奖赏也。"""

# Cell 13: Stage 3: Adding a Reference Model
lzh_markdown[13] = """\
### 第三阶：增参考模型

无约束之下，模型可「作弊于奖赏」 —— 寻得退化之输出，分虽高而重复无义。

**参考模型（Reference Model）** 乃初始策略之冻结副本。以 **KL 散度（KL Divergence）** 度量当前策略偏离几何：

$$\\text{KL}(\\pi_\\theta \\| \\pi_{\\text{ref}}) = \\mathbb{E}\\left[ \\log \\frac{\\pi_\\theta(a|s)}{\\pi_{\\text{ref}}(a|s)} \\right]$$

以 $\\beta \\cdot \\text{KL}$ 为惩罚项加于损失，使模型「近」于原始行为 —— 可改进而不崩塌。"""

# Cell 14: Reward Hacking in Action
lzh_markdown[14] = """\
### 奖赏作弊之实

今观无 KL 惩罚而激进训练之果。模型将寻一「秘技」 —— 一序列使奖赏最大 —— 而坍缩为反复生成之。"""

# Cell 16: KL Divergence: The Drift Detector
lzh_markdown[16] = """\
### KL 散度：偏移之探

**KL 散度**度量「新策略较参考偏离几何」。

| KL 值 | 含义 |
|--------|------|
| KL = 0 | 两策略一也 —— 未学也 |
| KL 小（0.01 - 0.1） | 适度改进 —— 安全之域 |
| KL 大（> 1.0） | 模型已剧变 —— 危险之域 |

以 $\\beta \\cdot \\text{KL}$ 为惩罚项加于损失，造一拉锯之势：
- 奖赏信号拉模型趋于高奖输出
- KL 惩罚拉模型返归参考

二力均衡处，得一模型：*优于*参考而仍*可辨其本*。"""

# Cell 17: Stage 4: Adding a Reward Model
lzh_markdown[17] = """\
### 第四阶：增奖赏模型

真正之 RLHF 中，奖赏非出自手工函数如吾之 `reward_fn`，而出自**习得之奖赏模型**，以人之偏好数据训练而成：

> 「给定同一提示之回答 A 与回答 B，人更好何者？」

奖赏模型将此等成对偏好化为任意回答之标量分。此为第四亦即末一模型。

**完整之 PPO/RLHF 设置：**

| 模型 | 职能 | 可训否 |
|------|------|--------|
| 策略（演员） | 生成文本 | 可 |
| 价值（评论） | 估期望奖赏 | 可 |
| 参考 | 初始策略之冻结副本 | 否 |
| 奖赏 | 为回答评分 | 否 |"""

# Cell 19: The PPO Clipping Trick
lzh_markdown[19] = """\
## PPO 裁剪之巧

此乃 PPO 之心脏 —— 使训练稳定之机制。

### 概率比

PPO 算新旧策略间之**概率比**：

$$r(\\theta) = \\frac{\\pi_{\\text{new}}(a|s)}{\\pi_{\\text{old}}(a|s)}$$

此比之义：
- $r = 1$：新策略与旧策略行为无异（未变也）
- $r = 2$：新策略取此动作之概率倍于旧（大变也）
- $r = 0.5$：新策略取此动作之概率减半（远离也）

### 裁剪

PPO 将此比**裁剪**于 $[1-\\varepsilon, \\; 1+\\varepsilon]$（通常 $\\varepsilon = 0.2$）：

$$L^{\\text{CLIP}} = \\min\\left( r(\\theta) \\cdot A, \\;\\; \\text{clip}(r(\\theta),\\; 1-\\varepsilon,\\; 1+\\varepsilon) \\cdot A \\right)$$

`min` 乃关键 —— 取*更保守*之选：
- 于**善动作**（A > 0）：比可增至 1.2 而不得更进
- 于**恶动作**（A < 0）：比可减至 0.8 而不得更退

此防模型于任一更新中过度投注。"""

# Cell 22: Why Multiple Updates Per Batch?
lzh_markdown[22] = """\
### 何以同批多次更新

PPO 较 REINFORCE 之一大优势乃**样本效率**。

| 算法 | 数据用法 | 效率 |
|------|---------|------|
| REINFORCE | 生成一批，更新一次，弃数据 | 浪费 |
| PPO | 生成一批，**同批 3-8 次更新** | 高效 |

此可行者，盖裁剪机制约束每次更新之幅。虽同批经数次更新，策略变化不剧 —— 故数据犹「新」，可再学焉。

实践中，PPO 通常于每批生成数据上运行 **4 次内部回合**。"""

# Cell 24: Full PPO Training
lzh_markdown[24] = """\
## 完整 PPO 训练

今合诸要件：裁剪、KL 惩罚、多次内部更新。以 PPO 训练吾之 TinyLM 八十回合。"""

# Cell 28: PPO vs REINFORCE: Side by Side
lzh_markdown[28] = """\
## PPO 与 REINFORCE：并列比较

今将二算法对决。重训 REINFORCE 以为公平之比，绘二训练曲线。"""

# Cell 30: Hyperparameter Sensitivity
lzh_markdown[30] = """\
## 超参敏感性

PPO 有数关键超参。设之得当，至为要紧：

| 超参 | 典型值 | 过小 | 过大 |
|------|--------|------|------|
| **ε**（裁剪范围） | 0.1 - 0.3 | 学习过缓（更新过于保守） | 不稳（裁剪形同虚设） |
| **β**（KL 系数） | 0.01 - 0.2 | 奖赏作弊（模型任意漂移） | 不学（模型困于参考旁） |
| **内部回合数** | 3 - 8 | 浪费（弃数据） | 数据过时（依过时信息更新） |
| **批大小** | 128 - 512 | 梯度噪杂 | 每步较慢，然更稳 |

今观裁剪范围 ε 之实际效果："""

# Cell 32: Summary
lzh_markdown[32] = """\
## 总结

### PPO 于 REINFORCE 之上所增"""

# Cell 34: Key Takeaways
lzh_markdown[34] = """\
### 要旨

1. **PPO 于 REINFORCE 之上增三件：** 价值函数、参考模型、裁剪机制
2. **裁剪**防策略每步变化过巨，予以稳定性
3. **KL 惩罚**防奖赏作弊 —— 模型近于其参考
4. **同批多次更新**使 PPO 样本效率远胜 REINFORCE
5. **四模型驻存：** 策略（可训）、价值（可训）、参考（冻结）、奖赏（冻结）
6. **PPO 乃 RLHF 之主力** —— ChatGPT 与 GPT-4 对齐所用之法

### 代价

PPO 虽效，然费：四模型驻存则 7B 模型于 fp16 需约 56 GB，加优化器状态则逾 100 GB。此促生更简之替代。"""

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
        new_cell = dict(cell)
        new_cell["outputs"] = []
        new_cell["execution_count"] = None
    out_nb["cells"].append(new_cell)

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(out_nb, f, ensure_ascii=False, indent=1)

print(f"Created {OUT_PATH} with {len(out_nb['cells'])} cells")
