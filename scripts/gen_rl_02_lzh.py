"""Generate the Classical Chinese (文言文) version of 02-dpo.ipynb."""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
EN_PATH = os.path.join(ROOT, "notebooks", "en", "reinforcement-learning", "02-dpo.ipynb")
OUT_PATH = os.path.join(ROOT, "notebooks", "lzh", "reinforcement-learning", "02-dpo.ipynb")

with open(EN_PATH) as f:
    en_nb = json.load(f)

lzh_markdown = {}

# Cell 0: Title header
lzh_markdown[0] = """\
---
**DPO — 直接偏好优化**

**篇目：** 强化学习用于大语言模型

**难易：** 中等 | **所需时辰：** 约三刻钟

---"""

# Cell 1: Overview
lzh_markdown[1] = """\
## 一 — 概论

前二篇以 **REINFORCE**（笔记本 00）与 **PPO**（笔记本 01）训练微型语言模型。PPO 虽效，然其流水线繁复：

- **四模型**同驻显存（演员、评论、奖赏模型、参考）
- 须另行训练奖赏模型
- 超参众多须调（裁剪范围、KL 系数、价值函数系数……）

2023 年，Rafailov 等人揭一出人意料之捷径：**可径优化人之偏好，无需训练奖赏模型。** 此法即**直接偏好优化（Direct Preference Optimization, DPO）**。

| PPO 流水线 | DPO 流水线 |
|---|---|
| 训奖赏模型 → 训价值函数 → 运行 PPO 循环 | 收集偏好对 → 运行 DPO（毕矣） |
| 四模型驻存 | **二模型**（策略 + 参考） |

本篇自头构建 DPO，逐步而行。"""

# Cell 4: The Complexity Problem
lzh_markdown[4] = """\
## 二 — 繁复之困

忆前篇之 PPO 流水线：

1. **训奖赏模型**于人之偏好数据
2. **训价值函数**（评论）以估期望未来奖赏
3. **运行 PPO 循环**：生成 → 评分 → 算优势 → 更新策略（带裁剪）

此需**四模型同驻 GPU 显存**：

| 模型 | 职能 | 可训否 |
|---|---|---|
| 演员（策略） | 生成文本 | 可 |
| 评论（价值函数） | 估优势 | 可 |
| 奖赏模型 | 为回答评分 | 冻结 |
| 参考模型 | KL 锚 | 冻结 |

七十亿参数之模型于 fp16，约 4 × 14 GB = **56 GB** 仅权重耳 —— 未计优化器状态、激活值、梯度。

> **若可径自偏好数据至更优策略，略过奖赏模型，奈何？**"""

# Cell 6: What Are Preference Pairs?
lzh_markdown[6] = """\
## 三 — 偏好对者何

RLHF 中收集**人之偏好**：给定一提示，人类标注者观两候选回答而曰*「A 优于 B」*。

此成每一提示 $x$ 之**偏好对**：

$$\\bigl(\\, y_w ,\\; y_l \\,\\bigr) \\quad \\text{其中 } y_w \\succ y_l$$

- $y_w$ —— **所好**（胜出）之回答
- $y_l$ —— **所恶**（败退）之回答

**例：**

| 提示 | 所好（$y_w$） | 所恶（$y_l$） |
|---|---|---|
| *「论猫」* | *「猫者，奇妙之生灵也……」* | *「不知，猫随便吧」* |

于吾之微型模型，以 `reward_fn` 构偏好对：含 "love" 之序列优于含 "hate" 者。"""

# Cell 8: The Bradley-Terry Preference Model
lzh_markdown[8] = """\
## 四 — Bradley-Terry 偏好模型

如何以数学模拟人之偏好乎？

**Bradley-Terry 模型**曰：人好回答 $A$ 甚于 $B$ 之概率为：

$$P(A \\succ B) \\;=\\; \\sigma\\bigl(r(A) - r(B)\\bigr)$$

其中 $\\sigma$ 乃 **sigmoid 函数** $\\sigma(x) = \\frac{1}{1 + e^{-x}}$。

直觉言之：

| 情形 | $r(A) - r(B)$ | $P(A \\succ B)$ |
|---|---|---|
| A 远优 | $\\gg 0$ | $\\approx 1$ |
| A、B 相当 | $= 0$ | $= 0.5$（掷钱也） |
| B 远优 | $\\ll 0$ | $\\approx 0$ |"""

# Cell 11: The Mathematical Shortcut
lzh_markdown[11] = """\
## 五 — 数学之捷径

此乃使 DPO 可行之关键推导。吾等逐步行之。

### 步骤一 — 闭式最优策略

标准 RLHF 之目标为：最大化奖赏而保持策略近于参考策略 $\\pi_{\\text{ref}}$（以 KL 散度度之）。此目标有**闭式解**：

$$\\pi^*(a \\mid s) \\;=\\; \\frac{1}{Z(s)}\\; \\pi_{\\text{ref}}(a \\mid s) \\;\\cdot\\; \\exp\\!\\left(\\frac{1}{\\beta}\\, r(a, s)\\right)$$

其中 $Z(s)$ 为归一化常数（配分函数），$\\beta$ 控策略可偏离参考之幅。

### 步骤二 — 以策略表奖赏

移项以解 $r$：

$$r(a, s) \\;=\\; \\beta \\log \\frac{\\pi^*(a \\mid s)}{\\pi_{\\text{ref}}(a \\mid s)} \\;+\\; \\beta \\log Z(s)$$

奖赏今以**策略对数比**加一常数表之。

### 步骤三 — 代入 Bradley-Terry

将奖赏表达代入 Bradley-Terry 偏好模型。$Z(s)$ 项**约去** —— 盖所好与所恶之回答共享同一提示：

$$\\boxed{L_{\\text{DPO}} = -\\log \\sigma\\!\\left(\\, \\beta \\left[\\, \\log \\frac{\\pi_\\theta(y_w)}{\\pi_{\\text{ref}}(y_w)} \\;-\\; \\log \\frac{\\pi_\\theta(y_l)}{\\pi_{\\text{ref}}(y_l)} \\,\\right] \\right)}$$

**白话释之：** 使模型更倾于生成人所好之输出而远人所恶者，*皆相对于参考模型而言。*"""

# Cell 13: Implementation
lzh_markdown[13] = """\
## 六 — 实现

观 DPO 之简。**全部损失**仅三行 PyTorch：

```python
lp_w = policy.log_probs_of(preferred) - ref.log_probs_of(preferred)
lp_l = policy.log_probs_of(rejected)  - ref.log_probs_of(rejected)
loss  = -F.logsigmoid(beta * (lp_w - lp_l)).mean()
```

较 PPO 之需裁剪机制、价值函数头、KL 惩罚计算、每批多次更新回合，DPO 将此一切凝于一简洁之目标函数。"""

# Cell 16: How DPO Updates the Policy
lzh_markdown[16] = """\
## 七 — DPO 如何更新策略

DPO 于模型权重实际为何？

对每一偏好对 $(y_w, y_l)$：

1. DPO **增** $\\log \\pi_\\theta(y_w)$ —— 使所好之输出更可能
2. DPO **减** $\\log \\pi_\\theta(y_l)$ —— 使所恶之输出更不可能

然此乃**相对于参考模型**而为之，此甚要：

- 若参考已强好 $y_w$，所可学者寡 —— DPO 几不改权重
- 若参考于 $y_w$ 与 $y_l$ 间**犹疑**，DPO 大力推之 —— 此乃模型可得最多之处

参考模型充锚之用：DPO 仅于当前模型犹疑处「用力」。"""

# Cell 18: Online vs Offline RL
lzh_markdown[18] = """\
## 八 — 在线与离线强化学习

一重要之辨：

- **DPO 为离线**：训于**固定之**偏好对数据集。模型于训练中从不生成新数据。
- **PPO 为在线**：每步皆自*当前*策略生成新数据而学之。

此乃根本之取舍：

| | 离线（DPO） | 在线（PPO） |
|---|---|---|
| **简易** | 甚简 | 繁复 |
| **稳定** | 甚稳 | 可不稳 |
| **探索** | 限于数据集 | 可发现新策略 |
| **算力** | 省（仅训练） | 费（生成 + 训练） |

离线更简更省，然模型仅能从数据集中已有之策略学。在线之法可发现标注者从未示范之新策略。"""

# Cell 20: DPO Variants
lzh_markdown[20] = """\
## 九 — DPO 之变体

DPO 之简催生一族相关算法，各解一具体之限：

### IPO — 恒等偏好优化（Identity Preference Optimisation）

以更简之**平方损失**代对数 sigmoid 损失于对数比之差。此避 DPO 久训于同一偏好数据时之过拟合。

### KTO — Kahneman-Tversky 优化

全无需偏好*对* —— 仅需标「好」或「差」之单独回答。此收集成本更低：标注者仅评一回答，无需比二。

### cDPO — 保守 DPO（Conservative DPO）

显式处理偏好数据中之**标签噪声**。若百分之十之对意外互换（标注者之误），cDPO 优雅退化而标准 DPO 则否。"""

# Cell 22: DPO vs PPO: Head-to-Head
lzh_markdown[22] = """\
## 十 — DPO 与 PPO：对决比较

今以二算法训于同一微型问题，比较训练过程中生成序列之质量。"""

# Cell 24: The Role of beta
lzh_markdown[24] = """\
## 十一 — $\\beta$ 之用

DPO 损失中之超参 $\\beta$ 控策略可偏离参考之幅：

$$L_{\\text{DPO}} = -\\log \\sigma\\!\\left(\\, \\beta \\bigl[\\, \\text{log-ratio}_w - \\text{log-ratio}_l \\,\\bigr] \\right)$$

- **小 $\\beta$**（如 0.05）：模型可大偏于参考。学之速然恐**模式坍缩** —— 模型或仅出一种序列。
- **大 $\\beta$**（如 0.5）：模型近于参考。安全然学之缓。
- **实践中典型值**：$\\beta \\in [0.1, 0.5]$。

此类 PPO 中之 KL 系数 —— 皆控探索与利用之取舍。"""

# Cell 26: Summary
lzh_markdown[26] = """\
## 总结

### 要旨

1. **DPO 去奖赏模型** —— 径优化偏好。数学之捷径：自闭式最优策略经 Bradley-Terry 至 DPO 损失。
2. 损失**仅三行 PyTorch**：算所好与所恶之对数比，推其差经对数 sigmoid。
3. **二模型代四** —— GPU 显存减半，较 PPO。
4. **离线训练**：简而稳，然无探索新策略之能。
5. **$\\beta$** 控策略可偏离参考之幅（类 PPO 中之 KL 系数）。
6. DPO 催生**一族变体**：IPO（平方损失）、KTO（无需对）、cDPO（抗噪声）。

### 迄今之演进

| 笔记本 | 算法 | 要旨 |
|---|---|---|
| 00 | REINFORCE | 策略梯度 —— 增高奖序列之概率 |
| 01 | PPO | 裁剪目标 + KL 惩罚以稳更新 |
| **02** | **DPO** | **略去奖赏模型 —— 径优化偏好** |
| 03 | GRPO | 兼去评论 —— 群组相对优势（次篇！） |"""

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
