"""Generate Chinese (ZH) translation: 03-grpo.ipynb — GRPO: 群组相对策略优化."""
import json
from pathlib import Path
import uuid


def code_cell(source):
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.strip().splitlines(True),
        "outputs": [],
        "execution_count": None,
    }


def md_cell(source):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source.strip().splitlines(True),
    }


# ── Read the EN notebook to extract code cells verbatim ──
en_path = Path(__file__).resolve().parent.parent / "notebooks" / "en" / "reinforcement-learning" / "03-grpo.ipynb"
with open(en_path) as f:
    en_nb = json.load(f)

en_code_cells = [c for c in en_nb["cells"] if c["cell_type"] == "code"]

cells = []

# ── 0: Header (ZH) ──────────────────────────────────────────────────────────
cells.append(md_cell(r"""---
**Title:** GRPO — 群组相对策略优化（Group Relative Policy Optimization）

**Chapter:** 强化学习与大语言模型

**Difficulty:** 中级

**Estimated Time:** 45 分钟

---"""))

# ── 1: Overview (ZH) ────────────────────────────────────────────────────────
cells.append(md_cell(r"""## 1 — 概述

在上一个 notebook 中，**DPO** 通过完全移除奖励模型（Reward Model）来简化 PPO——将 4 个模型减少到 2 个。但 DPO 是**离线（Offline）**的：它在固定的偏好对数据集上训练，永远不会生成新数据。模型只能从数据集中已有的策略中学习。

**GRPO**（群组相对策略优化，Group Relative Policy Optimization，DeepSeek 2025）走了一条不同的路。它不是移除奖励模型，而是移除了**评论家/价值函数（Critic / Value Function）**——PPO 中另一个昂贵的组件——同时保留了**在线学习（Online Learning）**。

核心思想：通过在**一组输出内部**进行比较来估计优势（Advantage）。对同一个提示生成多个输出，对它们全部评分，用组统计量作为基线。不需要学习评论家模型。

| 算法 | 内存中的模型数 | 在线？ | 关键简化 |
|---|---|---|---|
| PPO | 4（策略 + 评论家 + 参考 + 奖励） | 是 | 裁剪代理目标 |
| DPO | 2（策略 + 参考） | 否 | 消除奖励模型 |
| **GRPO** | **2（策略 + 参考）** | **是** | **消除评论家——使用组统计量** |

GRPO 被 **DeepSeek-R1**（2025 年 1 月）用于通过纯强化学习实现最先进的推理能力。"""))

# ── 2: Imports (code — verbatim from EN) ─────────────────────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[0]["source"], "outputs": [], "execution_count": None})

# ── 3: TinyLM Setup (code — verbatim from EN) ───────────────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[1]["source"], "outputs": [], "execution_count": None})

# ── 4: The Critic Problem (ZH) ─────────────────────────────────────────────
cells.append(md_cell(r"""## 2 — 评论家问题（The Critic Problem）

回想 PPO 那个 notebook：**价值函数（Value Function）**，即评论家（Critic），估计的是"这个状态有多好？"——也就是从序列中某个位置开始的期望未来奖励。

训练一个好的评论家**很难**：

- 需要大量数据才能收敛
- 可能**不准确**，尤其是在训练早期
- 糟糕的评论家会给出错误的优势估计，进而导致错误的策略更新
- 它需要自己的网络、优化器和训练循环——使复杂度翻倍

评论家是 PPO 管线中最脆弱的部分。优势估计中的小误差会级联放大为策略梯度（Policy Gradient）中的大误差。

> **如果我们根本不需要评论家呢？**"""))

# ── 5: Group Sampling Intuition (ZH) ───────────────────────────────────────
cells.append(md_cell(r"""## 3 — 组采样：核心思想

GRPO 不问"这个状态有多好？"（这需要一个学习的评论家），而是问一个更简单的问题：

> **"这个输出与同一提示的其他输出相比有多好？"**

方法：

1. 对同一个提示**生成 G 个输出**
2. 用奖励函数**对它们全部评分**
3. **优势 = 比组平均好多少/差多少**

就是这样。不需要学习的模型，不需要评论家的训练循环——只是一个简单的统计计算。

**类比：** 如果你可以比较所有学生的答案，你就不需要教授来评分。最好的答案高于平均水平，最差的低于平均水平。排名自然从群组中涌现出来。"""))

# ── 6: Key Insight info_box (code — verbatim from EN) ───────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[2]["source"], "outputs": [], "execution_count": None})

# ── 7: Group-Relative Advantage Formula (ZH) ───────────────────────────────
cells.append(md_cell(r"""## 4 — 群组相对优势（Group-Relative Advantage）

组中第 $i$ 个输出的**群组相对优势（Group-Relative Advantage）**为：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

这就是一个 **z 分数**——比组平均值高或低多少个标准差。

| 成分 | 含义 |
|---|---|
| $r_i$ | 第 $i$ 个输出的奖励 |
| $\text{mean}(r_1, \ldots, r_G)$ | 组平均值——充当我们的**基线**（替代评论家） |
| $\text{std}(r_1, \ldots, r_G)$ | 归一化——将优势放在标准尺度上 |
| $A_i > 0$ | 输出**优于平均**——强化它 |
| $A_i < 0$ | 输出**劣于平均**——抑制它 |

注意其优雅之处：基线会自动校准到当前策略（Policy）的能力水平。随着策略改进，组平均值上升，只有*相对于新水平*更好的输出才会获得正优势。"""))

# ── 8: Group Sampling Demo (code — verbatim from EN) ───────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[3]["source"], "outputs": [], "execution_count": None})

# ── 9: Group Ranking Visualization (code — verbatim from EN) ───────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[4]["source"], "outputs": [], "execution_count": None})

# ── 10: Group Size Matters (ZH) ────────────────────────────────────────────
cells.append(md_cell(r"""## 5 — 组大小的影响

组大小 $G$ 控制优势估计的质量。更大的组给出更稳定的估计，但需要更多计算（每个输出都需要通过策略进行一次前向传播）。

让我们直观地看看效果。"""))

# ── 11: Group Size Comparison (code — verbatim from EN) ────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[5]["source"], "outputs": [], "execution_count": None})

# ── 12: GRPO vs PPO: What Changed? (ZH) ───────────────────────────────────
cells.append(md_cell(r"""## 6 — GRPO vs PPO：变了什么？

PPO 和 GRPO 都计算优势并用它来更新策略。区别在于**基线从何而来**：

- **PPO**：使用**学习的价值函数** $V(s)$ 作为基线。这是一个必须与策略一起训练的神经网络。
- **GRPO**：使用**组平均值**作为基线。这是从当前批次计算的简单统计量——不需要训练模型。

两种方法都产生以零为中心的优势估计（好的输出获得正优势，差的输出获得负优势）。但 GRPO 的基线是**免费的**——不需要额外的模型、额外的训练，也不存在校准不良的评论家的风险。"""))

# ── 13: GRPO vs PPO Comparison Table (code — verbatim from EN) ─────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[6]["source"], "outputs": [], "execution_count": None})

# ── 14: GRPO Formula Breakdown (code — verbatim from EN) ──────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[7]["source"], "outputs": [], "execution_count": None})

# ── 15: GRPO Training (code — verbatim from EN) ──────────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[8]["source"], "outputs": [], "execution_count": None})

# ── 16: GRPO Training Curve (code — verbatim from EN) ────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[9]["source"], "outputs": [], "execution_count": None})

# ── 17: Verifier-Based Rewards (ZH) ───────────────────────────────────────
cells.append(md_cell(r"""## 7 — 基于验证器的奖励（Verifier-Based Rewards）

在许多实际任务中，我们使用**验证器（Verifier）**而非学习的奖励模型：

| 任务 | 验证器 | 奖励 |
|---|---|---|
| **数学** | 检查答案是否正确 | 1（正确）或 0（错误） |
| **代码** | 运行单元测试 | 通过的测试比例 |
| **事实问答** | 与标准答案对照 | 1（正确）或 0（错误） |

这与 GRPO 完美契合：

- 奖励信号**简单且客观**——不需要学习的奖励模型
- 二值奖励（正确/错误）与群组相对优势配合良好：正确的输出获得正优势，错误的获得负优势
- **DeepSeek-R1** 正是使用了这种方法：数学和代码验证作为奖励信号"""))

# ── 18: Verifier Example (code — verbatim from EN) ────────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[10]["source"], "outputs": [], "execution_count": None})

# ── 19: DeepSeek-R1 Case Study (ZH) ──────────────────────────────────────
cells.append(md_cell(r"""## 8 — DeepSeek-R1：GRPO 实战

**DeepSeek-R1**（2025 年 1 月）证明了纯强化学习可以教会模型推理——无需在人工编写的思维链（Chain of Thought）示例上进行任何监督微调（SFT）。

### 它是怎么做到的

1. **从预训练的基础模型开始**（没有指令微调，没有 SFT）
2. **应用 GRPO**，使用数学和代码验证作为奖励信号
3. **使用大组大小**（G = 64）以获得稳定的优势估计
4. **在多样的数学和代码问题上训练多轮**

### 涌现了什么

模型**自发地发展出了"思维链"推理能力**。在从未见过逐步推理示例的情况下，模型学会了：

- 将问题分解为子步骤
- 检查自己的工作
- 当推理路径失败时回溯
- 在最终确认前验证答案

这是 2025 年 AI 领域最引人注目的结果之一：推理行为纯粹从强化学习的奖励信号中涌现出来。"""))

# ── 20: Emergent Reasoning info_box (code — verbatim from EN) ─────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[11]["source"], "outputs": [], "execution_count": None})

# ── 21: Three-Way Algorithm Comparison (ZH) ───────────────────────────────
cells.append(md_cell(r"""## 9 — REINFORCE vs PPO vs GRPO：正面对比

让我们在同一个玩具问题上训练这三种算法，并在同一张图上比较它们的学习曲线。这将给我们一个具体的直观感受，了解算法在实践中的差异。"""))

# ── 22: Three-Way Comparison Plot (code — verbatim from EN) ───────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[12]["source"], "outputs": [], "execution_count": None})

# ── 23: When to Use What (ZH) ─────────────────────────────────────────────
cells.append(md_cell(r"""## 10 — 何时使用哪种算法

我们现在已经介绍了三种不同的 LLM 强化学习方法。每种都有其优势：

- **GRPO**：最适合有**客观、可验证奖励**的任务（数学、代码、事实问答）。简单，在线，无评论家。当你有可靠的奖励信号时，这是首选算法。
- **DPO**：最适合你有**固定偏好数据集**且追求简单的场景。离线且稳定，但受限于数据中已有的策略。
- **PPO**：最适合**复杂奖励场景**，探索很重要且需要最大灵活性的情况。最强大但最昂贵。"""))

# ── 24: When to Use What Table (code — verbatim from EN) ──────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[13]["source"], "outputs": [], "execution_count": None})

# ── 25: Summary (ZH) ──────────────────────────────────────────────────────
cells.append(md_cell(r"""## 总结

### 核心要点

1. **GRPO 用群组相对优势替代了学习的评论家**——不需要价值函数。
2. **组优势（Group Advantage）**：$A_i = (r_i - \mu) / \sigma$——在一组输出内进行简单的 z 分数归一化。
3. **只需 2 个模型**（策略 + 参考），与 DPO 相同——但 GRPO 是**在线（Online）**的。
4. **在线学习**意味着模型通过生成来探索新策略，不像 DPO 的固定数据集。
5. **非常适合可验证的任务**（数学、代码），具有客观的二值奖励。
6. **DeepSeek-R1** 使用 GRPO 发展出了涌现推理能力，且不依赖任何监督示范。

### 目前的演进历程

| Notebook | 算法 | 核心思想 |
|---|---|---|
| 00 | REINFORCE | 策略梯度——提高高奖励输出的概率 |
| 01 | PPO | 裁剪目标 + KL 惩罚实现稳定更新 |
| 02 | DPO | 跳过奖励模型——直接优化偏好 |
| **03** | **GRPO** | **跳过评论家——群组相对优势** |
| 04 | 全局视角 | 这些算法如何演进 + 未来方向 |"""))

# ── 26: What's Next info_box (code — verbatim from EN) ────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[14]["source"], "outputs": [], "execution_count": None})


# ── Assemble notebook ────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "cells": cells,
}

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "zh" / "reinforcement-learning" / "03-grpo.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Created {out_path}  ({len(cells)} cells)")
