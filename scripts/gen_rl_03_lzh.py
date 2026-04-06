"""Generate the Classical Chinese (文言文, LZH) version of 03-grpo.ipynb.

Reads the English notebook, copies code cells verbatim,
and replaces markdown cells with Classical Chinese translations.
"""
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


# ── Read English notebook to extract code cells ──────────────────────────
en_path = Path(__file__).resolve().parent.parent / "notebooks" / "en" / "reinforcement-learning" / "03-grpo.ipynb"
en_nb = json.loads(en_path.read_text())
en_code_cells = []
for c in en_nb["cells"]:
    if c["cell_type"] == "code":
        en_code_cells.append("".join(c["source"]))

# Verify we got the expected number
assert len(en_code_cells) == 15, f"Expected 15 code cells, got {len(en_code_cells)}"

cells = []

# ── 0: Header (markdown) ─────────────────────────────────────────────────
cells.append(md_cell(r"""---
**GRPO — 群组相对策略优化**

**类目：** 大语言模型之强化学习

**难易：** 中等 | **所需时辰：** 约三刻钟

---"""))

# ── 1: Overview (markdown) ───────────────────────────────────────────────
cells.append(md_cell(r"""## 一 — 概论

前篇之 **DPO** 径去奖赏模型（reward model），将四模型简至二。然 DPO 者，**离线（offline）** 之法也：训于固定之偏好数据，不生新数据。模型所学，限于数据中已有之策略。

**GRPO**（群组相对策略优化，Group Relative Policy Optimization，DeepSeek 2025）另辟蹊径。不去奖赏模型，而去**评论模型（critic / value function）** —— PPO 中另一耗资甚巨之组件——而保**在线学习（online learning）**。

要旨：以**群组内部比较**估算优势（advantage）。对同一提示生成数个输出，尽评其分，以群组统计量为基线。无需习得之评论模型。

| 算法 | 显存中模型数 | 在线否？ | 核心简化 |
|---|---|---|---|
| PPO | 四（策略 + 评论 + 参考 + 奖赏） | 然 | 裁剪代理目标 |
| DPO | 二（策略 + 参考） | 否 | 去奖赏模型 |
| **GRPO** | **二（策略 + 参考）** | **然** | **去评论模型——以群组统计量代之** |

GRPO 为 **DeepSeek-R1**（2025 年 1 月）所用，以纯强化学习而臻至先进之推理能力。"""))

# ── 2: Imports (code — verbatim) ─────────────────────────────────────────
cells.append(code_cell(en_code_cells[0]))

# ── 3: TinyLM + reward (code — verbatim) ─────────────────────────────────
cells.append(code_cell(en_code_cells[1]))

# ── 4: The Critic Problem (markdown) ─────────────────────────────────────
cells.append(md_cell(r"""## 二 — 评论模型之困

前篇 PPO 所述：**价值函数（value function，即评论模型）** 估「此状态优劣几何」——即于序列某位置所期之未来奖赏。

然训一良好之评论模型，**难也**：

- 需大量数据方能收敛
- 或**不精**，尤训练初期
- 评论不精，则优势估计不精，策略更新亦随之失当
- 需独立之网络、优化器与训练循环——复杂度倍增

评论模型乃 PPO 管线中最脆弱之环。优势估计之微误，经策略梯度放大，终成大谬。

> **若无需评论模型，何如？**"""))

# ── 5: Group Sampling: The Core Idea (markdown) ──────────────────────────
cells.append(md_cell(r"""## 三 — 群组采样：核心之理

不问「此状态优劣几何」（此须习得之评论模型），GRPO 问一更简之题：

> **「此输出较同组他输出孰优孰劣？」**

其法：

1. 对同一提示**生成 G 个输出**
2. 以奖赏函数**尽评之**
3. **优势 = 较群组均值优劣几何**

尽于此矣。无需习得之模型，无需评论模型之训练——唯一简单之统计计算。

**譬喻：** 评一试卷，不必延请名师。但将诸生之答比较，优者自见，劣者自明。排序由群组自身而生。"""))

# ── 6: Info box (code — verbatim) ────────────────────────────────────────
cells.append(code_cell(en_code_cells[2]))

# ── 7: Group-Relative Advantage (markdown) ───────────────────────────────
cells.append(md_cell(r"""## 四 — 群组相对优势

一群 $G$ 个输出中第 $i$ 个之**群组相对优势**为：

$$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$

此即 **z-score** —— 高于或低于群组均值几个标准差。

| 组分 | 释义 |
|---|---|
| $r_i$ | 第 $i$ 个输出之奖赏 |
| $\text{mean}(r_1, \ldots, r_G)$ | 群组均值——充**基线**（代评论模型之职） |
| $\text{std}(r_1, \ldots, r_G)$ | 归一化——使优势处于标准尺度 |
| $A_i > 0$ | 输出**优于均值**——强化之 |
| $A_i < 0$ | 输出**劣于均值**——抑制之 |

其巧处在此：基线自然校准于当前策略之水平。策略愈精，群组均值随之升，唯*相对于新水平*更优者方得正优势。"""))

# ── 8: Group advantage demo (code — verbatim) ───────────────────────────
cells.append(code_cell(en_code_cells[3]))

# ── 9: Group ranking viz (code — verbatim) ──────────────────────────────
cells.append(code_cell(en_code_cells[4]))

# ── 10: Group Size Matters (markdown) ────────────────────────────────────
cells.append(md_cell(r"""## 五 — 群组大小之影响

群组大小 $G$ 决优势估计之质。群组愈大，估计愈稳，然所耗算力亦愈多（每一输出皆须策略之前向传播）。

试以图观其效。"""))

# ── 11: Group size comparison (code — verbatim) ─────────────────────────
cells.append(code_cell(en_code_cells[5]))

# ── 12: GRPO vs PPO: What Changed? (markdown) ───────────────────────────
cells.append(md_cell(r"""## 六 — GRPO 与 PPO：何者异？

PPO 与 GRPO 皆算优势而以之更新策略。其异在于**基线之来源**：

- **PPO**：以**习得之价值函数** $V(s)$ 为基线。此乃神经网络，须与策略并训之。
- **GRPO**：以**群组均值**为基线。此乃当前批次之简单统计量——无需训练模型。

两法所得优势估计皆以零为中（优者得正优势，劣者得负优势）。然 GRPO 之基线**无额外代价** —— 无需模型，无需训练，无评论模型校准失当之虞。"""))

# ── 13: PPO vs GRPO comparison table (code — verbatim) ──────────────────
cells.append(code_cell(en_code_cells[6]))

# ── 14: GRPO formula breakdown (code — verbatim) ────────────────────────
cells.append(code_cell(en_code_cells[7]))

# ── 15: GRPO training (code — verbatim) ─────────────────────────────────
cells.append(code_cell(en_code_cells[8]))

# ── 16: GRPO training plots (code — verbatim) ──────────────────────────
cells.append(code_cell(en_code_cells[9]))

# ── 17: Verifier-Based Rewards (markdown) ───────────────────────────────
cells.append(md_cell(r"""## 七 — 验证器驱动之奖赏

实际诸多任务中，以**验证器（verifier）** 代习得之奖赏模型：

| 任务 | 验证器 | 奖赏 |
|---|---|---|
| **数学** | 验答案是否正确 | 1（正确）或 0（误） |
| **编程** | 运行单元测试 | 通过之测试比例 |
| **事实问答** | 核对真实答案 | 1（正确）或 0（误） |

此与 GRPO 天然契合：

- 奖赏信号**简而客观** —— 无需习得之奖赏模型
- 二值奖赏（正/误）与群组相对优势相得益彰：正确之输出得正优势，错误者得负优势
- **DeepSeek-R1** 正用此法：以数学与编程之验证为奖赏信号"""))

# ── 18: Math verifier demo (code — verbatim) ────────────────────────────
cells.append(code_cell(en_code_cells[10]))

# ── 19: DeepSeek-R1: GRPO in Practice (markdown) ────────────────────────
cells.append(md_cell(r"""## 八 — DeepSeek-R1：GRPO 之实践

**DeepSeek-R1**（2025 年 1 月）证明：纯强化学习足以教模型推理——无需任何人工撰写之思维链示范。

### 其法

1. **始于预训练基座模型**（无指令微调，无 SFT）
2. **施 GRPO**，以数学与编程验证为奖赏信号
3. **用大群组**（G = 64），使优势估计稳健
4. **长期训练**，遍及多样之数学与编程问题

### 涌现之能

模型**自发习得「思维链」推理**。未尝示之以逐步推理之范例，模型自学：

- 拆问题为子步
- 自检其作
- 遇推理死路则回溯
- 提交答案前先行验证

此乃 2025 年人工智能领域最惊人之成果之一：推理行为纯由强化学习之奖赏信号而涌现。"""))

# ── 20: DeepSeek info box (code — verbatim) ─────────────────────────────
cells.append(code_cell(en_code_cells[11]))

# ── 21: Head-to-Head (markdown) ──────────────────────────────────────────
cells.append(md_cell(r"""## 九 — REINFORCE、PPO、GRPO：对决

试以三法训于同一玩具问题，绘其学习曲线于一图。由此可具体感知算法之异同。"""))

# ── 22: Head-to-head training (code — verbatim) ─────────────────────────
cells.append(code_cell(en_code_cells[12]))

# ── 23: When to Use What (markdown) ──────────────────────────────────────
cells.append(md_cell(r"""## 十 — 何时用何法

至此已历三种大语言模型强化学习之法。各有所长：

- **GRPO**：最宜于有**客观可验之奖赏**之任务（数学、编程、事实问答）。简洁、在线、无评论模型。有可靠奖赏信号时之首选。
- **DPO**：最宜于有**固定偏好数据集**而求简者。离线而稳，然限于数据中已有之策略。
- **PPO**：最宜于**复杂奖赏景观**，探索为要，灵活至上。最为强大，亦最为昂贵。"""))

# ── 24: Algorithm selection guide (code — verbatim) ─────────────────────
cells.append(code_cell(en_code_cells[13]))

# ── 25: Summary (markdown) ──────────────────────────────────────────────
cells.append(md_cell(r"""## 总结

### 要旨

1. **GRPO 以群组相对优势代替习得之评论模型** —— 无需价值函数。
2. **群组优势**：$A_i = (r_i - \mu) / \sigma$ —— 群组输出内之简单 z-score 归一化。
3. **仅需二模型**（策略 + 参考），与 DPO 同——然 GRPO 为**在线**之法。
4. **在线学习**者，模型以生成探索新策略，非如 DPO 之固定数据集。
5. **最宜可验之任务**（数学、编程），奖赏客观而二值。
6. **DeepSeek-R1** 以 GRPO 开发涌现之推理能力，无需任何有监督之示范。

### 迄今之演进

| 篇目 | 算法 | 要旨 |
|---|---|---|
| 00 | REINFORCE | 策略梯度——增高奖赏之输出之概率 |
| 01 | PPO | 裁剪目标 + KL 惩罚，以求稳定更新 |
| 02 | DPO | 去奖赏模型——径优化偏好 |
| **03** | **GRPO** | **去评论模型——群组相对优势** |
| 04 | 全景 | 诸算法之演进与未来之路 |"""))

# ── 26: What's Next info box (code — verbatim) ──────────────────────────
cells.append(code_cell(en_code_cells[14]))


# ── Assemble notebook ────────────────────────────────────────────────────
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

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "lzh" / "reinforcement-learning" / "03-grpo.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Created {out_path}  ({len(cells)} cells)")
