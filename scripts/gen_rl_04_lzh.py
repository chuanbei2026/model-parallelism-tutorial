"""Generate the Classical Chinese (文言文, LZH) version of 04-frontiers-and-systems.ipynb.

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
en_path = Path(__file__).resolve().parent.parent / "notebooks" / "en" / "reinforcement-learning" / "04-frontiers-and-systems.ipynb"
en_nb = json.loads(en_path.read_text())
en_code_cells = []
for c in en_nb["cells"]:
    if c["cell_type"] == "code":
        en_code_cells.append("".join(c["source"]))

# Verify we got the expected number
assert len(en_code_cells) == 13, f"Expected 13 code cells, got {len(en_code_cells)}"

cells = []

# ── 0: Header (markdown) ─────────────────────────────────────────────────
cells.append(md_cell(r"""---
**前沿与分布式系统**

**类目：** 大语言模型之强化学习

**难易：** 中等至深 | **所需时辰：** 约二刻半

---"""))

# ── 1: Overview (markdown) ───────────────────────────────────────────────
cells.append(md_cell(r"""## 一 — 概论

前四篇建立了大语言模型强化学习对齐之核心算法：

| 篇目 | 算法 | 要旨 |
|---|---|---|
| 00 | REINFORCE | 策略梯度——最简之强化学习算法 |
| 01 | PPO | 裁剪目标 + KL 惩罚，以求稳定更新 |
| 02 | DPO | 去奖赏模型——径优化偏好 |
| 03 | GRPO | 去评论模型——群组相对优势 |

此终篇**鸟瞰全局**，论二大题：

1. **算法之演进与前沿之法** —— 吾等何以至此，前路何在？
2. **强化学习训练之分布式系统** —— 大规模强化学习对齐之工程难题。

读毕此篇，当得此领域之完整图景：算法、取舍、系统设计与未解之难。"""))

# ── 2: Imports (code — verbatim) ─────────────────────────────────────────
cells.append(code_cell(en_code_cells[0]))

# ── 3: Full Algorithm Comparison (markdown) ──────────────────────────────
cells.append(md_cell(r"""## 二 — 算法总览

四种强化学习算法已悉论之。入前沿之前，先列而较之，一览其取舍。"""))

# ── 4: Algorithm comparison table (code — verbatim) ─────────────────────
cells.append(code_cell(en_code_cells[1]))

# ── 5: Memory bar chart (code — verbatim) ────────────────────────────────
cells.append(code_cell(en_code_cells[2]))

# ── 6: Evolution of RL for LLMs (markdown) ──────────────────────────────
cells.append(md_cell(r"""## 三 — 大语言模型强化学习之演进

大语言模型强化学习之史，乃渐次简化之史也。每一新算法，皆为解前法之困：

| 年 | 里程碑 | 所解之难 |
|---|---|---|
| 2017 | **PPO**（Schulman 等） | 以裁剪实现稳定之策略优化 |
| 2022 | **大规模 RLHF**（InstructGPT / ChatGPT） | 证 PPO + 人类反馈可用于对话对齐 |
| 2023 | **DPO**（Rafailov 等） | 径去奖赏模型 |
| 2024 | **SAPO、KTO、IPO** | 自对齐、更简之损失、无需偏好对 |
| 2025 | **GRPO / DeepSeek-R1** | 去评论模型，涌现推理能力 |

趋势昭然：**模型更少、损失更简、改进更自主。**"""))

# ── 7: Method timeline (code — verbatim) ─────────────────────────────────
cells.append(code_cell(en_code_cells[3]))

# ── 8: SAPO (markdown) ──────────────────────────────────────────────────
cells.append(md_cell(r"""## 四 — SAPO：自对齐偏好优化

前述诸法皆需某种外部信号：奖赏函数、人工偏好标注或验证器。**SAPO**（Self-Aligned Preference Optimization，自对齐偏好优化）更进一步：**若模型可自我改进，何如？**

要旨：

1. 以当前策略对同一提示**生成**二（或更多）回答
2. 以质量信号**评分** —— 长度、连贯、自洽（模型再答一次是否一致）或轻量验证器
3. **构造偏好对**：高分者为「所好」，低分者为「所恶」
4. 以 **DPO 式损失**训于此自生之偏好对
5. **反复** —— 改进后之模型下一轮生成更佳之回答

由此成一**自我改进之环**：模型自举其训练数据。"""))

# ── 9: SAPO info box (code — verbatim) ──────────────────────────────────
cells.append(code_cell(en_code_cells[4]))

# ── 10: SAPO circular flow (code — verbatim) ────────────────────────────
cells.append(code_cell(en_code_cells[5]))

# ── 11: Evolution in One Line (markdown) ─────────────────────────────────
cells.append(md_cell(r"""### 演进一言以蔽之

> REINFORCE（噪杂）→ **PPO**（稳定，四模型）→ **DPO**（更简，二模型，离线）→ **GRPO**（更简，二模型，在线）→ **SAPO**（自对齐）

每步去一组件或一依赖。此领域趋向**更简、更廉、更自主**之算法。"""))

# ── 12: Which LLMs Use What? (markdown) ──────────────────────────────────
cells.append(md_cell(r"""## 五 — 诸大模型所用何法？

此诸算法非纸上之谈——实驱动日用之模型。"""))

# ── 13: LLM methods table (code — verbatim) ─────────────────────────────
cells.append(code_cell(en_code_cells[6]))

# ── 14: Part 2 header (markdown) ────────────────────────────────────────
cells.append(md_cell(r"""---

## 第二部分 — 强化学习训练之分布式系统

大规模训练大语言模型时，强化学习对齐引入独特之分布式系统挑战，**超乎寻常之预训练**。预训练不过大规模前向-反向循环于文本数据上。强化学习对齐则增生成、多模型与在策略之约束。

试探三大挑战。"""))

# ── 15: Challenge 1 (markdown) ──────────────────────────────────────────
cells.append(md_cell(r"""### 挑战其一：多模型之编排

预训练涉**一**模型。强化学习对齐涉**二至四**：

| 算法 | 模型 | 注 |
|---|---|---|
| PPO | 四 | 策略、评论、奖赏、参考 |
| DPO | 二 | 策略、参考 |
| GRPO | 二 | 策略、参考 |

此诸模型**计算模式各异**：
- 有**冻结**者（参考、奖赏模型）——仅推理
- 有**训练**者（策略、评论）——前向 + 反向 + 优化器步
- 步间须**传递**结果（奖赏、对数概率、优势）

调度与显存管理遂为关键。不可一概而论「数据并行」——各模型需求与生命周期各异。"""))

# ── 16: Challenge 2 (markdown) ──────────────────────────────────────────
cells.append(md_cell(r"""### 挑战其二：生成之瓶颈

强化学习对齐需模型**生成文本**（自回归解码，逐词元而出）。此与标准训练步截然不同——且远慢于之。"""))

# ── 17: Bottleneck chart (code — verbatim) ──────────────────────────────
cells.append(code_cell(en_code_cells[7]))

# ── 18: Challenge 3 (markdown) ──────────────────────────────────────────
cells.append(md_cell(r"""### 挑战其三：在策略数据之时效

PPO 与 GRPO 皆**在策略（on-policy）** 之算法：训练数据须出自**当前**模型，不可用预存之数据集（此乃 DPO 之法）。

由此成一 **生成 → 训练 → 生成 → 训练** 之循环：

1. 以当前策略**生成**一批回答
2. **评分**（奖赏模型或验证器）
3. 以此批回答**训练**策略
4. **弃之** —— 数据已为离策略（off-policy）
5. 自第一步**重复**

其困：**管线气泡（pipeline bubbles）**。生成时，训练 GPU 闲置；训练时，生成 GPU 闲置。昂贵之算力遂浪费焉。

> **至善之境：** 令生成与训练重叠，使一切 GPU 常忙不辍。"""))

# ── 19: GPU Placement Strategies (markdown) ──────────────────────────────
cells.append(md_cell(r"""## 六 — GPU 分置之策

二至四大模型如何置于 GPU 之上？两大策略：

**同置（Colocated）：** 诸模型共享同一组 GPU。实现简易，然显存极为吃紧——须同时容纳全部模型。

**分置（Separated）：** 各模型占据专属之 GPU 组。总显存更充裕，然编排复杂——模型间须跨组传送数据。"""))

# ── 20: Colocated diagram (code — verbatim) ─────────────────────────────
cells.append(code_cell(en_code_cells[8]))

# ── 21: Separated diagram (code — verbatim) ─────────────────────────────
cells.append(code_cell(en_code_cells[9]))

# ── 22: RL Training Frameworks (markdown) ────────────────────────────────
cells.append(md_cell(r"""## 七 — 强化学习训练框架

数框架应运而生，以解此分布式之难。各于简洁与性能间取舍不同。"""))

# ── 23: Framework comparison table (code — verbatim) ────────────────────
cells.append(code_cell(en_code_cells[10]))

# ── 24: Code reference (code — verbatim) ────────────────────────────────
cells.append(code_cell(en_code_cells[11]))

# ── 25: Sequential vs Overlapped timeline (code — verbatim) ─────────────
cells.append(code_cell(en_code_cells[12]))

# ── 26: The Full Picture (markdown) ─────────────────────────────────────
cells.append(md_cell(r"""## 八 — 全局之景

汇总述之。

**强化学习对齐乃大语言模型训练之末阶**，继预训练与有监督微调之后。将能而未齐之模型化为遵从指令、远避其害、善于推理之器。

**诸算法并存**，各于简洁、显存、探索力与数据需求间取舍不同。无所谓「最优」之算法——择之在于所受之约束。

**此领域演进甚速。** 新算法数月一出。趋势向更简之法，需更少之模型、更少之人工标注。

**工程挑战与算法进步同等重要。** 纵有天下之最优算法，若不能高效运行于数百 GPU 之上，亦不足为用。多模型编排、生成瓶颈与在策略数据之时效，乃实际部署之真约束。"""))

# ── 27: Summary (markdown) ──────────────────────────────────────────────
cells.append(md_cell(r"""## 总结 — 全章回顾

本章遍论大语言模型强化学习对齐之全景，自基本原理至生产系统。

### 逐篇回顾

1. **基石（第 00 篇）：** 强化学习 = 生成 → 评分 → 学习 → 反复。REINFORCE 为最简之算法——单一策略，方差甚大，然为一切之基。

2. **PPO（第 01 篇）：** 以裁剪代理目标增稳定性，以 KL 惩罚保安全，以多轮更新提效率。须四模型驻存。

3. **DPO（第 02 篇）：** 以数学捷径（Bradley-Terry → 闭式最优策略）径去奖赏模型。仅二模型，离线训练，损失函数极简。

4. **GRPO（第 03 篇）：** 以群组相对优势去评论模型。仅二模型，在线训练，无价值函数。为 DeepSeek-R1 涌现推理之动力。

5. **前沿（第 04 篇，即此篇）：** SAPO 自对齐——模型自生偏好数据。分布式挑战：多模型编排、生成瓶颈、在策略数据时效。框架全景：DeepSpeed-Chat、OpenRLHF、veRL、TRL。"""))

# ── 28: Evolution one-liner (markdown) ──────────────────────────────────
cells.append(md_cell(r"""> REINFORCE（噪杂）→ **PPO**（稳定，四模型）→ **DPO**（更简，二模型，离线）→ **GRPO**（更简，二模型，在线）→ **SAPO**（自对齐）"""))

# ── 29: Further Reading (markdown) ──────────────────────────────────────
cells.append(md_cell(r"""## 延伸阅读

### 论文

- **PPO** — Schulman et al., 2017. [arXiv 1707.06347](https://arxiv.org/abs/1707.06347)
- **RLHF (InstructGPT)** — Ouyang et al., 2022. [arXiv 2203.02155](https://arxiv.org/abs/2203.02155)
- **DPO** — Rafailov et al., 2023. [arXiv 2305.18290](https://arxiv.org/abs/2305.18290)
- **GRPO / DeepSeek-R1** — 2025. [arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
- **SAPO** — 2024. [arXiv 2405.07863](https://arxiv.org/abs/2405.07863)

### 框架

- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) — 微软端到端 RLHF 管线
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 基于 Ray 之可扩展 RLHF
- [veRL](https://github.com/volcengine/verl) — 火山引擎混合 SPMD 强化学习训练框架
- [TRL](https://github.com/huggingface/trl) — HuggingFace Transformer Reinforcement Learning 库"""))


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

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "lzh" / "reinforcement-learning" / "04-frontiers-and-systems.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Created {out_path}  ({len(cells)} cells)")
