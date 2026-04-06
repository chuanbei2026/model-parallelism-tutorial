"""Generate Chinese (ZH) translation: 04-frontiers-and-systems.ipynb — 前沿与分布式系统."""
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
en_path = Path(__file__).resolve().parent.parent / "notebooks" / "en" / "reinforcement-learning" / "04-frontiers-and-systems.ipynb"
with open(en_path) as f:
    en_nb = json.load(f)

en_code_cells = [c for c in en_nb["cells"] if c["cell_type"] == "code"]

cells = []

# ── 0: Header (ZH) ──────────────────────────────────────────────────────────
cells.append(md_cell(r"""---
**Title:** 前沿与分布式系统（Frontiers & Distributed Systems）

**Chapter:** 强化学习与大语言模型

**Difficulty:** 中高级

**Estimated Time:** 40 分钟

---"""))

# ── 1: Overview (ZH) ────────────────────────────────────────────────────────
cells.append(md_cell(r"""## 1 — 概述

在前面四个 notebook 中，我们构建了基于强化学习的 LLM 对齐的核心算法工具箱：

| Notebook | 算法 | 核心思想 |
|---|---|---|
| 00 | REINFORCE | 策略梯度（Policy Gradient）——最简单的强化学习算法 |
| 01 | PPO | 裁剪目标 + KL 惩罚实现稳定更新 |
| 02 | DPO | 跳过奖励模型——直接优化偏好 |
| 03 | GRPO | 跳过评论家（Critic）——群组相对优势（Group-Relative Advantage） |

这个最终的 notebook **拉远视角**。我们涵盖两大主题：

1. **算法演进与前沿方法**——我们是如何走到这一步的，下一步是什么？
2. **强化学习训练的分布式系统**——使大规模强化学习对齐变得困难的工程挑战。

学完后，你将拥有这个领域的完整心智地图：算法、权衡、系统设计和开放问题。"""))

# ── 2: Imports (code — verbatim from EN) ─────────────────────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[0]["source"], "outputs": [], "execution_count": None})

# ── 3: The Full Algorithm Comparison (ZH) ──────────────────────────────────
cells.append(md_cell(r"""## 2 — 完整算法对比

我们已经见过四种用于 LLM 对齐的强化学习算法。在深入前沿之前，让我们把它们并排放在一起，在一个地方比较它们的权衡。"""))

# ── 4: Algorithm Comparison Table (code — verbatim from EN) ──────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[1]["source"], "outputs": [], "execution_count": None})

# ── 5: Memory Comparison (code — verbatim from EN) ─────────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[2]["source"], "outputs": [], "execution_count": None})

# ── 6: Evolution Timeline (ZH) ─────────────────────────────────────────────
cells.append(md_cell(r"""## 3 — LLM 强化学习的演进

LLM 强化学习的发展史就是一个渐进简化的故事。每一种新算法都解决了前一种方法的特定痛点：

| 年份 | 里程碑 | 解决了什么问题 |
|---|---|---|
| 2017 | **PPO**（Schulman et al.） | 通过裁剪实现稳定的策略优化 |
| 2022 | **大规模 RLHF**（InstructGPT / ChatGPT） | 证明 PPO + 人类反馈适用于聊天对齐 |
| 2023 | **DPO**（Rafailov et al.） | 完全移除了奖励模型 |
| 2024 | **SAPO、KTO、IPO** | 自对齐、更简单的损失函数、无需偏好对 |
| 2025 | **GRPO / DeepSeek-R1** | 移除了评论家，实现了涌现推理 |

趋势清晰可见：**更少的模型、更简单的损失函数、更自主的改进。**"""))

# ── 7: Timeline Visualization (code — verbatim from EN) ──────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[3]["source"], "outputs": [], "execution_count": None})

# ── 8: SAPO (ZH) ───────────────────────────────────────────────────────────
cells.append(md_cell(r"""## 4 — SAPO：自对齐偏好优化（Self-Aligned Preference Optimization）

我们之前介绍的算法都需要某种形式的外部信号：奖励函数、人类偏好标注或验证器（Verifier）。**SAPO**（自对齐偏好优化，Self-Aligned Preference Optimization）将前沿推得更远：**如果模型能自我改进呢？**

核心思想：

1. 用当前策略对同一提示**生成**两个（或更多）回复。
2. 用质量信号**评分**这些回复——可以是长度、连贯性、自一致性（模型再次被问到时是否同意自己的答案？）或轻量级验证器。
3. **创建偏好对**：得分更高的回复成为"偏好"输出，得分更低的成为"拒绝"输出。
4. 用 **DPO 风格的损失函数**在这些自生成的偏好对上训练。
5. **重复**——改进后的模型在下一轮生成更好的回复。

这创建了一个**自我改进循环（Self-Improvement Loop）**：模型自举式地生成自己的训练数据。"""))

# ── 9: Why SAPO Matters info_box (code — verbatim from EN) ──────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[4]["source"], "outputs": [], "execution_count": None})

# ── 10: Self-Improvement Loop Visualization (code — verbatim from EN) ────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[5]["source"], "outputs": [], "execution_count": None})

# ── 11: The Evolution in One Line (ZH) ─────────────────────────────────────
cells.append(md_cell(r"""### 一句话概括演进

> REINFORCE（高噪声）→ **PPO**（稳定，4 个模型）→ **DPO**（更简单，2 个模型，离线）→ **GRPO**（更简单，2 个模型，在线）→ **SAPO**（自对齐）

每一步都移除了一个组件或一个依赖。这个领域正在向**更简单、更廉价、更自主**的算法收敛。"""))

# ── 12: Which LLMs Use What? (ZH) ──────────────────────────────────────────
cells.append(md_cell(r"""## 5 — 哪些 LLM 在使用什么方法？

这些算法不是学术上的好奇心——它们驱动着你每天使用的模型。"""))

# ── 13: LLM Methods Table (code — verbatim from EN) ────────────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[6]["source"], "outputs": [], "execution_count": None})

# ── 14: Part 2: Distributed Systems (ZH) ───────────────────────────────────
cells.append(md_cell(r"""---

## 第二部分 — 强化学习训练的分布式系统

在大规模训练 LLM 时，强化学习对齐引入了**超越标准预训练**的独特分布式系统挑战。预训练"只是"在文本数据上进行大规模的前向-反向循环。强化学习对齐增加了生成、多模型和在策略（On-Policy）约束。

让我们探讨三个主要挑战。"""))

# ── 15: Challenge 1: Multi-Model Orchestration (ZH) ────────────────────────
cells.append(md_cell(r"""### 挑战 1：多模型编排（Multi-Model Orchestration）

预训练涉及**一个模型**。强化学习对齐涉及**两到四个**：

| 算法 | 模型数 | 说明 |
|---|---|---|
| PPO | 4 | 演员（Actor）、评论家（Critic）、奖励（Reward）、参考（Reference） |
| DPO | 2 | 策略（Policy）、参考（Reference） |
| GRPO | 2 | 策略（Policy）、参考（Reference） |

这些模型有**不同的计算模式**：
- 有些是**冻结的**（参考模型、奖励模型）——仅推理
- 有些在**训练**（演员、评论家）——前向 + 反向 + 优化器步骤
- 它们需要在步骤之间**通信**结果（奖励、对数概率、优势）

调度和内存管理变得至关重要。你不能简单地对整个系统进行"数据并行"——每个模型有不同的资源需求和生命周期。"""))

# ── 16: Challenge 2: The Generation Bottleneck (ZH) ────────────────────────
cells.append(md_cell(r"""### 挑战 2：生成瓶颈（Generation Bottleneck）

强化学习对齐要求模型**生成文本**（自回归解码，逐 token 生成）。这与标准训练步骤根本不同——而且慢得多。"""))

# ── 17: Bottleneck Visualization (code — verbatim from EN) ──────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[7]["source"], "outputs": [], "execution_count": None})

# ── 18: Challenge 3: On-Policy Data Freshness (ZH) ────────────────────────
cells.append(md_cell(r"""### 挑战 3：在策略数据新鲜度（On-Policy Data Freshness）

PPO 和 GRPO 是**在策略（On-Policy）**算法：它们的训练数据必须来自**当前**模型。你不能预先生成一个大数据集然后反复迭代（那是 DPO 的方法）。

这创建了一个**生成 → 训练 → 生成 → 训练**的循环：

1. 用当前策略**生成**一批回复
2. 对它们**评分**（奖励模型或验证器）
3. 在这些回复上**训练**策略
4. **丢弃**这些数据——它们现在已经离策略了
5. 从步骤 1 **重复**

问题是：**流水线气泡（Pipeline Bubbles）**。生成期间，训练 GPU 闲置。训练期间，生成 GPU 闲置。这浪费了昂贵的计算资源。

> **终极目标：** 重叠生成和训练，让所有 GPU 始终保持忙碌。"""))

# ── 19: GPU Placement Strategies (ZH) ──────────────────────────────────────
cells.append(md_cell(r"""## 6 — GPU 放置策略（GPU Placement Strategies）

如何将 2 到 4 个大模型放到 GPU 上？两种主要方法：

**共置（Colocated）：** 所有模型共享相同的 GPU。实现简单但内存严重受限——你需要足够的内存同时容纳所有模型。

**分离（Separated）：** 每个模型有专用的 GPU 组。总可用内存更多，但编排复杂——模型必须跨组相互发送数据。"""))

# ── 20: GPU Placement Colocated (code — verbatim from EN) ──────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[8]["source"], "outputs": [], "execution_count": None})

# ── 21: GPU Placement Separated (code — verbatim from EN) ─────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[9]["source"], "outputs": [], "execution_count": None})

# ── 22: RL Training Frameworks (ZH) ───────────────────────────────────────
cells.append(md_cell(r"""## 7 — 强化学习训练框架

已有多个框架涌现来应对这些分布式挑战。每个框架在简单性和性能之间做出了不同的权衡。"""))

# ── 23: Framework Comparison Table (code — verbatim from EN) ───────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[10]["source"], "outputs": [], "execution_count": None})

# ── 24: OpenRLHF Code Reference (code — verbatim from EN) ─────────────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[11]["source"], "outputs": [], "execution_count": None})

# ── 25: Overlap Strategy Visualization (code — verbatim from EN) ───────────
cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
              "source": en_code_cells[12]["source"], "outputs": [], "execution_count": None})

# ── 26: The Full Picture (ZH) ─────────────────────────────────────────────
cells.append(md_cell(r"""## 8 — 全局视角

让我们把所有内容串联起来。

**强化学习对齐是 LLM 训练的最后阶段**，在预训练和监督微调之后。它将一个有能力但不受控的模型转变为一个能遵循指令、避免伤害、善于推理的模型。

**存在多种算法**，每种在简单性、内存成本、探索能力和数据需求之间有不同的权衡。没有单一的"最佳"算法——选择取决于你的约束条件。

**这个领域发展迅速。** 每隔几个月就会出现新算法。趋势是向更简单的方法发展，需要更少的模型和更少的人类标注。

**工程挑战与算法进步同等重要。** 你可以拥有世界上最好的算法，但如果不能在数百块 GPU 上高效运行，它就不实用。多模型编排、生成瓶颈和在策略数据新鲜度是影响实际部署的真实约束。"""))

# ── 27: Summary (ZH) ──────────────────────────────────────────────────────
cells.append(md_cell(r"""## 总结 — 完整章节回顾

本章涵盖了 LLM 强化学习对齐的完整版图，从第一性原理到生产系统。

### 逐 Notebook 回顾

1. **基础（nb 00）：** 强化学习 = 生成 → 评分 → 学习 → 重复。REINFORCE 是最简单的算法——单一策略，高方差，但为所有后续方法奠定了基础。

2. **PPO（nb 01）：** 通过裁剪代理目标增加稳定性，通过 KL 惩罚保障安全性，通过多轮更新提高效率。需要 4 个模型在内存中。

3. **DPO（nb 02）：** 通过数学捷径（Bradley-Terry → 封闭形式最优策略）消除了奖励模型。只需 2 个模型，离线训练，极其简单的损失函数。

4. **GRPO（nb 03）：** 通过群组相对优势估计消除了评论家。只需 2 个模型，在线训练，无价值函数。驱动了 DeepSeek-R1 的涌现推理。

5. **前沿（nb 04，本 notebook）：** SAPO 用于自对齐——模型生成自己的偏好数据。分布式挑战：多模型编排、生成瓶颈、在策略数据新鲜度。框架版图：DeepSpeed-Chat、OpenRLHF、veRL、TRL。"""))

# ── 28: The Evolution in One Line (ZH) ─────────────────────────────────────
cells.append(md_cell(r"""> REINFORCE（高噪声）→ **PPO**（稳定，4 个模型）→ **DPO**（更简单，2 个模型，离线）→ **GRPO**（更简单，2 个模型，在线）→ **SAPO**（自对齐）"""))

# ── 29: Further Reading (ZH) ──────────────────────────────────────────────
cells.append(md_cell(r"""## 延伸阅读

### 论文

- **PPO** — Schulman et al., 2017. [arXiv 1707.06347](https://arxiv.org/abs/1707.06347)
- **RLHF (InstructGPT)** — Ouyang et al., 2022. [arXiv 2203.02155](https://arxiv.org/abs/2203.02155)
- **DPO** — Rafailov et al., 2023. [arXiv 2305.18290](https://arxiv.org/abs/2305.18290)
- **GRPO / DeepSeek-R1** — 2025. [arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
- **SAPO** — 2024. [arXiv 2405.07863](https://arxiv.org/abs/2405.07863)

### 框架

- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) — 微软的端到端 RLHF 管线
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 基于 Ray 的可扩展 RLHF，支持模型分离
- [veRL](https://github.com/volcengine/verl) — 火山引擎的混合 SPMD 强化学习训练框架
- [TRL](https://github.com/huggingface/trl) — HuggingFace 的 Transformer 强化学习库"""))


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

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "zh" / "reinforcement-learning" / "04-frontiers-and-systems.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Created {out_path}  ({len(cells)} cells)")
