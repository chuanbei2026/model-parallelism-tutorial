"""Generate the Chinese (simplified) translation: 08-reinforcement-learning.ipynb."""
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {"name": "python", "version": "3.12.0"}
}

def md(src):
    return nbformat.v4.new_markdown_cell(src)

def code(src):
    return nbformat.v4.new_code_cell(src)

nb.cells = [
    # ── 0: Title ──
    md("""---
**Title:** 强化学习与大语言模型 (Reinforcement Learning for LLMs)

**Category:** reinforcement-learning

**Difficulty:** 中高级 (Intermediate–Advanced)

**Estimated Time:** 60 min

---"""),

    # ── 1: Imports ──
    code("""import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from mp_tutorial.fonts import configure_cjk_fonts
configure_cjk_fonts()

from mp_tutorial.viz import (
    draw_training_pipeline, draw_rlhf_architecture,
    draw_rl_algorithm_comparison, draw_rl_gpu_placement,
    draw_ppo_clip, draw_group_ranking, draw_progressive_models,
    draw_method_timeline,
)
from mp_tutorial.formatting import (
    info_box, comparison_table, code_reference, formula_breakdown,
)
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font", category=UserWarning)

torch.manual_seed(42)
print("Ready!")"""),

    # ── 2: Overview ──
    md("""## 概述

### LLM 训练的三个阶段

构建一个实用的大语言模型（LLM）需要三个阶段。前两个阶段你已经在之前的 notebook 中学习过：

1. **预训练（Pre-training）** — 在海量文本语料上预测下一个 token（学习语言）
2. **监督微调（Supervised Fine-Tuning, SFT）** — 在人工撰写的示范数据上训练（学习遵循指令）
3. **强化学习对齐（RL Alignment）** — 针对人类*偏好*进行优化（学习人类真正想要什么）

本 notebook 介绍**第三阶段**：强化学习如何将一个有能力但未对齐的模型转变为一个能生成人类偏好回复的模型。

### 为什么 SFT 还不够

SFT 教模型*模仿*示范，但存在以下问题：
- 不可能为每一个问题都编写示范
- 模仿无法区分"还行"和"优秀"的回复
- 模型没有信号来判断回复中*哪些部分*是重要的

强化学习通过让模型**尝试并从反馈中学习**来解决这个问题 —— "这个回复比那个好。"

### 前置知识

- PyTorch 基础（`nn.Module`、优化器、损失函数）
- 推荐：[01 — 数据并行](01-data-parallelism.ipynb)"""),

    # ── 3: Training pipeline diagram ──
    code("""fig, ax = draw_training_pipeline()
plt.show()"""),

    # ── 4: Foundations intro ──
    md("""## 基础概念

在深入具体算法之前，我们先建立三个核心概念：
1. **策略（Policy）** — 模型正在做什么（它的行为）
2. **奖励（Reward）** — 我们如何评分它的表现
3. **策略梯度（Policy Gradient）** — 如何利用奖励来改进策略

### 我们的玩具模型：一个 5 词语言模型

为了让强化学习的概念更加具体，我们将在整个 notebook 中使用一个微型语言模型。它的词汇表只有 5 个单词，生成长度为 3 的 token 序列。

> **为什么这么小？** 只有 5 个单词、长度为 3，总共只有 125 种可能的序列。我们可以*精确地*看到模型学到了什么。这些概念可以直接扩展到 GPT 规模的模型 —— 只是规模不同而已。"""),

    # ── 5: TinyLM + reward ──
    code("""# ── Our tiny language model ──
VOCAB = ["I", "love", "cats", "hate", "dogs"]
V = len(VOCAB)
SEQ_LEN = 3

def decode(token_ids):
    \"\"\"Convert token indices to words.\"\"\"
    return " ".join(VOCAB[t] for t in token_ids.tolist())


class TinyLM(nn.Module):
    \"\"\"A minimal autoregressive language model.

    Given a sequence of tokens, predicts the next token at each position.
    This is exactly what GPT does — just with 5 words instead of 50,000.
    \"\"\"
    def __init__(self, vocab_size=V, embed_dim=8, hidden_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        \"\"\"Forward pass (teacher-forced): returns logits at every position.\"\"\"
        e = self.embed(x)
        h, _ = self.rnn(e)
        return self.head(h)  # (batch, seq_len, vocab_size)

    def log_probs_of(self, sequences):
        \"\"\"Compute log P(sequence) under the current policy.

        Sums log-probabilities of tokens at positions 1, 2, ..., T-1
        (position 0 is the fixed start token).
        \"\"\"
        logits = self.forward(sequences)           # (B, T, V)
        dist = torch.distributions.Categorical(logits=logits[:, :-1, :])
        per_token = dist.log_prob(sequences[:, 1:])  # (B, T-1)
        return per_token.sum(dim=1)                   # (B,)

    @torch.no_grad()
    def generate(self, batch_size, seq_len=SEQ_LEN):
        \"\"\"Generate sequences autoregressively (always starts with 'I').\"\"\"
        tokens = [torch.zeros(batch_size, 1, dtype=torch.long)]
        for _ in range(seq_len - 1):
            inp = torch.cat(tokens, dim=1)
            logits = self.forward(inp)[:, -1, :]
            token = torch.distributions.Categorical(logits=logits).sample()
            tokens.append(token.unsqueeze(1))
        return torch.cat(tokens, dim=1)


def reward_fn(sequences):
    \"\"\"Score sequences: +1 for 'I love cats', penalize 'hate'.

    This is our stand-in for human preferences.
    In real RLHF, a learned reward model replaces this function.
    \"\"\"
    rewards = []
    for seq in sequences:
        words = [VOCAB[t] for t in seq.tolist()]
        r = 0.0
        if "love" in words and "cats" in words:
            r = 1.0
        elif "love" in words:
            r = 0.3
        if "hate" in words:
            r -= 0.5
        rewards.append(r)
    return torch.tensor(rewards)


# Quick test
model = TinyLM()
seqs = model.generate(8)
for seq in seqs:
    print(f"  {decode(seq):20s}  reward = {reward_fn(seq.unsqueeze(0)).item():+.1f}")"""),

    # ── 6: Policy gradient ──
    md("""### 策略梯度（Policy Gradient）：核心思想

这是强化学习应用于语言模型的根本洞察：

> **提高获得高奖励的序列的概率。
> 降低获得低奖励的序列的概率。**

就是这样。其他一切（PPO、DPO、GRPO）都是对这个思想的改进。

用数学表达，**REINFORCE** 算法告诉我们：

$$\\nabla J(\\theta) = \\mathbb{E}\\left[ R \\cdot \\nabla \\log \\pi_\\theta(\\text{sequence}) \\right]$$

让我们逐项拆解："""),

    # ── 7: Formula breakdown ──
    code("""formula_breakdown([
    (
        "π_θ(sequence) — the probability the model<br>assigns to this sequence",
        "π<sub>θ</sub>(a₁, a₂, … | s)",
        "probs = model(sequence).softmax(-1)",
    ),
    (
        "log π_θ — take the log (easier to optimize,<br>"
        "turns products into sums)",
        "log π<sub>θ</sub> = Σ log P(aₜ | a₁..ₜ₋₁)",
        "log_probs = model.log_probs_of(sequence)",
    ),
    (
        "R — the reward score for this sequence<br>"
        "(how good was it?)",
        "R(sequence)",
        "reward = reward_fn(sequence)",
    ),
    (
        "R · ∇log π_θ — push the model toward<br>"
        "high-reward sequences",
        "R · ∇ log π<sub>θ</sub>",
        "loss = -(reward * log_probs).mean()",
    ),
], title="REINFORCE — Term by Term")"""),

    # ── 8: REINFORCE training ──
    code("""# ── Train with REINFORCE ──
torch.manual_seed(42)
policy = TinyLM()
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
reward_history = []

for step in range(300):
    seqs = policy.generate(batch_size=64)
    rewards = reward_fn(seqs)

    # REINFORCE with baseline (subtract mean to reduce variance)
    baseline = rewards.mean()
    advantages = rewards - baseline

    log_probs = policy.log_probs_of(seqs)
    loss = -(log_probs * advantages).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    reward_history.append(rewards.mean().item())
    if step % 60 == 0:
        print(f"Step {step:3d}  avg reward = {rewards.mean():.3f}")

print(f"\\nFinal avg reward: {reward_history[-1]:.3f}")"""),

    # ── 9: Plot REINFORCE ──
    code("""fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(reward_history, color="#4C72B0", lw=1.5, alpha=0.6)
# Smoothed
window = 20
smoothed = np.convolve(reward_history, np.ones(window)/window, mode="valid")
ax.plot(range(window-1, len(reward_history)), smoothed, color="#4C72B0", lw=2.5)
ax.set_xlabel("Training Step", fontsize=11)
ax.set_ylabel("Average Reward", fontsize=11)
ax.set_title("REINFORCE Training — Reward Over Time", fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

info_box(
    "REINFORCE works! The model learns to generate higher-reward sequences. "
    "But look at the <b>variance</b> (noise) in the curve — this is a known "
    "problem with vanilla policy gradients. PPO fixes this.",
    title="Observation"
)"""),

    # ── 10: What the model learned ──
    code("""# What did the model learn to generate?
print("Most common sequences after REINFORCE training:\\n")
seqs = policy.generate(500)
from collections import Counter
counts = Counter(decode(s) for s in seqs)
for text, count in counts.most_common(8):
    r = reward_fn(torch.tensor([[VOCAB.index(w) for w in text.split()]])).item()
    print(f"  {text:20s}  freq={count/500:.0%}  reward={r:+.1f}")"""),

    # ── 11: PPO intro ──
    md("""## PPO：近端策略优化（Proximal Policy Optimization）

REINFORCE 有两个问题：
1. **高方差** — 训练信号噪声很大（你看到了锯齿状的曲线）
2. **没有安全网** — 一次糟糕的更新就可能破坏模型已学到的内容

PPO 同时解决了这两个问题。让我们一步步构建它，每次添加一个组件。

### 逐步构建 4 个模型

真实的 RLHF（用于 LLM 的 PPO）需要**同时在内存中保存四个模型**。这听起来很可怕，所以让我们通过逐一添加来理解*为什么*需要它们 —— 每个模型解决一个具体问题。"""),

    # ── 12: Progressive models diagram ──
    code("""fig, axes = plt.subplots(2, 2, figsize=(12, 6))
for i, ax_flat in enumerate(axes.flat):
    plt.sca(ax_flat)
    ax_flat.clear()

# Draw each stage
for stage in range(1, 5):
    fig_s, ax_s = draw_progressive_models(stage=stage)
    plt.show()"""),

    # ── 13: PPO clip explanation ──
    md("""### PPO 裁剪技巧（Clipping Trick）

PPO 的核心洞察：**不要让策略在一步中改变太多**。

它通过计算一个*概率比率*来实现：

$$r(\\theta) = \\frac{\\pi_\\text{new}(\\text{action})}{\\pi_\\text{old}(\\text{action})}$$

- 如果 $r = 1$：新策略与旧策略行为完全一致
- 如果 $r = 2$：新策略采取这个动作的概率是旧策略的 2 倍
- 如果 $r = 0.5$：新策略的概率只有旧策略的一半

PPO 将这个比率**裁剪**到 $[1-\\varepsilon, 1+\\varepsilon]$（通常 $\\varepsilon = 0.2$），防止剧烈变化：

$$L^{\\text{CLIP}} = \\min\\left( r(\\theta) \\cdot A, \\; \\text{clip}(r(\\theta), 1-\\varepsilon, 1+\\varepsilon) \\cdot A \\right)$$"""),

    # ── 14: PPO clip diagram ──
    code("""fig, axes = draw_ppo_clip(eps=0.2)
plt.show()

info_box(
    "<b>Left:</b> When the advantage is positive (good action), PPO lets the ratio "
    "increase up to 1+ε but no further — preventing overcommitment.<br>"
    "<b>Right:</b> When the advantage is negative (bad action), PPO lets the ratio "
    "decrease down to 1-ε — the model moves away, but not too aggressively.",
    title="Reading the PPO Clip Plot"
)"""),

    # ── 15: PPO formula breakdown ──
    code("""formula_breakdown([
    (
        "Probability ratio — how much did the<br>"
        "policy change for this action?",
        "r(θ) = π<sub>new</sub>(a|s) / π<sub>old</sub>(a|s)",
        "ratio = (new_lp - old_lp).exp()",
    ),
    (
        "Clip the ratio — prevent drastic changes",
        "clip(r, 1-ε, 1+ε)",
        "clipped = torch.clamp(ratio, 1-eps, 1+eps)",
    ),
    (
        "Take the conservative option —<br>"
        "min for good actions, prevents overconfidence",
        "min(r·A, clip(r)·A)",
        "loss = -torch.min(ratio*adv, clipped*adv).mean()",
    ),
    (
        "KL penalty — don't drift too far from<br>"
        "the original (reference) model",
        "β · KL(π<sub>θ</sub> ‖ π<sub>ref</sub>)",
        "kl = (new_lp - ref_lp).mean()",
    ),
], title="PPO Loss — Term by Term")"""),

    # ── 16: RLHF architecture diagram ──
    code("""fig, ax = draw_rlhf_architecture()
plt.show()"""),

    # ── 17: PPO training ──
    code("""# ── Train with PPO ──
torch.manual_seed(42)
policy = TinyLM()
ref_model = deepcopy(policy)  # Frozen reference
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
eps_clip = 0.2
kl_coeff = 0.15
reward_hist, kl_hist = [], []

for epoch in range(80):
    # 1. Collect data with current policy
    seqs = policy.generate(batch_size=128)
    rewards = reward_fn(seqs)
    with torch.no_grad():
        old_lp = policy.log_probs_of(seqs)
        ref_lp = ref_model.log_probs_of(seqs)

    advantages = rewards - rewards.mean()

    # 2. Multiple update steps on same batch (PPO's key feature)
    for _ in range(4):
        new_lp = policy.log_probs_of(seqs)
        ratio = (new_lp - old_lp).exp()

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()

        kl = (new_lp - ref_lp).mean()
        loss = ppo_loss + kl_coeff * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    reward_hist.append(rewards.mean().item())
    kl_hist.append(kl.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}  reward={rewards.mean():.3f}  KL={kl.item():.3f}")"""),

    # ── 18: Plot PPO results ──
    code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(reward_hist, color="#4C72B0", lw=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Avg Reward")
ax1.set_title("PPO: Reward", fontweight="bold")
ax1.grid(alpha=0.3)

ax2.plot(kl_hist, color="#DD8452", lw=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("KL Divergence")
ax2.set_title("PPO: KL from Reference", fontweight="bold")
ax2.grid(alpha=0.3)

plt.suptitle("PPO Training — Reward Increases While KL Stays Bounded",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

info_box(
    "Notice: reward goes up (model improves) while KL stays bounded "
    "(model doesn't drift too far from the reference). This is the "
    "PPO + KL penalty working together — <b>stable improvement</b>.",
    title="PPO vs REINFORCE"
)"""),

    # ── 19: DPO intro ──
    md("""## DPO：直接偏好优化（Direct Preference Optimization）

PPO 虽然有效，但很复杂：4 个模型、多个训练阶段、奖励模型训练、超参数敏感。

**DPO 的洞察**（Rafailov et al., 2023）：我们可以完全跳过奖励模型。

### 数学捷径

带 KL 惩罚的强化学习目标有一个封闭形式的最优策略：

$$\\pi^*(a|s) = \\frac{1}{Z(s)} \\pi_{\\text{ref}}(a|s) \\cdot \\exp\\left(\\frac{1}{\\beta} r(a, s)\\right)$$

重新整理后，我们可以用策略来表达奖励：

$$r(a, s) = \\beta \\log \\frac{\\pi^*(a|s)}{\\pi_{\\text{ref}}(a|s)} + \\text{const}$$

将其代入 Bradley-Terry 偏好模型并化简：

$$L_{\\text{DPO}} = -\\log \\sigma\\left( \\beta \\left[ \\log \\frac{\\pi_\\theta(y_w)}{\\pi_{\\text{ref}}(y_w)} - \\log \\frac{\\pi_\\theta(y_l)}{\\pi_{\\text{ref}}(y_l)} \\right] \\right)$$

用通俗的话说：**让模型更可能生成被偏好的输出（$y_w$），更不可能生成被拒绝的输出（$y_l$），都是相对于参考模型而言。**"""),

    # ── 20: DPO formula breakdown ──
    code("""formula_breakdown([
    (
        "Log-ratio for preferred output —<br>"
        "how much more likely is it under our model<br>"
        "vs the reference?",
        "log π<sub>θ</sub>(y<sub>w</sub>) − log π<sub>ref</sub>(y<sub>w</sub>)",
        "lp_w = policy.log_probs_of(preferred) - ref.log_probs_of(preferred)",
    ),
    (
        "Log-ratio for rejected output —<br>"
        "same comparison for the bad response",
        "log π<sub>θ</sub>(y<sub>l</sub>) − log π<sub>ref</sub>(y<sub>l</sub>)",
        "lp_l = policy.log_probs_of(rejected) - ref.log_probs_of(rejected)",
    ),
    (
        "DPO loss — push the gap apart,<br>"
        "making preferred more likely",
        "−log σ(β · (lp_w − lp_l))",
        "loss = -F.logsigmoid(beta * (lp_w - lp_l)).mean()",
    ),
], title="DPO Loss — Term by Term")"""),

    # ── 21: DPO training ──
    code("""# ── Train with DPO ──
# Step 1: Create preference pairs
# (In real RLHF, these come from human annotators)
torch.manual_seed(42)

n_pairs = 500
preferred, rejected = [], []
for _ in range(n_pairs):
    # Preferred: sequences with "love"
    good = [[0, 1, 2], [0, 1, 4], [2, 1, 0]][np.random.randint(3)]  # I love cats, etc.
    bad  = [[0, 3, 2], [0, 3, 4], [3, 4, 0]][np.random.randint(3)]  # I hate cats, etc.
    preferred.append(good)
    rejected.append(bad)
preferred = torch.tensor(preferred)
rejected = torch.tensor(rejected)

print(f"Preference dataset: {n_pairs} pairs")
print(f"  Example preferred: {decode(preferred[0])}")
print(f"  Example rejected:  {decode(rejected[0])}")

# Step 2: Train
policy = TinyLM()
ref_model = deepcopy(policy)
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
beta = 0.1
dpo_losses = []

for step in range(300):
    idx = torch.randint(n_pairs, (64,))
    pref_batch = preferred[idx]
    rej_batch = rejected[idx]

    # DPO loss (just 3 lines!)
    lp_w = policy.log_probs_of(pref_batch) - ref_model.log_probs_of(pref_batch)
    lp_l = policy.log_probs_of(rej_batch) - ref_model.log_probs_of(rej_batch)
    loss = -F.logsigmoid(beta * (lp_w - lp_l)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dpo_losses.append(loss.item())
    if step % 60 == 0:
        print(f"Step {step:3d}  loss = {loss.item():.4f}")

# Check what the DPO-trained model generates
print("\\nAfter DPO training:")
seqs = policy.generate(200)
counts = Counter(decode(s) for s in seqs)
for text, count in counts.most_common(5):
    print(f"  {text:20s}  freq={count/200:.0%}")"""),

    # ── 22: Algorithm comparison diagram ──
    code("""fig, axes = draw_rl_algorithm_comparison()
plt.show()

info_box(
    "DPO reduces the system from <b>4 models to 2</b> — policy + reference. "
    "No reward model to train, no value function to estimate, no "
    "multi-phase training loop. The core idea: optimize preferences "
    "<b>directly</b> instead of going through a reward model.",
    title="DPO's Simplification"
)"""),

    # ── 23: GRPO intro ──
    md("""## GRPO：组相对策略优化（Group Relative Policy Optimization）

DPO 通过移除奖励模型来简化 RLHF。**GRPO**（DeepSeek-R1 使用的方法）进一步简化，连**评论家/价值模型（Critic/Value Model）**也移除了。

### 核心思想：组采样（Group Sampling）

GRPO 不训练一个单独的模型来估计"这个状态有多好"（即评论家），而是：

1. 对每个提示，从当前策略**生成一组 G 个输出**
2. 用奖励函数（或验证器）对每个输出**打分**
3. 计算优势值为：**比组平均好多少/差多少？**

$$A_i = \\frac{r_i - \\text{mean}(r_1, ..., r_G)}{\\text{std}(r_1, ..., r_G)}$$

这被称为**组相对优势（Group-Relative Advantage）** —— 不需要评论家模型！

> **为什么这有效？** 如果你生成足够多的输出（G=8 或更多），组平均值就是期望奖励的一个合理估计。高于平均值的输出是"优于平均"（正优势），低于平均值的是"劣于平均"（负优势）。这很简单，但对 DeepSeek-R1 来说足够有效。"""),

    # ── 24: GRPO demo ──
    code("""# ── Demonstrate group-relative advantage ──
torch.manual_seed(42)
policy = TinyLM()

# Generate a group of 8 outputs
group = policy.generate(8)
scores = reward_fn(group).numpy()
labels = [decode(s) for s in group]

print("Group of 8 generated sequences:")
for label, score in zip(labels, scores):
    print(f"  {label:20s}  reward = {score:+.2f}")

print(f"\\nGroup mean = {scores.mean():.2f}")
print(f"Group std  = {scores.std():.2f}")

fig, ax = draw_group_ranking(scores, labels)
plt.show()

info_box(
    "Green bars: sequences that scored above the group mean → positive advantage "
    "(model should generate these more often).<br>"
    "Red bars: below the mean → negative advantage (generate less often).<br>"
    "No critic model needed — just compare within the group!",
    title="GRPO: Group-Relative Advantage"
)"""),

    # ── 25: GRPO training ──
    code("""# ── Train with GRPO ──
torch.manual_seed(42)
policy = TinyLM()
ref_model = deepcopy(policy)
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
kl_coeff = 0.15
grpo_rewards = []

for step in range(300):
    # 1. Generate a group of outputs
    group_size = 16
    seqs = policy.generate(group_size)
    rewards = reward_fn(seqs)

    # 2. Group-relative advantage (GRPO's key idea)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # 3. Policy gradient with KL penalty
    new_lp = policy.log_probs_of(seqs)
    ref_lp = ref_model.log_probs_of(seqs)
    kl = (new_lp - ref_lp).mean()

    loss = -(new_lp * advantages.detach()).mean() + kl_coeff * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    grpo_rewards.append(rewards.mean().item())
    if step % 60 == 0:
        print(f"Step {step:3d}  reward={rewards.mean():.3f}  kl={kl.item():.3f}")

print("\\nAfter GRPO training:")
seqs = policy.generate(200)
counts = Counter(decode(s) for s in seqs)
for text, count in counts.most_common(5):
    print(f"  {text:20s}  freq={count/200:.0%}")"""),

    # ── 26: Comparison ──
    md("""## 算法对比

我们已经看到了三种算法的实际效果，现在来进行比较："""),

    # ── 27: Comparison table ──
    code("""comparison_table(
    headers=["Algorithm", "Models in Memory", "Needs Reward Model?",
             "Training Data", "Key Advantage", "Key Limitation"],
    rows=[
        ["PPO (RLHF)", "4 (actor + critic + reward + ref)",
         "Yes", "On-policy (generated)", "Most flexible, well-studied",
         "Complex, memory-hungry"],
        ["DPO", "2 (policy + ref)",
         "No", "Offline preference pairs", "Simple, stable",
         "No online exploration"],
        ["GRPO", "2 (policy + ref)",
         "No (uses verifier)", "On-policy groups", "Simple, on-policy",
         "Needs good reward/verifier"],
    ],
    title="PPO vs DPO vs GRPO"
)"""),

    # ── 28: Memory comparison ──
    code("""# Memory comparison for a 7B model
params_7b = 7e9
bytes_per_param = 2  # fp16

model_gb = params_7b * bytes_per_param / 1e9
print(f"One 7B model in fp16: {model_gb:.1f} GB\\n")

configs = [
    ("PPO (RLHF)", 4, "Actor + Critic + Reward + Reference"),
    ("DPO",        2, "Policy + Reference"),
    ("GRPO",       2, "Policy + Reference"),
]

for name, n_models, desc in configs:
    total = model_gb * n_models
    print(f"  {name:12s}  {n_models} models × {model_gb:.0f} GB = {total:.0f} GB  ({desc})")

print(f"\\nPPO needs ~{4*model_gb:.0f} GB just for model weights — before optimizer states!")
print("This is why DPO and GRPO are popular: half the memory.")"""),

    # ── 29: Evolution & Frontier ──
    md("""## 演进与前沿

### 到目前为止的发展

每种算法都解决了前一种方法的特定问题："""),

    # ── 30: Timeline diagram ──
    code("""fig, ax = draw_method_timeline()
plt.show()"""),

    # ── 31: Frontier text ──
    md("""### SAPO：自对齐偏好优化（Self-Aligned Preference Optimization, 2024）

最新的前沿更进一步：**如果我们根本不需要人类偏好呢？**

SAPO（自对齐偏好优化）从模型自身生成偏好对：
1. 对同一提示生成两个回复
2. 使用简单的质量信号（长度、连贯性、自一致性）对它们排序
3. 用 DPO 风格的损失在这些自生成的偏好对上训练

这创建了一个**自我改进循环** —— 模型自举式地进行自身对齐。

### 哪些 LLM 使用了什么方法？

| 模型 | 方法 | 关键创新 |
|------|------|---------|
| ChatGPT / GPT-4 | PPO (RLHF) | 大规模开创了 RLHF |
| Llama-2-Chat | PPO (RLHF) | 拒绝采样 + PPO |
| Zephyr | DPO | 证明 DPO 可以匹配 RLHF |
| DeepSeek-R1 | GRPO | 组相对优势，无需评论家 |
| Qwen-2 | DPO + GRPO | 混合方法 |"""),

    # ── 32: Distributed challenges ──
    md("""## 分布式挑战

在大规模训练 LLM 时，强化学习对齐引入了超越标准预训练的独特分布式系统挑战：

### 1. 多模型编排
PPO 需要 4 个模型（参数总量约为 4 倍）同时驻留在 GPU 内存中。即使是 DPO/GRPO 也需要 2 倍。这与预训练有根本不同，预训练只有一个模型。

### 2. 生成瓶颈（Generation Bottleneck）
强化学习需要从模型**生成文本**（自回归，逐 token 生成）。这比预训练的前向/反向传播慢得多，通常占据主要的运行时间。

### 3. 在线策略数据新鲜度（On-Policy Data Freshness）
PPO 和 GRPO 是**在线策略（On-Policy）**方法 —— 训练数据必须来自*当前*模型，而不是存储的数据集。你不能预先生成数据，必须按 生成 → 训练 → 生成 → 训练 的循环进行。

### GPU 放置策略

各框架通过不同的放置策略来解决这些挑战："""),

    # ── 33: GPU placement diagrams ──
    code("""fig1, ax1 = draw_rl_gpu_placement(strategy="colocated")
plt.show()

fig2, ax2 = draw_rl_gpu_placement(strategy="separated")
plt.show()

comparison_table(
    headers=["Framework", "Strategy", "Key Idea"],
    rows=[
        ["DeepSpeed-Chat", "Colocated → Separated", "All models on same GPUs, then offload"],
        ["OpenRLHF", "Separated (Ray)", "Dedicated GPU groups per model, Ray orchestration"],
        ["veRL (Volcano Engine)", "Hybrid SPMD", "Flexible placement, overlaps generation + training"],
    ],
    title="RL Training Frameworks"
)"""),

    # ── 34: Framework reference ──
    code("""code_reference(
    code=\"\"\"# OpenRLHF: Actor-Critic separated across GPU groups
class PPOTrainer:
    def __init__(self, actor, critic, reward_model, ref_model):
        # Each model can be on different GPU groups
        self.actor = ActorModelRayActor(actor, gpu_group_0)
        self.critic = CriticModelRayActor(critic, gpu_group_1)
        self.reward = RewardModelRayActor(reward_model, gpu_group_2)
        self.ref = RefModelRayActor(ref_model, gpu_group_0)  # shares with actor

    def step(self, prompts):
        # 1. Generate with actor
        sequences = self.actor.generate(prompts)
        # 2. Score with reward model
        rewards = self.reward.score(sequences)
        # 3. Compute values with critic
        values = self.critic.evaluate(sequences)
        # 4. PPO update
        self.actor.ppo_step(sequences, rewards, values)\"\"\",
    source="OpenRLHF",
    filepath="openrlhf/trainer/ppo_trainer.py"
)"""),

    # ── 35: Summary ──
    md("""## 总结与延伸阅读

### 核心要点

1. **强化学习对齐（RL Alignment）**是 LLM 训练的第三阶段（在预训练和 SFT 之后）—— 优化人类*偏好*的内容，而非人类示范的内容
2. **策略梯度（Policy Gradient）**是核心思想：提高高奖励输出的概率，降低低奖励输出的概率
3. **PPO** 通过裁剪目标和 KL 惩罚稳定训练，但需要 4 个模型在内存中
4. **DPO** 通过直接优化偏好消除了奖励模型 —— 从 4 个模型减少到 2 个
5. **GRPO** 通过组相对优势消除了评论家 —— 更简单的在线策略训练（DeepSeek-R1 使用）
6. **SAPO** 更进一步：自生成偏好用于自举式对齐
7. 在大规模场景下，分布式强化学习面临独特挑战：多模型内存、生成瓶颈、在线策略数据新鲜度

### 一句话概括演进

> REINFORCE（高噪声）→ **PPO**（稳定，4 个模型）→ **DPO**（更简单，2 个模型，离线）→ **GRPO**（更简单，2 个模型，在线）→ **SAPO**（自对齐）

### 延伸阅读

**论文：**
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — 近端策略优化
- [RLHF (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — 用人类反馈训练语言模型遵循指令
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — 直接偏好优化
- [GRPO / DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — DeepSeek-R1: 通过强化学习激发 LLM 的推理能力
- [SAPO (2024)](https://arxiv.org/abs/2405.07863) — 自对齐偏好优化

**框架：**
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) — 端到端 RLHF 管线
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — 基于 Ray 的分布式 RLHF
- [veRL](https://github.com/volcengine/verl) — 火山引擎 LLM 强化学习框架
- [TRL](https://github.com/huggingface/trl) — Hugging Face Transformer 强化学习库"""),
]

nbformat.write(nb, "notebooks/zh/08-reinforcement-learning.ipynb")
print(f"Created notebook with {len(nb.cells)} cells")
