"""Generate notebook 00: RL Foundations — From Rewards to Policy Gradients.

Notebook 00 in the 5-notebook "Reinforcement Learning for LLMs" chapter.
Covers fundamentals for readers with zero RL background, using a 5-token
TinyLM as a running example.

Usage:
    python scripts/gen_rl_00_foundations.py
"""

import json
import uuid
from pathlib import Path


def _cell_id():
    return uuid.uuid4().hex[:8]


def code_cell(source, cell_id=None):
    return {
        "cell_type": "code",
        "id": cell_id or _cell_id(),
        "metadata": {},
        "source": source.strip("\n").split("\n") if isinstance(source, str) else source,
        "outputs": [],
        "execution_count": None,
    }


def md_cell(source, cell_id=None):
    return {
        "cell_type": "markdown",
        "id": cell_id or _cell_id(),
        "metadata": {},
        "source": source.strip("\n").split("\n") if isinstance(source, str) else source,
    }


def _join_lines(lines):
    """Ensure source is stored as a list of lines with newlines (nbformat convention)."""
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        else:
            result.append(line)
    return result


def build_cells():
    cells = []

    # ── 0: Header ──
    cells.append(md_cell("""\
---
**Title:** RL Foundations — From Rewards to Policy Gradients

**Chapter:** Reinforcement Learning for LLMs

**Difficulty:** Beginner–Intermediate

**Estimated Time:** 45 min

---"""))

    # ── 1: Overview ──
    cells.append(md_cell("""\
## 1. What Is Reinforcement Learning?

Imagine training a dog. You ask it to sit:
- **Dog sits → you give a treat** → the dog learns to sit more often.
- **Dog jumps on the table → no treat** → the dog learns to avoid that.

That's reinforcement learning in a nutshell: **try something, get feedback, adjust behavior**.

Now replace "dog" with "language model", "sit" with "generate a helpful response", and "treat" with "a high reward score". That's RL for LLMs.

### The 3 Stages of LLM Training

Building a useful LLM takes three stages:

| Stage | What It Does | Analogy |
|-------|-------------|---------|
| **1. Pre-training** | Predict next token on massive text | Reading every book in the library |
| **2. SFT** (Supervised Fine-Tuning) | Train on human-written examples | A tutor showing you worked examples |
| **3. RL Alignment** | Optimize for human *preferences* | Practice problems with a grading rubric |

**Why do we need stage 3?** SFT teaches the model to *imitate*, but imitation alone can't distinguish "okay" from "great". RL lets the model **try things and learn from feedback** — discovering responses that humans actually prefer.

### What This Notebook Covers

We'll build your RL intuition from scratch:
1. Core RL vocabulary (policy, reward, episode, etc.)
2. A tiny 5-word language model you can fully understand
3. The REINFORCE algorithm — the simplest policy gradient method
4. Why variance is a problem, and how baselines help

**Prerequisites:** PyTorch basics only (tensors, `nn.Module`, optimizers)."""))

    # ── 2: Imports ──
    cells.append(code_cell("""\
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from collections import Counter
import itertools

from mp_tutorial.fonts import configure_cjk_fonts
configure_cjk_fonts()

from mp_tutorial.viz import draw_training_pipeline, draw_progressive_models
from mp_tutorial.formatting import info_box, comparison_table, formula_breakdown

import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font", category=UserWarning)

torch.manual_seed(42)
print("Ready!")"""))

    # ── 3: Training Pipeline Diagram ──
    cells.append(md_cell("""\
### The Three-Stage Pipeline

The diagram below shows how modern LLMs are trained. This notebook focuses on the **third stage** — RL alignment."""))

    cells.append(code_cell("""\
fig, ax = draw_training_pipeline()
plt.show()"""))

    # ── 4: Why SFT Is Not Enough ──
    cells.append(md_cell("""\
## 2. Why SFT Is Not Enough

Supervised Fine-Tuning is powerful, but it has fundamental limitations:

- **Imitation, not judgment.** SFT teaches the model to copy demonstrations. But it can't distinguish an "okay" response from a "great" one — it treats all training examples equally.
- **No signal for what matters.** A response might be 95% good but have one misleading sentence. SFT has no way to say "this part is bad, fix it."
- **Can't cover everything.** You can't write a human demonstration for every possible question. The model needs to *generalize* beyond its training examples.

**RL solves this:** instead of showing the model what to say, we let it **try things and learn from feedback**. "This response scored 0.9, that one scored 0.3 — do more of the first kind."

This is a fundamentally different learning signal: **preferences over outcomes**, not just examples to copy."""))

    # ── 5: Core RL Concepts ──
    cells.append(md_cell("""\
## 3. Core RL Concepts

RL has its own vocabulary. Here's each term with its LLM equivalent:

| RL Term | General Meaning | LLM Equivalent |
|---------|----------------|----------------|
| **Agent** | The decision-maker | The language model |
| **Environment** | The world the agent interacts with | The user who reads the response |
| **Action** | A choice the agent makes | Generating a token (or a full sequence) |
| **State** | The current situation | The conversation so far (prompt + tokens generated so far) |
| **Reward** | Feedback signal (number) | Human preference score (or reward model output) |
| **Policy** | The agent's strategy — how it chooses actions | The model's probability distribution over tokens |
| **Episode** | One complete interaction | One full generation: prompt → complete response |"""))

    cells.append(code_cell("""\
info_box(
    "If you remember one thing from this notebook: "
    "<b>RL = generate → score → learn → repeat.</b> "
    "The model generates a response, gets a reward score, "
    "updates its parameters to make high-scoring responses more likely, "
    "and repeats.",
    title="The RL Loop"
)"""))

    # ── 6: Our Toy Model: TinyLM (markdown) ──
    cells.append(md_cell("""\
## 4. Our Toy Model: TinyLM

To make RL concepts concrete, we'll use a *tiny* language model throughout this notebook.

- **Vocabulary:** 5 words — `I`, `love`, `cats`, `hate`, `dogs`
- **Sequence length:** 3 tokens (always starts with `I`)
- **Total possible sequences:** 5 × 5 = **25** (since the first token is fixed to `I`, only positions 2 and 3 vary — wait, let's be precise: 1 × 5 × 5 = 25 sequences starting with `I`, but the model architecture allows any first token, so there are 5 × 5 × 5 = **125** total possible sequences)

> **Why so small?** With only 125 possible sequences, we can visualize *exactly* what the model learns. Every concept here scales directly to GPT-scale models — only the numbers change."""))

    # ── 7: TinyLM Definition (code) ──
    cells.append(code_cell("""\
# ── Our tiny language model ──
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
print("Sample generations from untrained TinyLM:\\n")
for seq in seqs:
    print(f"  {decode(seq):20s}  reward = {reward_fn(seq.unsqueeze(0)).item():+.1f}")"""))

    # ── 8: Visualizing the Policy (markdown + code) ──
    cells.append(md_cell("""\
### What Does a "Policy" Look Like?

We said the **policy** is the model's probability distribution over tokens. Let's see that concretely. Given the starting token `I`, what does the model think should come next?"""))

    cells.append(code_cell("""\
# Show the model's probability distribution for the next token
model = TinyLM()
start = torch.tensor([[0]])  # Token "I" has index 0
logits = model(start)[0, 0, :]
probs = torch.softmax(logits, dim=0)

fig, ax = plt.subplots(figsize=(8, 3.5))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
bars = ax.bar(VOCAB, probs.detach().numpy(), color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylabel("Probability", fontsize=12)
ax.set_title('Policy: P(next token | "I") — Before Training', fontsize=13, fontweight='bold')
ax.set_ylim(0, 0.5)
for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{p:.1%}', ha='center', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

info_box(
    "This is what a <b>policy</b> looks like: a probability distribution over the next token. "
    "Before training, it's roughly uniform — the model has no preference. "
    "RL will reshape this distribution so that good tokens become more likely.",
    title="Policy = Probability Distribution"
)"""))

    # ── 9: Reward Function Design (markdown) ──
    cells.append(md_cell("""\
## 5. Reward Function Design

The reward function is the **heart of RL** — it defines *what* the model should learn.

Our toy reward function:
- **+1.0** for sequences containing both "love" and "cats" (e.g., "I love cats")
- **+0.3** for sequences containing "love" (but not "cats")
- **-0.5** penalty for any sequence containing "hate"
- **0.0** otherwise

In real RLHF, this hand-coded function is replaced by a **reward model** — a neural network trained on human preference data. But the principle is the same: the reward signal tells the policy what's good and what's bad."""))

    cells.append(code_cell("""\
info_box(
    "The reward function defines <b>WHAT</b> the model should learn. "
    "Get it wrong, and the model optimizes the wrong thing — "
    "a phenomenon called <b>reward hacking</b>. For example, a reward model that "
    "favors longer responses will produce a model that rambles.",
    title="Reward Design Matters"
)"""))

    # ── 10: Reward Landscape Visualization ──
    cells.append(md_cell("""\
### The Reward Landscape

Let's map out the *entire* space of possible sequences and their rewards. With only 125 sequences, we can see everything."""))

    cells.append(code_cell("""\
# Generate ALL possible sequences and their rewards
all_seqs = list(itertools.product(range(V), repeat=SEQ_LEN))
all_rewards = []
for seq in all_seqs:
    t = torch.tensor([seq])
    all_rewards.append(reward_fn(t).item())

# Show distribution of rewards
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: histogram of rewards
ax1.hist(all_rewards, bins=20, color='#4C72B0', edgecolor='white', alpha=0.8)
ax1.set_xlabel("Reward", fontsize=11)
ax1.set_ylabel("Count", fontsize=11)
ax1.set_title("Reward Distribution (All 125 Sequences)", fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Right: top and bottom sequences
sorted_seqs = sorted(zip(all_rewards, all_seqs), reverse=True)
top5 = sorted_seqs[:5]
bot5 = sorted_seqs[-5:]

y_labels = [" ".join(VOCAB[t] for t in s) for _, s in top5 + bot5]
x_vals = [r for r, _ in top5 + bot5]
colors_bar = ['#55A868'] * 5 + ['#C44E52'] * 5

ax2.barh(range(10), x_vals, color=colors_bar, edgecolor='white')
ax2.set_yticks(range(10))
ax2.set_yticklabels(y_labels, fontsize=10)
ax2.set_xlabel("Reward", fontsize=11)
ax2.set_title("Best & Worst Sequences", fontsize=12, fontweight='bold')
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax2.grid(axis='x', alpha=0.3)
ax2.invert_yaxis()

plt.tight_layout()
plt.show()"""))

    # ── 11: Policy Gradient: The Core Idea ──
    cells.append(md_cell("""\
## 6. Policy Gradient: The Core Idea

Here's the fundamental insight of RL for language models:

> **Increase the probability of sequences that got high rewards.
> Decrease the probability of sequences that got low rewards.**

That's it. Everything else — PPO, DPO, GRPO — is refinements of this single idea.

In math, the **REINFORCE** algorithm (Williams, 1992) says:

$$\\nabla J(\\theta) = \\mathbb{E}\\left[ R \\cdot \\nabla \\log \\pi_\\theta(\\text{sequence}) \\right]$$

Let's break this down term by term:"""))

    # ── 12: REINFORCE Formula Breakdown ──
    cells.append(code_cell("""\
formula_breakdown([
    (
        "\\u03c0\\u2098(sequence) — the probability the model<br>"
        "assigns to this complete sequence",
        "\\u03c0<sub>\\u03b8</sub>(a\\u2081, a\\u2082, \\u2026 | s)",
        "probs = model(sequence).softmax(-1)",
    ),
    (
        "log \\u03c0\\u2098 — take the log (turns products<br>"
        "of probabilities into sums; numerically stable)",
        "log \\u03c0<sub>\\u03b8</sub> = \\u03a3 log P(a\\u209c | a\\u2081..\\u209c\\u208b\\u2081)",
        "log_probs = model.log_probs_of(sequence)",
    ),
    (
        "R — the reward score for this sequence<br>"
        "(how good was the complete response?)",
        "R(sequence)",
        "reward = reward_fn(sequence)",
    ),
    (
        "R \\u00b7 \\u2207log \\u03c0\\u2098 — push the model toward<br>"
        "high-reward sequences (positive R amplifies<br>"
        "the gradient; negative R reverses it)",
        "R \\u00b7 \\u2207 log \\u03c0<sub>\\u03b8</sub>",
        "loss = -(reward * log_probs).mean()",
    ),
], title="REINFORCE — Term by Term")"""))

    # ── 13: Why log probabilities? ──
    cells.append(md_cell("""\
### Why Log Probabilities?

You might wonder: why use *log* probabilities instead of raw probabilities?

Two practical reasons:

1. **Numerical stability.** The probability of a sequence is the *product* of per-token probabilities. For a 100-token sequence: $P = p_1 \\times p_2 \\times \\dots \\times p_{100}$. Even if each $p_i = 0.5$, the product is $2^{-100} \\approx 10^{-30}$ — too small for floating-point. Taking logs turns products into sums: $\\log P = \\log p_1 + \\log p_2 + \\dots + \\log p_{100}$, which is perfectly fine numerically.

2. **Clean gradient.** The gradient of $\\log \\pi_\\theta$ with respect to $\\theta$ has a particularly clean form (known as the "score function" in statistics). This is what makes REINFORCE work — the $\\nabla \\log \\pi$ term naturally tells us which direction to push the parameters."""))

    cells.append(code_cell("""\
info_box(
    "<b>The Score Function Trick:</b> "
    "The identity \\u2207 log f(x) = \\u2207f(x) / f(x) means we can compute "
    "policy gradients by sampling — no need to enumerate all possible sequences. "
    "This is what makes RL scalable to models with trillions of possible outputs.",
    title="Why This Works at Scale"
)"""))

    # ── 14: REINFORCE Training ──
    cells.append(md_cell("""\
## 7. REINFORCE Training

Let's train our TinyLM with REINFORCE. The algorithm is surprisingly simple:

1. **Generate** a batch of sequences from the current policy
2. **Score** each sequence with the reward function
3. **Compute** the policy gradient: $\\text{loss} = -(R \\cdot \\log \\pi_\\theta)$
4. **Update** the model parameters with gradient descent
5. **Repeat**"""))

    cells.append(code_cell("""\
# ── Train with REINFORCE ──
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

print(f"\\nFinal avg reward: {reward_history[-1]:.3f}")"""))

    # ── 15: Training Curve ──
    cells.append(code_cell("""\
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(reward_history, color="#4C72B0", lw=1.5, alpha=0.6, label="Raw")
# Smoothed curve
window = 20
smoothed = np.convolve(reward_history, np.ones(window)/window, mode="valid")
ax.plot(range(window-1, len(reward_history)), smoothed, color="#4C72B0", lw=2.5, label="Smoothed")
ax.set_xlabel("Training Step", fontsize=11)
ax.set_ylabel("Average Reward", fontsize=11)
ax.set_title("REINFORCE Training — Reward Over Time", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

info_box(
    "REINFORCE works! The model learns to generate higher-reward sequences. "
    "But look at the <b>variance</b> (noise) in the raw curve — the reward bounces "
    "around wildly from step to step. This high variance is a known problem with "
    "vanilla policy gradients.",
    title="Observation"
)"""))

    # ── 16: What Did the Model Learn? ──
    cells.append(code_cell("""\
# What did the model learn to generate?
print("Most common sequences after REINFORCE training:\\n")
seqs = policy.generate(500)
counts = Counter(decode(s) for s in seqs)
for text, count in counts.most_common(8):
    r = reward_fn(torch.tensor([[VOCAB.index(w) for w in text.split()]])).item()
    print(f"  {text:20s}  freq={count/500:.0%}  reward={r:+.1f}")"""))

    # ── 17: Visualizing Policy After Training ──
    cells.append(md_cell("""\
### Policy Before vs. After Training

Let's see how REINFORCE reshaped the model's probability distribution."""))

    cells.append(code_cell("""\
# What does the policy look like now?
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

start = torch.tensor([[0]])  # "I"
logits_after = policy(start)[0, 0, :]
probs_after = torch.softmax(logits_after, dim=0)

fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.bar(VOCAB, probs_after.detach().numpy(), color=colors, edgecolor='white', linewidth=1.5)
ax.set_ylabel("Probability", fontsize=12)
ax.set_title('Policy: P(next token | "I") — After REINFORCE', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.0)
for bar, p in zip(bars, probs_after):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{p:.1%}', ha='center', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

info_box(
    "After training, the model strongly prefers 'love' — exactly the token "
    "that leads to high rewards. Compare this to the roughly uniform distribution "
    "before training. The policy has been <b>shaped by the reward signal</b>.",
    title="Policy After Training"
)"""))

    # ── 18: The Variance Problem (markdown) ──
    cells.append(md_cell("""\
## 8. The Variance Problem

You saw the noisy training curve above. Let's make this problem more visible by running REINFORCE **5 times** with different random seeds."""))

    cells.append(code_cell("""\
fig, ax = plt.subplots(figsize=(10, 4))
for seed in [42, 123, 456, 789, 1024]:
    torch.manual_seed(seed)
    p = TinyLM()
    opt = torch.optim.Adam(p.parameters(), lr=5e-3)
    hist = []
    for step in range(200):
        seqs = p.generate(batch_size=64)
        rewards = reward_fn(seqs)
        baseline = rewards.mean()
        advantages = rewards - baseline
        log_probs = p.log_probs_of(seqs)
        loss = -(log_probs * advantages).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(rewards.mean().item())
    ax.plot(hist, alpha=0.6, lw=1.5, label=f'seed={seed}')

ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Avg Reward", fontsize=11)
ax.set_title("REINFORCE: 5 Runs with Different Seeds", fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

info_box(
    "Each run converges, but the <b>paths are wildly different</b>. Some converge fast, "
    "others are slow and noisy. This high variance is REINFORCE's Achilles heel. "
    "Next notebook: PPO fixes this with clipping and multiple update steps.",
    title="The Variance Problem"
)"""))

    # ── 19: The Baseline Trick (markdown) ──
    cells.append(md_cell("""\
## 9. The Baseline Trick

We've been subtracting the mean reward (the "baseline") in our training loop. Why?

**Intuition:** Imagine all your sequences get rewards between 0.5 and 1.0. Without a baseline, *every* sequence gets pushed up (all rewards are positive). The model learns slowly because there's no contrast.

With a baseline, we ask: "was this sequence **better or worse than average?**"
- Above average → positive advantage → increase probability
- Below average → negative advantage → decrease probability

It's like **grading on a curve** — what matters is relative performance, not absolute scores.

**Mathematically:** subtracting a constant baseline doesn't change the expected gradient (it's a zero-mean correction), but it **dramatically reduces variance**."""))

    # ── 20: Baseline Comparison ──
    cells.append(code_cell("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

for ax, use_baseline, title in [(ax1, False, "Without Baseline"), (ax2, True, "With Baseline")]:
    torch.manual_seed(42)
    p = TinyLM()
    opt = torch.optim.Adam(p.parameters(), lr=5e-3)
    hist = []
    for step in range(300):
        seqs = p.generate(batch_size=64)
        rewards = reward_fn(seqs)
        if use_baseline:
            advantages = rewards - rewards.mean()
        else:
            advantages = rewards
        log_probs = p.log_probs_of(seqs)
        loss = -(log_probs * advantages).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(rewards.mean().item())

    ax.plot(hist, color='#4C72B0', alpha=0.5, lw=1)
    window = 20
    smoothed = np.convolve(hist, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(hist)), smoothed, color='#4C72B0', lw=2.5)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Step")
    ax.set_ylabel("Avg Reward")
    ax.set_ylim(-0.2, 1.1)
    ax.grid(alpha=0.3)

plt.suptitle("Effect of Baseline on REINFORCE Training", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()"""))

    # ── 21: Summary ──
    cells.append(md_cell("""\
## Summary

### Key Takeaways

1. **RL = generate, score, learn, repeat.** The model generates responses, gets reward scores, and updates to make high-scoring responses more likely.

2. **Policy = probability distribution over tokens.** Before training it's roughly uniform; after training it's shaped by the reward signal.

3. **REINFORCE** is the simplest policy gradient: $\\text{loss} = -(R \\cdot \\log \\pi_\\theta)$. Increase P(high-reward sequences), decrease P(low-reward ones).

4. **The baseline trick** (subtracting mean reward) reduces variance without changing the expected gradient. It's like grading on a curve.

5. **Variance remains a problem.** Even with a baseline, REINFORCE training is noisy and sensitive to random seeds."""))

    # ── 22: What's Next ──
    cells.append(code_cell("""\
info_box(
    "REINFORCE works but is fragile — high variance, no safety net against "
    "destructive updates. In the <b>next notebook</b>, we build PPO step by step: "
    "adding a value model, clipping the policy ratio, and KL penalties to make "
    "training stable and reliable.",
    title="What's Next: PPO (Notebook 01)"
)"""))

    return cells


def build_notebook(cells):
    # Fix source format: ensure each cell's source is a list of lines with \n
    for cell in cells:
        if isinstance(cell["source"], list):
            cell["source"] = _join_lines(cell["source"])
        elif isinstance(cell["source"], str):
            cell["source"] = _join_lines(cell["source"].split("\n"))

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
    return notebook


def main():
    cells = build_cells()
    notebook = build_notebook(cells)

    output_path = Path(__file__).resolve().parent.parent / "notebooks" / "en" / "reinforcement-learning" / "00-rl-foundations.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Created {output_path}")
    print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type'] == 'code')} code, "
          f"{sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")


if __name__ == "__main__":
    main()
