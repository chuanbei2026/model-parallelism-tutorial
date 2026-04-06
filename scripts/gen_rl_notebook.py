"""Generate the skeleton for 08-reinforcement-learning.ipynb."""
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
**Title:** Reinforcement Learning for LLMs

**Category:** reinforcement-learning

**Difficulty:** Intermediate–Advanced

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
    md("""## Overview

### The Three Stages of LLM Training

Building a useful LLM takes three stages. You already know the first two from earlier notebooks:

1. **Pre-training** — predict the next token on massive text corpora (learns language)
2. **Supervised Fine-Tuning (SFT)** — train on human-written demonstrations (learns to follow instructions)
3. **RL Alignment** — optimize for human *preferences* (learns what humans actually want)

This notebook covers **stage 3**: how reinforcement learning turns a capable-but-unaligned model into one that produces responses humans prefer.

### Why SFT Is Not Enough

SFT teaches the model to *imitate* demonstrations, but:
- You can't write a demonstration for every possible question
- Imitation doesn't distinguish "okay" from "great" responses
- The model has no signal for *which parts* of a response matter

RL solves this by letting the model **try things and learn from feedback** — "this response was better than that one."

### Prerequisites

- PyTorch basics (`nn.Module`, optimizers, loss functions)
- Recommended: [01 — Data Parallelism](01-data-parallelism.ipynb)"""),

    # ── 3: Training pipeline diagram ──
    code("""fig, ax = draw_training_pipeline()
plt.show()"""),

    # ── 4: Foundations intro ──
    md("""## Foundations

Before diving into specific algorithms, let's build three essential concepts:
1. **Policy** — what the model is doing (its behavior)
2. **Reward** — how we score what it did
3. **Policy gradient** — how to improve the policy using rewards

### Our Toy Model: A 5-Word Language Model

To make RL concepts concrete, we'll use a tiny language model throughout this notebook. It has a vocabulary of just 5 words and generates 3-token sequences.

> **Why so small?** With only 5 words and length 3, there are just 125 possible sequences. We can see *exactly* what the model learns. The concepts scale directly to GPT-scale models — only the size changes."""),

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
    md("""### Policy Gradient: The Core Idea

Here's the fundamental insight of RL for language models:

> **Increase the probability of sequences that got high rewards.
> Decrease the probability of sequences that got low rewards.**

That's it. Everything else (PPO, DPO, GRPO) is refinements of this idea.

In math, the **REINFORCE** algorithm says:

$$\\nabla J(\\theta) = \\mathbb{E}\\left[ R \\cdot \\nabla \\log \\pi_\\theta(\\text{sequence}) \\right]$$

Let's break this down term by term:"""),

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
    md("""## PPO: Proximal Policy Optimization

REINFORCE has two problems:
1. **High variance** — the training signal is noisy (you saw the jagged curve)
2. **No safety net** — a single bad update can destroy what the model learned

PPO solves both. Let's build it up step by step, adding one component at a time.

### Building Up to 4 Models

Real RLHF (PPO for LLMs) needs **four models in memory simultaneously**. That sounds scary, so let's understand *why* by adding them one at a time — each solves a specific problem."""),

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
    md("""### The PPO Clipping Trick

PPO's key insight: **don't let the policy change too much in one step**.

It does this by computing a *probability ratio*:

$$r(\\theta) = \\frac{\\pi_\\text{new}(\\text{action})}{\\pi_\\text{old}(\\text{action})}$$

- If $r = 1$: the new policy behaves identically to the old one
- If $r = 2$: the new policy is 2x more likely to take this action
- If $r = 0.5$: the new policy is half as likely

PPO **clips** this ratio to $[1-\\varepsilon, 1+\\varepsilon]$ (typically $\\varepsilon = 0.2$), preventing drastic changes:

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
    md("""## DPO: Direct Preference Optimization

PPO works, but it's complicated: 4 models, multiple training phases, reward model training, hyperparameter-sensitive.

**DPO's insight** (Rafailov et al., 2023): we can skip the reward model entirely.

### The Mathematical Shortcut

The RL objective with a KL penalty has a closed-form optimal policy:

$$\\pi^*(a|s) = \\frac{1}{Z(s)} \\pi_{\\text{ref}}(a|s) \\cdot \\exp\\left(\\frac{1}{\\beta} r(a, s)\\right)$$

Rearranging, we can express the reward in terms of policies:

$$r(a, s) = \\beta \\log \\frac{\\pi^*(a|s)}{\\pi_{\\text{ref}}(a|s)} + \\text{const}$$

Substituting this into the Bradley-Terry preference model and simplifying:

$$L_{\\text{DPO}} = -\\log \\sigma\\left( \\beta \\left[ \\log \\frac{\\pi_\\theta(y_w)}{\\pi_{\\text{ref}}(y_w)} - \\log \\frac{\\pi_\\theta(y_l)}{\\pi_{\\text{ref}}(y_l)} \\right] \\right)$$

In plain English: **make the model more likely to produce preferred outputs ($y_w$) and less likely to produce rejected outputs ($y_l$), relative to the reference model.**"""),

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
    md("""## GRPO: Group Relative Policy Optimization

DPO simplified RLHF by removing the reward model. **GRPO** (used by DeepSeek-R1) simplifies it further by removing the **critic/value model** too.

### The Key Idea: Group Sampling

Instead of training a separate model to estimate "how good is this state?" (the critic), GRPO:

1. For each prompt, **generate a group of G outputs** from the current policy
2. **Score** each output with a reward function (or verifier)
3. Compute advantage as: **how much better/worse than the group average?**

$$A_i = \\frac{r_i - \\text{mean}(r_1, ..., r_G)}{\\text{std}(r_1, ..., r_G)}$$

This is called **group-relative advantage** — no critic model needed!

> **Why does this work?** If you generate enough outputs (G=8 or more), the group mean is a decent estimate of the expected reward. Outputs above the mean are "better than average" (positive advantage), below are "worse" (negative advantage). It's simple, but effective enough for DeepSeek-R1."""),

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
    md("""## Algorithm Comparison

Now that we've seen all three algorithms in action, let's compare:"""),

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
    md("""## Evolution & Frontier

### The Story So Far

Each algorithm solved a specific problem with the previous approach:"""),

    # ── 30: Timeline diagram ──
    code("""fig, ax = draw_method_timeline()
plt.show()"""),

    # ── 31: Frontier text ──
    md("""### SAPO: Self-Aligned Preference Optimization (2024)

The latest frontier goes further: **what if we don't need human preferences at all?**

SAPO (Self-Aligned Preference Optimization) generates preference pairs from the model itself:
1. Generate two responses to the same prompt
2. Use a simple quality signal (length, coherence, self-consistency) to rank them
3. Train with DPO-style loss on these self-generated pairs

This creates a **self-improvement loop** — the model bootstraps its own alignment.

### Which LLMs Use What?

| Model | Method | Key Innovation |
|-------|--------|---------------|
| ChatGPT / GPT-4 | PPO (RLHF) | Pioneered RLHF at scale |
| Llama-2-Chat | PPO (RLHF) | Rejection sampling + PPO |
| Zephyr | DPO | Showed DPO can match RLHF |
| DeepSeek-R1 | GRPO | Group-relative advantage, no critic |
| Qwen-2 | DPO + GRPO | Hybrid approach |"""),

    # ── 32: Distributed challenges ──
    md("""## Distributed Challenges

When training LLMs at scale, RL alignment introduces unique distributed-systems challenges beyond standard pre-training:

### 1. Multi-Model Orchestration
PPO needs 4 models (totaling ~4x the parameter count) in GPU memory simultaneously. Even DPO/GRPO need 2x. This is fundamentally different from pre-training, where there's only one model.

### 2. Generation Bottleneck
RL requires **generating text** from the model (autoregressive, token by token). This is much slower than the forward/backward passes of pre-training. It often dominates wall-clock time.

### 3. On-Policy Data Freshness
PPO and GRPO are **on-policy** — training data must come from the *current* model, not a stored dataset. You can't pre-generate data; you must generate → train → generate → train in a loop.

### GPU Placement Strategies

Frameworks solve these challenges with different placement strategies:"""),

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
    md("""## Summary & Further Reading

### Key Takeaways

1. **RL alignment** is the third stage of LLM training (after pre-training and SFT) — it optimizes for what humans *prefer*, not what they demonstrate
2. **Policy gradient** is the core idea: increase probability of high-reward outputs, decrease low-reward ones
3. **PPO** stabilizes training with clipped objectives and KL penalties, but needs 4 models in memory
4. **DPO** eliminates the reward model by optimizing preferences directly — from 4 models to 2
5. **GRPO** eliminates the critic by using group-relative advantages — simpler on-policy training (used by DeepSeek-R1)
6. **SAPO** pushes further: self-generated preferences for bootstrapped alignment
7. At scale, distributed RL faces unique challenges: multi-model memory, generation bottleneck, on-policy freshness

### The Evolution in One Line

> REINFORCE (noisy) → **PPO** (stable, 4 models) → **DPO** (simpler, 2 models, offline) → **GRPO** (simpler, 2 models, online) → **SAPO** (self-aligned)

### Further Reading

**Papers:**
- [PPO (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) — Proximal Policy Optimization
- [RLHF (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — Training language models to follow instructions with human feedback
- [DPO (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization
- [GRPO / DeepSeek-R1 (2025)](https://arxiv.org/abs/2501.12948) — DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL
- [SAPO (2024)](https://arxiv.org/abs/2405.07863) — Self-Aligned Preference Optimization

**Frameworks:**
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) — End-to-end RLHF pipeline
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Ray-based distributed RLHF
- [veRL](https://github.com/volcengine/verl) — Volcano Engine Reinforcement Learning for LLMs
- [TRL](https://github.com/huggingface/trl) — Hugging Face Transformer Reinforcement Learning"""),
]

nbformat.write(nb, "notebooks/en/08-reinforcement-learning.ipynb")
print(f"Created notebook with {len(nb.cells)} cells")
