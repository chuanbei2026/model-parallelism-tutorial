"""Generate notebook 04-frontiers-and-systems.ipynb for the Reinforcement Learning for LLMs chapter."""
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


cells = []

# ── 0: Header ─────────────────────────────────────────────────────────────
cells.append(md_cell(r"""---
**Title:** Frontiers & Distributed Systems

**Chapter:** Reinforcement Learning for LLMs

**Difficulty:** Intermediate–Advanced

**Estimated Time:** 40 min

---"""))

# ── 1: Overview ───────────────────────────────────────────────────────────
cells.append(md_cell(r"""## 1 — Overview

Over the previous four notebooks we built up the core algorithmic toolkit for RL-based LLM alignment:

| Notebook | Algorithm | Key Idea |
|---|---|---|
| 00 | REINFORCE | Policy gradient — the simplest RL algorithm |
| 01 | PPO | Clipped objective + KL penalty for stable updates |
| 02 | DPO | Skip the reward model — optimise preferences directly |
| 03 | GRPO | Skip the critic — group-relative advantages |

This final notebook **zooms out**. We cover two broad topics:

1. **Algorithm evolution & cutting-edge methods** — how did we get here, and what comes next?
2. **Distributed systems for RL training** — the engineering challenges that make large-scale RL alignment hard.

By the end you will have a complete mental map of the field: algorithms, trade-offs, system designs, and open problems."""))

# ── 2: Imports ────────────────────────────────────────────────────────────
cells.append(code_cell(r"""import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mp_tutorial.fonts import configure_cjk_fonts
configure_cjk_fonts()

from mp_tutorial.viz import draw_method_timeline, draw_rl_algorithm_comparison, draw_rl_gpu_placement
from mp_tutorial.formatting import info_box, comparison_table, formula_breakdown, code_reference

import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from font", category=UserWarning)

torch.manual_seed(42)
print("Ready!")"""))

# ── 3: The Full Algorithm Comparison (markdown) ──────────────────────────
cells.append(md_cell(r"""## 2 — The Full Algorithm Comparison

We have now seen four RL algorithms for LLM alignment. Before diving into frontiers, let us put them side by side and compare their trade-offs in one place."""))

# ── 4: Algorithm Comparison Table ─────────────────────────────────────────
cells.append(code_cell(r"""comparison_table(
    headers=["Algorithm", "Models in Memory", "Needs Reward Model?",
             "Training Data", "Key Advantage", "Key Limitation"],
    rows=[
        ["REINFORCE", "1 (policy)", "No (uses reward fn)",
         "On-policy", "Simplest possible", "High variance, unstable"],
        ["PPO (RLHF)", "4 (actor + critic + reward + ref)",
         "Yes", "On-policy", "Most flexible, well-studied",
         "Complex, memory-hungry"],
        ["DPO", "2 (policy + ref)",
         "No", "Offline preference pairs", "Simple, stable",
         "No online exploration"],
        ["GRPO", "2 (policy + ref)",
         "No (uses verifier)", "On-policy groups", "Simple, on-policy",
         "Needs good reward/verifier"],
    ],
    title="Algorithm Comparison — Complete Picture"
)"""))

# ── 5: Memory Comparison ─────────────────────────────────────────────────
cells.append(code_cell(r"""params_7b = 7e9
bytes_per_param = 2  # fp16
model_gb = params_7b * bytes_per_param / 1e9
print(f"One 7B model in fp16: {model_gb:.1f} GB\n")

configs = [
    ("REINFORCE", 1, "Policy only"),
    ("PPO (RLHF)", 4, "Actor + Critic + Reward + Reference"),
    ("DPO", 2, "Policy + Reference"),
    ("GRPO", 2, "Policy + Reference"),
]
fig, ax = plt.subplots(figsize=(8, 4))
names = [c[0] for c in configs]
mem = [c[1] * model_gb for c in configs]
colors = ['#C44E52', '#DD8452', '#4C72B0', '#55A868']
bars = ax.barh(names, mem, color=colors, edgecolor='white', height=0.6)
for bar, m, (_, n, desc) in zip(bars, mem, configs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{m:.0f} GB ({desc})', va='center', fontsize=9)
ax.set_xlabel("GPU Memory (GB) — 7B Model in fp16", fontsize=11)
ax.set_title("Memory Cost by Algorithm", fontsize=13, fontweight='bold')
ax.set_xlim(0, 70)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

info_box(f"PPO needs ~{4*model_gb:.0f} GB just for model weights — before optimizer states "
         f"and activations! DPO/GRPO halve this to ~{2*model_gb:.0f} GB.",
         title="Memory Reality Check")"""))

# ── 6: Evolution Timeline (markdown) ─────────────────────────────────────
cells.append(md_cell(r"""## 3 — The Evolution of RL for LLMs

The story of RL for LLMs is one of progressive simplification. Each new algorithm solved a specific pain point of its predecessor:

| Year | Milestone | What It Solved |
|---|---|---|
| 2017 | **PPO** (Schulman et al.) | Stable policy optimisation via clipping |
| 2022 | **RLHF at scale** (InstructGPT / ChatGPT) | Showed PPO + human feedback works for chat alignment |
| 2023 | **DPO** (Rafailov et al.) | Removed the reward model entirely |
| 2024 | **SAPO, KTO, IPO** | Self-alignment, simpler losses, no preference pairs |
| 2025 | **GRPO / DeepSeek-R1** | Removed the critic, enabled emergent reasoning |

The trend is clear: **fewer models, simpler losses, more autonomous improvement.**"""))

# ── 7: Timeline Visualization ─────────────────────────────────────────────
cells.append(code_cell(r"""fig, ax = draw_method_timeline()
plt.show()"""))

# ── 8: SAPO: Self-Aligned Preference Optimization (markdown) ─────────────
cells.append(md_cell(r"""## 4 — SAPO: Self-Aligned Preference Optimization

The algorithms we have covered so far all require some form of external signal: a reward function, human preference annotations, or a verifier. **SAPO** (Self-Aligned Preference Optimization) pushes the frontier further: **what if the model can improve itself?**

The key idea:

1. **Generate** two (or more) responses to the same prompt using the current policy.
2. **Score** the responses using a quality signal — this could be length, coherence, self-consistency (does the model agree with its own answer when asked again?), or a lightweight verifier.
3. **Create a preference pair**: the higher-scoring response becomes the "preferred" output, the lower-scoring one becomes "rejected".
4. **Train with a DPO-style loss** on these self-generated preference pairs.
5. **Repeat** — the improved model generates better responses next round.

This creates a **self-improvement loop**: the model bootstraps its own training data."""))

cells.append(code_cell(r"""info_box("SAPO is exciting because it removes the last human bottleneck — preference annotation. "
         "The model generates its own training signal and improves itself iteratively. "
         "This is a step toward truly autonomous alignment.",
         title="Why SAPO Matters")"""))

# ── 9: Self-Improvement Loop Visualization ────────────────────────────────
cells.append(code_cell(r"""fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# Draw circular flow
angles = [90, 0, -90, -180]
labels = ['Generate\n2 responses', 'Score with\nquality signal',
          'Create preference\npair (better > worse)', 'Train with\nDPO loss']
colors_c = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']

for i, (angle, label, color) in enumerate(zip(angles, labels, colors_c)):
    rad = np.radians(angle)
    x, y = np.cos(rad), np.sin(rad)
    circle = mpatches.FancyBboxPatch((x-0.4, y-0.25), 0.8, 0.5,
                                      boxstyle="round,pad=0.05",
                                      facecolor=color, alpha=0.2, edgecolor=color, lw=2)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# Draw arrows between nodes
for i in range(4):
    a1 = np.radians(angles[i])
    a2 = np.radians(angles[(i+1) % 4])
    # Arrow from node i to node i+1
    x1, y1 = 0.7*np.cos(a1 - 0.3), 0.7*np.sin(a1 - 0.3)
    x2, y2 = 0.7*np.cos(a2 + 0.3), 0.7*np.sin(a2 + 0.3)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))

ax.set_title("SAPO: Self-Improvement Loop", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()"""))

# ── 10: The Evolution in One Line ─────────────────────────────────────────
cells.append(md_cell(r"""### The Evolution in One Line

> REINFORCE (noisy) → **PPO** (stable, 4 models) → **DPO** (simpler, 2 models, offline) → **GRPO** (simpler, 2 models, online) → **SAPO** (self-aligned)

Each step removes a component or a dependency. The field is converging toward algorithms that are **simpler, cheaper, and more autonomous.**"""))

# ── 11: Which LLMs Use What? ─────────────────────────────────────────────
cells.append(md_cell(r"""## 5 — Which LLMs Use What?

These algorithms are not academic curiosities — they power the models you use every day."""))

cells.append(code_cell(r"""comparison_table(
    headers=["Model", "Method", "Key Innovation"],
    rows=[
        ["ChatGPT / GPT-4", "PPO (RLHF)", "Pioneered RLHF at scale"],
        ["Claude", "RLHF + Constitutional AI", "AI feedback for safety"],
        ["Llama-2-Chat", "PPO (RLHF)", "Rejection sampling + PPO"],
        ["Zephyr", "DPO", "Showed DPO can match RLHF"],
        ["DeepSeek-R1", "GRPO", "Group-relative advantage, emergent reasoning"],
        ["Qwen-2", "DPO + GRPO", "Hybrid approach"],
    ],
    title="RL Methods in Production LLMs"
)"""))

# ── 12: Part 2: Distributed Challenges ───────────────────────────────────
cells.append(md_cell(r"""---

## Part 2 — Distributed Systems for RL Training

When training LLMs at scale, RL alignment introduces unique distributed-systems challenges **beyond standard pre-training**. Pre-training is "just" a massive forward-backward loop on text data. RL alignment adds generation, multiple models, and on-policy constraints.

Let us explore the three main challenges."""))

# ── 13: Challenge 1: Multi-Model Orchestration ───────────────────────────
cells.append(md_cell(r"""### Challenge 1: Multi-Model Orchestration

Pre-training involves **one model**. RL alignment involves **two to four**:

| Algorithm | Models | Notes |
|---|---|---|
| PPO | 4 | Actor, Critic, Reward, Reference |
| DPO | 2 | Policy, Reference |
| GRPO | 2 | Policy, Reference |

These models have **different compute patterns**:
- Some are **frozen** (reference, reward model) — inference only
- Some are **training** (actor, critic) — forward + backward + optimizer step
- They need to **communicate** results between steps (rewards, log-probs, advantages)

Scheduling and memory management become critical. You cannot simply "data-parallel" the whole thing — each model has different resource needs and lifetimes."""))

# ── 14: Challenge 2: Generation Bottleneck ────────────────────────────────
cells.append(md_cell(r"""### Challenge 2: The Generation Bottleneck

RL alignment requires the model to **generate text** (autoregressive decoding, token by token). This is fundamentally different from — and much slower than — a standard training step."""))

cells.append(code_cell(r"""# Illustrate the bottleneck
fig, ax = plt.subplots(figsize=(10, 3.5))
tasks = ['Pre-training\n(forward+backward)', 'RL Generation\n(autoregressive)',
         'RL Training\n(PPO update)']
# Relative time estimates
times = [1.0, 3.5, 0.8]
colors_t = ['#4C72B0', '#C44E52', '#55A868']
bars = ax.barh(tasks, times, color=colors_t, edgecolor='white', height=0.5)
for bar, t in zip(bars, times):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f'{t:.1f}x', va='center', fontsize=11, fontweight='bold')
ax.set_xlabel("Relative Wall-Clock Time", fontsize=11)
ax.set_title("The Generation Bottleneck", fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.show()

info_box("Autoregressive generation is 3-4x slower than a training step. "
         "Optimizing generation throughput (batching, speculative decoding, vLLM) "
         "is critical for RL training speed.",
         title="The Generation Bottleneck")"""))

# ── 15: Challenge 3: On-Policy Data Freshness ────────────────────────────
cells.append(md_cell(r"""### Challenge 3: On-Policy Data Freshness

PPO and GRPO are **on-policy** algorithms: their training data must come from the **current** model. You cannot pre-generate a big dataset and iterate over it (that is DPO's approach).

This creates a **generate → train → generate → train** loop:

1. **Generate** a batch of responses with the current policy
2. **Score** them (reward model or verifier)
3. **Train** the policy on these responses
4. **Discard** the data — it is now off-policy
5. **Repeat** from step 1

The problem: **pipeline bubbles**. During generation, the training GPUs sit idle. During training, the generation GPUs sit idle. This wastes expensive compute.

> **The holy grail:** overlap generation and training to keep all GPUs busy at all times."""))

# ── 16: GPU Placement Strategies (markdown) ───────────────────────────────
cells.append(md_cell(r"""## 6 — GPU Placement Strategies

How do you fit 2–4 large models onto GPUs? Two main approaches:

**Colocated:** All models share the same GPUs. Simple to implement but severely memory-constrained — you need enough memory for all models simultaneously.

**Separated:** Dedicated GPU groups for each model. More total memory available, but orchestration is complex — models must send data to each other across groups."""))

# ── 17: GPU Placement Visualization ───────────────────────────────────────
cells.append(code_cell(r"""fig1, ax1 = draw_rl_gpu_placement(strategy="colocated")
plt.show()"""))

cells.append(code_cell(r"""fig2, ax2 = draw_rl_gpu_placement(strategy="separated")
plt.show()"""))

# ── 18: Framework Comparison ──────────────────────────────────────────────
cells.append(md_cell(r"""## 7 — RL Training Frameworks

Several frameworks have emerged to tackle these distributed challenges. Each makes different trade-offs between simplicity and performance."""))

cells.append(code_cell(r"""comparison_table(
    headers=["Framework", "Strategy", "Key Idea", "Best For"],
    rows=[
        ["DeepSpeed-Chat", "Colocated → Separated", "All models on same GPUs, then offload", "Getting started"],
        ["OpenRLHF", "Separated (Ray)", "Dedicated GPU groups, Ray orchestration", "Large-scale PPO"],
        ["veRL (Volcano Engine)", "Hybrid SPMD", "Flexible placement, overlaps gen + train", "Maximum throughput"],
        ["TRL", "Colocated (HF)", "HuggingFace ecosystem integration", "Prototyping"],
    ],
    title="RL Training Frameworks"
)"""))

# ── 19: OpenRLHF Code Reference ──────────────────────────────────────────
cells.append(code_cell(
    'code_reference(\n'
    '    code="""\\\n'
    '# OpenRLHF: Actor-Critic separated across GPU groups\n'
    'class PPOTrainer:\n'
    '    def __init__(self, actor, critic, reward_model, ref_model):\n'
    '        # Each model can be on different GPU groups\n'
    '        self.actor = ActorModelRayActor(actor, gpu_group_0)\n'
    '        self.critic = CriticModelRayActor(critic, gpu_group_1)\n'
    '        self.reward = RewardModelRayActor(reward_model, gpu_group_2)\n'
    '        self.ref = RefModelRayActor(ref_model, gpu_group_0)  # shares with actor\n'
    '\n'
    '    def step(self, prompts):\n'
    '        # 1. Generate with actor (SLOW — generation bottleneck)\n'
    '        sequences = self.actor.generate(prompts)\n'
    '        # 2. Score with reward model\n'
    '        rewards = self.reward.score(sequences)\n'
    '        # 3. Compute values with critic\n'
    '        values = self.critic.evaluate(sequences)\n'
    '        # 4. PPO update (FAST — standard training step)\n'
    '        self.actor.ppo_step(sequences, rewards, values)\"\"\",\n'
    '    source="OpenRLHF",\n'
    '    filepath="openrlhf/trainer/ppo_trainer.py"\n'
    ')'
))

# ── 20: Overlap Strategy Visualization ────────────────────────────────────
cells.append(code_cell(r"""fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

# Sequential timeline
tasks_seq = [
    ('Generate', 0, 4, '#C44E52'),
    ('Score', 4, 1, '#DD8452'),
    ('Train', 5, 2, '#4C72B0'),
    ('Generate', 7, 4, '#C44E52'),
    ('Score', 11, 1, '#DD8452'),
    ('Train', 12, 2, '#4C72B0'),
]
for name, start, dur, color in tasks_seq:
    ax1.barh(0, dur, left=start, height=0.5, color=color, edgecolor='white', alpha=0.8)
    ax1.text(start + dur/2, 0, name, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
ax1.set_title("Sequential (DeepSpeed-Chat)", fontweight='bold')
ax1.set_yticks([])
ax1.set_xlim(0, 15)
ax1.grid(axis='x', alpha=0.3)

# Overlapped timeline
tasks_over = [
    ('Generate\u2081', 0, 4, '#C44E52', 0),
    ('Score\u2081', 4, 1, '#DD8452', 0),
    ('Train\u2081', 5, 2, '#4C72B0', 0),
    ('Generate\u2082', 5, 4, '#C44E52', 1),  # Overlap!
    ('Score\u2082', 9, 1, '#DD8452', 1),
    ('Train\u2082', 7, 2, '#4C72B0', 1),
]
for name, start, dur, color, row in tasks_over:
    ax2.barh(row, dur, left=start, height=0.4, color=color, edgecolor='white', alpha=0.8)
    ax2.text(start + dur/2, row, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')
ax2.set_title("Overlapped (veRL)", fontweight='bold')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["GPU Group A", "GPU Group B"], fontsize=9)
ax2.set_xlabel("Time \u2192", fontsize=11)
ax2.set_xlim(0, 15)
ax2.grid(axis='x', alpha=0.3)

plt.suptitle("Sequential vs Overlapped RL Training", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

info_box("By overlapping generation and training across GPU groups, veRL achieves "
         "~1.5x throughput over sequential approaches. The key: start generating "
         "the next batch while training on the current one.",
         title="Overlap = Throughput")"""))

# ── 21: The Full Picture ─────────────────────────────────────────────────
cells.append(md_cell(r"""## 8 — The Full Picture

Let us bring everything together.

**RL alignment is the final stage of LLM training**, after pre-training and supervised fine-tuning. It turns a capable but uncontrolled model into one that follows instructions, avoids harm, and reasons well.

**Multiple algorithms exist**, each with different trade-offs between simplicity, memory cost, exploration ability, and data requirements. There is no single "best" algorithm — the choice depends on your constraints.

**The field evolves rapidly.** New algorithms appear every few months. The trend is toward simpler methods that need fewer models and less human annotation.

**Engineering challenges are just as important as algorithmic advances.** You can have the best algorithm in the world, but if you cannot run it efficiently across hundreds of GPUs, it is not practical. Multi-model orchestration, generation bottlenecks, and on-policy freshness are real constraints that shape what is deployable."""))

# ── 22: Summary ───────────────────────────────────────────────────────────
cells.append(md_cell(r"""## Summary — The Complete Chapter

This chapter covered the full landscape of RL for LLM alignment, from first principles to production systems.

### Notebook-by-Notebook Recap

1. **Foundations (nb 00):** RL = generate → score → learn → repeat. REINFORCE is the simplest algorithm — a single policy, high variance, but the foundation for everything else.

2. **PPO (nb 01):** Adds stability via clipped surrogate objective, safety via KL penalty to a reference model, and efficiency via multi-epoch updates. Requires 4 models in memory.

3. **DPO (nb 02):** Eliminates the reward model via a mathematical shortcut (Bradley-Terry → closed-form optimal policy). Only 2 models, offline training, extremely simple loss.

4. **GRPO (nb 03):** Eliminates the critic via group-relative advantage estimation. Only 2 models, online training, no value function. Powers DeepSeek-R1's emergent reasoning.

5. **Frontiers (nb 04, this notebook):** SAPO for self-alignment — the model generates its own preference data. Distributed challenges: multi-model orchestration, generation bottleneck, on-policy data freshness. Framework landscape: DeepSpeed-Chat, OpenRLHF, veRL, TRL."""))

# ── 23: The Evolution in One Line ─────────────────────────────────────────
cells.append(md_cell(r"""> REINFORCE (noisy) → **PPO** (stable, 4 models) → **DPO** (simpler, 2 models, offline) → **GRPO** (simpler, 2 models, online) → **SAPO** (self-aligned)"""))

# ── 24: Further Reading ───────────────────────────────────────────────────
cells.append(md_cell(r"""## Further Reading

### Papers

- **PPO** — Schulman et al., 2017. [arXiv 1707.06347](https://arxiv.org/abs/1707.06347)
- **RLHF (InstructGPT)** — Ouyang et al., 2022. [arXiv 2203.02155](https://arxiv.org/abs/2203.02155)
- **DPO** — Rafailov et al., 2023. [arXiv 2305.18290](https://arxiv.org/abs/2305.18290)
- **GRPO / DeepSeek-R1** — 2025. [arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
- **SAPO** — 2024. [arXiv 2405.07863](https://arxiv.org/abs/2405.07863)

### Frameworks

- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) — Microsoft's end-to-end RLHF pipeline
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Scalable RLHF with Ray-based model separation
- [veRL](https://github.com/volcengine/verl) — Volcano Engine's hybrid SPMD framework for RL training
- [TRL](https://github.com/huggingface/trl) — HuggingFace's Transformer Reinforcement Learning library"""))


# ── Assemble notebook ─────────────────────────────────────────────────────
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

out_path = Path(__file__).resolve().parent.parent / "notebooks" / "en" / "reinforcement-learning" / "04-frontiers-and-systems.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(f"Created {out_path}  ({len(cells)} cells)")
