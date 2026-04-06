"""Visualization helpers for inference optimization notebooks.

Provides reusable functions for KV-cache diagrams, batching timelines,
block table visualizations, quantization comparisons, and more.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import numpy as np


# Color palette consistent with the training notebooks
GPU_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

SEQ_COLORS = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373",
              "#BA68C8", "#4DB6AC", "#FF8A65", "#A1887F",
              "#90A4AE", "#F48FB1", "#80CBC4", "#FFCC80"]


def draw_kv_cache_growth(seq_lengths, n_layers, n_kv_heads, head_dim,
                         dtype_bytes=2, model_weight_gb=None,
                         title="KV-Cache Memory vs Sequence Length"):
    """Plot KV-cache memory as a function of sequence length.

    Args:
        seq_lengths: list of sequence lengths to plot
        n_layers, n_kv_heads, head_dim: model architecture parameters
        dtype_bytes: bytes per element (2 for fp16)
        model_weight_gb: optional — draw a horizontal line for model weights
        title: plot title
    """
    mem_gb = []
    for s in seq_lengths:
        m = 2 * n_layers * n_kv_heads * head_dim * s * dtype_bytes / (1024**3)
        mem_gb.append(m)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(seq_lengths, mem_gb, 'o-', color="#4C72B0", linewidth=2.5, markersize=6,
            label="KV-Cache (batch=1)")
    ax.fill_between(seq_lengths, mem_gb, alpha=0.1, color="#4C72B0")

    if model_weight_gb is not None:
        ax.axhline(model_weight_gb, color="#C44E52", linewidth=2, linestyle="--",
                   label=f"Model weights ({model_weight_gb:.1f} GB)")

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Memory (GB)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def draw_mha_vs_mqa_vs_gqa(n_q_heads=32, n_kv_heads_list=None, head_dim=128):
    """Visualize MHA vs MQA vs GQA head sharing patterns.

    Shows colored blocks: Q heads on top, K/V heads on bottom with
    lines showing which Q heads share which KV heads.
    """
    if n_kv_heads_list is None:
        n_kv_heads_list = [("MHA", n_q_heads), ("GQA-8", 8), ("GQA-4", 4), ("MQA", 1)]

    fig, axes = plt.subplots(1, len(n_kv_heads_list),
                              figsize=(4 * len(n_kv_heads_list), 4))
    if len(n_kv_heads_list) == 1:
        axes = [axes]

    q_colors = plt.cm.Set3(np.linspace(0, 1, n_q_heads))

    for ax, (label, n_kv) in zip(axes, n_kv_heads_list):
        group_size = n_q_heads // n_kv
        ax.set_xlim(-0.5, max(n_q_heads, n_kv) - 0.5)
        ax.set_ylim(-1.5, 2.5)
        ax.set_title(f"{label}\n({n_kv} KV heads)", fontsize=11, fontweight="bold")
        ax.axis("off")

        # Q heads (top row)
        for i in range(n_q_heads):
            kv_group = i // group_size
            color = q_colors[kv_group % len(q_colors)]
            rect = mpatches.FancyBboxPatch(
                (i - 0.4, 1.2), 0.8, 0.6,
                boxstyle="round,pad=0.03", facecolor=color,
                edgecolor="#333", linewidth=0.8
            )
            ax.add_patch(rect)
            if n_q_heads <= 16:
                ax.text(i, 1.5, f"Q{i}", ha="center", va="center", fontsize=6)

        # KV heads (bottom row)
        kv_spacing = n_q_heads / n_kv
        for j in range(n_kv):
            x_pos = j * kv_spacing + kv_spacing / 2 - 0.5
            color = q_colors[j % len(q_colors)]
            rect = mpatches.FancyBboxPatch(
                (x_pos - 0.4, -0.6), 0.8, 0.6,
                boxstyle="round,pad=0.03", facecolor=color,
                edgecolor="#333", linewidth=1.2
            )
            ax.add_patch(rect)
            if n_kv <= 16:
                ax.text(x_pos, -0.3, f"KV{j}", ha="center", va="center", fontsize=6)

            # Lines connecting Q heads to their KV head
            for qi in range(j * group_size, (j + 1) * group_size):
                ax.plot([qi, x_pos], [1.2, 0.0], color=color, alpha=0.3, linewidth=0.5)

        # Labels
        ax.text(-0.5, 1.5, "Q", fontsize=10, fontweight="bold", va="center")
        ax.text(-0.5, -0.3, "KV", fontsize=10, fontweight="bold", va="center")

        # Cache reduction factor
        reduction = n_q_heads / n_kv
        ax.text(n_q_heads / 2 - 0.5, -1.3,
                f"Cache: 1/{reduction:.0f}x" if reduction > 1 else "Cache: 1x",
                ha="center", fontsize=10, fontweight="bold",
                color="#C44E52" if reduction > 1 else "#333")

    fig.suptitle("Query-Key-Value Head Sharing Patterns", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


def draw_batching_timeline(timeline, title="Batching Timeline", max_slots=None):
    """Draw a timeline heatmap for batching simulations.

    Args:
        timeline: list of (timestep, [(request_id, is_padding), ...])
        title: plot title
        max_slots: max batch slots to show (auto if None)
    """
    if not timeline:
        return None, None

    n_steps = len(timeline)
    if max_slots is None:
        max_slots = max(len(step[1]) for step in timeline)

    fig, ax = plt.subplots(figsize=(max(8, n_steps * 0.5), max(3, max_slots * 0.6)))

    for t_idx, (t, slots) in enumerate(timeline):
        for s_idx, (req_id, is_padding) in enumerate(slots):
            if is_padding:
                color = "#f0f0f0"
                edge = "#ddd"
                label = "pad"
                fontcolor = "#bbb"
            else:
                color = SEQ_COLORS[req_id % len(SEQ_COLORS)]
                edge = "#333"
                label = f"R{req_id}"
                fontcolor = "white"

            rect = mpatches.FancyBboxPatch(
                (t_idx + 0.05, s_idx + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=edge, linewidth=0.8
            )
            ax.add_patch(rect)
            ax.text(t_idx + 0.5, s_idx + 0.5, label,
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color=fontcolor)

    ax.set_xlim(0, n_steps)
    ax.set_ylim(0, max_slots)
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Batch Slot", fontsize=11)
    ax.set_yticks([s + 0.5 for s in range(max_slots)])
    ax.set_yticklabels([f"Slot {s}" for s in range(max_slots)], fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig, ax


def draw_memory_map(mem_map, block_size=1, title="Memory Layout",
                    figsize=None, show_waste=True):
    """Draw a 1D memory map showing allocated vs free blocks.

    Args:
        mem_map: list where each element is a seq_id (int) or None (free)
        block_size: visual block size
        title: plot title
        figsize: optional figure size
        show_waste: if True, color free slots as wasted
    """
    n = len(mem_map)
    if figsize is None:
        figsize = (max(10, n * 0.3), 2)

    fig, ax = plt.subplots(figsize=figsize)

    for i, owner in enumerate(mem_map):
        if owner is not None:
            color = SEQ_COLORS[owner % len(SEQ_COLORS)]
            edge = "#333"
        elif show_waste:
            color = "#ffcdd2"
            edge = "#ef9a9a"
        else:
            color = "#f5f5f5"
            edge = "#e0e0e0"

        rect = mpatches.FancyBboxPatch(
            (i + 0.05, 0.1), 0.9, 0.8,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor=edge, linewidth=0.8
        )
        ax.add_patch(rect)
        if owner is not None and n <= 40:
            ax.text(i + 0.5, 0.5, f"S{owner}", ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white")

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Memory Slot", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_yticks([])

    # Stats
    used = sum(1 for x in mem_map if x is not None)
    free = n - used
    ax.text(n, 0.5, f"Used: {used}/{n}\nWaste: {free}",
            fontsize=9, va="center", ha="left", color="#555")

    plt.tight_layout()
    return fig, ax


def draw_block_table(manager, seq_ids=None, title="Block Tables (PagedAttention)"):
    """Visualize block tables for PagedAttention sequences.

    Shows logical → physical block mapping and physical memory layout.
    """
    if seq_ids is None:
        seq_ids = list(manager.sequence_tables.keys())

    n_seqs = len(seq_ids)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, n_seqs * 1.2)),
                              gridspec_kw={'width_ratios': [1, 2]})

    # Left: block tables
    ax = axes[0]
    ax.set_title("Logical Block Tables", fontsize=11, fontweight="bold")
    ax.axis("off")

    for i, seq_id in enumerate(seq_ids):
        table = manager.get_block_table(seq_id)
        y = n_seqs - 1 - i
        color = SEQ_COLORS[seq_id % len(SEQ_COLORS)]
        ax.text(-0.5, y, f"Seq {seq_id}:", fontsize=10, fontweight="bold",
                va="center", ha="right", color=color)
        for j, block_id in enumerate(table):
            rect = mpatches.FancyBboxPatch(
                (j * 1.2, y - 0.3), 1.0, 0.6,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="#333", linewidth=1, alpha=0.7
            )
            ax.add_patch(rect)
            ax.text(j * 1.2 + 0.5, y, f"→{block_id}",
                    ha="center", va="center", fontsize=9, fontweight="bold")

    max_blocks = max(len(manager.get_block_table(s)) for s in seq_ids) if seq_ids else 1
    ax.set_xlim(-1.5, max_blocks * 1.2 + 0.5)
    ax.set_ylim(-0.8, n_seqs)

    # Right: physical memory
    ax2 = axes[1]
    ax2.set_title("Physical Memory Blocks", fontsize=11, fontweight="bold")
    mem_map = manager.get_memory_map()
    n_blocks = manager.num_blocks
    cols = min(n_blocks, 16)
    rows = (n_blocks + cols - 1) // cols

    for idx in range(n_blocks):
        r = idx // cols
        c = idx % cols
        owner = mem_map[idx]
        if owner is not None:
            color = SEQ_COLORS[owner % len(SEQ_COLORS)]
            label = f"B{idx}\nS{owner}"
            edge = "#333"
        else:
            color = "#f5f5f5"
            label = f"B{idx}\nfree"
            edge = "#ddd"

        rect = mpatches.FancyBboxPatch(
            (c + 0.05, rows - 1 - r + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor=edge, linewidth=1, alpha=0.8
        )
        ax2.add_patch(rect)
        ax2.text(c + 0.5, rows - 1 - r + 0.5, label,
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white" if owner is not None else "#bbb")

    ax2.set_xlim(0, cols)
    ax2.set_ylim(0, rows)
    ax2.set_aspect("equal")
    ax2.axis("off")

    used, total, util = manager.memory_usage()
    ax2.text(cols / 2, -0.3, f"Utilization: {used}/{total} blocks ({util:.0%})",
             ha="center", fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


def draw_quantization_comparison(original, quantized_dict, title="Quantization Comparison"):
    """Draw histograms comparing original vs quantized weight distributions.

    Args:
        original: 1D tensor of original weights
        quantized_dict: dict of {label: dequantized_tensor}
        title: plot title
    """
    n = len(quantized_dict) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))

    orig_np = original.detach().numpy().flatten()
    axes[0].hist(orig_np, bins=50, color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[0].set_title("Original (FP32)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Value", fontsize=9)

    for i, (label, q_tensor) in enumerate(quantized_dict.items()):
        q_np = q_tensor.detach().numpy().flatten()
        error = orig_np - q_np
        axes[i + 1].hist(q_np, bins=50, color=SEQ_COLORS[i % len(SEQ_COLORS)],
                         alpha=0.8, edgecolor="white")
        rmse = np.sqrt(np.mean(error ** 2))
        axes[i + 1].set_title(f"{label}\nRMSE={rmse:.4f}", fontsize=11, fontweight="bold")
        axes[i + 1].set_xlabel("Value", fontsize=9)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


def draw_speculative_decoding_step(q_probs, p_probs, draft_tokens, accepted_mask,
                                    token_names=None,
                                    title="Speculative Decoding — One Round"):
    """Visualize one round of speculative decoding acceptance/rejection.

    Args:
        q_probs: (K, vocab_size) draft model probabilities
        p_probs: (K, vocab_size) target model probabilities
        draft_tokens: (K,) draft token ids
        accepted_mask: (K,) boolean acceptance mask
        token_names: optional list of token label strings
        title: plot title
    """
    K = len(draft_tokens)
    fig, ax = plt.subplots(figsize=(max(8, K * 2), 4))
    ax.set_xlim(-0.5, K + 0.5)
    ax.set_ylim(-1.5, 3)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(K):
        tok = draft_tokens[i].item() if hasattr(draft_tokens[i], 'item') else int(draft_tokens[i])
        accepted = bool(accepted_mask[i])
        p_target = float(p_probs[i, tok])
        p_draft = float(q_probs[i, tok])
        accept_prob = min(1.0, p_target / (p_draft + 1e-10))

        color = "#81C784" if accepted else "#E57373"
        status = "ACCEPT" if accepted else "REJECT"

        rect = mpatches.FancyBboxPatch(
            (i - 0.4, 0.8), 0.8, 1.2,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#333", linewidth=1.5
        )
        ax.add_patch(rect)

        tok_label = token_names[tok] if token_names else f"t{tok}"
        ax.text(i, 1.7, tok_label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(i, 1.2, status, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        ax.text(i, 0.9, f"p={accept_prob:.2f}", ha="center", va="center",
                fontsize=7, color="white")

        # Probability bars
        bar_y = -0.1
        ax.barh(bar_y, p_target, height=0.2, left=i - 0.4,
                color="#4C72B0", alpha=0.8)
        ax.text(i + p_target - 0.35, bar_y,
                f"p_t={p_target:.2f}", fontsize=6, va="center")
        ax.barh(bar_y - 0.3, p_draft, height=0.2, left=i - 0.4,
                color="#DD8452", alpha=0.8)
        ax.text(i + p_draft - 0.35, bar_y - 0.3,
                f"p_d={p_draft:.2f}", fontsize=6, va="center")

        if not accepted:
            break

    n_shown = int(accepted_mask.sum()) + (0 if accepted_mask.all() else 1)
    ax.annotate("", xy=(n_shown - 0.5, 2.5), xytext=(-0.5, 2.5),
                arrowprops=dict(arrowstyle="->", color="#666", lw=2))
    ax.text((n_shown - 1) / 2, 2.7, "Draft → Verify", ha="center",
            fontsize=10, fontweight="bold", color="#666")

    plt.tight_layout()
    return fig, ax


def draw_speedup_vs_acceptance(data=None, K_values=None, cost_ratio=0.1,
                                title="Speculative Decoding Speedup"):
    """Plot expected speedup vs acceptance rate for different K values.

    Args:
        data: dict from compute_speculative_speedup sweep mode, or None
        K_values: list of K values (used if data is None)
        cost_ratio: draft_cost / target_cost (used if data is None)
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    if data is not None and isinstance(data, dict):
        alphas = data["alphas"]
        K_vals = data["K_values"]
        cost_ratio = data.get("cost_ratio", cost_ratio)
        for K in K_vals:
            ax.plot(alphas, data["speedups"][K], linewidth=2, label=f"K={K}")
    else:
        if K_values is None:
            K_values = [1, 2, 4, 8, 16]
        alphas = np.linspace(0.01, 0.99, 100)
        for K in K_values:
            speedups = []
            for a in alphas:
                expected = (1 - a ** (K + 1)) / (1 - a)
                cost = K * cost_ratio + 1
                speedups.append(expected / cost)
            ax.plot(alphas, speedups, linewidth=2, label=f"K={K}")

    ax.axhline(1.0, color="#999", linewidth=1, linestyle="--", label="No speedup")
    ax.set_xlabel("Acceptance Rate (α)", fontsize=12)
    ax.set_ylabel("Speedup", fontsize=12)
    ax.set_title(f"{title}\n(cost_ratio={cost_ratio})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    return fig, ax


def draw_attention_memory_comparison(seq_lengths, d_model=64,
                                      sram_size=None,
                                      title="Standard vs Flash Attention Memory"):
    """Plot peak memory usage: standard O(n²) vs Flash Attention.

    Args:
        seq_lengths: list of sequence lengths
        d_model: model dimension
        sram_size: SRAM tile size (block_size² for Flash Attention)
        title: plot title
    """
    if sram_size is None:
        sram_size = 64  # block_size=8 → 64 elements per tile

    fig, ax = plt.subplots(figsize=(9, 5))

    standard_mem = [s * s for s in seq_lengths]
    flash_mem = [sram_size for _ in seq_lengths]  # constant SRAM usage

    ax.plot(seq_lengths, standard_mem, 'o-', color="#C44E52", linewidth=2.5,
            markersize=6, label="Standard Attention (O(n²))")
    ax.fill_between(seq_lengths, standard_mem, alpha=0.1, color="#C44E52")
    ax.plot(seq_lengths, flash_mem, 's-', color="#55A868", linewidth=2.5,
            markersize=6, label=f"Flash Attention (O(SRAM)={sram_size})")
    ax.fill_between(seq_lengths, flash_mem, alpha=0.1, color="#55A868")

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Peak Memory (elements)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


def draw_prefill_decode_comparison(title="Prefill vs Decode Phase"):
    """Draw a side-by-side comparison of prefill and decode phases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (phase, desc, compute, memory, color) in zip(axes, [
        ("Prefill Phase", "Process full prompt in parallel",
         "Compute-bound\n(high arithmetic intensity)", "Reads weights once\nfor many tokens", "#4C72B0"),
        ("Decode Phase", "Generate tokens one at a time",
         "Memory-bound\n(low arithmetic intensity)", "Reads full weights\nfor each token", "#DD8452"),
    ]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

        # Title box
        rect = mpatches.FancyBboxPatch(
            (0.5, 4.5), 9, 1.2,
            boxstyle="round,pad=0.1", facecolor=color,
            edgecolor="#333", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(5, 5.1, phase, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

        # Description
        ax.text(5, 3.8, desc, ha="center", va="center", fontsize=11, color="#333")

        # Compute box
        rect = mpatches.FancyBboxPatch(
            (0.5, 1.8), 4, 1.5,
            boxstyle="round,pad=0.1", facecolor="#E3F2FD",
            edgecolor="#1565C0", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(2.5, 2.9, "Compute", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1565C0")
        ax.text(2.5, 2.3, compute, ha="center", va="center",
                fontsize=9, color="#333")

        # Memory box
        rect = mpatches.FancyBboxPatch(
            (5.5, 1.8), 4, 1.5,
            boxstyle="round,pad=0.1", facecolor="#FFF3E0",
            edgecolor="#E65100", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(7.5, 2.9, "Memory", ha="center", va="center",
                fontsize=10, fontweight="bold", color="#E65100")
        ax.text(7.5, 2.3, memory, ha="center", va="center",
                fontsize=9, color="#333")

        # GPU utilization indicator
        if phase == "Prefill Phase":
            util = 0.85
            util_color = "#55A868"
        else:
            util = 0.15
            util_color = "#C44E52"

        ax.barh(0.5, util * 9, height=0.6, left=0.5,
                color=util_color, alpha=0.7, edgecolor="#333")
        ax.barh(0.5, (1 - util) * 9, height=0.6, left=0.5 + util * 9,
                color="#f0f0f0", edgecolor="#ddd")
        ax.text(5, 0.5, f"GPU Util: {util:.0%}", ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(5, 0.0, "Arithmetic Intensity", ha="center", va="center",
                fontsize=8, color="#666")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


def draw_radix_tree(tree_data, title="Prefix Cache — Radix Tree"):
    """Draw a simplified radix tree for prefix caching visualization.

    Args:
        tree_data: output from PrefixCache.get_tree_structure()
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")

    def _draw_node(node, x, y, dx, depth=0):
        # Draw this node
        if node["prefix"]:
            label = str(node["prefix"][-1])
        else:
            label = "root"

        color = SEQ_COLORS[depth % len(SEQ_COLORS)] if node["prefix"] else "#f5f5f5"
        rect = mpatches.FancyBboxPatch(
            (x - 0.3, y - 0.2), 0.6, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#333", linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if node["prefix"] else "#333")

        if node["blocks"]:
            ax.text(x, y - 0.35, f"[B{','.join(str(b) for b in node['blocks'])}]",
                    ha="center", fontsize=7, color="#666")

        # Draw children
        children = list(node["children"].values())
        if children:
            child_dx = dx / max(len(children), 1)
            start_x = x - dx / 2 + child_dx / 2
            for i, child in enumerate(children):
                cx = start_x + i * child_dx
                cy = y - 1.0
                ax.plot([x, cx], [y - 0.2, cy + 0.2], color="#999", linewidth=1)
                _draw_node(child, cx, cy, child_dx * 0.9, depth + 1)

    _draw_node(tree_data, 6, 5, 10)
    ax.set_xlim(0, 12)
    ax.set_ylim(-1, 6)
    plt.tight_layout()
    return fig, ax


def draw_operator_fusion(title="Operator Fusion: Before vs After"):
    """Draw a diagram showing unfused vs fused kernel execution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, ops, fused) in zip(axes, [
        ("Unfused (3 kernels)", ["Linear", "ReLU", "Dropout"], False),
        ("Fused (1 kernel)", ["Linear + ReLU + Dropout"], True),
    ]):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.set_title(label, fontsize=12, fontweight="bold")

        y = 6.5
        for i, op in enumerate(ops):
            # Kernel box
            color = "#55A868" if fused else GPU_COLORS[i % len(GPU_COLORS)]
            h = 4.5 if fused else 1.2
            rect = mpatches.FancyBboxPatch(
                (1, y - h), 5, h,
                boxstyle="round,pad=0.1", facecolor=color,
                edgecolor="#333", linewidth=2
            )
            ax.add_patch(rect)
            ax.text(3.5, y - h / 2, op, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")

            if not fused and i < len(ops) - 1:
                # HBM round-trip arrow
                ax.annotate("", xy=(7.5, y - h - 0.1), xytext=(7.5, y - h - 0.5),
                           arrowprops=dict(arrowstyle="<->", color="#C44E52", lw=2))
                ax.text(8.5, y - h - 0.3, "HBM\nread/write", ha="center",
                        fontsize=8, color="#C44E52", fontweight="bold")

            y -= h + 0.6

        if fused:
            ax.text(3.5, 1.0, "No intermediate\nHBM traffic!",
                    ha="center", fontsize=11, color="#55A868", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes
