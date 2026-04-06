"""Visualization helpers for diffusion tutorial notebooks.

Provides reusable plotting functions for noise schedules, forward/reverse
diffusion processes, training curves, architecture diagrams, and more.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


# Color palette consistent with the training/inference viz modules
GPU_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]

CLASS_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

SCHEDULE_COLORS = {"linear": "#4C72B0", "cosine": "#DD8452",
                   "sigmoid": "#55A868", "sqrt": "#C44E52"}


# ---------------------------------------------------------------------------
# Denoising trajectory
# ---------------------------------------------------------------------------

def draw_denoising_trajectory(intermediates, n_show=8, labels=None,
                              n_classes=4,
                              title="Denoising Trajectory"):
    """Show denoising steps as scatter plots for 2D point data.

    Args:
        intermediates: List of [B, 2] tensors from noisy to clean.
        n_show: Number of snapshots to display (evenly spaced).
        labels: Optional class labels [B] for coloring.
        n_classes: Number of distinct classes.
        title: Figure title.

    Returns:
        (fig, axes) tuple.
    """
    total = len(intermediates)
    if total <= n_show:
        indices = list(range(total))
    else:
        indices = [int(i * (total - 1) / (n_show - 1)) for i in range(n_show)]

    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3))
    if n == 1:
        axes = [axes]

    for ax_idx, snap_idx in enumerate(indices):
        ax = axes[ax_idx]
        snap = intermediates[snap_idx]
        pts = snap.numpy() if hasattr(snap, "numpy") else np.asarray(snap)

        if labels is not None:
            labs = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
            for c in range(n_classes):
                mask = labs == c
                ax.scatter(pts[mask, 0], pts[mask, 1], s=8, alpha=0.6,
                           c=CLASS_COLORS[c % len(CLASS_COLORS)], label=f"c={c}")
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, c="#3498db")

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")

        if snap_idx == 0:
            step_label = "t=T (noise)"
        elif snap_idx == total - 1:
            step_label = "t=0 (clean)"
        else:
            step_label = f"step {snap_idx}"
        ax.set_title(step_label, fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Noise schedule comparison
# ---------------------------------------------------------------------------

def draw_noise_schedule_comparison(schedules_dict, T=1000,
                                   title="Noise Schedule Comparison"):
    """Plot beta_t, alpha_bar_t, and SNR curves for multiple schedules.

    Args:
        schedules_dict: Dict mapping name -> betas tensor (T,).
        T: Number of timesteps (used for x-axis labeling).
        title: Figure title.

    Returns:
        (fig, axes) tuple with 3 subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    timesteps = np.arange(T)
    color_keys = list(SCHEDULE_COLORS.values())

    for i, (name, betas) in enumerate(schedules_dict.items()):
        betas_np = betas.numpy() if hasattr(betas, "numpy") else np.asarray(betas)
        alphas = 1.0 - betas_np
        alpha_bars = np.cumprod(alphas)
        snr = alpha_bars / np.clip(1.0 - alpha_bars, 1e-10, None)

        color = SCHEDULE_COLORS.get(name, color_keys[i % len(color_keys)])

        axes[0].plot(timesteps[:len(betas_np)], betas_np,
                     linewidth=2, label=name, color=color)
        axes[1].plot(timesteps[:len(alpha_bars)], alpha_bars,
                     linewidth=2, label=name, color=color)
        axes[2].plot(timesteps[:len(snr)], snr,
                     linewidth=2, label=name, color=color)

    axes[0].set_title("$\\beta_t$ (noise rate)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Timestep $t$", fontsize=10)
    axes[0].set_ylabel("$\\beta_t$", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("$\\bar{\\alpha}_t$ (signal retention)",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Timestep $t$", fontsize=10)
    axes[1].set_ylabel("$\\bar{\\alpha}_t$", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("SNR$(t)$ (signal-to-noise)",
                      fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Timestep $t$", fontsize=10)
    axes[2].set_ylabel("SNR", fontsize=10)
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Forward process visualization
# ---------------------------------------------------------------------------

def draw_forward_process(x_0, alpha_bars, timesteps, labels=None,
                         n_classes=4,
                         title="Forward Diffusion Process"):
    """Show noising at selected timesteps as scatter plots.

    Args:
        x_0: Clean data [N, 2].
        alpha_bars: Cumulative alpha products (T,).
        timesteps: List of integer timesteps to visualize.
        labels: Optional class labels [N].
        n_classes: Number of classes.
        title: Figure title.

    Returns:
        (fig, axes) tuple.
    """
    import torch
    n = len(timesteps) + 1  # +1 for original
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3))
    if n == 1:
        axes = [axes]

    x_0_t = x_0 if isinstance(x_0, torch.Tensor) else torch.tensor(x_0, dtype=torch.float32)
    pts_0 = x_0_t.numpy()

    def _scatter(ax, pts, lab_text):
        if labels is not None:
            labs = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
            for c in range(n_classes):
                mask = labs == c
                ax.scatter(pts[mask, 0], pts[mask, 1], s=8, alpha=0.6,
                           c=CLASS_COLORS[c % len(CLASS_COLORS)])
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, c="#3498db")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(lab_text, fontsize=10)
        ax.tick_params(labelsize=7)

    # Original
    _scatter(axes[0], pts_0, "t=0 (clean)")

    # Noisy versions
    for i, t_val in enumerate(timesteps):
        t_idx = torch.tensor([t_val] * x_0_t.shape[0])
        abar = alpha_bars[t_idx].unsqueeze(-1)
        noise = torch.randn_like(x_0_t)
        x_t = torch.sqrt(abar) * x_0_t + torch.sqrt(1.0 - abar) * noise
        _scatter(axes[i + 1], x_t.numpy(), f"t={t_val}")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def draw_training_curves(losses, title="Diffusion Training Loss",
                         smoothing=0.9):
    """Plot training loss curve with exponential smoothing.

    Args:
        losses: List of per-epoch (or per-step) loss values.
        title: Figure title.
        smoothing: Exponential moving average factor (0 = no smoothing).

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    epochs = np.arange(1, len(losses) + 1)
    ax.plot(epochs, losses, alpha=0.3, color="#4C72B0", linewidth=1,
            label="Raw loss")

    # Smoothed curve
    if smoothing > 0:
        smoothed = []
        running = losses[0]
        for v in losses:
            running = smoothing * running + (1 - smoothing) * v
            smoothed.append(running)
        ax.plot(epochs, smoothed, color="#4C72B0", linewidth=2.5,
                label=f"Smoothed (EMA {smoothing})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Guidance scale effect
# ---------------------------------------------------------------------------

def draw_guidance_scale_effect(samples_dict, true_data=None,
                               true_labels=None, n_classes=4,
                               title="Effect of Guidance Scale w"):
    """Grid showing effect of different classifier-free guidance scales.

    Args:
        samples_dict: Dict mapping w (guidance scale) -> [B, 2] samples
            or list of image tensors.
        true_data: Optional [N, 2] real data for reference panel.
        true_labels: Optional [N] labels for real data coloring.
        n_classes: Number of classes.
        title: Figure title.

    Returns:
        (fig, axes) tuple.
    """
    n = len(samples_dict) + (1 if true_data is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3))
    if n == 1:
        axes = [axes]

    col = 0
    if true_data is not None:
        ax = axes[col]
        pts = true_data.numpy() if hasattr(true_data, "numpy") else np.asarray(true_data)
        if true_labels is not None:
            labs = true_labels.numpy() if hasattr(true_labels, "numpy") else np.asarray(true_labels)
            for c in range(n_classes):
                mask = labs == c
                ax.scatter(pts[mask, 0], pts[mask, 1], s=8, alpha=0.6,
                           c=CLASS_COLORS[c % len(CLASS_COLORS)])
        else:
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, c="#aaa")
        ax.set_title("Ground truth", fontsize=10)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        col += 1

    for w, samples in samples_dict.items():
        ax = axes[col]
        pts = samples.numpy() if hasattr(samples, "numpy") else np.asarray(samples)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.5, c="#3498db")
        ax.set_title(f"w = {w}", fontsize=10)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        col += 1

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Architecture block diagrams
# ---------------------------------------------------------------------------

def _draw_box(ax, x, y, w, h, text, color, fontsize=9, text_color="white",
              edge="#333", lw=1.5, style="round,pad=0.05"):
    """Helper to draw a labeled rounded box."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=style,
        facecolor=color, edgecolor=edge, linewidth=lw
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color)


def _draw_arrow(ax, x1, y1, x2, y2, color="#666"):
    """Helper to draw an arrow between two points."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def draw_architecture_block(block_type, title=None):
    """Draw a matplotlib diagram of a diffusion architecture block.

    Supported block_type values: "resblock", "cross_attention",
    "adaln_zero", "patch_embed", "zero_conv".

    Args:
        block_type: One of the supported block type strings.
        title: Optional figure title override.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    if block_type == "resblock":
        t = title or "ResBlock with Time Conditioning"
        ax.set_title(t, fontsize=13, fontweight="bold")

        _draw_box(ax, 3, 8.2, 4, 0.8, "Input x", "#90A4AE")
        _draw_arrow(ax, 5, 8.2, 5, 7.6)
        _draw_box(ax, 3, 6.8, 4, 0.8, "GroupNorm + SiLU", "#4C72B0")
        _draw_arrow(ax, 5, 6.8, 5, 6.2)
        _draw_box(ax, 3, 5.4, 4, 0.8, "Conv 3x3", "#4C72B0")

        # Time embedding branch
        _draw_box(ax, 0.2, 5.4, 2.4, 0.8, "t_emb", "#DD8452")
        _draw_arrow(ax, 2.6, 5.8, 3, 5.8)
        ax.text(2.8, 5.3, "+", fontsize=14, fontweight="bold", color="#DD8452")

        _draw_arrow(ax, 5, 5.4, 5, 4.8)
        _draw_box(ax, 3, 4.0, 4, 0.8, "GroupNorm + SiLU", "#4C72B0")
        _draw_arrow(ax, 5, 4.0, 5, 3.4)
        _draw_box(ax, 3, 2.6, 4, 0.8, "Conv 3x3", "#4C72B0")
        _draw_arrow(ax, 5, 2.6, 5, 2.0)
        _draw_box(ax, 3, 1.2, 4, 0.8, "Output (x + h)", "#55A868")

        # Skip connection
        ax.annotate("", xy=(7.5, 1.6), xytext=(7.5, 8.6),
                    arrowprops=dict(arrowstyle="->", color="#C44E52",
                                    lw=2, linestyle="--"))
        ax.text(7.8, 5.0, "skip", fontsize=9, color="#C44E52", rotation=90,
                va="center")

    elif block_type == "cross_attention":
        t = title or "Cross-Attention Block"
        ax.set_title(t, fontsize=13, fontweight="bold")

        _draw_box(ax, 0.3, 7.5, 3.5, 0.8, "Image tokens x", "#4C72B0")
        _draw_box(ax, 6.2, 7.5, 3.5, 0.8, "Text tokens c", "#DD8452")

        _draw_arrow(ax, 2, 7.5, 2, 6.8)
        _draw_arrow(ax, 8, 7.5, 5, 6.8)
        _draw_arrow(ax, 8, 7.5, 8, 6.8)

        _draw_box(ax, 1, 6.0, 2, 0.8, "W_q", "#4C72B0")
        _draw_box(ax, 4, 6.0, 2, 0.8, "W_k", "#DD8452")
        _draw_box(ax, 7, 6.0, 2, 0.8, "W_v", "#DD8452")

        _draw_arrow(ax, 2, 6.0, 3, 5.2)
        _draw_arrow(ax, 5, 6.0, 4, 5.2)

        _draw_box(ax, 2.5, 4.4, 3, 0.8, "Q @ K^T / sqrt(d)", "#8172B3")
        _draw_arrow(ax, 4, 4.4, 4, 3.8)

        _draw_box(ax, 2.5, 3.0, 3, 0.8, "Softmax", "#8172B3")
        _draw_arrow(ax, 5.5, 3.4, 8, 6.0)
        _draw_arrow(ax, 4, 3.0, 4, 2.4)

        _draw_box(ax, 2.5, 1.6, 3, 0.8, "Attn @ V", "#8172B3")
        _draw_arrow(ax, 4, 1.6, 4, 1.0)

        _draw_box(ax, 2.5, 0.2, 3, 0.8, "W_o + Residual", "#55A868")

    elif block_type == "adaln_zero":
        t = title or "AdaLN-Zero Block (DiT)"
        ax.set_title(t, fontsize=13, fontweight="bold")

        _draw_box(ax, 3, 8.5, 4, 0.7, "Input x", "#90A4AE")
        _draw_arrow(ax, 5, 8.5, 5, 8.0)

        _draw_box(ax, 3, 7.3, 4, 0.7, "LayerNorm", "#4C72B0")

        # Conditioning branch
        _draw_box(ax, 0.2, 5.8, 2.4, 0.7, "cond (t+c)", "#DD8452")
        _draw_arrow(ax, 2.6, 6.15, 3.0, 6.15)

        _draw_box(ax, 3, 5.8, 4, 0.7, "Linear -> (shift, scale, gate)",
                  "#DD8452", fontsize=7)

        _draw_arrow(ax, 5, 7.3, 5, 6.8)
        _draw_box(ax, 3, 4.3, 4, 0.7, "x * (1 + scale) + shift", "#8172B3",
                  fontsize=8)
        _draw_arrow(ax, 5, 5.8, 5, 5.0)
        _draw_arrow(ax, 5, 4.3, 5, 3.8)
        _draw_box(ax, 3, 3.1, 4, 0.7, "gate * h", "#C44E52")
        _draw_arrow(ax, 5, 3.1, 5, 2.6)
        _draw_box(ax, 3, 1.9, 4, 0.7, "Output", "#55A868")

        ax.text(5, 1.2, "gate initialized to 0\n(identity at init)",
                ha="center", fontsize=9, color="#C44E52", style="italic")

    elif block_type == "patch_embed":
        t = title or "Patch Embedding (ViT / DiT)"
        ax.set_title(t, fontsize=13, fontweight="bold")

        # Draw image grid
        for r in range(4):
            for c in range(4):
                color = GPU_COLORS[(r * 4 + c) % len(GPU_COLORS)]
                rect = mpatches.FancyBboxPatch(
                    (1.5 + c * 0.8, 6.5 + (3 - r) * 0.8), 0.7, 0.7,
                    boxstyle="round,pad=0.02", facecolor=color,
                    edgecolor="#333", linewidth=0.8, alpha=0.6
                )
                ax.add_patch(rect)
        ax.text(3.1, 10.0, "Image (H x W)", ha="center", fontsize=10,
                fontweight="bold")

        _draw_arrow(ax, 3.1, 6.3, 3.1, 5.8)
        _draw_box(ax, 1.5, 5.0, 3.2, 0.8, "Conv2d (P x P stride)",
                  "#4C72B0", fontsize=8)
        _draw_arrow(ax, 3.1, 5.0, 3.1, 4.4)
        _draw_box(ax, 1.5, 3.6, 3.2, 0.8, "Flatten + Transpose",
                  "#8172B3", fontsize=8)
        _draw_arrow(ax, 3.1, 3.6, 3.1, 3.0)

        # Token sequence
        for i in range(6):
            color = GPU_COLORS[i % len(GPU_COLORS)]
            rect = mpatches.FancyBboxPatch(
                (0.5 + i * 1.0, 2.0), 0.8, 0.8,
                boxstyle="round,pad=0.02", facecolor=color,
                edgecolor="#333", linewidth=0.8, alpha=0.6
            )
            ax.add_patch(rect)
        ax.text(2.2, 2.4, "...", fontsize=12, fontweight="bold")
        ax.text(3.5, 1.2, "(B, n_patches, embed_dim)", ha="center",
                fontsize=10, fontweight="bold", color="#555")

    elif block_type == "zero_conv":
        t = title or "Zero-Initialized Convolution (ControlNet)"
        ax.set_title(t, fontsize=13, fontweight="bold")

        _draw_box(ax, 3, 7.5, 4, 0.8, "ControlNet features", "#DD8452")
        _draw_arrow(ax, 5, 7.5, 5, 6.8)
        _draw_box(ax, 3, 6.0, 4, 0.8, "Conv 1x1 (W=0, b=0)", "#C44E52")
        _draw_arrow(ax, 5, 6.0, 5, 5.2)

        ax.text(5, 4.6, "Initially outputs all zeros",
                ha="center", fontsize=10, color="#C44E52", style="italic")
        ax.text(5, 4.0, "Gradually learns residual signals",
                ha="center", fontsize=10, color="#55A868", style="italic")

        _draw_arrow(ax, 5, 3.5, 5, 2.8)
        _draw_box(ax, 3, 2.0, 4, 0.8, "Add to UNet features", "#55A868")
    else:
        ax.text(5, 5, f"Unknown block_type: {block_type!r}",
                ha="center", va="center", fontsize=12, color="#C44E52")

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# UNet architecture diagram
# ---------------------------------------------------------------------------

def draw_unet_architecture(title="Simplified UNet Architecture (DDPM)"):
    """Draw a simplified UNet diagram with labeled components.

    Shows the encoder-decoder structure with skip connections,
    ResBlocks, attention, and timestep conditioning.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Encoder blocks (left, going down)
    enc_specs = [
        (1.0, 6.0, 2.0, 0.9, "ResBlock\n64ch", "#4C72B0"),
        (1.0, 4.5, 2.0, 0.9, "ResBlock + Attn\n128ch", "#4C72B0"),
        (1.0, 3.0, 2.0, 0.9, "ResBlock + Attn\n256ch", "#4C72B0"),
    ]
    for x, y, w, h, text, color in enc_specs:
        _draw_box(ax, x, y, w, h, text, color, fontsize=8)

    # Downsample arrows
    for i in range(2):
        y_from = enc_specs[i][1]
        y_to = enc_specs[i + 1][1] + enc_specs[i + 1][3]
        _draw_arrow(ax, 2.0, y_from, 2.0, y_to)
        ax.text(2.3, (y_from + y_to) / 2, "Down", fontsize=7, color="#666")

    # Bottleneck
    _draw_box(ax, 5.5, 1.2, 3.0, 0.9, "Bottleneck\nResBlock + Attn + ResBlock",
              "#8172B3", fontsize=8)

    # Connect encoder to bottleneck
    _draw_arrow(ax, 2.0, 3.0, 5.5, 1.65)

    # Decoder blocks (right, going up)
    dec_specs = [
        (11.0, 3.0, 2.0, 0.9, "ResBlock + Attn\n256ch", "#55A868"),
        (11.0, 4.5, 2.0, 0.9, "ResBlock + Attn\n128ch", "#55A868"),
        (11.0, 6.0, 2.0, 0.9, "ResBlock\n64ch", "#55A868"),
    ]
    for x, y, w, h, text, color in dec_specs:
        _draw_box(ax, x, y, w, h, text, color, fontsize=8)

    # Upsample arrows
    for i in range(2):
        y_from = dec_specs[i][1] + dec_specs[i][3]
        y_to = dec_specs[i + 1][1]
        _draw_arrow(ax, 12.0, y_from, 12.0, y_to)
        ax.text(12.3, (y_from + y_to) / 2, "Up", fontsize=7, color="#666")

    # Connect bottleneck to decoder
    _draw_arrow(ax, 8.5, 1.65, 11.0, 3.4)

    # Skip connections (dashed)
    for enc, dec in zip(enc_specs, reversed(list(dec_specs))):
        y_enc = enc[1] + enc[3] / 2
        y_dec = dec[1] + dec[3] / 2
        ax.annotate("", xy=(dec[0], y_dec), xytext=(enc[0] + enc[2], y_enc),
                    arrowprops=dict(arrowstyle="->", color="#C44E52",
                                    lw=1.5, linestyle="--"))

    ax.text(7.0, 6.9, "skip connections", fontsize=9, color="#C44E52",
            ha="center", style="italic")

    # Timestep embedding
    _draw_box(ax, 5.5, 7.0, 3.0, 0.6, "Sinusoidal t_emb", "#DD8452",
              fontsize=9)
    ax.annotate("", xy=(2.0, 6.9), xytext=(5.5, 7.3),
                arrowprops=dict(arrowstyle="->", color="#DD8452",
                                lw=1, linestyle=":"))
    ax.annotate("", xy=(12.0, 6.9), xytext=(8.5, 7.3),
                arrowprops=dict(arrowstyle="->", color="#DD8452",
                                lw=1, linestyle=":"))

    # Input/Output labels
    ax.text(2.0, 7.3, "x_t (noisy input)", ha="center", fontsize=10,
            fontweight="bold", color="#333")
    ax.text(12.0, 7.3, "eps_pred (noise)", ha="center", fontsize=10,
            fontweight="bold", color="#333")

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# DiT architecture diagram
# ---------------------------------------------------------------------------

def draw_dit_architecture(title="Simplified DiT Architecture"):
    """Draw a simplified DiT (Diffusion Transformer) diagram.

    Shows patch embedding, transformer blocks with AdaLN-Zero,
    and final linear projection.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Input
    _draw_box(ax, 3, 8.8, 4, 0.6, "Noisy latent z_t", "#90A4AE", fontsize=9)
    _draw_arrow(ax, 5, 8.8, 5, 8.3)

    # Patch Embed
    _draw_box(ax, 3, 7.5, 4, 0.8, "Patch Embed\n+ Pos Encoding", "#4C72B0",
              fontsize=9)
    _draw_arrow(ax, 5, 7.5, 5, 7.0)

    # Conditioning
    _draw_box(ax, 0.2, 6.0, 2.3, 0.7, "t_emb + c_emb", "#DD8452", fontsize=8)

    # DiT blocks
    for i, y_base in enumerate([5.5, 3.8]):
        block_color = "#8172B3"
        _draw_box(ax, 3, y_base, 4, 1.2,
                  f"DiT Block {i+1}\nAdaLN-Zero + Self-Attn\n+ AdaLN-Zero + FFN",
                  block_color, fontsize=7)
        _draw_arrow(ax, 2.5, y_base + 0.6, 3.0, y_base + 0.6)
        if y_base > 4.0:
            _draw_arrow(ax, 5, y_base, 5, y_base - 0.5)

    ax.text(3.5, 5.6, "x N", fontsize=16, fontweight="bold", color="#666",
            ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#666",
                      alpha=0.8))

    # Final layer
    _draw_arrow(ax, 5, 3.8, 5, 3.2)
    _draw_box(ax, 3, 2.4, 4, 0.8, "AdaLN-Zero\n+ Linear Projection",
              "#55A868", fontsize=9)
    _draw_arrow(ax, 5, 2.4, 5, 1.8)

    # Output
    _draw_box(ax, 3, 1.0, 4, 0.8, "Predicted noise / v",
              "#C44E52", fontsize=9)

    # Conditioning arrows
    ax.annotate("", xy=(3.0, 5.0), xytext=(2.5, 6.0),
                arrowprops=dict(arrowstyle="->", color="#DD8452",
                                lw=1.5, linestyle=":"))

    ax.text(0.5, 5.0, "AdaLN-Zero\nmodulation", fontsize=8, color="#DD8452",
            ha="center", style="italic")

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Latent Diffusion / Stable Diffusion pipeline
# ---------------------------------------------------------------------------

def draw_latent_pipeline(title="Stable Diffusion Pipeline"):
    """Draw the Stable Diffusion pipeline diagram.

    Shows: text encoder -> cross-attention -> UNet in latent space ->
    VAE decoder -> output image.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Text Encoder
    _draw_box(ax, 0.3, 1.5, 2.4, 2.0, "Text\nEncoder\n(CLIP)", "#DD8452",
              fontsize=10)
    ax.text(1.5, 0.8, '"a cat on\na beach"', ha="center", fontsize=8,
            style="italic", color="#555")

    # Arrow to UNet
    _draw_arrow(ax, 2.7, 2.5, 4.0, 2.5)
    ax.text(3.35, 2.8, "text\nembeddings", ha="center", fontsize=7,
            color="#666")

    # Noise
    _draw_box(ax, 4.0, 3.8, 1.5, 0.8, "z_T ~ N(0,I)", "#90A4AE", fontsize=8)
    _draw_arrow(ax, 4.75, 3.8, 4.75, 3.5)

    # Timestep
    _draw_box(ax, 6.5, 3.8, 1.5, 0.8, "t_emb", "#937860", fontsize=9)
    _draw_arrow(ax, 7.25, 3.8, 7.25, 3.5)

    # UNet in latent space
    _draw_box(ax, 4.0, 1.5, 4.5, 2.0, "UNet\n(in latent space)\nCross-Attn + ResBlocks",
              "#4C72B0", fontsize=9)

    # Arrow to VAE
    _draw_arrow(ax, 8.5, 2.5, 10.0, 2.5)
    ax.text(9.25, 2.8, "z_0", ha="center", fontsize=9, fontweight="bold",
            color="#666")

    # VAE Decoder
    _draw_box(ax, 10.0, 1.5, 2.4, 2.0, "VAE\nDecoder", "#55A868",
              fontsize=10)

    # Arrow to output
    _draw_arrow(ax, 12.4, 2.5, 13.5, 2.5)

    # Output image placeholder
    rect = mpatches.FancyBboxPatch(
        (13.5, 1.5), 2.0, 2.0, boxstyle="round,pad=0.05",
        facecolor="#FFF3E0", edgecolor="#333", linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(14.5, 2.5, "Output\nImage", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333")

    # Loop annotation
    ax.annotate("", xy=(4.75, 1.5), xytext=(4.75, 0.6),
                arrowprops=dict(arrowstyle="->", color="#8172B3", lw=1.5,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(3.5, 0.4, "iterate T steps", fontsize=8, color="#8172B3",
            style="italic")

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Loss weighting comparison
# ---------------------------------------------------------------------------

def draw_loss_weighting_comparison(alpha_bars, T=1000,
                                   title="Loss Weighting Strategies"):
    """Compare uniform, SNR, and min-SNR loss weighting curves.

    Args:
        alpha_bars: Precomputed cumulative alphas (T,).
        T: Number of timesteps for x-axis.
        title: Figure title.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    abar = alpha_bars.numpy() if hasattr(alpha_bars, "numpy") else np.asarray(alpha_bars)
    timesteps = np.arange(len(abar))

    snr = abar / np.clip(1.0 - abar, 1e-10, None)

    # Uniform weighting
    uniform = np.ones_like(abar)
    ax.plot(timesteps, uniform, linewidth=2, label="Uniform",
            color="#4C72B0", linestyle="--")

    # SNR weighting
    snr_weight = snr / snr.max()
    ax.plot(timesteps, snr_weight, linewidth=2, label="SNR weighting",
            color="#DD8452")

    # Min-SNR (gamma=5)
    for gamma, color in [(5.0, "#55A868"), (1.0, "#C44E52")]:
        min_snr = np.minimum(snr, gamma) / np.clip(snr, 1e-10, None)
        ax.plot(timesteps, min_snr, linewidth=2,
                label=f"Min-SNR ($\\gamma$={gamma})", color=color)

    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("Loss Weight (normalized)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(abar) - 1)
    plt.tight_layout()
    return fig, ax
