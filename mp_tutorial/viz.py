"""Visualization helpers for model parallelism tutorials.

Provides reusable functions for creating diagrams commonly used
across notebooks: tensor split views, pipeline stage timelines,
communication pattern illustrations, GPU topology maps, etc.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import colormaps
import numpy as np

# Consistent GPU colors across all visualizations
GPU_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def show_matrix(tensor, ax=None, title="", gpu_label=None, cmap="YlOrRd",
                fmt=".1f", fontsize=10):
    """Display a 2D tensor as a color-coded grid with values annotated.

    Args:
        tensor: 2D torch.Tensor or numpy array.
        ax: matplotlib Axes (created if None).
        title: Title above the matrix.
        gpu_label: If set, adds a colored GPU badge (e.g., "GPU 0").
        cmap: Colormap name.
        fmt: Number format string.
        fontsize: Font size for cell values.

    Returns:
        matplotlib Axes.
    """
    import torch
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(arr.shape[1] * 0.9, 2.5),
                                         max(arr.shape[0] * 0.7, 1.5)))

    im = ax.imshow(arr, cmap=cmap, aspect="equal")

    # Annotate each cell with its value
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            color = "white" if abs(val) > (arr.max() + arr.min()) / 2 else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                    fontsize=fontsize, color=color, fontweight="bold")

    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    ax.set_xticklabels([f"c{j}" for j in range(arr.shape[1])], fontsize=8)
    ax.set_yticklabels([f"r{i}" for i in range(arr.shape[0])], fontsize=8)

    label = title
    if gpu_label is not None:
        gpu_idx = int(gpu_label.split()[-1]) if gpu_label.split()[-1].isdigit() else 0
        color = GPU_COLORS[gpu_idx % len(GPU_COLORS)]
        label = f"{title}  " if title else ""
        ax.set_title(label, fontsize=11, fontweight="bold", loc="left")
        ax.annotate(gpu_label, xy=(1, 1), xycoords="axes fraction",
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc=color, ec="none"),
                    ha="right", va="bottom")
        return ax

    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    return ax


def show_matrices_row(tensors, titles=None, gpu_labels=None, suptitle="",
                      cmap="YlOrRd", fmt=".1f"):
    """Show multiple matrices side by side in a single row.

    Args:
        tensors: List of 2D tensors/arrays.
        titles: List of titles (one per tensor).
        gpu_labels: List of GPU labels (e.g., ["GPU 0", "GPU 1"]).
        suptitle: Overall figure title.
        cmap: Colormap name.
        fmt: Number format string.

    Returns:
        matplotlib Figure.
    """
    import torch
    n = len(tensors)
    titles = titles or [""] * n
    gpu_labels = gpu_labels or [None] * n

    # Compute figure size based on tensor shapes
    shapes = []
    for t in tensors:
        arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        shapes.append(arr.shape)

    total_w = sum(max(s[1] * 0.9, 2.5) for s in shapes) + (n - 1) * 0.5
    max_h = max(max(s[0] * 0.7, 1.5) for s in shapes)
    fig, axes = plt.subplots(1, n, figsize=(total_w, max_h + 1.2))
    if n == 1:
        axes = [axes]

    for ax, tensor, title, glabel in zip(axes, tensors, titles, gpu_labels):
        show_matrix(tensor, ax=ax, title=title, gpu_label=glabel, cmap=cmap, fmt=fmt)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def draw_tensor_split(tensor_shape, split_dim, num_splits, title="Tensor Split"):
    """Draw a visual representation of a tensor being split along a dimension.

    Args:
        tensor_shape: Tuple of (rows, cols) for the 2D tensor.
        split_dim: Dimension to split along (0=rows, 1=cols).
        num_splits: Number of splits (GPUs).
        title: Plot title.

    Returns:
        matplotlib Figure and Axes.
    """
    rows, cols = tensor_shape
    cmap = plt.cm.get_cmap("tab10")

    fig, ax = plt.subplots(1, 1, figsize=(max(cols * 0.6, 4), max(rows * 0.6, 3)))

    # Draw grid cells colored by GPU assignment
    for r in range(rows):
        for c in range(cols):
            if split_dim == 1:  # column split
                gpu_idx = c // (cols // num_splits)
            else:  # row split
                gpu_idx = r // (rows // num_splits)
            gpu_idx = min(gpu_idx, num_splits - 1)
            color = cmap(gpu_idx % 10)
            rect = plt.Rectangle((c, rows - 1 - r), 1, 1,
                                 facecolor=color, edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)

    # Draw partition boundary lines
    for s in range(1, num_splits):
        if split_dim == 1:
            x = s * (cols // num_splits)
            ax.plot([x, x], [0, rows], color="black", linewidth=2.5)
        else:
            y = rows - s * (rows // num_splits)
            ax.plot([0, cols], [y, y], color="black", linewidth=2.5)

    # Add GPU labels
    for s in range(num_splits):
        if split_dim == 1:
            cx = s * (cols // num_splits) + (cols // num_splits) / 2
            cy = rows / 2
        else:
            cx = cols / 2
            cy = rows - s * (rows // num_splits) - (rows // num_splits) / 2
        ax.text(cx, cy, f"GPU {s}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xlabel("Columns" if split_dim == 1 else "Columns")
    ax.set_ylabel("Rows")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.tight_layout()
    return fig, ax


def _generate_gpipe_schedule(num_stages, num_microbatches):
    """Generate GPipe schedule: all forwards, then all backwards."""
    schedule = []
    # Forward passes: stage s starts micro-batch m at time s + m
    for m in range(num_microbatches):
        for s in range(num_stages):
            t = s + m
            schedule.append((t, s, m, "fwd"))
    # Backward passes start after all forwards complete
    fwd_end = num_stages + num_microbatches - 1
    for m in range(num_microbatches):
        for s in range(num_stages - 1, -1, -1):
            t = fwd_end + (num_stages - 1 - s) + m
            schedule.append((t, s, m, "bwd"))
    return schedule


def _generate_1f1b_schedule(num_stages, num_microbatches):
    """Generate 1F1B schedule: warmup, then steady state 1-fwd-1-bwd."""
    schedule = []
    # Track next available time for each stage
    next_time = [0] * num_stages
    fwd_done = [0] * num_stages  # count of forwards done per stage

    # Warmup phase: fill the pipeline with forwards
    for m in range(num_stages):
        for s in range(num_stages):
            if m < num_microbatches and s <= m:
                pass  # handled below

    # Simpler approach: compute schedule directly
    # Forward: micro-batch m arrives at stage s at time m + s
    fwd_times = {}
    for m in range(num_microbatches):
        for s in range(num_stages):
            fwd_times[(s, m)] = m + s
            schedule.append((m + s, s, m, "fwd"))

    # Backward: 1F1B means after warmup, each stage does 1 bwd after each fwd
    # Backward for micro-batch m at stage s:
    # - Last stage starts bwd for m=0 at time (num_stages - 1) + 1 = num_stages
    # - Then pipelines backward through stages
    bwd_times = {}
    for m in range(num_microbatches):
        for s in range(num_stages - 1, -1, -1):
            # Backward at last stage for micro-batch m starts after its forward
            if s == num_stages - 1:
                bwd_start = fwd_times[(s, m)] + 1
            else:
                # Must wait for backward from stage s+1 and any forward at this stage
                bwd_start = bwd_times[(s + 1, m)] + 1
            # Also must wait for this stage to finish its current work
            bwd_times[(s, m)] = bwd_start

    # Re-collect backward into schedule
    for m in range(num_microbatches):
        for s in range(num_stages - 1, -1, -1):
            schedule.append((bwd_times[(s, m)], s, m, "bwd"))

    return schedule


def _generate_interleaved_schedule(num_stages, num_microbatches):
    """Generate interleaved virtual pipeline schedule (2 virtual stages per GPU)."""
    # With interleaving, each GPU holds multiple non-contiguous chunks
    # For simplicity, show 2 virtual stages per physical GPU
    # This effectively doubles the stages but halves layers per stage
    virtual_stages = num_stages * 2
    schedule = []

    # Forward passes through virtual stages
    for m in range(num_microbatches):
        for vs in range(virtual_stages):
            physical_gpu = vs % num_stages
            t = m * 2 + vs  # interleaved timing
            schedule.append((t, physical_gpu, m, "fwd"))

    # Backward passes
    fwd_end = max(t for t, _, _, _ in schedule) + 1
    for m in range(num_microbatches):
        for vs in range(virtual_stages - 1, -1, -1):
            physical_gpu = vs % num_stages
            t = fwd_end + m * 2 + (virtual_stages - 1 - vs)
            schedule.append((t, physical_gpu, m, "bwd"))

    return schedule


def draw_pipeline_timeline(num_stages, num_microbatches, schedule="1f1b",
                           title=None, figsize=None):
    """Draw a pipeline parallelism execution timeline.

    Renders a grid where rows = GPU stages, columns = time steps, and colored
    blocks represent forward (blue) and backward (orange) micro-batch computations.

    Args:
        num_stages: Number of pipeline stages (GPUs).
        num_microbatches: Number of micro-batches.
        schedule: Scheduling strategy ("gpipe", "1f1b", "interleaved").
        title: Plot title. Defaults to schedule name.
        figsize: Figure size tuple. Auto-calculated if None.

    Returns:
        Tuple of (fig, ax) for further customization.
    """
    if title is None:
        names = {"gpipe": "GPipe", "1f1b": "1F1B", "interleaved": "Interleaved"}
        title = f"{names.get(schedule, schedule)} Schedule — {num_stages} stages, {num_microbatches} micro-batches"

    # Generate schedule
    if schedule == "gpipe":
        sched = _generate_gpipe_schedule(num_stages, num_microbatches)
    elif schedule == "1f1b":
        sched = _generate_1f1b_schedule(num_stages, num_microbatches)
    elif schedule == "interleaved":
        sched = _generate_interleaved_schedule(num_stages, num_microbatches)
    else:
        raise ValueError(f"Unknown schedule: {schedule}. Use 'gpipe', '1f1b', or 'interleaved'.")

    # Determine grid dimensions
    max_time = max(t for t, _, _, _ in sched) + 1
    if figsize is None:
        figsize = (max(10, max_time * 0.6), max(3, num_stages * 1.0))

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Color maps
    fwd_cmap = plt.cm.Blues
    bwd_cmap = plt.cm.Oranges

    # Draw blocks
    for t, s, m, kind in sched:
        color_val = 0.3 + 0.5 * (m % num_microbatches) / max(num_microbatches - 1, 1)
        if kind == "fwd":
            color = fwd_cmap(color_val)
        else:
            color = bwd_cmap(color_val)

        rect = patches.FancyBboxPatch(
            (t + 0.05, s + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", linewidth=1.0
        )
        ax.add_patch(rect)

        # Label: F0, B0, etc.
        label = f"{'F' if kind == 'fwd' else 'B'}{m}"
        ax.text(t + 0.5, s + 0.5, label, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white" if color_val > 0.5 else "black")

    # Shade bubble (empty) cells lightly
    occupied = set()
    for t, s, m, kind in sched:
        occupied.add((t, s))
    for t in range(max_time):
        for s in range(num_stages):
            if (t, s) not in occupied:
                rect = patches.Rectangle(
                    (t, s), 1, 1,
                    facecolor="#f0f0f0", edgecolor="#e0e0e0", linewidth=0.5
                )
                ax.add_patch(rect)

    # Formatting
    ax.set_xlim(0, max_time)
    ax.set_ylim(0, num_stages)
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("GPU / Stage", fontsize=11)
    ax.set_yticks([s + 0.5 for s in range(num_stages)])
    ax.set_yticklabels([f"GPU {s}" for s in range(num_stages)], fontsize=9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")

    # Legend
    fwd_patch = patches.Patch(facecolor=fwd_cmap(0.5), label="Forward")
    bwd_patch = patches.Patch(facecolor=bwd_cmap(0.5), label="Backward")
    bubble_patch = patches.Patch(facecolor="#f0f0f0", edgecolor="#e0e0e0", label="Bubble (idle)")
    ax.legend(handles=[fwd_patch, bwd_patch, bubble_patch], loc="upper right",
              fontsize=8, framealpha=0.9)

    plt.tight_layout()
    return fig, ax


def draw_data_flow(stage_labels, data_snapshots, title=None, figsize=None):
    """Draw a data flow diagram showing tensors moving through pipeline stages.

    Each stage is a box, with arrows between them. Below each arrow, the tensor
    values (or a summary) are shown as a small heatmap or value grid.

    Args:
        stage_labels: List of strings, one per stage (e.g., ["GPU 0\\nLayers 0-1", ...]).
        data_snapshots: List of tensors to show between/at stages. Length = len(stage_labels).
            First element is the input, rest are outputs of each stage.
        title: Optional plot title.
        figsize: Optional figure size.

    Returns:
        Tuple of (fig, axes).
    """
    n_stages = len(stage_labels)
    n_snapshots = len(data_snapshots)
    if figsize is None:
        figsize = (3.5 * n_stages, 4)

    fig, axes = plt.subplots(1, n_stages, figsize=figsize)
    if n_stages == 1:
        axes = [axes]

    gpu_colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373",
                  "#BA68C8", "#4DB6AC", "#FF8A65", "#A1887F"]

    for i, ax in enumerate(axes):
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 3.5)
        ax.axis("off")

        # Stage box
        color = gpu_colors[i % len(gpu_colors)]
        rect = patches.FancyBboxPatch(
            (0.0, 1.8), 1.0, 1.2,
            boxstyle="round,pad=0.08", facecolor=color, edgecolor="gray",
            linewidth=1.5, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(0.5, 2.4, stage_labels[i], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

        # Show tensor snapshot below the stage box
        if i < n_snapshots:
            t = data_snapshots[i]
            if hasattr(t, 'numpy'):
                arr = t.detach().cpu().numpy()
            else:
                arr = np.array(t)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # Truncate display to at most 4x6
            arr_show = arr[:4, :6]
            rows, cols = arr_show.shape

            cell_w = min(0.9 / cols, 0.18)
            cell_h = min(1.2 / rows, 0.28)
            x_start = 0.5 - (cols * cell_w) / 2
            y_start = 1.4 - rows * cell_h

            for r in range(rows):
                for c in range(cols):
                    val = arr_show[r, c]
                    # Color: blue for positive, red for negative
                    intensity = min(abs(val) / (np.abs(arr_show).max() + 1e-8), 1.0)
                    if val >= 0:
                        fc = (0.7 - 0.5 * intensity, 0.8 - 0.3 * intensity, 1.0, 0.8)
                    else:
                        fc = (1.0, 0.8 - 0.5 * intensity, 0.7 - 0.5 * intensity, 0.8)
                    cell_rect = patches.Rectangle(
                        (x_start + c * cell_w, y_start + (rows - 1 - r) * cell_h),
                        cell_w * 0.95, cell_h * 0.9,
                        facecolor=fc, edgecolor="#aaa", linewidth=0.5
                    )
                    ax.add_patch(cell_rect)
                    ax.text(x_start + c * cell_w + cell_w * 0.47,
                            y_start + (rows - 1 - r) * cell_h + cell_h * 0.45,
                            f"{val:.1f}", ha="center", va="center", fontsize=5.5)

            shape_str = "×".join(str(s) for s in t.shape) if hasattr(t, 'shape') else ""
            ax.text(0.5, y_start - 0.15, f"shape: ({shape_str})",
                    ha="center", va="top", fontsize=7, color="gray")

        # Arrow to next stage
        if i < n_stages - 1:
            ax.annotate("", xy=(1.4, 2.4), xytext=(1.05, 2.4),
                        arrowprops=dict(arrowstyle="->,head_width=0.12",
                                        color="#666", lw=1.5))

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig, axes


def draw_naive_vs_pipeline(num_stages, num_microbatches=1, title=None, figsize=None):
    """Draw side-by-side comparison: naive sequential vs pipelined execution.

    Shows which GPU is active at each time step. Naive has 1 GPU active;
    pipelined fills the pipeline with micro-batches.

    Args:
        num_stages: Number of pipeline stages.
        num_microbatches: Number of micro-batches (1 = naive).
        title: Optional title.
        figsize: Optional figure size.

    Returns:
        Tuple of (fig, axes).
    """
    if figsize is None:
        figsize = (12, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    gpu_colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373"]

    # --- Naive (no micro-batching) ---
    total_naive = num_stages
    for t in range(total_naive):
        for s in range(num_stages):
            if s == t:
                rect = patches.FancyBboxPatch(
                    (t + 0.05, s + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=gpu_colors[s % 4], edgecolor="white", lw=1.0, alpha=0.85
                )
                ax1.add_patch(rect)
                ax1.text(t + 0.5, s + 0.5, "F", ha="center", va="center",
                         fontsize=10, fontweight="bold", color="white")
            else:
                rect = patches.Rectangle((t, s), 1, 1, facecolor="#f5f5f5",
                                         edgecolor="#e0e0e0", lw=0.5)
                ax1.add_patch(rect)
                ax1.text(t + 0.5, s + 0.5, "idle", ha="center", va="center",
                         fontsize=7, color="#bbb")

    ax1.set_xlim(0, total_naive)
    ax1.set_ylim(0, num_stages)
    ax1.invert_yaxis()
    ax1.set_yticks([s + 0.5 for s in range(num_stages)])
    ax1.set_yticklabels([f"GPU {s}" for s in range(num_stages)], fontsize=9)
    ax1.set_xlabel("Time Step", fontsize=10)
    ax1.set_title("Naive: No Micro-batching", fontsize=11, fontweight="bold")
    ax1.set_aspect("equal")

    # --- Pipelined ---
    total_pipe = num_stages + num_microbatches - 1
    for t in range(total_pipe):
        for s in range(num_stages):
            m = t - s
            if 0 <= m < num_microbatches:
                alpha = 0.5 + 0.4 * (m / max(num_microbatches - 1, 1))
                rect = patches.FancyBboxPatch(
                    (t + 0.05, s + 0.05), 0.9, 0.9,
                    boxstyle="round,pad=0.02",
                    facecolor=gpu_colors[s % 4], edgecolor="white", lw=1.0, alpha=alpha
                )
                ax2.add_patch(rect)
                ax2.text(t + 0.5, s + 0.5, f"F{m}", ha="center", va="center",
                         fontsize=8, fontweight="bold", color="white")
            else:
                rect = patches.Rectangle((t, s), 1, 1, facecolor="#f5f5f5",
                                         edgecolor="#e0e0e0", lw=0.5)
                ax2.add_patch(rect)

    ax2.set_xlim(0, total_pipe)
    ax2.set_ylim(0, num_stages)
    ax2.invert_yaxis()
    ax2.set_yticks([s + 0.5 for s in range(num_stages)])
    ax2.set_yticklabels([f"GPU {s}" for s in range(num_stages)], fontsize=9)
    ax2.set_xlabel("Time Step", fontsize=10)
    ax2.set_title(f"Pipelined: {num_microbatches} Micro-batches", fontsize=11, fontweight="bold")
    ax2.set_aspect("equal")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    return fig, (ax1, ax2)


def draw_comm_pattern(pattern, num_gpus, title=None, figsize=(8, 6)):
    """Draw a GPU communication pattern diagram.

    Args:
        pattern: One of "allreduce", "allgather", "reduce_scatter",
                 "broadcast", "all_to_all", "ring".
        num_gpus: Number of GPUs to show.
        title: Plot title. Defaults to pattern name.
        figsize: Figure size tuple.

    Returns:
        Tuple of (fig, ax) for further customization.
    """
    if title is None:
        title = f"{pattern.replace('_', ' ').title()} — {num_gpus} GPUs"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)

    angles = np.linspace(0, 2 * np.pi, num_gpus, endpoint=False)
    angles = np.pi / 2 - angles
    radius = 1.0
    positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

    node_radius = 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, num_gpus))
    for i, (x, y) in enumerate(positions):
        circle = plt.Circle((x, y), node_radius, color=colors[i], ec="black", lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f"GPU\n{i}", ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

    arrow_base = dict(
        arrowstyle="->,head_width=0.08,head_length=0.06",
        lw=1.5, connectionstyle="arc3,rad=0.15", zorder=2,
    )

    def _arrow(src, dst, color="#444444"):
        xi, yi = positions[src]
        xj, yj = positions[dst]
        dx, dy = xj - xi, yj - yi
        d = np.sqrt(dx**2 + dy**2)
        s = node_radius / d
        ax.annotate("", xy=(xj - dx*s, yj - dy*s), xytext=(xi + dx*s, yi + dy*s),
                     arrowprops=dict(**arrow_base, color=color))

    if pattern == "allreduce":
        for i in range(num_gpus):
            j = (i + 1) % num_gpus
            _arrow(i, j, "#2196F3")
            _arrow(j, i, "#F44336")
        ax.plot([], [], color="#2196F3", label="Reduce (sum gradients)", lw=2)
        ax.plot([], [], color="#F44336", label="Broadcast (distribute result)", lw=2)
        ax.legend(loc="lower center", fontsize=9, ncol=2, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))
    elif pattern == "broadcast":
        for i in range(1, num_gpus):
            _arrow(0, i)
    elif pattern == "reduce_scatter":
        for i in range(num_gpus):
            _arrow(i, (i + 1) % num_gpus)
    elif pattern == "allgather":
        for i in range(num_gpus):
            _arrow(i, (i - 1) % num_gpus, "#4CAF50")
    else:
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i != j:
                    _arrow(i, j)

    plt.tight_layout()
    return fig, ax


def draw_ring_attention_steps(num_gpus, num_steps=None, title="Ring Attention Steps"):
    """Draw GPUs arranged in a ring with arrows showing KV block transfers.

    Each subplot shows one ring step, with GPU nodes in a circle and
    directional arrows indicating which KV block each GPU holds.

    Args:
        num_gpus: Number of GPUs to show.
        num_steps: Number of ring steps to display (default: min(num_gpus, 3)).
        title: Overall figure title.
    """
    if num_steps is None:
        num_steps = min(num_gpus, 3)

    fig, axes = plt.subplots(1, num_steps, figsize=(5 * num_steps, 5))
    if num_steps == 1:
        axes = [axes]

    # GPU positions in a circle
    angles = np.linspace(0, 2 * np.pi, num_gpus, endpoint=False) - np.pi / 2
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    colors = plt.cm.Set2(np.linspace(0, 1, num_gpus))

    for step, ax in enumerate(axes):
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal")
        ax.set_title(f"Step {step}", fontsize=13, fontweight="bold")
        ax.axis("off")

        # Draw ring arrows (KV transfer direction)
        for i in range(num_gpus):
            j = (i + 1) % num_gpus
            dx = x_pos[j] - x_pos[i]
            dy = y_pos[j] - y_pos[i]
            length = np.sqrt(dx**2 + dy**2)
            # Shorten arrow to not overlap with nodes
            shrink = 0.3 / length
            ax.annotate(
                "",
                xy=(x_pos[j] - dx * shrink, y_pos[j] - dy * shrink),
                xytext=(x_pos[i] + dx * shrink, y_pos[i] + dy * shrink),
                arrowprops=dict(arrowstyle="->", color="#888", lw=1.5),
            )

        # Draw GPU nodes with KV block labels
        for i in range(num_gpus):
            # Which KV chunk does GPU i hold at this step?
            kv_src = (i + step) % num_gpus
            ax.scatter(x_pos[i], y_pos[i], s=800, c=[colors[i]],
                       edgecolors="black", linewidths=1.5, zorder=3)
            ax.text(x_pos[i], y_pos[i] + 0.05, f"GPU {i}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.text(x_pos[i], y_pos[i] - 0.1, f"KV{kv_src}",
                    ha="center", va="top", fontsize=9, color="#444")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


def draw_attention_heatmap(scores, title="Attention Scores", chunk_boundaries=None,
                           token_labels=None, ax=None, cmap="Blues", annotate=True):
    """Draw a heatmap of attention scores with optional chunk boundary lines.

    Args:
        scores: 2D numpy array or torch tensor of attention scores/weights.
        title: Plot title.
        chunk_boundaries: List of positions to draw chunk divider lines.
        token_labels: Labels for rows/columns.
        ax: Matplotlib axes to draw on (creates new figure if None).
        cmap: Colormap name.
        annotate: If True and matrix is small (<= 12), show values in cells.
    """
    if hasattr(scores, 'numpy'):
        scores = scores.detach().numpy()
    scores = np.array(scores)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(max(4, scores.shape[1] * 0.7),
                                         max(3, scores.shape[0] * 0.6)))

    im = ax.imshow(scores, cmap=cmap, aspect="auto")

    if annotate and max(scores.shape) <= 12:
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                val = scores[i, j]
                color = "white" if val > (scores.max() + scores.min()) / 2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    if chunk_boundaries:
        for b in chunk_boundaries:
            ax.axhline(b - 0.5, color="red", linewidth=2, linestyle="--")
            ax.axvline(b - 0.5, color="red", linewidth=2, linestyle="--")

    if token_labels:
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, fontsize=8)
        ax.set_yticks(range(len(token_labels)))
        ax.set_yticklabels(token_labels, fontsize=8)
    ax.set_xlabel("Key position", fontsize=10)
    ax.set_ylabel("Query position", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    if own_fig:
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.show()
    return im


def draw_tensor_blocks(blocks, gpu_labels=None, title="Tensor Blocks per GPU",
                        highlight_gpu=None, cmap="Set2"):
    """Visualize tensor values distributed across GPUs as colored blocks.

    Shows each GPU's tensor chunk as a colored grid with actual values,
    making it easy to see what data each GPU holds.

    Args:
        blocks: List of 1D or 2D numpy arrays / torch tensors, one per GPU.
        gpu_labels: Labels for each GPU (default: "GPU 0", "GPU 1", ...).
        title: Overall title.
        highlight_gpu: Index of GPU to highlight with a border.
        cmap: Colormap for GPU coloring.
    """
    n = len(blocks)
    if gpu_labels is None:
        gpu_labels = [f"GPU {i}" for i in range(n)]

    # Convert to numpy
    arrs = []
    for b in blocks:
        if hasattr(b, 'numpy'):
            b = b.detach().numpy()
        b = np.atleast_2d(np.array(b, dtype=float))
        arrs.append(b)

    gpu_colors = colormaps.get_cmap(cmap)(np.linspace(0.1, 0.9, n))

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, max(2, arrs[0].shape[0] * 0.6 + 1)))
    if n == 1:
        axes = [axes]

    for i, (arr, ax) in enumerate(zip(arrs, axes)):
        rows, cols = arr.shape
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_aspect("equal")

        for r in range(rows):
            for c in range(cols):
                color = gpu_colors[i]
                if highlight_gpu is not None and i == highlight_gpu:
                    edgecolor, lw = "red", 3
                else:
                    edgecolor, lw = "#555", 1
                rect = mpatches.FancyBboxPatch(
                    (c - 0.45, r - 0.45), 0.9, 0.9,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor=edgecolor, linewidth=lw
                )
                ax.add_patch(rect)
                ax.text(c, r, f"{arr[r, c]:.1f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")

        ax.set_title(gpu_labels[i], fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def draw_ring_step_dataflow(q_chunks, k_chunks, v_chunks, scores_tiles,
                             output_tiles, step, num_gpus, title=None):
    """Visualize one ring attention step: show Q chunks, current KV, scores, and output.

    Draws a 4-column layout per GPU for one ring step:
    Q_i | K_current | scores tile | partial output

    Args:
        q_chunks: List of Q chunk tensors per GPU.
        k_chunks: List of current K block tensors per GPU (after rotation).
        v_chunks: List of current V block tensors per GPU.
        scores_tiles: List of score matrices per GPU for this step.
        output_tiles: List of partial output tensors per GPU.
        step: Ring step number.
        num_gpus: Number of GPUs.
        title: Optional title override.
    """
    if title is None:
        title = f"Ring Step {step}: Data Flow"

    fig, axes = plt.subplots(num_gpus, 4, figsize=(14, 2.5 * num_gpus))
    if num_gpus == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Q (local)", "K (received)", "Scores (Q @ K^T)", "Output tile"]

    for gpu in range(num_gpus):
        kv_src = (gpu + step) % num_gpus
        data = [q_chunks[gpu], k_chunks[gpu], scores_tiles[gpu], output_tiles[gpu]]

        for col, (d, ct) in enumerate(zip(data, col_titles)):
            ax = axes[gpu, col]
            if hasattr(d, 'numpy'):
                d = d.detach().numpy()
            d = np.atleast_2d(np.array(d, dtype=float))

            cmap_name = "Blues" if col < 2 else ("Oranges" if col == 2 else "Greens")
            im = ax.imshow(d, cmap=cmap_name, aspect="auto")

            if max(d.shape) <= 8:
                for r in range(d.shape[0]):
                    for c in range(d.shape[1]):
                        val = d[r, c]
                        color = "white" if val > (d.max() + d.min()) / 2 else "black"
                        ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                                fontsize=7, color=color)

            if gpu == 0:
                ax.set_title(ct, fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                label = f"GPU {gpu}"
                if col == 1:
                    label += f"\n(KV from {kv_src})"
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
            if col == 1:
                ax.set_xlabel(f"from chunk {kv_src}", fontsize=8, fontstyle="italic")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def draw_context_partition(token_labels, num_gpus, q_chunks=None, k_chunks=None,
                            v_chunks=None, title="Context Partitioning Across GPUs"):
    """Visualize how a sequence of tokens is split across GPUs for Context Parallelism.

    Shows:
    - Top row: the full sequence as a colored bar of tokens
    - Bottom rows: each GPU's chunk with Q, K, V tensor values (if provided)

    Args:
        token_labels: List of token strings (e.g., ["The", "cat", "sat", ...]).
        num_gpus: Number of GPUs.
        q_chunks: Optional list of Q chunk tensors (one per GPU).
        k_chunks: Optional list of K chunk tensors (one per GPU).
        v_chunks: Optional list of V chunk tensors (one per GPU).
        title: Plot title.
    """
    S = len(token_labels)
    chunk_size = S // num_gpus
    gpu_colors = colormaps.get_cmap("Set2")(np.linspace(0.1, 0.9, num_gpus))

    has_tensors = q_chunks is not None
    n_rows = 1 + num_gpus if not has_tensors else 1 + num_gpus
    height_ratios = [1.2] + [1.5 if has_tensors else 1.0] * num_gpus

    fig, axes = plt.subplots(n_rows, 1, figsize=(max(10, S * 0.9), 1.2 + num_gpus * (2.0 if has_tensors else 1.2)),
                              gridspec_kw={"height_ratios": height_ratios})

    # --- Top row: full sequence ---
    ax = axes[0]
    ax.set_xlim(-0.5, S - 0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.set_title("Full Sequence (input)", fontsize=12, fontweight="bold")

    for i, tok in enumerate(token_labels):
        gpu_idx = i // chunk_size
        if gpu_idx >= num_gpus:
            gpu_idx = num_gpus - 1
        color = gpu_colors[gpu_idx]
        rect = mpatches.FancyBboxPatch(
            (i - 0.45, -0.4), 0.9, 0.8,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#333", linewidth=1.2
        )
        ax.add_patch(rect)
        ax.text(i, 0, tok, ha="center", va="center", fontsize=9, fontweight="bold")

    # Draw chunk boundary markers
    for g in range(1, num_gpus):
        bx = g * chunk_size - 0.5
        ax.axvline(bx, color="red", linewidth=2, linestyle="--", ymin=0.1, ymax=0.9)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # --- Per-GPU rows ---
    for g in range(num_gpus):
        ax = axes[1 + g]
        start = g * chunk_size
        end = start + chunk_size
        my_tokens = token_labels[start:end]
        color = gpu_colors[g]

        ax.set_xlim(-0.5, chunk_size - 0.5 + (4.5 if has_tensors else 0))
        y_lo, y_hi = -0.6, 0.6
        if has_tensors and q_chunks is not None:
            n_dims = q_chunks[g].shape[-1] if hasattr(q_chunks[g], 'shape') else 1
            y_hi = max(0.6, n_dims * 0.45)
            y_lo = -0.6

        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")

        # GPU label
        ax.set_ylabel(f"GPU {g}", fontsize=11, fontweight="bold", rotation=0,
                       labelpad=40, va="center")

        # Token boxes
        for j, tok in enumerate(my_tokens):
            rect = mpatches.FancyBboxPatch(
                (j - 0.45, -0.4), 0.9, 0.8,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="#333", linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(j, 0, tok, ha="center", va="center", fontsize=9, fontweight="bold")

        # If tensors provided, show Q values as a mini heatmap to the right
        if has_tensors and q_chunks is not None:
            q_np = q_chunks[g].detach().numpy() if hasattr(q_chunks[g], 'numpy') else np.array(q_chunks[g])
            q_np = np.atleast_2d(q_np)
            rows, cols = q_np.shape
            x_off = chunk_size + 0.5
            # Label
            ax.text(x_off + cols / 2 - 0.5, -0.55, f"Q chunk (shape {rows}x{cols})",
                    ha="center", fontsize=8, fontstyle="italic")
            for r in range(min(rows, chunk_size)):
                for c in range(cols):
                    intensity = (q_np[r, c] - q_np.min()) / (q_np.max() - q_np.min() + 1e-8)
                    cell_color = plt.cm.Blues(0.2 + 0.6 * intensity)
                    rect = mpatches.FancyBboxPatch(
                        (x_off + c - 0.4, r - 0.4), 0.8, 0.8,
                        boxstyle="round,pad=0.02",
                        facecolor=cell_color, edgecolor="#aaa", linewidth=0.5
                    )
                    ax.add_patch(rect)
                    ax.text(x_off + c, r, f"{q_np[r, c]:.1f}", ha="center", va="center",
                            fontsize=6, color="#333")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Legend
    legend_handles = [mpatches.Patch(facecolor=gpu_colors[g], edgecolor="#333",
                      label=f"GPU {g}: tokens {g*chunk_size}-{(g+1)*chunk_size-1}")
                      for g in range(num_gpus)]
    axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8,
                    bbox_to_anchor=(1.0, -0.1), ncol=num_gpus)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def draw_cp_memory_scaling(seq_lengths, max_gpus=8, title="CP Memory Scaling"):
    """Plot per-GPU memory usage vs number of GPUs for standard vs CP attention.

    Shows how Context Parallelism reduces per-GPU attention memory from
    O(S^2) to O(S^2/N).

    Args:
        seq_lengths: List of sequence lengths to plot.
        max_gpus: Maximum number of GPUs on x-axis.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    gpu_counts = np.arange(1, max_gpus + 1)
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(seq_lengths)))
    linestyles = ["-", "--", "-."]

    for idx, seq_len in enumerate(seq_lengths):
        # Standard attention: memory is S^2 regardless of GPU count (replicated)
        standard_mem = np.full_like(gpu_counts, seq_len**2, dtype=float)
        # CP: attention memory splits as S^2 / N per GPU
        cp_mem = seq_len**2 / gpu_counts

        # Normalize to millions for readability
        scale = 1e6
        ls = linestyles[idx % len(linestyles)]
        ax.plot(gpu_counts, standard_mem / scale, ls, color=colors[idx],
                alpha=0.4, label=f"Standard (S={seq_len})")
        ax.plot(gpu_counts, cp_mem / scale, ls, color=colors[idx],
                linewidth=2, marker="o", markersize=5,
                label=f"CP (S={seq_len})")

    ax.set_xlabel("Number of GPUs", fontsize=12)
    ax.set_ylabel("Attention Memory per GPU (M elements)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(gpu_counts)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
