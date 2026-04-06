"""Distributed simulation helpers for model parallelism tutorials.

Provides toy simulation utilities that demonstrate distributed communication
patterns WITHOUT requiring actual multi-GPU hardware. Uses CPU-based
threads/processes to simulate multi-GPU behavior for local learning.

For real GPU execution, use a machine with 4+ CUDA GPUs.
"""

import torch
import torch.nn.functional as F


def check_gpu_env():
    """Check GPU environment and print guidance.

    Detects CUDA availability and reports GPU info. If no GPU is found,
    prints instructions to connect to the remote machine.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"✓ CUDA available — {num_gpus} GPU(s) detected:")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            print(f"  [{i}] {name} ({mem:.1f} GB)")
    else:
        print("✗ No CUDA GPU detected on this machine.")
        print()
        print("To run GPU-required cells, use a machine with CUDA GPUs (4+ recommended).")
        print("See README.md for remote Jupyter setup instructions.")


def simulate_pipeline_stages(model, num_stages, micro_batches):
    """Simulate pipeline-parallel execution of a sequential model on CPU.

    Splits an nn.Sequential model into stages and runs micro-batches through
    them, printing a log of which micro-batch is at which stage at each step.

    Args:
        model: A torch.nn.Sequential model.
        num_stages: Number of pipeline stages to split the model into.
        micro_batches: List of input tensors (one per micro-batch).

    Returns:
        List of output tensors (one per micro-batch).
    """
    import torch.nn as nn

    # Split model layers into stages
    layers = list(model.children())
    layers_per_stage = len(layers) // num_stages
    if layers_per_stage == 0:
        raise ValueError(f"Model has {len(layers)} layers but {num_stages} stages requested")

    stages = []
    for i in range(num_stages):
        start = i * layers_per_stage
        end = start + layers_per_stage if i < num_stages - 1 else len(layers)
        stages.append(nn.Sequential(*layers[start:end]))

    print(f"Pipeline: {len(layers)} layers split into {num_stages} stages")
    for i, stage in enumerate(stages):
        stage_layers = len(list(stage.children()))
        print(f"  Stage {i}: {stage_layers} layer(s)")
    print()

    # Simulate pipeline execution (GPipe-style: all forwards, then all backwards)
    num_mb = len(micro_batches)
    activations = {}  # (stage, micro_batch) -> tensor

    print("=== Forward Pass ===")
    for m in range(num_mb):
        x = micro_batches[m]
        for s in range(num_stages):
            with torch.no_grad():
                x = stages[s](x)
            activations[(s, m)] = x
            print(f"  [t={m + s:2d}] Stage {s} processes micro-batch {m} → shape {tuple(x.shape)}")

    print()
    print("=== Pipeline Summary ===")
    total_steps = (num_stages - 1) + num_mb
    bubble_steps = num_stages - 1
    print(f"  Total time steps: {total_steps}")
    print(f"  Bubble steps per GPU: {bubble_steps}")
    print(f"  Bubble fraction: {bubble_steps}/{total_steps} = {bubble_steps/total_steps:.1%}")

    # Return final outputs
    outputs = [activations[(num_stages - 1, m)] for m in range(num_mb)]
    return outputs


def simulate_allreduce(tensors):
    """Simulate AllReduce on CPU using a list of tensors (one per 'GPU').

    Args:
        tensors: List of tensors, each representing one GPU's data.

    Returns:
        List of tensors, each containing the sum of all inputs.
    """
    total = torch.stack(tensors).sum(dim=0)
    return [total.clone() for _ in tensors]


def simulate_allgather(tensors):
    """Simulate AllGather on CPU.

    Args:
        tensors: List of tensors, each representing one GPU's data.

    Returns:
        List of tensors, each containing the concatenation of all inputs along dim 0.
    """
    gathered = torch.cat(tensors, dim=0)
    return [gathered.clone() for _ in tensors]


def simulate_reduce_scatter(tensors):
    """Simulate ReduceScatter on CPU.

    Sums all input tensors element-wise (reduce), then splits the result
    into equal chunks along dim 0 and distributes chunk i to GPU i (scatter).

    Args:
        tensors: List of N tensors of identical shape, one per GPU.

    Returns:
        List of N tensors, where tensor[i] is chunk i of the reduced sum.
    """
    n = len(tensors)
    total = torch.stack(tensors).sum(dim=0)
    chunks = total.chunk(n, dim=0)
    return [chunk.clone() for chunk in chunks]


def simulate_p2p_kv_exchange(kv_blocks):
    """Simulate one round of point-to-point KV block rotation around a ring.

    Each GPU sends its KV block to the next GPU in the ring:
    GPU 0 → GPU 1, GPU 1 → GPU 2, ..., GPU N-1 → GPU 0.

    Args:
        kv_blocks: List of tensors, one per GPU. Each tensor is a KV block.

    Returns:
        List of tensors after one rotation step. The block that was on GPU i
        is now on GPU (i+1) % N.
    """
    n = len(kv_blocks)
    # Rotate: each GPU receives from the previous GPU in the ring
    return [kv_blocks[(i - 1) % n] for i in range(n)]


def simulate_ring_attention(queries, keys, values, num_gpus, verbose=False):
    """Simulate the Ring Attention algorithm on CPU.

    Splits the input sequence into chunks (one per GPU), then iterates
    over ring steps. At each step, each GPU computes partial attention
    with its local Q chunk and the current KV block, then KV blocks
    rotate one position around the ring. Uses online softmax correction
    to aggregate partial attention outputs numerically stably.

    Args:
        queries: Tensor of shape (seq_len, head_dim).
        keys: Tensor of shape (seq_len, head_dim).
        values: Tensor of shape (seq_len, head_dim).
        num_gpus: Number of simulated GPUs.
        verbose: If True, print which GPU computes with which KV block
                 at each ring step.

    Returns:
        Tensor of shape (seq_len, head_dim) — the attention output,
        matching standard scaled dot-product attention (within float tolerance).
    """
    seq_len, head_dim = queries.shape
    assert seq_len % num_gpus == 0, (
        f"seq_len ({seq_len}) must be divisible by num_gpus ({num_gpus})"
    )
    chunk_size = seq_len // num_gpus
    scale = head_dim ** -0.5

    # Split Q, K, V into per-GPU chunks
    q_chunks = list(queries.split(chunk_size, dim=0))
    k_chunks = list(keys.split(chunk_size, dim=0))
    v_chunks = list(values.split(chunk_size, dim=0))

    # Each GPU starts with its own KV block
    local_k = list(k_chunks)  # current K block on each GPU
    local_v = list(v_chunks)  # current V block on each GPU

    # Online softmax aggregation per GPU (FlashAttention-style):
    # Track running max (m), sum-of-exps (l), and unnormalized output (O)
    m_acc = [torch.full((chunk_size, 1), float("-inf")) for _ in range(num_gpus)]
    l_acc = [torch.zeros(chunk_size, 1) for _ in range(num_gpus)]
    out_acc = [torch.zeros(chunk_size, head_dim) for _ in range(num_gpus)]

    for step in range(num_gpus):
        if verbose:
            for gpu_id in range(num_gpus):
                src = (gpu_id + step) % num_gpus
                print(f"  Step {step}: GPU {gpu_id} attends to KV from chunk {src}")

        for gpu_id in range(num_gpus):
            q = q_chunks[gpu_id]        # (chunk_size, head_dim)
            k = local_k[gpu_id]         # (chunk_size, head_dim)
            v = local_v[gpu_id]         # (chunk_size, head_dim)

            # Compute raw attention scores for this tile
            scores = (q @ k.T) * scale  # (chunk_size, chunk_size)

            # Per-row max for numerical stability
            tile_max = scores.max(dim=-1, keepdim=True).values  # (chunk_size, 1)
            tile_exp = torch.exp(scores - tile_max)             # (chunk_size, chunk_size)
            tile_sum = tile_exp.sum(dim=-1, keepdim=True)       # (chunk_size, 1)
            tile_out = tile_exp @ v                             # (chunk_size, head_dim)

            # Online softmax correction: merge this tile with accumulated result
            m_prev = m_acc[gpu_id]
            m_new = torch.maximum(m_prev, tile_max)
            correction_prev = torch.exp(m_prev - m_new)
            correction_tile = torch.exp(tile_max - m_new)

            l_acc[gpu_id] = correction_prev * l_acc[gpu_id] + correction_tile * tile_sum
            out_acc[gpu_id] = correction_prev * out_acc[gpu_id] + correction_tile * tile_out
            m_acc[gpu_id] = m_new

        # Rotate KV blocks one step around the ring
        local_k = simulate_p2p_kv_exchange(local_k)
        local_v = simulate_p2p_kv_exchange(local_v)

    # Normalize each GPU's output: O / l
    for gpu_id in range(num_gpus):
        out_acc[gpu_id] = out_acc[gpu_id] / l_acc[gpu_id]

    # Concatenate chunks back into full sequence
    return torch.cat(out_acc, dim=0)
