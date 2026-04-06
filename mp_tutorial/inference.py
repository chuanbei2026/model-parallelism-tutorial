"""Inference optimization simulation helpers.

Provides CPU-based simulations for KV-cache, continuous batching,
PagedAttention, quantization, and speculative decoding — used by
the inference-tricks notebook series.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# KV-Cache helpers
# ---------------------------------------------------------------------------

class SimpleAttention(nn.Module):
    """Minimal multi-head attention for demonstrating KV-cache."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            kv_cache: optional tuple (cached_k, cached_v) each (batch, n_heads, cache_len, head_dim)
        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: tuple (k, v) each (batch, n_heads, total_len, head_dim)
        """
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.W_o(out)
        return out, (k, v)


class GroupedQueryAttention(nn.Module):
    """GQA: fewer KV heads than query heads."""

    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_q_heads == 0
        assert n_q_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_q_heads
        self.n_groups = n_q_heads // n_kv_heads

        self.W_q = nn.Linear(d_model, n_q_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.n_q_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.W_o(out)
        return out, (k[:, :self.n_kv_heads], v[:, :self.n_kv_heads])


def kv_cache_memory_bytes(n_layers, n_kv_heads, head_dim, seq_len,
                          batch_size=1, dtype_bytes=2):
    """Compute KV-cache memory in bytes.

    Formula: 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes
    (factor 2 = K + V)
    """
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes


def count_attention_flops(seq_len, d_model, n_heads, cached_len=0):
    """Count approximate FLOPs for one attention forward pass.

    Without cache: Q@K^T is (S, d) x (d, S) = 2*S*S*d FLOPs, plus attn@V = 2*S*S*d.
    With cache: Q is (1, d), K is (S+1, d), so 2*1*(S+1)*d + 2*1*(S+1)*d.
    """
    if cached_len == 0:
        # Full attention
        return 4 * seq_len * seq_len * d_model
    else:
        # Cached: only 1 new query token
        total_len = cached_len + 1
        return 4 * total_len * d_model


# ---------------------------------------------------------------------------
# Continuous Batching helpers
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequest:
    """A single inference request in the serving system."""
    request_id: int
    prompt_len: int
    output_len: int
    arrival_time: int = 0
    tokens_generated: int = 0
    start_time: Optional[int] = None
    end_time: Optional[int] = None

    @property
    def is_complete(self):
        return self.tokens_generated >= self.output_len

    @property
    def is_in_prefill(self):
        return self.tokens_generated == 0 and self.start_time is not None


def simulate_static_batching(requests, max_batch_size):
    """Simulate static batching: all requests in a batch wait for the longest.

    Returns list of (timestep, active_request_ids, is_padding) tuples.
    """
    timeline = []
    pending = list(requests)
    t = 0

    while pending:
        batch = pending[:max_batch_size]
        pending = pending[max_batch_size:]

        # Find max total length
        max_output = max(r.output_len for r in batch)
        for r in batch:
            r.start_time = t

        for step in range(max_output):
            active = []
            for r in batch:
                if r.tokens_generated < r.output_len:
                    r.tokens_generated += 1
                    active.append((r.request_id, False))  # not padding
                else:
                    active.append((r.request_id, True))   # padding (wasted)
            timeline.append((t, active))
            t += 1

        for r in batch:
            r.end_time = t

    return timeline


def simulate_continuous_batching(requests, max_batch_size):
    """Simulate continuous (iteration-level) batching.

    Completed requests are evicted and new ones inserted each step.
    Returns list of (timestep, active_request_ids) tuples.
    """
    timeline = []
    pending = sorted(requests, key=lambda r: r.arrival_time)
    active_batch = []
    t = 0
    pending_idx = 0

    while pending_idx < len(pending) or active_batch:
        # Admit new requests if there's room
        while (pending_idx < len(pending) and
               len(active_batch) < max_batch_size and
               pending[pending_idx].arrival_time <= t):
            r = pending[pending_idx]
            r.start_time = t
            active_batch.append(r)
            pending_idx += 1

        if not active_batch:
            t += 1
            continue

        # One decode step
        step_info = []
        for r in active_batch:
            r.tokens_generated += 1
            step_info.append((r.request_id, False))

        timeline.append((t, step_info))

        # Evict completed
        completed = [r for r in active_batch if r.is_complete]
        for r in completed:
            r.end_time = t + 1
        active_batch = [r for r in active_batch if not r.is_complete]

        t += 1

    return timeline


# ---------------------------------------------------------------------------
# PagedAttention helpers
# ---------------------------------------------------------------------------

@dataclass
class KVBlock:
    """A physical KV-cache block."""
    block_id: int
    block_size: int  # max tokens per block
    ref_count: int = 1
    tokens_used: int = 0

    @property
    def is_full(self):
        return self.tokens_used >= self.block_size


class PagedKVCacheManager:
    """Simulate PagedAttention's block-based KV-cache management."""

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.block_pool = {}  # block_id -> KVBlock
        self.sequence_tables = {}  # seq_id -> list of block_ids (block table)

    def allocate_sequence(self, seq_id):
        """Start a new sequence with one block."""
        if not self.free_blocks:
            raise RuntimeError("Out of blocks!")
        block_id = self.free_blocks.pop(0)
        block = KVBlock(block_id=block_id, block_size=self.block_size)
        self.block_pool[block_id] = block
        self.sequence_tables[seq_id] = [block_id]
        return block_id

    def append_token(self, seq_id):
        """Add a token to a sequence, allocating a new block if needed."""
        table = self.sequence_tables[seq_id]
        last_block = self.block_pool[table[-1]]
        if last_block.is_full:
            if not self.free_blocks:
                raise RuntimeError("Out of blocks!")
            new_id = self.free_blocks.pop(0)
            new_block = KVBlock(block_id=new_id, block_size=self.block_size)
            self.block_pool[new_id] = new_block
            table.append(new_id)
            last_block = new_block
        last_block.tokens_used += 1

    def free_sequence(self, seq_id):
        """Release all blocks for a sequence."""
        for block_id in self.sequence_tables.pop(seq_id, []):
            block = self.block_pool.pop(block_id)
            block.ref_count -= 1
            if block.ref_count <= 0:
                self.free_blocks.append(block_id)

    def fork_sequence(self, src_seq_id, new_seq_id):
        """Fork a sequence (for beam search) using copy-on-write."""
        src_table = self.sequence_tables[src_seq_id]
        # Share existing blocks (increment ref count)
        new_table = list(src_table)
        for block_id in new_table:
            self.block_pool[block_id].ref_count += 1
        self.sequence_tables[new_seq_id] = new_table

    def get_block_table(self, seq_id):
        """Return the block table for a sequence."""
        return list(self.sequence_tables.get(seq_id, []))

    def memory_usage(self):
        """Return (used_blocks, total_blocks, utilization)."""
        used = self.num_blocks - len(self.free_blocks)
        return used, self.num_blocks, used / self.num_blocks

    def get_memory_map(self):
        """Return a list showing which seq owns each block (or None if free)."""
        block_map = [None] * self.num_blocks
        for seq_id, table in self.sequence_tables.items():
            for block_id in table:
                block_map[block_id] = seq_id
        return block_map


def simulate_contiguous_allocation(sequences, total_memory, max_seq_len):
    """Simulate contiguous KV-cache allocation (pre-PagedAttention).

    Each sequence reserves max_seq_len slots upfront.
    Returns memory map and waste statistics.
    """
    mem_map = [None] * total_memory
    pos = 0
    waste = 0
    allocated = {}

    for seq_id, actual_len in sequences:
        if pos + max_seq_len > total_memory:
            break
        allocated[seq_id] = (pos, max_seq_len, actual_len)
        for i in range(actual_len):
            mem_map[pos + i] = seq_id
        waste += (max_seq_len - actual_len)
        pos += max_seq_len

    return mem_map, waste, allocated


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def quantize_symmetric(tensor, bits=8):
    """Symmetric quantization: map to [-2^(b-1), 2^(b-1)-1].

    Returns (quantized_tensor, scale).
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    abs_max = tensor.abs().max()
    scale = abs_max / qmax if abs_max > 0 else torch.tensor(1.0)
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8 if bits == 8 else torch.int32)
    return quantized, scale


def dequantize_symmetric(quantized, scale):
    """Dequantize symmetric quantized tensor."""
    return quantized.float() * scale


def quantize_asymmetric(tensor, bits=8):
    """Asymmetric quantization: map to [0, 2^b - 1].

    Returns (quantized_tensor, scale, zero_point).
    """
    qmin = 0
    qmax = 2 ** bits - 1
    t_min = tensor.min()
    t_max = tensor.max()
    scale = (t_max - t_min) / (qmax - qmin) if t_max > t_min else torch.tensor(1.0)
    zero_point = torch.round(qmin - t_min / scale).clamp(qmin, qmax)
    quantized = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax).to(torch.uint8 if bits == 8 else torch.int32)
    return quantized, scale, zero_point


def dequantize_asymmetric(quantized, scale, zero_point):
    """Dequantize asymmetric quantized tensor."""
    return (quantized.float() - zero_point.float()) * scale


def simulate_gptq_error_compensation(W, H_inv, bits=4):
    """Simplified GPTQ-style quantization with Hessian-based error compensation.

    Quantizes column by column, compensating remaining columns for quantization error.
    W: (out_features, in_features) weight matrix
    H_inv: (in_features, in_features) inverse Hessian (or approximation)
    """
    W_q = W.clone()
    n_cols = W.shape[1]
    errors = []

    for i in range(n_cols):
        col = W_q[:, i]
        q_col, scale = quantize_symmetric(col, bits=bits)
        dq_col = dequantize_symmetric(q_col, scale)
        error = col - dq_col
        errors.append(error.norm().item())
        W_q[:, i] = dq_col

        # Compensate remaining columns
        if i < n_cols - 1:
            delta = error.unsqueeze(1) * H_inv[i, i+1:].unsqueeze(0) / (H_inv[i, i] + 1e-8)
            W_q[:, i+1:] += delta

    return W_q, errors


def magnitude_prune(tensor, sparsity):
    """Prune tensor by zeroing out smallest-magnitude elements.

    Args:
        tensor: weight tensor
        sparsity: fraction of weights to prune (0.0 to 1.0)

    Returns (pruned_tensor, mask).
    """
    flat = tensor.abs().flatten()
    k = int(flat.numel() * sparsity)
    if k == 0:
        return tensor.clone(), torch.ones_like(tensor, dtype=torch.bool)
    threshold = torch.topk(flat, k, largest=False).values[-1]
    mask = tensor.abs() > threshold
    return tensor * mask, mask


def structured_prune_rows(weight, sparsity):
    """Structured pruning: remove entire rows with smallest L2 norms.

    Returns (pruned_weight, kept_row_indices).
    """
    row_norms = weight.norm(dim=1)
    k = int(weight.shape[0] * sparsity)
    if k == 0:
        return weight.clone(), list(range(weight.shape[0]))
    _, indices = torch.topk(row_norms, weight.shape[0] - k)
    indices = indices.sort().values
    return weight[indices], indices.tolist()


# ---------------------------------------------------------------------------
# Speculative Decoding helpers
# ---------------------------------------------------------------------------

def speculative_decode_step(draft_probs, target_probs, bonus_probs, K):
    """Simulate one round of speculative decoding.

    Samples K draft tokens from draft_probs, verifies each against
    target_probs, and returns the accepted/rejected results.

    Args:
        draft_probs: (K, vocab_size) — draft model probabilities
        target_probs: (K, vocab_size) — target model probabilities
        bonus_probs: (1, vocab_size) — target probs for bonus position
        K: number of speculative tokens

    Returns:
        dict with keys:
            draft_tokens: (K,) sampled draft token ids
            accepted_mask: (K,) boolean acceptance mask
            final_tokens: list of accepted tokens (+ bonus if all accepted)
            n_produced: number of tokens produced this round
    """
    # Sample draft tokens
    draft_tokens = torch.multinomial(draft_probs, num_samples=1).squeeze(-1)
    accepted_mask = torch.zeros(K, dtype=torch.bool)
    final_tokens = []

    for i in range(K):
        tok = draft_tokens[i].item()
        p_val = target_probs[i, tok].item()
        q_val = draft_probs[i, tok].item()

        accept_prob = min(1.0, p_val / (q_val + 1e-10))
        r = torch.rand(1).item()

        if r < accept_prob:
            accepted_mask[i] = True
            final_tokens.append(tok)
        else:
            # Resample from adjusted distribution max(0, p - q)
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            adjusted = adjusted / (adjusted.sum() + 1e-10)
            resampled = torch.multinomial(adjusted, 1).item()
            final_tokens.append(resampled)
            break

    # Bonus token if all K accepted
    if accepted_mask.all():
        bonus = torch.multinomial(bonus_probs.squeeze(0), 1).item()
        final_tokens.append(bonus)

    return {
        "draft_tokens": draft_tokens,
        "accepted_mask": accepted_mask,
        "final_tokens": torch.tensor(final_tokens),
        "n_produced": len(final_tokens),
    }


def compute_speculative_speedup(acceptance_rate=None, K=None, cost_ratio=1.0,
                                 alpha_range=None, K_values=None,
                                 draft_cost_ratio=None):
    """Compute expected speedup from speculative decoding.

    Can be called in two modes:
    1. Single point: compute_speculative_speedup(acceptance_rate=0.8, K=4)
    2. Sweep: compute_speculative_speedup(alpha_range=(0.3, 0.99), K_values=[2,4,6,8])

    Returns float in single mode, or dict with 'alphas', 'K_values',
    'speedups', 'cost_ratio' in sweep mode.
    """
    if draft_cost_ratio is not None:
        cost_ratio = draft_cost_ratio

    def _speedup(alpha, k):
        if alpha >= 1.0:
            return k + 1
        elif alpha <= 0.0:
            return 1.0
        expected = (1 - alpha ** (k + 1)) / (1 - alpha)
        return expected / (k * cost_ratio + 1)

    # Sweep mode
    if alpha_range is not None or K_values is not None:
        if alpha_range is None:
            alpha_range = (0.01, 0.99)
        if K_values is None:
            K_values = [1, 2, 4, 8]
        import numpy as _np
        alphas = _np.linspace(alpha_range[0], alpha_range[1], 100)
        speedups = {}
        for k in K_values:
            speedups[k] = [_speedup(a, k) for a in alphas]
        return {"alphas": alphas, "K_values": K_values,
                "speedups": speedups, "cost_ratio": cost_ratio}

    # Single point mode
    return _speedup(acceptance_rate, K)


# ---------------------------------------------------------------------------
# Flash Attention helpers
# ---------------------------------------------------------------------------

def standard_attention(Q, K, V):
    """Standard attention: materializes full S×S attention matrix.

    Returns (output, attention_weights, peak_memory_elements).
    """
    S = Q.shape[0]
    d = Q.shape[1]
    scores = Q @ K.T / math.sqrt(d)       # S × S
    attn = F.softmax(scores, dim=-1)       # S × S
    output = attn @ V                       # S × d
    peak_memory = S * S  # the attention matrix
    return output, attn, peak_memory


def flash_attention_tiled(Q, K, V, block_size=2):
    """Simplified Flash Attention with tiling and online softmax.

    Processes Q in blocks, streaming K/V blocks, never materializing full S×S.

    Returns (output, peak_memory_elements, hbm_reads, hbm_writes).
    """
    S, d = Q.shape
    output = torch.zeros_like(Q)
    # Running statistics for online softmax
    row_max = torch.full((S,), float('-inf'))
    row_sum = torch.zeros(S)

    n_q_blocks = math.ceil(S / block_size)
    n_kv_blocks = math.ceil(S / block_size)
    hbm_reads = 0
    hbm_writes = 0
    peak_sram = 0

    for i in range(n_q_blocks):
        q_start = i * block_size
        q_end = min(q_start + block_size, S)
        Qi = Q[q_start:q_end]             # block_size × d
        Oi = output[q_start:q_end]
        mi = row_max[q_start:q_end]
        li = row_sum[q_start:q_end]
        hbm_reads += Qi.numel()

        for j in range(n_kv_blocks):
            kv_start = j * block_size
            kv_end = min(kv_start + block_size, S)
            Kj = K[kv_start:kv_end]       # block_size × d
            Vj = V[kv_start:kv_end]       # block_size × d
            hbm_reads += Kj.numel() + Vj.numel()

            # Compute block scores in SRAM
            Sij = Qi @ Kj.T / math.sqrt(d)   # block_size × block_size
            sram_used = Sij.numel() + Qi.numel() + Kj.numel() + Vj.numel()
            peak_sram = max(peak_sram, sram_used)

            # Online softmax update
            block_max = Sij.max(dim=-1).values
            new_max = torch.maximum(mi, block_max)
            exp_old = torch.exp(mi - new_max)
            exp_new = torch.exp(Sij - new_max.unsqueeze(-1))

            new_sum = li * exp_old + exp_new.sum(dim=-1)
            # Update output: rescale old + add new
            Oi = Oi * (li * exp_old / (new_sum + 1e-10)).unsqueeze(-1)
            Oi = Oi + (exp_new / (new_sum + 1e-10).unsqueeze(-1)) @ Vj

            mi = new_max
            li = new_sum

        output[q_start:q_end] = Oi
        row_max[q_start:q_end] = mi
        row_sum[q_start:q_end] = li
        hbm_writes += Oi.numel()

    peak_memory = peak_sram  # only SRAM tiles, never full S×S
    return output, peak_memory, hbm_reads, hbm_writes


def online_softmax_demo(values, chunk_size=2):
    """Demonstrate online softmax: process a vector in chunks.

    Returns list of dicts showing running max, running sum, and partial result at each step.
    """
    n = len(values)
    if isinstance(values, torch.Tensor):
        x = values.float()
    else:
        x = torch.tensor(values, dtype=torch.float32)

    steps = []
    running_max = torch.tensor(float('-inf'))
    running_sum = torch.tensor(0.0)

    for i in range(0, n, chunk_size):
        chunk = x[i:i + chunk_size]
        chunk_max = chunk.max()
        new_max = torch.maximum(running_max, chunk_max)
        # Rescale old sum + add new
        running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(chunk - new_max).sum()
        running_max = new_max

        # Current softmax estimate
        current_softmax = torch.exp(x[:i + chunk_size] - running_max) / running_sum

        steps.append({
            "chunk_idx": i // chunk_size,
            "chunk_values": chunk.tolist(),
            "running_max": running_max.item(),
            "running_sum": running_sum.item(),
            "partial_softmax": current_softmax.tolist(),
        })

    return steps


# ---------------------------------------------------------------------------
# Prefix Caching helpers
# ---------------------------------------------------------------------------

class RadixTreeNode:
    """A node in the radix tree for prefix caching."""

    def __init__(self):
        self.children = {}  # token_id -> RadixTreeNode
        self.block_ids = []  # physical block IDs stored at this prefix
        self.ref_count = 0

    def __repr__(self):
        return f"RadixNode(children={list(self.children.keys())}, blocks={self.block_ids}, refs={self.ref_count})"


class PrefixCache:
    """Radix tree-based prefix cache for KV blocks."""

    def __init__(self, block_size=4):
        self.root = RadixTreeNode()
        self.block_size = block_size
        self.hits = 0
        self.misses = 0

    def insert(self, token_ids, block_ids):
        """Insert a prefix (token sequence) with associated block IDs."""
        node = self.root
        block_idx = 0
        for i, tok in enumerate(token_ids):
            if tok not in node.children:
                node.children[tok] = RadixTreeNode()
            node = node.children[tok]
            # Assign block at block boundaries
            if (i + 1) % self.block_size == 0 and block_idx < len(block_ids):
                node.block_ids.append(block_ids[block_idx])
                block_idx += 1
        node.ref_count += 1

    def lookup(self, token_ids):
        """Find the longest matching prefix.

        Returns (matched_length, cached_block_ids).
        """
        node = self.root
        matched = 0
        cached_blocks = []

        for tok in token_ids:
            if tok not in node.children:
                break
            node = node.children[tok]
            matched += 1
            if node.block_ids:
                cached_blocks.extend(node.block_ids)

        if matched > 0:
            self.hits += 1
        else:
            self.misses += 1

        return matched, cached_blocks

    def get_tree_structure(self, node=None, prefix=None):
        """Return the tree as a nested dict for visualization."""
        if node is None:
            node = self.root
        if prefix is None:
            prefix = []
        result = {
            "prefix": list(prefix),
            "blocks": list(node.block_ids),
            "refs": node.ref_count,
            "children": {}
        }
        for tok, child in node.children.items():
            result["children"][tok] = self.get_tree_structure(child, prefix + [tok])
        return result


# ---------------------------------------------------------------------------
# Attention variant helpers (MHA / MQA / GQA / MLA)
# ---------------------------------------------------------------------------

def calc_kv_cache_size(variant, n_heads, n_kv_heads, head_dim, seq_len,
                       n_layers, d_compressed=None, dtype_bytes=2):
    """Compute KV-cache size in bytes for different attention variants.

    Args:
        variant: one of "mha", "mqa", "gqa", "mla"
        n_heads: number of query heads
        n_kv_heads: number of KV heads (equals n_heads for MHA, 1 for MQA)
        head_dim: dimension per head
        seq_len: sequence length
        n_layers: number of transformer layers
        d_compressed: compressed latent dimension (required for MLA)
        dtype_bytes: bytes per element (2 for fp16)

    Returns:
        int: total KV-cache memory in bytes
    """
    if variant in ("mha", "mqa", "gqa"):
        return 2 * n_layers * n_kv_heads * seq_len * head_dim * dtype_bytes
    elif variant == "mla":
        if d_compressed is None:
            raise ValueError("d_compressed is required for MLA variant")
        return n_layers * seq_len * d_compressed * dtype_bytes
    else:
        raise ValueError(f"Unknown variant: {variant}")


def attention_forward_sim(variant, x, n_heads, n_kv_heads, head_dim,
                          d_compressed=None):
    """Simplified attention forward pass returning intermediate tensors.

    Uses random projections (no learned weights) to illustrate tensor shapes
    and the dataflow for each attention variant.

    Args:
        variant: one of "mha", "mqa", "gqa", "mla"
        x: input tensor of shape (batch, seq_len, d_model)
        n_heads: number of query heads
        n_kv_heads: number of KV heads
        head_dim: dimension per head
        d_compressed: compressed latent dimension (required for MLA)

    Returns:
        dict with keys depending on variant:
            - q, k, v: projected tensors
            - scores: attention scores
            - output: attention output
            - compressed_kv (MLA only): compressed latent representation
    """
    B, S, d_model = x.shape

    torch.manual_seed(42)

    if variant in ("mha", "mqa", "gqa"):
        W_q = torch.randn(d_model, n_heads * head_dim) * (d_model ** -0.5)
        W_k = torch.randn(d_model, n_kv_heads * head_dim) * (d_model ** -0.5)
        W_v = torch.randn(d_model, n_kv_heads * head_dim) * (d_model ** -0.5)

        q = (x @ W_q).view(B, S, n_heads, head_dim).transpose(1, 2)
        k = (x @ W_k).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
        v = (x @ W_v).view(B, S, n_kv_heads, head_dim).transpose(1, 2)

        # Expand KV heads to match Q heads via repeat_interleave
        n_groups = n_heads // n_kv_heads
        k_expanded = k.repeat_interleave(n_groups, dim=1)
        v_expanded = v.repeat_interleave(n_groups, dim=1)

        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_expanded)
        out = out.transpose(1, 2).contiguous().view(B, S, n_heads * head_dim)

        return {"q": q, "k": k, "v": v, "scores": scores, "output": out}

    elif variant == "mla":
        if d_compressed is None:
            raise ValueError("d_compressed is required for MLA variant")

        W_q = torch.randn(d_model, n_heads * head_dim) * (d_model ** -0.5)
        # Down-projection: d_model -> d_compressed
        W_down = torch.randn(d_model, d_compressed) * (d_model ** -0.5)
        # Up-projections: d_compressed -> n_kv_heads * head_dim (for K and V)
        W_up_k = torch.randn(d_compressed, n_kv_heads * head_dim) * (d_compressed ** -0.5)
        W_up_v = torch.randn(d_compressed, n_kv_heads * head_dim) * (d_compressed ** -0.5)

        q = (x @ W_q).view(B, S, n_heads, head_dim).transpose(1, 2)

        # Compress: this is what gets cached
        compressed_kv = x @ W_down  # (B, S, d_compressed)

        # Decompress for attention
        k = (compressed_kv @ W_up_k).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
        v = (compressed_kv @ W_up_v).view(B, S, n_kv_heads, head_dim).transpose(1, 2)

        n_groups = n_heads // n_kv_heads
        k_expanded = k.repeat_interleave(n_groups, dim=1)
        v_expanded = v.repeat_interleave(n_groups, dim=1)

        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_expanded)
        out = out.transpose(1, 2).contiguous().view(B, S, n_heads * head_dim)

        return {"q": q, "k": k, "v": v, "scores": scores, "output": out,
                "compressed_kv": compressed_kv}

    else:
        raise ValueError(f"Unknown variant: {variant}")
