"""Diffusion model simulation helpers.

Provides CPU-based implementations of noise schedules, forward/reverse
diffusion processes, simple denoisers, samplers, loss functions, and
architecture building blocks — used by the diffusion-family notebook series.

ALL computations are CPU-only.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Noise Schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """Linear noise schedule from DDPM (Ho et al., 2020).

    Returns:
        betas: (T,) linearly spaced from beta_start to beta_end.
    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    """Cosine noise schedule from Improved DDPM (Nichol & Dhariwal, 2021).

    Uses f(t) = cos((t/T + s) / (1+s) * pi/2)^2 to derive alpha_bar,
    then converts to betas.

    Returns:
        betas: (T,) clipped to [0, 0.999].
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bars = f / f[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return betas.clamp(0.0, 0.999).float()


def compute_alpha_bars(betas):
    """Compute cumulative product of (1 - beta_t).

    Returns:
        alpha_bars: (T,) where alpha_bar_t = prod_{s=1}^{t} (1 - beta_s).
    """
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def compute_snr(alpha_bars):
    """Compute signal-to-noise ratio: SNR(t) = alpha_bar_t / (1 - alpha_bar_t).

    Higher SNR means the signal dominates; lower means noise dominates.

    Returns:
        snr: (T,) signal-to-noise ratios.
    """
    return alpha_bars / (1.0 - alpha_bars).clamp(min=1e-10)


# ---------------------------------------------------------------------------
# Forward Process
# ---------------------------------------------------------------------------

def q_sample(x_0, t, alpha_bars, noise=None):
    """Sample from q(x_t | x_0) — the forward diffusion process.

    q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

    Args:
        x_0: Clean data [B, D].
        t: Timestep indices [B].
        alpha_bars: Cumulative alpha products [T].
        noise: Optional pre-generated noise.

    Returns:
        (x_t, noise) tuple.
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    abar = alpha_bars[t]
    # Reshape for broadcasting: add trailing dims to match x_0
    while abar.dim() < x_0.dim():
        abar = abar.unsqueeze(-1)

    x_t = torch.sqrt(abar) * x_0 + torch.sqrt(1.0 - abar) * noise
    return x_t, noise


# ---------------------------------------------------------------------------
# Sinusoidal Timestep Embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional/timestep embedding (Vaswani et al., 2017).

    Maps integer timesteps to continuous vectors using sin/cos at
    geometrically spaced frequencies.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: (batch,) integer or float timesteps.
        Returns:
            emb: (batch, dim) sinusoidal embeddings.
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Simple Denoiser (MLP-based, for 2D point data)
# ---------------------------------------------------------------------------

class SimpleDenoiser(nn.Module):
    """Small MLP denoiser for 2D toy diffusion experiments.

    Accepts data (x), timestep embedding (t), and optional class label (c).
    When c is None or set to n_classes (the "null class"), the model acts
    unconditionally — enabling classifier-free guidance training.
    Uses sinusoidal time embedding for continuous timestep conditioning.
    Small enough to train on CPU in seconds.
    """

    def __init__(self, data_dim=2, hidden=128, n_classes=4, time_dim=32):
        super().__init__()
        self.n_classes = n_classes
        self.time_emb = SinusoidalPosEmb(time_dim)
        # class embedding (n_classes + 1 for null/unconditional token)
        self.class_emb = nn.Embedding(n_classes + 1, time_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x, t, c=None):
        """Predict noise.

        Args:
            x: Noisy data [B, D].
            t: Timestep indices [B].
            c: Class labels [B] or None for unconditional.

        Returns:
            eps_pred: [B, D] predicted noise.
        """
        t_emb = self.time_emb(t)
        if c is None:
            c = torch.full((x.shape[0],), self.n_classes,
                           dtype=torch.long, device=x.device)
        c_emb = self.class_emb(c)
        inp = torch.cat([x, t_emb, c_emb], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

def ddpm_sample_step(model, x_t, t, alpha_bars, betas):
    """One reverse DDPM step: x_t -> x_{t-1}.

    Uses the reparametrization from Ho et al., 2020.

    Args:
        model: Denoiser that predicts noise given (x_t, t).
        x_t: Current noisy sample (batch, ...).
        t: Current integer timestep (scalar).
        alpha_bars: Precomputed cumulative alphas (T,).
        betas: Noise schedule (T,).

    Returns:
        x_{t-1}: denoised one step.
    """
    model.eval()
    with torch.no_grad():
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.long)

        beta_t = betas[t]
        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0)
        alpha_t = 1.0 - beta_t

        eps_pred = model(x_t, t_tensor)

        # Predicted x_0
        x_0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # Posterior mean
        coeff1 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coeff2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t)
        mean = coeff1 * x_0_pred + coeff2 * x_t

        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x_t)
            return mean + sigma_t * noise
        else:
            return mean


def ddim_sample_step(model, x_t, t, t_prev, alpha_bars):
    """One deterministic DDIM step: x_t -> x_{t_prev}.

    Song et al., 2020 (DDIM) with eta=0 (deterministic).  DDIM
    skips timesteps for faster sampling while maintaining quality.

    Args:
        model: Denoiser that predicts noise given (x_t, t).
        x_t: Current noisy sample (batch, ...).
        t: Current timestep (scalar).
        t_prev: Previous timestep (scalar, t_prev < t).
        alpha_bars: Precomputed cumulative alphas (T,).

    Returns:
        x_{t_prev}: denoised sample at timestep t_prev.
    """
    model.eval()
    with torch.no_grad():
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.long)

        eps_pred = model(x_t, t_tensor)

        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predicted x_0
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * eps_pred

        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
        return x_prev


@torch.no_grad()
def ddpm_sample_loop(model, betas, alpha_bars, shape, labels=None,
                     guidance_scale=1.0, record_every=None):
    """Full DDPM reverse sampling loop with optional classifier-free guidance.

    Args:
        model: SimpleDenoiser instance.
        betas: Beta schedule [T].
        alpha_bars: Cumulative alpha products [T].
        shape: Shape of samples to generate (B, D).
        labels: Class labels for conditional generation [B] or None.
        guidance_scale: CFG weight w.  w=1.0 means no extra guidance.
        record_every: If set, record trajectory every N steps.

    Returns:
        Final samples [B, D] (and trajectory list if record_every is set).
    """
    device = betas.device
    T = len(betas)
    alphas = 1.0 - betas

    x = torch.randn(shape, device=device)
    trajectory = [x.clone().cpu()] if record_every else None

    for i in reversed(range(T)):
        t_batch = torch.full((shape[0],), i, dtype=torch.long, device=device)

        if guidance_scale != 1.0 and labels is not None:
            # Classifier-free guidance: two forward passes
            eps_cond = model(x, t_batch, labels)
            null_labels = torch.full_like(labels, model.n_classes)
            eps_uncond = model(x, t_batch, null_labels)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = model(x, t_batch, labels)

        abar = alpha_bars[i]
        abar_prev = alpha_bars[i - 1] if i > 0 else torch.tensor(1.0, device=device)
        beta = betas[i]

        # Predicted x_0
        x_0_pred = (x - torch.sqrt(1.0 - abar) * eps) / torch.sqrt(abar)
        # Posterior mean
        coeff1 = beta * torch.sqrt(abar_prev) / (1.0 - abar)
        coeff2 = (1.0 - abar_prev) * torch.sqrt(alphas[i]) / (1.0 - abar)
        mean = coeff1 * x_0_pred + coeff2 * x

        if i > 0:
            sigma = torch.sqrt(beta)
            x = mean + sigma * torch.randn_like(x)
        else:
            x = mean

        if record_every and (i % record_every == 0 or i == 0):
            trajectory.append(x.clone().cpu())

    if trajectory is not None:
        return x, trajectory
    return x


@torch.no_grad()
def ddim_sample_loop(model, shape, T, alpha_bars, steps=50):
    """DDIM sampling with fewer steps (deterministic).

    Builds a sub-sequence of timesteps and uses ddim_sample_step to
    skip intermediate timesteps, achieving faster generation.

    Args:
        model: Denoiser.
        shape: Shape of samples to generate (B, D).
        T: Total number of training timesteps.
        alpha_bars: Precomputed cumulative alphas (T,).
        steps: Number of DDIM sampling steps (< T for speedup).

    Returns:
        intermediates: list of tensors at each DDIM step.
    """
    # Build sub-sequence of timesteps
    step_size = max(T // steps, 1)
    timesteps = list(range(T - 1, -1, -step_size))
    if timesteps[-1] != 0:
        timesteps.append(0)

    x = torch.randn(shape)
    intermediates = [x.clone()]

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_prev = timesteps[i + 1]
        x = ddim_sample_step(model, x, t, t_prev, alpha_bars)
        intermediates.append(x.clone())

    return intermediates


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

def diffusion_loss(model, x_0, alpha_bars, labels=None, p_uncond=0.0,
                   loss_type="eps"):
    """Compute diffusion training loss (epsilon-prediction).

    Args:
        model: Denoiser that predicts noise given (x_t, t, c).
        x_0: Clean data [B, D].
        alpha_bars: Cumulative alpha products [T].
        labels: Class labels [B] or None.
        p_uncond: Probability of dropping the label (for CFG training).
        loss_type: "eps" for epsilon-prediction MSE.

    Returns:
        Scalar MSE loss.
    """
    B = x_0.shape[0]
    T = len(alpha_bars)
    t = torch.randint(0, T, (B,), device=x_0.device)
    x_t, noise = q_sample(x_0, t, alpha_bars)

    # Label dropout for classifier-free guidance training
    if labels is not None and p_uncond > 0 and hasattr(model, 'n_classes'):
        mask = torch.rand(B, device=x_0.device) < p_uncond
        c = labels.clone()
        c[mask] = model.n_classes  # null class
    else:
        c = labels

    pred = model(x_t, t, c)

    if loss_type == "eps":
        return F.mse_loss(pred, noise)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def v_prediction_target(x_0, noise, alpha_bars, t):
    """Compute v-prediction target (Salimans & Ho, 2022).

    v = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * x_0

    This reparametrization improves training stability, especially
    at high noise levels where epsilon-prediction can be noisy.

    Args:
        x_0: Clean data [B, D].
        noise: Sampled noise [B, D].
        alpha_bars: Precomputed cumulative alphas (T,).
        t: Timestep indices (batch,).

    Returns:
        v: [B, D] v-prediction target.
    """
    alpha_bar_t = alpha_bars[t]
    while alpha_bar_t.dim() < x_0.dim():
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)

    return torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1.0 - alpha_bar_t) * x_0


def min_snr_weight(alpha_bars, t, gamma=5.0):
    """Min-SNR loss weighting (Hang et al., 2023).

    weight(t) = min(SNR(t), gamma) / SNR(t)

    Down-weights high-SNR (low-noise) timesteps to reduce gradient
    variance during training, leading to faster convergence.

    Args:
        alpha_bars: Precomputed cumulative alphas (T,).
        t: Timestep indices (batch,).
        gamma: SNR clipping threshold.

    Returns:
        weights: (batch,) per-sample loss weights.
    """
    snr = alpha_bars[t] / (1.0 - alpha_bars[t]).clamp(min=1e-10)
    return torch.minimum(snr, torch.tensor(gamma)) / snr.clamp(min=1e-10)


# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------

def train_diffusion_2d(n_points=2000, T=200, epochs=100, lr=1e-3):
    """Train a SimpleDenoiser on 2D Swiss roll data.

    A convenience function that sets up data, schedule, model, and
    optimizer, then runs a training loop.  Completes in <10 seconds
    on CPU.

    Args:
        n_points: Number of data points.
        T: Number of diffusion timesteps.
        epochs: Training epochs.
        lr: Learning rate.

    Returns:
        (model, losses): trained model and per-epoch loss list.
    """
    x_0 = make_swiss_roll(n_points)
    betas = linear_beta_schedule(T)
    alpha_bars = compute_alpha_bars(betas)

    model = SimpleDenoiser(n_classes=0, time_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    model.train()
    for epoch in range(epochs):
        t = torch.randint(0, T, (n_points,))
        loss = diffusion_loss(model, x_0, alpha_bars)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses


def train_diffusion_2d_batched(model, data, betas, alpha_bars,
                               n_epochs=200, batch_size=256, lr=3e-4,
                               seed=42, verbose=True):
    """Train a 2D denoiser on toy data with mini-batching.

    Works with both unconditional models (forward(x, t)) and conditional
    models (forward(x, t, c=None)).

    Args:
        model: Denoiser instance.
        data: Training data tensor [N, 2].
        betas: Beta schedule [T].
        alpha_bars: Cumulative alpha products [T].
        n_epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        seed: Random seed.
        verbose: Print progress every 50 epochs.

    Returns:
        List of per-epoch average losses.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    T = len(alpha_bars)
    N = data.shape[0]
    losses = []

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            batch = data[perm[i:i + batch_size]]
            B = batch.shape[0]
            t = torch.randint(0, T, (B,), device=batch.device)
            x_t, noise = q_sample(batch, t, alpha_bars)
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:>4d}/{n_epochs} | Loss: {avg_loss:.6f}")

    return losses


# ---------------------------------------------------------------------------
# Architecture Building Blocks (for demonstration, not training)
# ---------------------------------------------------------------------------

class SimpleResBlock(nn.Module):
    """Simplified ResNet block with time conditioning.

    Adds a time-embedding projection to the residual path,
    as used in DDPM / Improved DDPM U-Nets.  The block applies
    two GroupNorm + SiLU + Conv layers with a skip connection.
    """

    def __init__(self, channels, time_dim=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, t_emb):
        """
        Args:
            x: (B, C, H, W) feature map.
            t_emb: (B, time_dim) time embedding.
        Returns:
            out: (B, C, H, W) residual-added output.
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time conditioning
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_proj

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class SimpleCrossAttention(nn.Module):
    """Cross-attention: queries from image features, keys/values from text.

    Used in Stable Diffusion's UNet for text-to-image conditioning.
    Q is derived from image token features, while K and V come from
    text encoder outputs.
    """

    def __init__(self, dim, context_dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(context_dim, dim, bias=False)
        self.W_v = nn.Linear(context_dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        """
        Args:
            x: (B, N, dim) image token features.
            context: (B, M, context_dim) text token features.
        Returns:
            out: (B, N, dim) attended output with residual.
        """
        B, N, _ = x.shape
        x_norm = self.norm(x)

        q = self.W_q(x_norm).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(context).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        out = self.W_o(out)
        return x + out


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm-Zero conditioning block (Peebles & Xie, 2023).

    Used in DiT (Diffusion Transformer) to condition transformer blocks
    on timestep and class.  Modulates LayerNorm output with learned
    shift, scale, and gate parameters.  The gate is initialized to
    zero so the block initially acts as an identity — enabling stable
    training from scratch.
    """

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        # Outputs: (shift, scale, gate) — 3 * dim total
        self.proj = nn.Linear(cond_dim, 3 * dim)
        # Initialize gate projection to zero for identity init
        nn.init.zeros_(self.proj.weight[-dim:])
        nn.init.zeros_(self.proj.bias[-dim:])

    def forward(self, x, cond):
        """
        Args:
            x: (B, N, dim) input features.
            cond: (B, cond_dim) conditioning vector (e.g., timestep emb).
        Returns:
            out: (B, N, dim) modulated output.
        """
        shift, scale, gate = self.proj(cond).unsqueeze(1).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return gate * h


class PatchEmbed(nn.Module):
    """Image to patch token embedding.

    Splits an image into non-overlapping patches and linearly projects
    each patch to a token embedding, as in ViT / DiT.
    """

    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input image.
        Returns:
            tokens: (B, n_patches, embed_dim).
        """
        x = self.proj(x)                      # (B, embed_dim, H/P, W/P)
        return x.flatten(2).transpose(1, 2)    # (B, n_patches, embed_dim)


class ZeroConv(nn.Module):
    """Zero-initialized convolution for ControlNet (Zhang et al., 2023).

    A 1x1 convolution whose weights and biases are initialized to zero,
    so the ControlNet branch initially contributes nothing and
    gradually learns residual control signals.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Utility — 2D Data Generators
# ---------------------------------------------------------------------------

def make_swiss_roll(n_points=2000, noise=0.5, seed=42):
    """Generate 2D Swiss roll data, normalized to roughly [-1, 1].

    Args:
        n_points: Number of points to generate.
        noise: Standard deviation of Gaussian noise added to the spiral.
        seed: Random seed for reproducibility.

    Returns:
        data: (n_points, 2) tensor of 2D coordinates.
    """
    rng = np.random.RandomState(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.uniform(size=n_points))
    x = t * np.cos(t) + noise * rng.randn(n_points)
    y = t * np.sin(t) + noise * rng.randn(n_points)
    data = np.column_stack([x, y]).astype(np.float32)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.from_numpy(data)


def make_2d_gaussians(n_points=2000, n_modes=4, std=0.15, radius=2.0,
                      seed=42):
    """Generate a 2D mixture-of-Gaussians dataset with class labels.

    Args:
        n_points: Total number of points.
        n_modes: Number of Gaussian clusters.
        std: Standard deviation of each cluster.
        radius: Distance of cluster centers from origin.
        seed: Random seed for reproducibility.

    Returns:
        (data, labels): data is (n_points, 2), labels is (n_points,).
    """
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius

    per_mode = n_points // n_modes
    data_list, label_list = [], []
    for c in range(n_modes):
        pts = rng.randn(per_mode, 2) * std + centers[c]
        data_list.append(pts)
        label_list.append(np.full(per_mode, c, dtype=np.int64))

    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    idx = rng.permutation(len(data))
    return (torch.tensor(data[idx], dtype=torch.float32),
            torch.tensor(labels[idx]))
