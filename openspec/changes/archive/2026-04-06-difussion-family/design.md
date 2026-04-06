## Context

This project already has two notebook series: model parallelism (training, 8 notebooks) and inference optimization (9 notebooks). Both follow a consistent pattern: Python helper modules (`mp_tutorial/*.py`) provide simulation/visualization functions, notebooks in `notebooks/{en,zh,lzh}/` interleave markdown explanations with code cells using those helpers. All simulations are CPU-only.

The diffusion family series adds a third pillar covering generative models, with an engineering-first perspective: not just deriving the math, but showing how production systems are built, what breaks during training, and practical design decisions.

## Goals / Non-Goals

**Goals:**
- 8 notebooks covering diffusion models from DDPM basics to video generation and fine-tuning
- Engineering-focused: architecture diagrams, training pitfalls, debugging tips, production considerations
- Real tensor examples throughout (small-scale simulations on CPU)
- Three-language support (en, zh, lzh) matching existing series
- Consistent with existing notebook style and helper module patterns

**Non-Goals:**
- Not training actual diffusion models (no GPU required, all simulated)
- Not covering GANs, VAEs standalone, or other generative families
- Not building a usable diffusion model library — helpers are for teaching only
- Not covering deployment/serving of diffusion models (covered in inference series)

## Teaching Principles

### Beginner-friendly, shallow-to-deep
Every notebook starts from the simplest possible example and builds up. Assume the reader has basic PyTorch knowledge but zero diffusion background. Every new concept gets a plain-language explanation before any formula appears. For example: "Diffusion models learn to reverse a process of gradually adding noise — imagine slowly stirring ink into water, then learning to un-stir it."

### First-encounter explanations for all technical terms
When a term first appears (e.g., "score function", "ELBO", "classifier-free guidance", "AdaLN"), it MUST be explained in 1-2 sentences of plain language. Do not assume the reader knows what it means. Use `info_box` for key definitions that the reader might need to reference.

### Cross-links to existing series
The project already has model parallelism (training) and inference optimization notebooks. Diffusion notebooks should link back where relevant:
- When discussing distributed training of diffusion models → link to `notebooks/en/00-gpu-communication.ipynb` and data/tensor parallelism notebooks
- When discussing DiT and why Transformers scale better → link to tensor parallelism notebook
- When discussing inference optimization (fast sampling, serving) → link to `notebooks/en/inference/` series (continuous batching, KV-cache, model compilation)
- When discussing quantization of diffusion models → link to `notebooks/en/inference/05-quantization-pruning.ipynb`

Use markdown links like: `See [Tensor Parallelism](../../en/02-tensor-parallelism.ipynb) for how large models are split across GPUs.`

### Visualization-heavy
Every concept should have a corresponding visualization. Prefer showing over telling. Use matplotlib for data plots, PlantUML for architecture diagrams, and real tensor computations over pseudocode.

## Decisions

### 1. Notebook ordering: math-first then engineering-deep

Notebooks 00-02 establish mathematical foundations (DDPM, sampling, guidance). Notebooks 03-04 cover architectures (Latent Diffusion, DiT). Notebooks 05-07 focus on engineering (video generation, training, fine-tuning).

**Why:** Engineers need the mathematical vocabulary before the engineering discussion makes sense. But we front-load practical intuition even in the math notebooks (e.g., "why cosine schedule?" alongside the formula).

### 2. Helper module split: `diffusion.py` + `diffusion_viz.py`

Following the inference series pattern (`inference.py` + `inference_viz.py`):
- `diffusion.py`: noise schedules, forward/reverse process, simple UNet/DiT blocks, samplers (DDPM/DDIM/DPM-Solver), loss functions, training loop helpers
- `diffusion_viz.py`: denoising trajectory visualization, noise schedule comparison plots, architecture block diagrams, training curve simulators

**Why:** Keeps visualization separate from logic, matching project conventions.

### 3. CPU-only simulations with tiny models

All demonstrations use small tensors (e.g., 2D points, 8×8 "images", tiny UNets with <1K params). Denoising is shown on synthetic data (Swiss roll, 2D Gaussians, tiny grayscale patches).

**Why:** Notebooks must execute on CPU without GPU. Real diffusion training requires thousands of GPU hours — we simulate the dynamics at small scale to build intuition.

### 4. Video generation: architecture analysis, not training

Notebook 06 (video generation) covers temporal attention, 3D architectures, and memory analysis through diagrams and calculations, not actual video generation.

**Why:** Video diffusion is too compute-intensive to simulate meaningfully on CPU. The engineering insights (memory scaling, temporal consistency, progressive training) can be taught through analysis and calculations.

### 5. PlantUML for architecture diagrams

Use `mp_tutorial/plantuml.py` (already exists from inference series) for Stable Diffusion, DiT, ControlNet, and video model architecture diagrams.

**Why:** Reuses existing infrastructure. Complex architectures are better shown as diagrams than described in text.

## Risks / Trade-offs

- **[Risk] Diffusion math is dense** → Mitigate with heavy visualization (denoising trajectories, noise schedule plots) and building from simple 2D examples before scaling to images
- **[Risk] CPU-only limits realism** → Mitigate by showing real numbers from papers (FLOPs, memory, training time) alongside toy simulations. Use `info_box` to bridge "here's what happens at scale"
- **[Risk] Field moves fast (new architectures monthly)** → Focus on foundational concepts (noise process, guidance, latent space, transformers) that remain stable. Mention recent work in "Further Reading" sections
- **[Risk] 8 notebooks × 3 languages = 24 files** → Mitigate with parallel translation agents (proven workflow from inference series)
