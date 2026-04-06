## Why

The existing tutorial series covers model parallelism (training) and inference optimization, but does not cover **generative model architectures** — the family of models behind modern image and video generation (Stable Diffusion, DALL-E, Sora, HunyuanVideo). Diffusion models have become essential knowledge for ML engineers. Beyond theory, engineers need to understand the **engineering architecture** (how these systems are built in production), **training pitfalls** (common issues that waste GPU hours), and **practical design decisions** that papers don't cover.

## What Changes

- Add a new `notebooks/{en,zh,lzh}/diffusion/` folder with ~8 tutorial notebooks
- Add `mp_tutorial/diffusion.py` with diffusion-specific helpers (noise schedules, forward/reverse process, samplers, UNet/DiT components)
- Add `mp_tutorial/diffusion_viz.py` with visualization helpers (denoising trajectories, noise schedule plots, architecture diagrams, training curves)
- All notebooks follow existing conventions: en/zh/lzh translations, interleaved code and explanation, real tensor examples, matplotlib visualizations
- Focus on **engineering perspective**: not just "what is the math" but "how is this built, what breaks, and why"

## Capabilities

### New Capabilities
- `ddpm-foundations`: DDPM forward/reverse process, noise schedule (linear/cosine), training objective (simplified ELBO), step-by-step denoising with real tensors, practical: why cosine schedule works better and how to debug training divergence
- `sampling-acceleration`: DDIM deterministic sampling, DPM-Solver, distillation (progressive/consistency), engineering tradeoff: quality vs speed vs memory, how production systems choose samplers
- `classifier-free-guidance`: Conditional generation, classifier guidance vs CFG, guidance scale effects, CFG training (random label dropout), engineering: batched CFG inference (2x cost), negative prompts implementation
- `latent-diffusion`: Latent Diffusion Models (LDM), VAE encoder/decoder (training pitfalls: KL collapse, codebook), UNet architecture, text conditioning via cross-attention, Stable Diffusion full architecture walkthrough with code
- `dit-architecture`: Diffusion Transformer (DiT), replacing UNet with Transformer, AdaLN-Zero, scalability analysis, engineering: why DiT scales better for video, patchification strategies, comparison with UNet
- `video-generation`: Video diffusion models, temporal attention, 3D UNet vs temporal transformer, Sora/CogVideo/HunyuanVideo architecture analysis, engineering: memory explosion with video, temporal consistency tricks, progressive training (image→video)
- `training-engineering`: The engineering of training diffusion models: noise schedule design, loss weighting (SNR-based, min-SNR), v-prediction vs epsilon-prediction, EMA (decay schedule, when to start), mixed precision pitfalls (FP16 underflow in noise prediction), distributed training considerations, common training failures and debugging
- `finetuning-adaptation`: Fine-tuning diffusion models in practice: ControlNet (architecture, training recipe), LoRA/QLoRA for diffusion, DreamBooth (overfitting traps), IP-Adapter, training data pipeline engineering, common failure modes

### Modified Capabilities

(none)

## Impact

- New files: `mp_tutorial/diffusion.py`, `mp_tutorial/diffusion_viz.py`, 24 notebook files (8 × 3 languages)
- Dependencies: existing torch, numpy, matplotlib — no new packages required
- All simulations CPU-only (no GPU required for tutorial execution)
