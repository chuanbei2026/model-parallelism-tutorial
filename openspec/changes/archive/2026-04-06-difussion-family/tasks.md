## 1. Project Setup

- [x] 1.1 Create `notebooks/{en,zh,lzh}/diffusion/` directory structure
- [x] 1.2 Create `mp_tutorial/diffusion.py` with diffusion simulation helpers (noise schedules, forward/reverse process, simple denoiser, DDIM/DPM samplers, loss functions, UNet/DiT building blocks)
- [x] 1.3 Create `mp_tutorial/diffusion_viz.py` with diffusion-specific visualization helpers (denoising trajectories, noise schedule plots, architecture diagrams, training curve simulators)

## 2. Notebook 00: DDPM Foundations

- [x] 2.1 Create `notebooks/en/diffusion/00-ddpm-foundations.ipynb` — forward process, noise schedules (linear/cosine), training objective derivation, training on 2D data, ancestral sampling, engineering tips
- [x] 2.2 Translate to `notebooks/zh/diffusion/00-ddpm-foundations.ipynb`
- [x] 2.3 Translate to `notebooks/lzh/diffusion/00-ddpm-foundations.ipynb`

## 3. Notebook 01: Sampling Acceleration

- [x] 3.1 Create `notebooks/en/diffusion/01-sampling-acceleration.ipynb` — DDIM derivation, DPM-Solver, progressive distillation, consistency models, production sampler selection guide
- [x] 3.2 Translate to `notebooks/zh/diffusion/01-sampling-acceleration.ipynb`
- [x] 3.3 Translate to `notebooks/lzh/diffusion/01-sampling-acceleration.ipynb`

## 4. Notebook 02: Classifier-Free Guidance

- [x] 4.1 Create `notebooks/en/diffusion/02-classifier-free-guidance.ipynb` — conditional generation, classifier guidance, CFG derivation, guidance scale effects, batched CFG engineering, negative prompts
- [x] 4.2 Translate to `notebooks/zh/diffusion/02-classifier-free-guidance.ipynb`
- [x] 4.3 Translate to `notebooks/lzh/diffusion/02-classifier-free-guidance.ipynb`

## 5. Notebook 03: Latent Diffusion

- [x] 5.1 Create `notebooks/en/diffusion/03-latent-diffusion.ipynb` — pixel vs latent motivation, VAE architecture and pitfalls, UNet for LDM, cross-attention text conditioning, Stable Diffusion full pipeline
- [x] 5.2 Translate to `notebooks/zh/diffusion/03-latent-diffusion.ipynb`
- [x] 5.3 Translate to `notebooks/lzh/diffusion/03-latent-diffusion.ipynb`

## 6. Notebook 04: DiT Architecture

- [x] 6.1 Create `notebooks/en/diffusion/04-dit-architecture.ipynb` — UNet→Transformer motivation, patchification, AdaLN-Zero, scaling analysis, engineering advantages for video/parallelism
- [x] 6.2 Translate to `notebooks/zh/diffusion/04-dit-architecture.ipynb`
- [x] 6.3 Translate to `notebooks/lzh/diffusion/04-dit-architecture.ipynb`

## 7. Notebook 05: Video Generation

- [x] 7.1 Create `notebooks/en/diffusion/05-video-generation.ipynb` — video as temporal extension, memory analysis, temporal attention architectures, Sora/CogVideo/HunyuanVideo analysis, progressive training, temporal consistency
- [x] 7.2 Translate to `notebooks/zh/diffusion/05-video-generation.ipynb`
- [x] 7.3 Translate to `notebooks/lzh/diffusion/05-video-generation.ipynb`

## 8. Notebook 06: Training Engineering

- [x] 8.1 Create `notebooks/en/diffusion/06-training-engineering.ipynb` — noise schedule design, loss weighting (min-SNR, P2), prediction targets (eps/x0/v), EMA subtleties, mixed precision pitfalls, failure diagnosis guide
- [x] 8.2 Translate to `notebooks/zh/diffusion/06-training-engineering.ipynb`
- [x] 8.3 Translate to `notebooks/lzh/diffusion/06-training-engineering.ipynb`

## 9. Notebook 07: Fine-tuning & Adaptation

- [x] 9.1 Create `notebooks/en/diffusion/07-finetuning-adaptation.ipynb` — ControlNet architecture, LoRA for diffusion, DreamBooth (overfitting traps), data pipeline engineering, fine-tuning failure modes
- [x] 9.2 Translate to `notebooks/zh/diffusion/07-finetuning-adaptation.ipynb`
- [x] 9.3 Translate to `notebooks/lzh/diffusion/07-finetuning-adaptation.ipynb`
