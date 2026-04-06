In English | [中文版](README_zh.md)

# Deep Learning Systems Tutorial

Interactive, notebook-based tutorials on distributed training, inference optimization, and generative models. Every topic comes with formulas, diagrams, runnable code, and real-world references — in English, Chinese, and Classical Chinese.

## Modules

### 1. Model Parallelism & Distributed Training

How to scale training across GPUs: communication primitives, data/tensor/pipeline/sequence/context/expert parallelism, mixed strategies, and mixed-precision training.

| # | Topic | EN | 中文 | 文言 |
|---|-------|----|------|------|
| 00 | GPU Communication Primitives | [EN](notebooks/en/model-parallelism/00-gpu-communication.ipynb) | [中文](notebooks/zh/model-parallelism/00-gpu-communication.ipynb) | [文言](notebooks/lzh/model-parallelism/00-gpu-communication.ipynb) |
| 01 | Data Parallelism (DP) | [EN](notebooks/en/model-parallelism/01-data-parallelism.ipynb) | [中文](notebooks/zh/model-parallelism/01-data-parallelism.ipynb) | [文言](notebooks/lzh/model-parallelism/01-data-parallelism.ipynb) |
| 02 | Tensor Parallelism (TP) | [EN](notebooks/en/model-parallelism/02-tensor-parallelism.ipynb) | [中文](notebooks/zh/model-parallelism/02-tensor-parallelism.ipynb) | [文言](notebooks/lzh/model-parallelism/02-tensor-parallelism.ipynb) |
| 03 | Pipeline Parallelism (PP) | [EN](notebooks/en/model-parallelism/03-pipeline-parallelism.ipynb) | [中文](notebooks/zh/model-parallelism/03-pipeline-parallelism.ipynb) | [文言](notebooks/lzh/model-parallelism/03-pipeline-parallelism.ipynb) |
| 04 | Sequence Parallelism (SP) | [EN](notebooks/en/model-parallelism/04-sequence-parallelism.ipynb) | [中文](notebooks/zh/model-parallelism/04-sequence-parallelism.ipynb) | [文言](notebooks/lzh/model-parallelism/04-sequence-parallelism.ipynb) |
| 05 | Context Parallelism (CP) | [EN](notebooks/en/model-parallelism/05-context-parallelism.ipynb) | [中文](notebooks/zh/model-parallelism/05-context-parallelism.ipynb) | [文言](notebooks/lzh/model-parallelism/05-context-parallelism.ipynb) |
| 06 | Expert Parallelism (EP / MoE) | [EN](notebooks/en/model-parallelism/06-expert-parallelism.ipynb) | [中文](notebooks/zh/model-parallelism/06-expert-parallelism.ipynb) | [文言](notebooks/lzh/model-parallelism/06-expert-parallelism.ipynb) |
| 07 | Parallelism Mix Strategy | [EN](notebooks/en/model-parallelism/07-parallelism-mix-strategy.ipynb) | [中文](notebooks/zh/model-parallelism/07-parallelism-mix-strategy.ipynb) | [文言](notebooks/lzh/model-parallelism/07-parallelism-mix-strategy.ipynb) |
| 08 | Mixed-Precision Training | [EN](notebooks/en/model-parallelism/08-mixed-precision-training.ipynb) | [中文](notebooks/zh/model-parallelism/08-mixed-precision-training.ipynb) | [文言](notebooks/lzh/model-parallelism/08-mixed-precision-training.ipynb) |

### 2. Inference Optimization

How to serve models fast: KV cache, flash attention, continuous batching, paged attention, quantization, speculative decoding, compilation, and serving architecture.

| # | Topic | EN | 中文 | 文言 |
|---|-------|----|------|------|
| 00 | KV Cache | [EN](notebooks/en/inference/00-kv-cache.ipynb) | [中文](notebooks/zh/inference/00-kv-cache.ipynb) | [文言](notebooks/lzh/inference/00-kv-cache.ipynb) |
| 01 | Flash Attention | [EN](notebooks/en/inference/01-flash-attention.ipynb) | [中文](notebooks/zh/inference/01-flash-attention.ipynb) | [文言](notebooks/lzh/inference/01-flash-attention.ipynb) |
| 02 | Continuous Batching | [EN](notebooks/en/inference/02-continuous-batching.ipynb) | [中文](notebooks/zh/inference/02-continuous-batching.ipynb) | [文言](notebooks/lzh/inference/02-continuous-batching.ipynb) |
| 03 | Paged Attention | [EN](notebooks/en/inference/03-paged-attention.ipynb) | [中文](notebooks/zh/inference/03-paged-attention.ipynb) | [文言](notebooks/lzh/inference/03-paged-attention.ipynb) |
| 04 | Prefix Caching | [EN](notebooks/en/inference/04-prefix-caching.ipynb) | [中文](notebooks/zh/inference/04-prefix-caching.ipynb) | [文言](notebooks/lzh/inference/04-prefix-caching.ipynb) |
| 05 | Quantization & Pruning | [EN](notebooks/en/inference/05-quantization-pruning.ipynb) | [中文](notebooks/zh/inference/05-quantization-pruning.ipynb) | [文言](notebooks/lzh/inference/05-quantization-pruning.ipynb) |
| 06 | Speculative Decoding | [EN](notebooks/en/inference/06-speculative-decoding.ipynb) | [中文](notebooks/zh/inference/06-speculative-decoding.ipynb) | [文言](notebooks/lzh/inference/06-speculative-decoding.ipynb) |
| 07 | Model Compilation | [EN](notebooks/en/inference/07-model-compilation.ipynb) | [中文](notebooks/zh/inference/07-model-compilation.ipynb) | [文言](notebooks/lzh/inference/07-model-compilation.ipynb) |
| 08 | Serving Architecture | [EN](notebooks/en/inference/08-serving-architecture.ipynb) | [中文](notebooks/zh/inference/08-serving-architecture.ipynb) | [文言](notebooks/lzh/inference/08-serving-architecture.ipynb) |
| 09 | Attention Mechanisms | [EN](notebooks/en/inference/09-attention-mechanisms.ipynb) | [中文](notebooks/zh/inference/09-attention-mechanisms.ipynb) | [文言](notebooks/lzh/inference/09-attention-mechanisms.ipynb) |

### 3. Diffusion Models

From noise to images: DDPM, sampling acceleration, classifier-free guidance, latent diffusion, DiT, video generation, training engineering, and fine-tuning (ControlNet, LoRA, DreamBooth).

| # | Topic | EN | 中文 | 文言 |
|---|-------|----|------|------|
| 00 | DDPM Foundations | [EN](notebooks/en/diffusion/00-ddpm-foundations.ipynb) | [中文](notebooks/zh/diffusion/00-ddpm-foundations.ipynb) | [文言](notebooks/lzh/diffusion/00-ddpm-foundations.ipynb) |
| 01 | Sampling Acceleration | [EN](notebooks/en/diffusion/01-sampling-acceleration.ipynb) | [中文](notebooks/zh/diffusion/01-sampling-acceleration.ipynb) | [文言](notebooks/lzh/diffusion/01-sampling-acceleration.ipynb) |
| 02 | Classifier-Free Guidance | [EN](notebooks/en/diffusion/02-classifier-free-guidance.ipynb) | [中文](notebooks/zh/diffusion/02-classifier-free-guidance.ipynb) | [文言](notebooks/lzh/diffusion/02-classifier-free-guidance.ipynb) |
| 03 | Latent Diffusion | [EN](notebooks/en/diffusion/03-latent-diffusion.ipynb) | [中文](notebooks/zh/diffusion/03-latent-diffusion.ipynb) | [文言](notebooks/lzh/diffusion/03-latent-diffusion.ipynb) |
| 04 | DiT Architecture | [EN](notebooks/en/diffusion/04-dit-architecture.ipynb) | [中文](notebooks/zh/diffusion/04-dit-architecture.ipynb) | [文言](notebooks/lzh/diffusion/04-dit-architecture.ipynb) |
| 05 | Video Generation | [EN](notebooks/en/diffusion/05-video-generation.ipynb) | [中文](notebooks/zh/diffusion/05-video-generation.ipynb) | [文言](notebooks/lzh/diffusion/05-video-generation.ipynb) |
| 06 | Training Engineering | [EN](notebooks/en/diffusion/06-training-engineering.ipynb) | [中文](notebooks/zh/diffusion/06-training-engineering.ipynb) | [文言](notebooks/lzh/diffusion/06-training-engineering.ipynb) |
| 07 | Fine-tuning & Adaptation | [EN](notebooks/en/diffusion/07-finetuning-adaptation.ipynb) | [中文](notebooks/zh/diffusion/07-finetuning-adaptation.ipynb) | [文言](notebooks/lzh/diffusion/07-finetuning-adaptation.ipynb) |

## Inspiration

Last week Karpathy shared his [obsidian-llm](https://github.com/karpathy/obsidian-llm) knowledge base setup. I loved the idea — but beyond a queryable knowledge base, I wanted a **task head**: something that distills papers, PDFs, and blog posts into step-by-step, interactive tutorials. That's how this repo was born.

**Next step**: build an automated workflow where adding a paper or blog to the knowledge base triggers Claude Code to generate a new tutorial notebook and push it here. Knowledge in, tutorials out.

Happy to share with everyone — humans and AI alike. Let's learn together.
